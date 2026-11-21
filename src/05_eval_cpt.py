#!/usr/bin/env python3
"""CPT 전후 비교: (1) 세 종류 텍스트의 perplexity, (2) 문장 이어쓰기 프로브.

  k8s_seen     학습한 문서   -> 크게 내려가야 함 (지식 습득)
  k8s_heldout  안 본 문서    -> 조금 내려가면 도메인 일반화
  general_ko   일반 위키     -> 거의 그대로여야 함 (많이 오르면 망각)

백엔드
  --backend mlx    Mac (Apple Silicon). BASE + LoRA 어댑터 (mlx_lm.lora -c config_cpt.yaml 결과)
  --backend cuda   NVIDIA GPU. BASE + CPT 모델 (train_cpt_hf.py 결과, 병합 모델 또는 PEFT 어댑터 폴더)
  --backend auto   (기본) CUDA 가 있으면 cuda, 아니면 mlx

  python 05_eval_cpt.py                                  # 자동 선택, 기본 경로
  python 05_eval_cpt.py --backend mlx                    # BASE vs adapters/k8s-ko-cpt
  python 05_eval_cpt.py --backend cuda                   # BASE vs models/k8s-ko-cpt-hf/final (8,192 토큰)
  python 05_eval_cpt.py --backend cuda --tuned models/k8s-ko-cpt-hf/checkpoint-200
"""
import argparse
import json
import math
from pathlib import Path

EVAL_FILES = ["k8s_seen", "k8s_heldout", "general_ko"]

# base 모델은 대화가 아니라 '이어쓰기'를 하므로 문서 문체의 앞부분을 준다
PROBES = [
    "# 파드\n\n파드(Pod)는 쿠버네티스에서",
    "# 디플로이먼트\n\n디플로이먼트(Deployment)는",
    "퍼시스턴트볼륨클레임(PVC)은 사용자의",
    "kube-proxy는 클러스터의 각 노드에서 실행되는",
    "파드의 로그를 확인하려면 다음 명령을 실행한다.\n\n```shell\n",
    "스테이트풀셋(StatefulSet)이 디플로이먼트와 다른 점은",
]
REPETITION_PENALTY = 1.1   # 두 백엔드 모두 greedy + 같은 반복 패널티 (노트북 평가와 같은 조건)

# 백엔드별 기본값: 학습 스크립트가 만드는 결과 경로, 학습 블록 길이
DEFAULTS = {
    "mlx": {"tuned": "adapters/k8s-ko-cpt", "max_tokens": 2048},          # config_cpt.yaml
    "cuda": {"tuned": "models/k8s-ko-cpt-hf/final", "max_tokens": 8192},  # train_cpt_hf.py
}


def iter_texts(path):
    for line in path.open(encoding="utf-8"):
        yield json.loads(line)["text"]


# ── MLX (Mac) ───────────────────────────────────────────────────────────────
class MLXBackend:
    def __init__(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_logits_processors

        self.mx, self.nn, self._generate, self._load = mx, nn, generate, load
        self.processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY)

    def check_tuned(self, tuned):
        if not (Path(tuned) / "adapters.safetensors").exists():
            raise SystemExit(f"MLX 어댑터가 없습니다: {tuned}  (먼저 mlx_lm.lora -c config_cpt.yaml 로 학습)")

    def load(self, base, tuned):
        """tuned=None 이면 BASE, 아니면 BASE + LoRA 어댑터."""
        self.model, self.tok = self._load(base, adapter_path=tuned)

    def perplexity(self, path, max_tokens):
        mx, nn = self.mx, self.nn
        total_nll, total_tokens = 0.0, 0
        for text in iter_texts(path):
            ids = self.tok.encode(text)[:max_tokens]
            x = mx.array(ids)[None]
            logits = self.model(x[:, :-1])
            # 모델이 bf16 이라 로짓도 bf16 이다. 토큰 수천 개의 NLL 을 bf16 으로 더하면 반올림 오차가 쌓이므로 fp32 로 계산
            nll = nn.losses.cross_entropy(logits.astype(mx.float32), x[:, 1:], reduction="sum")
            total_nll += nll.item()
            total_tokens += len(ids) - 1
        return math.exp(total_nll / total_tokens)

    def complete(self, prompt, max_new_tokens):
        return self._generate(self.model, self.tok, prompt=prompt, max_tokens=max_new_tokens,
                              logits_processors=self.processors)

    def unload(self):
        del self.model
        self.mx.clear_cache()


# ── CUDA (NVIDIA, Hugging Face transformers) ────────────────────────────────
class CUDABackend:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU 를 찾을 수 없습니다. Mac 에서는 --backend mlx 를 쓰세요.")
        self.torch, self.AutoModel, self.AutoTokenizer = torch, AutoModelForCausalLM, AutoTokenizer

    def check_tuned(self, tuned):
        if not Path(tuned).exists():
            raise SystemExit(f"CPT 모델이 없습니다: {tuned}  (먼저 train_cpt_hf.py 로 학습)")

    def load(self, base, tuned):
        """tuned=None 이면 BASE. tuned 가 PEFT 어댑터 폴더면 BASE 에 얹고, 아니면 병합된 모델로 읽는다."""
        torch = self.torch
        is_adapter = tuned is not None and (Path(tuned) / "adapter_config.json").exists()
        path = base if tuned is None or is_adapter else tuned
        self.tok = self.AutoTokenizer.from_pretrained(base)   # CPT 는 토크나이저를 바꾸지 않는다
        self.model = self.AutoModel.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        if is_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, tuned).eval()

    def perplexity(self, path, max_tokens):
        total_nll, total_tokens = 0.0, 0
        with self.torch.no_grad():
            for text in iter_texts(path):
                ids = self.tok(text, return_tensors="pt", truncation=True,
                               max_length=max_tokens).input_ids.to(self.model.device)
                # transformers 의 loss 는 로짓을 fp32 로 올려서 계산한다 (평균 → 토큰 수를 곱해 합으로)
                total_nll += self.model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
                total_tokens += ids.shape[1] - 1
        return math.exp(total_nll / total_tokens)

    def complete(self, prompt, max_new_tokens):
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=REPETITION_PENALTY,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def unload(self):
        del self.model
        self.torch.cuda.empty_cache()


def pick_backend(name):
    if name != "auto":
        return name
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "mlx"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "mlx", "cuda"], default="auto",
                    help="mlx: Mac (Apple Silicon), cuda: NVIDIA GPU, auto: CUDA 가 있으면 cuda")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B", help="BASE 모델")
    ap.add_argument("--tuned", "--adapter", dest="tuned", default=None,
                    help="CPT 결과. mlx: LoRA 어댑터 폴더 (기본 adapters/k8s-ko-cpt) / "
                         "cuda: 병합 모델 또는 PEFT 어댑터 폴더 (기본 models/k8s-ko-cpt-hf/final)")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="문서당 최대 토큰. 학습 블록 길이와 같게 (기본 mlx 2048, cuda 8192)")
    ap.add_argument("--gen-tokens", type=int, default=120)
    ap.add_argument("--no-probes", action="store_true")
    args = ap.parse_args()

    name = pick_backend(args.backend)
    tuned = args.tuned or DEFAULTS[name]["tuned"]
    max_tokens = args.max_tokens or DEFAULTS[name]["max_tokens"]
    backend = MLXBackend() if name == "mlx" else CUDABackend()
    backend.check_tuned(tuned)   # BASE 평가를 다 돌린 뒤에 실패하지 않도록 먼저 확인
    print(f"backend={name}  BASE={args.model}  CPT={tuned}  max_tokens={max_tokens}")

    ppl, gens = {}, {}
    for label, path in [("BASE", None), ("CPT", tuned)]:
        backend.load(args.model, path)
        ppl[label] = {}
        for n in EVAL_FILES:
            f = Path(args.eval_dir) / f"{n}.jsonl"
            if f.exists():
                ppl[label][n] = backend.perplexity(f, max_tokens)
        if not args.no_probes:
            gens[label] = [backend.complete(p, args.gen_tokens) for p in PROBES]
        backend.unload()

    print(f"\n{'perplexity':<14}{'BASE':>10}{'CPT':>10}{'변화':>10}")
    for n in ppl["BASE"]:
        b, c = ppl["BASE"][n], ppl["CPT"][n]
        print(f"{n:<14}{b:>10.2f}{c:>10.2f}{(c - b) / b:>+10.1%}")

    for i, p in enumerate(PROBES if gens else []):
        print("\n" + "=" * 80 + f"\nPROMPT: {p!r}")
        for label in gens:
            print(f"\n[{label}] {p}{gens[label][i]}")


if __name__ == "__main__":
    main()
