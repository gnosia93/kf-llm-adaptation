#!/usr/bin/env python3
"""대화형 모델 비교: Instruct 원본 vs Chat Vector vs CPT+SFT (data/eval/sft_eval.jsonl, 06_build_sft.py 가 만듦).

  python 07_eval_sft.py                                   # 기본 3개 모델 (없는 경로는 건너뜀)
  python 07_eval_sft.py --models SFT=models/k8s-ko-sft-hf/final BASE_SFT=models/k8s-ko-base-sft-hf/final

지표
  answer ppl  정답 답변의 perplexity (질문은 조건으로만, 답변 토큰만 계산). 낮을수록 그 답을 할 법하다고 본다
      glossary_heldout  SFT 에 없던 용어 정의 → CPT 지식을 질의응답에서 꺼내 쓰는가   (낮을수록 좋음)
      general_heldout   일반 지시문의 정답    → 일반 대화 능력이 유지됐는가          (instruct 와 비슷하면 좋음)
  stop rate   생성이 max_new_tokens 전에 <|im_end|> 로 끝난 비율. 낮으면 종료 토큰 학습/설정 문제
  생성 결과    practical(실무 질문), practical_general(일반 질문), glossary_heldout 일부를 나란히 출력
  answer len  생성 답변의 평균 글자 수. DPO/PPO 뒤에 급증하면 '길게 쓰면 점수가 높다'를 배운 것 (보상 해킹)
  RM score    (--reward-model 을 주면) 생성 답변의 보상 모델 점수 평균. PPO 가 높인 값이지만, 이것만 오르고
              다른 지표가 나빠지면 보상 모델의 약점을 파고든 것

  python 07_eval_sft.py --models SFT=models/k8s-ko-sft-hf/final DPO=models/k8s-ko-dpo-hf/final \
                        PPO=models/k8s-ko-ppo-hf/final --reward-model models/k8s-ko-rm-hf/final

결과: logs/sft_eval.json (모든 생성 결과 포함)
"""
import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = [
    "INSTRUCT=Qwen/Qwen2.5-3B-Instruct",
    "CHATVEC=models/k8s-ko-3b-chat",
    "SFT=models/k8s-ko-sft-hf/final",
]
PPL_KINDS = ["glossary_heldout", "general_heldout"]


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def messages(row, answer=None):
    msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["question"]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def encode(tok, msgs, add_generation_prompt):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids


@torch.no_grad()
def answer_nll(model, tok, row, max_length):
    """(답변 토큰 NLL 합, 답변 토큰 수). 학습 때와 똑같이 프롬프트 부분은 loss 에서 뺀다."""
    prompt = encode(tok, messages(row), True)
    full = encode(tok, messages(row, row["reference"]), False)[:, :max_length].to(model.device)
    labels = full.clone()
    labels[:, : prompt.shape[1]] = -100
    n = int((labels[:, 1:] != -100).sum())
    if n == 0:
        return 0.0, 0
    return model(full, labels=labels).loss.item() * n, n


@torch.no_grad()
def generate(model, tok, row, max_new_tokens, stop_ids):
    ids = encode(tok, messages(row), True).to(model.device)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, repetition_penalty=1.1, eos_token_id=stop_ids, pad_token_id=stop_ids[-1])
    new = out[0, ids.shape[1]:].tolist()
    stopped = any(t in stop_ids for t in new)
    return tok.decode(new, skip_special_tokens=True).strip(), stopped


@torch.no_grad()
def score_with_rm(path, gen_rows, results, device, dtype):
    """모든 모델의 생성 답변을 같은 보상 모델로 채점한다 (모델별 평균)."""
    from transformers import AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(path)
    rm = AutoModelForSequenceClassification.from_pretrained(path, dtype=dtype).to(device).eval()
    rm.config.pad_token_id = tok.pad_token_id
    for res in results.values():
        scores = []
        for r, g in zip(gen_rows, res["generations"]):
            text = tok.apply_chat_template(messages(r, g["answer"]), tokenize=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False, truncation=True,
                      max_length=2048).input_ids.to(device)
            scores.append(rm(ids).logits[0, 0].float().item())
            g["rm_score"] = scores[-1]
        res["rm_score"] = sum(scores) / max(1, len(scores))
    del rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="이름=경로 ...")
    ap.add_argument("--eval", default="data/eval/sft_eval.jsonl")
    ap.add_argument("--out", default="logs/sft_eval.json")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--n-glossary-gen", type=int, default=6, help="생성까지 해 볼 glossary_heldout 개수")
    ap.add_argument("--reward-model", default=None, help="train_rm_hf.py 결과. 주면 생성 답변의 RM 점수도 잰다")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    gen_rows = [r for r in rows if r["kind"].startswith("practical")]
    gen_rows += [r for r in rows if r["kind"] == "glossary_heldout"][: args.n_glossary_gen]
    device, dtype = device_and_dtype()

    results = {}
    for spec in args.models:
        label, path = spec.split("=", 1)
        if path.startswith(("models/", "./", "/")) and not Path(path).exists():
            print(f"[{label}] {path} 없음 → 건너뜀")
            continue
        print(f"[{label}] {path} 평가 중 ...")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]

        ppl = {}
        for kind in PPL_KINDS:
            nll, n = 0.0, 0
            for r in rows:
                if r["kind"] == kind and r.get("reference"):
                    a, b = answer_nll(model, tok, r, args.max_length)
                    nll, n = nll + a, n + b
            if n:
                ppl[kind] = math.exp(nll / n)
        gens = [generate(model, tok, r, args.max_new_tokens, stop_ids) for r in gen_rows]
        results[label] = {
            "path": path,
            "ppl": ppl,
            "stop_rate": sum(s for _, s in gens) / max(1, len(gens)),
            "generations": [{"kind": r["kind"], "question": r["question"], "answer": g, "stopped": s}
                            for r, (g, s) in zip(gen_rows, gens)],
        }
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not results:
        raise SystemExit("평가할 모델이 없습니다.")
    for res in results.values():
        res["answer_len"] = sum(len(g["answer"]) for g in res["generations"]) / max(1, len(res["generations"]))
    if args.reward_model:
        score_with_rm(args.reward_model, gen_rows, results, device, dtype)
    labels = list(results)
    print("\n" + f"{'지표':<22}" + "".join(f"{l:>12}" for l in labels))
    for kind in PPL_KINDS:
        print(f"{'ppl ' + kind:<22}" + "".join(
            f"{results[l]['ppl'].get(kind, float('nan')):>12.2f}" for l in labels))
    print(f"{'stop rate':<22}" + "".join(f"{results[l]['stop_rate']:>12.0%}" for l in labels))
    print(f"{'answer len (글자)':<20}" + "".join(f"{results[l]['answer_len']:>12.0f}" for l in labels))
    if args.reward_model:
        print(f"{'RM score':<22}" + "".join(f"{results[l]['rm_score']:>12.3f}" for l in labels))

    for i, r in enumerate(gen_rows):
        print("\n" + "=" * 80 + f"\n[{r['kind']}] Q: {r['question']}")
        for l in labels:
            g = results[l]["generations"][i]
            print(f"\n--- {l}{'' if g['stopped'] else '  (max_new_tokens 도달: 멈추지 않음)'}\n{g['answer']}")
        if r.get("reference"):
            print(f"\n--- 정답(용어집 원문)\n{r['reference']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
