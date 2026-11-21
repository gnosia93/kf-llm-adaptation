#!/usr/bin/env python3
"""CPT 전후 비교: (1) 세 종류 텍스트의 perplexity, (2) 문장 이어쓰기 프로브.

  k8s_seen     학습한 문서   -> 크게 내려가야 함 (지식 습득)
  k8s_heldout  안 본 문서    -> 조금 내려가면 도메인 일반화
  general_ko   일반 위키     -> 거의 그대로여야 함 (많이 오르면 망각)
"""
import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate, load

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


def perplexity(model, tokenizer, path, max_tokens):
    total_nll, total_tokens = 0.0, 0
    for line in path.open(encoding="utf-8"):
        ids = tokenizer.encode(json.loads(line)["text"])[:max_tokens]
        x = mx.array(ids)[None]
        logits = model(x[:, :-1])
        nll = nn.losses.cross_entropy(logits, x[:, 1:], reduction="sum")
        total_nll += nll.item()
        total_tokens += len(ids) - 1
    return math.exp(total_nll / total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--adapter", default="adapters/k8s-ko-cpt")
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gen-tokens", type=int, default=120)
    ap.add_argument("--no-probes", action="store_true")
    args = ap.parse_args()

    ppl, gens = {}, {}
    for label, adapter in [("BASE", None), ("CPT", args.adapter)]:
        model, tokenizer = load(args.model, adapter_path=adapter)
        ppl[label] = {}
        for name in EVAL_FILES:
            path = Path(args.eval_dir) / f"{name}.jsonl"
            if path.exists():
                ppl[label][name] = perplexity(model, tokenizer, path, args.max_tokens)
        if not args.no_probes:
            gens[label] = [
                generate(model, tokenizer, prompt=p, max_tokens=args.gen_tokens)
                for p in PROBES
            ]
        del model

    print(f"\n{'perplexity':<14}{'BASE':>10}{'CPT':>10}{'변화':>10}")
    for name in ppl["BASE"]:
        b, c = ppl["BASE"][name], ppl["CPT"][name]
        print(f"{name:<14}{b:>10.2f}{c:>10.2f}{(c - b) / b:>+10.1%}")

    for i, p in enumerate(PROBES if gens else []):
        print("\n" + "=" * 80 + f"\nPROMPT: {p!r}")
        for label in gens:
            print(f"\n[{label}] {p}{gens[label][i]}")


if __name__ == "__main__":
    main()
