#!/usr/bin/env python3
"""계속 사전학습(CPT)용 데이터: 문서를 토큰 길이 기준으로 패킹하고 일반 한국어 텍스트를 섞는다.

학습 프레임워크는 샘플을 max_length 에서 잘라버리므로, 여기서 미리 문서 순서를 유지한 채
섹션 단위로 묶어 각 블록이 max_tokens 를 넘지 않게 만든다. 서로 다른 문서는 섞지 않는다.

출력:
  data/cpt/train.jsonl          {"text"}  쿠버네티스 블록 + 일반 한국어 replay
  data/cpt/valid.jsonl          {"text"}  학습에 쓰지 않은 문서의 블록 (문서 단위로 분리)
  data/cpt/valid_docs.txt       valid 로 뺀 문서 목록
  data/eval/k8s_seen.jsonl      학습에 포함된 블록 샘플   -> 지식 습득 정도
  data/eval/k8s_heldout.jsonl   valid 와 동일             -> 도메인 일반화
  data/eval/general_ko.jsonl    학습에 안 쓴 위키 문서    -> 일반 능력 망각 정도
"""
import argparse
import json
import random
from importlib import import_module
from pathlib import Path

from transformers import AutoTokenizer

prep = import_module("02_prepare_data")

PIECE_CHARS = 600   # 패킹 단위가 되는 문단 묶음 크기 (글자)
SAFETY_TOKENS = 32  # 조각별 토큰 수 합과 실제 토큰 수의 오차 여유 + EOS


def decode(tok, ids):
    """토큰 목록을 문자열로. 잘린 경계에서 반쪽 난 UTF-8 글자(�)는 버린다."""
    return tok.decode(ids).strip("�")


def split_to_budget(text, tok, budget):
    """예산보다 긴 조각(거대한 코드 블록 등)을 줄 경계에서 나눈다. 한 줄이 예산보다 길 때만 토큰 단위로 자른다."""
    parts, cur, cur_len = [], [], 0
    for line in text.split("\n"):
        n = len(tok.encode(line, add_special_tokens=False)) + 1
        if n > budget:
            if cur:
                parts.append("\n".join(cur))
                cur, cur_len = [], 0
            ids = tok.encode(line, add_special_tokens=False)
            parts += [decode(tok, ids[i:i + budget]) for i in range(0, len(ids), budget)]
            continue
        if cur and cur_len + n > budget:
            parts.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += n
    if cur:
        parts.append("\n".join(cur))
    return parts


def pack_document(title, body, tok, max_tokens):
    """하나의 문서를 '# 제목' 헤더가 붙은 max_tokens 이하 블록들로 나눈다."""
    def n_tokens(s):
        return len(tok.encode(s, add_special_tokens=False))

    header = f"# {title}\n\n"
    budget = max_tokens - SAFETY_TOKENS - n_tokens(header)
    blocks, cur, cur_len, cur_heading = [], [], 0, None

    def flush():
        if cur:
            blocks.append(header + "\n\n".join(cur))

    for heading, text in prep.split_sections(body):
        for piece in prep.split_long(text, PIECE_CHARS):
            # 블록이 새로 시작되거나 섹션이 바뀌면 소제목을 다시 붙여 문맥을 유지
            seg = f"## {heading}\n\n{piece}" if heading and (heading != cur_heading or not cur) else piece
            seg_len = n_tokens(seg) + 2
            if cur and cur_len + seg_len > budget:
                flush()
                cur, cur_len = [], 0
                seg = f"## {heading}\n\n{piece}" if heading else piece
                seg_len = n_tokens(seg) + 2
            if seg_len > budget:
                # 조각 하나가 블록보다 크다: 줄 경계에서 나눠 각각 독립 블록으로
                blocks += [header + part for part in split_to_budget(seg, tok, budget)]
            else:
                cur.append(seg)
                cur_len += seg_len
            cur_heading = heading
    flush()
    return blocks


def load_replay(args, tok, target_tokens, n_eval):
    """일반 한국어 위키 문서를 target_tokens 만큼 + 평가용 n_eval 개 가져온다."""
    from datasets import load_dataset

    ds = load_dataset(args.replay_dataset, args.replay_config, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
    train, evals, total = [], [], 0
    for row in ds:
        text = row["text"].strip()
        if len(text) < 300:
            continue
        ids = tok.encode(text, add_special_tokens=False)[: args.max_tokens - SAFETY_TOKENS]
        text = f"# {row['title']}\n\n{decode(tok, ids)}"
        if len(evals) < n_eval:
            evals.append({"text": text})
        elif total < target_tokens:
            train.append({"text": text})
            total += len(ids)
        else:
            break
    return train, evals, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", default=["raw/website/content/ko/docs"],
                    help="여러 개 지정 가능 (예: content/ko/docs content/en/docs)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="블록(JSONL 한 줄)의 최대 토큰 수. 학습 max_length 와 같게 (Mac MLX 는 2048)")
    ap.add_argument("--valid-ratio", type=float, default=0.02)
    ap.add_argument("--replay-ratio", type=float, default=0.2,
                    help="학습 토큰 중 일반 텍스트 비율 (0 이면 replay 없음)")
    ap.add_argument("--replay-dataset", default="wikimedia/wikipedia")
    ap.add_argument("--replay-config", default="20231101.ko")
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8, help="1 epoch 스텝 수 출력용 (배치 x 누적)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    blocks = []  # [(문서 경로, 블록 텍스트)]
    for docs in args.docs:
        n_docs = 0
        for path, title, body in prep.iter_docs(docs):
            n_docs += 1
            blocks += [(str(path), b) for b in pack_document(title, body, tok, args.max_tokens)]
        print(f"{docs}: 문서 {n_docs}개")
    prep.print_stats()

    blocks = [(src, b) for src, b in blocks if len(tok.encode(b)) >= 32]
    domain_train, valid, valid_docs = prep.split_by_document(blocks, lambda x: x[0], args.valid_ratio, rng)
    domain_train = [{"text": b} for _, b in domain_train]
    valid = [{"text": b} for _, b in valid]
    domain_tokens = sum(len(tok.encode(b["text"])) for b in domain_train)

    replay, general_eval, replay_tokens = [], [], 0
    if args.replay_ratio > 0:
        target = int(domain_tokens * args.replay_ratio / (1 - args.replay_ratio))
        replay, general_eval, replay_tokens = load_replay(args, tok, target, args.n_eval)

    train = domain_train + replay
    rng.shuffle(train)
    out = Path(args.out)
    prep.write_jsonl(out / "cpt" / "train.jsonl", train)
    prep.write_jsonl(out / "cpt" / "valid.jsonl", valid)
    (out / "cpt" / "valid_docs.txt").write_text("\n".join(valid_docs) + "\n", encoding="utf-8")
    prep.write_jsonl(out / "eval" / "k8s_seen.jsonl", rng.sample(domain_train, min(args.n_eval, len(domain_train))))
    prep.write_jsonl(out / "eval" / "k8s_heldout.jsonl", valid)
    if general_eval:
        prep.write_jsonl(out / "eval" / "general_ko.jsonl", general_eval)

    steps_per_epoch = -(-len(train) // args.batch_size)
    print(f"쿠버네티스 블록: train {len(domain_train)} ({domain_tokens:,} 토큰) / "
          f"valid {len(valid)} (문서 {len(valid_docs)}개)")
    print(f"일반 텍스트 replay: {len(replay)}개 ({replay_tokens:,} 토큰)")
    print(f"총 train 샘플 {len(train)} -> 실효 배치 {args.batch_size} 기준 1 epoch = {steps_per_epoch} 스텝")


if __name__ == "__main__":
    main()
