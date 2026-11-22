#!/usr/bin/env python3
"""(선택) 큰 LLM 으로 문서 청크마다 '실제 사용자가 할 법한 질문 + 근거 있는 답변'을 합성한다.

OpenAI 호환 API 면 무엇이든 된다. g6e 에서는 vLLM 으로 Qwen2.5-7B-Instruct (Apache-2.0) 를 띄워 쓰는 것을 권한다.
(학습과 GPU 를 같이 쓸 수 없으므로 SFT 학습 전에 돌리고 vLLM 을 끈다.)

  # 터미널 1 (별도 가상환경, 모듈 8.4 참고)
  vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.9
  # 터미널 2
  python synth_qa.py --limit 50              # 먼저 50개로 품질 확인
  python synth_qa.py                         # 전체 (이미 만든 청크는 건너뜀)
  python 06_build_sft.py --synth data/synth_qa.jsonl

출력: data/synth_qa.jsonl  {"source", "heading", "question", "answer"}  (청크 하나에 질문 최대 N개)
"""
import argparse
import json
import os
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path

prep = import_module("02_prepare_data")

PROMPT = """다음은 쿠버네티스 공식 한글 문서의 일부다.

<문서 제목>{title}</문서 제목>
<섹션>{heading}</섹션>
<본문>
{text}
</본문>

이 본문만 근거로, 쿠버네티스 사용자가 실제로 물어볼 법한 질문과 답변 {n}쌍을 만들어라.
규칙:
- 질문은 다양하게: 개념 설명, 방법(how-to), 비교, 문제 해결 등. 문서 제목을 그대로 베끼지 마라.
- 질문만 읽어도 무엇을 묻는지 알 수 있게 써라. "이 문서", "위 내용" 같은 표현 금지.
- 답변은 한국어 존댓말로, 본문에 있는 사실·명령어·YAML·숫자만 사용하라. 본문에 없는 내용은 추가하지 마라.
- 명령어와 YAML 은 마크다운 코드 블록으로 써라. 용어는 본문의 번역어(예: 파드, 디플로이먼트)를 따른다.
- 본문이 목차·링크 목록뿐이라 질문을 만들 수 없으면 빈 배열 [] 을 출력하라.

출력은 JSON 배열 하나만: [{{"question": "...", "answer": "..."}}, ...]"""


def call(api_base, api_key, model, prompt, temperature, timeout=300):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 2048,
    }).encode()
    req = urllib.request.Request(f"{api_base.rstrip('/')}/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse_pairs(text):
    """응답에서 JSON 배열을 꺼낸다. 코드 블록(```json)으로 감싸 오는 경우도 처리."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    pairs = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        q, a = str(d.get("question", "")).strip(), str(d.get("answer", "")).strip()
        if len(q) >= 5 and len(a) >= 20 and not re.search(r"이 문서|위 (내용|본문)|본문에서", q):
            pairs.append((q, a))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="raw/website/content/ko/docs")
    ap.add_argument("--out", default="data/synth_qa.jsonl")
    ap.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=3, help="청크당 질문 수")
    ap.add_argument("--min-chars", type=int, default=300, help="이보다 짧은 청크는 건너뜀")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개 청크만 (품질 확인용)")
    ap.add_argument("--workers", type=int, default=32, help="동시 요청 수 (vLLM 은 많이 보낼수록 빠르다)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _, chunks = prep.build_chunks(Path(args.docs), args.min_chars, 1800)
    random.Random(args.seed).shuffle(chunks)
    if args.limit:
        chunks = chunks[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():   # 이어서 실행: 이미 처리한 청크는 건너뛴다 (질문이 0개였던 청크 포함)
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["source"], r["heading"]))
    todo = [c for c in chunks if (c["source"], c["heading"]) not in done]
    print(f"청크 {len(chunks)}개 중 {len(todo)}개 생성 -> {out}  (model={args.model}, api={args.api_base})")

    def work(c):
        prompt = PROMPT.format(n=args.n, **c)
        return c, parse_pairs(call(args.api_base, args.api_key, args.model, prompt, args.temperature))

    n_pairs, n_fail = 0, 0
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, c) for c in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                c, pairs = fut.result()
            except Exception as e:  # 네트워크 오류 등: 다음 실행 때 다시 시도된다
                n_fail += 1
                print(f"  실패: {type(e).__name__}: {e}")
                continue
            base = {"source": c["source"], "heading": c["heading"]}
            # 질문이 0개여도 '처리함' 표시를 남겨서 다시 요청하지 않게 한다
            for q, a in pairs or [(None, None)]:
                f.write(json.dumps({**base, "question": q, "answer": a}, ensure_ascii=False) + "\n")
            n_pairs += len(pairs)
            if i % 50 == 0 or i == len(futures):
                f.flush()
                print(f"  {i}/{len(futures)} 청크, 질문 {n_pairs}개, 실패 {n_fail}")
    print(f"완료: 질문 {n_pairs}개 추가. 원문과 대조해서 샘플을 꼭 검수할 것.")


if __name__ == "__main__":
    main()
