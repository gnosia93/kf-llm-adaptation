#!/usr/bin/env python3
"""선호(preference) 데이터: SFT 모델이 같은 질문에 여러 답을 쓰고, 심판(judge)이 점수를 매겨 좋은 답 / 나쁜 답 쌍을 만든다.

  단계 1 generate  SFT 모델로 질문마다 답변 K개를 샘플링 (on-policy: 지금 모델이 실제로 하는 실수를 모은다)
  단계 2 judge     각 답변에 1~10점. 기준 답변(공식 문서에서 온 SFT 정답)을 심판에게 함께 준다
                     --judge api        OpenAI 호환 API (권장: vLLM + Qwen2.5-7B-Instruct, synth_qa.py 와 같은 서버)
                     --judge heuristic  모델 없이 계산 (기준 답변과 글자 bigram 겹침 + 종료/반복 감점). 약한 대용품
  단계 3 pairs     최고점 답 = chosen, 최저점 답 = rejected (점수 차가 --min-margin 이상인 것만)

  # 터미널 1 (vLLM, 모듈 8.4 참고). SFT 모델(약 7GB)과 같이 GPU 에 올라가도록 메모리를 줄인다
  vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.55
  # 터미널 2
  python 08_build_pref.py                         # generate → judge(api) → pairs
  python 08_build_pref.py --stage judge           # 후보는 그대로 두고 다시 채점만
  python 08_build_pref.py --judge heuristic       # 심판 모델 없이 (품질 낮음)

출력:
  data/pref/candidates.jsonl   {"id", "messages"(system,user), "reference", "kind", "answers": [...], "stopped": [...], "scores": [...]}
  data/pref/{train,valid}.jsonl  {"prompt": [system, user], "chosen": [assistant], "rejected": [assistant]}
      → train_dpo_hf.py (DPO), train_rm_hf.py (보상 모델) 가 같은 파일을 쓴다
"""
import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path

prep = import_module("02_prepare_data")

JUDGE_PROMPT = """당신은 쿠버네티스 한국어 Q&A 답변을 채점하는 엄격한 심사위원입니다.

<질문>
{question}
</질문>

<기준 답변 (공식 문서에서 가져온 내용. 사실 판단의 근거)>
{reference}
</기준 답변>

{answers}

각 답변을 1~10점으로 채점하세요.
- 정확성 (가장 중요): 기준 답변·쿠버네티스 사실과 어긋나는 내용, 지어낸 명령어·필드·버전이 있으면 크게 감점
- 질문 적합성: 질문에 직접 답하는가. 관계없는 문서 내용을 길게 옮겨 적으면 감점
- 표현: 자연스러운 한국어, 공식 번역어(파드, 디플로이먼트 등), 같은 말 반복이나 중간에 끊긴 답은 감점
- 기준 답변과 표현이 달라도 사실이 맞으면 감점하지 마세요. 길다고 좋은 답이 아닙니다.

출력은 JSON 하나만: {{"scores": [답변1 점수, 답변2 점수, ...]}}"""

GENERAL_JUDGE_PROMPT = """당신은 한국어 AI 어시스턴트의 답변을 채점하는 엄격한 심사위원입니다.

<질문>
{question}
</질문>

<참고 답변>
{reference}
</참고 답변>

{answers}

각 답변을 1~10점으로 채점하세요. 기준: 질문에 맞게 정확하고 도움이 되는가, 자연스러운 한국어인가,
반복·끊김·엉뚱한 주제(예: 묻지도 않은 쿠버네티스 이야기)가 없는가. 참고 답변과 표현이 달라도 좋은 답이면 높은 점수.

출력은 JSON 하나만: {{"scores": [답변1 점수, 답변2 점수, ...]}}"""


# ── 단계 1: 후보 생성 ───────────────────────────────────────────────────────
def load_prompts(sft_train, n, general_ratio, seed):
    """SFT 학습 데이터의 질문을 재사용한다 (평가용 held-out 용어는 이미 빠져 있음). 기준 답변 = SFT 정답."""
    rows = [json.loads(l)["messages"] for l in open(sft_train, encoding="utf-8")]
    k8s = [r for r in rows if r[0]["content"] == prep.SYSTEM_PROMPT]
    general = [r for r in rows if r[0]["content"] != prep.SYSTEM_PROMPT]
    rng = random.Random(seed)
    n_gen = min(len(general), int(n * general_ratio))
    picked = [("k8s", r) for r in rng.sample(k8s, min(len(k8s), n - n_gen))] + \
             [("general", r) for r in rng.sample(general, n_gen)]
    rng.shuffle(picked)
    return [{"id": i, "kind": kind, "messages": r[:2], "reference": r[2]["content"]} for i, (kind, r) in enumerate(picked)]


def generate_candidates(args, items):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.policy)
    tok.padding_side = "left"                      # 생성은 왼쪽 padding (오른쪽 끝에서 이어 쓴다)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.bfloat16, device_map=device).eval()
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    out_rows = []
    for s in range(0, len(items), args.gen_batch):
        batch = items[s: s + args.gen_batch]
        texts = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) for it in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=0.95,
                                 max_new_tokens=args.max_new_tokens, num_return_sequences=args.k,
                                 eos_token_id=stop, pad_token_id=stop[-1])
        new = out[:, enc.input_ids.shape[1]:].tolist()
        for i, it in enumerate(batch):
            seqs = new[i * args.k:(i + 1) * args.k]
            it = dict(it)
            it["stopped"] = [any(t in stop for t in q) for q in seqs]
            it["answers"] = [tok.decode(q, skip_special_tokens=True).strip() for q in seqs]
            out_rows.append(it)
        print(f"  생성 {min(s + args.gen_batch, len(items))}/{len(items)}")
    return out_rows


# ── 단계 2: 채점 ────────────────────────────────────────────────────────────
def bigrams(s):
    s = re.sub(r"\s+", "", s)
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def heuristic_scores(item):
    """기준 답변과의 bigram F1 (0~1) 을 1~10 점으로. 끝나지 않았거나 같은 줄을 반복하면 감점."""
    ref = bigrams(item["reference"])
    scores = []
    for ans, stopped in zip(item["answers"], item["stopped"]):
        cand = bigrams(ans)
        overlap = sum((ref & cand).values())
        p, r = overlap / max(1, sum(cand.values())), overlap / max(1, sum(ref.values()))
        f1 = 2 * p * r / max(1e-9, p + r)
        lines = [l for l in ans.splitlines() if l.strip()]
        repeat = 1 - len(set(lines)) / max(1, len(lines))
        scores.append(round(1 + 9 * f1 - (3 if not stopped else 0) - 5 * repeat, 2))
    return scores


def api_scores(args, item, rng):
    """답변 순서를 섞어서 보여준다 (심판 모델은 앞쪽 답에 점수를 더 주는 위치 편향이 있다)."""
    synth = import_module("synth_qa")
    order = list(range(len(item["answers"])))
    rng.shuffle(order)
    answers = "\n\n".join(
        f"<답변{j + 1}>\n{item['answers'][i]}{'' if item['stopped'][i] else ' …(중간에 끊김)'}\n</답변{j + 1}>"
        for j, i in enumerate(order))
    tpl = JUDGE_PROMPT if item["kind"] == "k8s" else GENERAL_JUDGE_PROMPT
    prompt = tpl.format(question=item["messages"][1]["content"], reference=item["reference"][:3000], answers=answers)
    text = synth.call(args.api_base, args.api_key, args.judge_model, prompt, temperature=0.0)
    m = re.search(r"\{.*\}", text, re.S)
    shuffled = json.loads(m.group(0))["scores"] if m else []
    if len(shuffled) != len(order):
        return None
    scores = [0.0] * len(order)
    for j, i in enumerate(order):
        scores[i] = float(shuffled[j])
    return scores


def judge(args, items):
    if args.judge == "heuristic":
        for it in items:
            it["scores"] = heuristic_scores(it)
        return items
    rngs = [random.Random(args.seed + it["id"]) for it in items]

    def work(pair):
        it, rng = pair
        try:
            return api_scores(args, it, rng)
        except Exception as e:   # 네트워크·JSON 오류: 이 질문만 건너뛴다
            print(f"  채점 실패 id={it['id']}: {type(e).__name__}: {e}")
            return None

    with ThreadPoolExecutor(args.workers) as pool:
        for i, (it, sc) in enumerate(zip(items, pool.map(work, zip(items, rngs))), 1):
            it["scores"] = sc
            if i % 100 == 0 or i == len(items):
                print(f"  채점 {i}/{len(items)}")
    return items


# ── 단계 3: 쌍 만들기 ───────────────────────────────────────────────────────
def make_pairs(items, min_margin):
    pairs, skipped = [], Counter()
    for it in items:
        sc = it.get("scores")
        if not sc:
            skipped["채점 없음"] += 1
            continue
        best, worst = max(range(len(sc)), key=sc.__getitem__), min(range(len(sc)), key=sc.__getitem__)
        if sc[best] - sc[worst] < min_margin:
            skipped["점수 차 부족"] += 1
            continue
        if it["answers"][best].strip() == it["answers"][worst].strip():
            skipped["같은 답"] += 1
            continue
        pairs.append({
            "id": it["id"], "kind": it["kind"],
            "prompt": it["messages"],
            "chosen": [{"role": "assistant", "content": it["answers"][best]}],
            "rejected": [{"role": "assistant", "content": it["answers"][worst]}],
            "score_chosen": sc[best], "score_rejected": sc[worst],
        })
    return pairs, skipped


def write_jsonl(path, rows):
    prep.write_jsonl(Path(path), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "generate", "judge", "pairs"], default="all")
    ap.add_argument("--policy", default="models/k8s-ko-sft-hf/final", help="후보를 생성할 모델 (보통 SFT 결과)")
    ap.add_argument("--sft-train", default="data/sft/train.jsonl")
    ap.add_argument("--out", default="data/pref")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--general-ratio", type=float, default=0.2, help="일반 질문 비율 (대화 능력 유지)")
    ap.add_argument("--k", type=int, default=4, help="질문당 후보 답변 수")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--gen-batch", type=int, default=16, help="질문 수 기준 (실제 시퀀스 = x k)")
    ap.add_argument("--judge", choices=["api", "heuristic"], default="api")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min-margin", type=float, default=None, help="chosen-rejected 최소 점수 차 (기본 api 2, heuristic 1)")
    ap.add_argument("--valid-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    cand_path = out / "candidates.jsonl"

    if args.stage in ("all", "generate"):
        items = load_prompts(args.sft_train, args.n_prompts, args.general_ratio, args.seed)
        print(f"[generate] 질문 {len(items)}개 x 답변 {args.k}개  (policy={args.policy})")
        items = generate_candidates(args, items)
        write_jsonl(cand_path, items)
        n = sum(len(it["stopped"]) for it in items)
        print(f"  -> {cand_path}  (끝까지 생성된 답 {sum(sum(it['stopped']) for it in items) / n:.0%})")
    if args.stage in ("all", "judge"):
        items = [json.loads(l) for l in open(cand_path, encoding="utf-8")]
        print(f"[judge] {args.judge}  ({len(items)}개)")
        items = judge(args, items)
        write_jsonl(cand_path, items)
    if args.stage in ("all", "pairs"):
        items = [json.loads(l) for l in open(cand_path, encoding="utf-8")]
        margin = args.min_margin if args.min_margin is not None else (2.0 if args.judge == "api" else 1.0)
        pairs, skipped = make_pairs(items, margin)
        random.Random(args.seed).shuffle(pairs)
        n_valid = max(1, int(len(pairs) * args.valid_ratio))
        write_jsonl(out / "valid.jsonl", pairs[:n_valid])
        write_jsonl(out / "train.jsonl", pairs[n_valid:])
        avg = lambda k: sum(p[k] for p in pairs) / max(1, len(pairs))
        print(f"[pairs] {len(pairs)}쌍 (train {len(pairs) - n_valid} / valid {n_valid}), 제외: {dict(skipped)}")
        print(f"  평균 점수 chosen {avg('score_chosen'):.2f} / rejected {avg('score_rejected'):.2f}, "
              f"평균 길이 chosen {avg_len(pairs, 'chosen'):.0f}자 / rejected {avg_len(pairs, 'rejected'):.0f}자")


def avg_len(pairs, key):
    return sum(len(p[key][0]["content"]) for p in pairs) / max(1, len(pairs))


if __name__ == "__main__":
    main()
