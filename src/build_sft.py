#!/usr/bin/env python3
"""지시 튜닝(SFT) 데이터: 쿠버네티스 질의응답 + 일반 한국어 지시 데이터(replay)를 섞는다.

CPT 로 넣은 '지식'은 그대로 두고, 질문을 받으면 답하는 '형식'을 가르치는 것이 목적이다.

소스 (모두 {"messages": [system, user, assistant]} 형식으로 변환)
  glossary  용어집(reference/glossary) 정의 → "X 이(가) 뭐야?" 형태. 짧고 정확한 답
  section   문서 섹션 → 템플릿 질문 + 섹션 원문. 긴 설명·명령어·YAML 답변 형식
  synth     (선택) synth_qa.py 로 큰 LLM 이 만든 질의응답. 가장 자연스러운 질문
  general   일반 한국어 지시 데이터 (nlpai-lab/kullm-v2 의 alpaca·dolly 부분) → 대화 능력 망각 방지

출력:
  data/sft/{train,valid}.jsonl  {"messages": [...]}  (train_sft_hf.py, train_hf.py, mlx_lm.lora 공용)
  data/eval/sft_eval.jsonl      {"kind", "system", "question", "reference"}  07_eval_sft.py 용
      kind = glossary_heldout  SFT 에서 뺀 용어 (CPT 에서는 봤음) → SFT 가 CPT 지식을 꺼내 쓰게 했나
             general_heldout   학습에 안 쓴 일반 지시문           → 일반 대화 능력 유지
             practical         손으로 쓴 실무 질문 (정답 없음)     → 생성 결과를 눈으로 비교
"""
import argparse
import json
import random
import re
from collections import Counter
from importlib import import_module
from pathlib import Path

prep = import_module("02_prepare_data")

SYSTEM_PROMPT = prep.SYSTEM_PROMPT
GENERAL_SYSTEM_PROMPT = "당신은 친절하고 정확한 한국어 AI 어시스턴트입니다."

# {t}=한글 용어, {e}=영어 원문, {이가}/{은는}/{을를}=받침에 맞는 조사
GLOSSARY_TEMPLATES = [
    "{t}{이가} 뭐야?",
    "쿠버네티스에서 {t}{은는} 무엇인가요?",
    "{t}({e})에 대해 간단히 설명해 주세요.",
    "쿠버네티스 용어 {e}의 의미를 알려줘.",
    "{t}의 개념을 알려줘.",
    "{t}{을를} 한두 문단으로 설명해 줄래?",
]

PRACTICAL_QUESTIONS = [
    "파드(Pod)가 무엇인지 설명해줘.",
    "디플로이먼트와 스테이트풀셋의 차이는 무엇인가요?",
    "kubectl로 파드의 로그를 확인하는 방법을 알려줘.",
    "퍼시스턴트 볼륨과 퍼시스턴트 볼륨 클레임의 관계를 설명해 주세요.",
    "서비스의 ClusterIP, NodePort, LoadBalancer 타입을 비교해줘.",
    "디플로이먼트를 이전 버전으로 롤백하려면 어떻게 하나요?",
    "컨피그맵과 시크릿은 언제 각각 사용하나요?",
    "파드가 CrashLoopBackOff 상태일 때 무엇부터 확인해야 하나요?",
]
GENERAL_QUESTIONS = [
    "김치찌개를 맛있게 끓이는 방법을 알려줘.",
    "회의 일정을 다음 주 화요일로 미루자는 정중한 이메일을 써 줘.",
    "광합성을 초등학생에게 설명하듯이 말해 줘.",
    "파이썬에서 리스트와 튜플의 차이는?",
]


# ── 한국어 조사 ─────────────────────────────────────────────────────────────
def has_batchim(word):
    """마지막 글자에 받침이 있나. 한글이 아니면 영어·숫자 발음을 대충 따른다 (Pod→'드', API→'이')."""
    w = re.sub(r"[\s)\]}>\"'.]+$", "", word)
    if not w:
        return False
    ch = w[-1]
    if "가" <= ch <= "힣":
        return (ord(ch) - ord("가")) % 28 != 0
    if ch.isdigit():
        return ch in "013678"           # 영, 일, 삼, 육, 칠, 팔
    if w.isupper():                     # 약어는 알파벳 이름으로 읽는다: L(엘) M(엠) N(엔) R(알)
        return ch in "LMNR"
    return ch.lower() in "lmnbkp" or w.lower().endswith("ng")   # 영어 단어 (대략: Job→잡, ConfigMap→맵)


def josa(word):
    b = has_batchim(word)
    return {"이가": "이" if b else "가", "은는": "은" if b else "는", "을를": "을" if b else "를"}


def fill(template, t, e):
    """'{t}{이가}' 처럼 조사 자리는 바로 앞 단어({t} 또는 {e})의 받침을 따른다."""
    if t == e:                                   # 원문 표기가 따로 없는 용어 (예: kube-proxy)
        template = template.replace("{t}({e})", "{t}")
    words = {"t": t, "e": e}
    return re.sub(r"\{([te])\}(?:\{(이가|은는|을를)\})?",
                  lambda m: words[m.group(1)] + (josa(words[m.group(1)])[m.group(2)] if m.group(2) else ""),
                  template)


# ── 소스별 변환 ─────────────────────────────────────────────────────────────
def chat(system, question, answer):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def reflow(text):
    """용어집 원문은 문장 중간에 줄바꿈이 있다. 문단 안의 줄바꿈을 공백으로 합친다 (코드 블록 제외)."""
    out = []
    for is_code, seg in prep.split_code(text):
        if is_code:
            out.append(seg)
            continue
        paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", seg)]
        out.append("\n\n".join(p for p in paras if p))
    return "\n\n".join(s.strip("\n") for s in out if s.strip())


def split_title(title):
    """'파드 (원문, Pod)' → ('파드', 'Pod'). 원문이 없으면 영어 자리에 같은 용어를 쓴다."""
    m = re.match(r"^(.*?)\s*\(원문,\s*(.+)\)\s*$", title)
    return (m.group(1).strip(), m.group(2).strip()) if m else (title.strip(), title.strip())


def load_glossary(docs_dir):
    items = []
    for path in sorted((Path(docs_dir) / "reference" / "glossary").glob("*.md")):
        if path.name == "_index.md":
            continue
        title, body = prep.parse_doc(path)
        body = reflow(body)
        if len(body) < 30:
            continue
        term, en = split_title(title)
        items.append({"source": path.stem, "term": term, "en": en, "text": body})
    return items


def glossary_rows(item, n_questions, rng):
    templates = rng.sample(GLOSSARY_TEMPLATES, min(n_questions, len(GLOSSARY_TEMPLATES)))
    return [chat(SYSTEM_PROMPT, fill(tpl, item["term"], item["en"]), item["text"]) for tpl in templates]


def load_synth(path):
    """synth_qa.py 결과: {"source", "question", "answer"} 한 줄에 하나."""
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("question") and r.get("answer"):
            rows.append(r)
    return rows


def load_general(name, n, seed):
    """kullm-v2 에서 alpaca / dolly 부분만 쓴다. vicuna 부분은 다중 대화를 한 턴씩 쪼갠 것이라 문맥이 끊긴다."""
    from datasets import load_dataset

    ds = load_dataset(name, split="train", streaming=True).shuffle(seed=seed, buffer_size=20_000)
    rows = []
    for r in ds:
        if not r["id"].startswith(("alpaca", "dolly")):
            continue
        q, a = r["instruction"].strip(), r["output"].strip()
        if r.get("input", "").strip():
            q = f"{q}\n\n{r['input'].strip()}"
        if not (5 <= len(q) <= 1500 and 20 <= len(a) <= 2000):
            continue
        if re.search(r"OpenAI|ChatGPT|GPT-\d|언어 ?모델로서|AI 언어 모델", a):   # 다른 모델의 자기소개 제거
            continue
        rows.append((q, a))
        if len(rows) >= n:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="raw/website/content/ko/docs")
    ap.add_argument("--out", default="data")
    ap.add_argument("--glossary-questions", type=int, default=3, help="용어 하나당 질문 변형 수")
    ap.add_argument("--glossary-heldout", type=float, default=0.15, help="SFT 에서 빼고 평가에 쓸 용어 비율")
    ap.add_argument("--max-sections", type=int, default=1500, help="섹션 원문 답변 최대 개수 (문서체 답변 편향 방지)")
    ap.add_argument("--synth", default=None, help="synth_qa.py 결과 JSONL (예: data/synth_qa.jsonl)")
    ap.add_argument("--general-dataset", default="nlpai-lab/kullm-v2")
    ap.add_argument("--n-general", type=int, default=1200, help="일반 지시 데이터 수 (0 이면 안 섞음)")
    ap.add_argument("--n-general-eval", type=int, default=50)
    ap.add_argument("--valid-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    train, valid, evals, counts = [], [], [], Counter()

    # 1) 용어집: 일부 용어는 SFT 에서 통째로 빼서 평가에 쓴다
    glossary = load_glossary(args.docs)
    rng.shuffle(glossary)
    n_held = max(1, int(len(glossary) * args.glossary_heldout))
    for item in glossary[:n_held]:
        q = fill(rng.choice(GLOSSARY_TEMPLATES), item["term"], item["en"])
        evals.append({"kind": "glossary_heldout", "system": SYSTEM_PROMPT, "question": q,
                      "reference": item["text"], "source": item["source"]})
    for item in glossary[n_held:]:
        rows = glossary_rows(item, args.glossary_questions, rng)
        train += rows
        counts["glossary"] += len(rows)
    held_terms = {f"reference/glossary/{i['source']}.md" for i in glossary[:n_held]}

    # 2) 문서 섹션: 문서 단위로 valid 분리. 평가용 용어집 문서는 제외
    _, chunks = prep.build_chunks(Path(args.docs), 150, 1800)
    chunks = [c for c in chunks if c["source"] not in held_terms and not c["source"].startswith("reference/glossary/")]
    sec_train, sec_valid, valid_docs = prep.split_by_document(chunks, lambda c: c["source"], args.valid_ratio, rng)
    rng.shuffle(sec_train)
    sec_train = sec_train[: args.max_sections]
    train += [prep.to_sft(c, rng) for c in sec_train]
    valid += [prep.to_sft(c, rng) for c in sec_valid]
    counts["section"] += len(sec_train)
    counts["section(valid)"] += len(sec_valid)

    # 3) 합성 질의응답 (선택). 섹션과 같은 문서 분리 기준을 따른다
    if args.synth:
        valid_set = set(valid_docs)
        for r in load_synth(args.synth):
            if r.get("source") in held_terms:
                continue
            row = chat(SYSTEM_PROMPT, r["question"], r["answer"])
            if r.get("source") in valid_set:
                valid.append(row)
                counts["synth(valid)"] += 1
            else:
                train.append(row)
                counts["synth"] += 1

    # 4) 일반 지시 데이터 replay
    if args.n_general > 0:
        general = load_general(args.general_dataset, args.n_general + args.n_general_eval + 30, args.seed)
        g_eval, g_valid, g_train = general[: args.n_general_eval], general[args.n_general_eval:][:30], \
            general[args.n_general_eval + 30:]
        train += [chat(GENERAL_SYSTEM_PROMPT, q, a) for q, a in g_train]
        valid += [chat(GENERAL_SYSTEM_PROMPT, q, a) for q, a in g_valid]
        evals += [{"kind": "general_heldout", "system": GENERAL_SYSTEM_PROMPT, "question": q, "reference": a}
                  for q, a in g_eval]
        counts["general"] += len(g_train)
        counts["general(valid)"] += len(g_valid)

    evals += [{"kind": "practical", "system": SYSTEM_PROMPT, "question": q, "reference": None}
              for q in PRACTICAL_QUESTIONS]
    evals += [{"kind": "practical_general", "system": GENERAL_SYSTEM_PROMPT, "question": q, "reference": None}
              for q in GENERAL_QUESTIONS]

    rng.shuffle(train)
    prep.write_jsonl(out / "sft" / "train.jsonl", train)
    prep.write_jsonl(out / "sft" / "valid.jsonl", valid)
    prep.write_jsonl(out / "eval" / "sft_eval.jsonl", evals)

    print(f"용어집 {len(glossary)}개 (평가용으로 {n_held}개 제외)")
    print("구성: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"train {len(train)} / valid {len(valid)} -> {out}/sft/")
    print(f"평가 {len(evals)}개 -> {out}/eval/sft_eval.jsonl  " +
          str(dict(Counter(e['kind'] for e in evals))))


if __name__ == "__main__":
    main()
