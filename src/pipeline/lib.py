"""KFP 파이프라인의 실제 로직. 컴포넌트(components.py)는 이 모듈을 호출하는 얇은 껍데기다.

KFP 와 분리해 둔 이유: 컨테이너 이미지 안(/app/pipelines/lib.py)에서 그대로 import 되고,
로컬에서도 KFP 없이 단위 테스트할 수 있다 (python -m pipelines.selftest).

구성
  1. Quality Gate 공통        check / run_gate
  2. Gate 1 데이터 품질       data_metrics (중복률, 유효 샘플 수, 개인정보·금지어)
  3. Gate 2 CPT              cpt_metrics (도메인 held-out ppl, 일반 ppl)
  4. Gate 3 SFT              sft_metrics (stop_rate, avg_len)
  5. Gate 4 DPO              dpo_metrics (SFT 대비 judge 승률, 길이 증가율)
  6. MLflow                  부모 run, 중첩 run, 학습 run 연결, Model Registry alias
  7. 배포                    vLLM Deployment 무중단 롤링 업데이트 + smoke test + 롤백
  8. Slack
"""
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from importlib import import_module
from pathlib import Path

APP_DIR = Path(os.environ.get("LLM_FT_APP", Path(__file__).resolve().parents[1]))   # 저장소 루트 (이미지에서는 /app)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ── 1. Quality Gate 공통 ─────────────────────────────────────────────────────
class GateFailed(RuntimeError):
    pass


OPS = {
    "<": lambda v, t: v < t, "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t, ">=": lambda v, t: v >= t, "==": lambda v, t: v == t,
}


def check(name, value, op, threshold):
    ok = value is not None and not (isinstance(value, float) and math.isnan(value)) and OPS[op](value, threshold)
    return {"name": name, "value": value, "op": op, "threshold": threshold, "passed": bool(ok)}


def run_gate(gate, checks):
    """모든 조건을 출력하고, 하나라도 실패하면 GateFailed. 메시지는 Slack 알림(오류 메시지)에 그대로 실린다."""
    lines = [f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']} = {fmt(c['value'])}  (조건 {c['op']} {fmt(c['threshold'])})"
             for c in checks]
    print(f"[{gate}]\n  " + "\n  ".join(lines))
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise GateFailed(f"{gate} 실패: " + "; ".join(
            f"{c['name']}={fmt(c['value'])} (조건 {c['op']} {fmt(c['threshold'])})" for c in failed))
    return checks


def gate_outcome(gate, checks):
    """dsl.If 분기용: ("PASS" | "FAIL", 사람이 읽는 요약). 실패 요약은 중단 단계가 예외 메시지로 써서 Slack 에 실린다."""
    failed = [c for c in checks if not c["passed"]]
    lines = [f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']} = {fmt(c['value'])}  (조건 {c['op']} {fmt(c['threshold'])})"
             for c in checks]
    print(f"[{gate}]\n  " + "\n  ".join(lines))
    summary = f"{gate}: " + ("통과" if not failed else "; ".join(
        f"{c['name']}={fmt(c['value'])} (조건 {c['op']} {fmt(c['threshold'])})" for c in failed))
    return ("FAIL" if failed else "PASS"), summary


def fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


# ── 2. Gate 1: 데이터 품질 ───────────────────────────────────────────────────
# 쿠버네티스 문서에는 예시용 주소가 많다. RFC 2606/6761 예약 도메인과 공개 프로젝트 주소는 개인정보가 아니다.
EMAIL_ALLOW = re.compile(
    r"@(?:[\w.-]+\.)?(?:example(?:\.com|\.org|\.net)?|test|invalid|localhost|local)$"
    r"|^git@github\.com$|@(?:[\w.-]+\.)?(?:kubernetes\.io|k8s\.io|cncf\.io|googlegroups\.com|linuxfoundation\.org|opensuse\.org)$",
    re.I)
PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "kr_phone": re.compile(r"(?<!\d)01[016789][- .]?\d{3,4}[- .]?\d{4}(?!\d)"),
    "kr_rrn": re.compile(r"(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])-[1-4]\d{6}(?!\d)"),   # 주민등록번호
    "card": re.compile(r"(?<!\d)(?:\d{4}[- ]){3}\d{4}(?!\d)"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----\n[A-Za-z0-9+/=]{40,}"),
}
CARD_EXAMPLES = {"1111222233334444", "5555666677778888", "4111111111111111", "1234567812345678"}


def luhn_ok(digits):
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


def find_pii(text):
    """(종류, 값) 목록. 예시 도메인 이메일과 Luhn 검사를 통과하지 못하거나 문서용 예시인 카드 번호는 제외."""
    hits = []
    for kind, pat in PII_PATTERNS.items():
        for m in pat.finditer(text):
            v = m.group(0)
            if kind == "email" and EMAIL_ALLOW.search(v):
                continue
            if kind == "card":
                digits = re.sub(r"\D", "", v)
                if digits in CARD_EXAMPLES or not luhn_ok(digits):
                    continue
            hits.append((kind, v))
    return hits


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def data_samples(data_dir):
    """[(종류, 정규화 전 텍스트, 유효한가)] — CPT 블록과 SFT 대화."""
    d = Path(data_dir)
    out = []
    for r in read_jsonl(d / "cpt" / "train.jsonl"):
        t = r.get("text", "")
        out.append(("cpt", t, len(t.strip()) >= 200))
    for r in read_jsonl(d / "sft" / "train.jsonl"):
        msgs = r.get("messages", [])
        roles_ok = [m.get("role") for m in msgs] == ["system", "user", "assistant"]
        q = msgs[1]["content"] if roles_ok else ""
        a = msgs[2]["content"] if roles_ok else ""
        out.append(("sft", f"{q}\n{a}", roles_ok and len(q.strip()) >= 5 and len(a.strip()) >= 20))
    return out


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def token_distribution(data_dir, tokenizer, cpt_max_tokens, sft_max_length):
    """토큰 길이 분포. 학습 max_length 를 넘는 샘플은 학습 때 잘려 나가므로 비율을 관문에서 막는다."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    d = Path(data_dir)
    cpt = [len(tok.encode(r["text"], add_special_tokens=False)) for r in read_jsonl(d / "cpt" / "train.jsonl")]
    # apply_chat_template(tokenize=True) 는 transformers 5 에서 dict 를 돌려준다 → 문자열로 만든 뒤 encode
    sft = [len(tok.encode(tok.apply_chat_template(r["messages"], tokenize=False), add_special_tokens=False))
           for r in read_jsonl(d / "sft" / "train.jsonl")]
    out = {"total_tokens": sum(cpt) + sum(sft)}
    for name, xs, limit in (("cpt", cpt, cpt_max_tokens), ("sft", sft, sft_max_length)):
        out.update({f"{name}_tokens_p50": percentile(xs, 0.5), f"{name}_tokens_p95": percentile(xs, 0.95),
                    f"{name}_tokens_max": max(xs, default=0),
                    f"{name}_overlength_ratio": sum(x > limit for x in xs) / max(1, len(xs))})
    out["overlength_ratio"] = max(out["cpt_overlength_ratio"], out["sft_overlength_ratio"])
    return out


def data_metrics(data_dir, forbidden_words=(), tokenizer=None, cpt_max_tokens=8192, sft_max_length=2048):
    samples = data_samples(data_dir)
    keys = [hashlib.sha1(_norm(t).encode()).hexdigest() for _, t, _ in samples]
    dup = sum(n - 1 for n in Counter(keys).values())
    pii = [h for _, t, _ in samples for h in find_pii(t)]
    words = [w.strip() for w in forbidden_words if w.strip()]
    forbidden = Counter(w for _, t, _ in samples for w in words if w in t)
    return {
        "n_samples": len(samples),
        "n_cpt": sum(k == "cpt" for k, _, _ in samples),
        "n_sft": sum(k == "sft" for k, _, _ in samples),
        "dup_ratio": dup / max(1, len(samples)),
        "valid_count": sum(v for _, _, v in samples),
        "pii_count": len(pii),
        "pii_by_kind": dict(Counter(k for k, _ in pii)),
        "pii_examples": [f"{k}:{v[:4]}***" for k, v in pii[:5]],     # 로그에 원문을 남기지 않는다
        "forbidden_count": sum(forbidden.values()),
        "forbidden_by_word": dict(forbidden),
        **(token_distribution(data_dir, tokenizer, cpt_max_tokens, sft_max_length) if tokenizer else {}),
    }


def data_checks(m, max_dup_ratio=0.05, min_valid_count=10000, max_pii_count=0, max_forbidden_count=0,
                max_overlength_ratio=0.01):
    checks = [
        check("dup_ratio", m["dup_ratio"], "<", max_dup_ratio),
        check("valid_count", m["valid_count"], ">=", min_valid_count),
        check("pii_count", m["pii_count"], "<=", max_pii_count),
        check("forbidden_count", m["forbidden_count"], "<=", max_forbidden_count),
    ]
    if "overlength_ratio" in m:   # 토큰 분포: 학습 max_length 를 넘어 잘리는 샘플 비율
        checks.append(check("overlength_ratio", m["overlength_ratio"], "<=", max_overlength_ratio))
    return checks


# ── 3. Gate 2: CPT ──────────────────────────────────────────────────────────
def cpt_metrics(base_model, cpt_model, eval_dir, max_tokens=8192):
    """BASE 와 CPT 의 perplexity 3종 (05_eval_cpt.py 의 CUDA 백엔드 재사용)."""
    ev = import_module("05_eval_cpt")
    backend = ev.CUDABackend()
    ppl = {}
    for label, tuned in [("base", None), ("cpt", cpt_model)]:
        backend.load(base_model, tuned)
        for name in ev.EVAL_FILES:
            f = Path(eval_dir) / f"{name}.jsonl"
            if f.exists():
                ppl[f"{label}_{name}"] = backend.perplexity(f, max_tokens)
        backend.unload()
    return derive_cpt(ppl)


def derive_cpt(ppl):
    m = {f"ppl_{k}": v for k, v in ppl.items()}
    m["domain_ppl_drop"] = (ppl["base_k8s_heldout"] - ppl["cpt_k8s_heldout"]) / ppl["base_k8s_heldout"]
    m["general_ppl_rise"] = (ppl["cpt_general_ko"] - ppl["base_general_ko"]) / ppl["base_general_ko"]
    if "base_k8s_seen" in ppl:
        m["seen_ppl_drop"] = (ppl["base_k8s_seen"] - ppl["cpt_k8s_seen"]) / ppl["base_k8s_seen"]
    return m


def cpt_checks(m, min_domain_ppl_drop=0.15, max_general_ppl_rise=0.05):
    return ([
        check("domain_ppl_drop (k8s_heldout)", m["domain_ppl_drop"], ">=", min_domain_ppl_drop),
        check("general_ppl_rise (general_ko)", m["general_ppl_rise"], "<", max_general_ppl_rise),
    ])


# ── 4. Gate 3: SFT ──────────────────────────────────────────────────────────
def eval_prompts(eval_dir, n, seed=0):
    """data/eval/sft_eval.jsonl 에서 생성 평가용 질문. 실무·일반 질문은 전부, 나머지는 held-out 에서 채운다."""
    rows = read_jsonl(Path(eval_dir) / "sft_eval.jsonl")
    fixed = [r for r in rows if r["kind"].startswith("practical")]
    rest = [r for r in rows if not r["kind"].startswith("practical")]
    random.Random(seed).shuffle(rest)
    return (fixed + rest)[: max(n, len(fixed))]


def generate_answers(model_path, rows, max_new_tokens=512, batch_size=16):
    """[{"answer", "n_tokens", "stopped"}] — greedy, repetition_penalty 1.1 (07_eval_sft.py 와 같은 조건)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device).eval()
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    out = []
    for s in range(0, len(rows), batch_size):
        batch = rows[s:s + batch_size]
        texts = [tok.apply_chat_template([{"role": "system", "content": r["system"]},
                                          {"role": "user", "content": r["question"]}],
                                         tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1,
                                 eos_token_id=stop, pad_token_id=stop[-1])
        for seq in gen[:, enc.input_ids.shape[1]:].tolist():
            stopped = any(t in stop for t in seq)
            body = seq[: next((i for i, t in enumerate(seq) if t in stop), len(seq))]
            out.append({"answer": tok.decode(body, skip_special_tokens=True).strip(),
                        "n_tokens": len(body), "stopped": stopped})
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


ROLE_LEAK = re.compile(r"<\|im_(?:start|end)\|>|<\|endoftext\|>|^\s*(?:user|assistant|system)\s*\n", re.M)


def format_issues(answer):
    """대화 형식이 깨진 흔적. 특수 토큰이 글자로 새어 나오거나, 다음 턴을 지어내거나, 같은 줄을 반복한다."""
    issues = []
    if not answer.strip():
        issues.append("empty")
    if ROLE_LEAK.search(answer):
        issues.append("role_leak")
    lines = [l.strip() for l in answer.splitlines() if len(l.strip()) >= 10]
    if lines and max(Counter(lines).values()) >= 3:
        issues.append("repetition")
    return issues


def sft_metrics_from(gens):
    bad = [g for g in gens if format_issues(g.get("answer", ""))]
    return {
        "format_error_rate": len(bad) / max(1, len(gens)),
        "format_errors": dict(Counter(i for g in gens for i in format_issues(g.get("answer", "")))),
        "n_prompts": len(gens),
        "stop_rate": sum(g["stopped"] for g in gens) / max(1, len(gens)),
        "avg_len": sum(g["n_tokens"] for g in gens) / max(1, len(gens)),     # 답변 토큰 수 (종료 토큰 제외)
        "max_len": max((g["n_tokens"] for g in gens), default=0),
    }


def sft_checks(m, min_stop_rate=0.95, min_avg_len=20, max_avg_len=480, max_format_error_rate=0.05):
    return ([
        check("stop_rate", m["stop_rate"], ">=", min_stop_rate),
        check("format_error_rate", m["format_error_rate"], "<=", max_format_error_rate),
        check("avg_len (tokens)", m["avg_len"], ">=", min_avg_len),
        check("avg_len (tokens)", m["avg_len"], "<=", max_avg_len),
    ])


# ── 5. Gate 4: DPO ──────────────────────────────────────────────────────────
PAIRWISE_PROMPT = """당신은 한국어 AI 어시스턴트의 답변 두 개를 비교하는 엄격한 심사위원입니다.

<질문>
{question}
</질문>
{reference}
<답변A>
{a}
</답변A>

<답변B>
{b}
</답변B>

어느 답변이 더 나은지 판단하세요. 기준: 사실 정확성(가장 중요), 질문에 직접 답하는가, 자연스러운 한국어.
**길이는 기준이 아닙니다.** 같은 내용이면 간결한 쪽이 낫습니다. 차이가 없으면 TIE.
출력은 JSON 하나만: {{"winner": "A" | "B" | "TIE"}}"""


def judge_once(api_base, model, question, reference, a, b, api_key="EMPTY"):
    synth = import_module("synth_qa")
    ref = f"\n<참고 답변 (공식 문서 기반, 사실 판단의 근거)>\n{reference[:3000]}\n</참고 답변>\n" if reference else ""
    text = synth.call(api_base, api_key, model, PAIRWISE_PROMPT.format(question=question, reference=ref, a=a, b=b),
                      temperature=0.0)
    m = re.search(r'"winner"\s*:\s*"(A|B|TIE)"', text)
    return m.group(1) if m else None


def pairwise_score(api_base, model, question, reference, cand, base, api_key="EMPTY"):
    """위치 편향을 없애려고 순서를 바꿔 두 번 묻는다. cand 가 이기면 1, 지면 0, 비기거나 판정이 엇갈리면 0.5."""
    r1 = judge_once(api_base, model, question, reference, cand, base, api_key)   # cand = A
    r2 = judge_once(api_base, model, question, reference, base, cand, api_key)   # cand = B
    if r1 is None or r2 is None:
        return None
    s = {"A": 1.0, "B": 0.0, "TIE": 0.5}
    return (s[r1] + (1 - s[r2])) / 2


def dpo_metrics_from(rows, sft_gens, dpo_gens, api_base, judge_model, workers=16, api_key="EMPTY"):
    from concurrent.futures import ThreadPoolExecutor

    def work(i):
        return pairwise_score(api_base, judge_model, rows[i]["question"], rows[i].get("reference"),
                              dpo_gens[i]["answer"], sft_gens[i]["answer"], api_key)

    with ThreadPoolExecutor(workers) as pool:
        scores = list(pool.map(work, range(len(rows))))
    valid = [s for s in scores if s is not None]
    sft_len = sum(g["n_tokens"] for g in sft_gens) / max(1, len(sft_gens))
    dpo_len = sum(g["n_tokens"] for g in dpo_gens) / max(1, len(dpo_gens))
    return {
        "n_prompts": len(rows), "n_judged": len(valid),
        "win_rate": sum(valid) / len(valid) if valid else float("nan"),
        "sft_avg_len": sft_len, "dpo_avg_len": dpo_len,
        "len_increase": (dpo_len - sft_len) / max(1e-9, sft_len),
        "dpo_stop_rate": sum(g["stopped"] for g in dpo_gens) / max(1, len(dpo_gens)),
    }


def dpo_checks(m, min_win_rate=0.55, max_len_increase=0.20, min_judged_ratio=0.9):
    return ([
        check("win_rate (vs SFT)", m["win_rate"], ">=", min_win_rate),
        check("len_increase (vs SFT)", m["len_increase"], "<", max_len_increase),
        check("judged_ratio", m["n_judged"] / max(1, m["n_prompts"]), ">=", min_judged_ratio),   # 심판 장애로 표본이 줄면 막는다
    ])


# ── 6. MLflow ───────────────────────────────────────────────────────────────
def mlflow_client(tracking_uri, experiment):
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment)
    return MlflowClient(), exp.experiment_id


def start_parent_run(tracking_uri, experiment, kfp_run_id, kfp_run_name, params, run_name=None):
    """파이프라인 실행 하나 = MLflow 부모 run 하나 (이름 예: qwen25-3b-pipeline-v1).
    이름은 실행마다 같을 수 있으므로, 종료 핸들러는 이름이 아니라 kfp.run_id 태그로 찾는다."""
    client, exp_id = mlflow_client(tracking_uri, experiment)
    run = client.create_run(exp_id, run_name=(run_name or f"pipeline-{kfp_run_name}")[:250],
                            tags={"kfp.run_id": kfp_run_id, "kfp.run_name": kfp_run_name, "llm_ft.level": "pipeline"})
    for k, v in params.items():
        client.log_param(run.info.run_id, k, str(v)[:6000])
    return run.info.run_id


def find_parent_run(tracking_uri, experiment, kfp_run_id):
    client, exp_id = mlflow_client(tracking_uri, experiment)
    runs = client.search_runs([exp_id], f"tags.`kfp.run_id` = '{kfp_run_id}' and tags.`llm_ft.level` = 'pipeline'",
                              max_results=1)
    return (client, runs[0].info.run_id) if runs else (client, None)


def failed_gates(client, experiment_id, parent_run_id):
    """종료 핸들러용: 부모 아래에서 실패한 gate 와 실패 조건. KFP 의 error_message 가 일반 문구일 때도 원인을 알린다."""
    runs = client.search_runs([experiment_id], f"tags.mlflow.parentRunId = '{parent_run_id}' "
                                               f"and tags.`llm_ft.gate_passed` = 'False'")
    return [(r.data.tags.get("llm_ft.gate"), r.data.tags.get("llm_ft.gate_failed_checks", "")) for r in runs]


def log_child_run(tracking_uri, experiment, parent_run_id, name, metrics=None, params=None, tags=None, text=None):
    """부모 아래 중첩 run (게이트, 등록, 배포). 숫자는 metric, 나머지는 param 으로."""
    client, exp_id = mlflow_client(tracking_uri, experiment)
    run = client.create_run(exp_id, run_name=name,
                            tags={"mlflow.parentRunId": parent_run_id, "llm_ft.level": "step", **(tags or {})})
    rid = run.info.run_id
    for k, v in (metrics or {}).items():
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            client.log_metric(rid, safe_key(k), float(v))
        else:
            client.log_param(rid, safe_key(k), json.dumps(v, ensure_ascii=False)[:6000])
    for k, v in (params or {}).items():
        client.log_param(rid, safe_key(k), str(v)[:6000])
    for fname, content in (text or {}).items():
        client.log_text(rid, content, fname)
    return client, rid


def safe_key(k):
    return re.sub(r"[^\w\-./ ]", "_", k)[:250]


def log_gate(tracking_uri, experiment, parent_run_id, gate, metrics, checks, passed, train_run_id=None):
    params = {f"threshold.{c['name']}": f"{c['op']} {c['threshold']}" for c in checks}
    m = dict(metrics)
    m.update({f"check.{c['name']}.passed": int(c["passed"]) for c in checks})
    m["gate_passed"] = int(passed)
    failed = "; ".join(f"{c['name']}={fmt(c['value'])} (조건 {c['op']} {fmt(c['threshold'])})"
                       for c in checks if not c["passed"])
    client, rid = log_child_run(tracking_uri, experiment, parent_run_id, gate, m, params,
                                tags={"llm_ft.gate": gate, "llm_ft.gate_passed": str(passed),
                                      "llm_ft.gate_failed_checks": failed[:5000]})
    if train_run_id:
        link_train_run(client, parent_run_id, train_run_id)
        client.log_param(rid, "train_run_id", train_run_id)
    client.set_terminated(rid, "FINISHED" if passed else "FAILED")
    return rid


def link_train_run(client, parent_run_id, train_run_id):
    """학습 run 은 KFP 캐시를 깨지 않으려고 부모 run id 를 입력으로 받지 않는다 (components.py 참고).
    처음 학습된 파이프라인 실행의 부모 아래로 한 번만 연결한다. 캐시로 재사용된 경우는 원래 부모에 그대로 둔다."""
    tags = client.get_run(train_run_id).data.tags
    if "mlflow.parentRunId" not in tags:
        client.set_tag(train_run_id, "mlflow.parentRunId", parent_run_id)
    else:
        client.set_tag(train_run_id, f"llm_ft.reused_by.{parent_run_id}", "cached")


def log_training_run(tracking_uri, experiment, stage, params, output_dir):
    """학습 run (부모 없음, 나중에 gate 가 연결). trainer_state.json 의 loss 곡선을 step 별 metric 으로 남긴다."""
    client, exp_id = mlflow_client(tracking_uri, experiment)
    run = client.create_run(exp_id, run_name=f"train-{stage}", tags={"llm_ft.level": "train", "llm_ft.stage": stage})
    rid = run.info.run_id
    for k, v in params.items():
        client.log_param(rid, safe_key(k), str(v)[:6000])
    states = sorted(Path(output_dir).glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if states:
        for h in json.loads(states[-1].read_text())["log_history"]:
            for k, v in h.items():
                if k not in ("step", "epoch") and isinstance(v, (int, float)):
                    client.log_metric(rid, safe_key(k), float(v), step=int(h.get("step", 0)))
    ppo_log = Path(output_dir) / "log.jsonl"
    if ppo_log.exists():
        for h in read_jsonl(ppo_log):
            for k, v in h.items():
                if k != "step" and isinstance(v, (int, float)):
                    client.log_metric(rid, safe_key(k), float(v), step=int(h["step"]))
    client.set_terminated(rid, "FINISHED")
    return rid


def to_s3_uri(uri):
    """KFP 의 artifact URI (minio://bucket/key, s3://, gs://) → vLLM 초기화 컨테이너가 받을 수 있는 s3:// 형식."""
    return re.sub(r"^minio://", "s3://", uri)


def register_model(tracking_uri, experiment, parent_run_id, model_name, source_uri, train_run_id, alias, tags):
    """Model Registry 에 새 버전 등록 + alias 지정. 모델 파일은 복사하지 않고 KFP artifact 위치를 source 로 가리킨다."""
    client, _ = mlflow_client(tracking_uri, experiment)
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name, description="k8s-ko 3B: CPT → SFT → DPO (KFP 파이프라인)")
    mv = client.create_model_version(model_name, source=to_s3_uri(source_uri), run_id=train_run_id,
                                     tags={k: str(v) for k, v in tags.items()})
    client.set_registered_model_alias(model_name, alias, mv.version)
    log_child_run(tracking_uri, experiment, parent_run_id, "register",
                  params={"model_name": model_name, "version": mv.version, "alias": alias, "source": mv.source})
    return mv.version, mv.source


# ── 7. vLLM 무중단 배포 ──────────────────────────────────────────────────────
MODEL_URI_ENV = "MODEL_URI"
VERSION_ANNOTATION = "llm-ft/model-version"


def _k8s():
    from kubernetes import client, config
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.AppsV1Api()


def current_model(dep):
    """Deployment 의 pod template 에서 (MODEL_URI, 모델 버전 annotation)."""
    tpl = dep.spec.template
    uri = None
    for c in (tpl.spec.init_containers or []) + tpl.spec.containers:
        for e in c.env or []:
            if e.name == MODEL_URI_ENV:
                uri = e.value
    return uri, (tpl.metadata.annotations or {}).get(VERSION_ANNOTATION)


def set_model(dep, uri, version):
    found = False
    for c in (dep.spec.template.spec.init_containers or []) + dep.spec.template.spec.containers:
        for e in c.env or []:
            if e.name == MODEL_URI_ENV:
                e.value = uri
                found = True
    if not found:
        raise RuntimeError(f"Deployment 에 {MODEL_URI_ENV} 환경변수가 없습니다 (pipelines/k8s/vllm-serving.yaml 참고)")
    dep.spec.template.metadata.annotations = {**(dep.spec.template.metadata.annotations or {}),
                                              VERSION_ANNOTATION: str(version)}


def check_zero_downtime(dep):
    """무중단 조건: maxUnavailable=0 (새 pod 가 Ready 가 된 뒤에만 옛 pod 를 내림) + readinessProbe."""
    problems = []
    ru = dep.spec.strategy.rolling_update if dep.spec.strategy else None
    if not dep.spec.strategy or dep.spec.strategy.type != "RollingUpdate":
        problems.append("strategy.type 이 RollingUpdate 가 아님")
    elif ru is None or str(ru.max_unavailable) not in ("0", "0%"):
        problems.append(f"maxUnavailable={getattr(ru, 'max_unavailable', None)} (0 이어야 함)")
    if not any(c.readiness_probe for c in dep.spec.template.spec.containers):
        problems.append("readinessProbe 없음")
    return problems


def wait_rollout(api, name, namespace, timeout_s, poll_s=15):
    """kubectl rollout status 와 같은 판정. 새 ReplicaSet 의 pod 가 모두 Ready 이고 옛 pod 가 남지 않으면 완료."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        d = api.read_namespaced_deployment(name, namespace)
        st, want = d.status, d.spec.replicas or 1
        if (d.metadata.generation <= (st.observed_generation or 0) and (st.updated_replicas or 0) == want
                and (st.available_replicas or 0) == want and (st.replicas or 0) == want):
            return True
        for c in st.conditions or []:
            if c.type == "Progressing" and c.reason == "ProgressDeadlineExceeded":
                return False
        print(f"  rollout: updated {st.updated_replicas}/{want}, available {st.available_replicas}, total {st.replicas}")
        time.sleep(poll_s)
    return False


def smoke_test(service_url, served_model, timeout_s=120):
    """배포된 엔드포인트에 실제 질문을 보낸다. 200 + 비어 있지 않은 답 + finish_reason=stop 이어야 통과."""
    body = json.dumps({"model": served_model, "max_tokens": 200, "temperature": 0,
                       "messages": [{"role": "user", "content": "파드(Pod)가 무엇인지 한 문장으로 설명해 주세요."}]}).encode()
    req = urllib.request.Request(f"{service_url.rstrip('/')}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        choice = json.load(r)["choices"][0]
    answer, reason = choice["message"]["content"].strip(), choice.get("finish_reason")
    print(f"  smoke test: finish_reason={reason}  answer={answer[:120]!r}")
    if not answer or reason != "stop":
        raise RuntimeError(f"smoke test 실패: finish_reason={reason}, answer={answer[:80]!r}")
    return answer


def deploy_rolling(name, namespace, model_uri, version, service_url, served_model, timeout_s=1800):
    """1) 무중단 조건 확인 2) pod template 의 MODEL_URI 교체 → 롤링 업데이트 3) 완료 대기 4) smoke test.
    3·4 가 실패하면 이전 MODEL_URI 로 되돌리고(롤백) 예외를 던진다."""
    api = _k8s()
    dep = api.read_namespaced_deployment(name, namespace)
    problems = check_zero_downtime(dep)
    if problems:
        raise RuntimeError("무중단 배포 조건 불충족: " + ", ".join(problems))
    old_uri, old_version = current_model(dep)
    if old_uri == model_uri:
        print(f"이미 배포된 모델입니다 ({model_uri}). smoke test 만 합니다.")
        smoke_test(service_url, served_model)
        return {"old_version": old_version, "new_version": version, "rolled_back": False, "skipped": True}
    print(f"배포: {old_version} ({old_uri}) → {version} ({model_uri})")
    set_model(dep, model_uri, version)
    api.replace_namespaced_deployment(name, namespace, dep)
    error = None
    if not wait_rollout(api, name, namespace, timeout_s):
        error = "롤링 업데이트 시간 초과 / ProgressDeadlineExceeded"
    else:
        try:
            smoke_test(service_url, served_model)
        except Exception as e:
            error = f"smoke test 실패: {e}"
    if error:
        print(f"{error} → 이전 모델로 롤백")
        dep = api.read_namespaced_deployment(name, namespace)
        set_model(dep, old_uri, old_version)
        api.replace_namespaced_deployment(name, namespace, dep)
        wait_rollout(api, name, namespace, timeout_s)
        raise RuntimeError(f"배포 실패 ({error}), {old_version} 으로 롤백함")
    return {"old_version": old_version, "new_version": version, "rolled_back": False, "skipped": False}


def promote_alias(tracking_uri, experiment, model_name, version, champion="champion", previous="previous"):
    # challenger(등록 직후) → 배포·smoke test 성공 → champion. 옛 champion → previous
    """배포가 성공한 뒤에만 champion 을 옮긴다. 옛 champion 은 previous 로 남겨 즉시 되돌릴 수 있게 한다."""
    client, _ = mlflow_client(tracking_uri, experiment)
    try:
        old = client.get_model_version_by_alias(model_name, champion).version
    except Exception:
        old = None
    if old and str(old) != str(version):
        client.set_registered_model_alias(model_name, previous, old)
    client.set_registered_model_alias(model_name, champion, version)
    return old


# ── 8. Slack ────────────────────────────────────────────────────────────────
def slack_post(webhook, text):
    if not webhook:
        print("SLACK_WEBHOOK_URL 이 없어 알림을 건너뜁니다:\n" + text)
        return False
    req = urllib.request.Request(webhook, data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status == 200
