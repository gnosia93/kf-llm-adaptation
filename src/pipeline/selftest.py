#!/usr/bin/env python3
"""클러스터 없이 파이프라인 로직을 검사한다 (GPU 학습 제외).

  python -m pipelines.selftest                 # 로컬 sqlite MLflow, 가짜 심판·가짜 Kubernetes
  python -m pipelines.selftest --data data     # Gate 1 을 실제 데이터로

검사 항목
  1. Gate 1 지표 (실제 data/): 중복률, 유효 샘플 수, 개인정보 탐지 (+ 일부러 넣은 개인정보를 잡는지)
  2. Gate 1~4 판정: 통과 / 실패 경계값
  3. MLflow: 부모 run → 게이트 중첩 run, 학습 run 연결(캐시 재사용 포함), Registry 등록, alias 승격, 부모 찾기
  4. Gate 4 심판: 순서를 바꿔 두 번 묻고 점수를 합치는 로직 (가짜 OpenAI 호환 서버)
  5. 배포: 무중단 조건 검사, MODEL_URI 교체, 롤아웃 실패 시 롤백 (가짜 AppsV1Api)
"""
import argparse
import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from pipelines import lib

PASSED = []


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    PASSED.append(msg)
    print(f"  ✓ {msg}")


def test_data(data_dir):
    print("[1] Gate 1 데이터 지표")
    m = lib.data_metrics(data_dir, ["금지어테스트"], tokenizer="Qwen/Qwen2.5-3B", cpt_max_tokens=8192, sft_max_length=2048)
    keys = ("n_cpt", "n_sft", "dup_ratio", "valid_count", "pii_count", "cpt_tokens_p50", "cpt_tokens_p95",
            "cpt_tokens_max", "sft_tokens_p95", "sft_tokens_max", "overlength_ratio", "total_tokens")
    print(f"    {json.dumps({k: m[k] for k in keys}, ensure_ascii=False)}")
    ok(m["cpt_tokens_max"] <= 8192, "토큰 분포: CPT 블록이 모두 8,192 토큰 이하 (03_build_cpt.py 패킹과 일치)")
    ok(100 < m["sft_tokens_p50"] < m["sft_tokens_max"] <= 4096, "토큰 분포: SFT 대화 길이가 실제 토큰 수로 계산됨")
    ok(m["pii_count"] == 0, "쿠버네티스 문서의 예시 이메일·카드번호는 개인정보로 잡지 않음")
    hits = lib.find_pii("문의: hong.gildong@gmail.com, 010-1234-5678, 900101-1234567, 4532 0151 1283 0366, AKIAIOSFODNN7EXAMPL0 "
                        "예시: user@example.com, 1111 2222 3333 4444, git@github.com")
    kinds = sorted(k for k, _ in hits)
    ok(kinds == ["aws_access_key", "card", "email", "kr_phone", "kr_rrn"], f"진짜 개인정보 5종은 잡음 ({kinds})")
    checks = lib.data_checks(m)
    by = {c["name"]: c["passed"] for c in checks}
    ok(by["dup_ratio"] and by["pii_count"] and by["overlength_ratio"], "중복률·개인정보·토큰 길이 조건 통과")
    decision, summary = lib.gate_outcome("Gate 1 (Data)", checks)
    print(f"    스펙 기본값으로 판정: {decision}  ({summary})")
    return m


def test_gates():
    print("[2] Gate 판정 경계")
    cpt = lib.derive_cpt({"base_k8s_heldout": 10.0, "cpt_k8s_heldout": 8.5, "base_general_ko": 5.0, "cpt_general_ko": 5.2})
    ok(all(c["passed"] for c in lib.cpt_checks(cpt)), "CPT: 도메인 -15.0%, 일반 +4.0% → 통과 (15% 는 '이상' 이라 포함)")
    cpt_bad = lib.derive_cpt({"base_k8s_heldout": 10.0, "cpt_k8s_heldout": 9.0, "base_general_ko": 5.0, "cpt_general_ko": 5.3})
    failed = [c["name"] for c in lib.cpt_checks(cpt_bad) if not c["passed"]]
    ok(len(failed) == 2, f"CPT: 도메인 -10%, 일반 +6% → 두 조건 모두 실패 ({failed})")
    decision, summary = lib.gate_outcome("Gate 2 (CPT)", lib.cpt_checks(cpt_bad))
    ok(decision == "FAIL" and "domain_ppl_drop" in summary and "general_ppl_rise" in summary,
       "FAIL 요약에 실패 조건이 모두 들어감 (stop_pipeline → Slack 에 실림)")
    ok(lib.gate_outcome("Gate 2 (CPT)", lib.cpt_checks(cpt))[0] == "PASS", "통과하면 decision=PASS (dsl.If 분기)")
    ok(lib.format_issues("파드는 컨테이너 묶음이다.") == [], "포맷: 정상 답변")
    ok("role_leak" in lib.format_issues("파드는 ...\nuser\n그럼 디플로이먼트는?"), "포맷: 다음 user 턴을 지어냄 → role_leak")
    ok("role_leak" in lib.format_issues("답<|im_end|>"), "포맷: 특수 토큰이 글자로 샘 → role_leak")
    ok("repetition" in lib.format_issues("\n".join(["파드는 가장 작은 배포 단위이다."] * 3)), "포맷: 같은 줄 3번 → repetition")
    gens = [{"answer": "정상 답변", "n_tokens": 100, "stopped": True}] * 19 + [{"answer": "끝없이", "n_tokens": 512, "stopped": False}]
    m = lib.sft_metrics_from(gens)
    ok(abs(m["stop_rate"] - 0.95) < 1e-9 and all(c["passed"] for c in lib.sft_checks(m)), "SFT: stop_rate 0.95 → 통과 (경계 포함)")
    m2 = lib.sft_metrics_from([{"answer": "짧음", "n_tokens": 10, "stopped": True}] * 10)
    ok(not all(c["passed"] for c in lib.sft_checks(m2)), "SFT: avg_len 10 → 실패")
    d = {"win_rate": 0.6, "len_increase": 0.25, "n_judged": 60, "n_prompts": 60}
    failed = [c["name"] for c in lib.dpo_checks(d) if not c["passed"]]
    ok(failed == ["len_increase (vs SFT)"], "DPO: 승률 0.6 이어도 길이 +25% 면 실패 (verbosity bias)")
    d = {"win_rate": float("nan"), "len_increase": 0.0, "n_judged": 0, "n_prompts": 60}
    ok(not any(c["passed"] for c in lib.dpo_checks(d)[:1]), "DPO: 심판 장애로 승률이 NaN 이면 실패")


def test_mlflow(tmp):
    print("[3] MLflow 계층 기록 · Registry")
    uri, exp = f"sqlite:///{tmp}/mlflow.db", "selftest"
    parent = lib.start_parent_run(uri, exp, "kfp-run-1", "run-1", {"base_model": "Qwen/Qwen2.5-3B"},
                                  run_name="qwen25-3b-pipeline-v1")
    ok(lib.mlflow_client(uri, exp)[0].get_run(parent).info.run_name == "qwen25-3b-pipeline-v1", "부모 run 이름")
    train = lib.log_training_run(uri, exp, "cpt", {"epochs": 4}, tmp)
    checks = lib.cpt_checks(lib.derive_cpt({"base_k8s_heldout": 10.0, "cpt_k8s_heldout": 8.0,
                                            "base_general_ko": 5.0, "cpt_general_ko": 5.1}))
    lib.log_gate(uri, exp, parent, "gate2-cpt", {"domain_ppl_drop": 0.2}, checks, True, train)
    client, _ = lib.mlflow_client(uri, exp)
    ok(client.get_run(train).data.tags["mlflow.parentRunId"] == parent, "학습 run 이 첫 실행의 부모 아래로 연결됨")
    parent2 = lib.start_parent_run(uri, exp, "kfp-run-2", "run-2", {}, run_name="qwen25-3b-pipeline-v1")   # 같은 이름
    lib.log_gate(uri, exp, parent2, "gate2-cpt", {"domain_ppl_drop": 0.2}, checks, True, train)   # 캐시된 학습 재사용
    tags = client.get_run(train).data.tags
    ok(tags["mlflow.parentRunId"] == parent and tags.get(f"llm_ft.reused_by.{parent2}") == "cached",
       "캐시 재사용 시 원래 부모 유지 + 재사용 표시")
    children = client.search_runs([client.get_experiment_by_name(exp).experiment_id],
                                  f"tags.mlflow.parentRunId = '{parent}'")
    ok(sorted(r.data.tags["mlflow.runName"] for r in children) == ["gate2-cpt", "train-cpt"], "부모 아래 gate·train 중첩 run")
    ok(lib.find_parent_run(uri, exp, "kfp-run-2")[1] == parent2, "부모 run 이름이 같아도 KFP run id 로 올바른 부모를 찾음")
    v1, src = lib.register_model(uri, exp, parent, "k8s-ko-3b", "minio://mlpipeline/v2/artifacts/a/model", train,
                                 "challenger", {"stage": "dpo"})
    ok(src == "s3://mlpipeline/v2/artifacts/a/model", "minio:// artifact URI 를 s3:// 로 바꿔 등록")
    ok(lib.promote_alias(uri, exp, "k8s-ko-3b", v1) is None, "첫 배포: champion = v1")
    ok(client.get_model_version_by_alias("k8s-ko-3b", "challenger").version == v1, "등록 직후 @challenger")
    v2, _ = lib.register_model(uri, exp, parent2, "k8s-ko-3b", "s3://b/model", train, "challenger", {})
    lib.promote_alias(uri, exp, "k8s-ko-3b", v2)
    ok(client.get_model_version_by_alias("k8s-ko-3b", "champion").version == v2
       and client.get_model_version_by_alias("k8s-ko-3b", "previous").version == v1, "두 번째 배포: champion=v2, previous=v1")


class FakeJudge(BaseHTTPRequestHandler):
    """'좋은답' 이 들어 있는 쪽을 이기게 한다. 항상 A 를 고르는 위치 편향 모드도 있다."""
    biased = False

    def do_POST(self):
        prompt = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["messages"][0]["content"]
        a = prompt.split("<답변A>")[1].split("</답변A>")[0]
        b = prompt.split("<답변B>")[1].split("</답변B>")[0]
        w = "A" if FakeJudge.biased else ("A" if "좋은답" in a else "B" if "좋은답" in b else "TIE")
        body = json.dumps({"choices": [{"message": {"content": f'{{"winner": "{w}"}}'}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_judge():
    print("[4] Gate 4 pairwise 심판")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeJudge)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    api = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    rows = [{"question": f"q{i}", "reference": "ref"} for i in range(4)]
    sft = [{"answer": "보통", "n_tokens": 100, "stopped": True}] * 4
    dpo = [{"answer": "좋은답", "n_tokens": 110, "stopped": True}] * 3 + [{"answer": "보통", "n_tokens": 110, "stopped": True}]
    m = lib.dpo_metrics_from(rows, sft, dpo, api, "fake", workers=4)
    ok(abs(m["win_rate"] - (3 + 0.5) / 4) < 1e-9, f"승 3 + 무 1 → win_rate 0.875 (실제 {m['win_rate']})")
    ok(abs(m["len_increase"] - 0.10) < 1e-9, "길이 증가율 +10%")
    FakeJudge.biased = True
    m = lib.dpo_metrics_from(rows, sft, dpo, api, "fake", workers=4)
    ok(m["win_rate"] == 0.5, "항상 A 를 고르는 심판(위치 편향)은 순서 교차로 0.5 로 상쇄됨")
    srv.shutdown()


class FakeApps:
    """AppsV1Api 흉내: replace 하면 새 template 으로 롤아웃. fail_uri 로 바꾸면 롤아웃이 끝나지 않는다."""

    def __init__(self, dep, fail_uri=None):
        self.dep, self.fail_uri, self.history = dep, fail_uri, []

    def read_namespaced_deployment(self, name, ns):
        import copy
        d = copy.deepcopy(self.dep)
        uri, _ = lib.current_model(d)
        good = uri != self.fail_uri
        d.metadata.generation = 1
        d.status = SimpleNamespace(observed_generation=1, updated_replicas=1 if good else 0,
                                   available_replicas=1, replicas=1 if good else 2,
                                   conditions=[] if good else [SimpleNamespace(
                                       type="Progressing", reason="ProgressDeadlineExceeded")])
        return d

    def replace_namespaced_deployment(self, name, ns, body):
        self.history.append(lib.current_model(body)[0])
        self.dep = body


def make_deployment(max_unavailable=0, probe=True):
    from kubernetes import client as k
    env = [k.V1EnvVar(name="MODEL_URI", value="s3://old/model")]
    return k.V1Deployment(
        metadata=k.V1ObjectMeta(name="k8s-ko-vllm", generation=1),
        spec=k.V1DeploymentSpec(
            replicas=1, selector=k.V1LabelSelector(match_labels={"app": "x"}),
            strategy=k.V1DeploymentStrategy(type="RollingUpdate", rolling_update=k.V1RollingUpdateDeployment(
                max_surge=1, max_unavailable=max_unavailable)),
            template=k.V1PodTemplateSpec(
                metadata=k.V1ObjectMeta(annotations={"llm-ft/model-version": "1"}),
                spec=k.V1PodSpec(
                    init_containers=[k.V1Container(name="model-downloader", image="aws-cli", env=env)],
                    containers=[k.V1Container(name="vllm", image="vllm", readiness_probe=k.V1Probe(
                        http_get=k.V1HTTPGetAction(path="/health", port=8000))) if probe else
                        k.V1Container(name="vllm", image="vllm")]))))


def test_deploy():
    print("[5] 무중단 배포 · 롤백")
    ok(lib.check_zero_downtime(make_deployment()) == [], "maxUnavailable=0 + readinessProbe → 무중단 조건 충족")
    probs = lib.check_zero_downtime(make_deployment(max_unavailable=1, probe=False))
    ok(len(probs) == 2, f"maxUnavailable=1, probe 없음 → 거부 ({probs})")
    orig_k8s, orig_smoke = lib._k8s, lib.smoke_test
    lib.smoke_test = lambda url, model, timeout_s=120: "ok"
    try:
        fake = FakeApps(make_deployment())
        lib._k8s = lambda: fake
        r = lib.deploy_rolling("k8s-ko-vllm", "ns", "s3://new/model", "2", "http://svc", "k8s-ko", timeout_s=5)
        ok(fake.history == ["s3://new/model"] and r["new_version"] == "2", "정상 롤아웃: MODEL_URI 교체 1회")
        fake = FakeApps(make_deployment(), fail_uri="s3://bad/model")
        lib._k8s = lambda: fake
        try:
            lib.deploy_rolling("k8s-ko-vllm", "ns", "s3://bad/model", "3", "http://svc", "k8s-ko", timeout_s=5)
            ok(False, "실패해야 함")
        except RuntimeError as e:
            ok(fake.history == ["s3://bad/model", "s3://old/model"] and "롤백" in str(e),
               "롤아웃 실패 → 이전 MODEL_URI 로 롤백 후 예외")
    finally:
        lib._k8s, lib.smoke_test = orig_k8s, orig_smoke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    tmp = tempfile.mkdtemp()
    try:
        if (Path(args.data) / "cpt" / "train.jsonl").exists():
            test_data(args.data)
        else:
            print(f"[1] 건너뜀: {args.data}/cpt/train.jsonl 없음")
        test_gates()
        test_mlflow(tmp)
        test_judge()
        test_deploy()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n모두 통과: {len(PASSED)}개")


if __name__ == "__main__":
    main()
