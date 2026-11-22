"""KFP v2 컴포넌트. 실제 로직은 pipelines/lib.py 와 저장소의 학습 스크립트에 있다.

설계 원칙
  * 학습(train_*) 과 평가(eval_*) 와 관문(gate_*) 을 다른 컴포넌트로 나눈다.
      학습: GPU, 수십 분~수 시간. 입력이 같으면 KFP 캐시로 건너뛴다.
      평가: GPU, 몇 분. 학습 결과(artifact)와 평가 설정만 입력 → 평가 기준을 바꿔도 학습은 캐시.
      관문: CPU, 수 초. 평가 결과(JSON 문자열)와 임계값만 입력 → 임계값을 바꾸면 관문만 다시 돈다.
  * 학습·평가 컴포넌트는 **MLflow 부모 run id 를 입력으로 받지 않는다.** 부모 run id 는 실행마다 바뀌므로
    입력에 넣으면 캐시 키가 달라져 매번 GPU 학습을 다시 한다. 학습 run 은 혼자 만들고 run id 를 출력하면,
    같은 실행의 관문이 부모 아래로 연결한다 (lib.link_train_run).
  * 관문은 ("PASS" | "FAIL", 요약) 을 출력한다. 파이프라인은 dsl.If(PASS) 안에서만 다음 단계로 가고,
    dsl.Else(FAIL) 에서는 stop_pipeline 이 요약을 담아 예외를 던진다 → 즉시 중단 → dsl.ExitHandler 의 Slack 알림에
    '어느 Gate 가 왜 실패했는지' 가 실린다.
"""
from typing import NamedTuple
import os

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output

# 파이프라인 이미지 (pipelines/Dockerfile). 컴파일할 때 PIPELINE_IMAGE 로 바꾼다.
IMAGE = os.environ.get("PIPELINE_IMAGE", "ghcr.io/your-org/llm-ft-pipeline:latest")
COMMON = dict(base_image=IMAGE, install_kfp_package=False)   # kfp 는 이미지에 이미 들어 있다


# ── MLflow 부모 run ─────────────────────────────────────────────────────────
@dsl.component(**COMMON)
def init_mlflow_run(mlflow_uri: str, experiment: str, run_name: str, kfp_run_id: str, kfp_run_name: str,
                    params_json: str) -> str:
    import json
    from pipelines import lib
    return lib.start_parent_run(mlflow_uri, experiment, kfp_run_id, kfp_run_name, json.loads(params_json), run_name)


# ── 데이터 ──────────────────────────────────────────────────────────────────
@dsl.component(**COMMON)
def prepare_data(data_version: str, include_en: bool, cpt_max_tokens: int, replay_ratio: float,
                 n_general: int, data: Output[Dataset]):
    """01 → 02 → 03 → 06. data_version 은 문서를 새로 받고 싶을 때 바꾸는 캐시 키 (값 자체는 기록용)."""
    import os
    import subprocess
    work = "/tmp/work"
    os.makedirs(work, exist_ok=True)
    langs = "ko en" if include_en else "ko"
    docs = [f"raw/website/content/{l}/docs" for l in langs.split()]
    out = data.path
    run = lambda cmd: subprocess.run(cmd, cwd=work, check=True, env={**os.environ, "LANGS": langs})
    run(["bash", "/app/01_download.sh"])
    run(["python", "/app/02_prepare_data.py", "--docs", docs[0], "--out", out])
    run(["python", "/app/03_build_cpt.py", "--docs", *docs, "--out", out, "--max-tokens", str(cpt_max_tokens),
         "--replay-ratio", str(replay_ratio)])
    run(["python", "/app/06_build_sft.py", "--docs", docs[0], "--out", out, "--n-general", str(n_general)])
    data.metadata.update({"data_version": data_version, "langs": langs, "cpt_max_tokens": cpt_max_tokens})


@dsl.component(**COMMON)
def gate_data(data: Input[Dataset], mlflow_uri: str, experiment: str, parent_run_id: str, tokenizer: str,
              cpt_max_tokens: int, sft_max_length: int, max_dup_ratio: float, min_valid_count: int,
              max_pii_count: int, forbidden_words: str, max_overlength_ratio: float,
              metrics: Output[Metrics]) -> NamedTuple("GateOut", [("decision", str), ("summary", str)]):
    """Gate 1: 중복률 · 유효 샘플 수 · 토큰 분포 · 개인정보/금지어. 지표 계산이 가벼워서 평가와 관문을 한 컴포넌트에서 한다."""
    from collections import namedtuple
    from pipelines import lib
    m = lib.data_metrics(data.path, forbidden_words.split(","), tokenizer, cpt_max_tokens, sft_max_length)
    for k, v in m.items():
        if isinstance(v, (int, float)):
            metrics.log_metric(k, v)
    checks = lib.data_checks(m, max_dup_ratio, min_valid_count, max_pii_count, 0, max_overlength_ratio)
    decision, summary = lib.gate_outcome("Gate 1 (Data)", checks)
    lib.log_gate(mlflow_uri, experiment, parent_run_id, "gate1-data", m, checks, decision == "PASS")
    return namedtuple("GateOut", ["decision", "summary"])(decision, summary)


@dsl.component(**COMMON)
def stop_pipeline(summary: str):
    """FAIL 분기: 파이프라인을 실패로 끝낸다. 이 메시지가 ExitHandler 의 status.error_message 로 Slack 에 간다."""
    raise RuntimeError(f"Quality Gate 실패 → 파이프라인 중단. {summary}")


# ── 학습 (GPU, 캐시 대상) ────────────────────────────────────────────────────
@dsl.component(**COMMON)
def train_cpt(data: Input[Dataset], base_model: str, full: bool, epochs: float, max_length: int,
              mlflow_uri: str, experiment: str, model: Output[Model]) -> str:
    import shutil
    import subprocess
    from pipelines import lib
    work = "/tmp/cpt"
    cmd = ["python", "/app/train_cpt_hf.py", "--model", base_model, "--data", f"{data.path}/cpt", "--output", work,
           "--epochs", str(epochs), "--max-length", str(max_length)] + (["--full"] if full else [])
    subprocess.run(cmd, check=True)
    shutil.copytree(f"{work}/final", model.path, dirs_exist_ok=True)   # 체크포인트는 올리지 않는다
    model.metadata.update({"stage": "cpt", "base_model": base_model})
    return lib.log_training_run(mlflow_uri, experiment, "cpt",
                                {"base_model": base_model, "full": full, "epochs": epochs, "max_length": max_length}, work)


@dsl.component(**COMMON)
def train_sft(cpt_model: Input[Model], data: Input[Dataset], full: bool, epochs: float,
              mlflow_uri: str, experiment: str, model: Output[Model]) -> str:
    import shutil
    import subprocess
    from pipelines import lib
    work = "/tmp/sft"
    cmd = ["python", "/app/train_sft_hf.py", "--model", cpt_model.path, "--data", f"{data.path}/sft",
           "--output", work, "--epochs", str(epochs)] + (["--full"] if full else [])
    subprocess.run(cmd, check=True)
    shutil.copytree(f"{work}/final", model.path, dirs_exist_ok=True)
    model.metadata.update({"stage": "sft"})
    return lib.log_training_run(mlflow_uri, experiment, "sft", {"full": full, "epochs": epochs}, work)


@dsl.component(**COMMON)
def build_preferences(sft_model: Input[Model], data: Input[Dataset], judge_api_base: str, judge_model: str,
                      n_prompts: int, pref: Output[Dataset]):
    """SFT 모델의 답 K개 → 심판 채점 → chosen/rejected (08_build_pref.py)."""
    import subprocess
    subprocess.run(["python", "/app/08_build_pref.py", "--policy", sft_model.path,
                    "--sft-train", f"{data.path}/sft/train.jsonl", "--out", pref.path,
                    "--n-prompts", str(n_prompts), "--judge", "api",
                    "--api-base", judge_api_base, "--judge-model", judge_model], check=True)


@dsl.component(**COMMON)
def train_dpo(sft_model: Input[Model], pref: Input[Dataset], beta: float, epochs: float,
              mlflow_uri: str, experiment: str, model: Output[Model]) -> str:
    """LoRA DPO → 병합된 최종 모델 (train_dpo_hf.py 가 merge 후 저장)."""
    import shutil
    import subprocess
    from pipelines import lib
    work = "/tmp/dpo"
    subprocess.run(["python", "/app/train_dpo_hf.py", "--model", sft_model.path, "--data", pref.path,
                    "--output", work, "--beta", str(beta), "--epochs", str(epochs)], check=True)
    shutil.copytree(f"{work}/final", model.path, dirs_exist_ok=True)
    model.metadata.update({"stage": "dpo", "beta": beta})
    return lib.log_training_run(mlflow_uri, experiment, "dpo", {"beta": beta, "epochs": epochs}, work)


# ── 평가 (GPU, 캐시 대상) ────────────────────────────────────────────────────
@dsl.component(**COMMON)
def eval_cpt(base_model: str, cpt_model: Input[Model], data: Input[Dataset], max_tokens: int,
             metrics: Output[Metrics]) -> str:
    import json
    from pipelines import lib
    m = lib.cpt_metrics(base_model, cpt_model.path, f"{data.path}/eval", max_tokens)
    for k, v in m.items():
        metrics.log_metric(k, v)
    return json.dumps(m)


@dsl.component(**COMMON)
def eval_sft(sft_model: Input[Model], data: Input[Dataset], n_prompts: int, max_new_tokens: int,
             metrics: Output[Metrics], generations: Output[Artifact]) -> str:
    """생성 결과(generations)는 Gate 4 의 SFT 기준 답변으로 재사용한다 (같은 질문, 다시 생성하지 않음)."""
    import json
    from pipelines import lib
    rows = lib.eval_prompts(f"{data.path}/eval", n_prompts)
    gens = lib.generate_answers(sft_model.path, rows, max_new_tokens)
    with open(generations.path, "w", encoding="utf-8") as f:
        for r, g in zip(rows, gens):
            f.write(json.dumps({**r, **g}, ensure_ascii=False) + "\n")
    m = lib.sft_metrics_from(gens)
    for k, v in m.items():
        metrics.log_metric(k, v)
    return json.dumps(m)


@dsl.component(**COMMON)
def eval_dpo(dpo_model: Input[Model], sft_generations: Input[Artifact], judge_api_base: str, judge_model: str,
             max_new_tokens: int, metrics: Output[Metrics], generations: Output[Artifact]) -> str:
    import json
    from pipelines import lib
    sft_rows = lib.read_jsonl(sft_generations.path)
    dpo_gens = lib.generate_answers(dpo_model.path, sft_rows, max_new_tokens)
    with open(generations.path, "w", encoding="utf-8") as f:
        for r, g in zip(sft_rows, dpo_gens):
            f.write(json.dumps({"question": r["question"], "sft": r["answer"], "dpo": g["answer"]},
                               ensure_ascii=False) + "\n")
    m = lib.dpo_metrics_from(sft_rows, sft_rows, dpo_gens, judge_api_base, judge_model)
    for k, v in m.items():
        metrics.log_metric(k, v)
    return json.dumps(m)


# ── 관문 (CPU, 수 초, 매번 실행) ─────────────────────────────────────────────
@dsl.component(**COMMON)
def gate_cpt(metrics_json: str, train_run_id: str, mlflow_uri: str, experiment: str, parent_run_id: str,
             min_domain_ppl_drop: float, max_general_ppl_rise: float) -> NamedTuple("GateOut", [("decision", str), ("summary", str)]):
    import json
    from collections import namedtuple
    from pipelines import lib
    m = json.loads(metrics_json)
    checks = lib.cpt_checks(m, min_domain_ppl_drop, max_general_ppl_rise)
    decision, summary = lib.gate_outcome("Gate 2 (CPT)", checks)
    lib.log_gate(mlflow_uri, experiment, parent_run_id, "gate2-cpt", m, checks, decision == "PASS", train_run_id)
    return namedtuple("GateOut", ["decision", "summary"])(decision, summary)


@dsl.component(**COMMON)
def gate_sft(metrics_json: str, train_run_id: str, mlflow_uri: str, experiment: str, parent_run_id: str,
             min_stop_rate: float, min_avg_len: float, max_avg_len: float, max_format_error_rate: float) -> NamedTuple("GateOut", [("decision", str), ("summary", str)]):
    import json
    from collections import namedtuple
    from pipelines import lib
    m = json.loads(metrics_json)
    checks = lib.sft_checks(m, min_stop_rate, min_avg_len, max_avg_len, max_format_error_rate)
    decision, summary = lib.gate_outcome("Gate 3 (SFT)", checks)
    lib.log_gate(mlflow_uri, experiment, parent_run_id, "gate3-sft", m, checks, decision == "PASS", train_run_id)
    return namedtuple("GateOut", ["decision", "summary"])(decision, summary)


@dsl.component(**COMMON)
def gate_dpo(metrics_json: str, train_run_id: str, mlflow_uri: str, experiment: str, parent_run_id: str,
             min_win_rate: float, max_len_increase: float) -> NamedTuple("GateOut", [("decision", str), ("summary", str)]):
    import json
    from collections import namedtuple
    from pipelines import lib
    m = json.loads(metrics_json)
    checks = lib.dpo_checks(m, min_win_rate, max_len_increase)
    decision, summary = lib.gate_outcome("Gate 4 (DPO)", checks)
    lib.log_gate(mlflow_uri, experiment, parent_run_id, "gate4-dpo", m, checks, decision == "PASS", train_run_id)
    return namedtuple("GateOut", ["decision", "summary"])(decision, summary)


# ── 등록 · 배포 ─────────────────────────────────────────────────────────────
@dsl.component(**COMMON)
def register_model(model: Input[Model], train_run_id: str, mlflow_uri: str, experiment: str, parent_run_id: str,
                   model_name: str, alias: str, kfp_run_id: str) -> str:
    """병합된 DPO 모델을 Registry 에 등록하고 alias(기본 challenger) 를 붙인다. 아직 서비스에는 안 나간다."""
    from pipelines import lib
    version, source = lib.register_model(mlflow_uri, experiment, parent_run_id, model_name, model.uri, train_run_id,
                                         alias, {"kfp.run_id": kfp_run_id, "stage": "dpo"})
    print(f"등록: {model_name} v{version} ({alias}) ← {source}")
    return version


@dsl.component(**COMMON)
def deploy_vllm(model_name: str, version: str, mlflow_uri: str, experiment: str, parent_run_id: str,
                namespace: str, deployment: str, service_url: str, served_model: str, rollout_timeout_s: int):
    """@challenger 버전을 vLLM Deployment 에 롤링 배포 → 성공하면 @champion 으로 승격 (실패 시 롤백 + 예외)."""
    from pipelines import lib
    client, _ = lib.mlflow_client(mlflow_uri, experiment)
    source = client.get_model_version(model_name, version).source
    result = lib.deploy_rolling(deployment, namespace, source, version, service_url, served_model, rollout_timeout_s)
    old = lib.promote_alias(mlflow_uri, experiment, model_name, version)
    lib.log_child_run(mlflow_uri, experiment, parent_run_id, "deploy",
                      params={"model_name": model_name, "version": version, "previous_champion": old,
                              "deployment": f"{namespace}/{deployment}", "source": source,
                              "skipped": result["skipped"]})


# ── 종료 핸들러 (성공·실패와 무관하게 마지막에 실행) ─────────────────────────────
@dsl.component(**COMMON)
def notify(status: dsl.PipelineTaskFinalStatus, mlflow_uri: str, experiment: str, kfp_run_id: str,
           notify_on_success: bool):
    """Slack 알림 + MLflow 부모 run 종료 상태 기록. 부모 run id 는 KFP run id 태그로 찾는다
    (종료 핸들러는 ExitHandler 안쪽 task 의 출력을 입력으로 받을 수 없다)."""
    import os
    from pipelines import lib
    state = status.state
    ok = state == "SUCCEEDED"
    parent, gates = None, []
    try:
        client, parent = lib.find_parent_run(mlflow_uri, experiment, kfp_run_id)
        if parent:
            gates = lib.failed_gates(client, client.get_run(parent).info.experiment_id, parent)
            client.set_tag(parent, "kfp.final_state", state)
            if status.error_message:
                client.set_tag(parent, "kfp.error", status.error_message[:5000])
            client.set_terminated(parent, "FINISHED" if ok else "FAILED")
    except Exception as e:   # MLflow 장애여도 알림은 보낸다
        print(f"MLflow 기록 실패: {e}")
    if ok and not notify_on_success:
        return
    icon = ":white_check_mark:" if ok else ":rotating_light:"
    text = (f"{icon} *llm-ft 파이프라인 {state}*\n"
            f"• run: `{status.pipeline_job_resource_name}`\n"
            f"• MLflow 부모 run: `{parent}`\n")
    for gate, why in gates:
        text += f"• :no_entry: *{gate} 실패*: {why}\n"
    if not ok:
        text += f"• 오류: ```{(status.error_message or '(메시지 없음)')[:1500]}```"
    lib.slack_post(os.environ.get("SLACK_WEBHOOK_URL"), text)
