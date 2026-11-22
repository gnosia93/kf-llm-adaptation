#!/usr/bin/env python3
"""Qwen2.5-3B k8s-ko 파이프라인 (KFP v2)

  [dsl.ExitHandler: 종료 시 성공/실패 + 실패 Gate 를 Slack 으로]
  [MLflow 부모 run: qwen25-3b-pipeline-v1]
   ├─ 1. Data Pre-processing ─▶ Gate 1 (중복률/토큰분포/PII) ──FAIL─▶ 즉시 중단 & Slack
   │                              │ PASS (dsl.If)
   ├─ 2. Train CPT (3B) ───────▶ Eval & Gate 2 (Held-out PPL) ──FAIL─▶ 즉시 중단 & Slack
   │                              │ PASS
   ├─ 3. Train SFT (LoRA) ─────▶ Eval & Gate 3 (Stop Rate/포맷) ─FAIL─▶ 즉시 중단 & Slack
   │                              │ PASS
   ├─ 4. Train DPO & Merge ────▶ Eval & Gate 4 (Judge 승률) ────FAIL─▶ 즉시 중단 & Slack
   │                              │ PASS
   └─ 5. MLflow Registry @challenger ─▶ 6. vLLM InitContainer 무중단 롤링 배포 → @champion

  PIPELINE_IMAGE=<레지스트리>/llm-ft-pipeline:<태그> python -m pipelines.pipeline      # → pipelines/llm_ft_pipeline.yaml
  python -m pipelines.pipeline --run --host https://<kubeflow>/pipeline               # 컴파일 + 실행 제출

필요한 Kubernetes 리소스 (pipelines/k8s/): secret llm-ft-secrets, 심판 vLLM, 서빙 vLLM Deployment, RBAC
"""
import argparse
import json
import os
from pathlib import Path

from kfp import compiler, dsl, kubernetes

from pipelines import components as c

SECRET = "llm-ft-secrets"
# secret 키 → 환경변수. optional=True 라서 없는 키는 무시된다 (예: MinIO 를 안 쓰면 AWS_* 가 없어도 됨)
SECRET_ENV = {
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",             # MLflow artifact store (S3/MinIO)
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_S3_ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL",   # MinIO 일 때
    "MLFLOW_TRACKING_USERNAME": "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD": "MLFLOW_TRACKING_PASSWORD",
    "HF_TOKEN": "HF_TOKEN",
    "SLACK_WEBHOOK_URL": "SLACK_WEBHOOK_URL",
}
GPU_NODE_SELECTOR = os.environ.get("GPU_NODE_SELECTOR", "")   # 예: "node.kubernetes.io/instance-type=g6e.4xlarge"


def cpu(task, memory="4Gi", cache=True):
    """CPU 단계. 관문·등록·배포·알림은 cache=False (매 실행마다 판정·기록해야 한다)."""
    task.set_cpu_request("1").set_memory_request(memory).set_memory_limit(memory)
    task.set_caching_options(cache)
    kubernetes.use_secret_as_env(task, SECRET, SECRET_ENV, optional=True)
    return task


def gpu(task, memory="100Gi", cpus="14", timeout_s=None):
    """L40S 48GB x1 (g6e.4xlarge: 16 vCPU, 128GB) 기준. 학습·평가 모두 캐시 대상."""
    task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    task.set_cpu_request(cpus).set_cpu_limit(cpus).set_memory_request(memory).set_memory_limit(memory)
    task.set_caching_options(True)
    kubernetes.use_secret_as_env(task, SECRET, SECRET_ENV, optional=True)
    kubernetes.empty_dir_mount(task, volume_name="dshm", mount_path="/dev/shm", medium="Memory", size_limit="16Gi")
    kubernetes.add_toleration(task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    if GPU_NODE_SELECTOR:
        k, v = GPU_NODE_SELECTOR.split("=", 1)
        kubernetes.add_node_selector(task, k, v)
    if timeout_s:
        kubernetes.set_timeout(task, timeout_s)
    return task


@dsl.pipeline(name="qwen25-3b-k8s-ko",
              description="데이터 → CPT → SFT(LoRA) → DPO → MLflow @challenger → vLLM 무중단 배포 @champion. "
                          "단계마다 Quality Gate (PASS: dsl.If / FAIL: 즉시 중단 + Slack)")
def llm_ft_pipeline(
    # 추적·배포
    mlflow_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",
    experiment: str = "llm-ft-k8s-ko",
    parent_run_name: str = "qwen25-3b-pipeline-v1",
    registered_model: str = "k8s-ko-3b",
    deploy: bool = True,
    deploy_namespace: str = "llm-serving",
    deploy_name: str = "k8s-ko-vllm",
    service_url: str = "http://k8s-ko-vllm.llm-serving.svc.cluster.local:8000",
    served_model: str = "k8s-ko",
    rollout_timeout_s: int = 1800,
    notify_on_success: bool = True,
    # 심판 LLM (선호 데이터, Gate 4). pipelines/k8s/judge-vllm.yaml
    judge_api_base: str = "http://judge-vllm.llm-serving.svc.cluster.local:8000/v1",
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
    # 데이터
    data_version: str = "v1",
    include_en: bool = False,
    cpt_max_tokens: int = 8192,
    sft_max_length: int = 2048,
    replay_ratio: float = 0.2,
    n_general: int = 1200,
    # 학습
    base_model: str = "Qwen/Qwen2.5-3B",
    cpt_full: bool = True,
    cpt_epochs: float = 4,
    sft_full: bool = False,          # 아키텍처: SFT 는 LoRA (train_sft_hf.py 가 학습 후 병합해서 저장)
    sft_epochs: float = 3,
    pref_prompts: int = 1000,
    dpo_beta: float = 0.1,
    dpo_epochs: float = 2,
    # 평가
    eval_prompts: int = 60,
    eval_max_new_tokens: int = 512,
    # Gate 1 (Data): 중복률 · 유효 샘플 · 토큰 분포 · PII/금지어
    max_dup_ratio: float = 0.05,
    min_valid_count: int = 10000,
    max_pii_count: int = 0,
    forbidden_words: str = "",
    max_overlength_ratio: float = 0.01,
    # Gate 2 (CPT): held-out PPL
    min_domain_ppl_drop: float = 0.15,
    max_general_ppl_rise: float = 0.05,
    # Gate 3 (SFT): stop rate · 포맷 · 길이
    min_stop_rate: float = 0.95,
    max_format_error_rate: float = 0.05,
    min_avg_len: float = 20,
    max_avg_len: float = 480,
    # Gate 4 (DPO): judge 승률 · 길이 증가
    min_win_rate: float = 0.55,
    max_len_increase: float = 0.20,
):
    track = dict(mlflow_uri=mlflow_uri, experiment=experiment)
    exit_task = cpu(c.notify(**track, kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
                             notify_on_success=notify_on_success), cache=False)
    exit_task.set_display_name("slack-notify (ExitHandler)")

    def stop(gate_task, name):
        """FAIL 분기: 즉시 중단. 요약이 예외 메시지 → ExitHandler 의 Slack 알림에 실린다."""
        t = cpu(c.stop_pipeline(summary=gate_task.outputs["summary"]), cache=False)
        t.set_display_name(f"stop: {name} FAIL")

    with dsl.ExitHandler(exit_task, name="qwen25-3b-pipeline"):
        init = cpu(c.init_mlflow_run(
            **track, run_name=parent_run_name, kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            kfp_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
            params_json=json.dumps({"note": "전체 파라미터는 KFP UI 의 run 상세에 있다"})), cache=False)
        init.set_display_name("mlflow-parent-run")
        parent = init.output

        # 1. Data Pre-processing → Gate 1
        data = cpu(c.prepare_data(data_version=data_version, include_en=include_en, cpt_max_tokens=cpt_max_tokens,
                                  replay_ratio=replay_ratio, n_general=n_general), memory="16Gi")
        data.set_cpu_request("4").set_display_name("1. data-preprocessing")
        g1 = cpu(c.gate_data(data=data.outputs["data"], **track, parent_run_id=parent, tokenizer=base_model,
                             cpt_max_tokens=cpt_max_tokens, sft_max_length=sft_max_length,
                             max_dup_ratio=max_dup_ratio, min_valid_count=min_valid_count,
                             max_pii_count=max_pii_count, forbidden_words=forbidden_words,
                             max_overlength_ratio=max_overlength_ratio), memory="8Gi", cache=False)
        g1.set_display_name("gate1: dup / token dist / PII")

        with dsl.If(g1.outputs["decision"] == "PASS", name="gate1-pass"):
            # 2. Train CPT → Eval & Gate 2
            cpt = gpu(c.train_cpt(data=data.outputs["data"], base_model=base_model, full=cpt_full, epochs=cpt_epochs,
                                  max_length=cpt_max_tokens, **track), timeout_s=6 * 3600)
            cpt.set_display_name("2. train-cpt")
            cpt_ev = gpu(c.eval_cpt(base_model=base_model, cpt_model=cpt.outputs["model"], data=data.outputs["data"],
                                    max_tokens=cpt_max_tokens))
            cpt_ev.set_display_name("eval-cpt: held-out PPL")
            g2 = cpu(c.gate_cpt(metrics_json=cpt_ev.outputs["Output"], train_run_id=cpt.outputs["Output"], **track,
                                parent_run_id=parent, min_domain_ppl_drop=min_domain_ppl_drop,
                                max_general_ppl_rise=max_general_ppl_rise), cache=False)
            g2.set_display_name("gate2: held-out PPL")

            with dsl.If(g2.outputs["decision"] == "PASS", name="gate2-pass"):
                # 3. Train SFT (LoRA) → Eval & Gate 3
                sft = gpu(c.train_sft(cpt_model=cpt.outputs["model"], data=data.outputs["data"], full=sft_full,
                                      epochs=sft_epochs, **track), timeout_s=4 * 3600)
                sft.set_display_name("3. train-sft (LoRA)")
                sft_ev = gpu(c.eval_sft(sft_model=sft.outputs["model"], data=data.outputs["data"],
                                        n_prompts=eval_prompts, max_new_tokens=eval_max_new_tokens))
                sft_ev.set_display_name("eval-sft: stop rate / format")
                g3 = cpu(c.gate_sft(metrics_json=sft_ev.outputs["Output"], train_run_id=sft.outputs["Output"], **track,
                                    parent_run_id=parent, min_stop_rate=min_stop_rate, min_avg_len=min_avg_len,
                                    max_avg_len=max_avg_len, max_format_error_rate=max_format_error_rate), cache=False)
                g3.set_display_name("gate3: stop rate / format")

                with dsl.If(g3.outputs["decision"] == "PASS", name="gate3-pass"):
                    # 4. Train DPO & Merge → Eval & Gate 4
                    pref = gpu(c.build_preferences(sft_model=sft.outputs["model"], data=data.outputs["data"],
                                                   judge_api_base=judge_api_base, judge_model=judge_model,
                                                   n_prompts=pref_prompts), timeout_s=3 * 3600)
                    pref.set_display_name("4a. build-preferences")
                    dpo = gpu(c.train_dpo(sft_model=sft.outputs["model"], pref=pref.outputs["pref"], beta=dpo_beta,
                                          epochs=dpo_epochs, **track), timeout_s=4 * 3600)
                    dpo.set_display_name("4b. train-dpo & merge")
                    dpo_ev = gpu(c.eval_dpo(dpo_model=dpo.outputs["model"], sft_generations=sft_ev.outputs["generations"],
                                            judge_api_base=judge_api_base, judge_model=judge_model,
                                            max_new_tokens=eval_max_new_tokens))
                    dpo_ev.set_display_name("eval-dpo: judge win rate")
                    g4 = cpu(c.gate_dpo(metrics_json=dpo_ev.outputs["Output"], train_run_id=dpo.outputs["Output"],
                                        **track, parent_run_id=parent, min_win_rate=min_win_rate,
                                        max_len_increase=max_len_increase), cache=False)
                    g4.set_display_name("gate4: judge win rate")

                    with dsl.If(g4.outputs["decision"] == "PASS", name="gate4-pass"):
                        # 5. MLflow Registry @challenger → 6. vLLM 무중단 롤링 배포 → @champion
                        reg = cpu(c.register_model(model=dpo.outputs["model"], train_run_id=dpo.outputs["Output"],
                                                   **track, parent_run_id=parent, model_name=registered_model,
                                                   alias="challenger", kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER),
                                  cache=False)
                        reg.set_display_name("5. registry @challenger")
                        with dsl.If(deploy == True, name="deploy-enabled"):   # noqa: E712 (KFP 조건식)
                            dep = cpu(c.deploy_vllm(model_name=registered_model, version=reg.output, **track,
                                                    parent_run_id=parent, namespace=deploy_namespace,
                                                    deployment=deploy_name, service_url=service_url,
                                                    served_model=served_model, rollout_timeout_s=rollout_timeout_s),
                                      cache=False)
                            dep.set_display_name("6. vLLM rolling deploy → @champion")
                            kubernetes.set_timeout(dep, 2 * 3600)   # 롤아웃 + 실패 시 롤백까지
                    with dsl.Else(name="gate4-fail"):
                        stop(g4, "Gate 4 (DPO)")
                with dsl.Else(name="gate3-fail"):
                    stop(g3, "Gate 3 (SFT)")
            with dsl.Else(name="gate2-fail"):
                stop(g2, "Gate 2 (CPT)")
        with dsl.Else(name="gate1-fail"):
            stop(g1, "Gate 1 (Data)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(Path(__file__).with_name("llm_ft_pipeline.yaml")))
    ap.add_argument("--run", action="store_true", help="컴파일 후 Kubeflow 에 실행 제출")
    ap.add_argument("--host", default=os.environ.get("KFP_HOST"))
    ap.add_argument("--namespace", default=os.environ.get("KFP_NAMESPACE"))
    ap.add_argument("--experiment", default="llm-ft")
    ap.add_argument("--param", action="append", default=[], help="파이프라인 파라미터 덮어쓰기 key=value (여러 번)")
    args = ap.parse_args()

    compiler.Compiler().compile(llm_ft_pipeline, args.output)
    print(f"컴파일: {args.output}  (image={c.IMAGE})")
    if args.run:
        import kfp
        params = {}
        for kv in args.param:
            k, v = kv.split("=", 1)
            params[k] = json.loads(v) if v[:1] in "0123456789-tf[{\"" and v not in ("-",) else v
        client = kfp.Client(host=args.host, namespace=args.namespace)
        run = client.create_run_from_pipeline_package(args.output, arguments=params, experiment_name=args.experiment,
                                                      enable_caching=True)
        print(f"제출: run_id={run.run_id}")


if __name__ == "__main__":
    main()
