_under construction & not fully implemented and tested..._

## Kubeflow Pipelines (KFP v2) ##

데이터 전처리 → CPT(연속 사전학습) → SFT(지도 미세조정) → DPO(선호도 정렬) → vLLM 무중단 추론 배포 로 이어지는 3B 모델 파이프라인을 **Kubeflow Pipelines (KFP v2)** 로 자동화하고, 각 단계 사이에 자동화된 품질 관문(Quality Gate), MLflow 계층형 트래킹 및 Model Registry Alias 배포, 그리고 실패 시 Slack 알림(`dsl.ExitHandler`) 발송을 하는 방법에 대해서 다룬다.

### 1. 전체 파이프라인 아키텍처 ###

핵심 설계 원칙은 **학습(Train) 컴포넌트와 평가(Eval/Gate) 컴포넌트를 물리적으로 분리** 하는 것이다. 이를 통해 KFP의 **Step Caching** 기능을 극대화하여, Gate 임계값이나 평가 기준을 수정할 때 무거운 GPU 학습을 다시 돌리지 않고 평가 단계만 1분 내에 재실행할 수 있다. 참고로 KFP Step Caching (단계별 캐싱) 은 이전 실행 때와 입력값이나 코드가 똑같은 박스는 다시 실행하지 않고, 이전에 만들어둔 결과물을 그대로 꺼내 쓰는(건너뛰는) 기능이다.

![](https://github.com/gnosia93/llm-pipeline/blob/main/images/kf-pipeline.jpg)


### 2. Quality Gate 검증 기준표 ###

| 관문 (Gate) | 평가 대상 | 핵심 측정 지표 (MLflow 기록) | 통과 조건 (Threshold) | 방어하는 실패 유형 |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1 (Data)** | 정제된 데이터셋 | • 중복률 (`dup_ratio`)<br>• 유효 샘플 수 (`valid_count`)<br>• 개인정보/금지어 검출 수 | • `dup_ratio < 0.05`<br>• `valid_count >= 10000`<br>• `pii_count == 0` | 오염된 데이터로 인한 시간·GPU 낭비 사전 차단 |
| **Gate 2 (CPT)** | CPT 완료 모델 | • 도메인 Held-out PPL<br>• 일반 상식(General) PPL | • 도메인 PPL **15% 이상 감소**<br>• 일반 PPL 상승폭 **< 5%** | 도메인 지식 주입 실패 또는 파국적 망각(Catastrophic Forgetting) |
| **Gate 3 (SFT)** | SFT 완료 모델 | • 종료 토큰 비율 (`stop_rate`)<br>• 평균 응답 길이 (`avg_len`) | • `stop_rate >= 0.95`<br>• `20 <= avg_len <= 480` | `<\|im_end\|>` 미생성으로 인한 무한 반복 생성 폭주 방지 |
| **Gate 4 (DPO)** | 병합된 최종 모델 | • SFT 대비 Judge 승률 (`win_rate`)<br>• SFT 대비 응답 길이 증가율 | • `win_rate >= 0.55`<br>• 길이 증가폭 **< 20%** | 정렬 실패 또는 '길게 쓰기만 하는 편향(Verbosity Bias)' 차단 |


### 3. 파이프라인 ###

| 파일 | 내용 |
|---|---|
| [`pipelines/pipeline.py`](https://github.com/gnosia93/llm-pipeline/blob/main/src/pipeline/pipeline.py) | 파이프라인 정의 (`dsl.ExitHandler`, `dsl.If/Else`, GPU·secret 설정), 컴파일·제출 CLI |
| [`pipelines/components.py`](https://github.com/gnosia93/llm-pipeline/blob/main/src/pipeline/components.py) | KFP 컴포넌트 (얇은 껍데기). 학습은 저장소의 `train_*.py` 를 그대로 호출 |
| [`pipelines/lib.py`](https://github.com/gnosia93/llm-pipeline/blob/main/src/pipeline/lib.py) | 관문 지표·판정, MLflow, Registry, 배포·롤백, Slack 로직 (KFP 없이 import 가능) |
| `pipelines/selftest.py` | 클러스터 없이 돌리는 검사 |
| `pipelines/Dockerfile`, `requirements-pipeline.txt` | 파이프라인 이미지 |
| `pipelines/k8s/vllm-serving.yaml` | 서빙 Deployment (InitContainer, RollingUpdate maxUnavailable 0, probe, PDB) |
| `pipelines/k8s/judge-vllm.yaml` | 심판 LLM (Qwen2.5-7B-Instruct) |
| `pipelines/k8s/rbac.yaml`, `secret.example.yaml` | 배포 권한, 비밀값 예시 |

#### 1. 준비 (한 번) ####

```bash
# 1) 이미지 (모든 단계 공용)
docker build -f pipelines/Dockerfile -t <레지스트리>/llm-ft-pipeline:v1 .
docker push <레지스트리>/llm-ft-pipeline:v1

# 2) 비밀값 (파이프라인이 도는 Kubeflow 사용자 네임스페이스)
cp pipelines/k8s/secret.example.yaml /tmp/llm-ft-secrets.yaml   # SLACK_WEBHOOK_URL 등 채우기
kubectl -n <사용자-네임스페이스> apply -f /tmp/llm-ft-secrets.yaml

# 3) 서빙 · 심판 · 권한 (rbac.yaml 의 ServiceAccount/네임스페이스를 환경에 맞게 고친다)
kubectl apply -f pipelines/k8s/rbac.yaml -f pipelines/k8s/vllm-serving.yaml -f pipelines/k8s/judge-vllm.yaml
kubectl -n llm-serving create secret generic model-store \
  --from-literal=AWS_ACCESS_KEY_ID=... --from-literal=AWS_SECRET_ACCESS_KEY=... \
  --from-literal=S3_ENDPOINT_URL=http://minio-service.kubeflow.svc.cluster.local:9000   # KFP artifact store 를 읽을 권한
```

#### 2. 컴파일 및 실행 ####

```bash
pip install -r requirements-pipeline.txt
export PIPELINE_IMAGE=<레지스트리>/llm-ft-pipeline:v1
export GPU_NODE_SELECTOR=node.kubernetes.io/instance-type=g6e.4xlarge     # 선택

python -m pipelines.pipeline                                  # → pipelines/llm_ft_pipeline.yaml (UI 에 업로드해도 됨)
python -m pipelines.pipeline --run --host https://<kubeflow>/pipeline \
  --param min_valid_count=3000 --param mlflow_uri=http://mlflow.mlflow:5000
```

#### 3. 임계값만 바꿔 다시 돌리기 (캐시 확인) ####

```bash
python -m pipelines.pipeline --run --host ... --param min_valid_count=3000 --param min_win_rate=0.5
```

KFP UI 에서 `train-*`, `eval-*` 가 **cached** 로 표시되고 `gate-*` 만 다시 도는지 확인한다.

#### 4. 클러스터 없이 검사 ####

```bash
python -m pipelines.selftest          # 관문 판정·PII·MLflow 계층·심판 승률·롤백 로직 (33개 검사)
```
