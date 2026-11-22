# 모듈 9. Kubeflow Pipelines (KFP v2) 로 전 과정 자동화

**소요 60분** · 강의 + 시연 (Kubeflow 클러스터가 있으면 실습)

데이터 전처리 → CPT(연속 사전학습) → SFT(지도 미세조정) → DPO(선호도 정렬) → vLLM 무중단 추론 배포 로 이어지는 3B 모델 파이프라인을
**Kubeflow Pipelines (KFP v2)** 로 자동화한다. 각 단계 사이에 자동화된 **품질 관문(Quality Gate)**, **MLflow 계층형 트래킹**과
**Model Registry Alias 배포**, 그리고 실패 시 **Slack 알림(`dsl.ExitHandler`)** 을 붙인다.

## 학습 목표

- 학습 · 평가 · 관문을 분리해서 **KFP Step Caching** 을 최대로 활용하는 이유를 설명할 수 있다.
- Quality Gate 4개가 각각 어떤 실패를 막는지 안다.
- MLflow 부모/자식 run 구조와 `@challenger` → `@champion` alias 승격 흐름을 이해한다.
- vLLM 을 InitContainer + 롤링 업데이트로 **무중단** 교체하는 조건을 안다.

---

## 9.1 전체 파이프라인 아키텍처

```
[dsl.ExitHandler: 파이프라인 종료 시 성공/실패 및 실패 Gate 자동 Slack 알림]
 │
[Parent MLflow Run: qwen25-3b-pipeline-v1]
 │
 ├─▶ [1. Data Pre-processing] ──▶ [Gate 1: 중복률/토큰분포/PII] ────(FAIL)──▶ [즉시 중단 & Slack 알림]
 │                                     │ (PASS: dsl.If)
 ├─▶ [2. Train CPT (3B)] ───────▶ [Eval & Gate 2: Held-out PPL] ───(FAIL)──▶ [즉시 중단 & Slack 알림]
 │                                     │ (PASS: dsl.If)
 ├─▶ [3. Train SFT (LoRA)] ─────▶ [Eval & Gate 3: Stop Rate/포맷] ─(FAIL)──▶ [즉시 중단 & Slack 알림]
 │                                     │ (PASS: dsl.If)
 ├─▶ [4. Train DPO & Merge] ────▶ [Eval & Gate 4: Judge 승률] ────(FAIL)──▶ [즉시 중단 & Slack 알림]
 │                                     │ (PASS: dsl.If)
 └─▶ [5. MLflow Registry '@challenger' 등록] ──▶ [6. vLLM InitContainer 무중단 롤링 배포('@champion')]
```

**핵심 설계 원칙은 학습(Train) 컴포넌트와 평가(Eval/Gate) 컴포넌트를 물리적으로 분리하는 것**이다.
이를 통해 KFP 의 Step Caching 기능을 극대화하여, Gate 임계값이나 평가 기준을 수정할 때 무거운 GPU 학습을 다시 돌리지 않고
평가 단계만 다시 실행할 수 있다.

> **KFP Step Caching (단계별 캐싱)** 은 이전 실행 때와 입력값이나 코드가 똑같은 박스는 다시 실행하지 않고,
> 이전에 만들어 둔 결과물을 그대로 꺼내 쓰는(건너뛰는) 기능이다.

### 이 저장소에서 구현한 방식

| 종류 | 컴포넌트 | 자원 | 캐시 | 입력 |
|---|---|---|---|---|
| 학습 | `train_cpt`, `train_sft`, `train_dpo`, `build_preferences` | GPU, 수십 분~수 시간 | ✓ | 데이터·모델 artifact + 하이퍼파라미터 |
| 평가 | `eval_cpt`, `eval_sft`, `eval_dpo` | GPU, 몇 분 | ✓ | 모델 artifact + 평가 설정 |
| 관문 | `gate_data`, `gate_cpt`, `gate_sft`, `gate_dpo` | CPU, 수 초 | ✗ (매번) | 평가 결과(JSON) + **임계값** |
| 기록·배포 | `init_mlflow_run`, `register_model`, `deploy_vllm`, `notify` | CPU | ✗ | |

| 바꾼 것 | 다시 도는 단계 |
|---|---|
| Gate 임계값 (예: `min_win_rate`) | 관문만 (**수 초**). 학습·평가는 캐시 |
| 평가 설정 (예: `eval_prompts`) | 그 평가 + 관문. 학습은 캐시 |
| DPO 하이퍼파라미터 (`dpo_beta`) | DPO 학습부터. CPT·SFT 는 캐시 |
| `data_version` | 전부 (문서를 새로 받는다) |

**함정: MLflow 부모 run id 를 학습 컴포넌트에 넘기면 캐시가 깨진다.** 부모 run id 는 실행마다 바뀌므로 입력에 들어가면
캐시 키가 달라져 매번 GPU 학습을 다시 한다. 그래서 학습 컴포넌트는 MLflow run 을 혼자 만들고 run id 만 출력한다.
같은 실행의 관문이 그 run 을 부모 아래로 연결한다 (`lib.link_train_run`). 캐시로 재사용된 학습 run 은 처음 학습한 실행의 부모 아래에 그대로 두고,
`llm_ft.reused_by.<부모>` 태그만 남긴다.

컴파일된 DAG (`python -m pipelines.pipeline` 결과를 풀어 본 것, 실측):

```
- exit-handler-1
  - init-mlflow-run        cache=False
  - prepare-data           cache=True
  - gate-data              cache=False
  - condition [gate-data == PASS]
      - train-cpt          cache=True     ← init-mlflow-run 에 의존하지 않음
      - eval-cpt           cache=True
      - gate-cpt           cache=False
      - condition [gate-cpt == PASS]
          - train-sft / eval-sft / gate-sft
          - condition [gate-sft == PASS]
              - build-preferences / train-dpo / eval-dpo / gate-dpo
              - condition [gate-dpo == PASS]
                  - register-model (@challenger)
                  - condition [deploy == true] → deploy-vllm (@champion)
              - condition [gate-dpo != PASS] → stop-pipeline
          - condition [gate-sft != PASS] → stop-pipeline
      - condition [gate-cpt != PASS] → stop-pipeline
  - condition [gate-data != PASS] → stop-pipeline
- notify (ExitHandler, 항상 실행)
```

## 9.2 Quality Gate 검증 기준표

| 관문 (Gate) | 평가 대상 | 핵심 측정 지표 (MLflow 기록) | 통과 조건 (기본값, 파이프라인 파라미터) | 방어하는 실패 유형 |
|---|---|---|---|---|
| **Gate 1 (Data)** | 정제된 데이터셋 | • 중복률 `dup_ratio`<br>• 유효 샘플 수 `valid_count`<br>• 토큰 분포 `*_tokens_p50/p95/max`, `overlength_ratio`<br>• 개인정보/금지어 검출 수 | • `dup_ratio < 0.05`<br>• `valid_count >= 10000`<br>• `overlength_ratio <= 0.01`<br>• `pii_count == 0`, 금지어 0 | 오염된 데이터로 인한 시간·GPU 낭비 사전 차단 |
| **Gate 2 (CPT)** | CPT 완료 모델 | • 도메인 Held-out PPL (`k8s_heldout`)<br>• 일반 상식(General) PPL (`general_ko`) | • 도메인 PPL **15% 이상 감소**<br>• 일반 PPL 상승폭 **< 5%** | 도메인 지식 주입 실패 또는 파국적 망각 |
| **Gate 3 (SFT)** | SFT 완료 모델 | • 종료 토큰 비율 `stop_rate`<br>• 포맷 오류율 `format_error_rate`<br>• 평균 응답 길이 `avg_len` (토큰) | • `stop_rate >= 0.95`<br>• `format_error_rate <= 0.05`<br>• `20 <= avg_len <= 480` | `<\|im_end\|>` 미생성으로 인한 무한 반복 생성 폭주 방지 |
| **Gate 4 (DPO)** | 병합된 최종 모델 | • SFT 대비 Judge 승률 `win_rate`<br>• SFT 대비 응답 길이 증가율 | • `win_rate >= 0.55`<br>• 길이 증가폭 **< 20%** | 정렬 실패 또는 '길게 쓰기만 하는 편향(Verbosity Bias)' 차단 |

구현 세부 (`pipelines/lib.py`):

- **중복률:** CPT 블록과 SFT 대화(질문+답)를 공백 정규화 후 해시가 같은 비율.
- **유효 샘플:** CPT 블록 200자 이상, SFT 는 system/user/assistant 형식 + 질문 5자·답변 20자 이상.
- **토큰 분포:** 학습 `max_length` (CPT 8,192 / SFT 2,048) 를 넘어 **학습 때 잘려 나갈 샘플 비율**을 막는다.
- **PII:** 이메일, 휴대전화, 주민등록번호, 카드번호(Luhn 검사), AWS 키, 개인키. 쿠버네티스 문서에는 예시 주소(`user@example.com`,
  `git@github.com`, 프로젝트 공식 주소)와 예시 카드번호(`1111 2222 3333 4444`)가 있어서 허용 목록으로 뺐다.
  정규식만 쓰면 실제 데이터에서 29건이 잡혔고, 모두 공개·예시 값이었다 (실측). 로그에는 앞 4글자만 남긴다.
- **포맷 오류 (Gate 3):** 빈 답, 특수 토큰이 글자로 새어 나옴, 다음 `user` 턴을 지어냄, 같은 줄 3번 이상 반복.
- **Judge 승률 (Gate 4):** 같은 질문에 대한 SFT 답과 DPO 답을 심판 LLM 에 **순서를 바꿔 두 번** 묻는다 (위치 편향 제거).
  두 번 다 DPO 가 이기면 1, 엇갈리거나 비기면 0.5. 프롬프트에 "길이는 기준이 아니다" 를 넣었다. 심판 장애로 판정 수가 90% 미만이면 실패로 본다.
  SFT 답은 Gate 3 평가 때 만든 것을 재사용한다 (`eval_sft` 의 `generations` artifact).

> **이 저장소의 데이터로는 Gate 1 이 기본값에서 실패한다.** 쿠버네티스 한글 문서는 CPT 726 + SFT 3,012 = 유효 3,667 개라서
> `valid_count >= 10000` 을 넘지 못한다 (실측). 사내 데이터처럼 더 큰 코퍼스를 위한 기준이다. 이 데이터로 돌릴 때는
> `--param min_valid_count=3000` 처럼 데이터 규모에 맞춰 정한다. 나머지 Gate 1 조건은 실측으로 통과한다
> (중복률 0.0, PII 0, 길이 초과 0.03%).
> Gate 2 의 "도메인 held-out PPL 15% 감소" 도 모듈 5 의 기대("안 본 문서는 약간 감소")보다 엄격하다. 리허설 결과를 보고 정한다.

## 9.3 PASS / FAIL 분기와 Slack 알림

관문은 예외를 던지지 않고 `("PASS" | "FAIL", 요약)` 을 출력한다.

```python
g2 = gate_cpt(metrics_json=cpt_ev.outputs["Output"], ...)
with dsl.If(g2.outputs["decision"] == "PASS"):
    sft = train_sft(...)                     # 다음 단계는 PASS 안에서만
    ...
with dsl.Else():
    stop_pipeline(summary=g2.outputs["summary"])   # 예외 → 파이프라인 FAILED → ExitHandler
```

`notify` 는 `dsl.ExitHandler` 의 종료 작업이라서 성공·실패와 관계없이 마지막에 돈다. Slack 메시지에는

1. KFP 상태와 `error_message`
2. **MLflow 에서 찾은 실패 Gate 와 실패 조건** (KFP 백엔드에 따라 `error_message` 가 "task failed" 같은 일반 문구일 수 있어서)

가 함께 실린다. 로컬 테스트에서 실제로 만든 메시지:

```
:rotating_light: *llm-ft 파이프라인 FAILED*
• run: `runs/kfp-xyz`
• MLflow 부모 run: `4145da93a7014a95b02eb35d592ed1a6`
• :no_entry: *gate3-sft 실패*: stop_rate=0.82 (조건 >= 0.95)
• 오류: ```Task stop-pipeline-2 failed```
```

> 종료 작업은 ExitHandler 안쪽 task 의 출력을 입력으로 받을 수 없다. 그래서 `notify` 는 부모 run id 를 입력으로 받지 않고
> `kfp.run_id` 태그로 MLflow 에서 찾는다. 부모 run 이름(`qwen25-3b-pipeline-v1`)은 실행마다 같을 수 있으므로 이름으로 찾으면 안 된다.

## 9.4 MLflow 계층형 트래킹과 Model Registry Alias

```
qwen25-3b-pipeline-v1          (부모, 파이프라인 실행 1회. tags: kfp.run_id, kfp.final_state)
 ├ gate1-data                  metrics: dup_ratio, valid_count, pii_count, *_tokens_p95 ... / params: threshold.* / gate_passed
 ├ train-cpt                   loss 곡선 (trainer_state.json → step 별 metric)
 ├ gate2-cpt                   ppl_base_*, ppl_cpt_*, domain_ppl_drop, general_ppl_rise
 ├ train-sft / gate3-sft       stop_rate, format_error_rate, avg_len
 ├ train-dpo / gate4-dpo       win_rate, len_increase, sft_avg_len, dpo_avg_len
 ├ register                    model_name, version, alias=challenger, source
 └ deploy                      version, previous_champion, deployment
```

로컬 KFP 실행(`kfp.local`) 으로 실제로 만들어진 구조 (실측):

```
부모 qwen25-3b-pipeline-v1 (c8ee8902)
   └ gate1-data   status=FAILED    gate_passed=0.0  valid_count=3667
부모 qwen25-3b-pipeline-v1 (b8cfa19b)
   └ gate1-data   status=FINISHED  gate_passed=1.0
   └ gate2-cpt    status=FAILED    gate_passed=0.0  domain_ppl_drop=0.0882   threshold: >= 0.15
```

**Alias 흐름:**

```
Gate 4 PASS → register: 새 버전 vN, @challenger = vN           (아직 서비스에 안 나감)
           → deploy:   롤링 업데이트 + smoke test 성공
                       @previous = 옛 @champion, @champion = vN   (실패하면 롤백하고 alias 는 그대로)
```

- 모델 파일은 MLflow 로 다시 올리지 않는다. 버전의 `source` 가 KFP artifact 위치(`minio://` → `s3://` 로 변환)를 가리킨다.
- 롤백이 필요하면 `@previous` 버전의 source 로 다시 배포하면 된다.

## 9.5 vLLM InitContainer 무중단 롤링 배포

`pipelines/k8s/vllm-serving.yaml`:

```
Pod
 ├ initContainer model-downloader (aws-cli)   MODEL_URI (s3://...) → emptyDir /models/current
 └ container vllm                             vllm serve /models/current --served-model-name k8s-ko
```

`deploy_vllm` 이 하는 일:

1. **무중단 조건 검사** — `strategy.rollingUpdate.maxUnavailable: 0` 과 `readinessProbe` 가 없으면 배포하지 않는다.
2. pod template 의 `MODEL_URI` 와 `llm-ft/model-version` annotation 만 바꿔 **롤링 업데이트**를 일으킨다.
3. `kubectl rollout status` 와 같은 기준으로 완료를 기다린다. `ProgressDeadlineExceeded` 면 실패.
4. Service 로 실제 질문을 보내는 **smoke test** (`finish_reason == "stop"` 이고 답이 비어 있지 않아야 함).
5. 3·4 가 실패하면 **이전 MODEL_URI 로 되돌리고** 예외 → Slack. 성공하면 `@champion` 승격.

무중단이 되는 이유: `maxUnavailable: 0` 이라 새 pod 가 모델을 받고 `/health` 가 Ready 가 될 때까지 옛 pod 가 계속 트래픽을 받는다.
`preStop sleep 15` 와 `terminationGracePeriodSeconds` 로 옛 pod 가 Service 에서 빠진 뒤 진행 중인 요청을 마무리하고 내려간다.
**롤아웃 동안 GPU 가 1장 더 필요하다** (`maxSurge: 1`).

## 9.6 실행 방법

### 준비 (한 번)

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

### 컴파일 · 실행

```bash
pip install -r requirements-pipeline.txt
export PIPELINE_IMAGE=<레지스트리>/llm-ft-pipeline:v1
export GPU_NODE_SELECTOR=node.kubernetes.io/instance-type=g6e.4xlarge     # 선택

python -m pipelines.pipeline                                  # → pipelines/llm_ft_pipeline.yaml (UI 에 업로드해도 됨)
python -m pipelines.pipeline --run --host https://<kubeflow>/pipeline \
  --param min_valid_count=3000 --param mlflow_uri=http://mlflow.mlflow:5000
```

### 임계값만 바꿔 다시 돌리기 (캐시 확인)

```bash
python -m pipelines.pipeline --run --host ... --param min_valid_count=3000 --param min_win_rate=0.5
```

KFP UI 에서 `train-*`, `eval-*` 가 **cached** 로 표시되고 `gate-*` 만 다시 도는지 확인한다.

### 클러스터 없이 검사

```bash
python -m pipelines.selftest          # 관문 판정·PII·MLflow 계층·심판 승률·롤백 로직 (33개 검사)
```

`kfp.local` 의 SubprocessRunner 로 실제 컴포넌트 코드를 돌려 Gate 1 실패 / Gate 2 실패 / 통과 세 분기가 의도대로 갈리는 것도 확인했다.
**GPU 학습 단계와 실제 클러스터 배포는 이 환경에서 실행하지 못했다** (Kubeflow 클러스터가 없음). 첫 실행은 `deploy=false` 로 해 보길 권한다.

## 9.7 파일

| 파일 | 내용 |
|---|---|
| `pipelines/pipeline.py` | 파이프라인 정의 (`dsl.ExitHandler`, `dsl.If/Else`, GPU·secret 설정), 컴파일·제출 CLI |
| `pipelines/components.py` | KFP 컴포넌트 (얇은 껍데기). 학습은 저장소의 `train_*.py` 를 그대로 호출 |
| `pipelines/lib.py` | 관문 지표·판정, MLflow, Registry, 배포·롤백, Slack 로직 (KFP 없이 import 가능) |
| `pipelines/selftest.py` | 클러스터 없이 돌리는 검사 |
| `pipelines/Dockerfile`, `requirements-pipeline.txt` | 파이프라인 이미지 |
| `pipelines/k8s/vllm-serving.yaml` | 서빙 Deployment (InitContainer, RollingUpdate maxUnavailable 0, probe, PDB) |
| `pipelines/k8s/judge-vllm.yaml` | 심판 LLM (Qwen2.5-7B-Instruct) |
| `pipelines/k8s/rbac.yaml`, `secret.example.yaml` | 배포 권한, 비밀값 예시 |

## 과제

1. Gate 2 의 `min_domain_ppl_drop` 을 0.10 으로 낮춰 다시 실행하고, KFP UI 에서 어떤 단계가 cached 인지 기록하자.
2. `eval_prompts` 를 60 → 100 으로 바꾸면 어떤 단계가 다시 도나? 왜 그런가?
3. `train_cpt` 에 `parent_run_id` 를 입력으로 추가하면 무슨 일이 생길까? (9.1 의 함정)
4. 롤아웃 중 새 pod 가 OOM 으로 계속 죽는다면 `deploy_vllm` 은 어떻게 동작하나? 사용자는 그동안 어떤 응답을 받나?
5. Gate 4 의 심판을 학습에 쓴 심판과 같은 모델로 쓰는 것의 위험은? (선호 데이터와 평가가 같은 편향을 공유한다)

---
← [모듈 8](08_next_steps.md) · [부록: 문제 해결 · FAQ · 용어집 →](99_troubleshooting.md)
