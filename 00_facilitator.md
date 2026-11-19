# 강사용 가이드

워크숍을 준비하고 운영하는 사람을 위한 문서다. 참가자에게는 공유하지 않아도 된다.

---

## 1. 운영 형태 결정

| 방식 | 장점 | 단점 | 권장 |
|---|---|---|---|
| **참가자 개인 AWS 계정** | 비용과 권한이 분리됨 | 쿼터 증설에 1일이 걸림, 결제 수단 필요 | 사내 교육, 소규모 |
| **강사 계정 1개 + 참가자별 IAM 사용자** | 쿼터를 한 번에 관리, 일괄 정리가 쉬움 | 비용이 한 계정에 몰림, 이름 충돌 주의 | **외부 워크숍 (권장)** |
| **2인 1조로 인스턴스 공유** | 비용 절반 | 한 사람만 노트북을 조작 | 예산이 빠듯할 때 |

### 공유 계정일 때

- **쿼터:** `Running On-Demand G and VT instances` ≥ `16 x 인스턴스 수 + 여유분`.
  현재 실습 계정(us-east-1)은 **768 vCPU = 최대 48대** 다.
- **이름 충돌 방지:** 참가자마다 고유 이름을 정해 주고, 셸에서 먼저 export 하게 한다.
  ```bash
  export NAME=llm-ft-p01 KEY_NAME=llm-ft-p01 SSH_HOST=gpu-p01
  ```
- **참가자용 IAM 정책 예시.** g6e.4xlarge 만 띄울 수 있게 제한한다. 워크숍 전에 반드시 테스트할 것.
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {"Sid": "Read", "Effect": "Allow",
       "Action": ["ec2:Describe*", "service-quotas:GetServiceQuota"], "Resource": "*"},
      {"Sid": "RunOnlyG6e4xlarge", "Effect": "Allow", "Action": "ec2:RunInstances",
       "Resource": "arn:aws:ec2:*:*:instance/*",
       "Condition": {"StringEquals": {"ec2:InstanceType": "g6e.4xlarge"}}},
      {"Sid": "RunDependencies", "Effect": "Allow", "Action": "ec2:RunInstances",
       "Resource": ["arn:aws:ec2:*::image/*", "arn:aws:ec2:*:*:subnet/*",
                    "arn:aws:ec2:*:*:network-interface/*", "arn:aws:ec2:*:*:security-group/*",
                    "arn:aws:ec2:*:*:key-pair/*", "arn:aws:ec2:*:*:volume/*"]},
      {"Sid": "Manage", "Effect": "Allow",
       "Action": ["ec2:CreateTags", "ec2:CreateKeyPair", "ec2:DeleteKeyPair",
                  "ec2:CreateSecurityGroup", "ec2:DeleteSecurityGroup",
                  "ec2:AuthorizeSecurityGroupIngress",
                  "ec2:StartInstances", "ec2:StopInstances", "ec2:TerminateInstances"],
       "Resource": "*"}
    ]
  }
  ```
  이 정책으로는 참가자가 서로의 인스턴스를 끌 수도 있다. 막으려면 `Manage` 에 `aws:ResourceTag/Name` 조건을 추가한다.

## 2. 사전 준비 체크리스트

| 시점 | 할 일 | 확인 |
|---|---|---|
| D-14 | 운영 형태 결정, 쿼터 확인·증설 요청 | `aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA` |
| D-14 | 예산 승인 (아래 비용 표) | |
| D-7 | **전체 리허설** (모듈 2~6 을 처음부터 끝까지). 아래 측정표 채우기 | |
| D-7 | 실습 저장소 공개 범위 결정, clone 주소 준비 | |
| D-7 | (선택) **완성된 CPT 모델 업로드** — 학습이 늦어지는 참가자용 (5절) | |
| D-3 | AMI 최신판 확인. `launch_g6e.sh` 가 자동으로 고르지만, AMI 가 바뀌면 한 번 더 테스트 | |
| D-1 | 참가자에게 사전 안내 (3절) | |
| 당일 시작 전 | 강사 인스턴스를 띄워 시연 화면 준비, 와이파이에서 22번 포트가 열리는지 확인 | `ssh k8s-gpu true` |
| 종료 후 | **남은 인스턴스와 볼륨 일괄 점검** (6절) | |

### 리허설 측정표

실제로 재서 채워 두고 참가자에게 안내한다. 학습 관련 값은 **아직 측정하지 않았다.**

| 항목 | 측정값 | 메모 |
|---|---|---|
| `launch_g6e.sh` (생성 ~ SSH 가능) | 분 | |
| `setup_remote.sh` | 분 | |
| 데이터 생성 (다운로드 + 03_build_cpt) | 분 | Mac 실측: 03_build_cpt 약 25초 (위키 스트리밍 포함) |
| Qwen2.5-3B 첫 다운로드 | 분 | 약 6GB |
| 학습 `s/it` / 전체 300 스텝 | s/it / 분 | |
| 학습 중 GPU 메모리 최대 | GB | 예측 36~39GB |
| 평가 셀 (모델 2개 x ppl 3종 + 프로브) | 분 | |
| perplexity: seen / heldout / general | BASE: / CPT: | |

## 3. 참가자 사전 안내문 (예시)

```
[워크숍 사전 준비] 3B LLM 에 쿠버네티스 지식 넣기

1. VS Code 설치 후 확장 설치: Remote - SSH, Python, Jupyter
2. AWS CLI v2 설치:  https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
3. 전달받은 자격 증명으로 `aws configure` (region: us-east-1) → `aws sts get-caller-identity` 확인
4. Windows 사용자는 WSL2 (Ubuntu) 를 설치해 오세요.
5. (권장) Hugging Face 계정 생성 및 Access Token 발급 (다운로드 제한 회피)
6. 회사 노트북이라면 외부 SSH(22번) 접속이 허용되는지 확인해 주세요.
```

## 4. 시간표 예시

### 하루 과정 (6시간, 권장)

| 시간 | 모듈 | 비고 |
|---|---|---|
| 09:30 – 09:40 | 오리엔테이션 | 비용 규칙(`stop`) 강조 |
| 09:40 – 10:20 | 1. 개념 | |
| 10:20 – 11:00 | 2. 환경 구성 | **인스턴스 생성을 가장 먼저.** 기다리는 동안 설명 |
| 11:00 – 11:10 | 휴식 | |
| 11:10 – 12:00 | 3. 데이터 | |
| 12:00 – 12:10 | **4.6 학습 시작** | 점심 전에 학습을 띄워 둔다 |
| 12:10 – 13:10 | 점심 | 학습 진행 |
| 13:10 – 14:00 | 4. 학습 (나머지 이론 + 모니터링 과제) | |
| 14:00 – 14:40 | 5. 평가 | |
| 14:40 – 14:50 | 휴식 | |
| 14:50 – 15:20 | 6. 다음 단계 (Chat Vector 시연) | |
| 15:20 – 15:30 | **리소스 정리 확인** + 회고 | 모든 참가자가 stop/terminate 했는지 확인 |

### 반나절 과정 (3시간 30분)

- 모듈 1 을 20분으로 압축하고, 모듈 3 은 과제 3-1, 3-2 만 한다.
- 학습은 `EPOCHS = 1` 로 돌리거나, 5절의 **완성 모델을 받아서** 평가만 한다.
- 모듈 6 은 강사 시연으로 대신한다.

## 5. 학습이 늦어질 때를 위한 완성 모델 배포

리허설에서 만든 `models/k8s-ko-cpt-hf/final` 을 올려 두면, 학습이 실패하거나 늦은 참가자도 모듈 5 를 진행할 수 있다.

**Hugging Face Hub (비공개 저장소) 예시:**

```bash
# 강사 (리허설 인스턴스)
hf auth login
hf upload <조직>/k8s-ko-cpt-3b-workshop models/k8s-ko-cpt-hf/final --private

# 참가자 (자기 인스턴스, 강사가 읽기 전용 토큰 공유)
hf download <조직>/k8s-ko-cpt-3b-workshop --local-dir models/k8s-ko-cpt-hf/final --token <읽기토큰>
```

(`hf` 는 최신 huggingface_hub CLI 다. 구버전은 `huggingface-cli upload/download`.)

> **라이선스 주의:** Qwen2.5-3B 의 파생 모델을 배포하는 조건(Qwen Research License)을 확인한다.
> 비공개 저장소에서 참가자에게만 공유하는 범위로 제한하는 것을 권한다.

## 6. 운영 중·종료 후 관리 명령 (강사용)

```bash
# 워크숍 인스턴스 전체 현황
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=llm-ft-*" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,InstanceId,State.Name,LaunchTime]' \
  --output table

# 실행 중인 것 일괄 중지 (점심시간이 길어질 때 등)
ids=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=llm-ft-*" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
[ -n "$ids" ] && aws ec2 stop-instances --instance-ids $ids

# 종료 후: 남은 볼륨 (인스턴스는 지웠는데 볼륨이 남는 경우 확인)
aws ec2 describe-volumes --filters Name=status,Values=available \
  --query 'Volumes[].[VolumeId,Size,CreateTime]' --output table

# 워크숍 보안 그룹 / 키 페어 목록
aws ec2 describe-security-groups --filters "Name=group-name,Values=llm-ft-*" --query 'SecurityGroups[].[GroupName,GroupId]' --output table
aws ec2 describe-key-pairs --filters "Name=key-name,Values=llm-ft-*" --query 'KeyPairs[].KeyName' --output table
```

**AWS Budgets 알림**을 예산의 50%, 80%, 100% 에 걸어 두는 것을 권한다.

## 7. 비용 추정

> 아래는 대략적인 값이다. 실제 요금은 [EC2 요금](https://aws.amazon.com/ec2/pricing/on-demand/) 과
> [EBS 요금](https://aws.amazon.com/ebs/pricing/) 으로 다시 확인한다.

| 항목 | 단가 (us-east-1, 대략) | 1인 하루 (6시간) |
|---|---|---|
| g6e.4xlarge | 시간당 약 $3 | 약 $18 |
| EBS gp3 300GB | 월 약 $24 (용량) | 하루 약 $0.8 |
| EBS 추가 IOPS (3,000 → 6,000) / 처리량 (125 → 500MB/s) | 월 약 $15 + $15 | 하루 약 $1 |
| **합계** | | **1인 약 $20 / 일** |

- 인스턴스를 **stop 하지 않고 하룻밤 두면** 1대당 약 $50 이 더 나온다.
- stop 해도 EBS 요금(하루 약 $1.8)은 계속 나간다. 워크숍이 끝나면 terminate 를 권한다.
- 비용을 줄이려면 `VOLUME_GB=150` 과 기본 gp3 성능을 쓰는 방법이 있다. `launch_g6e.sh` 의 `Iops`/`Throughput` 을 지운다.

## 8. 진행 팁

- **인스턴스 생성을 가장 먼저 시킨다.** 쿼터나 재고 문제를 일찍 발견할 수 있다.
- 노트북 셀 출력이 강사 화면과 같은지 **체크포인트(✅)마다 손을 들어 확인**하게 한다.
- 학습이 도는 동안이 가장 좋은 질문 시간이다. 4.3 메모리 계산이나 4.7 loss 곡선 해석을 그때 한다.
- 참가자가 막히면 먼저 `!tail -50 logs/train.log` 와 `nvidia-smi` 를 보게 한다.
- 모듈 5 에서는 **"seen ppl 이 내려갔다 = 지식이 들어갔다" 는 함정**을 꼭 짚는다.
- 마지막 10분은 리소스 정리만 한다. 강사가 6절의 현황 명령으로 확인한다.

---

## 9. 확인 문제 · 과제 정답

### 모듈 1 확인 문제

1. CPT 는 **모든 토큰**, SFT 는 **답변(assistant/completion) 토큰만** loss 를 계산한다.
2. CPT 데이터는 문서 이어쓰기라 사전학습과 목적이 같다. instruct 모델에 하면 대화 정렬(지시 따르기, 답변 형식)이 무너지기 쉽다. 대화 능력은 CPT 뒤에 SFT 나 Chat Vector 로 다시 입힌다.
3. 같은 글자 수에 토큰이 더 많이 든다. 2,048 토큰 블록 하나에 담기는 내용이 줄고, 같은 문서를 학습하는 데 시간·비용·메모리가 더 든다. CPT 로 그 조각들의 조합에 익숙해지는 효과는 있다.
4. 파국적 망각이다 (+44%). replay 비율 ↑, 학습률 ↓, epoch ↓ 또는 이른 체크포인트 사용, LoRA 로 전환하는 방법이 있다.
5. 아니다. 세부 사실은 틀릴 수 있다. RAG 로 원문을 근거로 답하게 하고, 버전 정보는 출처를 함께 보여준다.

### 모듈 3

- **3-2:** `쿠버네티스` 7 토큰 vs `Kubernetes` 2, `파드` 2 vs `Pod` 1, `디플로이먼트` 6, `kubectl get pods -n kube-system` 7 (실측).
  영어는 같은 내용을 더 적은 토큰으로 표현하지만, 영어 원문은 문서 수가 3배라서 전체 토큰은 오히려 늘어난다 (3-6).
- **3-3:** 최대 2,033 토큰, 2,048 초과 0개 (실측). 짧은 블록은 주로 `reference/glossary/` 용어집 문서다.
- **3-5 (`--replay-ratio 0.4`, 실측):** replay 746개 / 762,095 토큰, train 1,668개, 1 epoch = 105 스텝.
  0.2 일 때(1,193개, 75 스텝)보다 학습 시간이 약 1.4배 든다.
- **3-6 (영어 포함, replay 0 으로 실측):** 쿠버네티스 블록 3,393개 / **4,064,566 토큰** (한글만일 때의 3.6배), 1 epoch = 213 스텝.
  replay 20% 를 더하면 약 270 스텝 → 학습 시간이 약 3.6배 든다.

### 모듈 4

- **4-1:** `nvidia-smi` 값은 예측보다 보통 몇 GB 크다. PyTorch 캐시 할당자가 해제한 메모리를 바로 돌려주지 않고 쌓아 두고(reserved),
  CUDA 컨텍스트와 cuBLAS 작업 공간도 수백 MB~1GB 를 쓴다. `torch.cuda.max_memory_allocated()` 가 실제 텐서 사용량에 더 가깝다.
- **4-3-1:** Qwen2.5-7B (약 76억) 기준 fp32 가중치 30.5 + 기울기 30.5 + 8bit Adam 15.2 ≈ 76GB + 활성값 → L40S 48GB 에 **들어가지 않는다.**
- **4-3-2:** LoRA (bf16 7B ≈ 15GB + α), QLoRA (4bit ≈ 5GB + α), 여러 GPU 에 FSDP/ZeRO-3 로 나누기 (예: g6e.12xlarge = L40S x4), 옵티마이저 CPU offload.
- **4-3-3:** 3-6 에 따라 약 3.6배. 시간은 리허설 측정값 x 3.6 으로 계산한다.

### 모듈 5

- **5-1 / 5-2:** 정해진 답은 없다. 확인할 점은 세 가지다.
  ① 문체가 문서체("~한다")로 바뀌었는가 ② 공식 번역어(파드, 디플로이먼트, 퍼시스턴트볼륨클레임)를 쓰는가
  ③ grep 으로 대조했을 때 원문 재현인지, 바꿔 말한 것인지, 환각인지.
- **5-3:** "쿠버네티스 1.40 ..." 같은 미래 정보는 **CPT 모델이 더 그럴듯한 환각**을 만들 수 있다. 문체가 문서 같아져서 더 믿음직해 보인다는 점이 위험하다.
  도메인 밖 프롬프트에서 쿠버네티스 이야기로 빠지면 replay 가 부족하다는 신호다.
- **5-4:** ① base 모델이라 질문에 답하지 않고 질문을 이어 쓰거나 비슷한 질문을 나열한다
  ② SFT 또는 Chat Vector (모듈 6) ③ RAG 와 출처 표시, 그리고 평가 세트로 주기적으로 회귀 테스트한다.
