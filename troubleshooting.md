# 부록. 문제 해결 · FAQ · 용어집

## A. 문제 해결

### AWS / 인스턴스

| 증상 | 원인 | 해결 |
|---|---|---|
| `VcpuLimitExceeded` | G 인스턴스 쿼터 부족 | 모듈 2.2. Service Quotas 에서 `Running On-Demand G and VT instances` 증설 |
| `InsufficientInstanceCapacity` | 그 AZ 에 GPU 재고 없음 | `AZ=us-east-1b ./aws/launch_g6e.sh` (a~d 차례로 시도) |
| `이미 'llm-ft-g6e' 인스턴스가 있습니다` | 같은 이름의 인스턴스가 있음 | `./aws/ctl.sh start` 로 재사용하거나 `NAME=` 을 바꿈 |
| `AWS 에 키 페어 ... 는 있는데 ... 이 없습니다` | 다른 PC 에서 만든 키 | `KEY_NAME=llm-ft-g6e-2 ./aws/launch_g6e.sh` |
| `UnauthorizedOperation` | IAM 권한 부족 | EC2 (RunInstances, CreateSecurityGroup, CreateKeyPair, Describe*) 권한 필요. 강사에게 문의 |
| `SSH 준비 대기 중...` 에서 몇 분째 멈춤 | 보안 그룹 IP 불일치 (VPN, 사내 프록시) | VPN 을 끄고 `./aws/ctl.sh allow-my-ip`. 회사 네트워크가 22번을 막는 경우도 있음 |

### SSH / VS Code

| 증상 | 해결 |
|---|---|
| `ssh: connect to host ... port 22: Operation timed out` | ① `./aws/ctl.sh status` 로 running 인지 확인 ② 장소를 옮겼다면 `./aws/ctl.sh allow-my-ip` ③ 재시작했다면 `./aws/ctl.sh ssh-config` |
| `Permission denied (publickey)` | `ls -l ~/.ssh/llm-ft-g6e.pem` 권한이 `-r--------` 인지 확인 (`chmod 400`). `~/.ssh/config` 의 `IdentityFile` 경로 확인 |
| `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED` | 인스턴스를 새로 만들었는데 이름이 같은 경우. `ssh-keygen -R llm-ft-g6e` |
| VS Code: *Could not establish connection* | 먼저 터미널에서 `ssh k8s-gpu` 가 되는지 확인. 된다면 `F1` → **Remote-SSH: Kill VS Code Server on Host...** 후 다시 접속 |
| VS Code 가 접속 중에 계속 멈춤 | 원격 디스크가 가득 찼을 수 있음: `ssh k8s-gpu df -h /` |

#### Windows + WSL 에서 VS Code 가 `k8s-gpu` 를 못 찾음

Windows 용 VS Code 는 `C:\Users\<이름>\.ssh\config` 를 읽는다. 스크립트는 WSL 의 `~/.ssh/config` 를 고친다.

방법 1 — 키와 설정을 Windows 쪽으로 복사:
```bash
# WSL 에서
WIN_SSH=/mnt/c/Users/<윈도우사용자>/.ssh
mkdir -p $WIN_SSH && cp ~/.ssh/llm-ft-g6e.pem $WIN_SSH/
sed -n '/# >>> llm-ft k8s-gpu >>>/,/# <<< llm-ft k8s-gpu <<</p' ~/.ssh/config \
  | sed "s#IdentityFile .*#IdentityFile C:/Users/<윈도우사용자>/.ssh/llm-ft-g6e.pem#" >> $WIN_SSH/config
```
Windows 에서 `.pem` 파일 권한이 너무 넓다는 오류가 나면, 파일 속성 → 보안에서 본인 계정만 남긴다.

방법 2 — VS Code 를 WSL 안에서 실행: WSL 터미널에서 `code .` 로 연 뒤 Remote-SSH 를 쓴다.
WSL 확장이 설치되어 있어야 한다.

### Jupyter / 노트북

| 증상 | 해결 |
|---|---|
| 커널 목록에 `Python (llm-ft)` 이 없음 | ① 원격에 Jupyter 확장 설치 (Install in SSH) ② `F1` → **Developer: Reload Window** ③ `ssh k8s-gpu 'bash ~/llm-ft/aws/setup_remote.sh'` 재실행 |
| 셀에서 `!` 명령이 엉뚱한 Python 을 씀 | 노트북은 `.venv/bin/python` 을 명시해서 호출한다. 직접 추가한 셀에서도 `!python` 대신 `!.venv/bin/python` 을 쓴다 |
| `ModuleNotFoundError: 02_prepare_data` | 작업 디렉터리가 `~/llm-ft` 가 아님. 첫 셀(`os.chdir`)을 먼저 실행 |
| 평가 셀에서 CUDA OOM | 학습 프로세스가 아직 GPU 를 쓰는 중: `!pgrep -f train_cpt_hf.py`. 끝난 뒤 커널 재시작 후 실행 |

### 데이터

| 증상 | 해결 |
|---|---|
| 위키 스트리밍이 느리거나 실패 | 네트워크 문제. 우선 `--replay-ratio 0` 으로 진행하고 나중에 다시 만든다 |
| HF 다운로드 `429 Too Many Requests` | 공유 IP 에서 많이 받으면 발생. `huggingface-cli login` 또는 `export HF_TOKEN=...` |
| `03_build_cpt.py` 실행 위치 오류 | 반드시 `~/llm-ft` 에서 실행 (형제 모듈 `02_prepare_data.py` 를 import 함). `python -I` 로 실행하면 안 된다 |

### 학습

| 증상 | 해결 |
|---|---|
| `torch.OutOfMemoryError: CUDA out of memory` | `EXTRA = "--batch-size 1 --grad-accum 16"`. 그래도 안 되면 LoRA (`MODE = ""`) |
| `bitsandbytes` 관련 오류 | `uv pip install -U bitsandbytes`. 급하면 `train_cpt_hf.py` 의 `optim` 을 `adamw_torch_fused` 로 바꾸고 배치를 1로 (메모리 약 18GB 증가) |
| `TypeError: ... unexpected keyword argument 'max_length'` (SFTConfig) | TRL 이 너무 오래된 버전. `uv pip install -U "trl>=0.20"` |
| `torch_dtype is deprecated` 경고 | 최신 transformers 의 경고일 뿐이며 동작에는 문제 없음 |
| loss 가 `nan` | `--lr 1e-5` 로 낮춘다 |
| 학습이 시작되지 않음 | `!tail -50 logs/train.log` 로 오류 확인 |
| 학습을 중간에 멈추고 싶음 | `!pkill -f train_cpt_hf.py`. 마지막 체크포인트는 `models/k8s-ko-cpt-hf/checkpoint-*` 에 남는다 |

### Mac (MLX)

| 증상 | 해결 |
|---|---|
| `[METAL] ... Insufficient Memory` | `grad_checkpoint: true` 확인 (끄면 24GB 에서 OOM 이 실측됨). `batch_size: 1` |
| 학습이 너무 느림 | M4 Pro 에서 약 190 tok/s 는 정상. `iters` 를 줄여 1 epoch 만 돌려 본다 |



## B. FAQ

**Q. 왜 3B 인가? 7B 가 더 똑똑하지 않나?**
7B 는 full 학습에 메모리가 약 90GB 필요해서 L40S 1장에 들어가지 않는다. 실습 시간 안에 끝나는 크기가 3B 다.
방법 자체는 같으므로 7B 이상은 LoRA/QLoRA 나 여러 GPU 를 쓰면 된다.

**Q. epoch 를 많이 돌리면 지식이 더 많이 들어가나?**
어느 정도까지는 그렇다. 하지만 과적합(원문 암기)과 망각이 함께 커진다.
`k8s_seen` ↓ 와 `general_ko` ↑ 사이에서 균형점을 찾아야 한다. 같은 문장을 반복하는 것보다 **다른 표현으로 증강**하는 편이 효과적이다.

**Q. 사내 문서로 해도 되나?**
방법은 같다. 대신 **보안 등급**(외부 클라우드 GPU 사용 가능 여부), **개인정보 제거**,
**문서 최신성**(오래된 문서는 오래된 지식을 넣는다)을 먼저 확인한다.

**Q. CPT 를 하면 RAG 는 필요 없나?**
아니다. CPT 는 도메인 이해력을 높이지만 **정확한 사실 인용을 보장하지 않는다.**
버전마다 바뀌는 정보, 정확해야 하는 수치는 RAG 로 원문을 근거로 답하게 하는 것이 안전하다.

**Q. 학습한 모델을 공개해도 되나?**
쿠버네티스 문서(CC BY 4.0)는 출처를 표기하면 된다. 위키백과(CC BY-SA 4.0)도 섞었다.
그리고 기본 모델 라이선스(Qwen Research License)를 반드시 확인한다. 상업 이용은 제한될 수 있다.

**Q. 왜 Jupyter 포트를 열지 않고 SSH 를 쓰나?**
Jupyter 는 브라우저에서 임의의 코드를 실행하는 서버다. 인터넷에 노출되면 토큰이 유출되는 순간 서버 전체가 넘어간다.
SSH 터널은 키 인증과 암호화를 그대로 이용한다.


## C. 용어집

| 용어 | 설명 |
|---|---|
| **CPT** (Continued Pre-Training) | 사전학습된 모델에 도메인 원문으로 다음 토큰 예측 학습을 계속하는 것. DAPT 라고도 한다 |
| **SFT** (Supervised Fine-Tuning) | 질문-답변 쌍으로 대화·지시 따르기를 학습시키는 것 |
| **RAG** (Retrieval-Augmented Generation) | 질문과 관련된 문서를 검색해서 프롬프트에 넣고 답하게 하는 것 |
| **base / instruct 모델** | 사전학습만 한 모델 / 대화용으로 추가 학습한 모델 |
| **토큰** | 모델이 처리하는 텍스트 단위. 한국어는 이번 데이터에서 평균 1.85 글자 |
| **perplexity** | `exp(평균 loss)`. 낮을수록 텍스트를 잘 예측함 |
| **패킹** | 짧은 텍스트를 이어 붙여 최대 길이 블록을 채우는 것. 계산 낭비와 잘림을 줄인다 |
| **replay** | 망각을 막으려고 일반 데이터를 섞는 것 |
| **파국적 망각(Catastrophic Forgetting)** | 새 데이터를 학습하면서 기존 능력을 잃는 현상 |
| **epoch** | 학습 데이터를 한 바퀴 다 보는 것 |
| **step / iter** | 가중치를 한 번 갱신하는 단위. mlx-lm 의 `iters` 는 마이크로배치 단위라서 다르다 |
| **실효 배치** | `배치 크기 x gradient accumulation 횟수`. 한 번 갱신에 쓰는 샘플 수 |
| **gradient accumulation** | 작은 배치를 여러 번 돌려 기울기를 모은 뒤 한 번에 갱신 → 큰 배치 효과 |
| **gradient checkpointing** | 중간 활성값을 저장하지 않고 역전파 때 다시 계산 → 메모리 절약, 속도 감소 |
| **bf16 / fp32** | 16비트 / 32비트 부동소수. bf16 은 범위는 넓지만 정밀도가 낮다 |
| **autocast (혼합 정밀도)** | 가중치는 fp32 로 두고 계산만 bf16 으로 하는 것 |
| **8bit 옵티마이저** | Adam 의 모멘텀 상태를 8비트로 저장해 메모리를 1/4 로 줄인 것 |
| **LoRA** | 원래 가중치는 고정하고 작은 저랭크 행렬(rank r)만 학습하는 방법 |
| **full fine-tuning** | 모든 가중치를 학습하는 것 |
| **tied embeddings** | 입력 임베딩과 출력층(lm_head)이 같은 가중치를 공유하는 구조 |
| **Chat Vector** | `instruct + (cpt − base)` 로 학습 없이 대화 능력과 도메인 지식을 합치는 기법 |
| **warmup / cosine** | 학습률을 처음에 서서히 올리고(warmup), 이후 코사인 곡선으로 줄이는 스케줄 |
| **sparse checkout** | git 저장소에서 일부 디렉터리만 작업 트리에 받는 기능 |
| **Hugo 숏코드** | `{{< ... >}}` 형식의 정적 사이트 생성기 문법 |

