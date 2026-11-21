# 모듈 2. 실습 환경 구성 — AWS g6e.4xlarge + VS Code + Jupyter

**소요 40분** · 실습

## 학습 목표

- 스크립트로 GPU 인스턴스를 만들고, 비용이 나가지 않게 관리할 수 있다.
- 로컬 VS Code 에서 원격 GPU 서버의 파일을 편집하고 노트북을 실행할 수 있다.

## 전체 구조

```
┌──────────── 로컬 PC ────────────┐            ┌────────── AWS us-east-1 ──────────┐
│ VS Code                         │   SSH :22  │ g6e.4xlarge                       │
│  ├ Remote-SSH 확장 ─────────────┼───────────▶│  ├ NVIDIA L40S 48GB               │
│  └ 노트북 UI (화면만 로컬)       │ (내 IP 만) │  ├ ~/llm-ft/.venv (Python 3.12)   │
│                                 │            │  ├ Jupyter 커널 "Python (llm-ft)" │
│ 터미널                           │            │  └ 학습 프로세스 (nohup)          │
│  ├ aws/launch_g6e.sh  생성       │            │                                   │
│  ├ aws/sync.sh        업·다운로드│            │ 보안 그룹: 22번만, 내 IP/32 만    │
│  └ aws/ctl.sh         켜기/끄기  │            │ EBS 300GB gp3                     │
└─────────────────────────────────┘            └───────────────────────────────────┘
```

Jupyter 포트(8888)는 **인터넷에 열지 않는다.** VS Code 가 SSH 연결 안에서 커널과 통신한다.

### g6e.4xlarge 사양

| 항목 | 값 |
|---|---|
| GPU | NVIDIA L40S x1, 48GB (실제 사용 가능 약 44.7 GiB) |
| vCPU / RAM | 16 / 128 GiB |
| 로컬 NVMe | 600GB (`/opt/dlami/nvme`, **인스턴스를 중지하면 지워짐**) |
| 요금 (us-east-1, 온디맨드) | 대략 시간당 $3 — 실습 전에 [요금표](https://aws.amazon.com/ec2/pricing/on-demand/) 확인 |

> ⚠️ **비용 규칙: 실습이 끝나면 반드시 `./aws/ctl.sh stop`.** 켜 둔 채 잊어버리면 하루에 $70 이상 나온다.

---

## 2.1 로컬 PC 준비

### (1) 필요한 도구

| 도구 | 확인 명령 | 설치 |
|---|---|---|
| AWS CLI v2 | `aws --version` | https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html |
| VS Code | `code --version` | https://code.visualstudio.com |
| ssh, rsync, curl | `ssh -V; rsync --version` | macOS/Linux 기본 포함 |

> **Windows 사용자:** 스크립트가 bash 용이라 **WSL2 (Ubuntu)** 안에서 실행한다.
> 단, Windows 용 VS Code 는 WSL 의 `~/.ssh/config` 가 아니라 `C:\Users\<이름>\.ssh\config` 를 읽는다.
> [부록: Windows + WSL](99_troubleshooting.md#windows--wsl-에서-vs-code-가-k8s-gpu-를-못-찾음) 을 참고한다.

### (2) AWS 자격 증명

```bash
aws configure            # Access Key, Secret Key, region: us-east-1, output: json
aws sts get-caller-identity
```

✅ **확인:** 계정 ID 와 ARN 이 출력되어야 한다.

### (3) VS Code 확장 (로컬)

```bash
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

### (4) 실습 코드 받기

```bash
git clone <강사가 안내한 저장소 주소> ~/llm-ft     # 또는 받은 zip 파일을 ~/llm-ft 에 풀기
cd ~/llm-ft
chmod +x aws/*.sh *.sh
```

## 2.2 GPU 쿼터 확인

AWS 는 계정마다 GPU 인스턴스를 몇 vCPU 까지 띄울 수 있는지 한도(쿼터)를 둔다. 새 계정은 이 값이 **0** 인 경우가 많다.

```bash
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
  --query 'Quota.[QuotaName,Value]' --output text
# Running On-Demand G and VT instances   768.0
```

- g6e.4xlarge 1대에 **16** 이 필요하다.
- 계정 하나를 여러 참가자가 같이 쓰면 `16 x 인원수` 가 필요하다.
- 부족하면 콘솔의 **Service Quotas → Amazon EC2 → Running On-Demand G and VT instances** 에서 증설을 요청한다. 승인까지 수 시간~1일 걸린다.

## 2.3 인스턴스 생성

```bash
./aws/launch_g6e.sh
```

**공유 계정을 쓴다면** 다른 참가자와 겹치지 않게 이름을 붙인다:

```bash
NAME=llm-ft-kim KEY_NAME=llm-ft-kim SSH_HOST=gpu-kim ./aws/launch_g6e.sh
# 이후 모든 aws/*.sh 실행에도 같은 환경변수를 붙인다. 셸에 export 해 두면 편하다:
export NAME=llm-ft-kim KEY_NAME=llm-ft-kim SSH_HOST=gpu-kim
```

이렇게 하면 이 문서의 `k8s-gpu` 는 모두 `gpu-kim` 으로 바꿔 읽으면 된다.

### 스크립트가 하는 일

| 단계 | 하는 일 | 왜 |
|---|---|---|
| 1. 키 페어 | `llm-ft-g6e` 생성, 개인키를 `~/.ssh/llm-ft-g6e.pem` 에 저장 (권한 400) | 개인키는 로컬에만 둔다 |
| 2. 보안 그룹 | `llm-ft-g6e-sg`: **현재 내 공인 IP/32 → 22번** 만 허용 | 외부 노출 최소화 |
| 3. AMI 조회 | *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)* 최신판 | NVIDIA 드라이버와 CUDA 가 이미 설치됨 |
| 4. 인스턴스 생성 | g6e.4xlarge, 루트 EBS 300GB gp3 (6000 IOPS / 500MB/s), IMDSv2 강제 | 모델 다운로드와 체크포인트를 빠르게 읽고 씀 |
| 5. SSH 설정 | `~/.ssh/config` 에 `Host k8s-gpu` 블록 추가 | `ssh k8s-gpu` 로 바로 접속 |
| 6. 확인 | SSH 가 열릴 때까지 기다린 뒤 `nvidia-smi` 실행 | GPU 인식 확인 |

### 예상 출력

```
키 페어 생성: /Users/you/.ssh/llm-ft-g6e.pem
보안 그룹 sg-0abc... : SSH from 122.xx.xx.xx/32
AMI: ami-032f90794e9cc5df4 (Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261117)
인스턴스 생성: i-0123...  (running 대기 중...)
~/.ssh/config 갱신: Host k8s-gpu -> 54.xx.xx.xx
SSH 준비 대기 중...
name, memory.total [MiB], driver_version
NVIDIA L40S, 46068 MiB, 5xx.xx
```

`~/.ssh/config` 에 추가되는 블록:

```
# >>> llm-ft k8s-gpu >>>
Host k8s-gpu
  HostName 54.xx.xx.xx
  User ubuntu
  IdentityFile /Users/you/.ssh/llm-ft-g6e.pem
  HostKeyAlias llm-ft-g6e          ← IP 가 바뀌어도 호스트 키 경고가 나지 않게
  StrictHostKeyChecking accept-new
  ServerAliveInterval 60           ← 오래 켜 둬도 연결이 끊기지 않게
  ServerAliveCountMax 10
# <<< llm-ft k8s-gpu <<<
```

> `InsufficientInstanceCapacity` 오류가 나면 그 AZ 에 GPU 재고가 없다는 뜻이다. AZ 를 바꿔서 다시 실행한다:
> `AZ=us-east-1b ./aws/launch_g6e.sh` (us-east-1a~d 모두 g6e.4xlarge 를 제공한다)

✅ **확인:** `ssh k8s-gpu nvidia-smi` 를 실행하면 L40S 가 보여야 한다.

## 2.4 코드 업로드와 원격 환경 구성

```bash
./aws/sync.sh up                                   # 로컬 ~/llm-ft → 원격 ~/llm-ft
ssh k8s-gpu 'bash ~/llm-ft/aws/setup_remote.sh'    # 약 5분
```

`sync.sh up` 은 `.venv`, `raw`, `adapters`, `models` 를 **제외하고** 올린다.
로컬 Mac 의 가상환경을 Linux 서버로 보내면 동작하지 않기 때문이다.

`setup_remote.sh` 가 원격에서 하는 일:

```bash
sudo apt-get install -y tmux rsync git
curl -LsSf https://astral.sh/uv/install.sh | sh          # 빠른 Python 패키지 관리자
uv venv -p 3.12 .venv                                     # Python 3.12 가상환경
uv pip install -r requirements-cuda.txt ipykernel jupyterlab ipywidgets matplotlib
python -m ipykernel install --user --name llm-ft --display-name "Python (llm-ft)"
```

✅ **확인:** 마지막 줄이 아래처럼 나와야 한다.
```
torch 2.x.x | CUDA 12.x | available: True | NVIDIA L40S
```

## 2.5 VS Code 로 원격 접속

1. VS Code 에서 `F1` (또는 `Cmd/Ctrl+Shift+P`) → **Remote-SSH: Connect to Host...** → `k8s-gpu`
   - 터미널에서 바로 열 수도 있다: `code --remote ssh-remote+k8s-gpu /home/ubuntu/llm-ft`
2. 처음 접속하면 원격에 *VS Code Server* 가 자동으로 설치된다 (1~2분).
   창 왼쪽 아래에 **`SSH: k8s-gpu`** 가 표시되면 연결된 것이다.
3. **File → Open Folder...** → `/home/ubuntu/llm-ft`
4. 확장(Extensions) 탭에서 **Python**, **Jupyter** 를 찾아 **Install in SSH: k8s-gpu** 를 누른다.
   로컬에 설치한 확장은 원격에 자동으로 깔리지 않는다.
5. 터미널 열기 (`Ctrl+\``): 이 터미널은 **원격 서버의 셸**이다. `nvidia-smi` 를 쳐 보자.

## 2.6 노트북 커널 연결

1. 탐색기에서 **`k8s_cpt_g6e.ipynb`** 를 연다.
2. 오른쪽 위 **Select Kernel** → **Jupyter Kernel...** → **Python (llm-ft)**
   - 목록에 없으면 **Python Environments...** → `~/llm-ft/.venv/bin/python` 을 고르거나,
     `F1` → **Developer: Reload Window** 후 다시 시도한다.
3. 첫 번째 코드 셀을 실행한다 (`Shift+Enter`).

✅ **확인:** 출력에 다음이 보여야 한다.
```
torch 2.x.x | NVIDIA L40S | 44.x GiB
```

### (선택) 브라우저에서 JupyterLab 쓰기

```bash
# 원격 (VS Code 터미널)
tmux new -s jlab
cd ~/llm-ft && source .venv/bin/activate
jupyter lab --no-browser --ip 127.0.0.1 --port 8888
# 출력된 URL 의 token 을 복사한 뒤 Ctrl-b d 로 tmux 에서 빠져나옴 (서버는 계속 실행)

# 로컬 터미널
ssh -N -L 8888:localhost:8888 k8s-gpu
# 브라우저: http://localhost:8888/lab?token=<토큰>
```

VS Code 의 **PORTS** 탭에서 8888 을 포워딩해도 같은 효과를 낸다.

## 2.7 인스턴스 관리 명령

```bash
./aws/ctl.sh status       # i-0123  g6e.4xlarge  running  54.xx.xx.xx
./aws/ctl.sh stop         # 중지: GPU 과금이 멈추고 EBS 요금만 나감
./aws/ctl.sh start        # 재시작: 공인 IP 가 바뀌므로 ~/.ssh/config 를 자동 갱신
./aws/ctl.sh allow-my-ip  # 카페 등으로 장소를 옮겨 내 IP 가 바뀌었을 때
./aws/ctl.sh terminate    # 영구 삭제 (이름을 입력해 확인)
```

> 다시 시작한 뒤 VS Code 가 접속하지 못하면 **Remote-SSH: Connect to Host** 로 다시 접속한다.
> `~/.ssh/config` 의 IP 는 `ctl.sh start` 가 이미 바꿔 두었다.

---

## 체크포인트

다음이 모두 되면 모듈 3 으로 넘어간다.

- [ ] `ssh k8s-gpu nvidia-smi` 에 L40S 가 보인다
- [ ] VS Code 왼쪽 아래에 `SSH: k8s-gpu` 가 표시된다
- [ ] 노트북 커널이 `Python (llm-ft)` 이고 첫 셀이 GPU 를 출력한다
- [ ] `./aws/ctl.sh stop` 명령을 어디서 실행하는지 안다

---

## 대안: Apple Silicon Mac 에서 로컬로 (MLX)

AWS 를 쓸 수 없거나 집에서 복습할 때 쓴다. M 시리즈 칩에 메모리 24GB 이상이 필요하다.

```bash
cd ~/llm-ft
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-mac.txt datasets
./01_download.sh && python 02_prepare_data.py
python 03_build_cpt.py --max-tokens 2048 --batch-size 2   # Mac 은 2K 블록 (config_cpt.yaml 과 짝)
mlx_lm.lora -c config_cpt.yaml          # LoRA r=128, 모든 레이어
python 05_eval_cpt.py                   # 원본 vs CPT 비교
```

M4 Pro 24GB 실측: 약 **190 tok/s**, 1 epoch 약 2시간, 4 epoch 약 8시간.
gradient checkpointing 을 끄면 메모리 부족(OOM)이 난다.
워크숍 시간 안에 끝나지 않으므로 Mac 은 복습용이다.

---
← [모듈 1](01_concepts.md) · 다음: [모듈 3. 데이터 파이프라인 →](03_data.md)
