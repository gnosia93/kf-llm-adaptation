# 딥러닝 GPU 학습(싱글/멀티 노드) Top 20 장애 및 트러블슈팅 가이드

딥러닝 분산 학습의 장애는 크게 **① 메모리·자원 고갈**, **② 학습 수치·모델 오류**, **③ 멀티 GPU/노드 통신(NCCL)**, **④ 스토리지·데이터 파이프라인**, **⑤ 인프라·하드웨어 결함** 5가지 영역에서 발생한다. 실무에서 가장 빈번한 20가지 사례와 해결책을 정리했다.

---

## [영역 1] GPU 및 시스템 메모리 고갈 (1~4)

### 1. CUDA Out of Memory (OOM)
* **증상:** `RuntimeError: CUDA out of memory. Tried to allocate ...` 에러와 함께 학습 즉시 중단.
* **원인:** 모델 가중치, 옵티마이저 상태(AdamW 기준 모델의 약 4배), 활성화 값(Activation)의 합이 VRAM 용량을 초과함. 특히 유독 긴 시퀀스 배치가 들어올 때 갑자기 터진다.
* **해결책:** `per_device_train_batch_size`를 줄이고 `gradient_accumulation_steps`를 늘린다. `gradient_checkpointing=True`를 켜고, `bf16`, 8-bit 옵티마이저(`adamw_bnb_8bit`), DeepSpeed ZeRO-2/3를 적용한다.

### 2. GPU 메모리 단편화 (CUDA Memory Fragmentation)
* **증상:** `nvidia-smi`상 여유 VRAM이 수 GB 남아 있는데도 `Tried to allocate 20.00 MiB ... reserved in total by PyTorch` 메시지와 함께 OOM 발생.
* **원인:** 길이가 들쭉날쭉한 가변 길이 배치가 반복되면서 PyTorch 캐시 메모리 공간이 잘게 쪼개져 연속된 메모리 블록을 할당하지 못함.
* **해결책:** 실행 전 환경변수 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 설정해 동적 메모리 확장을 활성화하고, 배치 내 시퀀스 길이를 비슷하게 묶어주는 `group_by_length=True` 옵션을 사용한다.

### 3. CPU RAM 부족으로 인한 프로세스 강제 종료 (`Killed` / Exit Code 137)
* **증상:** 파이썬 에러 로그 없이 터미널에 `Killed`만 뜨거나 컨테이너가 Exit Code 137로 조용히 종료됨.
* **원인:** 호스트 시스템의 **CPU RAM이 고갈**되어 리눅스 커널(OOM Killer)이 프로세스를 강제 사살함. 대용량 데이터셋 토크나이징이나 모델 체크포인트 저장 순간에 주로 발생한다.
* **해결책:** `dmesg -T | grep -i oom`으로 커널 OOM 여부를 확인한다. 데이터셋은 스트리밍(`streaming=True`) 또는 메모리 매핑(Arrow)으로 읽고, DataLoader의 `num_workers` 수를 줄인다.

### 4. 공유 메모리 부족 (`DataLoader worker exited unexpectedly` / `/dev/shm`)
* **증상:** `RuntimeError: DataLoader worker (pid ...) is killed by signal: Bus error` 발생.
* **원인:** Docker나 쿠버네티스 컨테이너의 기본 공유 메모리(`/dev/shm`) 크기가 `64MB`로 제한되어 멀티 워커 데이터 전송 시 공간이 부족해짐.
* **해결책:** Docker 실행 시 `--shm-size=16g` 이상 또는 `--ipc=host` 옵션을 부여한다. 쿠버네티스 환경에서는 `emptyDir` 볼륨(`medium: Memory`)을 `/dev/shm`에 마운트한다.

---

## [영역 2] 학습 수치 불안정 및 설정 오류 (5~8)

### 5. Loss가 `NaN`으로 발산하거나 `grad_norm` 폭주
* **증상:** 학습 도중 갑자기 `loss: nan`이 찍히거나 Gradient Norm이 수천 단위로 치솟은 뒤 모델 출력이 붕괴됨.
* **원인:** `FP16` 학습 시 표현 범위 초과(Overflow)가 발생했거나 학습률(Learning Rate)이 과도함. 또는 유효 토큰이 하나도 없는(`labels` 전체가 `-100`) 배치가 입력됨.
* **해결책:** Ampere(A100, RTX 30/40시리즈) 이상 GPU에서는 반드시 **`bf16=True`**를 사용한다. `max_grad_norm=1.0`으로 그래디언트를 자르고, `labels` 전체가 `-100`인 빈 샘플을 전처리 단계에서 제거한다.

### 6. Loss가 전혀 줄어들지 않고 제자리걸음
* **증상:** 수백 스텝이 지나도 Loss 값이 초기값에서 거의 변하지 않거나 모델이 아무것도 배우지 못함.
* **원인:** 학습률이 너무 작거나, LoRA 설정 시 `target_modules` 이름을 잘못 지정해 실제 어댑터가 모델 레이어에 붙지 않음. 또는 `labels` 마스킹 오류로 답변 토큰까지 `-100` 처리됨.
* **해결책:** `model.print_trainable_parameters()`로 학습 가능한 파라미터 비율이 `0%`가 아닌지 확인하고, 데이터셋 1개를 디코딩하여 `labels`에 실제 답변 토큰 ID가 살아 있는지 검증한다.

### 7. 평가(Evaluation) 단계 진입 시 갑작스러운 OOM 발생
* **증상:** 훈련(Train) 스텝은 멀쩡히 잘 돌다가, `eval_steps`에 도달해 검증을 시작하는 순간 OOM으로 죽음.
* **원인:** 검증 과정에서 모든 배치의 예측 로짓(Logits, 어휘 크기 × 시퀀스 길이)을 CPU로 내리지 않고 GPU 메모리에 한꺼번에 누적하려다 메모리 초과 발생.
* **해결책:** `eval_accumulation_steps=1` 또는 `4`로 설정해 검증 결과를 매 스텝 CPU RAM으로 즉시 옮기도록 하고, `per_device_eval_batch_size`를 훈련 배치보다 작게 줄인다.

### 8. 파국적 망각(Catastrophic Forgetting) 및 답변 폭주(EOS 미인식)
* **증상:** 전문 지식 학습 후 일상 질문에도 딱딱한 문서체로 답하거나, 답변을 끝내지 못하고 가상의 사용자 질문까지 혼자 지어냄.
* **원인:** 일반 대화 데이터(Replay) 비율이 부족하거나 학습률이 너무 높음. 답변 폭주는 `generation_config`에 대화 턴 종료 토큰(`<|im_end|>`)이 EOS로 등록되지 않아 발생.
* **해결책:** 일반 대화 데이터 비율을 30~40%로 늘리고 학습률을 낮춘다. 모델 저장 시 `generation_config.eos_token_id`에 기본 EOS와 턴 종료 토큰을 모두 리스트로 추가한다.

---

## [영역 3] 멀티 GPU · 멀티 노드 분산 통신 장애 (9~14)

### 9. NCCL 타임아웃 및 무한 대기 (`Watchdog caught collective operation timeout`)
* **증상:** 학습이 특정 스텝에서 멈춘 채 GPU 사용률이 0%로 떨어지고, 10~30분 뒤 `NCCL timeout` 에러를 내며 종료됨.
* **원인:** 가중치 동기화(`AllReduce`) 시점에 **특정 1개 GPU(Rank)만 작업이 지연되거나 먼저 죽어서** 나머지 GPU들이 무한 대기 상태에 빠짐. (예: Rank 0만 혼자 체크포인트 저장/전처리를 오래 수행)
* **해결책:** `NCCL_DEBUG=INFO`와 `TORCH_DISTRIBUTED_DEBUG=DETAIL`을 켜서 어느 노드·Rank에서 멈췄는지 확인한다. Rank 0 전용 작업 후 `dist.barrier()` 동기화를 맞추고, 필요시 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`을 늘린다.

### 10. 멀티 GPU 환경에서 `device_map="auto"` 충돌
* **증상:** `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!`
* **원인:** DDP/FSDP/DeepSpeed 같은 분산 학습 런처(`torchrun`)를 쓰면서 모델 로드 옵션에 추론용 파이프라인 병렬화 옵션인 `device_map="auto"`를 함께 넣음.
* **해결책:** 분산 학습 시에는 반드시 **`device_map=None`**으로 설정하여 각 프로세스가 할당받은 단일 GPU(`LOCAL_RANK`)에만 모델을 올리도록 한다.

### 11. 멀티 노드 간 네트워크 인터페이스(NIC) 오인식
* **증상:** 싱글 노드에서는 정상 작동하나, 2대 이상의 노드를 묶으면 초기화 단계에서 `NCCL error: unhandled system error` 또는 연결 실패 발생.
* **원인:** 노드 간 고속 통신망(InfiniBand/RoCE)이 아닌 도커 가상 브릿지(`docker0`)나 저속 관리망 이더넷 인터페이스로 통신을 시도함.
* **해결책:** `ip addr`로 고속 통신망 인터페이스 이름(예: `eth0`, `ens5`, `ib0`)을 확인해 `export NCCL_SOCKET_IFNAME=인터페이스명`을 명시한다. InfiniBand 미지원 환경이면 `export NCCL_IB_DISABLE=1`을 추가한다.

### 12. 마스터 노드 포트 충돌 및 방화벽 차단 (`Address already in use` / `ECONNREFUSED`)
* **증상:** 멀티 노드 실행 시 워커 노드가 마스터 노드에 접속하지 못하거나, `socket.error: Address already in use` 발생.
* **원인:** 이전 학습에서 죽지 않고 남은 좀비 파이썬 프로세스가 포트를 점유하고 있거나, 노드 간 통신 포트(예: `29500` 및 동적 포트 대역)가 방화벽·보안 그룹에 막혀 있음.
* **해결책:** 모든 노드에서 `pkill -9 python`으로 잔여 프로세스를 정리하고, 노드 간 내부 네트워크의 모든 TCP 포트 통신이 열려 있는지 방화벽을 확인한다.

### 13. 노드 간 라이브러리·코드·토크나이저 버전 불일치
* **증상:** 학습이 시작되자마자 특정 노드만 모듈 에러가 나거나, 실행 중 텐서 크기 불일치(`size mismatch`)로 크래시 발생.
* **원인:** 노드 0번과 노드 1번의 파이썬 패키지 버전, 로컬 소스 코드 수정 내역, 또는 캐시된 토크나이저 파일이 서로 다름.
* **해결책:** 모든 노드가 동일한 **Docker 컨테이너 이미지**를 실행하도록 강제하고, 코드와 데이터는 공유 파일시스템(NFS/Lustre)이나 배포 스크립트로 100% 동기화한다.

### 14. FSDP / DeepSpeed ZeRO-3 체크포인트 저장·병합 실패
* **증상:** 분산 학습은 끝났는데 저장된 체크포인트에 가중치가 비어 있거나(`0 bytes`), 추론용 단일 모델로 불러올 때 키 누락 에러 발생.
* **원인:** ZeRO-3나 FSDP는 모델 가중치를 여러 GPU에 조각내어(Sharding) 들고 있기 때문에, Rank 0만 일반 `save_pretrained()`를 호출하면 자기 조각만 저장하게 됨.
* **해결책:** `FULL_STATE_DICT` 옵션을 켜서 모든 GPU의 조각을 모은 뒤 저장하거나, DeepSpeed 제공 스크립트(`zero_to_fp32.py`)를 실행해 통합 가중치 파일로 변환한다.

---

## [영역 4] 스토리지 및 데이터 파이프라인 병목 (15~17)

### 15. 스토리지 I/O 병목으로 인한 GPU 굶주림 (GPU Utilization 출렁임)
* **증상:** 에러는 없지만 `nvidia-smi`에서 GPU 사용률이 `0% ↔ 100%`를 반복하며 학습 속도가 극단적으로 느림.
* **원인:** 수십만 개의 작은 파일이나 대용량 데이터를 느린 네트워크 스토리지(NFS)에서 실시간으로 읽어오느라 GPU가 데이터를 기다리는 상태(I/O Bottleneck).
* **해결책:** 학습 전 데이터셋을 각 노드의 **로컬 초고속 디스크(NVMe SSD)**로 복사하고, 자잘한 파일 대신 `Parquet`, `Arrow`, `WebDataset(.tar)` 같은 대용량 순차 읽기 포맷으로 변환한다.

### 16. 디스크 용량 부족 (`OSError: [Errno 28] No space left on device`)
* **증상:** 학습 중반 이후 체크포인트 저장 시점에 디스크 풀 에러로 프로세스 중단.
* **원인:** LLM 체크포인트는 1개당 수십 GB에 달하는데, `save_total_limit` 설정 없이 매 에폭/스텝마다 저장하다 루트(`/`)나 임시 디렉터리(`/tmp`, Hugging Face 캐시)가 가득 참.
* **해결책:** `save_total_limit=2` 또는 `3`으로 오래된 체크포인트를 자동 삭제하게 하고, `HF_HOME` 및 `TMPDIR` 환경변수를 용량이 넉넉한 대용량 볼륨 경로로 지정한다.

### 17. 손상된 데이터(Corrupted File / NaN 입력)로 인한 중도 크래시
* **증상:** 학습이 50% 이상 잘 진행되다가 항상 **특정 스텝(예: 1,420 스텝)** 부근에서 깨진 이미지·잘못된 인코딩·빈 텍스트를 만나며 에러 발생.
* **원인:** 대용량 데이터셋 일부에 깨진 파일이나 특수 제어문자, 길이가 0인 샘플이 섞여 있어 해당 배치가 순서대로 들어올 때 터짐.
* **해결책:** 학습 전 전체 데이터셋을 1회 순회하며 파일 무결성, 최소/최대 토큰 길이, `NaN`/`Null` 값을 검증하는 사전 필터링 스크립트를 반드시 먼저 돌린다.

---

## [영역 5] 하드웨어 및 드라이버·클러스터 결함 (18~20)

### 18. 특정 GPU 하드웨어 결함 (`illegal memory access` / NVIDIA Xid 에러)
* **증상:** 코드와 데이터는 동일한데 특정 노드나 특정 GPU 번호에서만 간헐적으로 메모리 접근 오류나 커널 패닉 발생.
* **원인:** GPU VRAM 물리적 불량, NVLink/PCIe 슬롯 접촉 불량, 또는 냉각 불량으로 인한 과열(Thermal Throttling / Shutdown).
* **해결책:** `nvidia-smi -q -d ECC`로 메모리 오류 카운트를 조회하고, `dmesg -T | grep -i NVRM`으로 **Xid 에러 코드**(예: Xid 79는 GPU 버스 이탈, Xid 48은 치명적 ECC 오류)를 확인해 해당 노드를 클러스터에서 격리(Drain)한다.

### 19. CUDA 드라이버와 PyTorch / FlashAttention ABI 버전 충돌
* **증상:** `ImportError: undefined symbol: ...` 또는 `CUDA error: no kernel image is available for execution on the device` 발생.
* **원인:** 호스트 머신의 NVIDIA 드라이버 버전, 컨테이너 내부 CUDA Toolkit 버전, PyTorch 빌드 버전, 그리고 `flash-attn` 컴파일 버전이 서로 어긋남.
* **해결책:** NVIDIA 공식 검증 컨테이너(`nvcr.io/nvidia/pytorch:xx.xx-py3`)를 기반으로 환경을 구성하고, `flash-attn` 등 C++/CUDA 확장 라이브러리는 현재 컨테이너의 PyTorch·CUDA 버전에 맞춰 재빌드한다.

### 20. 클러스터 선점(Preemption) 및 스팟 인스턴스 중단으로 인한 학습 유실
* **증상:** 클라우드 스팟(Spot) 인스턴스 회수나 Slurm/쿠버네티스 우선순위 스케줄링으로 학습 노드가 예기치 않게 종료되어 처음부터 다시 학습해야 함.
* **원인:** 며칠씩 걸리는 장기 학습에서 중간 체크포인트 저장 주기가 너무 길거나 재개(Resume) 로직이 구현되어 있지 않음.
* **해결책:** `save_steps` 주기를 1~2시간 간격으로 설정하고, 장애 복구 시 가장 최근 체크포인트를 자동 탐색해 이어서 학습하는 `trainer.train(resume_from_checkpoint=True)` 로직을 기본으로 적용한다.

---

## 💡 장애 발생 시 1분 만에 범인 찾는 디버깅 루틴

> [!TIP] 실전 트러블슈팅 3단계 원칙
> 1. **인프라 생존 및 로그 확인:** `nvidia-smi` (GPU 점유·온도) → `free -h` (CPU RAM) → `df -h` (디스크 잔여량) → `dmesg -T | tail -n 30` (커널 OOM 및 GPU Xid 하드웨어 에러).
> 2. **분산 통신 상세 추적:** 멀티 GPU/노드에서 멈출 때는 아래 환경변수를 켜고 재실행하여 어느 노드·Rank에서 멈췄는지 특정한다.
>    ```bash
>    export NCCL_DEBUG=INFO
>    export TORCH_DISTRIBUTED_DEBUG=DETAIL
>    export CUDA_LAUNCH_BLOCKING=1  # 정확한 에러 코드 라인 추적용 (디버깅 완료 후 해제)
>    ```
> 3. **최소 단위 격리 테스트:** 멀티 노드에서 터지면 **싱글 노드(1대)**로 줄이고, 싱글 노드에서 터지면 **싱글 GPU(1장)**로 줄여서 **코드·데이터 문제**인지 **통신·인프라 문제**인지 즉시 분리한다.
