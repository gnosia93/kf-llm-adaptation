# MFU (Model FLOPs Utilization) 개념, 병목 진단 및 프로파일링 가이드

---

## 1. MFU란 무엇이며 왜 중요한가?

### ① 개념 정의
**MFU(Model FLOPs Utilization)**는 **하드웨어(GPU)가 이론적으로 낼 수 있는 최대 연산 성능(Peak FLOPs) 대비, 실제 모델 학습(순전파+역전파)에 유효하게 쓰인 연산량의 비율(%)**을 의미한다.

\[\text{MFU (\%)} = \frac{\text{실제 달성한 모델 연산량 (Achieved TFLOPs/s)}}{\text{GPU 스펙상 이론 최대 연산량 (Peak TFLOPs/s)}} \times 100\]

* **이론 최대치 기준:** H100 SXM5(BF16 Dense) 기준 **`989 TFLOPs`**, A100 SXM(BF16) 기준 **`312 TFLOPs`**를 분모로 삼는다. *(희소성 Sparsity 스펙은 제외)*
* **현업 표준 구간:** 잘 튜닝된 멀티 노드 분산 학습의 MFU는 보통 **`45% ~ 55%`** 수준이며, 네트워크나 스토리지 병목이 있으면 **`10% ~ 20%`대**로 추락한다.

### ② 왜 Account SA와 고객에게 결정적으로 중요한가?
1. **`nvidia-smi` 지표의 착시 해결:** 터미널에서 `nvidia-smi`를 쳤을 때 `GPU-Util 100%`가 찍혀도, 실제로는 GPU 코어가 네트워크 통신이나 메모리 대기를 하느라 공회전하고 있을 수 있다. **MFU만이 "비싼 GPU가 진짜 모델 학습에 얼마나 알차게 일했는가"를 보여주는 유일한 진실 지표**다.
2. **직접적인 TCO(총소유비용) 및 출시 일정(Time-to-Train) 결정:** MFU가 `15%`에서 `50%`로 오르면 동일한 클러스터에서 학습 속도가 **3.3배 빨라지고, 100만 토큰당 학습 비용이 약 70% 절감**된다.
3. **Neocloud 인프라 가치 입증:** 고객에게 왜 일반 이더넷(TCP/IP)과 NFS 대신 **InfiniBand와 초고속 병렬 스토리지**가 필요한지 재무적 ROI로 설득하는 가장 강력한 무기다.

---

## 2. MFU 저하를 일으키는 4대 핵심 병목 지점

학습 루프에서 GPU 연산 시간 외에 시간을 잡아먹는 병목은 크게 네 곳에서 발생한다.

| 병목 영역 | 구체적 발생 원인 | MFU 저하 메커니즘 | 해결 아키텍처 |
| :--- | :--- | :--- | :--- |
| **1. 노드 간 통신 병목**<br>*(Communication)* | • RDMA 대신 일반 **TCP/IP 소켓** 통신<br>• **4-byte(FP32)** 그래디언트 전송<br>• 연산과 통신의 미중첩(No Overlap) | 역전파 끝날 때마다 수백 GB의 그래디언트를 주고받느라(`AllReduce`) GPU 코어가 멍하니 대기함 | • **InfiniBand (400G RDMA)**<br>• **BF16(2-byte)** 그래디언트 전환<br>• NCCL 튜닝 및 통신 중첩 |
| **2. 스토리지 I/O 병목**<br>*(Data & Checkpoint)* | • 느린 **공유 NFS**에서 데이터셋 로딩<br>• ~1TB급 학습 체크포인트를 동기 방식으로 디스크에 쓰기 | 데이터 로딩 지연(I/O Starvation) 및 체크포인트 저장 중 전체 클러스터 수십 분간 정지 | • **로컬 NVMe SSD** 데이터 캐싱<br>• **Lustre/Weka** 병렬 파일시스템<br>• 비동기 체크포인트 저장 |
| **3. 메모리 대역폭 & 커널**<br>*(Memory-Bound)* | • 너무 작은 마이크로 배치 크기<br>• FlashAttention 미적용<br>• 비효율적인 파이썬/작은 커널 호출 | 연산(FLOPs)보다 VRAM에서 데이터를 읽고 쓰는 시간(Memory Bandwidth)이 더 오래 걸림 | • **FlashAttention-2/3** 적용<br>• 커널 퓨전(`torch.compile`)<br>• 최적 마이크로 배치 확보 |
| **4. 파이프라인 & 낙오자**<br>*(Bubbles & Stragglers)* | • 파이프라인 병렬화(PP)의 유휴 구간(Bubble)<br>• 특정 노드의 과열·하드웨어 저하(Straggler)<br>• CPU 토크나이징 병목 | 가장 느린 GPU 1장이 끝날 때까지 나머지 수십 장의 GPU가 동기화 지점(`Barrier`)에서 대기 | • 3D 병렬화(TP/PP/DP) 비율 최적화<br>• **DCGM** 기반 불량 노드 격리<br>• 데이터 사전 토크나이징 |

---

## 3. 정확한 Step Time 계측 및 병목 프로파일링 코드

### ① CUDA 이벤트 기반 정밀 Step Time 측정 (`torch.cuda.Event`)
파이썬 `time.time()`은 CPU-GPU 비동기 실행 때문에 부정확하므로 반드시 CUDA 하드웨어 타이머로 측정한다.

```python
import torch

def measure_accurate_step_time(train_step_fn, batch, warmup_steps=3, measure_steps=5) -> float:
    """CUDA 비동기 실행 오차를 제거하여 순수 1 스텝 평균 소요 시간(초)을 측정한다."""
    for _ in range(warmup_steps):
        train_step_fn(batch)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(measure_steps):
        train_step_fn(batch)
    end_event.record()
    torch.cuda.synchronize()

    return (start_event.elapsed_time(end_event) / 1000.0) / measure_steps
```

### ② PyTorch 내장 프로파일러 (`torch.profiler`)로 병목 커널 추적
2~3스텝만 캡처하여 **연산 커널(GEMM) vs NCCL 통신 커널(`ncclKernel`) vs CPU DataLoader 대기 시간** 비율을 확인한다.

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

def run_with_pytorch_profiler(model, dataloader, optimizer, trace_dir="./log/profiler_trace"):
    prof_schedule = schedule(wait=1, warmup=1, active=2, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for step, batch in enumerate(dataloader):
            if step >= 4:
                break
            with record_function("## 1. Data_to_GPU ##"):
                inputs = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with record_function("## 2. Forward_Pass ##"):
                loss = model(**inputs).loss
            with record_function("## 3. Backward_and_AllReduce ##"):
                loss.backward()
            with record_function("## 4. Optimizer_Step ##"):
                optimizer.step()
                optimizer.zero_grad()
            prof.step()

    # ncclKernel 비중이 상위권이면 네트워크 병목, GPU 타임라인이 비어있으면 스토리지/CPU 병목
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### ③ NVIDIA Nsight Systems (`nsys`) CLI 프로파일링
코드 수정 없이 실행 명령어 앞단에 붙여 InfiniBand RDMA 및 NVMe 하드웨어 레벨 타임라인을 캡처한다.

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi \
  --output=70b_multinode_rank0 \
  torchrun --nproc_per_node=8 --nnodes=8 train.py
```

---

## 4. 학습 로그 및 실측 Step Time 기반 MFU 산출 코드 (`calc_mfu.py`)

```python
def calculate_mfu(
    num_params_b: float,           # 모델 파라미터 수 (단위: Billion, 예: 70B -> 70.0)
    num_gpus: int,                 # 전체 사용 GPU 수 (예: 8노드 * 8장 = 64)
    gpu_peak_tflops: float,        # GPU 이론 최대 연산량 (H100 BF16: 989.0, A100 BF16: 312.0)
    global_batch_size: int = None, # 전체 배치 크기 (samples/step)
    seq_length: int = None,        # 시퀀스 길이 (tokens/sample)
    step_time_sec: float = None,   # 실측된 1 스텝 소요 시간 (초)
    samples_per_sec: float = None, # 로그의 초당 처리 샘플 수 (선택 입력)
    logged_tflops: float = None,   # Megatron/DeepSpeed 로그의 GPU당 TFLOPs/s (선택 입력)
) -> dict:
    if logged_tflops is not None:
        achieved_tflops_per_gpu = logged_tflops
    else:
        if step_time_sec is None:
            if samples_per_sec is None or global_batch_size is None:
                raise ValueError("step_time_sec 또는 (samples_per_sec + global_batch_size)가 필요합니다.")
            step_time_sec = global_batch_size / samples_per_sec

        # 표준 Transformer MFU 공식: 토큰당 순수 모델 연산량 = 6 * P
        total_tokens_per_step = global_batch_size * seq_length
        total_flops_per_step = 6 * (num_params_b * 1e9) * total_tokens_per_step
        achieved_tflops_per_gpu = total_flops_per_step / (step_time_sec * num_gpus * 1e12)

    mfu_percent = (achieved_tflops_per_gpu / gpu_peak_tflops) * 100

    return {
        "achieved_tflops_per_gpu": round(achieved_tflops_per_gpu, 2),
        "peak_tflops_per_gpu": gpu_peak_tflops,
        "mfu_percent": round(mfu_percent, 2),
    }

# [As-Is vs To-Be 비교]
as_is = calculate_mfu(num_params_b=70.0, num_gpus=64, gpu_peak_tflops=312.0,
                      global_batch_size=128, seq_length=4096, step_time_sec=68.5)
to_be = calculate_mfu(num_params_b=70.0, num_gpus=64, gpu_peak_tflops=989.0,
                      global_batch_size=128, seq_length=4096, step_time_sec=6.9)

print(f"As-Is 온프레미스 A100 MFU: {as_is['mfu_percent']}% ({as_is['achieved_tflops_per_gpu']} TFLOPs/GPU)")
print(f"To-Be Neocloud H100 MFU: {to_be['mfu_percent']}% ({to_be['achieved_tflops_per_gpu']} TFLOPs/GPU)")
```
