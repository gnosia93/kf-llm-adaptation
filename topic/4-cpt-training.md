## 연속 사전학습(Continued Pre-Training / CPT) ##


### 1. 학습 설정 ###

| 설정 | 값 | 이유 |
|---|---|---|
| 모델 | `Qwen/Qwen2.5-3B` (base) | 문서 이어쓰기를 배우는 데 적합 (모듈 1.3) |
| 방식 | **full fine-tuning** | 지식 주입 효과가 가장 크고, L40S 48GB 에 들어감 |
| 가중치 정밀도 | fp32 마스터 + bf16 autocast | 작은 업데이트가 반올림돼 사라지지 않게 |
| 옵티마이저 | `paged_adamw_8bit` | Adam 상태를 8bit 로 → 메모리 18GB 절약 |
| 시퀀스 길이 | **8,192** | 데이터를 이 길이에 맞춰 패킹함. 문서의 96% 가 블록 하나에 들어감 (모듈 3) |
| 배치 | **1** x 누적 8 = 실효 8 | 배치 1 이라 padding 이 0%. 한 스텝에 약 1.7만 토큰 |
| Liger 커널 | 켬 (`use_liger_kernel=True`) | 로짓 전체를 만들지 않고 loss 계산 → 8K 에서 로짓 메모리 10GB 이상 절약 |
| 학습률 | **2e-5**, cosine, warmup 5% | 사전학습된 지식을 크게 망가뜨리지 않는 범위 |
| epoch | 4 | 지식은 여러 번 봐야 남는다 (모듈 1.7) |
| gradient checkpointing | 켬 | 활성값 메모리 절약 (대신 약 30% 느려짐) |
| eval / save | 25 / 50 스텝마다 | 학습 중 held-out loss 를 추적 |


### 2. 메모리 계산 ###

파라미터 수 **P = 30.9억** (3.09B).

| 항목 | 계산 | full + 8bit Adam + Liger (이번 설정) | 참고: bf16 + 일반 AdamW, Liger 없음 |
|---|---|---|---|
| 가중치 | P x 4 byte (fp32) | **12.3 GB** | 6.2 GB (bf16) |
| 기울기 | P x 4 byte | **12.3 GB** | 6.2 GB |
| Adam 상태 (m, v) | P x 2 x 1 byte (8bit) | **6.2 GB** | 24.7 GB (fp32 x 2) |
| 활성값 (hidden, checkpointing) | 36층 x 8,192 토큰 x 2,048 x 2 byte | 약 1.2 GB | 약 1.2 GB |
| 활성값 (레이어 재계산, attention 등) | 한 층 분량 | 약 2~4 GB | 약 2~4 GB |
| 로짓 | 8,192 토큰 x 151,936 x (bf16 + fp32 사본 + 기울기) | **Liger: 거의 0** | 약 10~18 GB |
| **합계** | | **약 34~37 GB** ✅ L40S 48GB | 50GB 이상 ❌ |

**로짓(logits)** 은 출력 헤드가 계산한 결과 텐서다. 토큰마다 어휘 전체(151,936개)에 대한 점수가 있어서, hidden state(2,048개)의 74배다.
gradient checkpointing 을 켜면 레이어 활성값은 작아지지만 로짓은 줄지 않는다. 그래서 8K 에서는 로짓이 가장 큰 활성값이 된다.

**Liger 커널**(`liger-kernel` 패키지)의 fused linear cross-entropy 는 토큰을 작은 묶음으로 나눠서
"출력 헤드 계산 → loss → 기울기"를 바로 처리하고 버린다. 그래서 로짓 전체를 메모리에 만들지 않는다.

> 위 메모리는 **계산으로 추정한 값**이다. 리허설에서 `nvidia-smi` 로 실측해서 확인한다.
>
> **OOM 이 나면:** 노트북 데이터 셀에서 `MAX_TOKENS = 4096` 으로 데이터를 다시 만들고 `EXTRA = "--grad-accum 16"` 으로 학습한다.
> 마이크로배치 토큰이 절반이 되어 메모리가 줄고, 실효 배치 토큰 수는 비슷하게 유지된다.

### 3. 스텝과 epoch 계산 ###

```
train 샘플 726개 / 실효 배치 8   =  91 스텝 / epoch
91 스텝 x 4 epoch                 =  약 364 스텝
eval 25 스텝마다 → 약 14번,  체크포인트 50 스텝마다 (최근 2개만 유지)
스텝당 토큰 ≈ 8 x 2,070 (replay 포함 평균 길이) ≈ 1.7만 토큰  (스텝마다 들쭉날쭉)
```

**소요 시간 추정 방법:** 진행 막대에 표시되는 `s/it` (스텝당 초) x 364.
배치 1 이라 스텝마다 토큰 수가 달라서 `s/it` 도 들쭉날쭉하다. 몇십 스텝 평균을 본다.
L40S 기준 예상은 수십 분대다. 정확한 값은 강사가 리허설로 측정해서 공유한다.


### 4. 학습 실행 ###

* https://github.com/gnosia93/llm-train-ft/blob/main/src/train_cpt_hf.py
  
```python
MODE = "--full"
EPOCHS = 4
EXTRA = ""
...
subprocess.run(
    f"nohup .venv/bin/python train_cpt_hf.py {MODE} --epochs {EPOCHS} --max-length {MAX_TOKENS} {EXTRA} > logs/train.log 2>&1 &",
    shell=True,
)
```

* 진행 확인

```python
!tail -c 1500 logs/train.log
!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

* 원격 터미널에서 실시간으로 볼 수도 있다:

```bash
tail -f ~/llm-ft/logs/train.log         # Ctrl-C 로 종료 (학습은 계속됨)
watch -n 2 nvidia-smi
```

* 로그 예시 (형식만 참고, 숫자는 예시):

```
{'loss': 1.62, 'grad_norm': 3.1, 'learning_rate': 1.3e-05, 'epoch': 0.07}
{'loss': 1.48, 'grad_norm': 2.4, 'learning_rate': 2.0e-05, 'epoch': 0.13}
{'eval_loss': 1.45, 'eval_runtime': 6.2, 'epoch': 0.33}
 14%|█▍        | 50/364 [05:12<32:40,  6.24s/it]
```

### 5. 평가 (evaluation) ###

* https://github.com/gnosia93/llm-train-ft/blob/main/src/eval_cpt.py

#### 5.1 perplexity 및 Probe 문장 생성 ####
```
                 원본 Qwen2.5-3B (BASE)      CPT 모델
                 ─────────────────────      ───────────────────
k8s_seen     ─▶  ppl_b(seen)                ppl_c(seen)        → 많이 ↓ 해야 함
k8s_heldout  ─▶  ppl_b(heldout)             ppl_c(heldout)     → 조금 ↓
general_ko   ─▶  ppl_b(general)             ppl_c(general)     → 거의 유지 (오르면 망각)

PROBES (6개)  ─▶  BASE 이어쓰기              CPT 이어쓰기        → 나란히 비교
```

#### 5.2 환각 검증 ####

학습한 문서에 있는 내용과 **없는 내용**으로 프로브를 2개씩 만들어 보자.

```python
my_probes = [
    "# 잡\n\n잡(Job)은",                          # 문서에 있는 주제
    "# 컨피그맵\n\n컨피그맵(ConfigMap)은",
    "쿠버네티스 1.40 버전에서 새로 추가된 기능은",        # 문서에 없는 미래 정보 → 환각이 나오는지?
    "아이스 아메리카노를 맛있게 만드는 방법은",            # 도메인 밖 → 일반 능력이 유지되나?
]
for label, path in [("BASE", BASE), ("CPT", TUNED)]:
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for p in my_probes:
        print(f"[{label}] {p}{complete(model, p)}\n")
    del model; torch.cuda.empty_cache()
```


