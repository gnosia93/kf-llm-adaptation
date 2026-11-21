# 모듈 4. 계속 사전학습 — 코드 분석, 메모리 계산, 학습 실행

**소요 60분** · 강의 25분 + 실습 35분 (학습이 도는 동안 강의를 진행)

## 학습 목표

- `train_cpt_hf.py` 의 각 설정이 왜 그 값인지 설명할 수 있다.
- 모델 크기로 GPU 메모리 사용량을 계산할 수 있다.
- 학습을 백그라운드로 실행하고, loss 곡선을 읽고 이상 징후를 판단할 수 있다.

> **진행 팁:** 학습은 시간이 걸린다. **4.6 실행 절을 먼저 해서 학습을 띄워 놓고**,
> 도는 동안 4.1~4.5 를 설명하는 순서를 권한다.

---

## 4.1 학습 설정 한눈에

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

## 4.2 코드 분석: `train_cpt_hf.py`

### (1) 모델 로딩: 왜 fp32 인가

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.float32 if args.full else torch.bfloat16,
    attn_implementation="sdpa",
)
```

bf16 은 유효숫자가 약 3자리(가수부 8비트)뿐이다.

```
가중치 0.0200 에 업데이트 2e-5 x (기울기 ~1) = 0.00002 를 더하면
  fp32:  0.0200 + 0.00002 = 0.02002   ✅
  bf16:  0.0200 + 0.00002 = 0.0200    ❌ 업데이트가 반올림돼 사라짐
         (0.02 근처에서 bf16 이 표현할 수 있는 간격은 약 0.000122)
```

그래서 **가중치는 fp32 로 저장**하고, 계산(forward/backward)만 `bf16=True` 로 bf16 을 써서 빠르게 한다 (autocast).
LoRA 모드는 원래 가중치를 고정하므로 bf16 으로 로드해도 된다.

### (2) 데이터: `text` 필드 → 모든 토큰이 loss 대상

```python
ds = load_dataset("json", data_files={"train": ".../train.jsonl", "validation": ".../valid.jsonl"})

config = SFTConfig(
    dataset_text_field="text",   # 순수 텍스트 -> 모든 토큰에 loss (언어모델링)
    packing=False,               # 03_build_cpt.py 가 문서 단위로 이미 패킹함
    max_length=args.max_length,  # 8192 = 03_build_cpt.py --max-tokens
    use_liger_kernel=not args.no_liger,
    ...
)
```

TRL 의 `SFTTrainer` 는 이름과 달리 데이터 형식에 따라 동작이 바뀐다.

| 데이터 형식 | loss 대상 | 용도 |
|---|---|---|
| `{"text": ...}` | **모든 토큰** | **CPT (지금)** |
| `{"prompt": ..., "completion": ...}` | completion 만 | SFT (`train_sft_hf.py`, 모듈 6) |
| `{"messages": [...]}` | 설정에 따라 | 대화 SFT |

`packing=False` 인 이유: 모듈 3 에서 이미 문서 경계를 지키며 블록을 만들었다.
TRL 의 패킹을 켜면 서로 다른 문서가 한 시퀀스에 이어 붙는다.

`max_length` 는 **반드시 데이터의 블록 상한과 같아야 한다.** 블록보다 작으면 블록 뒷부분이 잘려 버려지고,
크면 메모리 계획이 어긋난다. 노트북은 데이터 셀의 `MAX_TOKENS` 를 학습과 평가에 그대로 넘긴다.

### (3) LoRA 는 선택

```python
peft_config = None
if not args.full:
    peft_config = LoraConfig(r=128, lora_alpha=128, lora_dropout=0.0,
                             target_modules="all-linear", task_type="CAUSAL_LM")
```

- `target_modules="all-linear"`: q, k, v, o, gate, up, down 프로젝션에 모두 적용한다. 지식은 주로 MLP 층에 저장된다고 알려져 있다.
- `modules_to_save` 를 쓰지 않는 이유: Qwen2.5-3B 는 임베딩과 `lm_head` 가 **가중치를 공유(tied)** 한다.
  둘 중 하나만 따로 학습하면 공유가 깨진다.

### (4) 최적화 설정

```python
learning_rate=args.lr or (2e-5 if args.full else 1e-4),
lr_scheduler_type="cosine",
warmup_steps=0.05,          # 1 미만이면 전체 스텝 대비 비율 (transformers 5)
optim="paged_adamw_8bit" if args.full else "adamw_torch_fused",
```

```
학습률
2e-5 ┤    ╭──╮
     │   ╱    ╲___
     │  ╱         ╲___
     │ ╱              ╲____
   0 ┼╱─────────────────────╲──▶ step
     └─5%─┘   cosine 감소
     warmup
```

- **warmup:** 처음부터 큰 학습률을 쓰면 사전학습 가중치가 크게 흔들린다. 처음 5% 동안 0에서 서서히 올린다.
- **full 은 2e-5, LoRA 는 1e-4:** LoRA 는 새로 추가한 작은 행렬만 학습하므로 더 큰 학습률이 필요하다.
- **paged:** GPU 메모리가 부족할 때 옵티마이저 상태를 CPU RAM 으로 잠시 내보낸다.

### (5) 저장

```python
final = trainer.model
if peft_config is not None:
    final = final.merge_and_unload()          # LoRA 가중치를 원래 가중치에 합침
final.to(torch.bfloat16).save_pretrained(f"{args.output}/final")
```

추론에는 bf16 이면 충분하므로 fp32(12GB) 대신 bf16(약 6GB)으로 저장한다.

## 4.3 메모리 계산

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

## 4.4 스텝과 epoch 계산

```
train 샘플 726개 / 실효 배치 8   =  91 스텝 / epoch
91 스텝 x 4 epoch                 =  약 364 스텝
eval 25 스텝마다 → 약 14번,  체크포인트 50 스텝마다 (최근 2개만 유지)
스텝당 토큰 ≈ 8 x 2,070 (replay 포함 평균 길이) ≈ 1.7만 토큰  (스텝마다 들쭉날쭉)
```

**소요 시간 추정 방법:** 진행 막대에 표시되는 `s/it` (스텝당 초) x 364.
배치 1 이라 스텝마다 토큰 수가 달라서 `s/it` 도 들쭉날쭉하다. 몇십 스텝 평균을 본다.
L40S 기준 예상은 수십 분대다. 정확한 값은 강사가 리허설로 측정해서 공유한다.

## 4.5 Mac(MLX) 설정과 비교

| | g6e (`train_cpt_hf.py --full`) | Mac (`config_cpt.yaml`) |
|---|---|---|
| 방식 | full | LoRA r=128, 모든 레이어 |
| 학습 파라미터 | 30.9억 | 2.39억 (7.8%) |
| 블록 길이 | 8,192 | 2,048 (`--max-tokens 2048` 으로 만든 데이터) |
| 배치 | 1 x 8 | 2 x 4 |
| 학습률 | 2e-5 | 5e-5 (scale 20) |
| iters 단위 | 옵티마이저 스텝 | **마이크로배치** (mlx-lm 특징) |
| 속도 | (리허설 측정) | 약 190 tok/s (M4 Pro 실측) |

> mlx-lm 의 `iters` 는 마이크로배치를 센다. 그래서 `grad_accumulation_steps: 4` 이면
> 옵티마이저 스텝은 `iters / 4` 이고, 학습률 스케줄도 그 기준으로 맞췄다.

---

## 4.6 실습: 학습 실행

노트북 **`2. 학습 (백그라운드)`** 섹션.

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

### 왜 노트북 커널에서 직접 학습하지 않나

| 노트북 커널에서 `trainer.train()` | `nohup` 백그라운드 프로세스 |
|---|---|
| VS Code 연결이 끊기면 커널과 함께 죽을 수 있음 | 연결과 상관없이 계속 실행 |
| 학습이 끝날 때까지 다른 셀을 실행 못 함 | 로그·그래프 셀을 언제든 실행 |
| 학습 후에도 커널이 GPU 메모리를 잡고 있음 | 끝나면 GPU 메모리를 바로 반납 |

### 진행 확인

```python
!tail -c 1500 logs/train.log
!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

원격 터미널에서 실시간으로 볼 수도 있다:

```bash
tail -f ~/llm-ft/logs/train.log         # Ctrl-C 로 종료 (학습은 계속됨)
watch -n 2 nvidia-smi
```

로그 예시 (형식만 참고, 숫자는 예시):

```
{'loss': 1.62, 'grad_norm': 3.1, 'learning_rate': 1.3e-05, 'epoch': 0.07}
{'loss': 1.48, 'grad_norm': 2.4, 'learning_rate': 2.0e-05, 'epoch': 0.13}
{'eval_loss': 1.45, 'eval_runtime': 6.2, 'epoch': 0.33}
 14%|█▍        | 50/364 [05:12<32:40,  6.24s/it]
```

### TensorBoard (선택)

```bash
source ~/llm-ft/.venv/bin/activate
tensorboard --logdir ~/llm-ft/models/k8s-ko-cpt-hf/runs --port 6006
```

VS Code 의 **PORTS** 탭에서 6006 을 포워딩한 뒤 로컬 브라우저에서 `http://localhost:6006` 을 연다.

## 4.7 loss 곡선 읽기

노트북의 loss 그래프 셀을 실행한다. 체크포인트가 저장된 뒤(50 스텝 이후)부터 그려진다.

| 모양 | 해석 | 조치 |
|---|---|---|
| train ↓, valid ↓ 또는 유지 | ✅ 정상 | 그대로 |
| train ↓↓, valid ↑ (뒤쪽 epoch) | 과적합. 학습한 블록을 외우는 중 | 더 이른 체크포인트 사용, epoch 줄이기 |
| epoch 경계마다 train 이 계단식으로 뚝 떨어짐 | 같은 데이터를 다시 보면서 외우는 신호 | 지식 주입에서는 어느 정도 정상. valid 를 함께 봐야 함 |
| loss 가 갑자기 튀거나 `nan` | 학습률이 너무 큼 | `--lr 1e-5` |
| loss 가 거의 안 내려감 | 학습률이 너무 작거나 데이터 문제 | 데이터 샘플 확인, lr 올리기 |

> **CPT 에서 valid loss 의 의미:** valid 는 학습하지 않은 쿠버네티스 문서다. 이 loss 가 내려간다는 것은
> 새 문서에도 통하는 **도메인 감각**(용어, 문체, YAML 구조)이 생겼다는 뜻이다.
> 반면 **특정 사실의 습득**은 `k8s_seen` perplexity 로 따로 본다 (모듈 5).

---

## 과제

### 과제 4-1. 메모리 예측과 실측 비교

`nvidia-smi` 의 `memory.used` 를 기록하고 4.3 의 예측(34~37 GB)과 비교하자.
차이가 난다면 원인은 무엇일까? (힌트: PyTorch 캐시 할당자, CUDA 컨텍스트)

### 과제 4-2. 완료 시간 예측

진행 막대의 `s/it` 로 전체 학습 시간을 계산하자. 1 epoch 만 돌렸다면 얼마였을까?

### 과제 4-3. 계산해 보기 (토론)

1. 7B 모델을 같은 방식(fp32 가중치 + 8bit Adam)으로 full 학습하면 메모리가 얼마나 필요한가? L40S 에 들어가나?
2. 들어가지 않는다면 어떤 선택지가 있을까? (LoRA, QLoRA, DeepSpeed ZeRO-3 / FSDP + 여러 GPU, CPU offload)
3. 이번 데이터에 영어 원문을 추가하면 토큰이 약 3~4배가 된다. 학습 시간과 비용은 얼마나 될까?

### 과제 4-4 (도전). LoRA 와 비교

full 학습이 끝난 뒤 `MODE = ""` 로 LoRA 학습을 돌려 본다. 결과가 덮어써지지 않게 `EXTRA = "--output models/k8s-ko-cpt-lora"` 를 준다.
모듈 5 에서 두 결과를 비교한다. 평가 셀의 `TUNED` 를 `models/k8s-ko-cpt-lora/final` 로,
loss 그래프 셀의 경로를 `models/k8s-ko-cpt-lora/checkpoint-*` 로 바꾸면 된다.

---
← [모듈 3](03_data.md) · 다음: [모듈 5. 평가 →](05_evaluation.md)
