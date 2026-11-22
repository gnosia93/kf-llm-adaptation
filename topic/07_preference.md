# 모듈 7. 선호 학습: DPO 와 PPO (RLHF)

**소요 80분** · 강의 30분 + 실습 50분 (학습은 백그라운드)

## 학습 목표

- SFT 만으로는 고치기 어려운 문제와, **선호(preference) 학습**이 무엇을 더 가르치는지 설명할 수 있다.
- on-policy 선호 데이터를 만든다: SFT 모델이 쓴 답 여러 개 → 심판 LLM 채점 → 좋은 답 / 나쁜 답 쌍.
- **DPO** 의 loss 와 β 의 역할을 이해하고, 학습 로그(margin, accuracy)를 읽는다.
- **PPO (RLHF)** 의 구성 요소(정책, 참조, 가치, 보상 모델)와 한 스텝의 흐름을 코드로 따라간다.
- 보상 해킹(reward hacking)을 지표로 알아챈다.

---

## 7.1 SFT 다음에 남은 문제

SFT 는 **정답을 따라 쓰는 것**만 배운다. "이 답이 저 답보다 낫다" 는 신호는 없다.
모듈 6 의 SFT 모델에서 흔히 보이는 문제:

| 증상 | 예 | SFT 로 잘 안 고쳐지는 이유 |
|---|---|---|
| 그럴듯한 환각 | 없는 `kubectl` 플래그, 틀린 기본값 | 정답 데이터에는 "틀린 답" 이 없다. 무엇을 하면 안 되는지 모른다 |
| 문서 덤프 | 질문과 상관없는 섹션을 길게 옮겨 적음 | section 원문 답변을 따라 하도록 배웠다 (모듈 6.3) |
| 반복·끊김 | 같은 문장 반복, 중간에 끊김 | 정답 하나만 보면 이런 실패를 피하는 법을 배우지 못한다 |

**선호 학습**은 같은 질문에 대한 답 두 개를 보여주고 "이쪽이 낫다" 를 가르친다.
모델이 **실제로 저지르는 실수**를 rejected 로 보여주면, 그 실수의 확률을 직접 낮춘다.

```
            ┌────────── A. DPO: 선호 쌍으로 바로 정책 학습 (분류 문제처럼) ──────────┐
SFT 모델 ───┤                                                                     ├──▶ 선호에 맞춘 모델
            └── B. RLHF: 선호 쌍 → 보상 모델 학습 → PPO 로 보상을 높이도록 강화학습 ─┘
```

B 가 InstructGPT·초기 ChatGPT 의 방식이고, A 는 B 와 같은 목표를 강화학습 없이 푸는 방법이다 (Rafailov 외, 2023).

## 7.2 선호 데이터 만들기

### (1) 파이프라인

```
data/sft/train.jsonl 의 질문 1,000개 (k8s 80% + 일반 20%)
        │  ① SFT 모델로 질문마다 답 4개 샘플링 (temperature 0.9)        ← on-policy
        ▼
data/pref/candidates.jsonl
        │  ② 심판 LLM (Qwen2.5-7B-Instruct) 이 기준 답변을 보고 4개를 1~10점 채점
        ▼
        │  ③ 최고점 = chosen, 최저점 = rejected  (점수 차 2점 이상만)
        ▼
data/pref/{train,valid}.jsonl   {"prompt": [...], "chosen": [...], "rejected": [...]}
        └─▶ DPO (train_dpo_hf.py) 와 보상 모델 (train_rm_hf.py) 이 같은 파일을 쓴다
```

### (2) 설계할 때 고려한 점

- **on-policy:** 다른 모델의 답이 아니라 **지금 학습할 SFT 모델의 답**으로 쌍을 만든다. 모델이 실제로 하는 실수를 고친다.
  남의 답(예: 큰 모델의 답 vs 작은 모델의 답)으로 만들면 "큰 모델 말투" 를 배우는 데 그칠 수 있다.
- **기준 답변을 심판에게 준다:** 7B 심판도 쿠버네티스 세부 사실은 틀릴 수 있다. SFT 정답(공식 문서에서 온 것)을
  `<기준 답변>` 으로 함께 줘서 사실 판단의 근거로 쓰게 한다 (`08_build_pref.py` 의 `JUDGE_PROMPT`).
- **위치 편향:** LLM 심판은 앞에 나온 답에 점수를 더 주는 경향이 있다. 답의 순서를 섞어서 보여주고 점수를 원래 순서로 되돌린다.
- **점수 차 기준(`--min-margin`):** 비슷한 점수의 쌍은 잡음이다. 차이가 뚜렷한 쌍만 쓴다.
- **길이 편향 확인:** 심판은 긴 답을 선호하기 쉽다. 결과 출력의 `평균 길이 chosen / rejected` 를 꼭 본다.
  chosen 이 훨씬 길면 모델은 "길게 쓰면 좋다" 를 배운다.
- **심판 없이 할 때(`--judge heuristic`):** 기준 답변과의 글자 겹침 + 종료·반복 감점으로 점수를 매긴다. 동작 확인용 대용품이다.
  겹침 점수는 "문서를 많이 베낄수록 좋다" 를 가르칠 수 있다.

> **라이선스:** 심판 모델의 출력을 학습 신호로 쓴다. Qwen2.5-7B-Instruct 는 Apache-2.0 이다.
> 상용 API 를 심판으로 쓸 때는 출력으로 모델을 학습시키는 것이 약관상 허용되는지 확인한다.

### (3) 실습

SFT 모델(`models/k8s-ko-sft-hf/final`) 이 있어야 한다. 노트북 **5. 선호 학습** 의 첫 셀, 또는 터미널에서:

```bash
# 터미널 1: 심판 (vLLM, 모듈 8.4 의 별도 가상환경). SFT 모델과 GPU 를 나눠 쓰도록 메모리를 55% 로 제한
source .venv-vllm/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.55

# 터미널 2
cd ~/llm-ft && source .venv/bin/activate
python 08_build_pref.py                     # generate → judge → pairs
# 끝나면 터미널 1 의 vLLM 을 Ctrl+C 로 끈다 (학습에 GPU 필요)
```

출력 예시 (형식만. **3B 실측값은 아직 없다**):

```
[generate] 질문 1000개 x 답변 4개  (policy=models/k8s-ko-sft-hf/final)
  -> data/pref/candidates.jsonl  (끝까지 생성된 답 ..%)
[judge] api  (1000개)
[pairs] ...쌍 (train ... / valid ...), 제외: {'점수 차 부족': ...}
  평균 점수 chosen ... / rejected ..., 평균 길이 chosen ...자 / rejected ...자
```

✅ **확인:** chosen / rejected 쌍을 몇 개 직접 읽어 보자. **심판의 판단에 동의하는가?**

```python
import json, random
pairs = [json.loads(l) for l in open("data/pref/train.jsonl", encoding="utf-8")]
for p in random.sample(pairs, 3):
    print("Q:", p["prompt"][1]["content"])
    print(f"\n[chosen {p['score_chosen']}]\n{p['chosen'][0]['content'][:500]}")
    print(f"\n[rejected {p['score_rejected']}]\n{p['rejected'][0]['content'][:500]}\n" + "=" * 60)
```

심판 기준을 바꾸고 싶으면 `JUDGE_PROMPT` 를 고친 뒤 `python 08_build_pref.py --stage judge` 로 **채점만 다시** 한다 (후보 생성은 재사용).

## 7.3 경로 A — DPO (Direct Preference Optimization)

### (1) 아이디어

RLHF 의 목표는 "보상은 높이되, 참조 모델(SFT)에서 너무 멀어지지 않기" 다. 이 목표의 최적해를 풀어 보면
**정책 자체가 보상 모델 역할을 한다**는 결론이 나온다. 그래서 보상 모델과 강화학습 없이 선호 쌍으로 바로 학습할 수 있다.

```
암묵적 보상   r(y) = β · [ log π(y|x) − log π_ref(y|x) ]      π_ref = 학습 시작 시점의 SFT 모델 (고정)
DPO loss      −log σ( r(chosen) − r(rejected) )
```

- chosen 의 확률은 참조 대비 **올리고**, rejected 의 확률은 참조 대비 **내리도록** 학습한다.
- **β** 는 참조 모델에 묶어 두는 강도다. 크면(0.3~0.5) 보수적, 작으면(0.01~0.05) 선호를 강하게 따르지만 망가지기 쉽다. 기본 0.1.
- 참조 모델이 필요하지만, LoRA 를 쓰면 **어댑터를 끈 같은 모델**이 참조 모델이다. 메모리가 추가로 들지 않는다.

### (2) 코드 (`train_dpo_hf.py`)

```python
config = DPOConfig(beta=args.beta, max_length=1536, learning_rate=5e-6, num_train_epochs=2, ...)
trainer = DPOTrainer(model=model, ref_model=None,             # peft 를 쓰면 어댑터 끈 모델이 참조
                     train_dataset=ds["train"], eval_dataset=ds["validation"],
                     processing_class=tokenizer,
                     peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM"))
```

학습률이 SFT(LoRA 1e-4) 보다 **20배 작다.** 선호 학습은 신호가 강해서 조금만 커도 모델이 망가진다.

```bash
nohup python train_dpo_hf.py > logs/train_dpo.log 2>&1 &
```

### (3) 로그 읽는 법

| 값 | 의미 | 정상 |
|---|---|---|
| `rewards/margins` | r(chosen) − r(rejected) 평균 | 0 에서 시작해 점점 커짐 |
| `rewards/accuracies` | r(chosen) > r(rejected) 인 비율 | 0.5 에서 시작해 0.7~0.9 |
| `rewards/chosen` | chosen 의 암묵적 보상 | 약간 음수까지는 흔함. 크게 음수면 chosen 확률까지 같이 떨어지는 중 |
| `eval_loss` | valid 쌍의 DPO loss | 0.693(= log 2) 에서 내려감. 다시 오르면 과적합 |

> **흔한 현상:** DPO 는 margin 을 키우기 위해 chosen 과 rejected 의 확률을 **둘 다** 낮추기도 한다.
> `rewards/chosen` 이 계속 크게 떨어지면 학습률을 낮추거나 β 를 올리거나 epoch 를 줄인다.

## 7.4 경로 B — RLHF: 보상 모델 + PPO

### (1) 보상 모델 (`train_rm_hf.py`)

(질문, 답변) → **점수 하나**를 내는 모델이다. SFT 모델 위에 `Linear(2048 → 1)` 점수 헤드를 얹고 같은 선호 쌍으로 학습한다.

```
loss = −log σ( RM(x, chosen) − RM(x, rejected) )        (Bradley-Terry 모델)
```

```bash
nohup python train_rm_hf.py > logs/train_rm.log 2>&1 &
```

확인할 값은 **`eval_accuracy`** (valid 쌍에서 chosen 점수가 더 높은 비율) 다. 0.5 면 동전 던지기고, 0.7 이상이면 PPO 에 쓸 만하다.
**보상 모델이 틀리는 만큼 PPO 는 그 틀린 방향으로 정책을 민다.** RM 품질이 PPO 결과의 상한이다.

### (2) PPO 의 구성

TRL 1.x 에서는 `PPOTrainer` 가 빠졌다 (TRL 1.18.2 에서 확인). 그래서 핵심 루프를 `train_ppo.py` 에 직접 구현했다 (약 300줄).

| 역할 | 모델 | 학습 | 메모리 |
|---|---|---|---|
| 정책 π | SFT 모델 + LoRA | ✓ | bf16 6GB |
| 참조 π_ref | 같은 모델, **LoRA 끈 상태** | ✗ | 0 (공유) |
| 가치 V | 정책의 마지막 hidden state 위 `Linear(2048→1)` | ✓ | 무시할 수준 |
| 보상 RM | `train_rm_hf.py` 결과 | ✗ | bf16 6GB |

원래 PPO 는 모델 4개(정책, 참조, 가치, 보상)를 따로 올린다. 3B 를 L40S 1장에 올리기 위해 정책·참조·가치가 가중치를 공유한다.

### (3) 한 스텝의 흐름

```
① rollout    질문 32개 → 정책이 답변 샘플링 (temperature 1)
② 점수        RM(질문, 답변) → 답변마다 점수 하나. 끝나지 않은 답은 −1
③ 토큰 보상    r_t = −β·(log π − log π_ref)  (모든 토큰)     마지막 토큰에만 + RM 점수
④ GAE        "이 토큰을 낸 것이 평균보다 얼마나 좋았나" = advantage A_t
⑤ 업데이트    같은 rollout 으로 2 epoch:
              ratio = π_new / π_old
              정책 loss = −min(ratio·A, clip(ratio, 0.8, 1.2)·A)      ← 한 번에 너무 많이 바뀌지 않게
              가치 loss = ½(V − return)²
```

핵심 코드 (`train_ppo.py` 에서 발췌):

```python
# ③ 토큰 보상: KL 벌점 + 마지막 토큰에 RM 점수
kl = (logp - ref_logp) * mask
rewards = -a.kl_coef * kl
rewards[torch.arange(B), last] += scores

# ④ GAE (γ=1, λ=0.95): 뒤에서부터 누적
for t in reversed(range(R)):
    next_v = values[:, t + 1] if t + 1 < R else 0
    delta = rewards[:, t] + a.gamma * next_v - values[:, t]
    lastgae = delta + a.gamma * a.lam * lastgae
    adv[:, t] = lastgae

# ⑤ clipped 정책 loss
ratio = torch.exp(logp - b["old_logp"][i])
pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - a.cliprange, 1 + a.cliprange))
```

**참조 모델 계산** 은 `with self.policy.disable_adapter():` 한 줄이다. LoRA 를 끄면 학습 전 SFT 모델과 정확히 같다.

### (4) 실행과 로그

```bash
nohup python train_ppo.py > logs/train_ppo.log 2>&1 &      # 100 스텝 x 질문 32개
tail -f logs/train_ppo.log
```

```
step   10  score +0.412  kl 1.873  len 143  eos 97%  clipfrac 0.14  vf 0.083  41.2s
```

(위 줄은 로그 형식 예시이며 3B 실측값이 아니다. 스텝당 시간도 **아직 측정하지 않았다**.)

| 값 | 의미 | 정상 / 위험 신호 |
|---|---|---|
| `score` | RM 점수 평균 | 천천히 오름 |
| `kl` | 답변 하나당 Σ(log π − log π_ref) | 천천히 오름. **수십 이상으로 급증하면** 정책이 SFT 에서 멀리 벗어남 → `--kl-coef` ↑ |
| `len` | 답변 토큰 수 | **score 와 함께 급증하면 길이 해킹** |
| `eos` | 끝까지 답한 비율 | 100% 근처 유지 |
| `clipfrac` | ratio 가 clip 된 비율 | 0.1~0.3. 0.5 이상이면 lr ↓ |

10 스텝마다 답변 하나를 출력한다. **score 가 오르는데 답이 이상해지면 보상 해킹이다.**

### (5) 구현 검증

`train_ppo.py` 는 직접 만든 코드라서 Mac CPU 에서 Qwen2.5-0.5B-Instruct 로 두 가지를 확인했다.

1. **log π 일관성:** rollout 직후 업데이트 전에 다시 계산한 log π 가 rollout 때 값과 같아야 한다 (ratio = 1).
   왼쪽 padding 된 질문 + 오른쪽 padding 된 답변에 `position_ids` 를 직접 주지 않으면 여기서 어긋난다.
2. **보상을 실제로 높이는가:** RM 대신 "짧을수록 좋음" 보상을 넣고 25 스텝 돌려 score 가 오르고 답이 짧아지는지 봤다.

결과는 [강사용 가이드](00_facilitator.md#ppo-구현-검증) 에 적어 두었다.

## 7.5 평가: SFT vs DPO vs PPO

모듈 6 의 평가 스크립트에 RM 점수와 답변 길이를 함께 본다.

```bash
python 07_eval_sft.py --models SFT=models/k8s-ko-sft-hf/final DPO=models/k8s-ko-dpo-hf/final \
  PPO=models/k8s-ko-ppo-hf/final --reward-model models/k8s-ko-rm-hf/final
```

| 지표 | SFT | DPO | PPO |
|---|---|---|---|
| ppl glossary_heldout | | | |
| ppl general_heldout | | | |
| stop rate | | | |
| answer len (글자) | | | |
| RM score | | | |

해석 가이드:

| 관찰 | 의미 | 다음 행동 |
|---|---|---|
| RM score ↑, ppl·답변 품질 유지 | ✅ 선호 학습 성공 | 모듈 8 |
| RM score 만 ↑, 답이 길어지거나 이상해짐 | 보상 해킹 (RM 의 약점을 파고듦) | `--kl-coef` ↑, 스텝 ↓, 선호 데이터의 길이 편향 점검 |
| `ppl glossary_heldout` 이 크게 ↑ | 선호 학습이 CPT 지식을 흐림 | lr ↓, β ↑ (DPO) / kl-coef ↑ (PPO) |
| DPO 와 PPO 의 차이가 거의 없음 | 흔한 결과. 같은 선호 데이터에서 나온 같은 목표 | 비용이 싼 DPO 를 쓴다 |

> RM score 는 PPO 가 직접 최적화한 값이라 PPO 에 유리하다. **학습에 쓴 RM 으로만 평가하면 안 된다.**
> 생성 결과를 직접 읽고(과제 6-2 의 채점표), 가능하면 심판 LLM 으로 따로 채점한다.

## 7.6 DPO vs PPO 정리

| | DPO | PPO (RLHF) |
|---|---|---|
| 필요한 것 | 선호 쌍 | 선호 쌍 → 보상 모델 → 강화학습 |
| 학습 데이터 | 고정 (오프라인) | 매 스텝 정책이 새로 생성 (온라인) |
| 메모리·시간 | SFT 와 비슷 | 생성이 들어가서 몇 배 느림 |
| 하이퍼파라미터 | β, lr | kl_coef, clip, lr, vf_coef, λ ... |
| 장점 | 단순하고 안정적 | 학습 중 새 답을 탐색. RM 을 다른 데이터로 재사용 |
| 위험 | 선호 데이터 밖으로 일반화가 약함 | 보상 해킹, 불안정 |

최근에는 PPO 대신 가치 모델이 없는 **GRPO**, **RLOO** 같은 온라인 방법도 많이 쓴다 (TRL 1.x 에 `GRPOTrainer`, `RLOOTrainer` 가 있다).
같은 질문에 답 여러 개를 생성해 **그 묶음의 평균 점수를 기준선**으로 쓴다. 정답을 자동으로 확인할 수 있는 과제(수학, 코드)에 특히 잘 맞는다.

---

## 과제

### 과제 7-1. 심판과 나의 판단 비교

7.2 (3) 의 코드로 쌍 10개를 읽고 심판에 동의하는지 표시한다. 동의하지 않은 쌍은 어떤 이유였나? (길이, 사실, 말투 ...)
`JUDGE_PROMPT` 를 고쳐서 `--stage judge` 로 다시 채점하고 결과를 비교해 보자.

### 과제 7-2. β 바꿔 보기 (DPO)

```bash
python train_dpo_hf.py --beta 0.02 --output models/k8s-ko-dpo-b002
python train_dpo_hf.py --beta 0.5  --output models/k8s-ko-dpo-b05
python 07_eval_sft.py --models SFT=models/k8s-ko-sft-hf/final B002=models/k8s-ko-dpo-b002/final \
  B01=models/k8s-ko-dpo-hf/final B05=models/k8s-ko-dpo-b05/final --reward-model models/k8s-ko-rm-hf/final
```

β 가 작을수록 `answer len`, `ppl glossary_heldout` 은 어떻게 바뀌나?

### 과제 7-3. PPO 의 KL 벌점 끄기

```bash
python train_ppo.py --kl-coef 0 --steps 40 --output models/k8s-ko-ppo-nokl
```

로그의 `kl`, `len`, 출력되는 답변이 기본값(0.05)과 어떻게 다른가? 보상 해킹의 징후를 찾아보자.

### 과제 7-4. 보상 모델 정확도와 PPO

`train_rm_hf.py` 의 `eval_accuracy` 를 기록하고, 7.5 에서 RM score 가 오른 만큼 실제 답변이 좋아졌는지 생성 결과로 판단해 보자.
RM 이 틀리게 채점한 답(점수는 높은데 실제로는 나쁜 답)을 하나 찾아보자.

### 과제 7-5 (토론). 우리 서비스에는?

DPO 와 PPO 중 무엇을 고르겠나? 선호 데이터는 어디서 얻을 수 있나? (사용자 👍/👎, 상담원 수정 이력, 심판 LLM ...)

(정답과 해설은 [강사용 가이드](00_facilitator.md#모듈-7) 에 있다.)

---
← [모듈 6](06_instruction_tuning.md) · 다음: [모듈 8. 다음 단계와 정리 →](08_next_steps.md)
