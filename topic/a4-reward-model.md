## 보상 모델(Reward Model, RM) 구조 및 학습·채점 가이드 ##

**보상 모델(Reward Model)** 은 SFT(지도 미세조정)가 끝난 모델이 생성한 답변의 품질을 사람 대신 자동으로 채점해 주는 **AI 심판관 모델** 이다. 주로 PPO/GRPO 같은 강화학습(RLHF) 단계에서 모델에게 보상 신호를 주거나, 여러 모델의 답변 품질을 정량적으로 비교 평가할 때 사용한다.

![](https://github.com/gnosia93/llm-train-ft/blob/main/images/reward-model.jpg)


### 1. SFT 모델에서 보상 모델로의 구조 변환 ###

보상 모델은 백지상태에서 새로 만들지 않고, 도메인 지식과 대화 형식을 이미 익힌 **SFT 모델의 몸통(Transformer Body)을 그대로 가져온 뒤 맨 끝 출력층(Head)만 교체**하여 만든다.

* **기존 SFT 모델 (`AutoModelForCausalLM`):**
  * `[트랜스포머 본체]` + **`[언어 모델 헤드 (lm_head)]`**
  * 수만~십수만 개의 어휘 사전(Vocabulary) 중 다음에 올 단어 확률을 계산한다.
* **보상 모델 (`AutoModelForSequenceClassification(num_labels=1)`):**
  * `[트랜스포머 본체 (SFT 가중치 복사)]` + **`[점수 채점 헤드 (score head)]`**
  * 기존 단어 예측 헤드를 제거하고, 대화 전체를 읽은 뒤 마지막 토큰 위치에서 **단일 실수 점수(Scalar Logit) 1개**만 출력하는 선형 레이어(Linear Layer)를 새로 부착한다.

---

## 2. 출력값의 범위와 해석 기준

* **출력 범위 (\(-\infty \sim +\infty\)):** 분류 헤드 끝에는 `Sigmoid`나 `Softmax`처럼 값을 `0~1` 사이로 누르는 활성화 함수가 붙지 않는다. 따라서 이론상 제한이 없는 모든 실수값이 나오며, 실무에서는 보통 **`-5.0`에서 `+5.0` (넓게는 `-10` ~ `+10`) 사이**에서 점수가 형성된다.
* **상대 비교 원칙:** 보상 모델은 "5점 만점에 몇 점"이라는 절대 기준이 아니라, **좋은 답변(`chosen`)과 나쁜 답변(`rejected`) 사이의 점수 차이(Margin)**만 크게 벌리도록 학습한다.
  \[\mathcal{L}_{\text{RM}} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})\]
* 따라서 단일 점수의 절대적인 크기 자체에는 의미가 없으며, **반드시 동일한 보상 모델로 채점한 두 모델 간의 평균 점수 차이**를 비교하는 용도로만 해석해야 한다.

---

## 3. 데이터 포맷 및 실전 파이썬 코드 (TRL 기준)

### ① 훈련용 데이터 포맷 (`chosen` / `rejected` 쌍)
같은 질문에 대해 더 선호되는 답변(`chosen`)과 덜 선호되는 답변(`rejected`)을 1:1 쌍으로 묶어 학습시킨다.

```json
{
  "chosen": [
    {"role": "system", "content": "당신은 쿠버네티스 전문가입니다."},
    {"role": "user", "content": "파드(Pod)가 무엇인지 설명해줘."},
    {"role": "assistant", "content": "파드는 쿠버네티스에서 배포할 수 있는 가장 작은 컴퓨팅 단위로, 하나 이상의 컨테이너를 포함합니다."}
  ],
  "rejected": [
    {"role": "system", "content": "당신은 쿠버네티스 전문가입니다."},
    {"role": "user", "content": "파드(Pod)가 무엇인지 설명해줘."},
    {"role": "assistant", "content": "파드는 그냥 도커 컨테이너랑 똑같은 거고 서버 한 대를 의미합니다."}
  ]
}
```

### ② 보상 모델 학습 코드 (`RewardTrainer`)
Hugging Face `transformers`와 `trl` 라이브러리를 사용하면 헤드 교체부터 마진 극대화 학습까지 간단히 구현할 수 있다.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import RewardConfig, RewardTrainer

model_id = "models/k8s-ko-sft-hf/final"  # SFT 완료된 모델 경로

# 1. 토크나이저 및 보상 모델 로드 (기존 lm_head 제거 후 num_labels=1 점수 헤드 자동 부착)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=1,
    torch_dtype=torch.bfloat16,
)
model.config.pad_token_id = tokenizer.pad_token_id

# 2. LoRA 설정 (SEQ_CLS 태스크 지정 시 새로 달린 score 헤드도 자동으로 학습 대상에 포함됨)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# 3. 학습 설정 및 실행
dataset = load_dataset("json", data_files="data/rm_train.jsonl", split="train")

training_args = RewardConfig(
    output_dir="models/k8s-ko-rm-hf",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    bf16=True,
    max_length=2048,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model("models/k8s-ko-rm-hf/final")
```

### ③ 학습된 보상 모델로 답변 채점하기 (추론)
채점 시에는 `chosen`/`rejected` 쌍이 아니라 **질문과 평가할 답변 1개가 합쳐진 단일 대화**를 입력한다.

```python
@torch.no_grad()
def get_reward_score(rm_path: str, question: str, answer: str) -> float:
    tok = AutoTokenizer.from_pretrained(rm_path)
    rm = AutoModelForSequenceClassification.from_pretrained(
        rm_path, torch_dtype=torch.bfloat16
    ).cuda().eval()

    messages = [
        {"role": "system", "content": "당신은 쿠버네티스 전문가입니다."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    text = tok.apply_chat_template(messages, tokenize=False)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

    # bfloat16 정밀도/타입 에러 방지를 위해 .float() 변환 후 스칼라 추출
    score = rm(**inputs).logits[0, 0].float().item()
    return score

# 사용 예시
score = get_reward_score(
    "models/k8s-ko-rm-hf/final",
    "파드가 무엇인가요?",
    "파드는 쿠버네티스에서 생성하고 관리할 수 있는 배포 가능한 가장 작은 단위입니다."
)
print(f"답변 보상 점수: {score:.4f}")
```

> [!WARNING] 보상 해킹(Reward Hacking) 주의
> PPO 같은 강화학습 단계에서 보상 모델 점수만 높이도록 훈련하다 보면, 모델이 실제 정답 정확도는 떨어뜨리면서 **"보상 모델이 좋아하는 긴 말투나 특정 키워드만 나열해 높은 점수를 따내는 현상(보상 해킹)"**이 발생할 수 있다. 따라서 반드시 답변 평균 길이(`answer len`)와 일반 대화 성능 지표(`general ppl`)를 함께 감시해야 한다.
