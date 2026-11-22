## Alignment(RM·DPO·GRPO) ##

**Alignment(정렬) 튜닝** 은 단순히 질문에 대답하는 형식만 배운 모델(SFT)에게 **사람이 진짜로 원하는 기준(정확성, 유용성, 안전성, 솔직함)에 눈높이를 맞추도록 다듬는 최종 마무리 훈련** 이다.

* **SFT의 한계:** SFT는 모범 답안 1개를 보고 그대로 따라 쓰도록 가르치는 과정("기출문제 풀이")이다. 따라서 SFT만 마친 모델은 모르는 내용도 그럴싸하게 지어내거나(환각), 불필요하게 장황하게 답하거나, 위험한 명령어를 그대로 안내하는 한계를 보인다.
* **Alignment의 핵심:** 여러 답변 후보 중 **"어느 쪽이 더 좋은 답변인가"를 비교하며 판단 기준(선호도)을 깨우치게 하는 과정("실전 면접·센스 코칭")**이다. 이를 통해 모델은 모르는 것은 모른다고 답하고, 간결하고 정확한 명령어를 제시하며, 안전하고 유용한 방식으로 대답하는 센스를 갖추게 된다.


`1. CPT (계속 사전학습)` - 전공 서적 통독 / 특정 분야(예: 쿠버네티스)의 단어, 문맥, 전문 지식을 머릿속에 주입 (Next Token Prediction)

`2. SFT (지도 미세조정)` - 기출문제 풀이 / 질문이 들어오면 대답하는 기본 대화 형식(`Chat Template`)과 말투 습득 

`3. Alignment (정렬 튜닝)` - 실전 센스 코칭 / 여러 답변 중 **사람이 선호하는 정확하고 안전한 답변** 을 선택하도록 눈높이 정렬 (RM+PPO, DPO, GRPO)



### LLM 전체 훈련 3단계 비교

| 훈련 단계 | 쉬운 비유 | 핵심 목표 | 대표 학습 기법 |
| :--- | :--- | :--- | :--- |

---

## 2. 대표 Alignment 기법 비교: RLHF(PPO) vs DPO vs GRPO

| 구분 | 기존 RLHF (RM + PPO) | DPO (직접 선호도 최적화) | **GRPO (그룹 상대 최적화)** |
| :--- | :--- | :--- | :--- |
| **핵심 작동 방식** | 심판관 모델(RM)과 가치 모델(Critic)을 띄워 실시간 강화학습 수행 | 별도 심판관 없이 선호 쌍(`chosen`/`rejected`)으로 확률 비율 직접 최적화 | Critic 모델을 제거하고, 동일 질문에서 생성한 $G$개 답변의 **그룹 내 평균 점수**로 상대평가 |
| **필요한 모델 수** | **4개** (Policy, Ref, Reward, Critic) | **2개** (Policy, Ref) | **2개** (Policy, Ref — Critic 제거) |
| **학습 데이터 형태** | 프롬프트 + 실시간 생성 | 사전에 구축된 `chosen` / `rejected` 쌍 | **프롬프트(질문) + 규칙 검증 함수(Verifier)** |
| **GPU 메모리 소모** | 매우 큼 | **절반 이하로 절약** (LoRA 시 Ref 별도 불필요) | **가벼움** (PPO 대비 메모리 대폭 절감) |
| **주요 강점 분야** | 범용 대화 정렬 | 말투, 스타일, 친절함, 안전성 교정 | **추론(Reasoning), 수학, 코딩, YAML·명령어 생성** |

* ① Policy 모델 (학습 대상): 지금 DPO 훈련을 받으며 실시간으로 가중치가 업데이트되는 모델입니다.
* ② Reference 모델 (기준점 / 고정됨): DPO를 시작하기 직전의 원래 SFT 모델 복사본입니다. 훈련 내내 가중치가 절대 변하지 않도록 자물쇠로 잠가둡니다.


## 3. 선호도 데이터(`chosen` / `rejected`) 자동 구축 파이프라인

수천 쌍의 `chosen`과 `rejected` 데이터를 사람이 직접 작성하는 대신, **현재 SFT 모델의 다중 샘플링 + 고성능 LLM 심판관(LLM-as-a-Judge)**을 결합해 자동으로 구축한다.

### ① 데이터 포맷 구조
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

### ② AI 심판관 자동 생성 스크립트
```python
import json
from openai import OpenAI

client = OpenAI()

JUDGE_PROMPT = """다음은 쿠버네티스 질문과 모델이 생성한 두 가지 답변이다.
원문 지식을 기준으로 정확성, 명확성, 한국어 자연스러움을 평가해 각 답변에 1~5점을 부여하라.
반드시 JSON 형식으로만 출력하라: {{"score_a": 점수, "score_b": 점수}}

[질문] {question}
[답변 A] {ans_a}
[답변 B] {ans_b}"""

def build_preference_pair(system_msg: str, question: str, candidates: list[str]) -> dict | None:
    ans_a, ans_b = candidates[0], candidates[1]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, ans_a=ans_a, ans_b=ans_b
        )}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    scores = json.loads(resp.choices[0].message.content)
    sa, sb = scores["score_a"], scores["score_b"]

    # 점수 차이가 1점 미만인 모호한 쌍은 노이즈 방지를 위해 제외
    if abs(sa - sb) < 1:
        return None

    chosen_ans, rejected_ans = (ans_a, ans_b) if sa > sb else (ans_b, ans_a)
    return {
        "chosen": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            {"role": "assistant", "content": chosen_ans},
        ],
        "rejected": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            {"role": "assistant", "content": rejected_ans},
        ],
    }
```

---

## 4. [Alignment 구현 1] 보상 모델(Reward Model, RM) 구축 및 채점

보상 모델은 SFT 모델의 문맥 이해 본체(Transformer Body)는 그대로 유지한 채, 맨 끝 **단어 예측 헤드(`lm_head`)를 제거하고 단일 실수 점수를 출력하는 선형 헤드(`score head`, `num_labels=1`)로 교체**하여 만든다.

* **출력값 범위 ($-\infty \sim +\infty$):** 별도의 활성화 함수(`Sigmoid` 등)가 없으므로 제한 없는 실수(보통 **`-5.0` ~ `+5.0` 내외**)가 출력된다.
* **상대 비교 원칙:** 절대 점수가 아니라 `chosen`과 `rejected` 사이의 **점수 격차(Margin)를 극대화**하도록 학습한다 ($\mathcal{L}_{	ext{RM}} = -\log \sigma(r_{	ext{chosen}} - r_{	ext{rejected}})$). 따라서 반드시 **동일한 보상 모델로 채점한 모델 간 평균 점수 차이**만 비교해야 한다.

### ① 보상 모델 학습 및 채점 코드 (`RewardTrainer`)
```python
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import RewardConfig, RewardTrainer

model_id = "models/k8s-ko-sft-hf/final"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 기존 lm_head 제거 후 단일 실수 점수를 출력하는 score 헤드 자동 부착
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=1, torch_dtype=torch.bfloat16
)
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

dataset = load_dataset("json", data_files="data/preference_train.jsonl", split="train")
training_args = RewardConfig(
    output_dir="models/k8s-ko-rm-hf",
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=1e-5, num_train_epochs=1, bf16=True,
    max_length=2048, logging_steps=10, save_strategy="epoch",
)

trainer = RewardTrainer(
    model=model, args=training_args, processing_class=tokenizer,
    train_dataset=dataset, peft_config=peft_config,
)
trainer.train()
trainer.save_model("models/k8s-ko-rm-hf/final")

# [추론/채점 함수] 단일 대화(질문 + 답변)를 입력해 보상 점수 추출
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
    return rm(**inputs).logits[0, 0].float().item()
```

---

## 5. [Alignment 구현 2] DPO (Direct Preference Optimization) 학습

DPO는 보상 모델을 따로 훈련하지 않고, **기준 모델(Reference Model, 초기 SFT 상태) 대비 현재 모델의 출력 확률 비율**을 직접 최적화한다.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

model_id = "models/k8s-ko-sft-hf/final"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)

dataset = load_dataset("json", data_files="data/preference_train.jsonl", split="train")
training_args = DPOConfig(
    output_dir="models/k8s-ko-dpo-hf",
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=5e-6,            # SFT(1e-5)보다 낮은 학습률 사용
    beta=0.1,                      # 기준 모델(SFT) 이탈 방지 계수 (0.1 ~ 0.5)
    lr_scheduler_type="cosine", warmup_ratio=0.05,
    num_train_epochs=1, bf16=True,
    max_length=2048, max_prompt_length=1024,
    logging_steps=10, save_strategy="epoch",
)

trainer = DPOTrainer(
    model=model, ref_model=None,   # LoRA 사용 시 어댑터 비활성화 원본을 자동 참조
    args=training_args, processing_class=tokenizer,
    train_dataset=dataset, peft_config=peft_config,
)
trainer.train()
trainer.save_model("models/k8s-ko-dpo-hf/final")
```

---

## 6. [Alignment 구현 3] GRPO (Group Relative Policy Optimization) 학습

**GRPO**는 DeepSeek 팀이 개발한 기법으로, 무거운 Critic 모델을 없애고 **하나의 질문에 대해 생성된 $G$개의 답변 그룹 내 상대적 평균 점수**를 기준으로 가중치를 업데이트한다. 코드·YAML 문법 검사나 정답 매칭 같은 **규칙 기반 보상 함수(Verifier)**와 결합할 때 추론 정확도를 극대화할 수 있다.

```python
import re
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

model_id = "models/k8s-ko-sft-hf/final"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 보상 함수 1: <think> 추론 과정 및 yaml 코드 블록 형식 준수 검증
def format_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for content in completions:
        text = content[0]["content"] if isinstance(content, list) else content
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        has_yaml = "```yaml" in text
        rewards.append((0.5 if has_think else 0.0) + (0.5 if has_yaml else 0.0))
    return rewards

# 보상 함수 2: 실제 YAML 문법 유효성 검증 (Verifier)
def yaml_syntax_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for content in completions:
        text = content[0]["content"] if isinstance(content, list) else content
        match = re.search(r"```yaml\n(.*?)```", text, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue
        try:
            parsed = yaml.safe_load(match.group(1))
            if isinstance(parsed, dict) and "apiVersion" in parsed and "kind" in parsed:
                rewards.append(2.0)
            else:
                rewards.append(0.5)
        except yaml.YAMLError:
            rewards.append(-1.0)
    return rewards

peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# 데이터셋은 'prompt' 컬럼만 포함하면 됨
dataset = load_dataset("json", data_files="data/grpo_prompts.jsonl", split="train")
training_args = GRPOConfig(
    output_dir="models/k8s-ko-grpo-hf",
    learning_rate=5e-6, lr_scheduler_type="cosine", warmup_ratio=0.05,
    num_generations=4,             # 질문당 생성할 그룹 답변 수 (G = 4)
    max_prompt_length=512, max_completion_length=1024,
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    beta=0.04, bf16=True, num_train_epochs=1,
    logging_steps=5, save_strategy="epoch",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_func, yaml_syntax_reward_func],
    args=training_args, train_dataset=dataset,
    processing_class=tokenizer, peft_config=peft_config,
)
trainer.train()
trainer.save_model("models/k8s-ko-grpo-hf/final")
```

> [!WARNING] Alignment 튜닝 시 사후 점검 필수 사항
> * **보상 해킹(Reward Hacking) 감시:** Alignment 학습 후 답변 평균 길이(`answer len`)가 비정상적으로 급증하거나, 일반 대화 성능(`general ppl`)이 나빠졌다면 모델이 편법을 배운 것이므로 `beta` 값을 높이거나 학습률을 낮춘다.

---

## 7. 추론 서빙 준비: LoRA 병합 및 종료 토큰(`EOS`) 고정

학습이 완료된 LoRA 어댑터를 고속 추론 엔진(**vLLM**)에 올리기 전, 원본 모델 가중치와 어댑터를 하나로 합치고(Merge) **대화 턴 종료 토큰(`<|im_end|>`)을 `generation_config`에 반드시 등록**해야 답변 후 말이 끊기지 않고 폭주하는 현상을 막을 수 있다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "models/k8s-ko-dpo-hf/final"
merged_save_path = "models/k8s-ko-merged"

# 1. 원본 모델 및 LoRA 어댑터 로드 후 영구 병합
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map="cpu"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# 2. 종료 토큰(EOS) 설정: <|endoftext|> 와 <|im_end|> 모두 등록
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
merged_model.generation_config.eos_token_id = list({im_end_id, endoftext_id})

# 3. 병합된 최종 모델 저장
merged_model.save_pretrained(merged_save_path, safe_serialization=True)
tokenizer.save_pretrained(merged_save_path)
print(f"병합 완료 및 저장 경로: {merged_save_path}")
```

---

## 8. vLLM 추론 서버 배포 (OpenAI 호환 API)

**vLLM**은 PagedAttention 기술을 사용해 대량의 동시 요청을 초고속으로 처리하며, 표준 OpenAI API 포맷(`/v1/chat/completions`)을 그대로 제공한다.

### ① CLI 명령어로 즉시 실행
```bash
vllm serve models/k8s-ko-merged     --served-model-name k8s-assistant     --host 0.0.0.0     --port 8000     --dtype bfloat16     --max-model-len 4096     --gpu-memory-utilization 0.90
```

### ② Docker Compose 운영 배포 예시 (`docker-compose.yml`)
```yaml
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    container_name: k8s-vllm
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    volumes:
      - ./models/k8s-ko-merged:/model
    ports:
      - "8000:8000"
    ipc: host
    command: >
      --model /model
      --served-model-name k8s-assistant
      --dtype bfloat16
      --max-model-len 4096
      --gpu-memory-utilization 0.90
```

---

## 9. RAG (검색 증강 생성) 파이프라인 연동

파인튜닝(CPT·SFT·Alignment)이 모델에게 **도메인 개념과 말투·추론 능력**을 내재화하는 과정이라면, **RAG**는 최신 버전 변경사항이나 사내 클러스터 설정처럼 **실시간·외부 문서 근거를 주입해 환각을 0%에 가깝게 줄이는 필수 결합 기술**이다.

### RAG + 파인튜닝 모델 통합 추론 코드 (FAISS + BGE-M3 + vLLM)
```python
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 한국어·다국어 성능이 뛰어난 임베딩 모델 로드 및 벡터 DB 구축
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

# 문서 청킹 및 인덱스 생성 (최초 1회 구축 후 로컬 저장)
raw_docs = [
    "파드(Pod)는 쿠버네티스에서 생성하고 관리할 수 있는 배포 가능한 가장 작은 컴퓨팅 단위다.",
    "디플로이먼트(Deployment)는 파드와 레플리카셋(ReplicaSet)에 대한 선언적 업데이트를 제공한다.",
]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents(raw_docs)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("data/k8s_faiss_index")

# 2. vLLM 추론 서버 클라이언트 연결
vllm_client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def ask_with_rag(query: str, top_k: int = 3) -> str:
    # 관련 공식 문서 청크 검색
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n\n".join([f"- {doc.page_content}" for doc in retrieved_docs])

    system_prompt = (
        "당신은 쿠버네티스 전문가입니다. 반드시 아래 제공된 [참고 문서]의 사실에만 근거하여 "
        "정확하고 간결하게 답변하세요. 문서에 없는 내용은 추측하지 말고 모른다고 답하세요."
    )
    user_prompt = f"[참고 문서]\n{context}\n\n[질문]\n{query}"

    response = vllm_client.chat.completions.create(
        model="k8s-assistant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # RAG 환경에서는 사실 충실도를 높이기 위해 낮은 온도 설정
        max_tokens=1024,
    )
    return response.choices[0].message.content

# 실행 테스트
print(ask_with_rag("쿠버네티스에서 배포 가능한 가장 작은 단위가 뭐야?"))
```

> [!TIP] 파인튜닝과 RAG의 시너지 효과
> * **일반 범용 모델 + RAG:** 문서를 찾아주어도 쿠버네티스 특수 용어나 YAML 구조를 어색하게 해석하거나 장황하게 답변하는 경우가 많다.
> * **도메인 튜닝(CPT+SFT+Alignment) 모델 + RAG:** 이미 도메인 문맥과 정확한 출력 형식을 체화한 상태이므로, 검색된 문서를 훨씬 정확하게 이해하고 군더더기 없는 실무형 답변을 생성한다.
