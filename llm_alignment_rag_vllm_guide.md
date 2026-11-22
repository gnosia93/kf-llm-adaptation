# LLM 파인튜닝부터 선호도 정렬(RM/DPO), RAG 연동 및 vLLM 서버 배포까지 완전 가이드

본 문서는 도메인 특화 LLM(예: 쿠버네티스 전문가 모델)을 구축하기 위한 **전 과정 엔드투엔드(End-to-End) 실무 매뉴얼**이다. 선호도 정렬(Reward Model & DPO)부터 모델 병합(LoRA Merge), **vLLM 고성능 추론 서버 배포**, 그리고 **RAG(검색 증강 생성) 파이프라인 연동**까지의 전체 흐름과 소스코드를 담았다.

---

## 1. 전체 서비스 아키텍처 흐름

```text
[1단계: 학습 및 정렬]
Base Model ──▶ CPT(도메인 지식) ──▶ SFT(대화 형식) ──▶ DPO / RM+PPO(선호도 정렬)
                                                               │
[2단계: 모델 병합 및 서빙]                                       ▼
LoRA Adapter + Base Model ──▶ Merge(단일 모델 저장) ──▶ vLLM 추론 서버 (OpenAI 호환 API)
                                                               ▲
[3단계: RAG 파이프라인 연동]                                     │ (프롬프트 + 검색 문서 전달)
사용자 질문 ──▶ 벡터 DB 검색(Retriever) ──▶ 컨텍스트 조립 ────────┘
```

---

## 2. 선호도 정렬 (Preference Alignment): Reward Model & DPO

SFT를 마친 모델이 환각(거짓말)이나 모호한 답변을 내놓는 빈도를 줄이고, 사람이 선호하는 정확하고 간결한 답변을 생성하도록 고도화하는 단계다.

### ① 접근 방식 비교: RLHF(RM + PPO) vs DPO

| 구분 | 기존 RLHF (RM + PPO) | DPO (직접 선호도 최적화) |
| :--- | :--- | :--- |
| **핵심 원리** | 심판관 모델(RM)을 먼저 학습한 뒤 강화학습(PPO)으로 정책 모델 업데이트 | 별도 보상 모델 없이 선호 데이터(`chosen`/`rejected`)로 SFT 모델을 직접 업데이트 |
| **필요한 모델 수** | **4개** (Policy, Reference, Reward, Value) | **2개** (학습할 Policy 모델, 기준이 되는 Reference 모델) |
| **GPU 메모리 소모** | 매우 큼 (4개 모델 동시 적재) | **절반 이하로 절약** (LoRA 사용 시 Reference 모델 별도 로드 불필요) |
| **학습 안정성** | 하이퍼파라미터에 민감하고 보상 해킹 위험이 높음 | 일반 SFT와 유사한 분류 손실 기반이라 **매우 안정적** |

### ② 선호도 데이터(`chosen` / `rejected`) 자동 구축 파이프라인
수천 쌍의 선호 데이터를 사람이 직접 작성하지 않고, **현재 SFT 모델의 다중 샘플링 + 고성능 LLM(LLM-as-a-Judge) 채점**을 결합해 자동으로 구축한다.

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

### ③ [방법 A] 보상 모델(Reward Model, RM) 학습 및 채점
SFT 모델의 단어 예측 헤드(`lm_head`)를 제거하고 단일 실수 점수(`-∞ ~ +∞`, 보통 `-5 ~ +5`)를 출력하는 선형 헤드(`num_labels=1`)를 부착해 학습시킨다.

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
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    bf16=True,
    max_length=2048,
)

trainer = RewardTrainer(
    model=model, args=training_args, processing_class=tokenizer,
    train_dataset=dataset, peft_config=peft_config,
)
trainer.train()
trainer.save_model("models/k8s-ko-rm-hf/final")
```

### ④ [방법 B] DPO (Direct Preference Optimization) 직접 학습
보상 모델 없이 기준 모델(SFT) 대비 확률 비율을 직접 최적화한다. LoRA 사용 시 `ref_model=None`으로 두면 어댑터가 비활성화된 원본 상태가 자동으로 Reference 역할을 수행한다.

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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,            # SFT(1e-5)보다 낮은 학습률 사용
    beta=0.1,                      # 기준 모델 이탈 방지 규제 계수
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=1,
    bf16=True,
    max_length=2048,
    max_prompt_length=1024,
)

trainer = DPOTrainer(
    model=model, ref_model=None, args=training_args,
    processing_class=tokenizer, train_dataset=dataset, peft_config=peft_config,
)
trainer.train()
trainer.save_model("models/k8s-ko-dpo-hf/final")
```

---

## 3. 모델 병합(LoRA Merge) 및 vLLM 추론 서버 배포

학습이 끝난 LoRA 어댑터를 실제 프로덕션 환경에서 초고속으로 서빙하려면, 먼저 원본 모델과 어댑터를 하나로 합친(Merge) 뒤 **vLLM(PagedAttention 기반 고성능 추론 엔진)**으로 OpenAI 호환 API 서버를 띄우는 것이 정석이다.

### ① LoRA 어댑터 병합 및 종료 토큰(`EOS`) 최종 점검 (`merge_lora.py`)
추론 엔진이 답변 끝에서 제때 말을 멈추도록 `<|im_end|>`와 `<|endoftext|>`를 `generation_config`에 함께 등록해 저장한다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-7B-Instruct"   # 또는 CPT 완료된 Base 경로
adapter_path = "models/k8s-ko-dpo-hf/final"  # SFT 또는 DPO 어댑터 경로
merged_save_path = "models/k8s-ko-merged"

# 1. 원본 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # 병합 시 VRAM 부족 방지를 위해 CPU RAM 활용
)

# 2. LoRA 가중치 병합 (W_new = W + scale * A * B) 후 어댑터 껍데기 제거
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# 3. 종료 토큰(EOS) 안전 설정 (답변 후 폭주 방지)
stop_ids = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
    tokenizer.convert_tokens_to_ids("<|endoftext|>"),
]
merged_model.generation_config.eos_token_id = [i for i in stop_ids if i is not None]

# 4. 단일 모델로 저장
merged_model.save_pretrained(merged_save_path, safe_serialization=True)
tokenizer.save_pretrained(merged_save_path)
print(f"병합 및 저장 완료: {merged_save_path}")
```

### ② vLLM 추론 서버 실행 (CLI & Docker)

#### 터미널(CLI) 직접 실행
```bash
# vLLM 설치: pip install vllm
vllm serve models/k8s-ko-merged \
    --served-model-name k8s-ko-assistant \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1          # 멀티 GPU 사용 시 GPU 개수(예: 2, 4)로 변경
```

#### 프로덕션용 Docker Compose 배포 (`docker-compose.yml`)
```yaml
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    container_name: k8s-vllm-server
    runtime: nvidia
    ipc: host                         # 멀티 GPU 통신 및 공유 메모리 확보 필수 옵션
    ports:
      - "8000:8000"
    volumes:
      - ./models/k8s-ko-merged:/models/k8s-ko-merged
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    command: >
      --model /models/k8s-ko-merged
      --served-model-name k8s-ko-assistant
      --dtype bfloat16
      --max-model-len 8192
      --gpu-memory-utilization 0.90
      --tensor-parallel-size 1
```

> [!TIP] vLLM 핵심 운영 옵션 가이드
> * **`--gpu-memory-utilization 0.90`:** 전체 GPU VRAM의 90%를 KV 캐시용으로 미리 확보하여 동시 접속자 처리량(Throughput)을 극대화한다.
> * **`--max-model-len 8192`:** RAG 연동 시 검색된 긴 문서 조각(Context) 여러 개가 프롬프트에 함께 들어가므로, 최소 `4096 ~ 8192` 이상으로 여유 있게 설정해야 문맥 잘림 에러가 발생하지 않는다.
> * **`--enable-lora` (선택 사항):** 모델을 병합하지 않고 여러 LoRA 어댑터를 동적으로 갈아끼우며 서빙하고 싶을 때 사용한다.

---

## 4. RAG (검색 증강 생성) 파이프라인 연동

파인튜닝(CPT/SFT/DPO)이 모델에게 **도메인 용어 이해력과 전문가다운 말투·형식**을 가르쳤다면, **RAG(Retrieval-Augmented Generation)**는 **최신 버전 정보나 사내 비공개 문서처럼 모델이 외우지 못한 구체적 사실을 실시간으로 검색해 근거로 제공**함으로써 환각(Hallucination)을 원천 차단한다.

### ① 문서 인덱싱 (Chunking & Vector DB 구축)
```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 마크다운 문서 로드 및 청킹 (문맥 유지를 위해 overlap 부여)
loader = DirectoryLoader("./docs/k8s-ko", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)
chunks = splitter.split_documents(docs)

# 2. 한국어 임베딩 모델 로드 및 FAISS 벡터 저장소 생성
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",         # 한국어 및 다국어 검색 성능이 검증된 임베딩 모델
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("./vectorstore/k8s_faiss")
print(f"총 {len(chunks)}개 청크 인덱싱 완료")
```

### ② vLLM 서버와 RAG 검색기 연동 추론 (`rag_client.py`)
vLLM은 표준 **OpenAI API 스펙(`http://localhost:8000/v1`)**을 완벽하게 지원하므로, 공식 `openai` 파이썬 클라이언트를 그대로 사용해 검색된 문서를 프롬프트에 주입할 수 있다.

```python
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. vLLM 서버 클라이언트 연결
vllm_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # 로컬 vLLM 서버는 별도 키 검증 없이 동작
)

# 2. 저장된 벡터 DB 로드
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local(
    "./vectorstore/k8s_faiss", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # 상위 4개 관련 문서 검색

SYSTEM_PROMPT = """당신은 쿠버네티스 전문가입니다.
반드시 아래 제공된 [참고 문서]의 내용만을 근거로 정확하고 간결하게 한국어로 답변하세요.
참고 문서에 없는 내용은 추측하지 말고 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요."""

def ask_rag(question: str) -> str:
    # 1단계: 사용자 질문과 유사한 문서 조각 검색
    retrieved_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(
        [f"[출처: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in retrieved_docs]
    )

    # 2단계: 검색된 문맥과 질문을 결합해 vLLM 서버에 생성 요청
    user_prompt = f"[참고 문서]\n{context}\n\n[질문]\n{question}"

    response = vllm_client.chat.completions.create(
        model="k8s-ko-assistant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,              # RAG 기반 사실 답변은 낮게 설정해 환각 억제
        max_tokens=1024,
    )
    return response.choices[0].message.content

# 실행 테스트
if __name__ == "__main__":
    query = "파드가 CrashLoopBackOff 상태일 때 원인과 해결 순서를 알려줘."
    print(ask_rag(query))
```

---

## 5. 프로덕션 운영 시 핵심 체크리스트

> [!WARNING] RAG + vLLM 통합 운영 시 자주 겪는 함정과 해결책
> * **시스템 프롬프트 일치성 유지:** SFT/DPO 훈련 시 사용했던 `system` 프롬프트의 핵심 어조와 역할을 RAG 시스템 프롬프트에도 일관되게 유지해야 파인튜닝된 답변 스타일과 코드 포맷팅 능력이 100% 발휘된다.
> * **컨텍스트 길이 초과 (`ContextLengthExceededError`):** 검색 문서 개수(`k`)를 너무 크게 잡으면 `시스템 프롬프트 + 검색 문서 + 질문 + 생성 토큰`의 합이 vLLM의 `--max-model-len`을 초과해 API 400 에러가 발생한다. 청크 크기(`800자`) × `k=3~4` 수준으로 총 입력 토큰을 `3,000~4,000` 이내로 관리한다.
> * **Temperature 설정:** 자유로운 창작이 아닌 문서 기반 질의응답(RAG)에서는 `temperature=0.0 ~ 0.2`로 낮게 설정해야 검색된 근거를 벗어나 엉뚱한 말을 지어내는 현상을 막을 수 있다.
