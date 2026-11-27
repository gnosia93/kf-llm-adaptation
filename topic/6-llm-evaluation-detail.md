# LLM 학습 단계별 평가 및 DPO 후 NLI 기반 할루시네이션 검증 가이드

LLM 학습 파이프라인의 각 단계(**CPT ➡️ SFT ➡️ DPO**)는 목표가 다르기 때문에 평가 지표와 테스트 방식도 완전히 달라집니다.

---

## 1. 학습 단계별 평가 지표 요약

| 학습 단계 | 핵심 목표 | 주요 평가 방식 | 대표적인 지표 / 벤치마크 |
| :--- | :--- | :--- | :--- |
| **CPT** (Continual Pre-Training) | 도메인 지식 습득, 언어 유창성 | 다음 토큰 예측 정확도 | **Perplexity (PPL)**, 도메인 객관식(KMMLU 등) |
| **SFT** (Supervised Fine-Tuning) | 질문 의도 파악, 지시 이행 능력 | 정답 일치도, 태스크 수행 능력 | **ROUGE / BLEU**, **Exact Match**, **LLM-as-a-Judge** |
| **DPO** (Direct Preference Optimization) | 선호 답변 생성, 유해성/환각 억제 | 모델 간 승률(Win-rate), 안전성 | **MT-Bench / AlpacaEval**, **Reward Score**, **NLI (환각 체크)** |

---

## 2. SFT 및 DPO 이후 테스트 방법

### 2.1 SFT 이후 테스트 방법
SFT 이후에는 모델이 "사람의 질문에 알맞은 포맷과 내용으로 대답하는가?"를 봅니다. (이 시점부터는 PPL이 낮다고 좋은 모델이 아니므로 PPL을 거의 보지 않습니다.)

* **정량 평가 (벤치마크 데이터셋 활용)**
  * **객관식/추론 능력:** KMMLU, HellaSwag, GSM8K(수학), HumanEval(코딩) 등을 돌려 CPT 때 쌓은 지식이 망가지지 않았는지(Catastrophic Forgetting) 확인합니다.
  * **텍스트 생성 일치도:** 요약이나 번역 태스크라면 정답지(Ground Truth)와 비교해 **ROUGE**, **BLEU**, **BERTScore** 등을 계산합니다.
* **정성/자동 평가 (LLM-as-a-Judge)**
  * GPT-4o나 Claude 3.5 Sonnet 같은 고성능 모델에게 심사위원 역할을 맡겨 평가합니다.
  * 평가 프롬프트에 질문과 SFT 모델의 답변을 넣고 **[지시 이행 여부, 정확성, 유창성]**을 1~5점 척도로 채점하게 합니다.

### 2.2 DPO 이후 테스트 방법
DPO 이후에는 SFT 모델 대비 **"사람이 더 선호할 만한(더 친절하고, 논리적이며, 환각이 없는) 답변을 하는가?"**를 비교 평가합니다.

* **사이드 바이 사이드 승률 비교 (Win-Rate)**
  * DPO의 가장 핵심적인 평가 방식입니다. 동일한 평가 프롬프트에 대해 **SFT 모델의 답변**과 **DPO 모델의 답변**을 나란히 두고, LLM 심사위원(또는 사람)이 어느 쪽이 더 나은지 고르게 합니다.
  * 보통 `승리(Win) + 무승부(Tie) 비율`이 50%를 넘어야 DPO 학습이 성공했다고 봅니다. (대표 벤치마크: **MT-Bench**, **AlpacaEval 2.0**, **LogicKor** 등)
* **보상 모델 점수 및 마진 (Reward Margin) 확인**
  * DPO 학습 검증 데이터셋(Chosen / Rejected 쌍)을 넣어, 모델이 `Chosen` 답변에 부여하는 확률(암묵적 보상)이 `Rejected` 답변보다 얼마나 높아졌는지(**Reward Accuracy & Margin**) 확인합니다.
* **Alignment Tax (정렬 세금) 점검**
  * DPO를 과하게 학습하면 말투는 좋아지지만 지능(추론/코딩 능력)이 떨어지거나, 안전성 때문에 정상적인 질문도 거절(Over-refusal)하는 부작용이 생깁니다.
  * SFT 때 돌렸던 벤치마크(KMMLU 등)와 **NLI 기반 할루시네이션 체크**를 다시 돌려 성능 하락이 없는지 검증합니다.

---

## 3. DPO 후 NLI 기반 할루시네이션 체크 파이프라인

DPO 학습 후 NLI(자연어 추론) 모델을 활용해 할루시네이션을 검증하려면, **모델의 답변(Hypothesis)이 주어진 근거 문서나 정답(Premise)에 의해 논리적으로 뒷받침되는지(Entailment) 판별하는 파이프라인**을 구축해야 합니다.

### 3.1 NLI 분류 기준
* **Entailment (함의):** 근거 문장으로부터 답변의 내용이 논리적으로 추론됨 ➡️ **사실(Faithful)**
* **Contradiction (모순):** 근거 문장과 답변의 내용이 충돌함 ➡️ **명백한 할루시네이션**
* **Neutral (중립):** 근거 문장만으로는 답변의 참/거짓을 알 수 없음 ➡️ **근거 없는 지어내기(할루시네이션 의심)**

### 3.2 검증 파이프라인 구축 단계
1. **평가 데이터셋 준비:** DPO 학습에 쓰이지 않은 검증용 프롬프트, 참조 문서(Context/Ground Truth), 그리고 DPO 모델이 생성한 답변을 준비합니다.
2. **문장 단위 분리 (Claim Decomposition):** 모델의 답변이 길 경우 문장 단위나 개별 사실(Atomic Facts) 단위로 쪼개어 각각을 `Hypothesis`로 만듭니다.
3. **NLI 모델 추론 및 점수화:** `Premise`(근거 문서)와 `Hypothesis`(쪼개진 답변 문장) 쌍을 NLI 모델에 입력해 각 문장별 `Entailment` 및 `Contradiction` 확률을 계산합니다.
4. **DPO 전후 비교 (SFT vs DPO):** 전체 생성 문장 중 `Contradiction` 또는 `Neutral`로 판정된 문장의 비율이 SFT 대비 줄어들었는지 확인합니다.

### 3.3 추천 NLI 모델 및 도구
* **한국어 특화 NLI 모델**
  * `KLUE-RoBERTa-large` (KLUE NLI 데이터셋으로 파인튜닝된 모델)
  * `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (한국어 지원 다국어 모델)
* **할루시네이션 탐지 전용 오픈소스**
  * **Vectara Hughes Hallucination Evaluation Model (HHEM):** 문서 기반 할루시네이션 측정에 특화된 경량 모델
  * **AlignScore / SummaC:** 긴 문서와 생성된 텍스트 간의 일관성을 NLI 기반으로 정밀하게 채점해 주는 프레임워크

### 3.4 Python 구현 예시 (Hugging Face)

```python
from transformers import pipeline

# 다국어 NLI 모델 로드
nli_pipeline = pipeline(
    "text-classification", 
    model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", 
    device=0 # GPU 사용 시
)

premise = "세종대왕은 1443년에 훈민정음을 창제하고, 1446년에 반포하였다."
hypothesis = "훈민정음은 1446년에 세상에 널리 반포되었다."

# NLI 모델 입력 포맷에 맞게 결합
result = nli_pipeline(f"{premise}</s></s>{hypothesis}")
print(result)
# 출력 예시: [{'label': 'entailment', 'score': 0.985}]
```
