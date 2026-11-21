## Instruction Tuning ##

### 1. CPT 모델의 한계 ###

CPT 모델은 base 모델이라 **이어쓰기만** 한다. 예를들어 "파드가 뭐야?" 라고 넣으면 답하지 않고 비슷한 질문을 나열하거나 문서를 이어 쓴다.  
반대로 일반 instruct 모델은 대화는 하지만 도메인 지식이 없다. 실습 준비 중에 소형 모델에 실제로 물어본 결과:
```
Q: 파드가 뭐야? 한 문장으로.
Qwen2.5-0.5B-Instruct: 파드는 컴퓨터의 메모리 공간을 의미하는 단어입니다.     ← 환각
```
필요한 것은 **CPT 로 넣은 지식 + 질문에 답하는 형식**의 결합이다. 이번 모듈에서 두 가지 방법을 비교한다.

```
                       ┌── A. 지시 튜닝 (SFT): 질의응답 데이터로 추가 학습 ──┐
CPT 모델 (지식) ───────┤                                                    ├──▶ 지식 + 대화
                       └── B. Chat Vector: 가중치 산수, 학습 없음 ──────────┘
```

**CPT 는 "무엇을 아는가"를, SFT 는 "어떻게 답하는가"를 바꾼다.**
SFT 데이터의 양(수천 개)은 CPT 데이터(120만 토큰)보다 훨씬 적다. 새 지식을 넣기보다는 이미 있는 지식을 **질문-답변 형식으로 꺼내는 법**을 가르치는 단계다.

### 2. chat 템플릿 ###

결국 모델이 입력으로 받는 것은 단순한 토큰들의 나열(토큰 열)뿐이며, 역할(system / user / assistant)은 특수 토큰으로 구분한다.
Qwen2.5 는 아래 형식을 쓰고, base 토크나이저에도 같은 template 이 있다.

```
<|im_start|>system
당신은 쿠버네티스 전문가입니다. ...<|im_end|>
<|im_start|>user
파드가 뭐야?<|im_end|>
<|im_start|>assistant
가장 작고 단순한 쿠버네티스 오브젝트. 파드는 ...<|im_end|>
```

`<|im_start|>`, `<|im_end|>` 는 단어가 아니라 **특수 토큰 하나씩**이다. 직접 확인해 보자.

### 3. Loss 계산 ###

```
토큰:  <|im_start|>system ... <|im_end|> <|im_start|>user 파드가 뭐야? <|im_end|> <|im_start|>assistant │ 가장 작고 ... <|im_end|>
loss:  ─────────────────────────────── 0 (labels = -100) ───────────────────────────────────────────── │ ████ 학습 ████████
```

질문까지 학습하면 모델이 **질문을 만들어 내는 법**까지 배우느라 용량을 쓴다. 그리고 질문이 길고 답이 짧은 데이터에서는 loss 의 대부분이 질문에서 나와 답변 학습이 약해진다.
이를 방지하기 위해서 prompt 의 labels 를 `-100` 으로 설정한다.


### 4. 지시 데이터 설계 ###

모델에게 질문하고 대답하는 법(SFT)을 가르치기 위해, 06_build_sft.py 스크립트는 네 가지 자료를 모아 대화 형식([system, user, assistant])으로 조립한다.

구체적으로는 핵심 용어의 짧은 정의를 묻는 데이터(312개), 명령어와 코드가 포함된 긴 매뉴얼 설명(최대 1,500개), 일상 대화 능력을 유지하기 위한 기존 한국어 대화 데이터(kullm-v2, 1,200개)를 기본으로 섞는다. 여기에 필요시 성능이 좋은 큰 AI가 문서를 읽고 생성한 다양한 가상 질의응답을 추가한다.

이 데이터를 설계할 때는 아래 기준을 적용하였다.
* 모델이 딱딱한 매뉴얼 말투('~한다')에만 물들지 않도록 긴 문서 설명은 1,500개로 상한선을 두었다.
* 질문의 자연스러움을 높이기 위해 단어의 받침 유무를 계산하여 '파드가', '컨피그맵이'처럼 한국어 조사를 정확하게 맞추었다.
* 일상 대화 능력을 잃지 않도록 일반 대화 데이터도 함께 학습시켰다.
* 전체 용어 122개 중 18개(15%)는 이번 학습에서 의도적으로 제외하엿다. 모델이 이 18개의 질문에도 잘 대답한다면, 단순히 문답을 외운 것이 아니라 앞 단계에서 배운 지식을 스스로 꺼내어 활용하는 법을 터득했다는 증거이다.


`데이터 만들기`
```bash
cd ~/llm-ft && source .venv/bin/activate
python 06_build_sft.py
```

`데이터 확인`
```python
import json, random
rows = [json.loads(l) for l in open("data/sft/train.jsonl", encoding="utf-8")]
for r in random.sample(rows, 3):
    for m in r["messages"]:
        print(f"[{m['role']}] {m['content'][:300]}\n")
    print("-" * 60)
```

### 5 (선택) 합성 질의응답으로 데이터 품질 올리기 ####

템플릿 질문은 단조롭고, 섹션 원문은 질문에 딱 맞는 답이 아니다. 실무에서는 **큰 LLM 으로 청크마다 실제 사용자가 할 법한 질문과
근거 있는 답을 합성**한다. g6e 에서는 학습 전에 vLLM 으로 7B instruct 모델을 띄워서 만들 수 있다.

```bash
# 터미널 1: vLLM (별도 가상환경, 모듈 7.4 참고). Qwen2.5-7B-Instruct 는 Apache-2.0
source .venv-vllm/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.9

# 터미널 2
cd ~/llm-ft && source .venv/bin/activate
python synth_qa.py --limit 50                      # 먼저 50개 청크로 품질 확인 → data/synth_qa.jsonl
python synth_qa.py                                 # 전체 (이미 만든 청크는 건너뜀)
python 06_build_sft.py --synth data/synth_qa.jsonl
# 끝나면 터미널 1 의 vLLM 을 Ctrl+C 로 꺼서 GPU 를 비운다 (학습과 같이 못 씀)
```

- 프롬프트(`synth_qa.py` 의 `PROMPT`)는 **"본문에 있는 사실만"**, **"이 문서 같은 표현 금지"**, **"번역어를 따를 것"** 을 요구한다.
- 합성 데이터도 **사실을 지어낼 수 있다.** 샘플을 뽑아 원문(`source`, `heading`)과 대조하는 검수 단계를 꼭 둔다.
- 평가용 held-out 용어 문서에서 나온 질문은 `06_build_sft.py` 가 자동으로 뺀다.
- 생성에 쓴 LLM 의 **이용 약관**을 확인한다. 상용 API 중에는 출력으로 다른 모델을 학습시키는 것을 금지하는 곳이 있다.


### 6. 학습 ###

```bash
# 터미널에서 직접 실행할 때
nohup python train_sft_hf.py --full > logs/train_sft.log 2>&1 &
tail -f logs/train_sft.log
```

### 7. 평가 ###

SFT 학습이 끝나면 노트북의 평가 셀을 실행한다.

```bash
python 07_eval_sft.py      # INSTRUCT / CHATVEC / SFT. 경로가 없는 모델은 건너뜀 → logs/sft_eval.json
```

| 지표 | 무엇을 재나 | 좋은 결과 |
|---|---|---|
| `ppl glossary_heldout` | SFT 에 없던 용어 18개의 **정답 정의**를 답변으로 낼 확률 (답변 토큰만, 모듈 5 의 perplexity 와 같은 방식) | INSTRUCT 보다 크게 낮음 |
| `ppl general_heldout` | 학습에 안 쓴 일반 지시문 50개의 정답 답변 | INSTRUCT 와 비슷 (망각 없음) |
| `stop rate` | 생성이 `max_new_tokens` 전에 `<|im_end|>` 로 끝난 비율 | 100% 에 가까움 |
| 생성 결과 | 실무 질문 8개, 일반 질문 4개, held-out 용어 6개를 나란히 출력 | 직접 읽고 판단 (과제 6-2) |

결과를 기록해 두자.

| 지표 | INSTRUCT | CHATVEC | SFT |
|---|---|---|---|
| ppl glossary_heldout | | | |
| ppl general_heldout | | | |
| stop rate | | | |

해석 가이드:

| 관찰 | 의미 | 다음 행동 |
|---|---|---|
| glossary ppl: SFT ≪ INSTRUCT, general ppl ≈ | ✅ 이상적. 지식은 꺼내 쓰고 대화 능력은 유지 | 모듈 7 (RAG, 서빙) |
| glossary ppl 은 낮은데 답변이 문서를 길게 복사함 | section 원문 답변의 형식을 따라 함 | `--max-sections` ↓, 합성 데이터(6.5) |
| general ppl 이 크게 ↑, 일반 질문에도 문서체 | 도메인 데이터에 치우침 (SFT 의 망각) | `--n-general` ↑, 학습률 ↓ |
| stop rate 가 낮고 `user` 턴을 지어냄 | EOS 설정 문제 | 6.2 (3), 부록 A |
| CHATVEC 이 SFT 보다 대화가 자연스러움 | 우리 SFT 데이터가 Qwen instruct 데이터보다 단조로움 | 정상. 합성 데이터나 Chat Vector + SFT 조합 |

> **ppl 지표의 한계:** 정답이 용어집 원문 한 가지뿐이다. 맞는 내용을 다른 말로 설명하면 ppl 이 높게 나온다.
> 그래서 SFT 처럼 **용어집 문체로 학습한 모델에 유리**하다. 숫자와 함께 생성 결과를 반드시 읽는다.



---
← [모듈 5](05_evaluation.md) · 다음: [모듈 7. 다음 단계와 정리 →](07_next_steps.md)
