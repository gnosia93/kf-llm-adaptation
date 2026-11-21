## Instruction Tuning ##

### 1 지금 모델의 한계 ###

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

### 3. loss 계산 ###

```
토큰:  <|im_start|>system ... <|im_end|> <|im_start|>user 파드가 뭐야? <|im_end|> <|im_start|>assistant │ 가장 작고 ... <|im_end|>
loss:  ─────────────────────────────── 0 (labels = -100) ───────────────────────────────────────────── │ ████ 학습 ████████
```

질문까지 학습하면 모델이 **질문을 만들어 내는 법**까지 배우느라 용량을 쓴다. 그리고 질문이 길고 답이 짧은 데이터에서는 loss 의 대부분이 질문에서 나와 답변 학습이 약해진다.
이를 방지하기 위해서 prompt 의 labels 를 `-100` 으로 설정한다.


## 6.3 지시 데이터 설계

`06_build_sft.py` 가 네 가지 소스를 `{"messages": [system, user, assistant]}` 형식으로 합친다.

| 소스 | 만드는 방법 | 개수 (실측) | 가르치는 것 |
|---|---|---|---|
| **glossary** | 용어집 정의 122개 → 용어당 질문 3개 (`"{용어}가 뭐야?"` 등 6가지 템플릿) | 312 | 짧고 정확한 정의형 답변 |
| **section** | 문서 섹션 → 템플릿 질문 + 섹션 원문 (모듈 3 의 `02_prepare_data.py` 재사용) | 1,500 (상한) | 명령어·YAML 이 들어간 긴 설명 |
| **general** | `nlpai-lab/kullm-v2` 의 alpaca·dolly 부분 | 1,200 | 일반 대화 능력 유지 (**SFT 의 replay**) |
| synth (선택) | 큰 LLM 이 청크마다 질문·답변 생성 (6.5) | 청크당 최대 3 | 실제 사용자 같은 다양한 질문 |

설계할 때 고려한 점:

- **답변 품질이 곧 모델 품질이다.** SFT 는 답변을 그대로 따라 하도록 학습한다. 섹션 원문 답변은 문서체("~한다")이고
  질문과 딱 맞지 않을 때도 있다. 그래서 개수에 상한(`--max-sections 1500`)을 둬서 이런 답변이 전체 형식을 지배하지 않게 한다.
- **조사 처리.** `"{용어}이(가) 뭐야?"` 같은 어색한 질문 대신 받침을 계산해 `파드가`, `컨피그맵이`, `잡을` 처럼 만든다
  (`06_build_sft.py` 의 `has_batchim`). 질문 품질도 데이터 품질이다.
- **일반 지시 데이터 replay.** 쿠버네티스 질의응답만 학습하면 "김치찌개 끓이는 법"에도 문서체로 쿠버네티스처럼 답한다.
  모듈 1.6 의 replay 를 SFT 에도 똑같이 적용한다. vicuna 부분은 여러 턴 대화를 한 턴씩 쪼갠 것이라 문맥이 끊겨서 쓰지 않고,
  다른 모델의 자기소개("OpenAI 가 만든 언어 모델로서...")가 들어간 답변은 걸러 낸다.
- **평가용 held-out (가장 중요).** 용어 122개 중 **18개(15%)는 SFT 에서 통째로 뺀다.** 이 용어들은 CPT 에서는 봤지만
  질의응답 형식으로는 한 번도 보지 못했다. 이 질문에 잘 답하면 SFT 가 **CPT 지식을 꺼내 쓰는 법**을 가르쳤다는 뜻이다.

> **라이선스:** kullm-v2 는 Apache-2.0 으로 공개되어 있다. 하지만 원본은 Alpaca(OpenAI 모델 출력)와 Dolly(CC BY-SA 3.0)를
> 번역한 것이다. 연구·교육용으로는 문제가 적지만, 상업용 모델에 쓸 때는 원본 데이터의 조건을 확인한다.

## 6.4 실습: 데이터 만들기

노트북 **4. 지시 튜닝** 의 첫 셀을 실행한다. 터미널에서는 다음과 같다.

```bash
cd ~/llm-ft && source .venv/bin/activate
python 06_build_sft.py
```

```
용어집 122개 (평가용으로 18개 제외)
구성: glossary 312, section 1500, section(valid) 108, general 1200, general(valid) 30
train 3012 / valid 138 -> data/sft/
평가 80개 -> data/eval/sft_eval.jsonl  {'glossary_heldout': 18, 'general_heldout': 50, 'practical': 8, 'practical_general': 4}
```

(위 숫자는 Mac 에서 같은 기본값으로 실측했다. general 은 kullm-v2 를 스트리밍하므로 첫 실행에 수십 초가 걸린다.)

> `02_prepare_data.py` 도 `data/sft/` 를 만든다 (섹션 템플릿 질문만). `06_build_sft.py` 가 그 결과를 덮어쓴다.
> Mac 의 `config_lora.yaml`, 간단 버전 `train_hf.py` 도 같은 형식이라 그대로 쓸 수 있다.

✅ **확인:** 샘플을 하나씩 직접 읽어 보자. 학습 전에 데이터를 눈으로 보는 습관이 가장 싸고 효과적인 디버깅이다.

```python
import json, random
rows = [json.loads(l) for l in open("data/sft/train.jsonl", encoding="utf-8")]
for r in random.sample(rows, 3):
    for m in r["messages"]:
        print(f"[{m['role']}] {m['content'][:300]}\n")
    print("-" * 60)
```

## 6.5 (선택) 합성 질의응답으로 데이터 품질 올리기

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

## 6.6 학습 코드 읽기 (`train_sft_hf.py`)

구조는 모듈 4 의 `train_cpt_hf.py` 와 거의 같다. 다른 부분만 보자.

```python
def to_prompt_completion(example):
    msgs = example["messages"]
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}     # 마지막 assistant 만 completion

config = SFTConfig(
    completion_only_loss=True,      # prompt(system+user) 토큰은 loss 에서 제외
    max_length=2048,                # 답변이 1,800자 이하라 2K 면 충분 (넘으면 뒷부분이 잘림)
    num_train_epochs=3,
    per_device_train_batch_size=4, gradient_accumulation_steps=4,    # 실효 배치 16
    learning_rate=1e-5 if args.full else 1e-4,
    warmup_steps=0.03,              # 1 미만이면 비율
    ...
)
```

| | CPT (모듈 4) | SFT (이번 모듈) | 이유 |
|---|---|---|---|
| 시작 모델 | `Qwen/Qwen2.5-3B` | `models/k8s-ko-cpt-hf/final` | CPT 지식 위에 형식을 얹는다 |
| loss | 모든 토큰 | 답변 토큰만 | 6.2 (2) |
| 학습률 (full) | 2e-5 | **1e-5** | 형식만 가르치고 CPT 지식은 덜 건드린다 |
| epoch | 4 | 3 | 데이터가 작다. 많이 돌리면 답변을 통째로 외운다 |
| 시퀀스 | 8,192 x 배치 1 | 2,048 x 배치 4 | 샘플이 짧다. padding 은 배치 안에서만 생긴다 |
| 저장 | 모델 | 모델 + **EOS 에 `<|im_end|>`** | 6.2 (3) |

full 모드 메모리는 모듈 4.3 과 같은 계산이다 (fp32 가중치 12 + 기울기 12 + 8bit Adam 6 = 30GB + 활성값).
시퀀스가 짧아서 활성값은 CPT 보다 작다.

## 6.7 실습: 학습 실행

노트북 **4-1. SFT 학습** 셀을 실행한다 (CPT 학습이 끝나 `models/k8s-ko-cpt-hf/final` 이 있어야 한다).

```bash
# 터미널에서 직접 실행할 때
nohup python train_sft_hf.py --full > logs/train_sft.log 2>&1 &
tail -f logs/train_sft.log
```

학습 데이터 3,012개 / 실효 배치 16 → 1 epoch 약 188 스텝, 3 epoch 약 565 스텝이다.
**학습 시간은 아직 측정하지 않았다** (리허설 측정표, [강사용 가이드](00_facilitator.md#리허설-측정표)).

loss 곡선 읽는 법:

| 관찰 | 의미 |
|---|---|
| train loss 가 첫 epoch 안에 빠르게 내려감 | 정상. 형식(템플릿, 문체)은 금방 배운다 |
| epoch 경계마다 train loss 가 계단처럼 뚝 떨어짐 | 같은 답변을 다시 보면서 외우기 시작한 것 |
| valid loss 가 오르기 시작 | 과적합. 그 앞의 체크포인트를 쓰거나 `--epochs 2` |

## 6.8 비교 대상: Chat Vector (학습 없이 가중치 산수)

SFT 학습이 도는 동안 비교 대상을 만들어 둔다. *Chat Vector* (Huang 외, 2023) 아이디어:

```
CPT 로 생긴 변화량   Δ = W_cpt − W_base
대화형 + 도메인       W  = W_instruct + Δ
```

```bash
python chat_vector.py                      # → models/k8s-ko-3b-chat
python chat_vector.py --ratio 0.5          # 대화가 어색해지면 변화량을 절반만 더함
```

핵심 코드 (`chat_vector.py`):

```python
base, cpt, inst = load(args.base).state_dict(), load(args.cpt).state_dict(), load(args.instruct)
with torch.no_grad():
    for name, param in inst.named_parameters():
        if any(s in name for s in ("embed_tokens", "lm_head")):   # 임베딩은 instruct 것을 그대로
            continue
        delta = cpt[name].float() - base[name].float()
        param.copy_((param.float() + args.ratio * delta).to(param.dtype))
```

- **같은 계열·같은 크기**여야 한다: `Qwen2.5-3B` / `Qwen2.5-3B-Instruct` / CPT 결과물.
- 계산은 fp32 로 한다. bf16 끼리 빼면 작은 변화량이 사라진다 (모듈 4.2 와 같은 이유).
- 검증: 0.5B 모델로 Δ=0 (cpt=base) 을 넣었을 때 결과가 instruct 모델과 **정확히 일치**하는 것을 확인했다.
- 메모리: 3B 모델 3개를 CPU RAM 에 올린다 (약 18GB). g6e 의 128GB 로 충분하다. GPU 는 쓰지 않아서 SFT 학습과 동시에 돌려도 된다.

| | SFT (A) | Chat Vector (B) |
|---|---|---|
| 비용 | GPU 학습 (수십 분) | CPU 몇 분 |
| 대화 품질 | 우리 데이터 수준 | Qwen 팀의 instruct 학습 (SFT + RLHF) 수준 |
| 답변 형식 제어 | 데이터로 원하는 대로 | 못 함 |
| 위험 | 데이터가 나쁘면 그대로 배움 | Δ 와 instruct 변화량이 충돌하면 어색해짐 |

실무에서는 둘을 합치기도 한다: **Chat Vector 모델 위에 소량의 SFT.** (과제 6-5)

## 6.9 평가: 세 모델 비교

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

## 과제

### 과제 6-1. loss 가 어디에 걸리는지 눈으로 확인하기

TRL 이 실제로 만드는 labels 를 디코딩해 본다. `-100` 인 부분은 loss 에서 빠진다.

```python
import json
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
row = json.loads(open("data/sft/train.jsonl", encoding="utf-8").readline())["messages"]
ds = Dataset.from_list([{"prompt": row[:-1], "completion": row[-1:]}])
trainer = SFTTrainer(model="Qwen/Qwen2.5-0.5B", train_dataset=ds, processing_class=tok,
                     args=SFTConfig(output_dir="/tmp/sft-mask", max_length=512, report_to="none"))
batch = trainer.data_collator([trainer.train_dataset[0]])
ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
print("[loss 없음]", tok.decode([i for i, l in zip(ids, labels) if l == -100]))
print("[loss 있음]", tok.decode([l for l in labels if l != -100]))
```

1. `<|im_start|>assistant\n` 은 어느 쪽에 들어가나? 답변 끝의 `<|im_end|>` 는?
2. 답변 끝의 `<|im_end|>` 가 loss 에서 빠졌다면 학습된 모델은 어떻게 행동할까?

### 과제 6-2. 답변 읽고 채점하기

`07_eval_sft.py` 출력에서 실무 질문 8개를 골라 모델마다 표시해 보자. 사실 확인은 과제 5-2 의 `find_in_docs` 를 쓴다.

| 질문 | 질문에 맞게 답했나 | 공식 번역어를 쓰나 | 사실이 맞나 | 길이·형식이 적당한가 |
|---|---|---|---|---|
| 파드 | | | | |
| 디플로이먼트 vs 스테이트풀셋 | | | | |
| ... | | | | |

### 과제 6-3. 종료 토큰 실험

SFT 모델에서 `<|im_end|>` 를 멈춤 토큰에서 빼고 생성해 보자.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "models/k8s-ko-sft-hf/final"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
msgs = [{"role": "user", "content": "파드가 뭐야?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
for eos in ([151645, 151643], [151643]):          # <|im_end|> 포함 / base 모델 기본값
    out = model.generate(**ids, max_new_tokens=200, do_sample=False, eos_token_id=eos)
    print(eos, repr(tok.decode(out[0, ids["input_ids"].shape[1]:])), "\n")
```

두 번째 결과에서 `<|im_end|>` 뒤에 무엇이 나오나? 서빙할 때 이 문제가 생기면 사용자에게는 어떻게 보일까?

### 과제 6-4. 대조 실험: CPT 는 정말 도움이 됐나?

같은 SFT 데이터로 **CPT 를 하지 않은** base 모델을 학습해 비교한다.

```bash
nohup python train_sft_hf.py --full --model Qwen/Qwen2.5-3B --output models/k8s-ko-base-sft-hf \
  > logs/train_base_sft.log 2>&1 &
# 끝나면
python 07_eval_sft.py --models SFT=models/k8s-ko-sft-hf/final BASE_SFT=models/k8s-ko-base-sft-hf/final
```

`glossary_heldout` 에서 차이가 나야 CPT 의 효과다. 학습한 용어(SFT 데이터에 있는 용어)로 물어봤을 때는 차이가 작을 수 있다. 왜일까?

### 과제 6-5 (선택). 데이터 구성 바꿔 보기

하나씩 바꿔서 다시 학습하고 `07_eval_sft.py` 결과를 비교한다. 각 학습은 `--output` 을 다르게 준다.

| 실험 | 명령 | 볼 지표 |
|---|---|---|
| replay 없음 | `python 06_build_sft.py --n-general 0` | general ppl, 일반 질문 답변 |
| 섹션 원문 줄이기 | `python 06_build_sft.py --max-sections 300` | 답변 길이·형식 |
| 합성 데이터 추가 | 6.5 | 실무 질문 답변 |
| Chat Vector 위에 SFT | `python train_sft_hf.py --full --model models/k8s-ko-3b-chat --output models/k8s-ko-cv-sft-hf` | 전부 |

(정답과 해설은 [강사용 가이드](00_facilitator.md#모듈-6) 에 있다.)

---
← [모듈 5](05_evaluation.md) · 다음: [모듈 7. 다음 단계와 정리 →](07_next_steps.md)
