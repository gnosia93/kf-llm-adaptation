## 선호 학습: DPO ##

### 1. SFT 다음에 남은 문제 ###

SFT 는 **정답을 따라 쓰는 것**만 배운다. "이 답이 저 답보다 낫다" 는 신호는 없다.
SFT 모델에서 흔히 보이는 문제:

| 증상 | 예 | SFT 로 잘 안 고쳐지는 이유 |
|---|---|---|
| 그럴듯한 환각 | 없는 `kubectl` 플래그, 틀린 기본값 | 정답 데이터에는 "틀린 답" 이 없다. 무엇을 하면 안 되는지 모른다 |
| 문서 덤프 | 질문과 상관없는 섹션을 길게 옮겨 적음 | section 원문 답변을 따라 하도록 배웠다 |
| 반복·끊김 | 같은 문장 반복, 중간에 끊김 | 정답 하나만 보면 이런 실패를 피하는 법을 배우지 못한다 |

**선호 학습** 은 같은 질문에 대한 답 두 개를 보여주고 "이쪽이 낫다" 를 가르친다.
모델이 **실제로 저지르는 실수** 를 rejected 로 보여주면, 그 실수의 확률을 직접 낮춘다.

```
            ┌────────── A. DPO: 선호 쌍으로 바로 정책 학습 (분류 문제처럼) ──────────┐
SFT 모델 ───┤                                                                     ├──▶ 선호에 맞춘 모델
            └── B. RLHF: 선호 쌍 → 보상 모델 학습 → PPO 로 보상을 높이도록 강화학습 ─┘
```

B 가 InstructGPT·초기 ChatGPT 의 방식이고, A 는 B 와 같은 목표를 강화학습 없이 푸는 방법이다 (Rafailov 외, 2023).

### 2. 선호 데이터 만들기 ###

* https://github.com/gnosia93/kf-llm-pipeline/blob/main/src/dpo/build_pref.py

#### (1) 파이프라인 ####

![](https://github.com/gnosia93/kf-llm-pipeline/blob/main/images/dpo-data-generation.jpg)


![](https://github.com/gnosia93/kf-llm-pipeline/blob/main/images/dpo-data-generation-flow.jpg)


### 설계시 고려사항 ####

- **on-policy:** 다른 모델의 답이 아니라 **지금 학습할 SFT 모델의 답**으로 쌍을 만든다. 모델이 실제로 하는 실수를 고친다.
  남의 답(예: 큰 모델의 답 vs 작은 모델의 답)으로 만들면 "큰 모델 말투" 를 배우는 데 그칠 수 있다.
- **기준 답변을 심판에게 준다:** 7B 심판도 쿠버네티스 세부 사실은 틀릴 수 있다. SFT 정답(공식 문서에서 온 것)을
  `<기준 답변>` 으로 함께 줘서 사실 판단의 근거로 쓰게 한다 (`build_pref.py` 의 `JUDGE_PROMPT`).
- **위치 편향:** LLM 심판은 앞에 나온 답에 점수를 더 주는 경향이 있다. 답의 순서를 섞어서 보여주고 점수를 원래 순서로 되돌린다.
- **점수 차 기준(`--min-margin`):** 비슷한 점수의 쌍은 잡음이다. 차이가 뚜렷한 쌍만 쓴다.
- **길이 편향 확인:** 심판은 긴 답을 선호하기 쉽다. 결과 출력의 `평균 길이 chosen / rejected` 를 꼭 본다.
  chosen 이 훨씬 길면 모델은 "길게 쓰면 좋다" 를 배운다.
- **심판 없이 할 때(`--judge heuristic`):** 기준 답변과의 글자 겹침 + 종료·반복 감점으로 점수를 매긴다. 동작 확인용 대용품이다.
  겹침 점수는 "문서를 많이 베낄수록 좋다" 를 가르칠 수 있다.

```bash
# 터미널 1: 심판 (vLLM, 모듈 8.4 의 별도 가상환경). SFT 모델과 GPU 를 나눠 쓰도록 메모리를 55% 로 제한
source .venv-vllm/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.55

# 터미널 2
cd ~/llm-ft && source .venv/bin/activate
python build_pref.py                     # generate → judge → pairs
# 끝나면 터미널 1 의 vLLM 을 Ctrl+C 로 끈다 (학습에 GPU 필요)
```

chosen / rejected 쌍을 확인한다 (점수도 확인)

```python
import json, random
pairs = [json.loads(l) for l in open("data/pref/train.jsonl", encoding="utf-8")]
for p in random.sample(pairs, 3):
    print("Q:", p["prompt"][1]["content"])
    print(f"\n[chosen {p['score_chosen']}]\n{p['chosen'][0]['content'][:500]}")
    print(f"\n[rejected {p['score_rejected']}]\n{p['rejected'][0]['content'][:500]}\n" + "=" * 60)
```

심판 기준을 바꾸고 싶으면 `JUDGE_PROMPT` 를 고친 뒤 `python build_pref.py --stage judge` 로 **채점만 다시** 한다.

### 3. DPO (Direct Preference Optimization) ###
* https://github.com/gnosia93/kf-llm-pipeline/blob/main/src/dpo/train_dpo_hf.py
  
RLHF 의 목표는 "보상은 높이되, 참조 모델(SFT)에서 너무 멀어지지 않기" 다. 이 목표의 최적해를 풀어 보면
**정책 자체가 보상 모델 역할을 한다** 는 결론이 나온다. 그래서 보상 모델과 강화학습 없이 선호 쌍으로 바로 학습할 수 있다.

```
암묵적 보상   r(y) = β · [ log π(y|x) − log π_ref(y|x) ]      π_ref = 학습 시작 시점의 SFT 모델 (고정)
DPO loss      −log σ( r(chosen) − r(rejected) )
```

- chosen 의 확률은 참조 대비 **올리고**, rejected 의 확률은 참조 대비 **내리도록** 학습한다.
- **β** 는 참조 모델에 묶어 두는 강도다. 크면(0.3 ~ 0.5) 보수적, 작으면(0.01 ~ 0.05) 선호를 강하게 따르지만 망가지기 쉽다. 기본 0.1.
- 참조 모델이 필요하지만, LoRA 를 쓰면 **어댑터를 끈 같은 모델**이 참조 모델이다. 메모리가 추가로 들지 않는다.

```bash
nohup python train_dpo_hf.py > logs/train_dpo.log 2>&1 &
```

#### 로그 읽는 법 ####
* `rewards/margins` : r(chosen) − r(rejected) 평균, 0 에서 시작해 점점 커짐 
* `rewards/accuracies` : r(chosen) > r(rejected) 인 비율, 0.5 에서 시작해 0.7~0.9 
* `rewards/chosen` : chosen 의 암묵적 보상으로 약간 음수까지는 흔함. 크게 음수면 chosen 확률까지 같이 떨어지는 중 
* `eval_loss` : valid 쌍의 DPO loss, 0.693(= log 2) 에서 내려감. 다시 오르면 과적합 

