## Kubeflow Pipelines (KFP v2) ##

**데이터 전처리 → CPT(연속 사전학습) → SFT(지도 미세조정) → DPO(선호도 정렬) → vLLM 무중단 추론 배포** 로 이어지는 3B 모델 파이프라인을 **Kubeflow Pipelines (KFP v2)**로 자동화하고, 각 단계 사이에 **자동화된 품질 관문(Quality Gate)**, **MLflow 계층형 트래킹 및 Model Registry Alias 배포**, 그리고 **실패 시 Slack 알림(`dsl.ExitHandler`)** 발송을 하는 방법에 대해서 다룬다.

### 1. 전체 파이프라인 및 Quality Gate 아키텍처 ###

핵심 설계 원칙은 **학습(Train) 컴포넌트와 평가(Eval/Gate) 컴포넌트를 물리적으로 분리** 하는 것이다. 이를 통해 KFP의 **Step Caching** 기능을 극대화하여, Gate 임계값이나 평가 기준을 수정할 때 무거운 GPU 학습을 다시 돌리지 않고 평가 단계만 1분 내에 재실행할 수 있다.

```text
[dsl.ExitHandler: 파이프라인 종료 시 성공/실패 및 실패 Gate 자동 Slack 알림]
 │
[Parent MLflow Run: qwen25-3b-pipeline-v1]
 │
 ├─▶ [1. Data Pre-processing] ──▶ [Gate 1: 중복률/토큰분포/PII] ──(FAIL)──▶ [즉시 중단 & Slack 알림]
 │                                        │ (PASS: dsl.If)
 ├─▶ [2. Train CPT (3B)] ───────▶ [Eval & Gate 2: Held-out PPL] ──(FAIL)──▶ [즉시 중단 & Slack 알림]
 │                                        │ (PASS: dsl.If)
 ├─▶ [3. Train SFT (LoRA)] ─────▶ [Eval & Gate 3: Stop Rate/포맷] ─(FAIL)──▶ [즉시 중단 & Slack 알림]
 │                                        │ (PASS: dsl.If)
 ├─▶ [4. Train DPO & Merge] ────▶ [Eval & Gate 4: Judge 승률] ────(FAIL)──▶ [즉시 중단 & Slack 알림]
 │                                        │ (PASS: dsl.If)
 └─▶ [5. MLflow Registry '@challenger' 등록] ──▶ [6. vLLM InitContainer 무중단 롤링 배포('@champion')]
```

---

## 2. 스테이지별 Quality Gate 검증 기준표 (3B 모델 권장)

| 관문 (Gate) | 평가 대상 | 핵심 측정 지표 (MLflow 기록) | 통과 조건 (Threshold) | 방어하는 실패 유형 |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1 (Data)** | 정제된 데이터셋 | • 중복률 (`dup_ratio`)<br>• 유효 샘플 수 (`valid_count`)<br>• 개인정보/금지어 검출 수 | • `dup_ratio < 0.05`<br>• `valid_count >= 10000`<br>• `pii_count == 0` | 오염된 데이터로 인한 시간·GPU 낭비 사전 차단 |
| **Gate 2 (CPT)** | CPT 완료 모델 | • 도메인 Held-out PPL<br>• 일반 상식(General) PPL | • 도메인 PPL **15% 이상 감소**<br>• 일반 PPL 상승폭 **< 5%** | 도메인 지식 주입 실패 또는 파국적 망각(Catastrophic Forgetting) |
| **Gate 3 (SFT)** | SFT 완료 모델 | • 종료 토큰 비율 (`stop_rate`)<br>• 평균 응답 길이 (`avg_len`) | • `stop_rate >= 0.95`<br>• `20 <= avg_len <= 480` | `<\|im_end\|>` 미생성으로 인한 무한 반복 생성 폭주 방지 |
| **Gate 4 (DPO)** | 병합된 최종 모델 | • SFT 대비 Judge 승률 (`win_rate`)<br>• SFT 대비 응답 길이 증가율 | • `win_rate >= 0.55`<br>• 길이 증가폭 **< 20%** | 정렬 실패 또는 '길게 쓰기만 하는 편향(Verbosity Bias)' 차단 |

---

## 3. 실전 파이썬 구현 코드 (KFP v2 + 전체 Gate 평가 로직 + MLflow)

### ① Gate 2, Gate 3, Gate 4 평가 컴포넌트 (`components/eval_gates.py`)

```python
from kfp import dsl
from kfp.dsl import Input, Model, Dataset

BASE_IMAGE = "my-registry/llm-pipeline-3b:v1"  # torch, transformers, peft, trl, mlflow, openai 포함

# ------------------------------------------------------------------------------
# Gate 2: CPT Perplexity 검증 (도메인 PPL 하락 & 일반 상식 망각 방지)
# ------------------------------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE)
def eval_cpt_gate(
    base_model_id: str,
    cpt_model: Input[Model],
    eval_dataset: Input[Dataset],
    mlflow_uri: str,
    parent_run_id: str,
    max_general_ppl_increase: float = 0.05,
) -> str:
    import math
    import torch
    import mlflow
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mlflow.set_tracking_uri(mlflow_uri)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def compute_ppl(model_path: str, texts: list[str]) -> float:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device).eval()
        losses = []
        with torch.no_grad():
            for t in texts:
                ids = tok(t, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
                if ids.size(1) < 2:
                    continue
                losses.append(model(ids, labels=ids).loss.item())
        del model
        torch.cuda.empty_cache()
        return math.exp(sum(losses) / max(1, len(losses)))

    ds = load_from_disk(eval_dataset.path)
    domain_texts = ds["domain_heldout"]["text"][:100]
    general_texts = ds["general_heldout"]["text"][:100]

    base_domain_ppl = compute_ppl(base_model_id, domain_texts)
    cpt_domain_ppl = compute_ppl(cpt_model.path, domain_texts)
    base_gen_ppl = compute_ppl(base_model_id, general_texts)
    cpt_gen_ppl = compute_ppl(cpt_model.path, general_texts)

    domain_drop_ratio = (base_domain_ppl - cpt_domain_ppl) / base_domain_ppl
    gen_inc_ratio = (cpt_gen_ppl - base_gen_ppl) / base_gen_ppl

    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="gate_2_cpt_eval", nested=True):
            mlflow.log_metrics({
                "base_domain_ppl": base_domain_ppl,
                "cpt_domain_ppl": cpt_domain_ppl,
                "domain_ppl_drop_ratio": domain_drop_ratio,
                "general_ppl_inc_ratio": gen_inc_ratio,
            })
            passed = (domain_drop_ratio >= 0.15) and (gen_inc_ratio <= max_general_ppl_increase)
            status = "PASS" if passed else "FAIL"
            mlflow.set_tag("gate_2_status", status)
            return status


# ------------------------------------------------------------------------------
# Gate 3: SFT 종료 토큰(stop_rate) 및 응답 형식 검증
# ------------------------------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE)
def eval_sft_gate(
    sft_model: Input[Model],
    eval_dataset: Input[Dataset],
    mlflow_uri: str,
    parent_run_id: str,
    min_stop_rate: float = 0.95,
) -> str:
    import torch
    import mlflow
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mlflow.set_tracking_uri(mlflow_uri)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(sft_model.path)
    model = AutoModelForCausalLM.from_pretrained(
        sft_model.path, torch_dtype=torch.bfloat16
    ).to(device).eval()

    stop_ids = [tok.eos_token_id]
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id != tok.unk_token_id:
        stop_ids.append(im_end_id)

    ds = load_from_disk(eval_dataset.path)["sft_eval"]
    stopped_count = 0
    lengths = []

    with torch.no_grad():
        for row in ds:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": row["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tok(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=stop_ids,
            )
            gen_ids = out[inputs.input_ids.shape:]
            if int(gen_ids[-1].item()) in stop_ids:
                stopped_count += 1
            lengths.append(len(gen_ids))

    stop_rate = stopped_count / max(1, len(ds))
    avg_len = sum(lengths) / max(1, len(lengths))

    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="gate_3_sft_eval", nested=True):
            mlflow.log_metrics({"sft_stop_rate": stop_rate, "sft_avg_gen_len": avg_len})
            passed = (stop_rate >= min_stop_rate) and (20 <= avg_len <= 480)
            status = "PASS" if passed else "FAIL"
            mlflow.set_tag("gate_3_status", status)
            return status


# ------------------------------------------------------------------------------
# Gate 4: DPO LLM-as-a-Judge 승률 및 응답 길이 편향(Verbosity Bias) 검증
# ------------------------------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE)
def eval_dpo_judge_gate(
    sft_model: Input[Model],
    dpo_model: Input[Model],
    eval_dataset: Input[Dataset],
    mlflow_uri: str,
    parent_run_id: str,
    min_win_rate: float = 0.55,
    max_len_increase_ratio: float = 0.20,
) -> str:
    import json
    import random
    import torch
    import mlflow
    from openai import OpenAI
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mlflow.set_tracking_uri(mlflow_uri)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_from_disk(eval_dataset.path)["dpo_eval"].select(range(50))

    def generate_answers(model_path: str, questions: list[str]) -> list[str]:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device).eval()
        answers = []
        with torch.no_grad():
            for q in questions:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tok(prompt, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                ans = tok.decode(out[inputs.input_ids.shape:], skip_special_tokens=True)
                answers.append(ans.strip())
        del model
        torch.cuda.empty_cache()
        return answers

    questions = [row["question"] for row in ds]
    sft_answers = generate_answers(sft_model.path, questions)
    dpo_answers = generate_answers(dpo_model.path, questions)

    # 위치 편향(Position Bias) 방지를 위해 A/B 순서를 무작위로 섞어 LLM 심판관 채점
    client = OpenAI()
    JUDGE_PROMPT = """다음 질문에 대한 두 AI 답변(A, B)의 정확성, 명확성, 환각 여부를 비교 평가하라.
더 우수한 답변을 선택하여 반드시 JSON 형식으로만 출력하라: {{"winner": "A" | "B" | "TIE"}}

[질문] {question}
[답변 A] {ans_a}
[답변 B] {ans_b}"""

    dpo_wins, ties = 0, 0
    for q, sft_ans, dpo_ans in zip(questions, sft_answers, dpo_answers):
        swap = random.choice([True, False])
