# 모듈 5. 평가 — 지식이 들어갔는지 어떻게 아나

**소요 40분** · 강의 10분 + 실습 30분

## 학습 목표

- perplexity 3종으로 **지식 습득, 도메인 일반화, 망각**을 분리해서 측정한다.
- 이어쓰기 프로브로 모델의 변화를 직접 눈으로 확인한다.
- 생성 결과를 **원문과 대조**해서 사실인지 환각인지 판별한다.

---

## 5.1 평가 설계

```
                 원본 Qwen2.5-3B (BASE)      CPT 모델
                 ─────────────────────      ───────────────────
k8s_seen     ─▶  ppl_b(seen)                ppl_c(seen)        → 많이 ↓ 해야 함
k8s_heldout  ─▶  ppl_b(heldout)             ppl_c(heldout)     → 조금 ↓
general_ko   ─▶  ppl_b(general)             ppl_c(general)     → 거의 유지 (오르면 망각)

PROBES (6개)  ─▶  BASE 이어쓰기              CPT 이어쓰기        → 나란히 비교
```

**base 모델은 질문에 답하지 않는다.** 그래서 프로브는 질문 형식이 아니라 **문서 앞부분**이다.

```python
PROBES = [
    "# 파드\n\n파드(Pod)는 쿠버네티스에서",
    "# 디플로이먼트\n\n디플로이먼트(Deployment)는",
    "퍼시스턴트볼륨클레임(PVC)은 사용자의",
    "kube-proxy는 클러스터의 각 노드에서 실행되는",
    "파드의 로그를 확인하려면 다음 명령을 실행한다.\n\n```shell\n",
    "스테이트풀셋(StatefulSet)이 디플로이먼트와 다른 점은",
]
```

## 5.2 perplexity 계산 코드

```python
@torch.no_grad()
def perplexity(model, path):
    nll, n = 0.0, 0
    for line in open(path, encoding="utf-8"):
        ids = tok(json.loads(line)["text"], return_tensors="pt",
                  truncation=True, max_length=MAX_TOKENS).input_ids.to(model.device)   # 학습 블록 상한과 같게
        nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)   # 평균 loss → 합계로 되돌림
        n += ids.shape[1] - 1
    return math.exp(nll / n)                                              # 전체 토큰 평균 → exp
```

- `labels=ids` 를 주면 모델이 내부에서 한 칸씩 밀어서 다음 토큰 예측 loss 를 계산한다.
- 문서마다 길이가 다르다. 그래서 **문서별 ppl 을 평균내지 않고**, 전체 토큰의 loss 합을 토큰 수로 나눈 뒤 `exp` 한다.

## 5.3 생성 설정

```python
out = model.generate(**enc, max_new_tokens=120, do_sample=False,
                     repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
```

| 설정 | 이유 |
|---|---|
| `do_sample=False` (greedy) | 매번 같은 결과가 나와서 BASE 와 CPT 를 공정하게 비교할 수 있음 |
| `repetition_penalty=1.1` | 3B 모델의 greedy 생성은 같은 문장을 반복하기 쉬움 |
| `max_new_tokens=120` | 비교하기에 충분한 길이 |

---

## 5.4 실습

학습이 끝났는지 먼저 확인한다.

```python
!ls models/k8s-ko-cpt-hf/final       # config.json, model-*.safetensors 가 있어야 함
!pgrep -f train_cpt_hf.py || echo "학습 종료"
```

> 학습이 아직 돌고 있을 때 평가를 시작하면 GPU 메모리가 부족해진다. 끝난 뒤에 실행한다.

노트북 **`3. 평가`** 섹션의 두 셀을 실행한다 (모델 2개를 차례로 로드, 수 분 소요).

### 결과 기록표

| perplexity | BASE | CPT | 변화 |
|---|---|---|---|
| k8s_seen | | | |
| k8s_heldout | | | |
| general_ko | | | |

참고로 원본 모델로 샘플 3개만 미리 재 본 값은 k8s 3.43, 위키 5.04 이다 (모듈 1.8).

### 결과 해석 가이드

| 관찰 | 의미 | 다음 행동 |
|---|---|---|
| seen ↓↓, heldout ↓, general ≈ | ✅ 이상적 | 모듈 6 으로 |
| seen ↓↓, heldout ↑ | 외우기만 하고 일반화하지 못함 | epoch 줄이기, 데이터 증강 |
| general 이 크게 ↑ (예: +20% 이상) | 망각 | replay 비율 ↑, 학습률 ↓, epoch ↓, LoRA 고려 |
| seen 이 별로 안 내려감 | 학습 부족 | epoch ↑, 학습률 ↑ |

> 가장 흔한 함정: seen 이 크게 내려간 것만 보고 "지식이 들어갔다"고 결론 내리는 것.
> seen ppl 은 **원래 문장을 그대로 재현하는 능력**을 잰다. 이게 지식을 활용하는 능력과 항상 같지는 않다.
> 그래서 아래 프로브와 사실 검증이 함께 필요하다.

## 과제

### 과제 5-1. 프로브 비교

BASE 와 CPT 의 이어쓰기를 비교해서 각 프로브마다 표시해 보자.

| 프로브 | 문체가 문서체로 바뀌었나 | 공식 번역 용어를 쓰나 (파드, 디플로이먼트...) | 사실이 맞나 |
|---|---|---|---|
| 파드 | | | |
| 디플로이먼트 | | | |
| PVC | | | |
| kube-proxy | | | |
| 로그 명령 | | | |
| 스테이트풀셋 | | | |

### 과제 5-2. 원문과 대조하기 (사실 검증)

생성된 문장이 원문에 있는 내용인지, 지어낸 내용(환각)인지 확인한다.

```python
import subprocess
def find_in_docs(phrase, lang="ko"):
    r = subprocess.run(["grep", "-rn", "--include=*.md", phrase, f"raw/website/content/{lang}/docs"],
                       capture_output=True, text=True)
    print(r.stdout[:2000] or "(원문에 없음)")

find_in_docs("가장 작은 컴퓨팅 단위")       # CPT 출력에서 고른 구절을 넣어 보자
```

- 원문을 **그대로 재현**했나 (암기), **바꿔 말했나** (이해), **원문에 없는 내용**인가 (환각)?
- 숫자, 플래그 이름, API 버전 같은 세부 사항을 특히 확인하자.

### 과제 5-3. 나만의 프로브

학습한 문서에 있는 내용과 **없는 내용**으로 프로브를 2개씩 만들어 보자.

```python
my_probes = [
    "# 잡\n\n잡(Job)은",                          # 문서에 있는 주제
    "# 컨피그맵\n\n컨피그맵(ConfigMap)은",
    "쿠버네티스 1.40 버전에서 새로 추가된 기능은",   # 문서에 없는 미래 정보 → 환각이 나오는지?
    "아이스 아메리카노를 맛있게 만드는 방법은",       # 도메인 밖 → 일반 능력이 유지되나?
]
for label, path in [("BASE", BASE), ("CPT", TUNED)]:
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for p in my_probes:
        print(f"[{label}] {p}{complete(model, p)}\n")
    del model; torch.cuda.empty_cache()
```

- 도메인 밖 프롬프트에서도 CPT 모델이 쿠버네티스 이야기로 빠지지는 않나?

### 과제 5-4 (토론). 이 모델을 서비스에 쓸 수 있을까?

1. 지금 모델에게 "파드가 뭐야?" 라고 물으면 어떻게 될까? 직접 해 보자.
2. 사용자가 질문하는 챗봇으로 만들려면 무엇이 더 필요한가?
3. 운영 환경의 최신 버전 정보처럼 **틀리면 안 되는** 답은 어떻게 보장할까?

---
← [모듈 4](04_training.md) · 다음: [모듈 6. 다음 단계와 정리 →](06_next_steps.md)
