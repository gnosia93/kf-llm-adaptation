# 모듈 6. 다음 단계와 정리

**소요 30분** · 강의 + 시연 (시간이 남으면 실습)

## 학습 목표

- CPT 모델을 **대화형 모델**로 바꾸는 방법 세 가지(SFT, Chat Vector, RAG 결합)를 안다.
- 서빙 방법과 데이터 증강 방향을 안다.
- 실습 리소스를 **깨끗이 정리**한다.

---

## 6.1 지금 모델의 한계

모듈 5 에서 본 것처럼 CPT 모델은 base 모델이라 **이어쓰기만** 한다.
그렇다고 일반 instruct 모델에 바로 물어보면 도메인 지식이 없다. 실습 준비 중에 소형 모델로 실제로 물어본 결과:

```
Q: 파드가 뭐야? 한 문장으로.
Qwen2.5-0.5B-Instruct: 파드는 컴퓨터의 메모리 공간을 의미하는 단어입니다.     ← 환각
```

필요한 것은 **CPT 로 넣은 지식 + 대화 능력** 의 결합이다.

```
                    ┌──────────── A. SFT ─────────────┐
CPT 모델 (지식) ────┤                                  ├──▶ 지식 + 대화
                    └── B. Chat Vector (학습 없음) ────┘
                                        +
                    C. RAG: 질문할 때 원문을 찾아서 같이 넣어 줌 → 정확성
```

## 6.2 경로 A — CPT 모델 위에 SFT

```bash
# 원격 터미널 (g6e)
cd ~/llm-ft && source .venv/bin/activate
python 02_prepare_data.py                         # data/sft/{train,valid}.jsonl (템플릿 질문 3,088개)
python train_hf.py --model models/k8s-ko-cpt-hf/final \
  --output adapters/k8s-ko-cpt-sft --merged models/k8s-ko-cpt-sft
```

- `train_hf.py` 는 LoRA SFT 를 한다. 대화를 prompt/completion 으로 나눠서 **답변 토큰에만** loss 를 준다.
- Qwen2.5 base 토크나이저에도 chat template 이 들어 있다 (확인함). 그래서 CPT 모델에 바로 SFT 할 수 있다.
- `02_prepare_data.py` 의 질문은 템플릿(`"{title}의 {heading} 부분을 정리해 주세요."`)으로 만든 단순한 것이다.
  실무에서는 **큰 LLM 으로 청크마다 실제 사용자가 할 법한 질문과 답을 합성**하면 품질이 크게 오른다 (6.5 참고).

## 6.3 경로 B — Chat Vector (학습 없이 가중치 산수)

*Chat Vector* (Huang 외, 2023) 아이디어:

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
- 메모리: 3B 모델 3개를 CPU RAM 에 올린다 (약 18GB). g6e 의 128GB 로 충분하다.

대화 테스트:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "models/k8s-ko-3b-chat"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
msgs = [{"role": "user", "content": "파드가 뭐야? 두세 문장으로 설명해줘."}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
out = model.generate(**enc, max_new_tokens=200, do_sample=False, repetition_penalty=1.1)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

`Qwen/Qwen2.5-3B-Instruct` 원본에도 같은 질문을 해서 비교해 보자.

## 6.4 경로 C — RAG 맛보기

3B 모델은 CPT 를 해도 세부 사실(플래그, 버전, 기본값)을 틀릴 수 있다.
**질문과 관련된 원문을 찾아 프롬프트에 함께 넣으면** 정확도가 크게 오른다.

`02_prepare_data.py` 가 만든 `data/chunks.jsonl` (3,184개, 출처 포함)을 그대로 검색 색인으로 쓴다.

```bash
uv pip install sentence-transformers
```

```python
import json, numpy as np
from sentence_transformers import SentenceTransformer

chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
emb = SentenceTransformer("intfloat/multilingual-e5-small")          # 다국어 임베딩 (약 470MB)
vecs = emb.encode([f"passage: {c['title']} {c['heading']}\n{c['text']}" for c in chunks],
                  normalize_embeddings=True, batch_size=64)

def search(q, k=3):
    qv = emb.encode([f"query: {q}"], normalize_embeddings=True)[0]
    return [(float(vecs[i] @ qv), chunks[i]) for i in np.argsort(-(vecs @ qv))[:k]]
```

실제 검색 결과 (Mac 에서 색인 생성까지 약 20초, 모델 다운로드 제외):

```
Q: 파드의 로그를 보려면 어떻게 하나요?
  0.887  tasks/debug/debug-application/debug-running-pod.md | 동작 중인 파드 디버그 > 파드의 로그 확인하기
  0.866  reference/glossary/pod.md | 파드 (원문, Pod)
  0.862  concepts/scheduling-eviction/assign-pod-node.md | 노드에 파드 할당하기 > ... > 파드 간 어피니티 및 안티-어피니티 종류
Q: 스테이트풀셋과 디플로이먼트 차이
  0.885  reference/glossary/statefulset.md | 스테이트풀셋 (원문, StatefulSet)
  0.877  concepts/workloads/controllers/statefulset.md | 스테이트풀셋
  0.876  concepts/workloads/controllers/statefulset.md | 스테이트풀셋 > 스테이트풀셋 사용
```

검색 결과를 대화형 모델(6.2 또는 6.3 의 결과)에 넣는다:

```python
def rag_answer(q, model, tok, k=3):
    ctx = "\n\n".join(f"[출처: {c['source']}]\n{c['text']}" for _, c in search(q, k))
    msgs = [
        {"role": "system", "content": "아래 쿠버네티스 문서만 근거로 한국어로 답하고, 근거가 없으면 모른다고 답하세요."},
        {"role": "user", "content": f"{ctx}\n\n질문: {q}"},
    ]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=300, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
```

> **CPT + RAG 가 좋은 이유:** CPT 모델은 용어와 문맥을 이미 알아서 검색된 원문을 더 잘 이해하고 요약한다.
> RAG 는 모델이 틀릴 수 있는 세부 사실을 보정해 준다.

## 6.5 데이터 증강: 다르게 쓴 문장으로 다시 학습

모듈 1.7 에서 본 것처럼 같은 지식을 **여러 표현으로** 보여주면 지식을 더 잘 꺼내 쓴다.
큰 LLM(API 또는 오픈 모델)으로 각 블록을 다시 쓴 데이터를 `data/cpt/train.jsonl` 에 추가한다.

프롬프트 예시:

```
다음은 쿠버네티스 공식 문서의 일부다. 사실 관계, 명령어, YAML, 숫자는 그대로 유지하면서
(1) 초보자에게 설명하는 강의 노트 문체, (2) 질의응답 FAQ 형식 으로 각각 다시 써라.
원문에 없는 내용은 추가하지 마라.

<문서>
{block}
</문서>
```

주의할 점:
- 원문의 라이선스(CC BY 4.0)와 **생성에 쓴 LLM 의 이용 약관**을 모두 확인한다.
- 증강 데이터는 **사실을 지어낼 수 있다.** 샘플을 뽑아 원문과 대조하는 검수 단계를 둔다.

## 6.6 서빙 (g6e)

vLLM 은 PyTorch 버전 요구 사항이 다르므로 **별도 가상환경**에 설치한다.

```bash
cd ~/llm-ft
uv venv -p 3.12 .venv-vllm && source .venv-vllm/bin/activate
uv pip install vllm

# 대화형 모델 (6.2 / 6.3 결과)
vllm serve models/k8s-ko-3b-chat --served-model-name k8s-ko --max-model-len 4096
```

```bash
# 다른 터미널 (원격)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "k8s-ko",
  "messages": [{"role": "user", "content": "디플로이먼트를 롤백하는 방법은?"}],
  "max_tokens": 300, "temperature": 0
}' | python3 -m json.tool
```

- OpenAI 호환 API 다. VS Code PORTS 탭에서 8000 을 포워딩하면 로컬 앱에서도 호출할 수 있다.
- base CPT 모델을 띄울 때는 `/v1/completions` 에 `"prompt"` 로 호출한다.
- Mac 에서 쓸 때는 `./aws/sync.sh down` 으로 받은 뒤 `mlx_lm.server --model models/k8s-ko-3b-chat` 로 띄운다.

---

## 6.7 리소스 정리 (필수)

### 오늘만 끝내고 나중에 이어서 할 때

```bash
./aws/sync.sh down       # 결과물 회수 (models/, 체크포인트 제외)
./aws/ctl.sh stop        # GPU 과금 중지. EBS 300GB 요금은 계속 나감
./aws/ctl.sh status      # stopped 확인
```

### 완전히 끝낼 때

```bash
./aws/sync.sh down
./aws/ctl.sh terminate                                     # 인스턴스 + EBS 삭제
aws ec2 wait instance-terminated --filters Name=tag:Name,Values=llm-ft-g6e
aws ec2 delete-security-group --group-name llm-ft-g6e-sg   # 보안 그룹
aws ec2 delete-key-pair --key-name llm-ft-g6e              # 키 페어
rm -f ~/.ssh/llm-ft-g6e.pem
# ~/.ssh/config 에서 '# >>> llm-ft k8s-gpu >>>' ~ '# <<< llm-ft k8s-gpu <<<' 블록 삭제
```

공유 계정에서 이름을 바꿔 썼다면 `llm-ft-g6e` 를 자신의 `NAME`/`KEY_NAME` 으로 바꾼다.

✅ **확인:** EC2 콘솔 → 인스턴스 목록에 내 인스턴스가 `terminated` 로 표시되고, 볼륨 목록에 남은 볼륨이 없다.

---

## 6.8 회고

함께 이야기해 보자.

1. CPT 전후로 가장 크게 달라진 점은 무엇이었나? 수치와 생성 결과 중 무엇이 더 설득력 있었나?
2. 우리 조직의 문서(사내 위키, 운영 매뉴얼)에 적용한다면 데이터 정제에서 무엇이 가장 어려울까?
3. CPT, SFT, RAG 중 우리 문제에는 어떤 조합이 맞을까? 비용과 정확성 관점에서 생각해 보자.

## 더 읽을거리

- Gururangan 외, *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks* (2020) — 도메인 적응 사전학습
- Allen-Zhu & Li, *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction* (2023) — 지식 증강
- Biderman 외, *LoRA Learns Less and Forgets Less* (2024) — LoRA vs full
- Huang 외, *Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages* (2023)
- Hugging Face TRL 문서: https://huggingface.co/docs/trl
- mlx-lm: https://github.com/ml-explore/mlx-lm

---
← [모듈 5](05_evaluation.md) · [부록: 문제 해결 · FAQ · 용어집 →](99_troubleshooting.md)
