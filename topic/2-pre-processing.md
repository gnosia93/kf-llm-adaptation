## 데이터 전처리 ##

> 학습 코드를 보기 전에 학습 데이터 한 줄이 어떻게 만들어지는지 끝까지 따라가는 것이 목표다.


### 1. 전체 흐름과 경로 ###

```
                     ┌────────────────────────── 입력 (raw/website/) ──────────────────────────┐
GitHub                content/ko/docs/**/*.md          문서 본문 569개 (headless 8개 제외 → 561)
kubernetes/website ─▶ content/ko/examples/**          예제 YAML 등 (code_sample 숏코드)
 download.sh          content/ko/includes/*.md        공통 조각 (include 숏코드)
                      content/en/examples/**          한글에 없는 예제의 대체본
                      content/en/docs/.../feature-gates/*.md   기능 게이트 단계·버전
                      i18n/{ko,en}/*.toml             번역 문자열 (heading, feature-state)
                      data/releases/schedule.yaml     패치 버전 (skew)
                      hugo.toml                       현재 버전 (skew, param)
                     └──────────────────────────────────────────────────────────────────────────┘
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
   prepare_data.py (SFT·RAG용)                           build_cpt.py (CPT용)
   iter_docs → split_sections → split_long              iter_docs → pack_document
   → build_chunks → split_by_document → to_sft          → split_by_document (+ load_replay)
             │                                                       │
             ▼                                                       ▼
   data/chunks.jsonl         3,184 줄                     data/cpt/train.jsonl         726 줄
   data/sft/train.jsonl      3,088 줄                     data/cpt/valid.jsonl          11 줄
   data/sft/valid.jsonl         96 줄                     data/cpt/valid_docs.txt       11 줄
                                                          data/eval/k8s_seen.jsonl     100 줄
                                                          data/eval/k8s_heldout.jsonl   11 줄
                                                          data/eval/general_ko.jsonl   100 줄


   build_sft.py                                      build_pref.py
   용어집 + 섹션(02 재사용) + kullm-v2 replay             data/sft/train.jsonl 의 질문 → SFT 모델이 답 4개
   → 평가용 용어 18개 제외, 누수 제거                       → 심판 LLM 채점 → 최고점/최저점 쌍
             │                                                       │
             ▼                                                       ▼
   data/sft/train.jsonl      3,012 줄                     data/pref/candidates.jsonl   300 줄 (질문당 답 4개 + 점수)
   data/sft/valid.jsonl        138 줄                     data/pref/train.jsonl        266 줄  → DPO, 보상 모델, PPO(질문만)
   data/eval/sft_eval.jsonl     80 줄                     data/pref/valid.jsonl         14 줄
```


### 2. 원본 데이터 구조 ###

```
raw/website/content/ko/docs/
├── concepts/       155   개념 설명 (파드, 서비스, 스토리지 ...)
├── tasks/          162   작업 가이드 (클러스터 관리, 파드 설정, 디버깅 ...)
├── reference/      168   레퍼런스 (그중 glossary/ 124 = 용어집)
├── tutorials/       30   튜토리얼
├── contribute/      29   번역·기여 가이드
├── setup/           22   설치 가이드
├── home/             2   문서 홈
└── images/           0
```
용어집(`reference/glossary/`)이 문서 수의 22% 를 차지하지만 대부분 아주 짧다 (예: `glossary/pod.md` ≈ 128 토큰).

#### pod 설명파일 예시  ####

`raw/website/content/ko/docs/concepts/workloads/pods/_index.md` 의 앞부분:

```markdown
---                                                       ┐
# reviewers:                                              │
# - erictune                                              │ front matter (YAML)
title: 파드                                               │ ← title 만 사용
api_metadata:                                             │
- apiVersion: "v1"                                        │
  kind: "Pod"                                             │
content_type: concept                                     │
weight: 10                                                │
no_list: true                                             │
---                                                       ┘

<!-- overview -->                                         ← HTML 주석 (Hugo 레이아웃용 표식)

_파드(Pod)_ 는 쿠버네티스에서 생성하고 관리할 수 있는 배포 가능한 가장 작은 컴퓨팅 단위이다.

_파드_ (고래 떼나 완두콩 꼬투리처럼)는 하나 이상의
{{< glossary_tooltip text="컨테이너" term_id="container" >}}의 그룹이다. ...   ← 숏코드

<!-- body -->

## 파드란 무엇인가?                                        ← 제목 (섹션 분할 기준)

{{< note >}}                                              ← 강조 상자 숏코드
클러스터의 각 노드에 [컨테이너 런타임](/docs/setup/...)을   ← 마크다운 링크
설치해야 노드에서 파드가 실행될 수 있다.
{{< /note >}}
```

**front matter 키 사용 빈도** (569개 문서, 실측):

| 키 | 문서 수 | 의미 | 사용 여부 |
|---|---|---|---|
| `title` | 568 | 페이지 제목 | ✅ 블록 헤더 `# 제목` |
| `weight` | 402 | 메뉴 정렬 순서 | ✗ |
| `content_type` | 366 | concept / task / tutorial / reference | ✗ (11절) |
| `id`, `tags`, `short_description`, `aka`, `full_link` | 116~123 | 용어집 메타데이터 | ✗ |
| `description` | 90 | 요약 | ✗ |
| `headless` | 9 (그중 `true` 8) | 조각 문서 표시 | ✅ 단독 문서에서 제외 |


### 3. 출력 파일 형식 ###

모든 파일은 **JSON Lines** 형식이다. 한 줄 = JSON 객체 하나 = **샘플 하나**이고, UTF-8 로 저장한다.

| 파일 | 줄 수 | 크기 | 스키마 | 쓰는 곳 |
|---|---|---|---|---|
| `data/cpt/train.jsonl` | 726 | 5.2 MB | `{"text": str}` | CPT 학습 |
| `data/cpt/valid.jsonl` | 11 | 0.1 MB | `{"text": str}` | CPT 학습 중 eval loss |
| `data/cpt/valid_docs.txt` | 11 | — | 문서 경로 (텍스트) | valid 로 뺀 문서 기록 |
| `data/eval/k8s_seen.jsonl` | 100 | 0.8 MB | `{"text": str}` | 평가: 학습한 문서 |
| `data/eval/k8s_heldout.jsonl` | 11 | 0.1 MB | `{"text": str}` | 평가: 안 본 문서 (= valid) |
| `data/eval/general_ko.jsonl` | 100 | 0.4 MB | `{"text": str}` | 평가: 망각 |
| `data/chunks.jsonl` | 3,184 | 4.5 MB | `{"source", "title", "heading", "text"}` | RAG (모듈 8) |
| `data/sft/train.jsonl` | 3,012 | 4.0 MB | `{"messages": [{role, content} x 3]}` | SFT 학습 (`06_build_sft.py`, 6절). `02` 단독 실행 시 3,088 줄 |
| `data/sft/valid.jsonl` | 138 | 0.3 MB | 위와 같음 | SFT 학습 중 eval loss |
| `data/eval/sft_eval.jsonl` | 80 | 0.1 MB | `{"kind", "system", "question", "reference", "source"?}` | 대화형 모델 평가 (`07_eval_sft.py`, KFP Gate 3·4) |
| `data/pref/candidates.jsonl` | 300 | 1.7 MB | `{"id", "kind", "messages", "reference", "answers": [4], "stopped": [4], "scores": [4]}` | 재채점용 원본 (7절) |
| `data/pref/train.jsonl` | 266 | 0.7 MB | `{"id", "kind", "prompt", "chosen", "rejected", "score_chosen", "score_rejected"}` | DPO, 보상 모델, PPO(질문만) |
| `data/pref/valid.jsonl` | 14 | 0.04 MB | 위와 같음 | DPO·보상 모델 eval |

실제 줄의 예 (값은 앞부분만):

```jsonc
// data/cpt/train.jsonl — 쿠버네티스 블록 (용어집)
{"text": "# 파드 프라이어리티 (원문, Pod Priority)\n\n파드 프라이어리티는 다른 파드에 대한 상대적인 중요도를 나타낸다.\n\n..."}

// data/cpt/train.jsonl — 예제가 삽입된 블록
{"text": "# 로깅 아키텍처\n\n...\n\n`debug/counter-pod.yaml`\n\n```yaml\napiVersion: v1\nkind: Pod\nmetadata:\n  name: counter\n..."}

// data/cpt/train.jsonl — replay 블록 (위키)
{"text": "# 1989년\n\n1989년은 일요일로 시작하는 평년이다.\n\n..."}

// data/chunks.jsonl
{"source": "concepts/services-networking/service-topology.md", "title": "토폴로지 키를 사용하여 ...",
 "heading": "예시들 > 노드 로컬 엔드포인트 선호", "text": "노드 로컬 엔드포인트를 선호하지만, ...\n\n```yaml\n..."}

// data/sft/train.jsonl — 용어집 질문
{"messages": [{"role": "system", "content": "당신은 쿠버네티스 전문가입니다. ..."},
              {"role": "user", "content": "파드의 개념을 알려줘."},
              {"role": "assistant", "content": "가장 작고 단순한 쿠버네티스 오브젝트. 파드는 사용자 클러스터에서 동작하는 컨테이너의 집합을 나타낸다.\n\n..."}]}

// data/eval/sft_eval.jsonl — 평가용으로 뺀 용어
{"kind": "glossary_heldout", "system": "당신은 쿠버네티스 전문가입니다. ...", "question": "스테이트풀셋의 개념을 알려줘.",
 "reference": "파드 집합의 디플로이먼트와 스케일링을 관리하며, ...", "source": "statefulset"}

// data/pref/train.jsonl — 선호 쌍 (DPO 는 prompt/chosen/rejected 만 쓴다)
{"id": 84, "kind": "k8s",
 "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "파드 시큐리티 어드미션의 메트릭 부분을 정리해 주세요."}],
 "chosen": [{"role": "assistant", "content": "쿠버네티스의 파드 시큐리티 어드미션에서 메트릭은 ..."}],
 "rejected": [{"role": "assistant", "content": "CUber네타스 파드 시큐리티 어드미션은 ..."}],
 "score_chosen": 6.0, "score_rejected": 2.0}
```

> **"한 줄의 길이"는 무엇으로 재나:** 파일에서 한 줄은 수백 바이트 ~ 수십 KB 지만, 학습에서 의미 있는 길이는
> **`text` 를 토크나이저로 바꾼 토큰 수**다. 다음 절의 "블록 길이"는 모두 토큰 수다.


### 4. 고려사항 ###

| # | 고려사항 | 왜 중요한가 | 이 데이터의 실측 / 판단 |
|---|---|---|---|
| 1 | **문서 길이 분포** | 상한이 문서보다 짧으면 문서가 여러 블록으로 잘린다 | 중앙값 1,207 / 90% 5,103 / 99% 15,910 / 최대 19,535 토큰 |
| 2 | **맥락 연속성** | 잘린 블록끼리는 서로를 볼 수 없다. 앞 섹션에서 정의한 개념을 뒤 섹션에서 쓰는 연결이 끊긴다 | 2K: 210개 문서가 잘림 → 8K: 27개 |
| 3 | **학습 프레임워크의 잘림** | 상한을 넘는 샘플은 **뒷부분이 버려진다** (mlx-lm `max_seq_length`, TRL `max_length`) | 블록 상한 = 학습 `max_length`. EOS 1 토큰 + `SAFETY_TOKENS` 를 여유로 둔다 |
| 4 | **GPU 메모리** | **마이크로배치당 토큰 수**(배치 x 길이)에 비례한다. 특히 로짓 = 토큰 x 어휘 151,936 | 4,096 토큰/마이크로배치 ≈ 로짓 5~9GB. 8K x 배치1 은 Liger 커널로 로짓을 만들지 않게 한다 |
| 5 | **padding 낭비** | 길이가 다른 샘플을 한 배치로 묶으면 짧은 쪽을 채워야 한다 | 배치2: 2K 24%, 4K 31%, 8K 36% / **배치1: 0%** |
| 6 | **attention 계산량** | attention 은 길이에 비례해서 커진다 (토큰당) | 같은 토큰 수 기준 대략 4K +10%, 8K +20%, 32K 약 2배 |
| 7 | **모델의 사전학습 문맥 길이** | 짧게만 CPT 하면 긴 문맥 능력이 약해질 수 있다 | Qwen2.5-3B 최대 32,768. 4K~8K 가 절충 |
| 8 | **짧은 샘플의 GPU 효율** | 128 토큰 샘플 하나로는 GPU 를 거의 놀린다 | 8K 기준 블록의 26% 가 256 토큰 미만 (토큰 비중 2%) |
| 9 | **스텝당 토큰 수** | 배치1 + 누적이면 스텝마다 토큰 수가 들쭉날쭉 | transformers ≥ 4.46 은 누적 구간 전체 토큰 수로 loss 를 나눈다 → 토큰마다 같은 무게 |
| 10 | **평가 길이 일관성** | 평가에서 더 짧게 자르면 학습과 조건이 달라진다 | 노트북 평가는 `MAX_TOKENS` 를 그대로 쓴다 |
| 11 | **replay 문서 길이** | `load_replay` 도 같은 상한으로 자른다 | 상한을 늘리면 위키 문서는 길어지고 개수는 줄어든다 (2K 289개 → 8K 184개) |
| 12 | **토크나이저 의존성** | 같은 텍스트도 토크나이저마다 토큰 수가 다르다 | 모델을 바꾸면 `--tokenizer` 를 바꿔 다시 생성 |
| 13 | **valid 분리 단위** | valid 는 문서 단위로 분리한다. 블록이 크면 valid 문서 수와 블록 수가 비슷해진다 | 8K: 문서 11개 = 블록 11개 / 2K: 문서 14개 = 블록 19개 |

#### 1. 문서 길이 분포 (정제 후) ####

```
   길이 기준   이보다 긴 문서     그 문서들이 차지하는 토큰 비율
  > 2,048       201개 (36%)          81.2%
  > 4,096        88개 (16%)          54.4%
  > 8,192        25개  (4%)          25.3%
  > 16,384        4개  (1%)           6.1%
  > 32,768        0개
가장 긴 문서: job.md 19,535 / deployment.md 19,360 / feature-gates/index.md 17,740
전체 1,201,047 토큰
```

문서 **개수**로는 대부분 짧지만 토큰 **양**의 81% 는 2K 보다 긴 문서에 있다.
"대부분 짧으니 2K 면 충분하다"는 판단은 개수만 본 것이라 틀린다.
정제를 고치면서 예제 YAML 과 include 가 들어가 문서가 조금 길어졌다 (중앙값 1,104 → 1,207).

#### 2. 상한별 패킹 결과 비교 (실측) ####

| 상한 | 블록 수 | 쪼개진 문서 | 평균 / 최대 토큰 | 줄 단위 강제 분할 | padding (배치2 / 배치1) |
|---|---|---|---|---|---|
| 2,048 | 973 | 210 | 1,253 / 2,007 | 2 | 24% / 0% |
| 4,096 | 656 | 90 | 1,850 / 4,032 | 0 | 31% / 0% |
| **8,192 (기본)** | **553** | **27** | 2,192 / 8,111 | 0 | 36% / **0%** |

8,192 기준 블록 길이 분포:

| 블록 길이 | 블록 수 | 토큰 비중 |
|---|---|---|
| 0 ~ 255 | 143 (26%) | 2% |
| 256 ~ 1,023 | 82 (15%) | 5% |
| 1,024 ~ 2,047 | 106 (19%) | 13% |
| 2,048 ~ 4,095 | 123 (22%) | 29% |
| 4,096 ~ 8,192 | 99 (18%) | 52% |


