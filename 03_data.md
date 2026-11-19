# 모듈 3. 데이터 파이프라인 — 수집 → 정제 → 패킹 → replay

**소요 50분** · 강의 20분 + 실습 30분

## 학습 목표

- 공개 문서 저장소에서 필요한 부분만 효율적으로 받는다.
- 마크업(Hugo 숏코드, HTML)을 제거하되 **코드 블록은 보존하는** 정제를 구현한다.
- 왜 토큰 단위로 **패킹**하는지 설명하고, 패킹 결과를 검증한다.
- replay 데이터를 섞고, 평가 세트 3종을 만든다.

## 파이프라인 한눈에 보기

```
01_download.sh          02_prepare_data.py                 03_build_cpt.py
──────────────          ───────────────────────────        ─────────────────────────────────────
GitHub                  parse_doc()                        pack_document()
kubernetes/website  ─▶   front matter → title       ─▶     섹션 조각을 2,048 토큰 이하로 묶음
content/ko/docs          clean_markdown()                   + "# 제목", "## 소제목" 헤더
(569개 .md)              숏코드·링크·HTML 제거             load_replay()
                         (코드 블록은 그대로)               위키백과에서 20% 만큼 가져옴
                        split_sections()                         │
                         제목 기준 섹션 분할                     ▼
                        split_long()                       data/cpt/train.jsonl   1,193 개
                         긴 섹션을 문단 단위로              data/cpt/valid.jsonl      18 개
                                                           data/eval/{k8s_seen, k8s_heldout, general_ko}.jsonl
```

---

## 3.1 수집: sparse checkout

`kubernetes/website` 저장소는 모든 언어의 문서와 이미지를 합쳐 크다.
필요한 것은 `content/ko` 하나뿐이므로 **필요한 디렉터리만** 받는다.

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/kubernetes/website.git raw/website
git -C raw/website sparse-checkout set content/ko
```

| 옵션 | 의미 |
|---|---|
| `--depth 1` | 최신 커밋만 (히스토리 제외) |
| `--filter=blob:none` | 파일 내용은 실제로 필요할 때만 받음 |
| `--sparse` + `sparse-checkout set` | 지정한 디렉터리만 작업 트리에 풀어 놓음 |

```bash
./01_download.sh                    # [ko] 마크다운 문서 수: 569
LANGS="ko en" ./01_download.sh      # [en] 마크다운 문서 수: 1729  (선택 과제)
```

> 한글 번역은 영어 원문의 약 1/3 만 다룬다. 번역되지 않은 주제는 영어 원문을 넣어야 배울 수 있다.

## 3.2 정제: 무엇을 지우고 무엇을 남길까

쿠버네티스 문서는 정적 사이트 생성기 **Hugo** 로 만든다. 원문에는 사람이 읽지 않는 문법이 섞여 있다.

### 원문 (`concepts/workloads/pods/_index.md`)

```markdown
---
title: 파드
content_type: concept
weight: 10
---

<!-- overview -->

_파드(Pod)_ 는 쿠버네티스에서 생성하고 관리할 수 있는 배포 가능한 가장 작은 컴퓨팅 단위이다.

_파드_ (고래 떼나 완두콩 꼬투리처럼)는 하나 이상의
{{< glossary_tooltip text="컨테이너" term_id="container" >}}의 그룹이다. ...

## 파드란 무엇인가?

{{< note >}}
클러스터의 각 노드에 [컨테이너 런타임](/docs/setup/production-environment/container-runtimes/)을
설치해야 노드에서 파드가 실행될 수 있다.
{{< /note >}}
```

### 정제 후

```markdown
_파드(Pod)_ 는 쿠버네티스에서 생성하고 관리할 수 있는 배포 가능한 가장 작은 컴퓨팅 단위이다.

_파드_ (고래 떼나 완두콩 꼬투리처럼)는 하나 이상의
컨테이너의 그룹이다. ...

## 파드란 무엇인가?

참고:
클러스터의 각 노드에 컨테이너 런타임을
설치해야 노드에서 파드가 실행될 수 있다.
```

### 정제 규칙 (`02_prepare_data.py`)

| 대상 | 예 | 처리 | 정규식 |
|---|---|---|---|
| front matter | `---\ntitle: 파드\n---` | `title` 만 뽑고 제거 | `FRONT_MATTER_RE`, `TITLE_RE` |
| HTML 주석 | `<!-- overview -->` | 제거 | `COMMENT_RE` |
| 용어 툴팁 | `{{< glossary_tooltip text="컨테이너" ...>}}` | `컨테이너` | `TOOLTIP_TEXT_RE` |
| 툴팁 (text 없음) | `{{< glossary_tooltip term_id="pod" >}}` | `pod` | `TOOLTIP_TERM_RE` |
| 강조 상자 | `{{< note >}}` / `caution` / `warning` | `참고:` / `주의:` / `경고:` | `ADMONITION_RE` |
| 기타 숏코드 | `{{< feature-state ...>}}`, `{{< /note >}}` | 제거 | `SHORTCODE_RE` |
| 이미지 | `![diagram](/images/...)` | 제거 | `IMAGE_RE` |
| 링크 | `[컨테이너 런타임](/docs/...)` | `컨테이너 런타임` | `LINK_RE` |
| HTML 태그 | `<br>`, `<table>` ... | 지정한 태그만 제거 | `HTML_TAG_RE` |
| **코드 블록** | ` ```yaml ... ``` ` | **그대로 둔다** | `CODE_FENCE_RE` 로 분리 |

### 코드 블록은 왜 건드리지 않나

```python
def clean_markdown(text):
    text = COMMENT_RE.sub("", text)
    # 코드 블록은 그대로 두고 나머지 부분만 정리 (kubectl 예제, YAML 보존)
    parts = CODE_FENCE_RE.split(text)
    text = "".join(p if i % 2 else clean_prose(p) for i, p in enumerate(parts))
    return re.sub(r"\n{3,}", "\n\n", text).strip()
```

- `re.split` 에 **괄호로 묶은 그룹**을 주면 구분자(코드 블록)도 결과에 들어간다.
  그래서 짝수 번째는 본문, 홀수 번째는 코드 블록이 된다.
- 코드 블록 안의 `<pod-name>` 같은 표현을 HTML 태그로 오인해서 지우지 않게 하고,
  `kubectl` 명령과 YAML 매니페스트를 **그대로 학습**시킨다. 쿠버네티스 지식의 핵심이 여기에 있다.

### 섹션 분할: 코드 안의 `#` 은 제목이 아니다

```python
for line in body.splitlines():
    if is_fence(line):            # ``` 를 만날 때마다 코드 블록 안/밖 전환
        in_code = not in_code
    m = None if in_code else HEADING_RE.match(line)   # 코드 블록 안에서는 제목 검사 안 함
```

bash 예제의 `# 파드 목록 확인` 같은 주석이 제목으로 잘못 인식되는 것을 막는다.
섹션 제목은 `상위 > 하위` 경로로 붙는다. pods 문서는 이렇게 나뉜다:

```
'파드란 무엇인가?'
'파드의 사용'
'파드의 사용 > 파드를 관리하기 위한 워크로드 리소스'
'파드 작업 > 파드 OS'
'파드 작업 > 파드와 컨트롤러'
...
```

## 3.3 패킹: 왜 토큰 길이를 직접 맞추나

학습 프레임워크는 한 번에 최대 `max_seq_length` (2,048) 토큰을 본다. 이보다 긴 샘플이나 짧은 샘플을 그대로 넣으면 문제가 생긴다.

| 상황 | 문제 |
|---|---|
| 샘플이 2,048 토큰보다 길다 | **뒷부분이 잘려서 버려진다** (mlx-lm 소스에서 확인). 그만큼 지식이 사라진다 |
| 샘플이 아주 짧다 (200 토큰) | 배치를 채우려고 padding 이 들어가서 GPU 계산이 낭비된다 |
| 문서 중간부터 시작하는 샘플 | 무슨 문서에 대한 내용인지 모른 채 학습한다 |

### `pack_document()` 알고리즘

```
문서 "파드" ─ 섹션 조각들 (각 600자 내외, 코드 블록은 쪼개지 않음)
 [소개] [파드란 무엇인가? 1] [파드란 무엇인가? 2] [파드의 사용] [파드 작업 > 파드 OS] ...

  블록 1 (≤ 2,016 토큰)                       블록 2
 ┌─────────────────────────────────┐        ┌───────────────────────────────┐
 │ # 파드                          │        │ # 파드                        │ ← 제목을 매 블록 반복
 │ [소개]                          │        │ ## 파드 작업 > 파드 OS         │ ← 섹션 중간에서 시작하면
 │ ## 파드란 무엇인가?              │        │ [파드 작업 > 파드 OS]          │   소제목도 다시 붙임
 │ [조각 1] [조각 2]               │  넘침  │ ...                           │
 │ ## 파드의 사용                   │ ────▶ │                               │
 └─────────────────────────────────┘        └───────────────────────────────┘
```

```python
for heading, text in prep.split_sections(body):
    for piece in prep.split_long(text, PIECE_CHARS):
        seg = f"## {heading}\n\n{piece}" if heading and (heading != cur_heading or not cur) else piece
        seg_len = n_tokens(seg) + 2
        if cur and cur_len + seg_len > budget:      # 넘치면 현재 블록 확정
            flush()
            ...
        if seg_len > budget:                        # 조각 하나가 블록보다 크면 (거대한 YAML 등)
            ids = tok.encode(seg, add_special_tokens=False)
            for i in range(0, len(ids), budget):    #   토큰 단위로 강제 분할
                blocks.append(header + tok.decode(ids[i:i + budget]))
```

- `budget = 2048 − 32(여유) − 헤더 토큰 수`. 32 토큰 여유는 조각마다 따로 센 토큰 수의 합과
  이어 붙인 뒤 실제 토큰 수가 조금 다를 수 있어서 둔다. 처음엔 16 이었는데, 경계에 걸리는 블록이 나와서 늘렸다.
- **문서 경계는 넘지 않는다.** 서로 다른 문서를 한 블록에 섞지 않는다.

### 실측 결과

```
쿠버네티스 블록: train 922 (1,141,311 토큰) / valid 18
train 전체(replay 포함 1,193개) 길이 분포:
  < 512 토큰: 320 | 512~1023: 174 | 1024~1535: 139 | ≥ 1536: 560
  평균 1,197 토큰, 최대 2,033 토큰 → 2,048 안에 모두 들어감 (잘리는 샘플 없음)
```

짧은 블록(<512)이 많은 이유는 용어집처럼 문서 자체가 짧기 때문이다.

## 3.4 replay 와 평가 세트

```python
target = int(domain_tokens * args.replay_ratio / (1 - args.replay_ratio))
#        1,141,311 * 0.2 / 0.8 = 285,327 토큰 → 위키 문서 271개
```

- `wikimedia/wikipedia` 의 `20231101.ko` 를 **스트리밍**으로 읽는다. 전체를 받지 않는다.
- 300자 미만 문서는 건너뛰고, 각 문서는 2,016 토큰에서 자른다.
- 처음 100개는 평가용(`general_ko`)으로 따로 떼어 둔다. **학습에는 쓰지 않는다.**

### 평가 세트 3종

| 파일 | 내용 | 학습에 포함? | 무엇을 보나 |
|---|---|---|---|
| `data/eval/k8s_seen.jsonl` | train 블록 중 100개 | ✅ | 지식 습득 (암기 포함) |
| `data/eval/k8s_heldout.jsonl` | valid 블록 18개 | ❌ | 도메인 일반화 |
| `data/eval/general_ko.jsonl` | 위키 문서 100개 | ❌ | 망각 |

> **생각해 보기:** valid 18개 블록은 학습에서 빠지므로 그 내용은 모델이 배우지 못한다.
> 지식 주입이 목적이면 `--valid-ratio` 를 0.01 로 줄이는 것도 방법이다. 무엇을 얻고 무엇을 잃을까?

---

## 실습

노트북 **`1. 데이터 준비`** 섹션의 셀을 실행한다.

```python
INCLUDE_EN = False
...
!LANGS="{langs}" bash 01_download.sh
!.venv/bin/python 03_build_cpt.py --docs {docs} --batch-size 16
```

✅ **확인:** 출력이 다음과 비슷해야 한다. 위키 샘플링 때문에 replay 숫자는 조금 다를 수 있다.
```
raw/website/content/ko/docs: 문서 569개
쿠버네티스 블록: train 922 (1,141,311 토큰) / valid 18
일반 텍스트 replay: 271개 (285,564 토큰)
총 train 샘플 1193 -> batch 16 기준 1 epoch = 75 iters
```

아래 과제는 노트북에 새 셀을 추가해서 진행한다.

### 과제 3-1. 정제 전후 비교

```python
from importlib import import_module
from pathlib import Path
prep = import_module("02_prepare_data")

p = Path("raw/website/content/ko/docs/concepts/services-networking/service.md")
print(p.read_text()[:1200])
print("=" * 60)
title, body = prep.parse_doc(p)
print("TITLE:", title)
print(body[:1200])
```

- 원문의 어떤 문법이 어떻게 바뀌었는지 3가지 찾아보자.
- 코드 블록이 그대로 남아 있는지 확인하자.

### 과제 3-2. 토크나이저 관찰

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
for w in ["쿠버네티스", "Kubernetes", "파드", "Pod", "디플로이먼트", "kubectl get pods -n kube-system"]:
    ids = tok.encode(w)
    print(f"{w!r:36} {len(ids):2} tokens", [tok.decode([i]) for i in ids])
```

- 한국어 용어와 영어 용어 중 어느 쪽이 토큰을 더 많이 쓰나?
- 같은 내용의 영어 원문을 넣으면 학습 토큰은 어떻게 달라질까?

### 과제 3-3. 블록 길이 분포 그리기

```python
import json, matplotlib.pyplot as plt
rows = [json.loads(l)["text"] for l in open("data/cpt/train.jsonl", encoding="utf-8")]
lens = [len(tok.encode(r)) for r in rows]
plt.hist(lens, bins=40); plt.xlabel("tokens per sample"); plt.ylabel("count"); plt.show()
print("max", max(lens), "| 2,048 초과:", sum(l > 2048 for l in lens))
```

- 2,048 을 넘는 샘플이 없는지 확인하자.
- 오른쪽 끝(2,000 근처)에 몰린 블록들과 왼쪽(짧은 블록)은 각각 어떤 문서일까?

### 과제 3-4. 블록 하나 뜯어보기

```python
import random
r = random.choice([r for r in rows if r.startswith("# ") and "```" in r])
print(r[:1500])
```

- `# 제목` 과 `## 소제목` 헤더가 제대로 붙어 있나?
- 코드 블록이 중간에서 잘리지 않았나?

### 과제 3-5 (도전). replay 비율 바꿔 보기

```python
!.venv/bin/python 03_build_cpt.py --replay-ratio 0.4 --out /tmp/data_r40 --batch-size 16
```

- 학습 샘플 수와 토큰 수는 어떻게 바뀌나? 학습 시간에는 어떤 영향이 있을까?
- **주의:** `--out` 을 지정하지 않으면 실제 학습용 `data/` 를 덮어쓴다.

### 과제 3-6 (도전). 영어 원문 추가

노트북에서 `INCLUDE_EN = True` 로 바꾸고 데이터를 다시 만든다.
토큰 수가 몇 배가 되는지, 1 epoch 의 iters 가 얼마가 되는지 기록하자.
학습 시간도 그만큼 늘어나므로 강사의 안내에 따라 결정한다.

---
← [모듈 2](02_environment.md) · 다음: [모듈 4. 계속 사전학습 →](04_training.md)
