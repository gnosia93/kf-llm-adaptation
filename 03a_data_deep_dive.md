# 모듈 3-A. 데이터 전처리 상세 해설 — 원본 구조부터 함수·라인별 동작까지

> [모듈 3](03_data.md) 이 "무엇을 하는가"의 개요라면, 이 문서는 **"정확히 어떻게 하는가"** 를 다룬다.
> 학습 코드를 보기 전에 학습 데이터 한 줄이 어떻게 만들어지는지 끝까지 따라가는 것이 목표다.

**이 문서의 기준 (재현 조건)**

| 항목 | 값 |
|---|---|
| 원본 저장소 | `kubernetes/website` 커밋 `80f8465` (2026-11-18) |
| 작업 디렉터리 | `~/llm-ft` (모든 경로는 여기 기준 상대 경로) |
| 토크나이저 | `Qwen/Qwen2.5-3B` |
| 시드 | 42 |
| 라인 번호 | 이 문서를 쓴 시점의 `01_download.sh`, `02_prepare_data.py`, `03_build_cpt.py` 기준 |

문서에 나오는 숫자는 모두 이 조건에서 **실제로 측정한 값**이다. 같은 조건이면 같은 값이 나와야 한다.

---

## 목차

1. [전체 흐름과 경로](#1-전체-흐름과-경로)
2. [원본 데이터 구조](#2-원본-데이터-구조)
3. [`01_download.sh` — 라인별 해설](#3-01_downloadsh--라인별-해설)
4. [`02_prepare_data.py` — 상수와 함수별 해설](#4-02_prepare_datapy--상수와-함수별-해설)
5. [`03_build_cpt.py` — 상수와 함수별 해설](#5-03_build_cptpy--상수와-함수별-해설)
6. [출력 파일 형식](#6-출력-파일-형식)
7. [블록 길이(JSONL 한 줄의 길이) 선택 고려사항](#7-블록-길이jsonl-한-줄의-길이-선택-고려사항)
8. [현재 전처리의 알려진 문제 (검증 결과)](#8-현재-전처리의-알려진-문제-검증-결과)
9. [직접 확인해 보는 명령 모음](#9-직접-확인해-보는-명령-모음)

---

## 1. 전체 흐름과 경로

```
                    ┌──────────────────── 입력 ────────────────────┐
GitHub              raw/website/content/ko/docs/**/*.md   (569개)
kubernetes/website ─▶ raw/website/content/ko/examples/**    (224개, 현재 미사용 → 8.2)
 01_download.sh      raw/website/content/ko/includes/*.md   (4개,   현재 미사용 → 8.3)
                    └──────────────────────────────────────────────┘
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
   02_prepare_data.py (SFT·RAG용)                        03_build_cpt.py (CPT용)
   parse_doc → split_sections → split_long              parse_doc → split_sections → split_long
   → build_chunks → to_sft                               → pack_document → (+ load_replay)
             │                                                       │
             ▼                                                       ▼
   data/chunks.jsonl         2,997 줄                     data/cpt/train.jsonl       1,193 줄
   data/sft/train.jsonl      2,908 줄                     data/cpt/valid.jsonl          18 줄
   data/sft/valid.jsonl         89 줄                     data/eval/k8s_seen.jsonl     100 줄
                                                          data/eval/k8s_heldout.jsonl   18 줄
                                                          data/eval/general_ko.jsonl   100 줄
```

- `03_build_cpt.py` 는 `02_prepare_data.py` 의 함수(`parse_doc`, `split_sections`, `split_long`)를 **import 해서 재사용**한다 (`03_build_cpt.py:22`).
  그래서 정제 규칙을 고치면 SFT 데이터와 CPT 데이터에 동시에 반영된다.
- CPT 학습에 실제로 쓰이는 것은 **`data/cpt/` 와 `data/eval/`** 이다. `data/sft/` 와 `data/chunks.jsonl` 은 모듈 6 (SFT, RAG) 에서 쓴다.

실행 명령 (항상 `~/llm-ft` 에서):

```bash
./01_download.sh
python 02_prepare_data.py
python 03_build_cpt.py          # python -I 로 실행하면 02_prepare_data 를 import 하지 못한다
```

---

## 2. 원본 데이터 구조

### 2.1 저장소와 받아 오는 범위

`kubernetes/website` 는 kubernetes.io 사이트의 소스다. 정적 사이트 생성기 **Hugo** 로 빌드한다.
우리는 sparse checkout 으로 **`content/ko` 디렉터리만** 받는다.

```
raw/website/                         ← git 저장소 루트 (sparse checkout)
├── content/
│   └── ko/                          ← 받아 오는 유일한 디렉터리
│       ├── _index.html              사이트 홈 (HTML, 미사용)
│       ├── README.md, OWNERS        번역팀 메타 파일 (미사용)
│       ├── docs/          ★         문서 본문: .md 569개 ← 학습 데이터
│       ├── examples/      ☆         문서가 참조하는 YAML 등 224개 (현재 미사용)
│       ├── includes/      ☆         여러 문서가 끼워 넣는 공통 조각 .md 4개 (현재 미사용)
│       ├── blog/  community/  partners/  releases/  training/  _common-resources/   (미사용)
├── i18n/ko/ko.toml                  숏코드 번역 문자열 (작업 트리에는 없음, git show 로 열람 가능)
└── hugo.toml                        사이트 설정: 버전 변수 등 (작업 트리에는 없음)
```

`content/ko` 전체의 파일 형식 분포 (실측):

| 확장자 | 개수 | 위치 |
|---|---|---|
| `.md` | 596 | `docs/` 569, 나머지는 `includes/`, `blog/` 등 |
| `.yaml` | 216 | 거의 모두 `examples/` |
| `.html` | 16 | 사이트 페이지 |
| `.png`, `.svg` | 8 | 이미지 |
| `.sh`, `.py`, `.json`, `Dockerfile` | 6 | `examples/` |

### 2.2 `docs/` 디렉터리 구조

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

하위 디렉터리 중 문서가 많은 곳 (실측 상위 10개):

| 경로 (`docs/` 기준) | `.md` 수 |
|---|---|
| `reference/glossary/` | 124 |
| `tasks/administer-cluster/` | 42 |
| `concepts/workloads/` | 26 |
| `tasks/configure-pod-container/` | 20 |
| `tasks/debug/` | 19 |
| `concepts/cluster-administration/` | 16 |
| `concepts/storage/` | 15 |
| `setup/production-environment/` | 14 |
| `concepts/services-networking/` | 14 |
| `concepts/scheduling-eviction/` | 14 |

> 용어집(`reference/glossary/`)이 전체 문서의 22% 를 차지하지만 대부분 아주 짧다 (예: `glossary/pod.md` = 128 토큰).
> 문서 **개수**와 **토큰 양**은 다르게 봐야 한다 (7.2 참고).

### 2.3 파일의 종류

| 종류 | 경로 예 | 특징 | 코드에서의 처리 |
|---|---|---|---|
| 일반 페이지 | `concepts/architecture/nodes.md` | front matter + 본문 | 그대로 |
| 섹션 인덱스 | `concepts/workloads/pods/_index.md` | 디렉터리의 대표 페이지 | `title` 이 없으면 **디렉터리 이름**을 제목으로 (`02:80-81`) |
| 페이지 번들 | `reference/command-line-tools-reference/feature-gates/index.md` | 디렉터리 + `index.md` | 그대로 (제목은 front matter 에서) |
| 용어집 | `reference/glossary/pod.md` | `id`, `short_description`, `aka` 등 메타데이터가 많음 | 메타데이터는 버리고 `title` + 본문만 |
| headless 조각 | front matter 에 `headless: true` (9개) | 단독 페이지가 아니라 다른 문서에 끼워 넣는 조각 | **단독 문서로 학습됨** |
| front matter 없음 | `contribute/generate-ref-docs/prerequisites-ref-docs.md` (1개) | `---` 로 시작하지 않음 | 제목이 파일명(`prerequisites-ref-docs`)이 됨 |

### 2.4 파일 하나 해부하기

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
| `title` | 568 | 페이지 제목 | ✅ 블록 헤더 `# 제목` 으로 사용 |
| `weight` | 402 | 메뉴 정렬 순서 | ✗ |
| `content_type` | 366 | concept / task / tutorial / reference | ✗ (8 절 개선안 참고) |
| `id`, `tags`, `short_description`, `aka`, `full_link` | 116~123 | 용어집 메타데이터 | ✗ |
| `description` | 90 | 요약 | ✗ |
| `headless` | 9 | 조각 문서 표시 | ✗ |

### 2.5 Hugo 숏코드 전체 현황

숏코드는 `{{< 이름 인자 >}}` 또는 `{{% 이름 인자 %}}` 형식의 Hugo 템플릿 호출이다.
**사이트를 빌드할 때 다른 내용으로 바뀌는** 자리 표시자라서, 원문 그대로 학습하면 안 된다.

`docs/` 안의 사용 현황과 **현재 코드의 처리 결과** (실측):

| 숏코드 | 등장 횟수 | 쓰는 문서 | 사이트에서 바뀌는 내용 | 현재 처리 결과 | 판정 |
|---|---|---|---|---|---|
| `glossary_tooltip` | 861 | — | 용어 링크 (`text` 를 표시) | `text` 값으로 치환 | ✅ |
| `note` / `caution` / `warning` | 580 / 59 / 50 | — | 강조 상자 | `참고:` / `주의:` / `경고:` | ✅ |
| `heading` | 427 | 287 | 번역된 제목 ("다음 내용", "시작하기 전에" ...) | **빈 `## ` 줄이 남음** | ❌ 8.1 |
| `feature-state` | 241 | 116 | "FEATURE STATE: v1.25 [stable]" | 삭제 | ⚠️ 8.7 |
| `skew` / `param` | 171 / 113 | 51 / 53 | 버전 문자열 (`1.37` 등) | 삭제 → **"쿠버네티스 에는"** | ❌ 8.5 |
| `codenew` / `code_sample` | 140 / 99 | 64 / 35 | `examples/` 의 YAML 파일 내용 | **삭제 → YAML 손실** | ❌ 8.2 |
| `include` | 109 | 100 | `includes/` 의 공통 조각 내용 | **삭제 → 내용 손실** | ❌ 8.3 |
| `tabs` / `tab` | 44 / 108 | 25 | 탭 UI (리눅스/윈도우 등) | 표식만 삭제, 내용은 남음. **탭 이름 손실** | ⚠️ 8.8 |
| `version-check` | 63 | 61 | "쿠버네티스 서버 버전은 ..." 안내문 | 삭제 | ✅ (반복 안내문) |
| `glossary_definition` | 29 | 21 | 용어집 정의 본문 | **삭제 → 내용 손실** | ❌ 8.4 |
| `thirdparty-content` | 28 | — | 서드파티 고지문 | 삭제 | ✅ |
| `mermaid` | 27 | 12 | 다이어그램 | 표식만 삭제, mermaid 코드가 본문에 남음 | ⚠️ |
| `figure` | 20 | 14 | 이미지 | 삭제 (`alt`/캡션 손실) | ⚠️ |
| `table` | 19 | 13 | 표 캡션 | 표식만 삭제, 표 본문은 남음 | ✅ |
| `comment` | 11 | 8 | **사이트에 표시되지 않는 주석** | 표식만 삭제, **숨김 내용이 본문에 노출** | ❌ 8.6 |

### 2.6 본문 밖에서 끌어오는 자원

사이트에서는 문서 본문이 다른 파일의 내용을 **끌어와서** 완성된다. 현재 전처리는 이 부분을 처리하지 않는다.

| 자원 | 경로 | 참조하는 숏코드 | 참조 수 | 예 |
|---|---|---|---|---|
| 예제 매니페스트 | `raw/website/content/ko/examples/<경로>` | `codenew`, `code_sample` | 239 | `{{% code_sample file="debug/counter-pod.yaml" %}}` → `examples/debug/counter-pod.yaml` |
| 공통 조각 | `raw/website/content/ko/includes/<파일>` 또는 문서와 같은 디렉터리 | `include` | 109 | `{{< include "task-tutorial-prereqs.md" >}}` (95회) |
| 용어 정의 | `raw/website/content/ko/docs/reference/glossary/<term_id>.md` | `glossary_definition` | 29 | `{{< glossary_definition term_id="kube-apiserver" length="all" >}}` |
| 번역 문자열 | `i18n/ko/ko.toml` (git 에만 있음) | `heading` | 427 | `whatsnext_heading` → `"다음 내용"` |
| 버전 변수 | `hugo.toml` (git 에만 있음) | `skew`, `param` | 284 | `latest = "v1.37"` |

---

## 3. `01_download.sh` — 라인별 해설

```bash
 6  set -euo pipefail
 7  LANGS=${LANGS:-ko}
```
- `6`: 명령이 하나라도 실패하면(`-e`), 정의되지 않은 변수를 쓰면(`-u`), 파이프 중간에서 실패하면(`pipefail`) 즉시 멈춘다.
- `7`: 받을 언어 목록. 환경변수가 없으면 `ko`. `LANGS="ko en" ./01_download.sh` 처럼 여러 개를 지정할 수 있다.

```bash
 9  mkdir -p raw
10  if [ ! -d raw/website ]; then
11    git clone --depth 1 --filter=blob:none --sparse \
12      https://github.com/kubernetes/website.git raw/website
13  fi
```
- `10`: 이미 받았으면 clone 을 건너뛴다 (재실행해도 안전하다).
- `11`: `--depth 1` 은 최신 커밋 하나만, `--filter=blob:none` 은 파일 내용을 필요할 때만 받기,
  `--sparse` 는 처음에는 루트의 파일만 작업 트리에 풀기.

```bash
15  git -C raw/website sparse-checkout set $(printf 'content/%s ' $LANGS)
```
- 작업 트리에 풀 디렉터리를 **지정한 목록으로 교체**한다. `LANGS=ko` 면 `content/ko` 만.
- ⚠️ "추가"가 아니라 "교체"다. 전에 `ko en` 으로 받았어도 `ko` 로 다시 실행하면 `content/en` 은 작업 트리에서 사라진다 (git 객체는 남아 있다).

```bash
17  for l in $LANGS; do
18    echo "[$l] 마크다운 문서 수: $(find raw/website/content/$l/docs -name '*.md' | wc -l)"
19  done
```
- 결과 확인용. `[ko] 마크다운 문서 수: 569`.

---

## 4. `02_prepare_data.py` — 상수와 함수별 해설

### 4.1 상수 (`15–49`)

**`SYSTEM_PROMPT`, `QUESTION_TEMPLATES*` (`15–30`)** — SFT 데이터용. CPT 에는 쓰이지 않는다.

**정규식 (`32–48`)**

| 라인 | 이름 | 패턴 | 하는 일 | 입력 → 출력 예 |
|---|---|---|---|---|
| 32 | `FRONT_MATTER_RE` | `\A---\s*\n(.*?)\n---\s*\n` (`re.S`) | 파일 **맨 앞**(`\A`)의 `---` ~ `---` 블록. 그룹 1 = YAML 본문 | `---\ntitle: 파드\n---\n본문` → 그룹1 `title: 파드` |
| 33 | `TITLE_RE` | `^title:\s*(.+?)\s*$` (`re.M`) | front matter 안에서 `title:` 줄의 값 | `title: "파드"` → `"파드"` (따옴표는 `79`행에서 제거) |
| 34 | `COMMENT_RE` | `<!--.*?-->` (`re.S`) | HTML 주석. 여러 줄도 가능 | `<!-- overview -->` → `` |
| 35 | `CODE_FENCE_RE` | ``(```.*?```\|~~~.*?~~~)`` (`re.S`) | 코드 블록 전체. **괄호로 묶여 있어서** `split` 결과에 코드 블록도 포함된다 | 4.3 참고 |
| 36 | `TOOLTIP_TEXT_RE` | `\{\{<\s*glossary_tooltip[^}]*?text="([^"]*)"[^}]*?>\}\}` | `text` 속성이 있는 용어 툴팁 | `{{< glossary_tooltip text="컨테이너" term_id="container" >}}` → `컨테이너` |
| 37 | `TOOLTIP_TERM_RE` | 위와 같고 `term_id="..."` 를 잡음 | `text` 가 없는 툴팁 | `{{< glossary_tooltip term_id="pod" >}}` → `pod` |
| 38 | `ADMONITION_RE` | `\{\{[<%]\s*(note\|caution\|warning)\s*[>%]\}\}` | 강조 상자 **여는** 태그 | `{{< note >}}` → `참고:` |
| 39 | `SHORTCODE_RE` | `\{\{[<%].*?[>%]\}\}` (`re.S`) | **나머지 모든 숏코드** (닫는 태그 포함) | `{{< /note >}}` → `` |
| 40 | `IMAGE_RE` | `!\[[^\]]*\]\([^)]*\)` | 마크다운 이미지 | `![그림](/images/a.svg)` → `` |
| 41 | `LINK_RE` | `\[([^\]]+)\]\([^)]*\)` | 마크다운 링크 → 링크 텍스트만 | `[컨테이너 런타임](/docs/...)` → `컨테이너 런타임` |
| 42–46 | `HTML_TAG_RE` | `</?(br\|div\|...\|center)\b[^>]*>` (`re.I`) | **지정한 태그 이름만** 제거 | `줄<br>바꿈` → `줄바꿈` |
| 47 | `HEADING_RE` | `^(#{1,4})\s+(.+?)\s*#*\s*$` | 제목 줄. 그룹1 = `#` 개수(레벨), 그룹2 = 제목 | `## 파드 OS` → (`##`, `파드 OS`) |
| 48 | `ANCHOR_RE` | `\s*\{#[^}]*\}` | 제목 뒤 앵커 | `## 개요 {#overview}` → `## 개요` |
| 49 | `ADMONITION_LABEL` | dict | 강조 상자 이름 → 한글 라벨 | `note` → `참고:` |

> **왜 HTML 태그를 전부 지우지 않나 (`42–46`):** `<[^>]+>` 처럼 모두 지우면 본문에 쓰인
> `<pod-name>`, `<namespace>` 같은 **자리 표시자까지** 사라진다. 그래서 실제 HTML 태그 이름만 골라서 지운다.

> **`HEADING_RE` 가 `#{1,4}` 인 이유 (`47`):** `#####` 이상은 문서에서 거의 쓰이지 않는 세부 제목이라 섹션을 나누지 않고 본문으로 둔다.
> `#####` 줄에서는 `#{1,4}` 가 앞의 4개를 먹은 뒤 다음 글자가 `\s` 가 아니라 `#` 이라서 매칭이 실패한다.

### 4.2 `clean_prose(text)` (`52–60`) — 코드 바깥 텍스트 정리

```python
52 def clean_prose(text):
54     text = TOOLTIP_TEXT_RE.sub(r"\1", text)       # ① text="..." 가 있는 툴팁 → 그 텍스트
55     text = TOOLTIP_TERM_RE.sub(r"\1", text)       # ② 남은 툴팁 → term_id
56     text = ADMONITION_RE.sub(lambda m: ADMONITION_LABEL[m.group(1)], text)   # ③ note → 참고:
57     text = SHORTCODE_RE.sub("", text)             # ④ 나머지 숏코드 전부 삭제
58     text = IMAGE_RE.sub("", text)                 # ⑤ 이미지 삭제
59     text = LINK_RE.sub(r"\1", text)               # ⑥ 링크 → 텍스트
60     return HTML_TAG_RE.sub("", text)              # ⑦ 지정한 HTML 태그 삭제
```

**순서가 중요하다.**
- ①②가 ④보다 먼저: ④는 모든 숏코드를 지우므로, 툴팁의 텍스트를 먼저 꺼내 놓아야 한다.
- ①이 ②보다 먼저: `text` 와 `term_id` 가 둘 다 있으면 사람이 읽는 `text`(한글)를 써야 한다. `term_id`(영문 ID)는 `text` 가 없을 때만 쓴다.
- ③이 ④보다 먼저: 여는 태그를 라벨로 바꾸고, 닫는 태그(`{{< /note >}}`)는 ④에서 지운다.
- ⑤가 ⑥보다 먼저: 이미지 문법 `![a](b)` 는 링크 문법 `[a](b)` 를 포함한다. 먼저 지우지 않으면 ⑥이 `!a` 로 바꿔 버린다.

입력 → 출력 (실제 실행):

```
'{{< glossary_tooltip term_id="pod" text="파드" >}}를 만든다.'   →  '파드를 만든다.'
'{{< note >}}\n클러스터의 각 노드에 [컨테이너 런타임](/docs/x)을\n설치해야 한다.\n{{< /note >}}'
                                                                 →  '참고:\n클러스터의 각 노드에 컨테이너 런타임을\n설치해야 한다.\n'
```

### 4.3 `clean_markdown(text)` (`63–68`) — 코드 블록을 보존하며 정리

```python
63 def clean_markdown(text):
64     text = COMMENT_RE.sub("", text)                                   # ① HTML 주석 삭제 (코드 안 포함)
66     parts = CODE_FENCE_RE.split(text)                                 # ② 본문/코드 블록 분리
67     text = "".join(p if i % 2 else clean_prose(p) for i, p in enumerate(parts))  # ③ 본문만 정리
68     return re.sub(r"\n{3,}", "\n\n", text).strip()                   # ④ 빈 줄 정리
```

- `64`: 주석은 코드 블록 안에 있어도 지운다. YAML 안의 `<!-- -->` 는 거의 없어서 문제가 되지 않는다.
- `66`: `re.split` 은 패턴에 **캡처 그룹**이 있으면 구분자도 결과 리스트에 넣는다.
  ```
  "앞 문장\n```yaml\na: 1\n```\n뒤 문장"
     → ['앞 문장\n', '```yaml\na: 1\n```', '\n뒤 문장']
         i=0 (본문)     i=1 (코드)           i=2 (본문)
  ```
  그래서 **짝수 인덱스는 본문, 홀수 인덱스는 코드 블록**이다.
- `67`: 홀수(`i % 2 == 1`)는 그대로, 짝수만 `clean_prose()` 를 적용한 뒤 다시 이어 붙인다.
  코드 블록 안의 `{{ .data }}` (Go 템플릿), `<pod-name>`, `[a](b)` 같은 문자열이 보존된다.
- `68`: 숏코드를 지우고 나면 빈 줄이 여러 개 생긴다. 3줄 이상 연속된 줄바꿈을 2줄(빈 줄 하나)로 줄인다.

### 4.4 `parse_doc(path)` (`71–82`) — 파일 하나 → (제목, 정제된 본문)

```python
71 def parse_doc(path):
72     raw = path.read_text(encoding="utf-8")
73     title, body = path.stem, raw                  # 기본값: 제목 = 파일명(확장자 제외), 본문 = 전체
74     m = FRONT_MATTER_RE.match(raw)
75     if m:
76         body = raw[m.end():]                      # front matter 다음부터가 본문
77         t = TITLE_RE.search(m.group(1))           # front matter 안에서 title 찾기
78         if t:
79             title = t.group(1).strip("'\"")       # 양끝 따옴표 제거: "파드" → 파드
80     if title == "_index":
81         title = path.parent.name                  # _index.md 인데 title 이 없으면 디렉터리 이름
82     return title, clean_markdown(body)
```

| 상황 | 결과 |
|---|---|
| 일반 (`title: 파드`) | `("파드", 정제된 본문)` |
| front matter 없음 (1개 파일) | 제목 = 파일명 `prerequisites-ref-docs`, 본문 = 파일 전체 |
| `_index.md` 에 `title` 없음 | 제목 = 디렉터리 이름 (예: `pods`) |
| 파일이 BOM 이나 빈 줄로 시작 | `\A---` 가 매칭되지 않아 front matter 가 **본문에 섞인다**. 현재 데이터에서는 없음 |

### 4.5 `is_fence(line)` (`85–86`)

```python
85 def is_fence(line):
86     return line.lstrip().startswith(("```", "~~~"))
```

- 앞 공백을 무시한다. 목록 안에 들여쓴 코드 블록(`   ```shell`)도 펜스로 인식한다.
- ⚠️ **한 줄 안에서 열고 닫는 경우**(` ```kubectl -A``` `)도 토글 한 번으로 센다. 그러면 그 뒤로 코드 안/밖이 **뒤집힌다**. 8.9 참고.

### 4.6 `split_sections(body)` (`89–110`) — 제목 기준 섹션 분할

```python
 91     sections, stack, buf, in_code = [], [], [], False
 93     def flush():
 94         text = "\n".join(buf).strip()
 95         if text:                                              # 빈 섹션(제목만 있고 본문 없음)은 버림
 96             sections.append((" > ".join(h for _, h in stack), text))
 98     for line in body.splitlines():
 99         if is_fence(line):
100             in_code = not in_code                             # 펜스를 만날 때마다 코드 안/밖 전환
101         m = None if in_code else HEADING_RE.match(line)       # 코드 안에서는 제목 검사를 하지 않음
102         if m:
103             flush()                                           # 지금까지 모은 본문을 이전 제목으로 확정
104             buf = []
105             level, heading = len(m.group(1)), ANCHOR_RE.sub("", m.group(2)).strip()
106             stack = [(lv, h) for lv, h in stack if lv < level] + [(level, heading)]
107         else:
108             buf.append(line)
109     flush()                                                   # 마지막 섹션
110     return sections
```

**`stack` 의 동작 (`106`)** — 현재 위치의 제목 경로를 유지한다. 새 제목보다 **레벨이 낮은(상위)** 것만 남기고 새 제목을 넣는다.

```
줄                       stack 이후                                    섹션 이름
## 파드의 사용            [(2,'파드의 사용')]                           '파드의 사용'
### 워크로드 리소스       [(2,'파드의 사용'), (3,'워크로드 리소스')]     '파드의 사용 > 워크로드 리소스'
## 파드 작업              [(2,'파드 작업')]                             '파드 작업'          ← 같은 레벨: 교체
### 파드 OS               [(2,'파드 작업'), (3,'파드 OS')]              '파드 작업 > 파드 OS'
```

- **첫 제목 이전의 본문**(도입부)은 `stack` 이 비어 있으므로 섹션 이름이 `''` 이다.
- **`101`행이 핵심**: 코드 블록 안의 bash 주석 `# 파드 목록 확인` 을 제목으로 오인하지 않는다.

### 4.7 `split_long(text, max_chars)` (`113–127`) — 긴 섹션을 조각으로

```python
115     parts, buf, size, in_code = [], [], 0, False
116     for line in text.splitlines():
117         if is_fence(line):
118             in_code = not in_code
119         buf.append(line)
120         size += len(line) + 1                                   # +1 = 줄바꿈 문자
121         if not in_code and (
122             (size >= max_chars and not line.strip()) or size >= max_chars * 2
123         ):
124             parts.append("\n".join(buf).strip())
125             buf, size = [], 0
126     parts.append("\n".join(buf).strip())
127     return [p for p in parts if p]
```

자르는 조건 (`121–122`): **코드 블록 밖이면서**, 다음 둘 중 하나일 때.

| 조건 | 의미 |
|---|---|
| `size >= max_chars and not line.strip()` | 목표 크기를 넘었고 **지금 줄이 빈 줄** → 문단 경계에서 자름 (자연스러움) |
| `size >= max_chars * 2` | 빈 줄 없이 2배를 넘음 → 아무 줄 경계에서나 자름 (안전장치) |

- **코드 블록 안에서는 절대 자르지 않는다.** 그래서 거대한 YAML 은 조각 하나가 `max_chars` 보다 훨씬 커질 수 있다.
  이 경우는 `03_build_cpt.py` 의 강제 분할(5.2)이 처리한다.
- 단위가 **글자 수**다. 토큰 수가 아니다. 토큰 기준 길이 제어는 `pack_document()` 가 한다.
- 호출하는 쪽마다 `max_chars` 가 다르다. `02_prepare_data.py` 는 1,800 (SFT 답변 하나의 크기), `03_build_cpt.py` 는 600 (패킹 단위).

### 4.8 `build_chunks(docs_dir, min_chars, max_chars)` (`130–149`) — SFT·RAG 용 청크

```python
132     files = sorted(docs_dir.rglob("*.md"))                      # 정렬 → 항상 같은 순서 (재현성)
133     for path in files:
134         title, body = parse_doc(path)
135         for heading, text in split_sections(body):
136             pieces = split_long(text, max_chars)
137             for i, piece in enumerate(pieces):
138                 if len(piece) < min_chars:                      # 150자 미만 조각은 버림
139                     continue
140                 h = heading
141                 if len(pieces) > 1:
142                     h = f"{heading or title} ({i + 1}/{len(pieces)})"   # 예: '파드 작업 (2/3)'
143                 chunks.append({"source": ..., "title": ..., "heading": h, "text": piece})
```

- `138`: SFT 답변이나 RAG 검색 결과로 쓰기에 너무 짧은 조각은 버린다.
  ⚠️ CPT 쪽(`pack_document`)에는 **이 필터가 없다**. 짧은 섹션도 모두 학습에 들어간다.
- `142`: 한 섹션이 여러 조각이면 제목 뒤에 `(i/n)` 을 붙여 구분한다.

### 4.9 `to_sft`, `write_jsonl`, `main` (`152–194`)

- `to_sft` (`152–161`): 청크 하나를 system/user/assistant 대화로 바꾼다. 질문은 템플릿에서 무작위로 고른다. 답변은 청크 원문이다.
- `write_jsonl` (`164–168`): 한 줄에 JSON 하나. `ensure_ascii=False` 라서 한글이 `\uXXXX` 로 바뀌지 않고 그대로 저장된다 (파일 크기 약 1/2).
- `main` (`171–194`): 청크를 섞어서 3% 를 valid 로 떼고, `data/sft/` 와 `data/chunks.jsonl` 을 쓴다.

---

## 5. `03_build_cpt.py` — 상수와 함수별 해설

### 5.1 import 와 상수 (`14–25`)

```python
20 from transformers import AutoTokenizer
22 prep = import_module("02_prepare_data")      # 파일명이 숫자로 시작해서 import 문을 못 씀 → import_module
24 PIECE_CHARS = 600                            # split_long 에 넘기는 조각 크기 (글자)
25 SAFETY_TOKENS = 32                           # 토큰 예산 여유분
```

- `22`: `import 02_prepare_data` 는 문법 오류다. 모듈 이름이 숫자로 시작하기 때문이다. `import_module` 은 문자열로 받으므로 가능하다.
  스크립트 디렉터리가 `sys.path` 에 있어야 하므로 **`python -I` 로 실행하면 실패**한다.
- `24` **왜 600자인가:** 패킹은 조각 단위로 블록을 채운다. 조각이 크면 블록 끝에 빈 공간이 많이 남고, 너무 작으면 토큰을 세는 횟수가 늘어 느려진다.
  600자는 약 320 토큰이다 (1 토큰 ≈ 1.85 글자). 2,048 토큰 블록에 6개 정도가 들어간다.
- `25` **왜 32 토큰 여유인가:** `pack_document` 는 조각마다 **따로** 토큰 수를 센 뒤 더한다. 하지만 조각을 이어 붙이고 나서 한 번에 토큰화하면 경계에서 토큰이 조금 달라질 수 있다 (BPE 는 문맥에 따라 병합이 달라진다).
  처음에는 16 이었는데 2,048 을 넘는 블록이 1개 나와서 32 로 늘렸다. 지금은 최대 2,033 토큰이다.

### 5.2 `pack_document(title, body, tok, max_tokens)` (`28–61`) — 문서 하나 → 블록들

```python
30     def n_tokens(s):
31         return len(tok.encode(s, add_special_tokens=False))      # BOS/EOS 없이 순수 텍스트 토큰 수
33     header = f"# {title}\n\n"                                   # 모든 블록 맨 앞에 붙는 제목
34     budget = max_tokens - SAFETY_TOKENS - n_tokens(header)      # 블록 본문에 쓸 수 있는 토큰 수
35     blocks, cur, cur_len, cur_heading = [], [], 0, None
       #  blocks: 완성된 블록 / cur: 만들고 있는 블록의 조각들 / cur_len: 그 토큰 수 / cur_heading: 직전 조각의 섹션
37     def flush():
38         if cur:
39             blocks.append(header + "\n\n".join(cur))           # 헤더 + 조각들을 빈 줄로 연결
```

- `34`: 예를 들어 `max_tokens=2048`, 제목 "파드"면 헤더 `# 파드\n\n` 이 약 4 토큰이므로 `budget = 2048 − 32 − 4 = 2012`.

```python
41     for heading, text in prep.split_sections(body):            # ① 섹션 단위로
42         for piece in prep.split_long(text, PIECE_CHARS):       # ② 600자 조각 단위로
44             seg = f"## {heading}\n\n{piece}" if heading and (heading != cur_heading or not cur) else piece
45             seg_len = n_tokens(seg) + 2                         # +2 = 조각 사이 "\n\n" 몫
```

- `44` **소제목을 언제 붙이나:** 섹션 이름이 있고, 다음 둘 중 하나일 때 `## 소제목` 을 앞에 붙인다.
  - `heading != cur_heading`: 새 섹션이 시작됐다.
  - `not cur`: 블록이 비어 있다. 즉 새 블록의 첫 조각이다.

  같은 섹션의 두 번째 조각부터는 소제목 없이 이어 붙인다.
- `45`: `"\n\n".join` 으로 조각 사이에 들어가는 구분자 몫을 대략 2 토큰으로 잡는다.

```python
46             if cur and cur_len + seg_len > budget:             # ③ 이 조각을 넣으면 넘친다
47                 flush()                                         #    → 지금 블록 확정
48                 cur, cur_len = [], 0
49                 seg = f"## {heading}\n\n{piece}" if heading else piece   # 새 블록이므로 소제목 다시 붙임
50                 seg_len = n_tokens(seg) + 2
```

- `46`: `cur` 가 비어 있지 않을 때만 확정한다. 빈 블록을 만들지 않기 위해서다.
- `49`: 섹션 중간에서 새 블록이 시작돼도, 블록만 떼어 보면 어느 섹션인지 알 수 있게 소제목을 다시 붙인다.

```python
51             if seg_len > budget:                                # ④ 조각 하나가 블록보다 크다 (거대 YAML 등)
53                 ids = tok.encode(seg, add_special_tokens=False)
54                 for i in range(0, len(ids), budget):
55                     blocks.append(header + tok.decode(ids[i:i + budget]))   # 토큰 단위 강제 분할
56             else:
57                 cur.append(seg)                                  # ⑤ 보통의 경우: 현재 블록에 추가
58                 cur_len += seg_len
59             cur_heading = heading
60     flush()                                                     # ⑥ 마지막 블록
61     return blocks
```

- `51–55`: `split_long` 은 코드 블록 안에서 자르지 않는다. 그래서 거대한 코드 블록은 이 경로로 온다.
  실측으로 2,048 에서 **4개 블록** (`job.md`, `zookeeper.md`), 4,096 이상에서는 0개. 강제 분할된 블록은 코드 블록 중간에서 끊긴다.
- 문서 경계: 함수가 **문서 하나만** 받으므로 서로 다른 문서가 한 블록에 섞이지 않는다.

**동작 예시** (문서 "파드", `budget` 2012 토큰):

```
조각 (섹션, 토큰)                       판단                              cur_len
(소개, 300)                            heading 없음 → 그대로 추가          302
(파드란 무엇인가?, 310)                 새 섹션 → "## 파드란 무엇인가?" 추가  ~620
(파드란 무엇인가?, 320)                 같은 섹션 → 그대로 추가              ~942
(파드의 사용, 330)                      새 섹션 → 소제목 붙여 추가           ~1280
(파드의 사용 > 워크로드 리소스, 400)     새 섹션 → 추가                       ~1690
(파드 작업, 380)                        1690+385 > 2012 → flush, 새 블록     ~385
                                        ("# 파드\n\n## 파드 작업\n\n...")
```

### 5.3 `load_replay(args, tok, target_tokens, n_eval)` (`64–84`) — 일반 한국어 텍스트

```python
66     from datasets import load_dataset                           # replay 를 쓸 때만 import (선택 의존성)
68     ds = load_dataset(args.replay_dataset, args.replay_config, split="train", streaming=True)
69     ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
```

- `68`: `wikimedia/wikipedia` 의 `20231101.ko`. `streaming=True` 라서 **전체(수 GB)를 받지 않고** 필요한 만큼만 읽는다.
- `69`: 스트리밍 셔플은 1만 개 버퍼 안에서만 섞는다. 앞쪽 샤드에서 주로 뽑히므로 완전한 무작위 표본은 아니다.

```python
71     for row in ds:
72         text = row["text"].strip()
73         if len(text) < 300:                                      # 너무 짧은 문서(토막 글)는 건너뜀
74             continue
75         ids = tok.encode(text, add_special_tokens=False)[: args.max_tokens - SAFETY_TOKENS]   # 길이 제한
76         text = f"# {row['title']}\n\n{tok.decode(ids)}"           # 쿠버네티스 블록과 같은 헤더 형식
77         if len(evals) < n_eval:
78             evals.append({"text": text})                         # 처음 100개 → 평가용 (학습에 안 씀)
79         elif total < target_tokens:
80             train.append({"text": text})                         # 그다음 → 목표 토큰까지 학습용
81             total += len(ids)
82         else:
83             break
```

- `75`: 문서를 `max_tokens − 32` 토큰에서 **자른다**. 긴 위키 문서는 앞부분만 쓴다.
  ⚠️ 토큰 경계에서 자르면 한글 글자의 UTF-8 바이트 중간에서 끊길 수 있다. 그러면 `decode` 결과의 끝에 `�` 가 생긴다. 실측 **4개 샘플**. 8.10 참고.
- `76`: 쿠버네티스 블록과 형식을 맞춘다. 모델이 "# 제목 다음에 본문"이라는 같은 구조로 배우게 하려는 것이다.
- `77–80`: **평가용을 먼저** 채운다. 학습용과 겹치지 않는다.

### 5.4 `main()` (`94–147`)

```python
112     tok = AutoTokenizer.from_pretrained(args.tokenizer)        # 학습 모델과 같은 토크나이저여야 함
115     for docs in args.docs:                                     # --docs 는 여러 개 가능 (ko, en)
117         files = sorted(docs_dir.rglob("*.md"))
119             title, body = prep.parse_doc(path)
120             blocks += pack_document(title, body, tok, args.max_tokens)

123     blocks = [{"text": b} for b in blocks if len(tok.encode(b)) >= 32]   # 32 토큰 미만 블록 제거 (실측 2개)
124     rng.shuffle(blocks)
125     n_valid = max(1, int(len(blocks) * args.valid_ratio))     # 2% → 940 x 0.02 = 18개
126     valid, domain_train = blocks[:n_valid], blocks[n_valid:]
127     domain_tokens = sum(len(tok.encode(b["text"])) for b in domain_train)
```

- `123`: 헤더만 있고 내용이 거의 없는 블록을 버린다.
- `124–126`: **블록 단위**로 섞어서 나눈다. 같은 문서의 다른 블록이 train 과 valid 에 나뉘어 들어갈 수 있다.
  실측으로 valid 18개 중 **11개**가 이런 경우다. 8.11 참고.

```python
130     if args.replay_ratio > 0:
131         target = int(domain_tokens * args.replay_ratio / (1 - args.replay_ratio))
```

- `131` **replay 목표 토큰 계산:** `replay / (domain + replay) = r` 이 되게 하려면 `replay = domain × r / (1 − r)` 이다.
  `1,141,311 × 0.2 / 0.8 = 285,327` → 실제로는 문서 단위로 넘치게 채워서 285,564 토큰 (271개).

```python
134     train = domain_train + replay
135     rng.shuffle(train)                                         # 도메인과 replay 를 섞음 → 학습 중 고르게 분포
137–142 write_jsonl(...)                                        # 출력 파일 5개 (6절)
139     ... rng.sample(domain_train, min(args.n_eval, len(domain_train)))   # k8s_seen: train 에서 100개 표본
144     iters_per_epoch = -(-len(train) // args.batch_size)       # 올림 나눗셈 (출력용)
```

---

## 6. 출력 파일 형식

모든 파일은 **JSON Lines** 형식이다. 한 줄 = JSON 객체 하나 = **샘플 하나**이고, UTF-8 로 저장한다.

| 파일 | 줄 수 | 크기 | 스키마 | 쓰는 곳 |
|---|---|---|---|---|
| `data/cpt/train.jsonl` | 1,193 | 4.9 MB | `{"text": str}` | CPT 학습 |
| `data/cpt/valid.jsonl` | 18 | 0.1 MB | `{"text": str}` | CPT 학습 중 eval loss |
| `data/eval/k8s_seen.jsonl` | 100 | 0.4 MB | `{"text": str}` | 평가: 학습한 문서 |
| `data/eval/k8s_heldout.jsonl` | 18 | 0.1 MB | `{"text": str}` | 평가: 안 본 문서 (= valid) |
| `data/eval/general_ko.jsonl` | 100 | 0.3 MB | `{"text": str}` | 평가: 망각 |
| `data/sft/train.jsonl` | 2,908 | 4.7 MB | `{"messages": [{role, content} x 3]}` | SFT (모듈 6) |
| `data/sft/valid.jsonl` | 89 | 0.2 MB | 위와 같음 | SFT (모듈 6) |
| `data/chunks.jsonl` | 2,997 | 4.3 MB | `{"source", "title", "heading", "text"}` | RAG (모듈 6) |

실제 줄의 예 (값은 앞부분만):

```jsonc
// data/cpt/train.jsonl — 쿠버네티스 블록 (용어집, 이 줄은 404 바이트)
{"text": "# 파드 프라이어리티 (원문, Pod Priority)\n\n파드 프라이어리티는 다른 파드에 대한 상대적인 중요도를 나타낸다.\n\n..."}

// data/cpt/train.jsonl — replay 블록 (위키)
{"text": "# 1989년\n\n1989년은 일요일로 시작하는 평년이다.\n\n연호\n\n사건 \n 1월 1일 - ..."}

// data/chunks.jsonl
{"source": "concepts/services-networking/service-topology.md",
 "title": "토폴로지 키를 사용하여 토폴로지-인지 트래픽 라우팅",
 "heading": "예시들 > 노드 로컬 엔드포인트 선호",
 "text": "노드 로컬 엔드포인트를 선호하지만, ...\n\n```yaml\napiVersion: v1 ..."}
```

> **"한 줄의 길이"는 무엇으로 재나:** 파일에서 한 줄은 바이트로 수백 B ~ 수십 KB 지만, 학습에서 의미 있는 길이는
> **`text` 를 토크나이저로 바꾼 토큰 수**다. 위 404 바이트짜리 줄은 128 토큰 내외다. 다음 절의 "블록 길이"는 모두 토큰 수다.

---

## 7. 블록 길이(JSONL 한 줄의 길이) 선택 고려사항

`03_build_cpt.py --max-tokens N` 이 정하는 것은 **한 줄(샘플)의 최대 토큰 수**다.
목표 길이가 아니라 **상한**이다. 짧은 문서는 그 길이 그대로 한 줄이 된다.

### 7.1 고려사항 체크리스트

| # | 고려사항 | 왜 중요한가 | 이 데이터의 실측 / 판단 |
|---|---|---|---|
| 1 | **문서 길이 분포** | 상한이 문서보다 짧으면 문서가 여러 블록으로 잘린다 | 중앙값 1,104 / 90% 5,036 / 99% 15,722 / 최대 19,222 토큰 |
| 2 | **맥락 연속성** | 잘린 블록끼리는 서로를 볼 수 없다. 앞 섹션의 정의와 뒤 섹션의 사용이 끊긴다 | 2K: 191개 문서가 잘림 → 8K: 25개 |
| 3 | **학습 프레임워크의 잘림 동작** | 상한을 넘는 샘플은 **뒷부분이 버려진다** (mlx-lm `max_seq_length`, TRL `max_length`) | 블록 상한 = 학습 `max_length` 로 맞추고, EOS 1 토큰 + `SAFETY_TOKENS` 를 여유로 둔다 |
| 4 | **GPU 메모리** | 메모리는 **마이크로배치당 토큰 수**(배치 x 길이)에 비례한다. 특히 로짓 = 토큰 x 151,936 | 4,096 토큰/마이크로배치 ≈ 로짓 5~9GB. 8K x 배치1 은 L40S 에서 Liger 커널 권장 |
| 5 | **padding 낭비** | 길이가 다른 샘플을 한 배치로 묶으면 짧은 쪽을 padding 으로 채운다 | 2K x 배치2 = **24% 낭비**, 8K x 배치2 = 36%, **배치1 = 0%** |
| 6 | **attention 계산량** | attention 계산은 길이에 비례해 커진다 (토큰당) | 같은 토큰 수 기준 4K +10%, 8K +20%, 32K 약 2배 (대략적 계산) |
| 7 | **모델의 사전학습 문맥 길이** | 짧은 길이로만 CPT 하면 긴 문맥 능력이 약해질 수 있다 | Qwen2.5-3B 최대 32,768. 4K~8K 가 절충 |
| 8 | **짧은 샘플의 GPU 효율** | 128 토큰 샘플 하나로는 GPU 를 거의 놀린다 | 8K 기준 블록의 27% 가 256 토큰 미만 (토큰 비중은 2%) |
| 9 | **스텝당 토큰 수** | 배치1 + gradient accumulation 이면 스텝마다 토큰 수가 들쭉날쭉 | transformers ≥ 4.46 은 누적 구간 전체 토큰 수로 loss 를 정규화한다 → 토큰당 같은 무게 |
| 10 | **평가 길이 일관성** | 평가에서 더 짧게 자르면 학습과 조건이 다르다 | 노트북 평가 `max_length=2048` → 블록 길이를 바꾸면 **같이 바꿔야 함** |
| 11 | **replay 문서 길이** | `load_replay` 도 같은 `max_tokens` 로 자른다 (`75`행) | 상한을 늘리면 위키 문서도 길어지고 개수는 줄어든다 |
| 12 | **토크나이저 의존성** | 같은 텍스트도 토크나이저마다 토큰 수가 다르다 | 모델을 바꾸면 `--tokenizer` 를 바꿔 다시 생성 |
| 13 | **valid 분리 단위** | 블록이 클수록 문서 단위 분리에 가까워진다 | 8K 면 문서의 96% 가 블록 1개 → 누수가 줄어든다 (8.11) |

### 7.2 문서 길이 분포 (정제 후, `# 제목` 포함, Qwen 토큰)

```
   길이 기준   이보다 긴 문서     그 문서들이 차지하는 토큰 비율
  > 2,048       181개 (32%)          77.8%
  > 4,096        78개 (14%)          51.8%
  > 8,192        23개  (4%)          24.6%
  > 16,384        3개  (1%)           4.9%
  > 32,768        0개
가장 긴 문서: deployment.md 19,222 / job.md 18,575 / feature-gates/index.md 17,734
```

문서 **개수**로는 대부분 짧지만, 토큰 **양**의 78% 가 2K 보다 긴 문서에 있다.
"대부분 짧으니 2K 면 충분하다"는 판단은 문서 수만 보고 내린 것이라 틀린다.

### 7.3 상한별 패킹 결과 비교 (실측)

| 상한 | 블록 수 | 쪼개진 문서 | 평균 / 최대 토큰 | 강제 분할 블록 | padding 낭비 (배치2 / 배치1) |
|---|---|---|---|---|---|
| 2,048 (현재) | 942 | 191 | 1,232 / 2,016 | 4 | 24% / 0% |
| 4,096 | 654 | 81 | 1,767 / 4,032 | 0 | — / 0% |
| 8,192 | 561 | 25 | 2,057 / 8,086 | 0 | 36% / 0% |

8,192 기준 블록 길이 분포:

| 블록 길이 | 블록 수 | 토큰 비중 |
|---|---|---|
| 0 ~ 255 | 150 (27%) | 2% |
| 256 ~ 1,023 | 99 (18%) | 6% |
| 1,024 ~ 2,047 | 113 (20%) | 15% |
| 2,048 ~ 4,095 | 109 (19%) | 27% |
| 4,096 ~ 8,192 | 90 (16%) | 50% |

### 7.4 환경별 권장

| 환경 | 권장 | 이유 |
|---|---|---|
| g6e (L40S 48GB), full | **8,192 x 배치1 x 누적 8 + Liger 커널** | 쪼개지는 문서 25개, padding 0%. 메모리는 Liger 커널로 확보 (**리허설에서 실측 필요**) |
| g6e, Liger 없이 | 4,096 x 배치1 x 누적 16 | 지금(2K x 2)과 마이크로배치 토큰이 같아서 메모리도 같다. padding 24% → 0% |
| Mac 24GB (MLX) | 2,048 (현재) 또는 4,096 x 배치1 | 이미 메모리가 빠듯하다. 4K 는 실측 필요 |
| 32,768 | 권장하지 않음 | 32K 를 넘는 문서가 없다. 채우려면 문서를 섞어야 하고, 메모리와 계산이 크게 는다 |

### 7.5 상한을 바꿀 때 같이 바꿔야 하는 곳

```bash
python 03_build_cpt.py --max-tokens 8192 --batch-size 8   # ① 데이터 재생성 (--batch-size 는 출력용)
```

| # | 파일 | 바꿀 것 |
|---|---|---|
| ② | `train_cpt_hf.py` | `SFTConfig(max_length=...)`, `per_device_train_batch_size=1`, `gradient_accumulation_steps`, (8K) `use_liger_kernel=True` |
| ③ | `config_cpt.yaml` (Mac) | `max_seq_length`, `batch_size`, `grad_accumulation_steps` |
| ④ | `k8s_cpt_g6e.ipynb` | 데이터 생성 셀의 `--max-tokens`, 평가 셀 `perplexity()` 의 `max_length=2048` |
| ⑤ | `05_eval_cpt.py` (Mac) | `--max-tokens` 기본값 |
| ⑥ | `requirements-cuda.txt` | (8K) `liger-kernel` |

---

## 8. 현재 전처리의 알려진 문제 (검증 결과)

이 문서를 쓰면서 실제 데이터로 검증해서 찾은 문제들이다. **아직 고치지 않았다.** 영향이 큰 순서로 정리했다.

### 8.1 ❌ `heading` 숏코드가 빈 제목 줄로 남는다

| | |
|---|---|
| 위치 | `02_prepare_data.py:57` (`SHORTCODE_RE` 로 삭제) → `:47` (`HEADING_RE` 불일치) |
| 현상 | `## {{% heading "whatsnext" %}}` → `## ` (제목 텍스트 없음). `HEADING_RE` 의 `(.+?)` 가 빈 문자열에는 매칭되지 않아 **섹션이 나뉘지 않고**, `## ` 줄이 본문에 그대로 남는다 |
| 규모 | 427회 / 287개 문서. CPT 샘플 1,211개 중 **334개**, chunks 2,997개 중 **330개** 에 빈 `##` 줄 |
| 영향 | "다음 내용", "시작하기 전에" 섹션이 앞 섹션 본문에 붙는다. 모델이 의미 없는 `## ` 줄을 학습한다 |
| 개선안 | `i18n/ko/ko.toml` 의 번역 문자열로 치환: `whatsnext`→`다음 내용`, `prerequisites`→`시작하기 전에`, `objectives`→`목적`, `cleanup`→`정리하기`, `seealso`→`더 보기`, `synopsis`→`시놉시스`, `options`→`옵션`, `envvars`→`환경 변수` |

### 8.2 ❌ `codenew` / `code_sample` 의 YAML 이 삽입되지 않는다

| | |
|---|---|
| 위치 | `02_prepare_data.py:57` |
| 현상 | `{{% code_sample file="debug/counter-pod.yaml" %}}` → 삭제. 사이트에서는 `content/ko/examples/debug/counter-pod.yaml` 의 내용이 코드 블록으로 들어간다 |
| 규모 | 239회 참조. 그중 199회는 `raw/website/content/ko/examples/` 에 파일이 있다 (약 **27,905 토큰**, 중복 포함). 40회는 한글 examples 에 없다 (`/controllers/job-success-policy.yaml` 등, 영어 `content/en/examples` 에 있을 것으로 추정) |
| 영향 | 쿠버네티스 지식의 핵심인 **매니페스트 예제가 빠진다**. 본문은 "다음 매니페스트를 사용한다." 로 끝나고 매니페스트가 없다 |
| 개선안 | 숏코드를 `examples/<file>` 의 내용을 담은 ` ```yaml ` 블록으로 치환한다. 한글에 없으면 `content/en/examples` 로 대체한다 (`LANGS="ko en"` 필요) |

### 8.3 ❌ `include` 내용이 삽입되지 않는다

| | |
|---|---|
| 현상 | `{{< include "task-tutorial-prereqs.md" >}}` → 삭제 |
| 규모 | 109회 / 100개 문서. `task-tutorial-prereqs.md` 가 95회 |
| 영향 | "시작하기 전에" 섹션의 공통 안내(클러스터와 kubectl 준비)가 빠진다. 단, 같은 내용이 95번 반복되므로 **넣으면 중복 학습**이 된다 |
| 개선안 | 넣되 문서당 한 번만 넣거나, 반복 안내문이라 일부러 빼거나. 어느 쪽이든 **의도적으로 결정**해야 한다 |

### 8.4 ❌ `glossary_definition` 내용이 삽입되지 않는다

| | |
|---|---|
| 현상 | `{{< glossary_definition term_id="kube-apiserver" length="all" >}}` → 삭제 |
| 규모 | 29회 / 21개 문서 (예: `concepts/architecture/_index.md` 의 컴포넌트 설명) |
| 개선안 | `docs/reference/glossary/<term_id>.md` 의 본문으로 치환한다 (`length="short"` 면 `<!--more-->` 앞부분만) |

### 8.5 ❌ 버전 숏코드(`skew`, `param`)가 빈칸이 된다

| | |
|---|---|
| 현상 | `쿠버네티스 {{< skew currentVersion >}}에는` → `쿠버네티스 에는` |
| 규모 | `skew` 171회 / 51개 문서, `param` 113회 / 53개 문서. 코드 블록 안의 것은 정제하지 않아 원문 그대로 남는다 |
| 영향 | 문법이 깨진 문장을 학습한다 |
| 개선안 | `hugo.toml` 의 값(`latest = "v1.37"`)으로 치환한다. 버전 정보를 넣지 않으려면 "현재 버전" 같은 말로 바꾼다 |

### 8.6 ❌ `comment` 숏코드 안의 숨김 내용이 노출된다

| | |
|---|---|
| 현상 | `{{< comment >}}번역자 메모...{{< /comment >}}` → 표식만 지우고 **안의 내용은 남긴다** |
| 규모 | 11회 / 8개 문서 |
| 개선안 | `{{< comment >}}` ~ `{{< /comment >}}` 를 내용째 삭제한다 |

### 8.7 ⚠️ `feature-state` 정보가 사라진다

`{{< feature-state for_k8s_version="v1.25" state="stable" >}}` (241회 / 116개 문서) → 삭제.
기능의 성숙도(alpha/beta/stable)와 도입 버전은 유용한 지식이다. `기능 상태: v1.25 [stable]` 같은 문장으로 바꿀 수 있다.

### 8.8 ⚠️ 탭 이름이 사라진다

`{{% tab name="리눅스" %}}리눅스 설명{{% /tab %}}{{% tab name="윈도우" %}}...` → `리눅스 설명\n\n윈도우 설명`.
어느 내용이 어느 OS 용인지 구분이 없어진다 (25개 문서). `**리눅스**` 같은 소제목으로 바꿀 수 있다.

### 8.9 ❌ 한 줄짜리 코드 펜스 때문에 섹션 분할이 뒤집힌다

| | |
|---|---|
| 위치 | `02_prepare_data.py:85–86` (`is_fence`), `:99–100`, `:117–118` |
| 현상 | `reference/kubectl/cheatsheet.md` 의 ` ```kubectl -A``` ` 줄을 펜스 하나로 세서 토글한다. 그 뒤로 **코드 안/밖이 뒤집혀** bash 주석 `# 기본 출력을 위한 Get 커맨드` 등이 제목으로 인식된다 |
| 규모 | 1개 문서 (펜스 수가 홀수인 문서는 이것뿐) |
| 영향 | kubectl 치트 시트(가장 실용적인 문서 중 하나)의 코드 블록이 섹션 경계에서 쪼개지고 `## # 주석` 같은 제목이 생긴다 |
| 개선안 | 펜스 줄이 ` ``` ` 를 두 번 포함하면 토글하지 않는다. 또는 펜스를 여닫는 쌍으로 추적한다 |

### 8.10 ⚠️ replay 문서를 자를 때 글자가 깨진다

| | |
|---|---|
| 위치 | `03_build_cpt.py:75–76` |
| 현상 | 토큰 목록을 중간에서 자른 뒤 `decode` 하면 한글 글자의 바이트가 반만 남아 `�` 가 된다 |
| 규모 | train 의 replay 샘플 **4개** (모두 끝 글자). 쿠버네티스 블록의 강제 분할(`:55`)은 실측 0개 |
| 개선안 | 디코드 결과 끝의 `�` 를 지우거나, 문장/줄 경계에서 자른다 |

### 8.11 ⚠️ valid 가 문서 단위로 분리되지 않는다

| | |
|---|---|
| 위치 | `03_build_cpt.py:124–126` |
| 현상 | 블록 단위로 섞어서 나눈다. 같은 문서의 다른 블록이 train 에 있다 |
| 규모 | valid 18개 중 **11개** (예: `deployment.md` 블록 11개 중 일부가 valid) |
| 영향 | `k8s_heldout` perplexity 가 "완전히 처음 보는 문서"보다 낮게 나온다 (같은 문서의 앞뒤 맥락을 학습했으므로) |
| 개선안 | 문서 단위로 valid 를 고른다. 또는 이 한계를 알고 결과를 해석한다 |

### 8.12 그 밖의 작은 것들

- headless 조각 문서(9개)가 단독 문서로 학습된다. include 를 구현하면(8.3) 중복이 된다.
- `mermaid` 다이어그램 코드가 본문에 남는다 (27회). 학습에 쓸모가 적은 노이즈다.
- `figure` 의 `alt` 텍스트(그림 설명)가 버려진다. 아키텍처 그림의 설명에 정보가 있다.
- front matter 의 `content_type`(concept/task/...)을 쓰지 않는다. 블록 헤더에 넣으면 문서 성격을 학습시킬 수 있다.

---

## 9. 직접 확인해 보는 명령 모음

모두 `~/llm-ft` 에서 `source .venv/bin/activate` 후에 실행한다.

**원문 vs 정제 결과**
```bash
python - <<'EOF'
from importlib import import_module; from pathlib import Path
prep = import_module("02_prepare_data")
p = Path("raw/website/content/ko/docs/concepts/workloads/pods/_index.md")
print(p.read_text()[:800]); print("=" * 60)
t, body = prep.parse_doc(p); print("TITLE:", t); print(body[:800])
EOF
```

**섹션 분할 결과**
```bash
python -c "
from importlib import import_module; from pathlib import Path
prep = import_module('02_prepare_data')
t, b = prep.parse_doc(Path('raw/website/content/ko/docs/concepts/workloads/pods/_index.md'))
for h, s in prep.split_sections(b): print(f'{len(s):5}자 | {h!r}')"
```

**한 문서의 블록 나누기 (상한별)**
```bash
python - <<'EOF'
from importlib import import_module; from pathlib import Path
from transformers import AutoTokenizer
b = import_module("03_build_cpt"); tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
t, body = b.prep.parse_doc(Path("raw/website/content/ko/docs/concepts/workloads/controllers/deployment.md"))
for n in (2048, 4096, 8192):
    bl = b.pack_document(t, body, tok, n)
    print(n, "->", [len(tok.encode(x)) for x in bl])
EOF
```

**숏코드 사용 현황**
```bash
grep -rhoE '\{\{[<%] *[/]?[a-zA-Z_-]+' raw/website/content/ko/docs --include='*.md' \
  | sed -E 's/\{\{[<%] *//' | sort | uniq -c | sort -rn | head -20
```

**출력 데이터 점검 (빈 제목 줄, 깨진 문자, 최대 길이)**
```bash
python - <<'EOF'
import json, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
rows = [json.loads(l)["text"] for l in open("data/cpt/train.jsonl", encoding="utf-8")]
strip = lambda t: re.sub(r"```.*?```", "", t, flags=re.S)
print("빈 제목 줄:", sum(bool(re.search(r"(?m)^#{1,4}\s*$", strip(t))) for t in rows))
print("깨진 문자 �:", sum("�" in t for t in rows))
print("최대 토큰:", max(len(tok.encode(t)) for t in rows))
EOF
```

**번역 문자열·버전 변수 보기 (작업 트리에 없는 파일)**
```bash
git -C raw/website show HEAD:i18n/ko/ko.toml | grep -A1 '_heading\]'
git -C raw/website show HEAD:hugo.toml | grep -nE '^\s*(latest|version)\s*='
```

---
← [모듈 3. 데이터 파이프라인](03_data.md) · 다음: [모듈 4. 계속 사전학습 →](04_training.md)
