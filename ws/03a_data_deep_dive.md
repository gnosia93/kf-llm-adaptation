# 모듈 3-A. 데이터 전처리 상세 해설 — 원본 구조부터 함수·라인별 동작까지

> [모듈 3](03_data.md) 이 "무엇을 하는가"의 개요라면, 이 문서는 **"정확히 어떻게 하는가"** 를 다룬다.
> 학습 코드를 보기 전에 학습 데이터 한 줄이 어떻게 만들어지는지 끝까지 따라가는 것이 목표다.

**이 문서의 기준 (재현 조건)**

| 항목 | 값 |
|---|---|
| 원본 저장소 | `kubernetes/website` 커밋 `80f8465` (2026-11-18) |
| 작업 디렉터리 | `~/llm-ft` (모든 경로는 여기 기준 상대 경로) |
| 토크나이저 | `Qwen/Qwen2.5-3B` |
| 블록 상한 | 8,192 토큰 (g6e 기본값. Mac 은 2,048) |
| 시드 | 42 |
| 라인 번호 | 현재 `01_download.sh`, `02_prepare_data.py`, `03_build_cpt.py` 기준 |

문서에 나오는 숫자는 모두 이 조건에서 **실제로 측정한 값**이다. 같은 조건이면 같은 값이 나와야 한다.

---

## 목차

1. [전체 흐름과 경로](#1-전체-흐름과-경로)
2. [원본 데이터 구조](#2-원본-데이터-구조)
3. [`01_download.sh` — 라인별 해설](#3-01_downloadsh--라인별-해설)
4. [`02_prepare_data.py` — 정제 엔진](#4-02_prepare_datapy--정제-엔진)
5. [`03_build_cpt.py` — 패킹과 데이터셋 구성](#5-03_build_cptpy--패킹과-데이터셋-구성)
6. [출력 파일 형식](#6-출력-파일-형식)
7. [블록 길이(JSONL 한 줄의 길이) 선택 고려사항](#7-블록-길이jsonl-한-줄의-길이-선택-고려사항)
8. [수정 이력: 검증으로 찾아 고친 문제들](#8-수정-이력-검증으로-찾아-고친-문제들)
9. [남아 있는 한계](#9-남아-있는-한계)
10. [직접 확인해 보는 명령 모음](#10-직접-확인해-보는-명령-모음)

---

## 1. 전체 흐름과 경로

```
                     ┌────────────────────────── 입력 (raw/website/) ──────────────────────────┐
GitHub               content/ko/docs/**/*.md          문서 본문 569개 (headless 8개 제외 → 561)
kubernetes/website ─▶ content/ko/examples/**          예제 YAML 등 (code_sample 숏코드)
 01_download.sh       content/ko/includes/*.md        공통 조각 (include 숏코드)
                      content/en/examples/**          한글에 없는 예제의 대체본
                      content/en/docs/.../feature-gates/*.md   기능 게이트 단계·버전
                      i18n/{ko,en}/*.toml             번역 문자열 (heading, feature-state)
                      data/releases/schedule.yaml     패치 버전 (skew)
                      hugo.toml                       현재 버전 (skew, param)
                     └──────────────────────────────────────────────────────────────────────────┘
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
   02_prepare_data.py (SFT·RAG용)                        03_build_cpt.py (CPT용)
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
```

- 정제 로직은 모두 `02_prepare_data.py` 에 있다. `03_build_cpt.py` 는 그것을 **import 해서 재사용**한다 (`03_build_cpt.py:22`).
  정제 규칙을 고치면 SFT, RAG, CPT 데이터에 한꺼번에 반영된다.
- CPT 학습에 쓰는 것은 **`data/cpt/` 와 `data/eval/`** 이다. `data/sft/` 와 `data/chunks.jsonl` 은 모듈 6 (SFT, RAG) 에서 쓴다.

실행 명령 (항상 `~/llm-ft` 에서):

```bash
./01_download.sh
python 02_prepare_data.py
python 03_build_cpt.py                    # g6e: 8,192 토큰 블록
python 03_build_cpt.py --max-tokens 2048 --batch-size 2   # Mac(MLX)
# python -I 로 실행하면 형제 모듈 02_prepare_data 를 import 하지 못한다
```

---

## 2. 원본 데이터 구조

### 2.1 저장소와 받아 오는 범위

`kubernetes/website` 는 kubernetes.io 사이트의 소스다. 정적 사이트 생성기 **Hugo** 로 빌드한다.
문서 본문은 마크다운이지만, **사이트에 보이는 최종 내용은 다른 파일들을 조합해서 만들어진다.**
그래서 본문만이 아니라 조합에 필요한 자원도 함께 받는다.

```
raw/website/                              ← git 저장소 루트 (sparse checkout)
├── hugo.toml                             사이트 설정. [params] version = "v1.37", latest = "v1.37"
├── i18n/
│   ├── ko/ko.toml                        한글 번역 문자열 (whatsnext_heading = "다음 내용" ...)
│   └── en/en.toml                        영어 (한글에 없는 키의 대체본)
├── data/releases/schedule.yaml           릴리스·패치 일정 (1.37 의 최신 패치 = 1.37.1)
└── content/
    ├── ko/
    │   ├── docs/              ★ 569      문서 본문 ← 학습 데이터
    │   ├── examples/          ☆          문서가 끼워 넣는 YAML 등 (216개 .yaml 외)
    │   ├── includes/          ☆ 4        여러 문서가 끼워 넣는 공통 조각
    │   └── blog/ community/ partners/ releases/ training/ _common-resources/  (미사용)
    └── en/
        ├── examples/          ☆          한글에 없는 예제는 사이트도 영어판을 쓴다
        └── docs/reference/command-line-tools-reference/feature-gates/   ☆ 491   기능 게이트 정보
```

`content/ko` 의 파일 형식 분포 (실측):

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

> 용어집(`reference/glossary/`)이 문서 수의 22% 를 차지하지만 대부분 아주 짧다 (예: `glossary/pod.md` ≈ 128 토큰).
> 문서 **개수**와 **토큰 양**은 다르게 봐야 한다 (7.2 참고).

### 2.3 파일의 종류

| 종류 | 경로 예 | 특징 | 코드에서의 처리 |
|---|---|---|---|
| 일반 페이지 | `concepts/architecture/nodes.md` | front matter + 본문 | 그대로 |
| 섹션 인덱스 | `concepts/workloads/pods/_index.md` | 디렉터리의 대표 페이지 | `title` 이 없으면 디렉터리 이름이 제목 (`02:388–389`) |
| 페이지 번들 | `.../feature-gates/index.md` | 디렉터리 + `index.md` | 위와 같음 |
| 용어집 | `reference/glossary/pod.md` | `id`, `short_description`, `aka` 등 메타데이터가 많음 | `title` + 본문만. `glossary_definition` 의 출처로도 쓰임 |
| **headless 조각** | `tasks/tools/included/verify-kubectl.md` 등 **8개** | front matter `headless: true`. 단독 페이지가 아니라 include 로 끼워 넣는 조각 | **단독 문서에서 제외** (`02:399–406`). include 될 때만 들어감 |
| front matter 없음 | `contribute/generate-ref-docs/prerequisites-ref-docs.md` (1개) | `---` 로 시작하지 않음 | 제목 = 파일명 |

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
| `title` | 568 | 페이지 제목 | ✅ 블록 헤더 `# 제목` |
| `weight` | 402 | 메뉴 정렬 순서 | ✗ |
| `content_type` | 366 | concept / task / tutorial / reference | ✗ (9절) |
| `id`, `tags`, `short_description`, `aka`, `full_link` | 116~123 | 용어집 메타데이터 | ✗ |
| `description` | 90 | 요약 | ✗ |
| `headless` | 9 (그중 `true` 8) | 조각 문서 표시 | ✅ 단독 문서에서 제외 |

### 2.5 Hugo 숏코드 전체 현황과 처리 방식

숏코드는 `{{< 이름 인자 >}}` 또는 `{{% 이름 인자 %}}` 형식의 Hugo 템플릿 호출이다. 사이트를 빌드할 때 **다른 내용으로 바뀐다.**
처리 원칙은 **"사이트가 렌더링한 결과에 가깝게"** 다. 각 숏코드의 동작은 저장소의 `layouts/shortcodes/<이름>.html` 을 읽고 그대로 옮겼다.

| 숏코드 | 등장 | 사이트에서 바뀌는 내용 | 처리 (`02_prepare_data.py`) | 예 (실제 출력) |
|---|---|---|---|---|
| `glossary_tooltip` | 861 | 용어 링크 | `text` 값 (없으면 `term_id`) | `파드를 만든다.` |
| `note` / `caution` / `warning` | 580 / 59 / 50 | 강조 상자 | `참고:` / `주의:` / `경고:` | |
| `heading` | 427 | 번역된 제목 | `i18n/ko/ko.toml` 의 `<키>_heading` | `## 다음 내용` |
| `feature-state` | 241 | 기능 상태 상자 | 수동형: 인자 그대로. 기능 게이트형(72회): `feature-gates/<이름>.md` 의 단계 중 현재 버전 이하의 마지막 것 | `기능 상태: Kubernetes v1.33 [stable]` |
| `skew` | 171 | 버전 문자열 | `hugo.toml` + `schedule.yaml` 로 계산 (코드 블록 안도 포함) | `1.37`, `1.37.1`, `1-36` |
| `param` | 113 | 사이트 변수 | `version` → `v1.37` (코드 블록 안도 포함) | `.../kubernetes-api/v1.37/#node` |
| `codenew` / `code_sample` | 140 / 99 | 예제 파일 내용 | `examples/<file>` 을 읽어 언어가 표시된 코드 블록으로. 한글에 없으면 영어판 | `` `debug/counter-pod.yaml` `` + ` ```yaml ... ``` ` |
| `include` | 109 (+탭 6) | 공통 조각 내용 | `includes/<이름>*` 또는 문서 기준 상대 경로의 파일을 **재귀적으로 정제**해서 삽입 (깊이 3까지) | "쿠버네티스 클러스터가 필요하고, ..." |
| `glossary_definition` | 29 | 용어 정의 | `glossary/<term_id>.md` 본문. `length="short"` 면 첫 문단 | "API 서버는 쿠버네티스 API를 노출하는 ..." |
| `tabs` / `tab` | 44 / 108 | 탭 UI | 탭 이름을 `**이름**` 줄로. `codelang` 탭은 코드 블록으로, `include` 탭은 파일 삽입 | `**리눅스**` |
| `comment` | 11 | **화면에 안 보이는 주석** | 내용째 삭제 | |
| `mermaid` | 27 | 다이어그램 | 내용째 삭제 | |
| `figure` | 20 | 이미지 | `(그림: 캡션 또는 alt)` | `(그림: 클러스터 구성 요소)` |
| `version-check`, `thirdparty-content` 등 | 63, 28 | 반복 안내문 | 삭제 | |
| `table`, `alert` 등 나머지 | | 감싸는 표식 | 표식만 삭제, 내용은 남김 | |
| 이스케이프 `{{</* ... */>}}` | 4 | 숏코드 **문법 예시** (기여 가이드) | `{{< ... >}}` 글자 그대로 복원 | `{{% code_sample file="..." %}}` |

정제 결과 집계 (`python 02_prepare_data.py` 출력, 실측):

```
숏코드 치환: code_sample 239, code_sample_missing 0, include 115, include_missing 0,
            glossary_definition 29, glossary_definition_missing 0, feature-state 241, skew 171, headless_skipped 8
```

모든 참조가 원본 파일을 찾았다 (`_missing` 0). code_sample 239건 중 40건은 영어판 `content/en/examples` 에서 찾았다.

---

## 3. `01_download.sh` — 라인별 해설

```bash
 6  set -euo pipefail
 7  LANGS=${LANGS:-ko}
```
- `6`: 명령이 하나라도 실패하면(`-e`), 정의되지 않은 변수를 쓰면(`-u`), 파이프 중간에서 실패하면(`pipefail`) 즉시 멈춘다.
- `7`: 받을 언어 목록. 기본값은 `ko`. `LANGS="ko en" ./01_download.sh` 처럼 여러 개를 줄 수 있다.

```bash
 9  mkdir -p raw
10  if [ ! -d raw/website ]; then
11    git clone --depth 1 --filter=blob:none --sparse \
12      https://github.com/kubernetes/website.git raw/website
13  fi
```
- `10`: 이미 받았으면 clone 을 건너뛴다. 다시 실행해도 안전하다.
- `11`: `--depth 1` 최신 커밋만 / `--filter=blob:none` 파일 내용은 필요할 때만 / `--sparse` 처음엔 루트 파일만 작업 트리에 풀기.
  루트 파일(`hugo.toml` 등)은 sparse checkout 에서도 **항상 포함**된다.

```bash
21  git -C raw/website sparse-checkout set $(printf 'content/%s ' $LANGS) \
22    content/en/examples \
23    content/en/docs/reference/command-line-tools-reference/feature-gates \
24    i18n data/releases
```
- 작업 트리에 풀 디렉터리를 **이 목록으로 교체**한다.
  - 문서 본문: `content/<언어>`
  - 정제에 필요한 자원: 영어 예제, 기능 게이트, 번역, 릴리스 일정 (14~19행 주석 참고)
- ⚠️ "추가"가 아니라 "교체"다. `ko en` 으로 받은 뒤 `ko` 로 다시 실행하면 `content/en/docs` 본문은 작업 트리에서 빠진다.
  예제와 기능 게이트는 목록에 있으므로 남는다.

```bash
26  for l in $LANGS; do
27    echo "[$l] 마크다운 문서 수: $(find raw/website/content/$l/docs -name '*.md' | wc -l)"
28  done
```
- 확인용 출력. `[ko] 마크다운 문서 수: 569` (headless 포함 개수).

---

## 4. `02_prepare_data.py` — 정제 엔진

### 4.1 파일 구성

| 라인 | 구역 | 내용 |
|---|---|---|
| 23–38 | SFT 템플릿 | `SYSTEM_PROMPT`, 질문 템플릿 (CPT 에는 안 씀) |
| 41–47 | 문서 구조 정규식 | front matter, 제목, 펜스, headless |
| 50–75 | 숏코드 정규식 | `_sc()` / `_close()` 로 생성 |
| 78–89 | 마크다운/HTML 정규식, 상수 | 링크, 이미지, 태그, 자리표시자 |
| 93–164 | 사이트 정보 | `Site`, `load_site()`, `Page`, `page_for()` |
| 168–202 | 코드 블록 판별 | `fence_states()`, `split_code()` |
| 206–338 | 숏코드 치환 | `_versions`, `_feature_state`, `_include`, `_glossary_definition`, `_code_sample`, `expand_shortcodes` |
| 341–372 | 정제 | `clean_prose`, `clean_code`, `clean_markdown` |
| 376–406 | 문서 읽기 | `read_front_matter`, `parse_doc`, `is_headless`, `iter_docs` |
| 410–446 | 나누기 | `split_sections`, `split_long` |
| 450–541 | SFT/RAG 출력 | `build_chunks`, `split_by_document`, `to_sft`, `write_jsonl`, `print_stats`, `main` |

**호출 흐름** (문서 하나):

```
iter_docs(docs_dir)                                         399
 ├─ is_headless(path) → True 면 건너뜀                        393
 └─ parse_doc(path)                                         381
     ├─ read_front_matter(raw) → (meta, body)               376
     ├─ TITLE_RE 로 제목
     └─ clean_markdown(body, page_for(path))                358
         ├─ COMMENT_RE / COMMENT_SC_RE / MERMAID_SC_RE 삭제   360–362
         ├─ split_code(text) → [(코드?, 조각)]                193
         │   └─ fence_states(lines)                         168
         ├─ 코드 조각 → clean_code()                          352
         │   └─ _versions()                                 210
         ├─ 본문 조각 → expand_shortcodes() → clean_prose()   305 / 341
         │   ├─ _code_sample() / _include() / _glossary_definition()   ← 재귀적으로 clean_markdown 호출
         │   └─ _feature_state() / _versions()
         └─ 자리표시자 복원, 빈 줄 정리                          370–372
```

### 4.2 정규식

**문서 구조 (`41–47`)**

| 라인 | 이름 | 패턴 | 하는 일 |
|---|---|---|---|
| 41 | `FRONT_MATTER_RE` | `\A---\s*\n(.*?)\n---\s*\n` (`re.S`) | 파일 **맨 앞**(`\A`)의 `---` ~ `---`. 그룹1 = YAML |
| 42 | `TITLE_RE` | `^title:\s*(.+?)\s*$` (`re.M`) | front matter 의 `title:` 값 |
| 43 | `HEADLESS_RE` | `^headless:\s*true\s*$` (`re.M`) | 조각 문서 표시 |
| 44 | `COMMENT_RE` | `<!--.*?-->` (`re.S`) | HTML 주석 (여러 줄 가능) |
| 45 | `FENCE_OPEN_RE` | ``[ \t]*(`{3,}\|~{3,})(.*)$`` | 펜스 후보 줄. 그룹1 = 펜스 문자열, 그룹2 = 나머지 (언어 표시 등) |
| 46 | `HEADING_RE` | `^(#{1,4})\s+(.+?)\s*#*\s*$` | 제목 줄. 그룹1 = 레벨, 그룹2 = 제목 |
| 47 | `ANCHOR_RE` | `\s*\{#[^}]*\}` | 제목 뒤 앵커 `{#overview}` |

> `HEADING_RE` 가 `#{1,4}` 인 이유: `#####` 이상은 드물게 쓰이는 세부 제목이라 섹션을 나누지 않고 본문으로 둔다.

**숏코드 정규식 생성기 (`50–56`)**

```python
50 def _sc(name, args=r"(?P<args>[^}]*?)"):
51     return rf"\{{\{{[<%]\s*{name}\b{args}\s*/?\s*[>%]\}}\}}"
54 def _close(name):
55     return rf"\{{\{{[<%]\s*/\s*{name}\s*[>%]\}}\}}"
```

- `_sc("tab")` 는 `{{< tab name="x" >}}`, `{{% tab name="x" %}}`, 자기 닫는 `{{< tab include="f" />}}` 를 모두 잡는다.
  `\b` 덕분에 `tabs` 는 잡지 않는다.
- `_close("mermaid")` 는 `{{< /mermaid >}}` 와 `{{</ mermaid >}}` 를 모두 잡는다. 실제 데이터에 두 형태가 섞여 있다 (18회 / 9회).
- `rf"..."` 안의 `\{{\{{` 는 f-string 에서 `{` 를 쓰기 위한 이스케이프다. 정규식으로는 `\{\{` 가 된다.

**숏코드 정규식 (`58–75`)** — 각각의 처리는 4.5 에서 설명한다.

| 라인 | 이름 | 잡는 것 |
|---|---|---|
| 58 | `ESCAPED_SC_RE` | `{{</* ... */>}}` 이스케이프된 숏코드 (문법 예시) |
| 59 | `COMMENT_SC_RE` | `{{< comment >}}` ~ `{{< /comment >}}` 전체 |
| 60 | `MERMAID_SC_RE` | `{{< mermaid >}}` ~ 닫는 태그 전체 |
| 61 | `HEADING_SC_RE` | `{{% heading "whatsnext" %}}` → 그룹 `key` |
| 62 | `CODE_SAMPLE_RE` | `code_sample` 또는 `codenew` |
| 63 | `TAB_CODE_RE` | `codelang` 이 있는 탭과 그 본문, 닫는 태그까지 |
| 64 | `TAB_RE` | 나머지 탭 여는 태그 |
| 65 | `INCLUDE_RE` | `{{< include "이름" >}}` → 그룹 `name` |
| 66–70 | `GLOSSARY_DEF_RE`, `FEATURE_STATE_RE`, `FIGURE_RE`, `SKEW_RE`, `PARAM_RE` | 각 숏코드 |
| 71–72 | `TOOLTIP_TEXT_RE`, `TOOLTIP_TERM_RE` | 용어 툴팁 (`text` 우선, 없으면 `term_id`) |
| 73 | `ADMONITION_RE` | `note` / `caution` / `warning` 여는 태그 |
| 74 | `SHORTCODE_RE` | **남은 모든 숏코드** (최후의 정리) |
| 75 | `ARG_RE` | `키="값"` 인자 추출 |

**마크다운 / HTML (`78–87`)**

| 라인 | 이름 | 하는 일 | 예 |
|---|---|---|---|
| 78 | `IMAGE_RE` | 이미지 삭제 | `![a](/x.svg)` → `` |
| 79 | `LINK_RE` | 링크 → 텍스트 | `[컨테이너 런타임](/docs/..)` → `컨테이너 런타임` |
| 80–84 | `HTML_TAG_RE` | **지정한 태그 이름만** 삭제 | `줄<br>바꿈` → `줄바꿈` |
| 86 | `PLACEHOLDER_RE` | 자리표시자 `\x00번호\x00` (4.6) | |
| 87 | `MAX_INCLUDE_DEPTH` | include 재귀 최대 깊이 3 | |

> HTML 태그를 전부(`<[^>]+>`) 지우지 않는 이유: 본문의 `<pod-name>`, `<namespace>` 같은 **자리 표시자까지** 사라지기 때문이다.

### 4.3 사이트 정보: `Site`, `load_site`, `page_for` (`93–164`)

숏코드를 렌더링하려면 "이 문서가 어느 사이트·언어에 속하나"를 알아야 한다.

```python
 94 class Site:
 95     repo: Path        # raw/website
 96     lang: str         # ko
 97     i18n: dict        # {"whatsnext_heading": "다음 내용", ...}  (없는 키는 영어로 대체)
 98     version: str      # hugo.toml params.version  예: "v1.37"
 99     latest: str       # hugo.toml params.latest
100     patch: str        # 현재 버전의 최신 패치  예: "1.37.1"
109     def find(self, *relative):
110         """content/<lang>/ 에서 찾고, 없으면 content/en/ (사이트 기본 언어)에서 찾는다."""
```

- `find()` (`109–115`): 사이트의 대체 규칙을 그대로 따른다. `code_sample.html` 은 한글에 파일이 없으면 기본 언어(영어)에서 찾는다.
- `_load_i18n()` (`118–125`): **영어를 먼저 읽고 한글로 덮어쓴다.** 한글에 없는 키는 영어 값이 남는다.
  예를 들어 `feature_gate_stage_stable` 은 한글에 없어서 `Stable` 이 된다.
- `_patch_version()` (`128–137`): `schedule.yaml` 에서 `release: "1.37"` 항목의 `previousPatches` 첫 번째 값 (`1.37.1`).
  Hugo `skew.html` 의 `currentPatchVersion` 과 같은 규칙이다. 없으면 `1.37.0`.
- `load_site()` (`140–147`): `@functools.lru_cache` 로 저장소·언어당 **한 번만** 읽는다.
- `page_for(path)` (`156–164`): 경로에서 **마지막 `content` 디렉터리**를 찾는다. 그 위가 저장소, 바로 아래가 언어다.
  ```
  raw/website/content/ko/docs/concepts/.../logging.md
  └─ repo ─┘         └lang┘
  ```
  이 구조가 아니면 `None` 을 돌려준다. 그러면 사이트 자원이 필요한 치환(예제, include, 번역, 버전)은 건너뛰고 기본 정리만 한다.

### 4.4 코드 블록 판별: `fence_states`, `split_code` (`168–202`)

코드 블록 안은 **정제하면 안 된다.** YAML, kubectl 명령, Go 템플릿(`{{ .data }}`)이 망가지기 때문이다.
그래서 "이 줄이 코드인가"를 정확히 아는 것이 전처리 전체의 기반이다.

```python
174     fence, states = None, []                     # fence: 지금 열려 있는 펜스 문자열 (없으면 None)
175     for line in lines:
176         if fence is None:                        # ── 코드 밖
177             m = FENCE_OPEN_RE.match(line)
178             if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
179                 fence = m.group(1)               # 여는 펜스: 이 줄부터 코드
180                 states.append((True, True))
181             else:
182                 states.append((False, False))    # 일반 줄
183         else:                                    # ── 코드 안
184             s = line.strip()
185             if s and set(s) == {fence[0]} and len(s) >= len(fence):
186                 fence = None                     # 닫는 펜스: 이 줄까지 코드
187                 states.append((True, False))
188             else:
189                 states.append((True, True))
```

돌려주는 값은 줄마다 `(이 줄이 코드인가, 이 줄 다음에도 코드가 열려 있나)` 이다.

| 줄 | `fence` 전 → 후 | 반환 | 설명 |
|---|---|---|---|
| `본문` | None → None | (False, False) | |
| `` ```bash `` | None → `` ``` `` | (True, True) | 여는 펜스 |
| `# 주석` | `` ``` `` | (True, True) | **코드 안의 #** — 제목이 아님 |
| `` ``` `` | `` ``` `` → None | (True, False) | 닫는 펜스 |
| `` ```kubectl -A``` `` | None → None | (False, False) | **같은 줄에서 열고 닫음** → 펜스 아님 (178행) |

- **178행:** CommonMark 규칙상 백틱 펜스의 정보 문자열에는 백틱이 들어갈 수 없다. 그래서 같은 줄에 백틱이 또 있으면 펜스가 아니다.
  예전 코드는 이 줄을 펜스로 세서 `cheatsheet.md` 의 코드 안/밖이 뒤집혔다 (8.9).
- **185행:** 닫는 펜스는 **여는 펜스와 같은 문자로만** 된 줄이고, 길이가 같거나 길어야 한다. `` ```` `` 로 연 블록은 `` ``` `` 로 닫히지 않는다.
- 닫히지 않은 펜스는 문서 끝까지 코드로 본다 (CommonMark 와 같다).

`split_code(text)` (`193–202`) 는 이 판정으로 텍스트를 **연속된 코드 줄 / 본문 줄 묶음**으로 나눈다.

```
"앞 문장\n```yaml\na: 1\n```\n뒤 문장"
  → [(False, '앞 문장'), (True, '```yaml\na: 1\n```'), (False, '뒤 문장')]
```

### 4.5 숏코드 치환 (`206–338`)

#### `_versions(site, text)` (`210–237`) — 버전 숏코드

`layouts/shortcodes/skew.html` 의 규칙을 옮겼다. 현재 `version = latest = v1.37` 이다.

| 숏코드 | 계산 | 결과 |
|---|---|---|
| `{{< skew currentVersion >}}` | version 에서 v 제거 | `1.37` |
| `{{< skew latestVersion >}}` | latest 에서 v 제거 | `1.37` |
| `{{< skew currentPatchVersion >}}` | `Site.patch` | `1.37.1` |
| `{{< skew nextMinorVersion >}}` / `prevMinorVersion` / `oldestMinorVersion` | latest 의 minor ±1 / −1 / −2 | `1.38` / `1.36` / `1.35` |
| `{{< skew currentVersionAddMinor -1 "-" >}}` | major + 구분자 + (minor + n) | `1-36` |
| `{{< param "version" >}}` | `hugo.toml` params | `v1.37` |

- 인자는 `shlex.split` 으로 나눈다 (`213`). 따옴표로 묶인 구분자 `"-"` 를 올바르게 처리하기 위해서다.
- **코드 블록 안에도 적용한다** (`clean_code`, 4.6). Hugo 는 코드 펜스 안의 숏코드도 처리한다.
  예: `curl -LO https://dl.k8s.io/release/v{{< skew currentPatchVersion >}}/bin/...` → `.../v1.37.1/bin/...`

#### `_feature_state(site, args)` (`240–256`) — 기능 상태

```python
243     state, version = a.get("state"), a.get("for_k8s_version", site.version)   # 수동형
244     if a.get("feature_gate_name"):                                             # 기능 게이트형
246         path = site.find("docs/reference/command-line-tools-reference/feature-gates", f"{...}.md")
247         current = (1, 37)                                                      # 사이트 버전
249             for stage, frm in re.findall(r'-\s*stage:\s*(\w+).*?fromVersion:\s*"?([\d.]+)', ...):
250                 if tuple(...frm...) <= current:                                # 현재 버전 이하의 마지막 단계
251                     state, version = stage, frm
252         if not state:                                                          # 게이트 파일이 없으면
253             return "기능 상태: 기능 게이트 <이름>"
```

- 수동형 (`for_k8s_version="v1.25" state="stable"`, 168회): 인자 그대로 쓴다. 버전이 없으면 사이트 버전.
- 기능 게이트형 (`feature_gate_name="SidecarContainers"`, 72회): 기능 게이트 파일의 `stages` 목록에서 **`fromVersion` 이 현재 버전(1.37) 이하인 마지막 단계**를 고른다.
  `SidecarContainers` 는 alpha 1.28 → beta 1.29 → stable 1.33 이므로 `v1.33 [stable]`.
- 출력: `기능 상태: Kubernetes v1.33 [stable]`. `기능 상태:` 는 한글 i18n, `Kubernetes` 는 한글 키가 없어 영어 대체본이다. 사이트 표시와 같다.

#### `_include(page, name, depth)` (`265–274`) — 공통 조각

```python
268     candidates = sorted((site.content / "includes").glob(f"{name}*")) + [page.path.parent / name]
269     for path in candidates:
270         if path.is_file() and path.name != "index.md":
272             return clean_markdown(_read_body(path), Page(path, site), depth + 1)
```

- `layouts/shortcodes/include.html` 과 같은 순서로 찾는다:
  ① `includes` 번들에서 이름으로 시작하는 파일 ② 현재 문서 디렉터리 기준 상대 경로.
  - `task-tutorial-prereqs.md` → ①로 `content/ko/includes/task-tutorial-prereqs.md`
  - `included/verify-kubectl.md` → ②로 `docs/tasks/tools/included/verify-kubectl.md` (headless 조각)
- **재귀 정제** (`272`): 끼워 넣을 파일도 숏코드·링크를 포함하므로 `clean_markdown` 을 다시 호출한다.
  `depth + 1` 로 깊이를 세고, `MAX_INCLUDE_DEPTH`(3) 에 도달하면 더 들어가지 않는다 (`327`). 서로를 include 하는 무한 루프를 막는다.

#### `_glossary_definition(page, args, depth)` (`277–289`) — 용어 정의

- `docs/reference/glossary/<term_id>.md` 를 읽어 재귀 정제한다.
- `length="short"` 면 **첫 문단만** (`284–285`). 사이트도 렌더링된 첫 `<p>` 만 보여준다. `all` / `long` 이면 전체.
- `prepend="..."` 가 있으면 앞에 붙인다.

#### `_code_sample(page, args)` (`292–302`) — 예제 파일

```python
295     file = a.get("file", "").lstrip("/")                         # "/controllers/x.yaml" 같은 앞 / 제거
296     path = page.site.find("examples", file) or (...문서 디렉터리 기준...)
301     lang = a.get("language") or Path(file).suffix.lstrip(".")    # 확장자 → 코드 블록 언어
302     return f"`{file}`\n\n```{lang}\n{...파일 내용...}\n```"
```

- 사이트처럼 **파일 경로를 코드 블록 위에 표시**한다. 본문에서 "`application/nginx-app.yaml` 을 적용한다" 같은 문장과 연결된다.
- 239건 모두 찾았다. 그중 40건은 영어 `content/en/examples` 에서 찾았다 (한글 저장소에 아직 번역·복사되지 않은 예제).

#### `expand_shortcodes(text, page, depth, vault)` (`305–338`) — 치환 순서

```python
311  ① 이스케이프 {{</* x */>}}      → keep("{{< x >}}")        문법 예시를 글자 그대로 보존
315  ② heading                       → "다음 내용" 등
316  ③ code_sample / codenew         → keep(코드 블록)
317  ④ tab (codelang)                → "**이름**" + keep(코드 블록)
326  ⑤ tab (나머지)                  → "**이름**" (+ include 탭이면 keep(파일 내용))
328  ⑥ include                       → keep(재귀 정제 결과)
329  ⑦ glossary_definition           → keep(재귀 정제 결과)
330  ⑧ feature-state                 → "기능 상태: ..."
337  ⑨ figure                        → "(그림: ...)"
338  ⑩ skew / param                  → 버전 문자열
```

- **`keep()`** (`307–309`): 결과를 `vault` 리스트에 넣고 자리표시자 `\x00번호\x00` 를 대신 둔다.
  이어서 실행되는 `clean_prose()` 가 **이미 완성된 코드 블록이나 정제가 끝난 include 결과를 다시 건드리지 않게** 하려는 것이다.
  예를 들어 YAML 안의 `[a](b)` 가 링크 규칙에 걸리면 안 된다.
- **①이 맨 앞:** 이스케이프된 문법 예시가 ②~⑩ 이나 `SHORTCODE_RE` 에 걸려 지워지지 않게 먼저 보관한다.
- **④가 ⑤보다 먼저:** `codelang` 탭은 본문과 닫는 태그까지 한 번에 잡아야 한다. ⑤의 일반 탭 규칙이 여는 태그만 먼저 바꿔 버리면 안 된다.
- `page` 가 없으면 (`312`) ①만 하고 끝낸다.

### 4.6 정제: `clean_prose`, `clean_code`, `clean_markdown` (`341–372`)

**`clean_prose(text)` (`341–349`)** — 치환 후 남은 것을 정리한다.

```python
343     text = TOOLTIP_TEXT_RE.sub(r"\1", text)      # ① text="..." 툴팁 → 텍스트
344     text = TOOLTIP_TERM_RE.sub(r"\1", text)      # ② 남은 툴팁 → term_id
345     text = ADMONITION_RE.sub(...)                # ③ note → 참고:
346     text = SHORTCODE_RE.sub("", text)            # ④ 나머지 숏코드 (닫는 태그, tabs, version-check ...) 삭제
347     text = IMAGE_RE.sub("", text)                # ⑤ 이미지 삭제
348     text = LINK_RE.sub(r"\1", text)              # ⑥ 링크 → 텍스트
349     return HTML_TAG_RE.sub("", text)             # ⑦ 지정한 HTML 태그 삭제
```

순서의 이유:
- ①②③이 ④보다 먼저: ④는 모든 숏코드를 지우므로, 남길 텍스트를 먼저 꺼내 둬야 한다.
- ①이 ②보다 먼저: `text`(한글)가 있으면 그것을 쓰고, 없을 때만 `term_id`(영문 ID)를 쓴다.
- ⑤가 ⑥보다 먼저: 이미지 `![a](b)` 는 링크 `[a](b)` 를 포함한다. 먼저 지우지 않으면 `!a` 가 남는다.

**`clean_code(text, page)` (`352–355`)** — 코드 블록 안에서는 **두 가지만** 한다. 이스케이프 복원과 버전 숏코드 치환이다. 나머지는 원문 그대로 둔다.

**`clean_markdown(text, page=None, depth=0)` (`358–372`)** — 전체 조립:

```python
360     text = COMMENT_RE.sub("", text)          # HTML 주석
361     text = COMMENT_SC_RE.sub("", text)       # {{< comment >}} 숨김 주석 (내용째)
362     text = MERMAID_SC_RE.sub("", text)       # 다이어그램 (내용째)
364     for is_code, seg in split_code(text):
365         if is_code:  parts.append(clean_code(seg, page))
367         else:        parts.append(clean_prose(expand_shortcodes(seg, page, depth, vault)))
369     text = "\n".join(parts)
370     while PLACEHOLDER_RE.search(text):       # 자리표시자 복원 (중첩 대비 반복)
371         text = PLACEHOLDER_RE.sub(lambda m: vault[int(m.group(1))], text)
372     return re.sub(r"\n{3,}", "\n\n", text).strip()   # 3줄 이상 빈 줄 → 1줄
```

- `360–362` 는 코드/본문을 나누기 **전에** 한다. `comment` 나 `mermaid` 블록 안에 코드 펜스가 있을 수 있어서, 나눈 뒤에는 블록 전체를 한 번에 잡을 수 없다.
- 자리표시자 문자 `\x00` 은 문서에 나오지 않는 제어 문자라서 다른 정규식과 충돌하지 않는다.

### 4.7 문서 읽기: `parse_doc`, `is_headless`, `iter_docs` (`376–406`)

```python
381 def parse_doc(path):
383     meta, body = read_front_matter(path.read_text(encoding="utf-8"))
384     title = path.stem                            # 기본값: 파일명
385     t = TITLE_RE.search(meta)
386     if t:
387         title = t.group(1).strip("'\"")          # "파드" → 파드
388     if title in ("_index", "index"):
389         title = path.parent.name                 # 섹션/번들인데 title 이 없으면 디렉터리 이름
390     return title, clean_markdown(body, page_for(path))
```

| 상황 | 결과 |
|---|---|
| 일반 (`title: 파드`) | `("파드", 정제된 본문)` |
| front matter 없음 (1개) | 제목 = 파일명 `prerequisites-ref-docs`, 본문 = 파일 전체 |
| 파일이 BOM 이나 빈 줄로 시작 | `\A---` 가 맞지 않아 front matter 가 본문에 섞인다. 현재 데이터에는 없음 |

`iter_docs(docs_dir)` (`399–406`): `sorted(rglob("*.md"))` 로 **항상 같은 순서**로 돌면서 headless 조각을 건너뛰고 `(경로, 제목, 본문)` 을 돌려준다.
`02` 와 `03` 모두 이 함수로 문서를 읽는다.

### 4.8 나누기: `split_sections`, `split_long` (`410–446`)

**`split_sections(body)`** — 제목 기준 섹션 분할

```python
419     lines = body.splitlines()
420     for line, (is_code, _) in zip(lines, fence_states(lines)):
421         m = None if is_code else HEADING_RE.match(line)       # 코드 줄은 제목 검사 안 함
422         if m:
423             flush()                                           # 지금까지의 본문을 이전 제목으로 확정
425             level, heading = len(m.group(1)), ANCHOR_RE.sub("", m.group(2)).strip()
426             stack = [(lv, h) for lv, h in stack if lv < level] + [(level, heading)]
```

`stack` (`426`) 은 현재 위치의 제목 경로다. 새 제목보다 **상위 레벨**만 남기고 새 제목을 넣는다.

```
줄                       stack 이후                                    섹션 이름
## 파드의 사용            [(2,'파드의 사용')]                           '파드의 사용'
### 워크로드 리소스       [(2,'파드의 사용'), (3,'워크로드 리소스')]     '파드의 사용 > 워크로드 리소스'
## 파드 작업              [(2,'파드 작업')]                             '파드 작업'          ← 같은 레벨: 교체
## 다음 내용              [(2,'다음 내용')]                             '다음 내용'          ← heading 숏코드에서 온 제목
```

- 첫 제목 이전의 도입부는 섹션 이름이 `''` 이다.
- 본문이 빈 섹션(제목만 있음)은 `flush()` 에서 버린다 (`416`).

**`split_long(text, max_chars)`** — 긴 섹션을 조각으로

```python
437     for line, (_, open_after) in zip(lines, fence_states(lines)):
438         buf.append(line)
439         size += len(line) + 1                                   # +1 = 줄바꿈
440         if not open_after and (
441             (size >= max_chars and not line.strip()) or size >= max_chars * 2
442         ):
443             parts.append("\n".join(buf).strip())                # 여기서 자름
```

자르는 조건: **이 줄 다음에 코드 블록이 열려 있지 않고** (`open_after == False`), 다음 둘 중 하나일 때.

| 조건 | 의미 |
|---|---|
| `size >= max_chars and not line.strip()` | 목표를 넘었고 **지금 줄이 빈 줄** → 문단 경계에서 자름 |
| `size >= max_chars * 2` | 빈 줄 없이 2배를 넘음 → 아무 줄 경계에서나 (안전장치) |

- 코드 블록 중간에서는 **절대 자르지 않는다.** 그래서 거대한 YAML 은 조각 하나가 `max_chars` 보다 훨씬 클 수 있다. 이 경우는 `03` 의 `split_to_budget` 이 처리한다 (5.2).
- 단위는 **글자 수**다. `02` 는 1,800 (SFT 답변 하나), `03` 은 600 (패킹 단위) 을 쓴다.

### 4.9 SFT·RAG 출력 (`450–534`)

- `build_chunks()` (`450–468`): 섹션 → 1,800자 조각 → 150자 미만 버림 → `{"source", "title", "heading", "text"}`. 한 섹션이 여러 조각이면 제목에 `(i/n)` 을 붙인다.
- **`split_by_document(items, key, ratio, rng)`** (`471–486`): 문서 단위로 valid 를 고른다.
  1. 항목을 문서별로 묶는다 (`473–475`).
  2. 문서 이름을 정렬한 뒤 섞는다 (`476–477`). 정렬은 재현성, 섞기는 무작위 선택을 위한 것이다.
  3. 목표 개수(`전체 x ratio`)가 찰 때까지 **문서를 통째로** valid 에 넣는다 (`480–485`).

  `02` 와 `03` 이 함께 쓴다. 이렇게 하면 같은 문서의 일부가 train 에 들어가는 누수가 없다 (8.11).
- `to_sft()` (`489–498`): 청크 → system/user/assistant 대화. 질문은 템플릿에서 무작위로 고른다.
- `write_jsonl()` (`501–505`): 한 줄에 JSON 하나. `ensure_ascii=False` 로 한글을 그대로 저장한다.

---

## 5. `03_build_cpt.py` — 패킹과 데이터셋 구성

### 5.1 import 와 상수 (`14–26`)

```python
20 from transformers import AutoTokenizer
22 prep = import_module("02_prepare_data")     # 숫자로 시작하는 파일명은 import 문으로 못 씀
25 PIECE_CHARS = 600                           # split_long 에 넘기는 조각 크기 (글자)
26 SAFETY_TOKENS = 32                          # 토큰 예산 여유분 (+EOS)
```

- `22`: `import 02_prepare_data` 는 문법 오류라서 `import_module` 을 쓴다. 스크립트 디렉터리가 `sys.path` 에 있어야 한다.
- `25` **왜 600자인가:** 패킹은 조각 단위로 블록을 채운다. 조각이 크면 블록 끝에 빈 공간이 많이 남고, 너무 작으면 토큰 세는 횟수가 늘어 느려진다.
  600자는 약 320 토큰이다 (1 토큰 ≈ 1.88 글자).
- `26` **왜 32 토큰 여유인가:** 조각마다 **따로** 센 토큰 수를 더해서 예산을 맞추는데, 이어 붙인 뒤 한 번에 세면 경계에서 조금 달라진다 (BPE 는 문맥에 따라 병합이 다르다).
  학습 때 붙는 EOS 1 토큰도 이 여유 안에 들어가야 한다. 16 으로 시작했다가 상한을 넘는 블록이 생겨서 32 로 늘렸다.

### 5.2 보조 함수: `decode`, `split_to_budget` (`29–53`)

```python
29 def decode(tok, ids):
31     return tok.decode(ids).strip("�")
```

토큰 목록을 중간에서 자르면 한글 글자의 UTF-8 바이트가 반만 남을 수 있다. 그러면 디코딩 결과의 양 끝에 `�`(U+FFFD) 가 생긴다.
한글 한 글자는 3바이트이고, 토크나이저가 그 3바이트를 여러 토큰으로 나누기도 한다 (`"쿠버네티스"` = 7 토큰). 양 끝의 `�` 를 지운다.

```python
34 def split_to_budget(text, tok, budget):
37     for line in text.split("\n"):
38         n = len(tok.encode(line, add_special_tokens=False)) + 1
39         if n > budget:                       # 한 줄이 예산보다 긴 경우만 (사실상 없음)
44             parts += [decode(tok, ids[i:i + budget]) for ...]   # 토큰 단위로 자름
46         if cur and cur_len + n > budget:     # 넘치면 지금까지를 한 조각으로
47             parts.append("\n".join(cur))
```

조각 하나가 블록 예산보다 클 때(거대한 코드 블록) **줄 경계에서** 나눈다. 토큰 단위로 자르는 것은 한 줄이 예산보다 긴 극단적인 경우뿐이다.
실측: 8,192 와 4,096 에서는 호출 0회, 2,048 에서 2회.

### 5.3 `pack_document(title, body, tok, max_tokens)` (`56–87`) — 문서 하나 → 블록들

```python
61     header = f"# {title}\n\n"                                  # 모든 블록 맨 앞의 제목
62     budget = max_tokens - SAFETY_TOKENS - n_tokens(header)     # 본문에 쓸 수 있는 토큰
63     blocks, cur, cur_len, cur_heading = [], [], 0, None
       #  blocks: 완성된 블록 / cur: 만드는 중인 블록의 조각들 / cur_len: 그 토큰 수 / cur_heading: 직전 조각의 섹션
```

`62`: `max_tokens=8192`, 제목 "파드"면 헤더 `# 파드\n\n` 이 약 4 토큰이므로 `budget = 8192 − 32 − 4 = 8156`.

```python
69     for heading, text in prep.split_sections(body):           # ① 섹션 단위로
70         for piece in prep.split_long(text, PIECE_CHARS):      # ② 600자 조각 단위로
72             seg = f"## {heading}\n\n{piece}" if heading and (heading != cur_heading or not cur) else piece
73             seg_len = n_tokens(seg) + 2                        # +2 = 조각 사이 "\n\n"
```

- `72` **소제목을 언제 붙이나:** 섹션 이름이 있고, 다음 둘 중 하나일 때.
  - `heading != cur_heading`: 새 섹션이 시작됐다.
  - `not cur`: 새 블록의 첫 조각이다.

  같은 섹션의 두 번째 조각부터는 소제목 없이 이어 붙인다.

```python
74             if cur and cur_len + seg_len > budget:            # ③ 넣으면 넘친다 → 지금 블록 확정
75                 flush()
76                 cur, cur_len = [], 0
77                 seg = f"## {heading}\n\n{piece}" if heading else piece   # 새 블록이므로 소제목 다시
79             if seg_len > budget:                               # ④ 조각 하나가 블록보다 큼
81                 blocks += [header + part for part in split_to_budget(seg, tok, budget)]
82             else:
83                 cur.append(seg)                                # ⑤ 보통: 현재 블록에 추가
84                 cur_len += seg_len
85             cur_heading = heading
86     flush()                                                    # ⑥ 마지막 블록
```

- `77`: 섹션 중간에서 새 블록이 시작돼도 블록만 떼어 보면 어느 섹션인지 알 수 있게 한다.
- 함수가 **문서 하나만** 받으므로 서로 다른 문서가 한 블록에 섞이지 않는다.
- **8K 보다 짧은 문서**는 블록 1개가 된다. 문서 길이 그대로이고, 다른 문서를 붙이거나 padding 하지 않는다. 블록 길이 분포는 7.3 참고.

**동작 예시** (`max_tokens=2048` 처럼 작을 때, 문서 "파드", budget ≈ 2012):

```
조각 (섹션, 토큰)                        판단                                cur_len
(도입부, 300)                           heading 없음 → 그대로 추가            302
(파드란 무엇인가?, 310)                  새 섹션 → "## 파드란 무엇인가?" 붙여 추가  ~620
(파드란 무엇인가?, 320)                  같은 섹션 → 그대로 추가               ~942
(파드의 사용, 330)                       새 섹션 → 소제목 붙여 추가            ~1280
(파드의 사용 > 워크로드 리소스, 400)      새 섹션 → 추가                        ~1690
(파드 작업, 380)                         1690+385 > 2012 → flush, 새 블록      ~385
                                         ("# 파드\n\n## 파드 작업\n\n...")
```

### 5.4 `load_replay(args, tok, target_tokens, n_eval)` (`90–110`) — 일반 한국어 텍스트

```python
94     ds = load_dataset(args.replay_dataset, args.replay_config, split="train", streaming=True)
95     ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
99         if len(text) < 300:  continue                          # 토막 글 제외
101        ids = tok.encode(text, ...)[: args.max_tokens - SAFETY_TOKENS]   # 블록 상한에 맞춰 자름
102        text = f"# {row['title']}\n\n{decode(tok, ids)}"       # 쿠버네티스 블록과 같은 헤더 형식
103        if len(evals) < n_eval:        evals.append(...)       # 처음 100개 → 평가용 (학습 X)
105        elif total < target_tokens:    train.append(...)       # 그다음 → 목표 토큰까지 학습용
```

- `94`: `wikimedia/wikipedia` 의 `20231101.ko` 를 **스트리밍**으로 읽는다. 수 GB 를 다 받지 않는다.
- `95`: 스트리밍 셔플은 1만 개 버퍼 안에서만 섞는다. 앞쪽 샤드 위주로 뽑히므로 완전한 무작위 표본은 아니다.
- `101`: 블록 상한이 8K 면 위키 문서도 8K 까지 쓴다. 상한을 바꾸면 **replay 문서 개수와 길이도 바뀐다** (2K: 289개, 8K: 184개).
- `102`: `decode()` 로 끝의 깨진 글자를 지운다 (8.10).

### 5.5 `main()` (`113–169`)

```python
134     blocks = []  # [(문서 경로, 블록 텍스트)]
137         for path, title, body in prep.iter_docs(docs):
139             blocks += [(str(path), b) for b in pack_document(title, body, tok, args.max_tokens)]
141     prep.print_stats()                                         # 숏코드 치환 집계 출력

143     blocks = [(src, b) for src, b in blocks if len(tok.encode(b)) >= 32]   # 32 토큰 미만 제거
144     domain_train, valid, valid_docs = prep.split_by_document(blocks, lambda x: x[0], args.valid_ratio, rng)
```

- `134–139`: 블록마다 **출처 문서 경로**를 같이 들고 다닌다. 문서 단위로 valid 를 나누는 데 쓴다.
- `144`: 553 블록 x 0.02 = 11 블록을 목표로, 문서를 통째로 골라 valid 에 넣는다 → 문서 11개, 블록 11개.
  valid 로 뺀 문서 목록은 `data/cpt/valid_docs.txt` 에 남는다 (`159`).

```python
151         target = int(domain_tokens * args.replay_ratio / (1 - args.replay_ratio))
```

- **replay 목표 토큰:** `replay / (domain + replay) = r` 이 되려면 `replay = domain x r / (1 − r)`.
  `1,196,430 x 0.2 / 0.8 = 299,107` → 문서 단위로 넘치게 채워서 실제로는 306,745 토큰 (184개).

```python
154     train = domain_train + replay
155     rng.shuffle(train)                                         # 도메인과 replay 를 골고루 섞음
157–163 write_jsonl(...)                                        # 출력 (6절)
160     ... rng.sample(domain_train, ...)                          # k8s_seen: train 에서 100개 표본
165     steps_per_epoch = -(-len(train) // args.batch_size)       # 올림 나눗셈 (--batch-size = 배치 x 누적)
```

실행 결과 (8K, 실측):

```
raw/website/content/ko/docs: 문서 561개
숏코드 치환: code_sample 239, code_sample_missing 0, include 115, ... headless_skipped 8
쿠버네티스 블록: train 542 (1,196,430 토큰) / valid 11 (문서 11개)
일반 텍스트 replay: 184개 (306,745 토큰)
총 train 샘플 726 -> 실효 배치 8 기준 1 epoch = 91 스텝
```

---

## 6. 출력 파일 형식

모든 파일은 **JSON Lines** 형식이다. 한 줄 = JSON 객체 하나 = **샘플 하나**이고, UTF-8 로 저장한다.

| 파일 | 줄 수 | 크기 | 스키마 | 쓰는 곳 |
|---|---|---|---|---|
| `data/cpt/train.jsonl` | 726 | 5.2 MB | `{"text": str}` | CPT 학습 |
| `data/cpt/valid.jsonl` | 11 | 0.1 MB | `{"text": str}` | CPT 학습 중 eval loss |
| `data/cpt/valid_docs.txt` | 11 | — | 문서 경로 (텍스트) | valid 로 뺀 문서 기록 |
| `data/eval/k8s_seen.jsonl` | 100 | 0.8 MB | `{"text": str}` | 평가: 학습한 문서 |
| `data/eval/k8s_heldout.jsonl` | 11 | 0.1 MB | `{"text": str}` | 평가: 안 본 문서 (= valid) |
| `data/eval/general_ko.jsonl` | 100 | 0.4 MB | `{"text": str}` | 평가: 망각 |
| `data/sft/train.jsonl` | 3,088 | 5.0 MB | `{"messages": [{role, content} x 3]}` | SFT (모듈 6) |
| `data/sft/valid.jsonl` | 96 | 0.1 MB | 위와 같음 | SFT (모듈 6) |
| `data/chunks.jsonl` | 3,184 | 4.5 MB | `{"source", "title", "heading", "text"}` | RAG (모듈 6) |

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
```

> **"한 줄의 길이"는 무엇으로 재나:** 파일에서 한 줄은 수백 바이트 ~ 수십 KB 지만, 학습에서 의미 있는 길이는
> **`text` 를 토크나이저로 바꾼 토큰 수**다. 다음 절의 "블록 길이"는 모두 토큰 수다.

---

## 7. 블록 길이(JSONL 한 줄의 길이) 선택 고려사항

`03_build_cpt.py --max-tokens N` 은 **한 줄(샘플)의 최대 토큰 수**를 정한다.
**목표 길이가 아니라 상한**이다. 짧은 문서는 그 길이 그대로 한 줄이 된다.

### 7.1 고려사항 체크리스트

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

### 7.2 문서 길이 분포 (정제 후, `# 제목` 포함, Qwen 토큰, headless 제외 561개)

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

### 7.3 상한별 패킹 결과 비교 (실측)

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

### 7.4 환경별 권장

| 환경 | 설정 | 이유 |
|---|---|---|
| **g6e (L40S 48GB), full** (기본) | 8,192 x 배치1 x 누적8 + Liger 커널 | 쪼개지는 문서 27개, padding 0%. 메모리는 Liger 로 확보 (**리허설에서 실측 필요**) |
| g6e, OOM 이 나면 | 4,096 x 배치1 x 누적16 | 마이크로배치 토큰이 줄어 메모리 여유. padding 0% |
| Mac 24GB (MLX) | 2,048 x 배치2 x 누적4 (`config_cpt.yaml`) | 메모리가 이미 빠듯하다. mlx-lm 에는 Liger 같은 기능이 없다 |
| 32,768 | 권장하지 않음 | 32K 를 넘는 문서가 없다. 채우려면 문서를 섞어야 하고, 메모리와 계산이 크게 는다 |

### 7.5 상한을 바꿀 때 같이 바꿔야 하는 곳

| # | 파일 | 바꿀 것 |
|---|---|---|
| ① | 데이터 | `python 03_build_cpt.py --max-tokens N --batch-size <배치x누적>` |
| ② | `train_cpt_hf.py` | `--max-length N` (필요하면 `--grad-accum`) |
| ③ | `k8s_cpt_g6e.ipynb` | 데이터 셀의 `MAX_TOKENS`. 학습 셀과 평가 셀이 같은 값을 쓴다 |
| ④ | `config_cpt.yaml` (Mac) | `max_seq_length`, `batch_size`, `iters` |
| ⑤ | `05_eval_cpt.py` (Mac) | `--max-tokens` |

---

## 8. 수정 이력: 검증으로 찾아 고친 문제들

처음 버전의 전처리를 실제 데이터로 검증해서 아래 문제들을 찾았고, 모두 고쳤다.
"수정 전" 수치는 같은 원본에 처음 버전 코드를 돌렸을 때의 실측값이다. 데이터 전처리 코드를 검증하는 방법의 예시로도 볼 수 있다.

| # | 문제 | 수정 전 | 원인 | 수정 | 수정 후 |
|---|---|---|---|---|---|
| 8.1 | `heading` 숏코드가 빈 `## ` 줄로 남음 | CPT 샘플 334개, chunks 330개에 빈 제목. 섹션이 나뉘지 않음 | 모든 숏코드를 지운 뒤 `## ` 만 남음 → `HEADING_RE` 불일치 | `i18n/ko/ko.toml` 의 `<키>_heading` 으로 치환 | 빈 제목 0 |
| 8.2 | 예제 YAML 미삽입 | 239건 삭제 (본문은 "다음 매니페스트를 사용한다."에서 끝남) | 숏코드를 지우기만 함 | `examples/` 파일을 코드 블록으로 삽입. 없으면 영어판 | 239건 삽입, 누락 0 |
| 8.3 | `include` 미삽입 | 109건 삭제 | 위와 같음 | `includes/` 또는 상대 경로 파일을 재귀 정제해서 삽입 | 115건 (탭 include 포함), 누락 0 |
| 8.4 | `glossary_definition` 미삽입 | 29건 삭제 | 위와 같음 | 용어집 본문 삽입 (short 는 첫 문단) | 29건 |
| 8.5 | 버전 숏코드가 빈칸 | "쿠버네티스 에는" 등 284건. 코드 안 `{{< param "version" >}}` 은 원문 그대로 | 삭제만 함. 코드 안은 처리 안 함 | `hugo.toml`·`schedule.yaml` 로 계산, 코드 안에도 적용 | 남은 버전 숏코드 0 |
| 8.6 | `comment` 숨김 내용 노출 | 8개 문서 | 여닫는 표식만 지움 | 블록을 내용째 삭제 | |
| 8.7 | `feature-state` 정보 손실 | 241건 삭제 | 삭제만 함 | "기능 상태: Kubernetes v1.33 [stable]" (기능 게이트형은 게이트 파일 조회) | 241건 |
| 8.8 | 탭 이름 손실 | 25개 문서 | 표식만 지움 | `**탭 이름**` 줄, codelang 탭은 코드 블록 | |
| 8.9 | 한 줄 펜스 때문에 코드 안/밖 뒤집힘 | `cheatsheet.md` 의 bash 주석들이 제목이 됨 | ` ```kubectl -A``` ` 를 펜스 토글 1회로 셈 | `fence_states()`: 같은 줄에 백틱이 또 있으면 펜스 아님, 닫는 펜스는 같은 문자로만 | 가짜 제목 0 (cheatsheet 21개 섹션 정상) |
| 8.10 | replay 끝 글자 깨짐 (`�`) | 4개 샘플 | 토큰 경계에서 잘라 디코딩 | `decode()` 가 양 끝 `�` 제거. 강제 분할도 줄 단위로 | `�` 0 |
| 8.11 | valid 문서 누수 | valid 18개 중 11개는 같은 문서의 다른 블록이 train 에 있음 | 블록 단위로 섞어서 나눔 | `split_by_document()` | 누수 0 (문서 11개 통째로 valid) |
| 8.12 | headless 조각이 단독 문서로 학습 | 8개 문서 | front matter 를 보지 않음 | `iter_docs()` 에서 제외 (include 될 때만 들어감) | 중복 제거 |
| 8.13 | 이스케이프 숏코드 손실 | 기여 가이드의 `{{</* ... */>}}` 문법 예시 4건 | 일반 숏코드처럼 지움 | `{{< ... >}}` 글자 그대로 복원 | |
| 8.14 | `mermaid` 코드가 본문에 남음 | 12개 문서 | 표식만 지움 | 블록을 내용째 삭제 | |
| 8.15 | `figure` 설명 손실 | 14개 문서 | 삭제만 함 | `(그림: 캡션 또는 alt)` | |

수정 후 검증 (실측, 8K 데이터의 train + valid 737개 샘플):

```
최대 토큰 8,172 (상한 8,192 초과 0개)    빈 제목 줄 0    깨진 문자 0
코드 밖에 남은 Hugo 숏코드 0             코드 안에 남은 skew/param 0
삽입된 ```yaml 블록 675개                cheatsheet 섹션 21개 (가짜 제목 0)
```

---

## 9. 남아 있는 한계

| 한계 | 설명 | 대응 방향 |
|---|---|---|
| `task-tutorial-prereqs.md` 반복 | 같은 안내문(정제 후 152 토큰)이 95개 문서에 삽입된다. 사이트에 보이는 그대로지만 같은 문단을 95번 학습한다 | 문서 수가 늘면 dedup (같은 문단 반복 제거)을 고려 |
| `version-check` 등 안내문 삭제 | 사이트에는 "쿠버네티스 서버 버전은 ... 이상이어야 한다" 같은 문장이 보인다 | 필요하면 i18n 문자열로 치환 |
| 기능 게이트 단계 추출 | 정규식으로 `stages` 를 읽는다. `fromVersion` 이 없는 단계가 있으면 다음 단계 값을 가져갈 수 있다 | YAML 파서로 교체 |
| 강제 분할 시 코드 펜스 | 거대한 코드 블록을 줄 단위로 나누면 각 조각에 여닫는 펜스가 없다 | 8K/4K 에서는 0건, 2K 에서 2건 |
| `content_type` 미사용 | concept/task/tutorial 같은 문서 성격이 블록에 표시되지 않는다 | 헤더에 넣는 실험 가능 |
| HTML 주석 | 코드 블록 안의 `<!-- -->` 도 지운다 | 현재 데이터에서는 영향 없음 |
| 영어 문서 대체 | 한글 저장소에 없는 예제 40건은 영어판이다. 예제 속 주석이 영어다 | 사이트와 같은 동작이라 그대로 둠 |

---

## 10. 직접 확인해 보는 명령 모음

모두 `~/llm-ft` 에서 `source .venv/bin/activate` 후에 실행한다.

**원문 vs 정제 결과**
```bash
python - <<'EOF'
from importlib import import_module; from pathlib import Path
prep = import_module("02_prepare_data")
p = Path("raw/website/content/ko/docs/concepts/cluster-administration/logging.md")
print(p.read_text()[:1500]); print("=" * 60)
t, body = prep.parse_doc(p); print("TITLE:", t); print(body[:1500])
EOF
```

**숏코드 하나씩 시험해 보기**
```bash
python - <<'EOF'
from importlib import import_module; from pathlib import Path
prep = import_module("02_prepare_data")
page = prep.page_for(Path("raw/website/content/ko/docs/concepts/overview/_index.md"))
for s in ['## {{% heading "whatsnext" %}}',
          '쿠버네티스 {{< skew currentVersion >}} / 패치 {{< skew currentPatchVersion >}}',
          '{{< feature-state feature_gate_name="SidecarContainers" >}}',
          '{{% code_sample file="pods/simple-pod.yaml" %}}']:
    print(repr(s), "\n →", prep.clean_markdown(s, page), "\n")
EOF
```

**섹션 분할 결과**
```bash
python -c "
from importlib import import_module; from pathlib import Path
prep = import_module('02_prepare_data')
t, b = prep.parse_doc(Path('raw/website/content/ko/docs/reference/kubectl/cheatsheet.md'))
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
    print(n, "->", [len(tok.encode(x)) for x in b.pack_document(t, body, tok, n)])
EOF
```

**출력 데이터 점검 (빈 제목 줄, 깨진 문자, 남은 숏코드, 최대 길이)**
```bash
python - <<'EOF'
import json, re
from importlib import import_module
from transformers import AutoTokenizer
prep = import_module("02_prepare_data"); tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
rows = [json.loads(l)["text"] for f in ("data/cpt/train.jsonl", "data/cpt/valid.jsonl") for l in open(f, encoding="utf-8")]
def prose(t):
    ls = t.split("\n"); return "\n".join(l for l, (c, _) in zip(ls, prep.fence_states(ls)) if not c)
print("빈 제목 줄:", sum(bool(re.search(r"(?m)^#{1,4}\s*$", prose(t))) for t in rows))
print("깨진 문자 �:", sum("�" in t for t in rows))
print("코드 밖 숏코드:", sum(bool(re.search(r"\{\{[<%]", prose(t))) for t in rows))
print("최대 토큰:", max(len(tok.encode(t)) for t in rows))
EOF
```

**숏코드 사용 현황 (원본)**
```bash
grep -rhoE '\{\{[<%] *[/]?[a-zA-Z_-]+' raw/website/content/ko/docs --include='*.md' \
  | sed -E 's/\{\{[<%] *//' | sort | uniq -c | sort -rn | head -20
```

---
← [모듈 3. 데이터 파이프라인](03_data.md) · 다음: [모듈 4. 계속 사전학습 →](04_training.md)
