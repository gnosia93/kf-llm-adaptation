# 모듈 3. 데이터 파이프라인 — 수집 → 정제 → 패킹 → replay

**소요 50분** · 강의 20분 + 실습 30분

> 함수별·라인별 상세 해설, 원본 데이터 구조, 블록 길이 선택 기준, 검증으로 찾아 고친 문제 목록은
> **[모듈 3-A. 데이터 전처리 상세 해설](03a_data_deep_dive.md)** 에 있다.

## 학습 목표

- 공개 문서 저장소에서 필요한 부분만 효율적으로 받는다.
- 마크업(Hugo 숏코드)을 **사이트에 보이는 내용 그대로** 복원하되 코드 블록은 보존하는 정제를 이해한다.
- 왜 토큰 단위로 **패킹**하는지, 블록 길이를 어떻게 정하는지 설명한다.
- replay 데이터를 섞고, 평가 세트 3종을 만든다.

## 파이프라인 한눈에 보기

```
01_download.sh            02_prepare_data.py                     03_build_cpt.py
──────────────            ────────────────────────────────       ──────────────────────────────────
GitHub                    iter_docs()                            pack_document()
kubernetes/website   ─▶    headless 조각 제외 (561개)       ─▶    섹션 조각을 8,192 토큰 이하로 묶음
content/ko/docs           parse_doc() → clean_markdown()          + "# 제목", "## 소제목" 헤더
(569개 .md)                숏코드를 렌더링 결과로 치환           split_by_document()
+ 예제·include·번역·        (예제 YAML 삽입, 번역 제목, 버전)       valid 를 문서 단위로 분리
  버전 정보                링크·HTML 정리, 코드 블록은 보존       load_replay()
                          split_sections() / split_long()         위키백과에서 20% 만큼
                                                                         │
                                                                         ▼
                                                   data/cpt/train.jsonl        726 개
                                                   data/cpt/valid.jsonl         11 개
                                                   data/eval/{k8s_seen, k8s_heldout, general_ko}.jsonl
```

---

## 3.1 수집: sparse checkout

`kubernetes/website` 저장소는 모든 언어의 문서와 이미지를 합쳐 크다. **필요한 디렉터리만** 받는다.

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/kubernetes/website.git raw/website
git -C raw/website sparse-checkout set content/ko \
  content/en/examples content/en/docs/reference/command-line-tools-reference/feature-gates \
  i18n data/releases
```

| 옵션 | 의미 |
|---|---|
| `--depth 1` | 최신 커밋만 (히스토리 제외) |
| `--filter=blob:none` | 파일 내용은 실제로 필요할 때만 받음 |
| `--sparse` + `sparse-checkout set` | 지정한 디렉터리만 작업 트리에 풀어 놓음. 루트 파일(`hugo.toml`)은 항상 포함 |

**왜 본문 말고 다른 것도 받나:** 사이트에 보이는 문서는 본문 하나로 완성되지 않는다.
예제 YAML(`examples/`), 공통 조각(`includes/`), 번역된 제목(`i18n/`), 버전 번호(`hugo.toml`)를 **빌드할 때 끼워 넣어서** 만든다.
이것들이 없으면 "다음 매니페스트를 사용한다." 다음에 매니페스트가 비어 있는 문서를 학습하게 된다.

```bash
./01_download.sh                    # [ko] 마크다운 문서 수: 569
LANGS="ko en" ./01_download.sh      # [en] 마크다운 문서 수: 1729  (선택 과제)
```

> 한글 번역은 영어 원문의 약 1/3 만 다룬다. 번역되지 않은 주제는 영어 원문을 넣어야 배울 수 있다.

## 3.2 정제: 사이트에 보이는 내용으로 되돌리기

쿠버네티스 문서는 정적 사이트 생성기 **Hugo** 로 만든다. 원문에는 `{{< ... >}}` 형식의 **숏코드**가 섞여 있다.
숏코드는 빌드할 때 다른 내용으로 바뀌는 자리 표시자다.

### 원문 → 정제 결과 (실제 출력)

```markdown
## {{% heading "whatsnext" %}}                          →  ## 다음 내용

쿠버네티스 {{< skew currentVersion >}}에는             →  쿠버네티스 1.37에는

{{< feature-state for_k8s_version="v1.25"              →  기능 상태: Kubernetes v1.25 [stable]
    state="stable" >}}

하나 이상의 {{< glossary_tooltip text="컨테이너"         →  하나 이상의 컨테이너의 그룹이다.
    term_id="container" >}}의 그룹이다.

{{< note >}} ... {{< /note >}}                          →  참고: ...

다음 매니페스트를 사용한다.                               →  다음 매니페스트를 사용한다.
{{% code_sample file="debug/counter-pod.yaml" %}}       →  `debug/counter-pod.yaml`
                                                            ```yaml
                                                            apiVersion: v1
                                                            kind: Pod ...        ← examples/ 의 파일 내용
                                                            ```

[컨테이너 런타임](/docs/setup/...)을                     →  컨테이너 런타임을
```

### 정제 규칙 요약

| 대상 | 처리 | 횟수 (실측) |
|---|---|---|
| front matter (`---` ~ `---`) | `title` 만 뽑고 제거 | 569 |
| HTML 주석, `{{< comment >}}` 숨김 블록, `mermaid` 다이어그램 | 내용째 삭제 | |
| `heading` | 번역된 제목 (`i18n/ko/ko.toml`) | 427 |
| `code_sample` / `codenew` | `examples/` 파일을 코드 블록으로 삽입 (한글에 없으면 영어판) | 239 |
| `include` | 공통 조각 파일을 재귀적으로 정제해 삽입 | 115 |
| `glossary_definition` | 용어집 본문 삽입 | 29 |
| `skew` / `param` | 버전 번호 (`hugo.toml`, `schedule.yaml`) — **코드 블록 안에도** 적용 | 284 |
| `feature-state` | `기능 상태: Kubernetes v1.25 [stable]` | 241 |
| `glossary_tooltip` | `text` 값 | 861 |
| `note` / `caution` / `warning` | `참고:` / `주의:` / `경고:` | 689 |
| `tab` | `**탭 이름**` 줄 (codelang 탭은 코드 블록) | 108 |
| `figure` | `(그림: 캡션)` | 20 |
| 나머지 숏코드, 이미지 | 삭제 | |
| 링크 `[텍스트](주소)` | 텍스트만 | |
| 지정한 HTML 태그 (`<br>`, `<table>` ...) | 삭제 | |
| **코드 블록** | **그대로** (버전 숏코드만 치환) | |
| `headless: true` 조각 문서 | 단독 문서에서 제외 (include 될 때만 들어감) | 8 |

> 각 숏코드의 동작은 저장소의 `layouts/shortcodes/<이름>.html` 을 읽고 그대로 옮겼다.
> 처음 버전은 숏코드를 지우기만 해서 예제 YAML 239건과 제목 427개가 사라졌다.
> 실제 데이터로 검증해서 찾아 고친 과정은 [3-A 의 8절](03a_data_deep_dive.md#8-수정-이력-검증으로-찾아-고친-문제들) 에 있다.

### 코드 블록은 어떻게 알아보나

전처리 전체의 기반은 **"이 줄이 코드 블록 안인가"** 를 정확히 아는 것이다.

```python
def fence_states(lines):          # 02_prepare_data.py:168
    ...
    if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):   # ```kubectl -A``` 처럼 한 줄에서
        fence = m.group(1)                                       # 열고 닫는 것은 펜스가 아니다
```

- 코드 블록 안의 bash 주석 `# 파드 목록 확인` 을 **제목으로 오인하지 않는다.**
- 코드 블록 안의 `<pod-name>`, `{{ .data }}` (Go 템플릿), `[a](b)` 같은 문자열을 **지우지 않는다.**
- 처음 버전은 ` ```kubectl -A``` ` 한 줄을 펜스로 세는 바람에 `kubectl 치트 시트` 의 코드 안/밖이 뒤집혔다.

### 섹션 분할

제목 `#` ~ `####` 기준으로 나누고, 제목은 `상위 > 하위` 경로로 붙는다. pods 문서는 이렇게 나뉜다:

```
'파드란 무엇인가?'
'파드의 사용'
'파드의 사용 > 파드를 관리하기 위한 워크로드 리소스'
'파드 작업 > 파드 OS'
...
'다음 내용'              ← heading 숏코드에서 복원된 제목
```

## 3.3 패킹과 블록 길이

학습 프레임워크는 한 샘플을 최대 `max_length` 토큰까지만 본다.

| 상황 | 문제 |
|---|---|
| 샘플이 `max_length` 보다 길다 | **뒷부분이 잘려 버려진다.** 그만큼 지식이 사라진다 |
| 길이가 다른 샘플을 한 배치로 | 짧은 쪽을 padding 으로 채운다 → GPU 계산 낭비 (배치2 기준 24~36%) |
| 문서 중간부터 시작하는 샘플 | 무슨 문서에 대한 내용인지 모른 채 학습한다 |

그래서 `03_build_cpt.py` 가 **미리 토크나이저로 길이를 재서** 블록을 만든다.

### `pack_document()` 의 동작

```
문서 "파드" ─ 섹션 조각들 (각 600자 내외, 코드 블록은 쪼개지 않음)
 [소개] [파드란 무엇인가? 1] [파드란 무엇인가? 2] [파드의 사용] [파드 작업 > 파드 OS] ...

  블록 1 (≤ 8,192 토큰)                        블록 2 (문서가 더 길 때만)
 ┌─────────────────────────────────┐        ┌───────────────────────────────┐
 │ # 파드                          │        │ # 파드                        │ ← 제목을 매 블록 반복
 │ [소개]                          │        │ ## 파드 작업 > 파드 OS         │ ← 섹션 중간에서 시작하면
 │ ## 파드란 무엇인가?              │        │ [파드 작업 > 파드 OS]          │   소제목도 다시 붙임
 │ [조각 1] [조각 2]               │  넘침  │ ...                           │
 │ ## 파드의 사용                   │ ────▶ │                               │
 └─────────────────────────────────┘        └───────────────────────────────┘
```

- **문서 경계는 넘지 않는다.** 서로 다른 문서를 한 블록에 섞지 않는다.
- 상한보다 **짧은 문서는 블록 1개**가 된다. 길이 그대로이고, padding 이나 다른 문서로 채우지 않는다.
- `budget = 8192 − 32(여유) − 헤더 토큰 수`. 32 토큰 여유는 조각마다 센 토큰 수의 합과 실제 토큰 수의 차이, 그리고 EOS 몫이다.

### 블록 길이(JSONL 한 줄의 최대 토큰)를 8,192 로 정한 이유

문서 561개의 길이 분포 (정제 후, Qwen 토큰, 실측):

```
중앙값 1,207 | 상위 10% 5,103 | 상위 1% 15,910 | 최대 19,535 (job.md)

   상한     이보다 긴 문서   그 문서들의 토큰 비중   블록 수   쪼개진 문서
  2,048      201 (36%)          81%               973        210
  4,096       88 (16%)          54%               656         90
  8,192       25  (4%)          25%               553         27   ← 기본값
```

- 문서 **개수**로는 대부분 짧지만 토큰 **양**의 81% 가 2K 보다 긴 문서에 있다. 2K 로 하면 210개 문서가 잘린다.
- 8K 면 문서의 96% 가 블록 하나에 통째로 들어간다.
- 32K 를 넘는 문서는 없으므로 32K 까지 늘릴 이유가 없다.
- 메모리는 **마이크로배치당 토큰 수**로 정해진다. 그래서 배치 1 로 학습하고(padding 0%), 로짓 메모리는 Liger 커널로 줄인다 (모듈 4).
- Mac(24GB)은 메모리가 빠듯해서 2,048 을 쓴다.

13가지 고려사항 전체는 [3-A 의 7절](03a_data_deep_dive.md#7-블록-길이jsonl-한-줄의-길이-선택-고려사항) 에 있다.

### 실측 결과 (8,192)

```
쿠버네티스 블록: train 542 (1,196,430 토큰) / valid 11 (문서 11개)
블록 길이 분포:  < 256: 26% | 256~1023: 15% | 1024~2047: 19% | 2048~4095: 22% | ≥ 4096: 18%
                (토큰 비중은 각각 2% | 5% | 13% | 29% | 52%)
평균 2,192 토큰, 최대 8,172 토큰 → 8,192 안에 모두 들어감
```

## 3.4 replay 와 평가 세트

```python
target = int(domain_tokens * args.replay_ratio / (1 - args.replay_ratio))
#        1,196,430 * 0.2 / 0.8 = 299,107 토큰 → 위키 문서 184개 (306,745 토큰)
```

- `wikimedia/wikipedia` 의 `20231101.ko` 를 **스트리밍**으로 읽는다. 전체를 받지 않는다.
- 300자 미만 문서는 건너뛰고, 각 문서는 블록 상한(8,160 토큰)에서 자른다. 잘린 경계의 깨진 글자는 지운다.
- 처음 100개는 평가용(`general_ko`)으로 따로 떼어 둔다. **학습에는 쓰지 않는다.**

### valid 는 문서 단위로 나눈다

블록 단위로 무작위로 나누면 같은 문서의 앞부분은 train, 뒷부분은 valid 에 들어간다.
그러면 "처음 보는 문서" 라는 valid 의 전제가 깨진다. 처음 버전에서는 valid 18개 중 11개가 이랬다.
지금은 `split_by_document()` 가 **문서를 통째로** valid 에 넣는다. 고른 문서는 `data/cpt/valid_docs.txt` 에 기록된다.

### 평가 세트 3종

| 파일 | 내용 | 학습에 포함? | 무엇을 보나 |
|---|---|---|---|
| `data/eval/k8s_seen.jsonl` | train 블록 중 100개 | ✅ | 지식 습득 (암기 포함) |
| `data/eval/k8s_heldout.jsonl` | valid 블록 11개 (문서 11개) | ❌ | 도메인 일반화 |
| `data/eval/general_ko.jsonl` | 위키 문서 100개 | ❌ | 망각 |

> **생각해 보기:** valid 로 뺀 문서 11개의 내용은 모델이 배우지 못한다.
> 지식 주입이 목적이면 `--valid-ratio` 를 0.01 로 줄이는 것도 방법이다. 무엇을 얻고 무엇을 잃을까?

---

## 실습

노트북 **`1. 데이터 준비`** 섹션의 셀을 실행한다.

```python
INCLUDE_EN = False
...
!LANGS="{langs}" bash 01_download.sh
MAX_TOKENS = 8192
!.venv/bin/python 02_prepare_data.py
!.venv/bin/python 03_build_cpt.py --docs {docs} --max-tokens {MAX_TOKENS} --batch-size 8
```

✅ **확인:** 출력이 다음과 비슷해야 한다 (위키 샘플링 때문에 replay 숫자는 조금 다를 수 있다).
```
raw/website/content/ko/docs: 문서 561개
숏코드 치환: code_sample 239, code_sample_missing 0, include 115, include_missing 0, ...
쿠버네티스 블록: train 542 (1,196,430 토큰) / valid 11 (문서 11개)
일반 텍스트 replay: 184개 (306,745 토큰)
총 train 샘플 726 -> 실효 배치 8 기준 1 epoch = 91 스텝
```

`_missing` 이 0 이 아니면 sparse checkout 에 필요한 디렉터리가 빠진 것이다. `./01_download.sh` 를 다시 실행한다.

아래 과제는 노트북에 새 셀을 추가해서 진행한다.

### 과제 3-1. 정제 전후 비교

```python
from importlib import import_module
from pathlib import Path
prep = import_module("02_prepare_data")

p = Path("raw/website/content/ko/docs/concepts/cluster-administration/logging.md")
print(p.read_text()[:1500])
print("=" * 60)
title, body = prep.parse_doc(p)
print("TITLE:", title)
print(body[:1500])
```

- 원문의 숏코드가 어떻게 바뀌었는지 3가지 찾아보자. `code_sample` 자리에 무엇이 들어갔나?
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
print("max", max(lens), "| 8,192 초과:", sum(l > 8192 for l in lens))
```

- 상한을 넘는 샘플이 없는지 확인하자.
- 왼쪽(아주 짧은 블록)과 오른쪽(8,000 근처)에 몰린 블록은 각각 어떤 문서일까?

### 과제 3-4. 블록 하나 뜯어보기

```python
import random
r = random.choice([r for r in rows if r.startswith("# ") and "```yaml" in r])
print(r[:2000])
```

- `# 제목` 과 `## 소제목` 헤더가 제대로 붙어 있나?
- 예제 파일 이름(`` `xxx.yaml` ``)과 YAML 코드 블록이 함께 들어 있나?

### 과제 3-5 (도전). replay 비율 바꿔 보기

```python
!.venv/bin/python 03_build_cpt.py --replay-ratio 0.4 --out /tmp/data_r40
```

- 학습 샘플 수와 토큰 수는 어떻게 바뀌나? 학습 시간에는 어떤 영향이 있을까?
- **주의:** `--out` 을 지정하지 않으면 실제 학습용 `data/` 를 덮어쓴다.

### 과제 3-6 (도전). 영어 원문 추가

노트북에서 `INCLUDE_EN = True` 로 바꾸고 데이터를 다시 만든다.
토큰 수가 몇 배가 되는지, 1 epoch 의 스텝 수가 얼마가 되는지 기록하자.
학습 시간도 그만큼 늘어나므로 강사의 안내에 따라 결정한다.

---
← [모듈 2](02_environment.md) · [3-A. 상세 해설](03a_data_deep_dive.md) · 다음: [모듈 4. 계속 사전학습 →](04_training.md)
