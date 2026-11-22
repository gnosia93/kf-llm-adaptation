## 파이썬 정규표현식(Regex) ##

## 1. 정규표현식 핵심 기호 ##

패턴을 해석할 때 가장 자주 등장하는 핵심 기호와 개념입니다.

### ① 수량자 (반복 기호) 및 탐욕적 vs 게으른 매칭
| 기호 | 이름 | 의미 | 예시 |
| :--- | :--- | :--- | :--- |
| **`+`** | 1개 이상 | 앞의 패턴이 **최소 1번 이상** 연속됨 (0개는 실패) | `a+` → `"a"`, `"aa"` |
| **`*`** | 0개 이상 | 앞의 패턴이 **없거나 여러 번** 나옴 | `a*` → `""`, `"a"`, `"aa"` |
| **`?`** | 0개 또는 1개 | 앞의 패턴이 **있거나 없음** | `/?` → `""` 또는 `"/"` |
| **`{m,n}`** | 횟수 범위 지정 | 앞의 패턴이 **최소 `m`번, 최대 `n`번** 반복됨 | `#{1,4}` → `#`, `##`, `###`, `####` |
| **`+?` / `*?`** | **게으른 매칭 (Lazy)** | 조건이 맞는 한 **가장 짧게(최소한으로)** 매칭함. 뒤쪽 기호를 침범하지 않도록 막을 때 필수 | `(.+?)` |

### ② 그룹화(Grouping), 역참조(Backreference), 선택(`|`)
| 기호 | 의미 | 설명 |
| :--- | :--- | :--- |
| **`( ... )`** | **캡처 그룹 (Grouping)** | 여러 글자를 한 덩어리로 묶고, 매칭된 내용을 별도 바구니(**그룹 1, 2...**)에 기억합니다. |
| **`r"\1"`** | **역참조 (Backreference)** | `re.sub()`에서 **1번 소괄호 `( ... )` 안에 잡힌 텍스트**를 그대로 꺼내어 치환할 때 씁니다. (`r`은 백슬래시를 글자 그대로 전달하는 Raw String 접두사) |
| **`(A\|B)`** | **OR 선택 (Alternation)** | 파이프(`\|`)로 구분된 여러 단어/패턴 중 하나와 일치합니다. |

### ③ 문자 클래스 `[ ... ]` 와 이스케이프 `\`
| 기호 | 의미 | 설명 |
| :--- | :--- | :--- |
| **`[abc]`** | 문자 클래스 | 대괄호 안의 문자 중 **단 1글자**와 일치합니다. (예: `[<%]`는 `<` 또는 `%`) |
| **`[^abc]`** | **부정 문자 클래스** | 대괄호 맨 앞의 `^`는 제어를 뜻합니다. 지정한 문자를 **제외한 모든 1글자**를 의미합니다. (예: `[^}]`는 `}`를 제외한 모든 문자) |
| **`\`** | **이스케이프 (Escape)** | `.`, `(`, `)`, `[`, `]`, `{`, `}`, `?`, `+`, `*` 등 특수 기능 기호를 **일반 문자 그대로** 찾을 때 앞에 붙입니다. (예: `\( 나 \)`) |
| **대괄호 `[ ]` 안의 예외** | 특수기능 해제 | `[ ]` 안에서는 `)`, `.`, `*`, `+` 등이 일반 문자가 되므로 `[^)]`처럼 `\`를 생략해도 됩니다. 단, **`\]`**, **`\\`**, **`^`(맨 앞)**, **`-`(범위)**는 여전히 이스케이프가 필요합니다. |

### ④ 위치 지정자 및 플래그(옵션)
| 기호 / 플래그 | 의미 | 설명 |
| :--- | :--- | :--- |
| **`\A`** | 문자열의 맨 시작 | 멀티라인 여부와 상관없이 파일/문자열의 **가장 첫 위치**만 의미합니다. |
| **`^` / `$`** | 줄의 시작 / 줄의 끝 | 한 줄의 맨 앞과 맨 뒤를 의미합니다. |
| **`\b`** | **단어 경계 (Word Boundary)** | 단어와 공백/기호 사이의 경계를 뜻합니다. `<br>`을 찾을 때 `<bridge>`가 잘못 걸리는 것을 막아줍니다. |
| **`re.S`** | **Single line (`re.DOTALL`)** | 점(`.`)이 **줄바꿈(`\n`)까지 포함**해서 매칭하도록 허용합니다. (개행 자체가 삭제되는 것이 아니라 `.`의 인식 범위가 넓어짐) |
| **`re.M`** | **Multiline (`re.MULTILINE`)** | 여러 줄 텍스트에서 `^`와 `$`가 각 줄의 시작과 끝으로 작동하게 합니다. |
| **`re.I`** | **Ignore case (`re.IGNORECASE`)** | 영문 대소문자를 구분하지 않고 매칭합니다. |


## 2. 실전 정규표현식 패턴 상세 해설 및 파이썬 예제 ##

### 1) `FRONT_MATTER_RE`: 마크다운 상단 메타데이터 분리
* **패턴:** `r"\A---\s*\n(.*?)\n---\s*\n"` (플래그: `re.S`)
* **기호 설명:**
  * `\A---\s*\n`: 파일 맨 첫 줄의 `---`와 줄바꿈
  * `(.*?)`: 여러 줄에 걸친 YAML 본문을 최소 일치로 캡처 (**그룹 1**)
  * `\n---\s*\n`: 닫는 `---`와 줄바꿈
* **코드 예제:**
```python
import re
import yaml  # pip install PyYAML

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)

sample_md = """---
title: "파드"
content_type: concept
---
# 파드(Pod)란?
본문 내용입니다.
"""

match = FRONT_MATTER_RE.match(sample_md)
if match:
    # yaml.safe_load 결과가 None(빈 블록)일 때를 대비해 'or {}'로 빈 딕셔너리 기본값 지정
    metadata = yaml.safe_load(match.group(1)) or {}
    body = sample_md[match.end():]
    print("메타데이터:", metadata)
    print("본문:", body.strip())
```

---

### 2) `TITLE_RE`: YAML 문자열에서 `title` 값 직접 추출
* **패턴:** `r"^title:\s*(.+?)\s*$"` (플래그: `re.M`)
* **기호 설명:**
  * `^title:\s*`: 줄 맨 앞이 `title:`로 시작하고 뒤따르는 공백 무시
  * `(.+?)`: 실제 제목 텍스트만 캡처 (**그룹 1**). 뒤쪽 공백까지 삼키지 않도록 게으른 매칭(`+?`) 사용
  * `\s*$`: 제목 뒤 불필요한 공백을 제외하고 줄 끝까지 매칭
* **코드 예제:**
```python
import re

TITLE_RE = re.compile(r"^title:\s*(.+?)\s*$", re.M)
yaml_block = 'content_type: concept\ntitle: "파드"   \nweight: 10'

match = TITLE_RE.search(yaml_block)
if match:
    clean_title = match.group(1).strip('"\'')  # 양끝 따옴표 제거 -> 파드
    print(clean_title)
```

---

### 3) `COMMENT_RE`: 여러 줄 HTML 주석 제거
* **패턴:** `r"<!--.*?-->"` (플래그: `re.S`)
* **기호 설명:**
  * `<!--` 와 `-->` 사이의 모든 내용을 `re.S`(줄바꿈 포함)와 `*?`(가장 가까운 닫는 태그에서 멈춤)로 매칭하여 빈 문자열로 삭제합니다.
* **코드 예제:**
```python
import re

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
text = "<!-- overview -->\n파드는 기본 단위입니다.\n<!-- 여러 줄\n주석 -->"

print(COMMENT_RE.sub("", text).strip())
# 출력: 파드는 기본 단위입니다.
```

---

### 4) `TOOLTIP_TEXT_RE` & `TOOLTIP_TERM_RE`: Hugo 용어 툴팁 숏코드 변환
* **패턴 1 (text 우선):** `r'\{\{<\s*glossary_tooltip[^}]*?text="([^"]*)"[^}]*?>\}\}'`
* **패턴 2 (term_id 대체):** `r'\{\{<\s*glossary_tooltip[^}]*?term_id="([^"]*)"[^}]*?>\}\}'`
* **기호 설명:**
  * `[^}]*?`: 현재 숏코드가 닫히는 `}` 밖으로 넘어가지 않으면서 속성들을 건너뛰는 안전장치
  * `text="([^"]*)"`: 큰따옴표 안의 표시 단어만 캡처 (**그룹 1**)
* **코드 예제 (실행 순서 중요):**
```python
import re

TOOLTIP_TEXT_RE = re.compile(r'\{\{<\s*glossary_tooltip[^}]*?text="([^"]*)"[^}]*?>\}\}')
TOOLTIP_TERM_RE = re.compile(r'\{\{<\s*glossary_tooltip[^}]*?term_id="([^"]*)"[^}]*?>\}\}')

text = '{{< glossary_tooltip text="컨테이너" term_id="container" >}}와 {{< glossary_tooltip term_id="pod" >}}'

# 1단계: text 속성이 있는 것 먼저 치환
text = TOOLTIP_TEXT_RE.sub(r"\1", text)
# 2단계: 남은 것 중 term_id만 있는 것 치환
text = TOOLTIP_TERM_RE.sub(r"\1", text)

print(text)  # 출력: 컨테이너와 pod
```

---

### 5) `ADMONITION_RE` & `SHORTCODE_RE`: 강조 상자 변환 및 나머지 숏코드 청소
* **패턴 1 (강조 상자):** `r"\{\{[<%]\s*(note|caution|warning)\s*[>%]\}\}"`
* **패턴 2 (나머지 숏코드):** `r"\{\{[<%].*?[>%]\}\}"` (플래그: `re.S`)
* **기호 설명:**
  * `[<%]` / `[>%]`: `{{< note >}}`와 `{{% note %}}` 두 문법을 모두 허용
  * `(note|caution|warning)`: 세 종류 중 하나를 **그룹 1**로 캡처
* **코드 예제:**
```python
import re

ADMONITION_RE = re.compile(r"\{\{[<%]\s*(note|caution|warning)\s*[>%]\}\}")
SHORTCODE_RE  = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)
LABELS = {"note": "참고:", "caution": "주의:", "warning": "경고:"}

text = "{{< note >}}\n베타 기능입니다.\n{{< /note >}}\n{{< figure src=\"/a.png\" >}}"

# 1단계: 여는 강조 태그를 한글 라벨로 변환
text = ADMONITION_RE.sub(lambda m: LABELS[m.group(1)], text)
# 2단계: 닫는 태그({{< /note >}}) 및 기타 모든 숏코드 일괄 삭제
text = SHORTCODE_RE.sub("", text)

print(text.strip())
# 출력:
# 참고:
# 베타 기능입니다.
```

---

### 6) `IMAGE_RE` & `LINK_RE`: 마크다운 이미지 삭제 및 링크 텍스트만 추출
* **패턴 1 (이미지 삭제):** `r"!\[[^\]]*\]\([^)]*\)"`
* **패턴 2 (링크 텍스트 추출):** `r"\[([^\]]+)\]\([^)]*\)"`
* **기호 설명:**
  * `[^\]]+`: 닫는 대괄호 `]`가 나오기 전까지의 텍스트를 **그룹 1**로 캡처
  * `[^)]*`: 닫는 소괄호 `)`가 나오기 전까지의 URL 주소 (캡처하지 않고 버림)
* **코드 예제 (반드시 이미지 먼저 실행):**
```python
import re

IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE  = re.compile(r"\[([^\]]+)\]\([^)]*\)")

text = "자세한 내용은 [컨테이너 런타임](/docs/runtime/)을 보세요. ![그림](/images/a.svg)"

text = IMAGE_RE.sub("", text)       # 이미지 통째로 제거
text = LINK_RE.sub(r"\1", text)     # 링크는 그룹 1(텍스트)만 남김

print(text.strip())
# 출력: 자세한 내용은 컨테이너 런타임을 보세요.
```

---

### 7) `HTML_TAG_RE`: 지정된 HTML 태그만 안전하게 제거
* **패턴:** `r"</?(br|div|span|p|table|tr|td|th|center)\b[^>]*>"` (플래그: `re.I`)
* **기호 설명:**
  * `/?`: 여는 태그(`<br>`)와 닫는 태그(`</div>`) 모두 매칭
  * `\b`: `<bridge>` 같은 오탐 방지용 단어 경계
  * `[^>]*`: 태그 내부 속성(`class="..."` 등) 흡수
  * **장점:** `<pod-name>`처럼 본문 명령어 예시에 쓰인 꺾쇠 표기는 지우지 않고 보호합니다.
* **코드 예제:**
```python
import re

HTML_TAG_RE = re.compile(r"</?(br|div|span|p|table|tr|td|th|center)\b[^>]*>", re.I)
text = '<div class="box">줄<br>바꿈 후 <pod-name> 입력</div>'

print(HTML_TAG_RE.sub("", text))
# 출력: 줄바꿈 후 <pod-name> 입력
```

---

### 8) `ANCHOR_RE` & `HEADING_RE`: 제목 앵커 제거 및 레벨·제목 분리
* **패턴 1 (앵커 제거):** `r"\s*\{#[^}]*\}"`
* **패턴 2 (제목 파싱):** `r"^(#{1,4})\s+(.+?)\s*#*\s*$"`
* **기호 설명:**
  * `\s*\{#[^}]*\}`: 제목 뒤의 `{#overview}`와 그 앞의 공백까지 함께 삭제
  * `(#{1,4})`: `#` 1~4개를 **그룹 1**로 캡처 (`len(group(1))` = 제목 레벨)
  * `(.+?)`: 순수 제목만 **그룹 2**로 캡처 (뒤에 붙는 닫는 `#` 제거)
* **코드 예제:**
```python
import re

ANCHOR_RE  = re.compile(r"\s*\{#[^}]*\}")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")

line = "## 파드 OS {#pod-os}"

# 1단계: 뒤쪽 앵커 제거 -> "## 파드 OS"
clean_line = ANCHOR_RE.sub("", line)

# 2단계: 레벨과 순수 제목 분리
match = HEADING_RE.match(clean_line)
if match:
    level = len(match.group(1))  # 2
    title = match.group(2)       # '파드 OS'
    print(f"H{level}: {title}")  # 출력: H2: 파드 OS
```
