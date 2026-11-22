#!/usr/bin/env python3
"""쿠버네티스 문서(content/<언어>/docs)를 정제해서 지시학습(SFT)·RAG 용 JSONL 로 만든다.
계속 사전학습(CPT)용 데이터는 03_build_cpt.py 가 이 모듈의 파서(iter_docs 등)를 재사용해 만든다.

Hugo 숏코드는 사이트가 렌더링하는 결과에 가깝게 치환한다:
예제 YAML·include·용어 정의는 원본 파일을 읽어 끼워 넣고, 제목·버전·기능 상태는 문자열로 바꾼다.

출력:
  data/sft/{train,valid}.jsonl  {"messages": [...]}  질의응답 형식 지시학습용
  data/chunks.jsonl             RAG 등 재사용용 원본 청크
"""
import argparse
import functools
import json
import random
import re
import shlex
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SYSTEM_PROMPT = (
    "당신은 쿠버네티스 전문가입니다. 쿠버네티스 공식 한글 문서에 근거하여 "
    "정확하고 간결하게 한국어로 답변하세요."
)

QUESTION_TEMPLATES = [
    "{title} 문서에서 '{heading}' 내용을 설명해줘.",
    "쿠버네티스 {title} 관련해서 {heading}에 대해 알려줘.",
    "{title}의 {heading} 부분을 정리해 주세요.",
    "{heading}에 대해 {title} 관점에서 설명해 주세요.",
]
QUESTION_TEMPLATES_NO_HEADING = [
    "쿠버네티스의 {title}에 대해 설명해줘.",
    "{title}이(가) 무엇인가요?",
    "{title} 개념을 정리해 주세요.",
]

# ── 문서 구조 ────────────────────────────────────────────────────────────────
FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
TITLE_RE = re.compile(r"^title:\s*(.+?)\s*$", re.M)
HEADLESS_RE = re.compile(r"^headless:\s*true\s*$", re.M)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
FENCE_OPEN_RE = re.compile(r"[ \t]*(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
ANCHOR_RE = re.compile(r"\s*\{#[^}]*\}")

# ── Hugo 숏코드: {{< 이름 인자 >}} / {{% 이름 인자 %}}, 닫는 태그는 {{< /이름 >}} 또는 {{</ 이름 >}} ──
def _sc(name, args=r"(?P<args>[^}]*?)"):
    return rf"\{{\{{[<%]\s*{name}\b{args}\s*/?\s*[>%]\}}\}}"


def _close(name):
    return rf"\{{\{{[<%]\s*/\s*{name}\s*[>%]\}}\}}"


ESCAPED_SC_RE = re.compile(r"\{\{([<%])/\*(.*?)\*/([>%])\}\}", re.S)       # {{</* note */>}} = 문법 예시
COMMENT_SC_RE = re.compile(_sc("comment") + r".*?" + _close("comment"), re.S)
MERMAID_SC_RE = re.compile(_sc("mermaid") + r".*?" + _close("mermaid"), re.S)
HEADING_SC_RE = re.compile(_sc("heading", r'\s+"(?P<key>[^"]+)"'))
CODE_SAMPLE_RE = re.compile(_sc("(?:code_sample|codenew)"))
TAB_CODE_RE = re.compile(_sc("tab", r'(?P<args>[^}]*?codelang="[^"]*"[^}]*?)') + r"(?P<body>.*?)" + _close("tab"), re.S)
TAB_RE = re.compile(_sc("tab"))
INCLUDE_RE = re.compile(_sc("include", r'\s+"(?P<name>[^"]+)"'))
GLOSSARY_DEF_RE = re.compile(_sc("glossary_definition"))
FEATURE_STATE_RE = re.compile(_sc("feature-state"))
FIGURE_RE = re.compile(_sc("figure"))
SKEW_RE = re.compile(_sc("skew"))
PARAM_RE = re.compile(_sc("param", r'\s*"?(?P<name>[A-Za-z_]+)"?'))
TOOLTIP_TEXT_RE = re.compile(r'\{\{<\s*glossary_tooltip[^}]*?text="([^"]*)"[^}]*?>\}\}')
TOOLTIP_TERM_RE = re.compile(r'\{\{<\s*glossary_tooltip[^}]*?term_id="([^"]*)"[^}]*?>\}\}')
ADMONITION_RE = re.compile(r"\{\{[<%]\s*(note|caution|warning)\s*[>%]\}\}")
SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)
ARG_RE = re.compile(r'(\w+)="([^"]*)"')

# ── 마크다운 / HTML ─────────────────────────────────────────────────────────
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_TAG_RE = re.compile(
    r"</?(br|div|span|p|a|img|table|thead|tbody|tr|td|th|ul|ol|li|b|i|em|strong"
    r"|sup|sub|details|summary|center)\b[^>]*>",
    re.I,
)
ADMONITION_LABEL = {"note": "참고:", "caution": "주의:", "warning": "경고:"}
PLACEHOLDER_RE = re.compile("\x00(\\d+)\x00")
MAX_INCLUDE_DEPTH = 3

STATS = Counter()   # 치환 결과 집계 (main 에서 출력)


# ── 사이트 정보 (번역 문자열, 버전, 자원 경로) ─────────────────────────────────
@dataclass(frozen=True)
class Site:
    repo: Path        # raw/website
    lang: str         # ko
    i18n: dict        # {"whatsnext_heading": "다음 내용", ...}  (없는 키는 영어로 대체)
    version: str      # hugo.toml params.version  예: "v1.37"
    latest: str       # hugo.toml params.latest
    patch: str        # 현재 버전의 최신 패치  예: "1.37.1"

    @property
    def content(self):
        return self.repo / "content" / self.lang

    def t(self, key, default=""):
        return self.i18n.get(key, default)

    def find(self, *relative):
        """content/<lang>/ 에서 찾고, 없으면 content/en/ (사이트 기본 언어)에서 찾는다."""
        for lang in dict.fromkeys([self.lang, "en"]):
            path = self.repo / "content" / lang / Path(*relative)
            if path.is_file():
                return path
        return None


def _load_i18n(repo, lang):
    table = {}
    for lg in ("en", lang):                       # 영어를 먼저 읽고 해당 언어로 덮어씀
        path = repo / "i18n" / lg / f"{lg}.toml"
        if path.exists():
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            table.update({k: v["other"] for k, v in data.items() if isinstance(v, dict) and "other" in v})
    return table


def _patch_version(repo, current):
    """data/releases/schedule.yaml 에서 current 릴리스의 가장 최근 패치 (Hugo skew 와 같은 규칙)."""
    path = repo / "data" / "releases" / "schedule.yaml"
    if path.exists():
        for entry in re.split(r"(?m)^- ", path.read_text(encoding="utf-8")):
            if re.search(rf'(?m)^  release:\s*"?{re.escape(current)}"?\s*$', entry) and "previousPatches:" in entry:
                m = re.search(r"release:\s*\"?(\d+\.\d+\.\d+)", entry.split("previousPatches:", 1)[1])
                if m:
                    return m.group(1)
    return f"{current}.0"


@functools.lru_cache(maxsize=None)
def load_site(repo, lang):
    params = {}
    if (repo / "hugo.toml").exists():
        params = tomllib.loads((repo / "hugo.toml").read_text(encoding="utf-8")).get("params", {})
    version = params.get("version", "v1.0")
    return Site(repo, lang, _load_i18n(repo, lang), version, params.get("latest", version),
                _patch_version(repo, version.lstrip("v")))


@dataclass(frozen=True)
class Page:
    path: Path
    site: Site


def page_for(path):
    """raw/website/content/<lang>/docs/... 경로에서 사이트 정보를 찾는다. 해당 구조가 아니면 None."""
    parts = path.resolve().parts
    if "content" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("content")
    if i + 1 >= len(parts):
        return None
    return Page(path.resolve(), load_site(Path(*parts[:i]), parts[i + 1]))


# ── 코드 블록 판별 ──────────────────────────────────────────────────────────
def fence_states(lines):
    """줄마다 (이 줄이 코드 블록에 속하나, 이 줄 다음에도 코드 블록이 열려 있나) 를 돌려준다.

    여는 펜스: ``` 또는 ~~~ 3개 이상. 같은 줄에 ` 가 또 있으면 (```kubectl -A```) 펜스가 아니다.
    닫는 펜스: 여는 펜스와 같은 문자로만 된 줄, 길이는 여는 펜스 이상.
    """
    fence, states = None, []
    for line in lines:
        if fence is None:
            m = FENCE_OPEN_RE.match(line)
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                fence = m.group(1)
                states.append((True, True))
            else:
                states.append((False, False))
        else:
            s = line.strip()
            if s and set(s) == {fence[0]} and len(s) >= len(fence):
                fence = None
                states.append((True, False))
            else:
                states.append((True, True))
    return states


def split_code(text):
    """텍스트를 [(코드 여부, 조각), ...] 로 나눈다. 연속된 같은 종류의 줄을 한 조각으로 묶는다."""
    lines = text.split("\n")
    segments = []
    for line, (is_code, _) in zip(lines, fence_states(lines)):
        if segments and segments[-1][0] == is_code:
            segments[-1][1].append(line)
        else:
            segments.append((is_code, [line]))
    return [(is_code, "\n".join(ls)) for is_code, ls in segments]


# ── 숏코드 치환 ─────────────────────────────────────────────────────────────
def _args(s):
    return dict(ARG_RE.findall(s or ""))


def _versions(site, text):
    """skew / param 버전 숏코드를 hugo.toml 값으로 치환한다 (Hugo layouts/shortcodes/skew.html 규칙)."""
    def skew(m):
        args = shlex.split(m.group("args"))
        if not args:
            return ""
        kind, base = args[0], site.version.lstrip("v")
        if kind.startswith("latest") or kind in ("nextMinorVersion", "prevMinorVersion", "oldestMinorVersion"):
            base = site.latest.lstrip("v")
        major, minor = base.split(".")[:2]
        offset = {"nextMinorVersion": 1, "prevMinorVersion": -1, "oldestMinorVersion": -2}.get(kind)
        if kind in ("currentVersion", "latestVersion"):
            out = base
        elif kind == "currentPatchVersion":
            out = site.patch
        elif kind.endswith("AddMinor"):
            sep = args[2] if len(args) > 2 and args[2] else "."
            out = f"{major}{sep}{int(minor) + int(args[1])}"
        elif offset is not None:
            out = f"{major}.{int(minor) + offset}"
        else:
            return m.group(0)
        STATS["skew"] += 1
        return out

    text = SKEW_RE.sub(skew, text)
    params = {"version": site.version, "latest": site.latest}
    return PARAM_RE.sub(lambda m: params.get(m.group("name"), ""), text)


def _feature_state(site, args):
    """feature-state → "기능 상태: Kubernetes v1.25 [stable]" (사이트의 기능 상태 상자와 같은 정보)."""
    a = _args(args)
    state, version = a.get("state"), a.get("for_k8s_version", site.version)
    if a.get("feature_gate_name"):
        state, version = None, None
        path = site.find("docs/reference/command-line-tools-reference/feature-gates", f"{a['feature_gate_name']}.md")
        current = tuple(int(x) for x in site.version.lstrip("v").split(".")[:2])
        if path:
            for stage, frm in re.findall(r'-\s*stage:\s*(\w+).*?fromVersion:\s*"?([\d.]+)', path.read_text(encoding="utf-8"), re.S):
                if tuple(int(x) for x in frm.split(".")[:2]) <= current:
                    state, version = stage, frm
        if not state:
            return f"{site.t('feature_state', '기능 상태:')} 기능 게이트 {a['feature_gate_name']}"
    STATS["feature-state"] += 1
    label = site.t("feature_state_kubernetes_label", "Kubernetes")
    return f"{site.t('feature_state', '기능 상태:')} {label} v{str(version).lstrip('v')} [{state}]"


def _read_body(path):
    raw = path.read_text(encoding="utf-8")
    m = FRONT_MATTER_RE.match(raw)
    return raw[m.end():] if m else raw


def _include(page, name, depth):
    """include "이름": content/<lang>/includes/ 에서 이름으로 시작하는 파일, 없으면 문서 기준 상대 경로."""
    site = page.site
    candidates = sorted((site.content / "includes").glob(f"{name}*")) + [page.path.parent / name]
    for path in candidates:
        if path.is_file() and path.name != "index.md":
            STATS["include"] += 1
            return clean_markdown(_read_body(path), Page(path, site), depth + 1)
    STATS["include_missing"] += 1
    return ""


def _glossary_definition(page, args, depth):
    a = _args(args)
    path = page.site.find("docs/reference/glossary", f"{a.get('term_id', '')}.md")
    if not path:
        STATS["glossary_definition_missing"] += 1
        return ""
    text = clean_markdown(_read_body(path), Page(path, page.site), depth + 1)
    if a.get("length") == "short":                # 사이트는 첫 문단만 보여준다
        text = text.split("\n\n")[0]
    if a.get("prepend"):
        text = f"{a['prepend']} {text}"
    STATS["glossary_definition"] += 1
    return text


def _code_sample(page, args):
    """code_sample / codenew: examples/ 의 파일을 언어가 표시된 코드 블록으로 끼워 넣는다."""
    a = _args(args)
    file = a.get("file", "").lstrip("/")
    path = page.site.find("examples", file) or (page.path.parent / file if (page.path.parent / file).is_file() else None)
    if not path:
        STATS["code_sample_missing"] += 1
        return ""
    STATS["code_sample"] += 1
    lang = a.get("language") or Path(file).suffix.lstrip(".")
    return f"`{file}`\n\n```{lang}\n{path.read_text(encoding='utf-8').rstrip()}\n```"


def expand_shortcodes(text, page, depth, vault):
    """코드 블록 밖 텍스트의 숏코드를 렌더링 결과로 바꾼다. 다시 정제되면 안 되는 결과는 vault 에 보관한다."""
    def keep(s):
        vault.append(s)
        return f"\x00{len(vault) - 1}\x00"

    text = ESCAPED_SC_RE.sub(lambda m: keep(f"{{{{{m.group(1)}{m.group(2)}{m.group(3)}}}}}"), text)
    if page is None:
        return text
    site = page.site
    text = HEADING_SC_RE.sub(lambda m: site.t(f"{m.group('key')}_heading", m.group("key")), text)
    text = CODE_SAMPLE_RE.sub(lambda m: keep(_code_sample(page, m.group("args"))), text)
    text = TAB_CODE_RE.sub(
        lambda m: f"\n**{_args(m.group('args')).get('name', '')}**\n\n"
        + keep(f"```{_args(m.group('args'))['codelang']}\n{m.group('body').strip()}\n```"), text)

    def tab(m):
        a = _args(m.group("args"))
        head = f"\n**{a['name']}**\n\n" if a.get("name") else ""
        return head + (keep(_include(page, a["include"], depth)) if a.get("include") else "")

    text = TAB_RE.sub(tab, text)
    if depth < MAX_INCLUDE_DEPTH:
        text = INCLUDE_RE.sub(lambda m: keep(_include(page, m.group("name"), depth)), text)
        text = GLOSSARY_DEF_RE.sub(lambda m: keep(_glossary_definition(page, m.group("args"), depth)), text)
    text = FEATURE_STATE_RE.sub(lambda m: _feature_state(site, m.group("args")), text)

    def figure(m):
        a = _args(m.group("args"))
        desc = a.get("caption") or a.get("alt") or a.get("title")
        return f"(그림: {desc})" if desc else ""

    text = FIGURE_RE.sub(figure, text)
    return _versions(site, text)


def clean_prose(text):
    """코드 블록 바깥의 텍스트에서 남은 Hugo 숏코드/링크/HTML 을 정리한다."""
    text = TOOLTIP_TEXT_RE.sub(r"\1", text)
    text = TOOLTIP_TERM_RE.sub(r"\1", text)
    text = ADMONITION_RE.sub(lambda m: ADMONITION_LABEL[m.group(1)], text)
    text = SHORTCODE_RE.sub("", text)
    text = IMAGE_RE.sub("", text)
    text = LINK_RE.sub(r"\1", text)
    return HTML_TAG_RE.sub("", text)


def clean_code(text, page):
    """코드 블록 안: Hugo 는 코드 안의 숏코드도 처리하므로 버전 값과 이스케이프만 풀어 준다."""
    text = ESCAPED_SC_RE.sub(lambda m: f"{{{{{m.group(1)}{m.group(2)}{m.group(3)}}}}}", text)
    return _versions(page.site, text) if page else text


def clean_markdown(text, page=None, depth=0):
    """마크다운 본문을 정제한다. page 를 주면 사이트 자원(예제, include, 번역, 버전)으로 숏코드를 치환한다."""
    text = COMMENT_RE.sub("", text)
    text = COMMENT_SC_RE.sub("", text)            # 사이트에 표시되지 않는 숨김 주석
    text = MERMAID_SC_RE.sub("", text)            # 다이어그램 코드 (학습 가치 낮음)
    vault, parts = [], []
    for is_code, seg in split_code(text):
        if is_code:
            parts.append(clean_code(seg, page))
        else:
            parts.append(clean_prose(expand_shortcodes(seg, page, depth, vault)))
    text = "\n".join(parts)
    while PLACEHOLDER_RE.search(text):            # 보관해 둔 코드·include 결과를 제자리에 복원
        text = PLACEHOLDER_RE.sub(lambda m: vault[int(m.group(1))], text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ── 문서 읽기 ───────────────────────────────────────────────────────────────
def read_front_matter(raw):
    m = FRONT_MATTER_RE.match(raw)
    return (m.group(1), raw[m.end():]) if m else ("", raw)


def parse_doc(path):
    """문서 파일 하나 → (제목, 정제된 본문)."""
    meta, body = read_front_matter(path.read_text(encoding="utf-8"))
    title = path.stem
    t = TITLE_RE.search(meta)
    if t:
        title = t.group(1).strip("'\"")
    if title in ("_index", "index"):
        title = path.parent.name
    return title, clean_markdown(body, page_for(path))


def is_headless(path):
    """headless: true 문서는 단독 페이지가 아니라 include 로 끼워 넣는 조각이다."""
    meta, _ = read_front_matter(path.read_text(encoding="utf-8"))
    return bool(HEADLESS_RE.search(meta))


def iter_docs(docs_dir):
    """docs_dir 아래 문서를 (경로, 제목, 본문) 으로 돌려준다. headless 조각은 건너뛴다."""
    for path in sorted(Path(docs_dir).rglob("*.md")):
        if is_headless(path):
            STATS["headless_skipped"] += 1
            continue
        title, body = parse_doc(path)
        yield path, title, body


# ── 섹션 / 조각 나누기 ───────────────────────────────────────────────────────
def split_sections(body):
    """제목(#~####) 기준으로 (상위 > 하위 제목, 본문) 목록으로 나눈다. 코드 블록 안의 # 은 무시한다."""
    sections, stack, buf = [], [], []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((" > ".join(h for _, h in stack), text))

    lines = body.splitlines()
    for line, (is_code, _) in zip(lines, fence_states(lines)):
        m = None if is_code else HEADING_RE.match(line)
        if m:
            flush()
            buf = []
            level, heading = len(m.group(1)), ANCHOR_RE.sub("", m.group(2)).strip()
            stack = [(lv, h) for lv, h in stack if lv < level] + [(level, heading)]
        else:
            buf.append(line)
    flush()
    return sections


def split_long(text, max_chars):
    """코드 블록을 깨지 않도록 빈 줄 기준으로 max_chars 근처에서 자른다."""
    parts, buf, size = [], [], 0
    lines = text.splitlines()
    for line, (_, open_after) in zip(lines, fence_states(lines)):
        buf.append(line)
        size += len(line) + 1
        if not open_after and (
            (size >= max_chars and not line.strip()) or size >= max_chars * 2
        ):
            parts.append("\n".join(buf).strip())
            buf, size = [], 0
    parts.append("\n".join(buf).strip())
    return [p for p in parts if p]


# ── SFT / RAG 데이터 ─────────────────────────────────────────────────────────
def build_chunks(docs_dir, min_chars, max_chars):
    chunks, n_docs = [], 0
    for path, title, body in iter_docs(docs_dir):
        n_docs += 1
        for heading, text in split_sections(body):
            pieces = split_long(text, max_chars)
            for i, piece in enumerate(pieces):
                if len(piece) < min_chars:
                    continue
                h = heading
                if len(pieces) > 1:
                    h = f"{heading or title} ({i + 1}/{len(pieces)})"
                chunks.append({
                    "source": str(path.relative_to(docs_dir)),
                    "title": title,
                    "heading": h,
                    "text": piece,
                })
    return n_docs, chunks


def split_by_document(items, key, ratio, rng):
    """같은 문서의 항목이 train 과 valid 에 나뉘지 않도록 문서 단위로 valid 를 고른다."""
    docs = {}
    for item in items:
        docs.setdefault(key(item), []).append(item)
    names = sorted(docs)
    rng.shuffle(names)
    target = max(1, int(len(items) * ratio))
    valid, train, valid_docs = [], [], []
    for name in names:
        if len(valid) < target:
            valid += docs[name]
            valid_docs.append(name)
        else:
            train += docs[name]
    return train, valid, valid_docs


def to_sft(chunk, rng):
    if chunk["heading"]:
        q = rng.choice(QUESTION_TEMPLATES).format(**chunk)
    else:
        q = rng.choice(QUESTION_TEMPLATES_NO_HEADING).format(**chunk)
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        {"role": "assistant", "content": chunk["text"]},
    ]}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_stats():
    keys = ["code_sample", "code_sample_missing", "include", "include_missing", "glossary_definition",
            "glossary_definition_missing", "feature-state", "skew", "headless_skipped"]
    print("숏코드 치환: " + ", ".join(f"{k} {STATS[k]}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="raw/website/content/ko/docs")
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-chars", type=int, default=150)
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--valid-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_docs, chunks = build_chunks(Path(args.docs), args.min_chars, args.max_chars)
    train, valid, _ = split_by_document(chunks, lambda c: c["source"], args.valid_ratio, rng)
    rng.shuffle(train)

    out = Path(args.out)
    write_jsonl(out / "sft" / "train.jsonl", [to_sft(c, rng) for c in train])
    write_jsonl(out / "sft" / "valid.jsonl", [to_sft(c, rng) for c in valid])
    write_jsonl(out / "chunks.jsonl", chunks)  # RAG 등 재사용용 원본 청크

    avg = sum(len(c["text"]) for c in chunks) / max(1, len(chunks))
    print(f"문서 {n_docs}개 -> 청크 {len(chunks)}개 (평균 {avg:.0f}자)")
    print(f"train {len(train)} / valid {len(valid)} -> {out}/")
    print_stats()


if __name__ == "__main__":
    main()
