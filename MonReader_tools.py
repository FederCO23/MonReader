import re
from pathlib import Path
import json, time
import pandas as pd
from typing import List, Dict, Optional



def join_hyphenated_linebreaks(lines):
    """Join words split by end-of-line hyphenation: 'escu-' + 'tavam' -> 'escutavam'."""
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i].rstrip()
        if cur.endswith("-") and i + 1 < len(lines):
            nxt = lines[i+1].lstrip()
            # join only if next starts with a letter (unicode-friendly)
            if nxt and re.match(r"^\w", nxt, flags=re.UNICODE):
                cur = cur[:-1] + nxt  # remove '-' and join directly
                out.append(cur)
                i += 2
                continue
        out.append(cur)
        i += 1
    return out

def normalize_for_eval(lines, *, keep_punct=True, keep_accents=True):
    """Normalize text for fair OCR eval (layout-agnostic)."""
    lines = [l for l in lines if l is not None]
    lines = join_hyphenated_linebreaks(lines)

    text = "\n".join(lines)

    # unify whitespace (treat line breaks as spaces for layout-agnostic eval)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text).strip()

    if not keep_punct:
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()

    if not keep_accents:
        # optional: remove accents if you want accent-insensitive scoring
        import unicodedata
        text = "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
    return text

def cer(ref, hyp):
    """Character Error Rate."""
    # simple Levenshtein via dynamic programming
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    dist = dp[m]
    return dist / max(1, n)

def wer(ref, hyp):
    """Word Error Rate (token-based)."""
    r = ref.split()
    h = hyp.split()
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    dist = dp[m]
    return dist / max(1, n)


# Step H config
# ----------------------------

# Image extensions you might have
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_images_sorted(folder: Path):
    imgs = [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS]

    def page_key(p: Path):
        # expected patterns: pag12, pag12-foo, pag_12, etc.
        m = re.search(r"(\d+)", p.stem)   # first number anywhere in stem
        n = int(m.group(1)) if m else 10**9
        # tie-breaker by name to keep stable ordering if needed
        return (n, p.name.lower())

    return sorted(imgs, key=page_key)

def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8", errors="replace")

def safe_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def run_qwen_on_image(
    img_path,
    *,
    model_name,
    prompt,
    options,
    run_ollama_ocr_fn,
    parse_ocr_text_fn,
    looks_degenerate_fn,
):
    out = run_ollama_ocr_fn(model_name, img_path, prompt, options)
    parsed = parse_ocr_text_fn(out["text"])

    lines = parsed.get("lines", [])
    row = {
        "image": img_path.name,
        "image_path": str(img_path),
        "model": model_name,
        "status": out.get("status"),
        "latency_s": out.get("latency_s"),
        "parsed_json": parsed.get("parsed_json", False),
        "language": parsed.get("language", "guess"),
        "json_objs": parsed.get("json_objs", 0),
        "n_lines": len(lines),
        "n_chars_raw": len(out.get("text") or ""),
        "degenerate": looks_degenerate_fn(out.get("text")),
        "parse_error": parsed.get("parse_error"),
        "error": None,
    }
    return row, (out.get("text") or ""), lines


def already_done(txt_out: Path, json_out: Path):
    # treat as done if both exist and non-empty
    return txt_out.exists() and txt_out.stat().st_size > 0 and json_out.exists() and json_out.stat().st_size > 0


# Part 4 helper functions:

# Step I.2 — Explicit verification of page ordering
# Page filenames like "pag2.JPEG" vs "pag10.JPEG" will sort incorrectly with plain string sorting.
# Here we define an explicit "natural sort" that extracts the numeric page id and sorts by it.


BASE = Path.cwd()
WORK_DIR = BASE / "work"

# OCR outputs from Part 3
STEPH_DIR = WORK_DIR / "stepH_qwen2p5vl_full"

# Part 4 outputs
STEP4_DIR = WORK_DIR / "step4_tts"


def page_number_from_name(name: str) -> int:
    """
    Extracts the first integer from filenames like:
      pag0.JPEG, pag2.jpeg, pag10.json, etc.
    Returns a large number if no integer is found (so it goes last).
    """
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else 10**12

def list_pages_sorted(img_or_json_dir: Path, exts=(".JPEG", ".JPG", ".PNG", ".json", ".txt")):
    """
    Lists files in directory, filters by extension, and sorts by numeric page id.
    """
    files = [p for p in img_or_json_dir.iterdir() if p.is_file() and p.suffix.lower() in {e.lower() for e in exts}]
    return sorted(files, key=lambda p: page_number_from_name(p.name))

def verify_page_order(book_id: str, *, which: str = "json", preview: int = 30):
    """
    which: 'json' or 'images' or 'txt'
    Prints the order and returns a dataframe with the computed page numbers.
    """
    base = STEPH_DIR / book_id
    if which == "json":
        d = base / "json"
        exts = (".json",)
    elif which == "txt":
        d = base / "txt"
        exts = (".txt",)
    elif which == "images":
        # if you want to verify original images order, point to your data/books/.../images instead
        raise ValueError("Use which='json' or 'txt' here (Step H outputs). For images, pass your images dir directly.")
    else:
        raise ValueError("which must be one of: 'json', 'txt'.")

    files = list_pages_sorted(d, exts=exts)
    rows = [{"name": f.name, "page_n": page_number_from_name(f.name)} for f in files]
    df = pd.DataFrame(rows).sort_values("page_n").reset_index(drop=True)

    print(f"\n>>> {book_id} ({which})")
    print(f"Directory: {d}")
    print("Order preview:")
    for n in df["name"].head(preview).tolist():
        print(" -", n)

    # quick sanity checks
    if df["page_n"].isna().any():
        print("WARNING: some files had no numeric page id.")
    if df["page_n"].duplicated().any():
        dups = df[df["page_n"].duplicated(keep=False)].sort_values("page_n")
        print("WARNING: duplicated page numbers detected:")
        display(dups)

    return df

# Step I.3 — Extraction and concatenation of page-level text (from Step H JSON)
#
# Goal:
# - Read all per-page JSON artifacts in *correct numeric order*
# - Extract OCR text lines robustly (prefer parsed.lines when valid; otherwise parse raw_response)
# - Concatenate into a single continuous text per book (still preserving paragraph breaks)


# Robust extraction helpers

def list_json_pages_sorted(book_id: str) -> List[Path]:
    json_dir = STEPH_DIR / book_id / "json"
    files = [p for p in json_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    return sorted(files, key=lambda p: page_number_from_name(p.name))

def coerce_lines(lines) -> List[str]:
    if lines is None:
        return []
    if isinstance(lines, list):
        return [str(x) for x in lines]
    if isinstance(lines, str):
        return lines.splitlines()
    return [str(lines)]

def extract_first_json_obj(text: str) -> Optional[dict]:
    if not text:
        return None
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        s = m.group(0)
        try:
            return json.loads(s)
        except Exception:
            continue
    return None

def extract_lines_from_page_obj(page_obj: dict) -> Dict:
    """
    Returns:
      {
        'language': str,
        'lines': List[str],
        'source': 'parsed' | 'raw_response' | 'raw_fallback'
      }
    """
    parsed = page_obj.get("parsed", {}) or {}
    parsed_ok = bool(parsed.get("parsed_json", False))

    # language preference
    lang = parsed.get("language") or page_obj.get("language") or "guess"

    # 1) Use parsed.lines only if parsed_ok=True
    if parsed_ok:
        lines = coerce_lines(parsed.get("lines"))
        # guard: sometimes "lines" got polluted with a JSON dump
        first = (lines[0].lstrip() if lines else "")
        if not (first.startswith("{") and '"lines"' in first):
            return {"language": lang, "lines": lines, "source": "parsed"}

    # 2) Parse raw_response as JSON (or find first JSON object inside it)
    raw = page_obj.get("raw_response", "") or ""
    js = None
    try:
        js = json.loads(raw)
    except Exception:
        js = extract_first_json_obj(raw)

    if isinstance(js, dict) and ("lines" in js):
        lines = coerce_lines(js.get("lines"))
        return {"language": js.get("language", lang), "lines": lines, "source": "raw_response"}

    # 3) Last resort: split raw_response as plain text
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    return {"language": lang, "lines": lines, "source": "raw_fallback"}

def join_lines_for_page(lines: List[str]) -> str:
    """
    Preserve paragraph breaks:
    - keep explicit blank lines as paragraph separators
    - otherwise join as newline (we'll convert to speech-friendly later)
    """
    # normalize None/whitespace-only to empty string
    out = []
    for ln in lines:
        s = (ln or "").rstrip()
        out.append(s)
    return "\n".join(out).strip()

# Book-level extraction and concatenation

def extract_book_pages(book_id: str) -> pd.DataFrame:
    """
    Returns a per-page dataframe in correct order with extracted text.
    Columns: page, page_n, language, source, n_lines, n_chars, text
    """
    pages = list_json_pages_sorted(book_id)
    rows = []

    for p in pages:
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        info = extract_lines_from_page_obj(obj)
        text = join_lines_for_page(info["lines"])
        rows.append({
            "book": book_id,
            "page": p.name,
            "page_n": page_number_from_name(p.name),
            "language": info["language"],
            "source": info["source"],
            "n_lines": len(info["lines"]),
            "n_chars": len(text),
            "text": text,
            "source_path": str(p),
        })

    df = pd.DataFrame(rows).sort_values("page_n").reset_index(drop=True)
    return df

def concatenate_book_text(df_pages: pd.DataFrame) -> str:
    """
    Concatenate pages with a clear page break marker (kept as blank lines).
    """
    parts = []
    for _, r in df_pages.iterrows():
        t = (r["text"] or "").strip()
        if not t:
            continue
        parts.append(t)
        parts.append("\n\n")  # page break
    return "".join(parts).strip()


# Step I.4 — Removal of OCR + formatting artifacts
#
# Goal:
# Remove JSON wrappers, list punctuation, and other non-content artifacts that
# appear in the concatenated text, while keeping the book text intact.

def remove_json_wrapper_and_list_syntax(text: str) -> str:
    """
    Converts a raw concatenation that still contains:
      { "language": "...", "lines": [ "text", ... ] }
    into plain text lines.
    Safe even if the wrapper isn't present.
    """
    if not text:
        return ""

    # 1) Remove obvious JSON scaffolding lines
    #    (keep only meaningful quoted lines content below)
    text = re.sub(r'^\s*\{\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\}\s*,?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*"language"\s*:\s*".*?"\s*,?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*"lines"\s*:\s*\[\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\]\s*,?\s*$', '', text, flags=re.MULTILINE)

    # 2) Extract content inside quotes when the line looks like: "something",
    #    while leaving non-quoted lines untouched.
    def unquote_line(m):
        s = m.group(1)
        # unescape common JSON escapes lightly (no heavy parsing)
        s = s.replace(r'\"', '"').replace(r'\n', '\n').replace(r'\t', '\t')
        return s

    text = re.sub(r'^\s*"(.*)"\s*,?\s*$', unquote_line, text, flags=re.MULTILINE)

    # 3) Remove trailing commas that survived weird formatting
    text = re.sub(r',\s*$', '', text, flags=re.MULTILINE)

    # 4) Normalize excessive blank lines (keep paragraph intent)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# Step I.5 — Repair hyphenated + line-wrapped words
#
# Goal:
# - Join hyphenated line breaks: "ex-\nample" -> "example"
# - Remove artificial line wraps inside paragraphs, while keeping paragraph breaks

def repair_hyphenation_and_wraps(text: str) -> str:
    if not text:
        return ""

    # 1) Join hyphenated line breaks (letters on both sides)
    #    ex-\nample -> example
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 2) Preserve paragraph breaks: mark blank lines temporarily
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)          # normalize blank lines
    text = text.replace("\n\n", "<PARA>")           # protect paragraph breaks

    # 3) Convert remaining newlines (line wraps) to spaces
    text = text.replace("\n", " ")

    # 4) Restore paragraph breaks
    text = text.replace("<PARA>", "\n\n")

    # 5) Cleanup spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Step I.6 — Canonical continuous prose (book-level)
#
# Goal:
# Convert the repaired text into a stable, canonical "prose" representation:
# - Preserve paragraph breaks as "\n\n"
# - Normalize Unicode (NFC)
# - Normalize whitespace and punctuation spacing (minimal, no rewriting)

import unicodedata

def canonicalize_prose(text: str) -> str:
    if not text:
        return ""

    # 1) Unicode normalization (safe for multilingual TTS)
    text = unicodedata.normalize("NFC", text)

    # 2) Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive blank lines

    # 3) Normalize spaces BUT preserve paragraph breaks
    text = text.replace("\n\n", "<PARA>")
    text = re.sub(r"[ \t]+", " ", text)     # collapse runs of spaces
    text = re.sub(r" *\n *", "\n", text)    # trim around any remaining single newlines
    text = text.replace("<PARA>", "\n\n")

    # 4) Minimal punctuation spacing cleanup (safe)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # remove spaces before punctuation
    text = re.sub(r"([,.;:!?])([A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 \2", text)  # ensure space after punctuation

    # 5) Normalize EM DASH only (—). Do NOT touch hyphen-minus "-"
    text = re.sub(r"\s*—\s*", " — ", text)
    text = re.sub(r"[ \t]+", " ", text)     # re-collapse after em-dash spacing

    # final clean
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_paragraphs(canonical: str) -> int:
    if not canonical.strip():
        return 0
    # count paragraph blocks separated by one-or-more blank lines
    return len([p for p in re.split(r"\n\s*\n", canonical.strip()) if p.strip()])


# Step I.7 — Sentence-aware chunking to respect TTS model limits

# Sentence-aware chunker (keeps sentences intact when possible)
def chunk_text_sentence_aware(text: str, max_chars: int = 900) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Split into sentences (simple + robust enough for OCR prose)
    # Keeps ., !, ? boundaries. (Portuguese works fine with this baseline.)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    buf = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # If a single sentence is longer than max_chars, hard-split it
        if len(s) > max_chars:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            for i in range(0, len(s), max_chars):
                chunks.append(s[i:i+max_chars].strip())
            continue

        # Normal case: pack sentences into chunks
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf += " " + s
        else:
            chunks.append(buf.strip())
            buf = s

    if buf:
        chunks.append(buf.strip())

    return chunks


def build_tts_chunks_from_canonical(
    book_texts_canonical: Dict[str, str],
    book_lang: Dict[str, str],
    max_chars: int = 900,
) -> pd.DataFrame:
    rows = []

    for book_id, canon_text in book_texts_canonical.items():
        lang = book_lang.get(book_id, "guess")

        chunks = chunk_text_sentence_aware(canon_text, max_chars=max_chars)
        for i, ch in enumerate(chunks):
            rows.append({
                "book": book_id,
                "chunk_id": i,
                "language": lang,
                "n_chars": len(ch),
                "text": ch,
            })

    return pd.DataFrame(rows)


