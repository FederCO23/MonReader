import re
from pathlib import Path
import json, time
import pandas as pd



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


