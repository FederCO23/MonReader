from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
from tqdm import tqdm

# small utils

def _natural_key(p: Path):
    """Sort 'line_2.png' < 'line_10.png'."""
    s = p.stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _avg_conf_from_tessdata(df: pd.DataFrame) -> float:
    """Get average confidence from pytesseract image_to_data output."""
    if df is None or df.empty or "conf" not in df.columns:
        return 0.0
    # Keep numeric confidences >= 0
    try:
        confs = pd.to_numeric(df["conf"], errors="coerce")
        confs = confs[confs >= 0]
        if len(confs) == 0:
            return 0.0
        return float(confs.mean())
    except Exception:
        return 0.0

def _unhyphenate_lines(lines: List[str]) -> List[str]:
    """
    Join line-ending hyphenations: 'palavra-\nseguinte' -> 'palavra seguinte'.
    Conservative: only if next line starts with a letter (keeps edge cases safer).
    """
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if cur.rstrip().endswith("-") and (i + 1) < len(lines):
            nxt = lines[i + 1]
            if re.match(r"^[A-Za-zÀ-ÿ]", nxt.strip()):
                # drop hyphen and join with a space
                cur = cur.rstrip()[:-1]  # remove trailing '-'
                merged = (cur + nxt.lstrip()).strip()
                out.append(merged)
                i += 2
                continue
        out.append(cur)
        i += 1
    return out

def _clean_text(s: str) -> str:
    """Light normalization: squeeze spaces, normalize quotes/dashes."""
    s = s.replace("\u201C", '"').replace("\u201D", '"')  # smart quotes -> "
    s = s.replace("\u2018", "'").replace("\u2019", "'")  # curly apostrophes -> '
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash -> hyphen
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


# OCR core

def ocr_line_image(img_path: Path, lang: str = "eng", oem: int = 1, psm: int = 6) -> Tuple[str, float]:
    """
    OCR a single line image and return (text, avg_conf).
    Uses pytesseract.image_to_data for confidences.
    """
    pil = Image.open(img_path).convert("L")  # grayscale
    cfg = f"--oem {oem} --psm {psm}"
    data = pytesseract.image_to_data(pil, lang=lang, config=cfg, output_type=pytesseract.Output.DATAFRAME)
    text = ""
    if data is not None and not data.empty and "text" in data.columns:
        # Rebuild line-level text by space-joining word-level tokens
        tokens = [t for t in data["text"].astype(str).tolist() if t and t != "nan"]
        text = " ".join(tokens)
    avg_conf = _avg_conf_from_tessdata(data)
    return _clean_text(text), float(avg_conf)

def _infer_lang_from_path(page_dir: Path) -> str:
    """
    Infer 'eng' or 'por' from the <lang> folder under C_layout.
    Adjust mapping as needed.
    """
    lang_folder = page_dir.parent.name.lower()
    if "portuguese" in lang_folder or "por" in lang_folder:
        return "por"
    if "english" in lang_folder or "eng" in lang_folder:
        return "eng"
    # fallback to both (slower but safer if mixed)
    return "eng+por"

def _collect_line_paths(layout_root: Path) -> List[Tuple[str, Path, List[Path]]]:
    """
    Return a list of (lang, page_stem, [line_paths...]) discovered in C_layout.
    """
    results = []
    for lang_dir in sorted(layout_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        for page_dir in sorted(lang_dir.iterdir(), key=_natural_key):
            if not page_dir.is_dir():
                continue
            lines = sorted(page_dir.glob("line_*.png"), key=_natural_key)
            if not lines:
                continue
            results.append((lang_dir.name, page_dir.name, lines))
    return results

def run_ocr_from_layout(layout_root: Path,
                        out_root: Path,
                        oem: int = 1,
                        psm: int = 6,
                        lang_override: Optional[str] = None) -> Path:
    """
    Walk C_layout/<lang>/<page>/line_*.png, run Tesseract, and write:
      - E_text/ocr_lines.csv   (language, page, line_id, text, conf, oem, psm)
      - E_text/pages/<lang>/<page>.txt    (stitched + cleaned per-page)
      - E_text/books/<lang>.txt           (concatenated per-book)
    """
    out_root = Path(out_root)
    csv_rows = []
    page_text_map: Dict[Tuple[str, str], List[str]] = {}

    jobs = _collect_line_paths(Path(layout_root))
    for lang_name, page_stem, line_paths in tqdm(jobs, desc="OCR lines"):
        # language
        lang = lang_override or _infer_lang_from_path(Path(layout_root) / lang_name / page_stem)

        # OCR lines (sorted)
        page_lines_text = []
        for lp in line_paths:
            text, conf = ocr_line_image(lp, lang=lang, oem=oem, psm=psm)
            line_id = lp.stem  # e.g., 'line_012'
            csv_rows.append({
                "language": lang_name,
                "page": page_stem,
                "line_id": line_id,
                "text_raw": text,
                "conf": conf,
                "oem": oem,
                "psm": psm,
                "lang_used": lang,
            })
            page_lines_text.append(text)

        # assemble per-page text
        page_lines_text = _unhyphenate_lines(page_lines_text)
        page_text_map[(lang_name, page_stem)] = page_lines_text

    # --- write CSV
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "ocr_lines.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    # --- write per-page txt
    pages_root = out_root / "pages"
    books_root = out_root / "books"
    pages_root.mkdir(parents=True, exist_ok=True)
    books_root.mkdir(parents=True, exist_ok=True)

    # aggregate per-book
    book_texts: Dict[str, List[str]] = {}

    for (lang_name, page_stem), lines in sorted(page_text_map.items()):
        page_dir = pages_root / lang_name
        page_dir.mkdir(parents=True, exist_ok=True)
        page_text = "\n".join(lines).strip()
        (page_dir / f"{page_stem}.txt").write_text(page_text, encoding="utf-8")

        book_texts.setdefault(lang_name, []).append(page_text)

    # --- write book txts
    for lang_name, page_texts in book_texts.items():
        book_txt = "\n\n".join(page_texts).strip()
        (books_root / f"{lang_name}.txt").write_text(book_txt, encoding="utf-8")

    print(f"Saved OCR CSV -> {csv_path}")
    print(f"Saved per-page .txt -> {pages_root}/<lang>/<page>.txt")
    print(f"Saved per-book .txt -> {books_root}/<lang>.txt")
    return csv_path
