from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import ast

# core segmentation

def _row_density(mask: np.ndarray) -> np.ndarray:
    """Fraction of foreground per row (text=255)."""
    H, W = mask.shape[:2]
    return (mask > 0).sum(axis=1).astype(np.float32) / float(W)

def _col_density(mask: np.ndarray) -> np.ndarray:
    """Fraction of foreground per column (text=255)."""
    H, W = mask.shape[:2]
    return (mask > 0).sum(axis=0).astype(np.float32) / float(H)

def _smooth_1d(x: np.ndarray, k: int = 11) -> np.ndarray:
    """Simple moving average with odd window size."""
    k = max(3, k | 1)  # force odd
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")

def _intervals_from_binary(flags: np.ndarray, min_len: int = 3) -> List[Tuple[int, int]]:
    """Return [(start, end)] runs where flags==True; end is exclusive."""
    runs = []
    in_run = False
    s = 0
    for i, v in enumerate(flags):
        if v and not in_run:
            in_run = True; s = i
        elif not v and in_run:
            if i - s >= min_len:
                runs.append((s, i))
            in_run = False
    if in_run and len(flags) - s >= min_len:
        runs.append((s, len(flags)))
    return runs

def segment_lines(mask: np.ndarray,
                  row_smooth_k: int = 25,
                  row_bin_method: str = "otsu",
                  min_line_height: int = 12,
                  pad: int = 2) -> List[Tuple[int,int,int,int]]:
    """
    Detect line boxes from a text=white mask.
    Returns list of (x, y, w, h) boxes.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    H, W = mask.shape[:2]

    # Row density -> smooth -> binarize rows-as-text
    dens_r = _row_density(mask)
    dens_r_s = _smooth_1d(dens_r, k=row_smooth_k)

    if row_bin_method == "otsu":
        # Otsu on scaled densities
        vals = (dens_r_s * 255).astype(np.uint8)
        _, thr = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu returns threshold value (scalar), not mask, we recompute density threshold
        # Convert that pixel-level threshold back to normalized 0..1 scale
        thv = _ / 255.0 if _ > 0 else float(np.percentile(dens_r_s, 40))
        
    else:
        # fixed fallback (very conservative)
        thv = max(0.02, float(np.percentile(dens_r_s, 40)))

    # Ensure it's a scalar float
    thv = float(np.squeeze(thv))

    row_flags = dens_r_s >= thv
    line_rows = _intervals_from_binary(row_flags, min_len=min_line_height)

    boxes: List[Tuple[int,int,int,int]] = []
    for y0, y1 in line_rows:
        # For each line band, get horizontal crop and determine column extents
        band = mask[y0:y1, :]
        dens_c = _col_density(band)
        # threshold columns with small smoothing to ignore side noise
        dens_c_s = _smooth_1d(dens_c, k=9)
        # adaptive threshold as fraction of max (handles ragged lines)
        if dens_c_s.max() > 0:
            c_thr = 0.1 * float(dens_c_s.max())
        else:
            c_thr = 0.0
        col_flags = dens_c_s >= c_thr
        cols = np.where(col_flags)[0]
        if cols.size == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1] + 1)

        # pad and clip
        x0 = max(0, x0 - pad); x1 = min(W, x1 + pad)
        y0p = max(0, y0 - pad); y1p = min(H, y1 + pad)
        w = x1 - x0; h = y1p - y0p
        if h >= min_line_height and w >= 16:
            boxes.append((x0, y0p, w, h))

    # optional small merge of very close lines (touching after pad)
    merged: List[Tuple[int,int,int,int]] = []
    for b in boxes:
        if not merged:
            merged.append(b); continue
        x, y, w, h = b
        X, Y, Wb, Hb = merged[-1]
        # if vertical gap small and horizontal overlap large, merge
        gap = y - (Y + Hb)
        overlap = min(X+Wb, x+w) - max(X, x)
        if gap <= 2 and overlap > 0.5 * min(Wb, w):
            ny = Y; nh = (y + h) - Y
            nx = min(X, x); nw = max(X+Wb, x+w) - nx
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(b)

    return merged

# IO helpers

def load_selected_mask(page_dir: Path) -> np.ndarray | None:
    """Read bw_selected_mask.png (text=white) from a binarized page folder."""
    p = page_dir / "bw_selected_mask.png"
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return img

def save_line_overlays(view_img: np.ndarray,
                       boxes: List[Tuple[int,int,int,int]],
                       out_path: Path):
    """Draw line boxes over a view image (text black), save PNG."""
    if view_img.ndim == 2:
        vis = cv2.cvtColor(view_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = view_img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(str(out_path), vis)

# runner
def run_layout_from_binarization(bin_csv: Path,
                                 out_root: Path,
                                 method: str = "morph",          # "morph" or "proj"
                                 ingest_csv: Path | None = None  # optional: crop to Step-0 bbox
                                 ):
    """
    For each row in B_binarization/binarization_log.csv:
      - load selected mask/view
      - segment lines (morph or projection; optionally cropped to Step-0 bbox)
      - save overlay + line crops
      - append layout stats to C_layout/layout_log.csv
    """
    df = pd.read_csv(bin_csv, dtype={"page_out": "string"}).dropna(subset=["page_out"]).copy()
    bbox_map = _load_ingest_bboxes(ingest_csv) if (ingest_csv and Path(ingest_csv).exists()) else {}

    logs: List[Dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Layout"):
        page_dir = Path(str(row["page_out"]))  # .../B_binarization/<lang>/<page>
        mask = load_selected_mask(page_dir)
        if mask is None:
            continue

        # Human-friendly background for overlays/crops
        view_path = page_dir / "bw_selected_view.png"
        view = cv2.imread(str(view_path), cv2.IMREAD_GRAYSCALE) if view_path.exists() else (255 - mask)

        # Determine bbox (if provided in ingest CSV)
        page_stem = page_dir.name
        bbox = bbox_map.get(page_stem, None)

        # ----- segmentation choice -----
        if method == "morph":
            boxes = segment_lines_morph(mask, bbox=bbox)
        else:
            if bbox is not None:
                x, y, w, h = bbox
                sub = mask[y:y+h, x:x+w]
                sub_boxes = segment_lines(sub)
                boxes = [(bx + x, by + y, bw, bh) for (bx, by, bw, bh) in sub_boxes]
            else:
                boxes = segment_lines(mask)

        # ----- prepare output dir in C_layout -----
        lang = page_dir.parent.name
        page = page_dir.name
        page_out = out_root / lang / page
        page_out.mkdir(parents=True, exist_ok=True)

        # ----- save overlay + line crops -----
        save_line_overlays(view, boxes, page_out / "overlay_lines.png")
        for i, (lx, ly, lw, lh) in enumerate(boxes):
            crop = view[ly:ly+lh, lx:lx+lw]
            cv2.imwrite(str(page_out / f"line_{i:03d}.png"), crop)

        # ----- log stats -----
        heights = [h for (_, _, _, h) in boxes]
        widths  = [w for (_, _, w, _) in boxes]
        logs.append({
            "language": lang,
            "page_out": str(page_out),
            "n_lines": len(boxes),
            "avg_line_height": float(np.mean(heights)) if heights else 0.0,
            "avg_line_width":  float(np.mean(widths))  if widths  else 0.0,
        })

    # write log
    out_root.mkdir(parents=True, exist_ok=True)
    log_csv = out_root / "layout_log.csv"
    pd.DataFrame(logs).to_csv(log_csv, index=False)
    print(f"Saved layout artifacts to: {out_root}\nLog -> {log_csv}")


# -----------------------------
# ----- WORD SEGMENTATION -----
# -----------------------------

def _segment_words_in_line(line_mask: np.ndarray,
                           min_word_w: int = 8,
                           gap_dilate_frac: float = 0.02,
                           pad: int = 1) -> list[tuple[int,int,int,int]]:
    """
    Split a single line (mask, text=white) into word boxes.
    Uses a small horizontal dilation to bridge intra-word gaps.
    Returns list of (x, y, w, h) in the line's coordinate frame.
    """
    if line_mask.dtype != np.uint8:
        line_mask = ((line_mask > 0).astype(np.uint8) * 255)

    H, W = line_mask.shape[:2]
    if W < 10 or H < 6:
        return []

    # Gentle dilation to merge letters inside a word
    kx = max(1, int(W * gap_dilate_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    merged = cv2.morphologyEx(line_mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Find connected components (each should correspond roughly to a word)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    boxes = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
        if w < min_word_w or h < max(6, H // 4):
            continue
        # pad and clip
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    # sort left→right
    boxes.sort(key=lambda b: b[0])
    return boxes


def run_word_segmentation_from_binarization(bin_csv: Path, out_root: Path):
    """
    Re-segment lines from the selected mask (to get coords),
    then segment words inside each line. Saves:
      - overlay_words.png (words on page view)
      - line_XXX/word_YYY.png crops (views for inspection)
    Logs per-page stats to out_root/words_log.csv
    """
    df = pd.read_csv(bin_csv, dtype={"page_out": "string"}).dropna(subset=["page_out"]).copy()
    logs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Words"):
        page_dir = Path(str(row["page_out"]))  # .../B_binarization/<lang>/<page>
        mask = load_selected_mask(page_dir)
        if mask is None:
            continue

        # view for overlays/crops
        view_path = page_dir / "bw_selected_view.png"
        view = cv2.imread(str(view_path), cv2.IMREAD_GRAYSCALE) if view_path.exists() else (255 - mask)

        # segment lines (re-use current logic)
        line_boxes = segment_lines(mask)

        # prepare output in D_words
        lang = page_dir.parent.name
        page = page_dir.name
        page_out = out_root / lang / page
        page_out.mkdir(parents=True, exist_ok=True)

        # overlay canvas
        if view.ndim == 2:
            overlay = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        else:
            overlay = view.copy()

        total_words = 0
        avg_word_w_all = []

        for li, (x, y, w, h) in enumerate(line_boxes):
            line_mask = mask[y:y+h, x:x+w]
            words = _segment_words_in_line(line_mask)

            # save words as crops (use view for readability)
            line_dir = page_out / f"line_{li:03d}"
            line_dir.mkdir(parents=True, exist_ok=True)

            for wi, (wx, wy, ww, wh) in enumerate(words):
                crop = view[y+wy:y+wy+wh, x+wx:x+wx+ww]
                cv2.imwrite(str(line_dir / f"word_{wi:03d}.png"), crop)
                # draw overlay
                cv2.rectangle(overlay, (x+wx, y+wy), (x+wx+ww, y+wy+wh), (0, 165, 255), 2)

            total_words += len(words)
            if words:
                avg_word_w_all.extend([ww for (_, _, ww, _) in words])

        # save overlay
        cv2.imwrite(str(page_out / "overlay_words.png"), overlay)

        logs.append({
            "language": lang,
            "page_out": str(page_out),
            "n_lines": len(line_boxes),
            "n_words": int(total_words),
            "avg_word_width": float(np.mean(avg_word_w_all)) if avg_word_w_all else 0.0,
            "words_per_line_avg": float(total_words / max(1, len(line_boxes))),
        })

    # write log
    out_root.mkdir(parents=True, exist_ok=True)
    log_csv = out_root / "words_log.csv"
    pd.DataFrame(logs).to_csv(log_csv, index=False)
    print(f"Saved word artifacts to: {out_root}\nLog -> {log_csv}")
    
    
def _estimate_xheight(mask: np.ndarray) -> int:
    """Rough x-height ~ median CC height (robust to outliers)."""
    m = ((mask > 0).astype(np.uint8) * 255)
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return 16
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    h = h[(h > 5) & (h < np.percentile(h, 95))]
    if h.size == 0:
        return 16
    return int(np.clip(np.median(h), 12, 48))  # clamp to a sane range


def segment_lines_morph(mask: np.ndarray,
                        bbox: tuple[int,int,int,int] | None = None) -> List[Tuple[int,int,int,int]]:
    """
    Robust line detection using morphology; handles curved baselines/gutter.
    Returns page-coords (x,y,w,h).
    """
    H, W = mask.shape[:2]
    if bbox is not None:
        bx, by, bw, bh = bbox
        sub = mask[by:by+bh, bx:bx+bw]
    else:
        bx = by = 0
        sub = mask

    if sub.dtype != np.uint8:
        sub = ((sub > 0).astype(np.uint8) * 255)

    # Estimate text scale and build kernels that scale with it
    xh = _estimate_xheight(sub)
    kx = max(10, 3 * xh)                           # connect letters into line bands
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    band = cv2.dilate(sub, kernel_h, iterations=1)

    # Suppress tall blobs (figures/gutter streaks)
    ky = max(1, xh // 2)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel_v, iterations=1)

    num, _, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    boxes: List[Tuple[int,int,int,int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if h < max(8, xh // 2) or w < 32:
            continue
        boxes.append((x + bx, y + by, w, h))

    boxes.sort(key=lambda b: b[1])  # top→bottom
    return boxes


def _parse_bbox_tuple(val) -> tuple[int,int,int,int] | None:
    """Parse a bbox from various formats safely."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    # Already a tuple/list?
    if isinstance(val, (tuple, list)) and len(val) == 4:
        x, y, w, h = val
        return int(x), int(y), int(w), int(h)
    # String like "(0, 0, 2048, 1536)" or "[...]"
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            t = ast.literal_eval(s)
            if isinstance(t, (tuple, list)) and len(t) == 4:
                x, y, w, h = t
                return int(x), int(y), int(w), int(h)
        except Exception:
            pass
    return None

def _load_ingest_bboxes(ingest_csv: Path) -> dict[str, tuple[int,int,int,int] | None]:
    """
    Map <page_stem> -> (x,y,w,h) from Step-0 ingest_log.csv.
    Accepts:
      - columns bbox_x, bbox_y, bbox_w, bbox_h (numeric or stringified tuples),
      - or a single 'bbox' column like '(x, y, w, h)'.
    Returns None when bbox is missing/unparseable.
    """
    bmap: dict[str, tuple[int,int,int,int] | None] = {}
    df = pd.read_csv(ingest_csv, dtype={"out_dir": "string"}, keep_default_na=True)

    # Decide which format we have
    has_separate = all(c in df.columns for c in ["bbox_x","bbox_y","bbox_w","bbox_h"])
    has_single = "bbox" in df.columns

    for _, r in df.iterrows():
        stem = Path(str(r["out_dir"])).name if pd.notna(r.get("out_dir")) else None
        if not stem:
            continue

        bbox: tuple[int,int,int,int] | None = None

        if has_separate:
            bx, by, bw, bh = r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]
            # Case 1: already numeric
            if all(pd.notna(v) and isinstance(v, (int, float, np.integer, np.floating)) for v in [bx, by, bw, bh]):
                bbox = (int(bx), int(by), int(bw), int(bh))
            else:
                # Case 2: some fields are strings like "(x, y, w, h)"
                # Try parsing any field as a tuple; if that fails, None
                parsed = None
                for v in [bx, by, bw, bh]:
                    parsed = _parse_bbox_tuple(v)
                    if parsed is not None:
                        break
                bbox = parsed
        elif has_single:
            bbox = _parse_bbox_tuple(r["bbox"])
        else:
            bbox = None

        bmap[stem] = bbox
    return bmap





