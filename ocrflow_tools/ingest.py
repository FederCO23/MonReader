from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from numpy.polynomial.polyutils import RankWarning

from PIL import Image
import csv
import pandas as pd
from tqdm import tqdm

import warnings
from typing import Optional, Tuple, List

from .common import natural_key, clip_small_angle, rotate_image


def collect_pages(img_dir: Path) -> List[Path]:
    """Collect page images and return a naturally sorted, de-duplicated list of paths."""
    imgs = []
    for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp","*.webp"):
        imgs.extend(img_dir.glob(ext))
    return sorted(set(imgs), key=natural_key)

def estimate_skew_angle_projection(gray: np.ndarray) -> float:
    """Fallback: small-angle sweep maximizing horizontal projection variance."""
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    scale = 1000 / max(gray.shape[:2])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw

    best_angle, best_score = 0.0, -1.0
    for a in np.linspace(-10.0, 10.0, 41):  # step 0.5°
        h, w = inv.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), a, 1.0)
        rot = cv2.warpAffine(inv, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        proj = rot.sum(axis=1).astype(np.float32)
        score = proj.var()
        if score > best_score:
            best_score, best_angle = score, a
    return float(best_angle)

def _text_bands(gray: np.ndarray) -> np.ndarray:
    """Return thickened horizontal 'bands' where text lines live."""
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw  # text white
    h, w = gray.shape
    kx = max(15, w // 70)     # a bit wider than before
    ky = max(1,  h // 300)    # thinner than before (more sensitive)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    return cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)

def _robust_baselines(gray: np.ndarray, poly_order: int = 2) -> List[np.ndarray]:
    """Fit low-order polynomials to band contours with better conditioning."""
    bands = _text_bands(gray)
    cnts, _ = cv2.findContours((bands > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curves = []
    for c in cnts:
        if len(c) < 60: 
            continue
        pts = c.reshape(-1, 2)
        x = pts[:, 0].astype(np.float32)
        y = pts[:, 1].astype(np.float32)

        # uniform subsample along width
        sel = np.linspace(0, len(x) - 1, num=min(400, len(x))).astype(int)
        x = x[sel]; y = y[sel]

        # normalize x to [-1, 1] for a well-conditioned fit
        xmin, xmax = float(x.min()), float(x.max())
        if xmax <= xmin + 1e-6:
            continue
        x_n = 2.0 * (x - xmin) / (xmax - xmin) - 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RankWarning)
            try:
                coeff = np.polyfit(x_n, y, deg=poly_order)  # y = c2*x^2 + c1*x + c0
            except Exception:
                continue

        # store coeff with (xmin,xmax) so we can evaluate later
        curves.append((coeff, xmin, xmax))
    return curves

def _median_curvature(gray: np.ndarray) -> Tuple[float, int]:
    """Return (median absolute baseline deviation in px, number of lines). Lower is straighter."""
    curves = _robust_baselines(gray, poly_order=1)  # straight line fit to measure deviation
    if not curves:
        return 0.0, 0
    h, w = gray.shape
    devs = []
    for (coeff, xmin, xmax) in curves:
        # evaluate deviation from best straight line over support
        x = np.linspace(xmin, xmax, 200, dtype=np.float32)
        x_n = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
        a1, a0 = coeff[0], coeff[1]  # for deg=1 -> y = a1*x + a0 in normalized x
        y_hat = a1 * x_n + a0
        # straightness metric = MAD of residuals against mean y (cheap proxy)
        devs.append(float(np.median(np.abs(y_hat - np.median(y_hat)))))
    return (float(np.median(devs)), len(devs))

def _clahe_unsharp(gray: np.ndarray) -> np.ndarray:
    """Slightly stronger local contrast & unsharp for OCR clarity."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    g = clahe.apply(gray)
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=0.8)
    sharp = cv2.addWeighted(g, 1.7, blur, -0.7, 0)  # stronger unsharp
    return sharp

def dewarp_by_line_baselines(bgr: np.ndarray,
                             poly_order: int = 2,
                             curvature_threshold_px: float = 1.8,
                             return_if_not_needed: bool = True) -> np.ndarray:
    """
    Dewarp only if median baseline deviation exceeds curvature_threshold_px.
    After remap, apply mild CLAHE + unsharp to preserve edges for binarization.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # gate by curvature
    dev, n_lines = _median_curvature(gray)
    if n_lines < 5 or dev < curvature_threshold_px:
        return bgr if return_if_not_needed else bgr.copy()

    # fit robust curves (quadratic) and compute column-wise median baseline
    curves = _robust_baselines(gray, poly_order=poly_order)
    if not curves:
        return bgr

    xgrid = np.arange(w, dtype=np.float32)
    y_list = []
    for (coeff, xmin, xmax) in curves:
        # evaluate only over the curve's support; extrapolate flatly outside
        x = xgrid.copy()
        x = np.clip(x, xmin, xmax)
        x_n = 2.0 * (x - xmin) / max(1e-6, (xmax - xmin)) - 1.0
        y = np.polyval(coeff, x_n)
        y_list.append(y.astype(np.float32))
    y_mean = np.median(np.stack(y_list, axis=0), axis=0)

    # build vertical displacement and remap
    target = float(np.median(y_mean))
    disp = (target - y_mean).astype(np.float32)  # per-column shift
    xmap = np.tile(xgrid, (h, 1)).astype(np.float32)
    ybase = np.arange(h, dtype=np.float32).reshape(-1, 1)
    ymap = np.clip(ybase + disp.reshape(1, -1), 0, h - 1).astype(np.float32)

    dewarped = cv2.remap(bgr, xmap, ymap, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # anti-soften for downstream thresholding
    g = cv2.cvtColor(dewarped, cv2.COLOR_BGR2GRAY)
    g = _clahe_unsharp(g)
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return out

def estimate_skew_angle_hough(gray: np.ndarray) -> float:
    """Estimate skew using Hough lines on text baselines; falls back to projection."""
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape[:2]
    bx = int(w * 0.05); by = int(h * 0.05)
    roi = gray[by:h-by, bx:w-bx]

    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw

    k = max(1, h // 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k*3+1, 1))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    edges = cv2.Canny(closed, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=max(80, int(min(h,w)*0.1)))

    if lines is None:
        return estimate_skew_angle_projection(gray)

    angles = []
    for l in lines[:200]:
        theta = l[0][1]
        baseline_deg = np.degrees(theta) - 90.0
        angles.append(clip_small_angle(baseline_deg, limit=15.0))

    if len(angles) == 0:
        return estimate_skew_angle_projection(gray)

    return float(np.median(angles))

def detect_main_text_bbox(gray: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """Detect a coarse bbox for the main text region on a deskewed grayscale page."""
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw

    k = max(1, gray.shape[0] // 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k*2+1, k*2+1))
    proc = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (int(x), int(y), int(w), int(h))

def save_overlay(img_bgr: np.ndarray, bbox: Optional[Tuple[int,int,int,int]]) -> np.ndarray:
    """Draw a green rectangle over the main text region (if available)."""
    vis = img_bgr.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return vis

def process_image_folder(lang: str, img_dir: Path, work_dir: Path, enable_dewarp: bool = False):
    """
    Phase A1 (ingestion & conditioning). Saves (per produced page):
      - page_original.png
      - page_deskewed.png    (deskewed OR dewarped+deskewed if enable_dewarp=True)
      - [optional] page_dewarped.png (extra QA artifact when enable_dewarp=True)
      - overlay_text_region.png
    Appends metadata to work/.../ingest_log.csv
    """
    out_lang_dir = work_dir / lang
    out_lang_dir.mkdir(parents=True, exist_ok=True)

    pages = collect_pages(img_dir)
    log_rows = []

    for page_path in tqdm(pages, desc=f"[{lang}] ingest (images)"):
        pil = Image.open(page_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        angle = estimate_skew_angle_hough(gray)
        deskewed_bgr = rotate_image(bgr, -angle)

        # optional dewarp
        used_bgr = deskewed_bgr
        if enable_dewarp:
            dewarped_bgr = dewarp_by_line_baselines(deskewed_bgr)
            used_bgr = dewarped_bgr

        gray_used = cv2.cvtColor(used_bgr, cv2.COLOR_BGR2GRAY)
        bbox = detect_main_text_bbox(gray_used)
        overlay = save_overlay(used_bgr, bbox)

        # create output dir for this page
        page_out = out_lang_dir / page_path.stem
        page_out.mkdir(parents=True, exist_ok=True)
        
        if enable_dewarp:
            # Quick QA: binarized preview of the dewarped image (does not affect Step B)
            bw = cv2.adaptiveThreshold(
                gray_used, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                41, 15
            )
            cv2.imwrite(str(page_out / "page_dewarped_bw_preview.png"), bw)
        
        cv2.imwrite(str(page_out / "page_original.png"), bgr)
        cv2.imwrite(str(page_out / "page_deskewed.png"), used_bgr)  # working image
        if enable_dewarp:
            cv2.imwrite(str(page_out / "page_dewarped.png"), used_bgr)  # QA duplicate
        cv2.imwrite(str(page_out / "overlay_text_region.png"), overlay)

        H, W = gray.shape[:2]
        bx = by = bw_ = bh = (None, None, None, None) if bbox is None else bbox

        log_rows.append({
            "language": lang,
            "page_path": str(page_path),
            "out_dir": str(page_out),
            "width": W, "height": H,
            "skew_angle_deg": float(angle),
            "bbox_x": bx, "bbox_y": by, "bbox_w": bw_, "bbox_h": bh,
            "enable_dewarp": bool(enable_dewarp),
        })

    log_csv = work_dir / "ingest_log.csv"
    headers = ["language","page_path","out_dir","width","height","skew_angle_deg",
               "bbox_x","bbox_y","bbox_w","bbox_h","enable_dewarp"]
    write_header = not log_csv.exists()
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if write_header: w.writeheader()
        w.writerows(log_rows)

    print(f"Done: {lang}. Pages: {len(pages)}. Log -> {log_csv}")



# --- add: helpers for double-page detection & split ---

def _detect_gutter_x(bgr: np.ndarray, min_ratio: float = 1.3) -> Optional[int]:
    """
    Return the x position of the page gutter (vertical split) if this looks like a 2-page scan,
    otherwise return None.
    Heuristic:
      - require wide aspect (w/h >= min_ratio)
      - Otsu binarize -> vertical ink projection -> smooth
      - look for a strong minimum in the 35%..65% band of image width
    """
    h, w = bgr.shape[:2]
    if w / float(h) < min_ratio:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw

    col_proj = inv.sum(axis=0).astype(np.float32)
    k = max(5, w // 200)
    col_proj_s = cv2.blur(col_proj.reshape(1, -1), (1, k)).ravel()

    lo = int(w * 0.35); hi = int(w * 0.65)
    if hi <= lo:
        return None
    band = col_proj_s[lo:hi]
    x_rel = int(np.argmin(band))
    x = lo + x_rel

    # require a strong valley (deep minimum) to avoid false splits
    if col_proj_s[x] > 0.4 * col_proj_s.max():
        return None
    return int(x)


def _iter_single_or_split_pages(bgr: np.ndarray):
    """
    Yield tuples: (suffix, sub_bgr) where suffix is '' for single page,
    or '_L'/'_R' for double-page splits.
    """
    gx = _detect_gutter_x(bgr)
    if gx is None:
        yield "", bgr
    else:
        yield "_L", bgr[:, :gx]
        yield "_R", bgr[:, gx:]

