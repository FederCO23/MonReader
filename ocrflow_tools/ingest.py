from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import csv
import pandas as pd
from tqdm import tqdm

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

def process_image_folder(lang: str, img_dir: Path, work_dir: Path):
    """
    Phase A1 (ingestion & conditioning). Saves:
      - page_original.png
      - page_deskewed.png
      - overlay_text_region.png
    Appends metadata to work/A_ingest/ingest_log.csv
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

        gray_deskew = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2GRAY)
        bbox = detect_main_text_bbox(gray_deskew)
        overlay = save_overlay(deskewed_bgr, bbox)

        page_out = out_lang_dir / page_path.stem
        page_out.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(page_out / "page_original.png"), bgr)
        cv2.imwrite(str(page_out / "page_deskewed.png"), deskewed_bgr)
        cv2.imwrite(str(page_out / "overlay_text_region.png"), overlay)

        H, W = gray.shape[:2]
        bx = by = bw = bh = (None, None, None, None) if bbox is None else bbox

        log_rows.append({
            "language": lang,
            "page_path": str(page_path),
            "out_dir": str(page_out),
            "width": W,
            "height": H,
            "skew_angle_deg": float(angle),
            "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh
        })

    log_csv = work_dir / "ingest_log.csv"
    headers = ["language","page_path","out_dir","width","height","skew_angle_deg","bbox_x","bbox_y","bbox_w","bbox_h"]
    write_header = not log_csv.exists()
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if write_header: w.writeheader()
        w.writerows(log_rows)

    print(f"Done: {lang}. Pages: {len(pages)}. Log -> {log_csv}")
