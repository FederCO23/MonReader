from pathlib import Path
from typing import Dict
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def binarize_variants(gray: np.ndarray) -> Dict[str, np.ndarray]:
    """Return multiple binarized candidates as 0/255 images with text=255 (white)."""
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.medianBlur(gray, 3)

    _, bw_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_ad_mean = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    bw_ad_gauss = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)

    return {
        "otsu": 255 - bw_otsu,
        "adaptive_mean": 255 - bw_ad_mean,
        "adaptive_gaussian": 255 - bw_ad_gauss,
    }

def _connected_components_stats(fg: np.ndarray):
    """Count and area stats on a foreground (255) mask."""
    mask255 = ((fg > 0).astype(np.uint8) * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask255, connectivity=8)
    if num_labels <= 1:
        return 0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    return int(len(areas)), float(np.median(areas)), float(np.percentile(areas, 95))

def bin_metrics(fg_white: np.ndarray) -> Dict[str, float]:
    """Quick metrics for selection and QA."""
    H, W = fg_white.shape[:2]
    fg_ratio = float((fg_white > 0).sum() / (H * W))
    proj = fg_white.sum(axis=1).astype(np.float32)
    proj_var = float(proj.var())
    cc_count, cc_med, cc_p95 = _connected_components_stats(fg_white)
    return {
        "fg_ratio": fg_ratio,
        "proj_var": proj_var,
        "cc_count": cc_count,
        "cc_median_area": cc_med,
        "cc_p95_area": cc_p95,
    }

def choose_best(cands: Dict[str, np.ndarray]) -> str:
    """Prefer highest projection variance within a plausible foreground band."""
    metrics = {k: bin_metrics(v) for k, v in cands.items()}
    band = {k: m for k, m in metrics.items() if 0.03 <= m["fg_ratio"] <= 0.25}
    pool = band if band else metrics
    best = max(pool.items(), key=lambda kv: kv[1]["proj_var"])[0]
    return best

def run_binarization_from_ingest(ingest_csv: Path, out_root: Path):
    df = pd.read_csv(ingest_csv)
    logs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Binarization"):
        lang = row["language"]
        out_dir = Path(row["out_dir"])
        page_name = out_dir.name
        deskew_path = out_dir / "page_deskewed.png"
        if not deskew_path.exists():
            continue

        bgr = cv2.imread(str(deskew_path), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        cands = binarize_variants(gray)
        best_name = choose_best(cands)

        page_out = out_root / lang / page_name
        page_out.mkdir(parents=True, exist_ok=True)

        for name, fg_white in cands.items():
            vis = 255 - fg_white
            cv2.imwrite(str(page_out / f"bw_{name}.png"), vis)

        best_fg = cands[best_name]
        cv2.imwrite(str(page_out / f"bw_selected_mask.png"), best_fg)
        cv2.imwrite(str(page_out / f"bw_selected_view.png"), 255 - best_fg)

        row_log = {"language": lang, "page_out": str(page_out), "selected": best_name}
        for name, fg_white in cands.items():
            m = bin_metrics(fg_white)
            for k, v in m.items():
                row_log[f"{name}.{k}"] = v
        logs.append(row_log)

    log_csv = out_root / "binarization_log.csv"
    pd.DataFrame(logs).to_csv(log_csv, index=False)
    print(f"Saved binarization artifacts to: {out_root}\nLog -> {log_csv}")
