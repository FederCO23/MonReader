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

def _border_foreground_ratio(fg_white: np.ndarray, margin_frac: float = 0.06) -> float:
    """
    Fraction of foreground (white) inside left+right vertical margins.
    Clean binarization => small value; noisy gutter/shading => large value.
    """
    H, W = fg_white.shape[:2]
    m = max(1, int(W * margin_frac))
    band = np.zeros_like(fg_white, dtype=bool)
    band[:, :m] = True
    band[:, -m:] = True
    fg = (fg_white > 0)
    if band.sum() == 0:
        return 0.0
    return float(fg[band].mean())

def choose_best(cands: dict[str, np.ndarray]) -> str:
    """
    Score = proj_var / (1 + a*margin_ink + b*cc_median_area_norm + c*fg_penalty)
    Higher score wins. Keeps a plausible fg_ratio band first.
    """
    # gather metrics
    metrics = {}
    for name, fg in cands.items():
        m = bin_metrics(fg)  # fg_ratio, proj_var, cc_count, cc_median_area, cc_p95_area
        m["margin_ink"] = _border_foreground_ratio(fg)  # new
        metrics[name] = m

    # keep plausible foreground band
    band = {k: v for k, v in metrics.items() if 0.03 <= v["fg_ratio"] <= 0.25}
    pool = band if band else metrics

    # normalize per-pool to keep scales comparable
    def _norm(key):
        vals = np.array([m[key] for m in pool.values()], dtype=float)
        vmin, vmax = float(vals.min()), float(vals.max())
        return {k: 0.0 if vmax == vmin else (metrics[k][key] - vmin) / (vmax - vmin) for k in pool.keys()}

    proj_var_n = _norm("proj_var")            # higher is better
    cc_med_n   = _norm("cc_median_area")      # lower is better
    margin_n   = _norm("margin_ink")          # lower is better
    fg_ratio_n = _norm("fg_ratio")            # prefer mid-range → use distance to 0.12 as penalty

    # score with penalties
    a, b, c = 3.0, 1.5, 1.0  # weights: margin, CC size, fg mid-range
    scores = {}
    for k in pool.keys():
        fg_penalty = abs((metrics[k]["fg_ratio"] - 0.12)) / 0.12  # 0 at 12% ink, grows away from it
        denom = (1.0 + a*margin_n[k] + b*cc_med_n[k] + c*fg_penalty)
        scores[k] = (1e-6 + proj_var_n[k]) / denom  # small epsilon for safety

    # pick best
    best = max(scores.items(), key=lambda kv: kv[1])[0]
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
