from pathlib import Path
import re
import numpy as np
import cv2

def natural_key(p: Path):
    """Key for natural (human-like) sorting of filenames."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]

def clip_small_angle(a_deg: float, limit: float = 15.0) -> float:
    """Map any angle to nearest equivalent within [-limit, +limit] degrees."""
    a = ((a_deg + 90) % 180) - 90  # [-90, 90)
    if a >  limit: a -= 180
    if a < -limit: a += 180
    return float(np.clip(a, -limit, limit))

def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around center with border replication."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
