"""Shared letterbox preprocessing (self-contained, cv2-only).

``letterbox`` mirrors the ``resize()`` helper from the original YOLO infer
scripts verbatim: letterbox to a square, scale-down only (``augment=False``),
``BORDER_CONSTANT`` pad, returns ``(image, (r, r), (pad_w, pad_h))``. It now
lives here directly so the package has no dependency on the sibling
``eval_yolo`` folder.
"""
from typing import Tuple

import cv2
import numpy as np


def letterbox(img0: np.ndarray, imgsz: int) -> Tuple[np.ndarray, tuple, tuple]:
    """Letterbox ``img0`` (BGR uint8) to a square ``imgsz``.

    Returns the letterboxed image plus ``(ratio=(r,r), pad=(pad_w,pad_h))`` so
    detections in letterbox coords can be mapped back to original-image coords.
    """
    shape = img0.shape[:2]  # [h, w]
    r = min(imgsz / shape[0], imgsz / shape[1], 1.0)  # scale down only

    new_w, new_h = int(round(shape[1] * r)), int(round(shape[0] * r))
    w_pad = (imgsz - new_w) / 2.0
    h_pad = (imgsz - new_h) / 2.0

    if shape[::-1] != (new_w, new_h):
        img0 = cv2.resize(img0, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
    left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))
    img0 = cv2.copyMakeBorder(img0, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return img0, (r, r), (w_pad, h_pad)


def to_chw_rgb_float(img_lb: np.ndarray, half: bool = False) -> np.ndarray:
    """Letterboxed BGR HxWx3 uint8 -> CHW RGB float tensor (not yet batched)."""
    sample = np.ascontiguousarray(img_lb.transpose((2, 0, 1))[::-1])
    sample = sample.astype(np.float16 if half else np.float32) / 255.0
    return sample
