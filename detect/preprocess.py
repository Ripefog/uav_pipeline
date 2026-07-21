"""Shared letterbox preprocessing (self-contained, cv2-only).

``letterbox`` mirrors the ``resize()`` helper from the original YOLO infer
scripts verbatim: letterbox to a square, scale-down only (``augment=False``),
``BORDER_CONSTANT`` pad, returns ``(image, (r, r), (pad_w, pad_h))``. It now
lives here directly so the package has no dependency on the sibling
``eval_yolo`` folder.
"""
from typing import Tuple, Union

import cv2
import numpy as np


def letterbox(img0: np.ndarray, imgsz: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, tuple, tuple]:
    """Letterbox ``img0`` (BGR uint8) to ``imgsz``.

    ``imgsz`` is either a single int (square target) or an ``(h, w)`` pair
    (e.g. a fixed non-square ONNX export like 736x1280).

    Returns the letterboxed image plus ``(ratio=(r,r), pad=(pad_w,pad_h))`` so
    detections in letterbox coords can be mapped back to original-image coords.
    """
    target_h, target_w = imgsz if isinstance(imgsz, (tuple, list)) else (imgsz, imgsz)
    shape = img0.shape[:2]  # [h, w]
    r = min(target_h / shape[0], target_w / shape[1], 1.0)  # scale down only

    new_w, new_h = int(round(shape[1] * r)), int(round(shape[0] * r))
    w_pad = (target_w - new_w) / 2.0
    h_pad = (target_h - new_h) / 2.0

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


def letterbox_yolox(img0: np.ndarray, imgsz: Tuple[int, int]) -> Tuple[np.ndarray, tuple, tuple]:
    """Letterbox to ``imgsz`` matching YOLOX's own ``preproc()`` exactly
    (``yolox/data/data_augment.py``): resized image placed top-left (no
    centering), 114 pad fill, and — unlike the Ultralytics-style ``letterbox``
    above — the scale is *not* capped at 1.0 (YOLOX upscales small images).

    Returns ``pad=(0, 0)`` since there's no centering offset to invert.
    """
    target_h, target_w = imgsz
    shape = img0.shape[:2]  # [h, w]
    r = min(target_h / shape[0], target_w / shape[1])

    new_w, new_h = int(shape[1] * r), int(shape[0] * r)
    resized = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, (r, r), (0.0, 0.0)


def to_chw_bgr_float(img_lb: np.ndarray, half: bool = False) -> np.ndarray:
    """Letterboxed BGR HxWx3 uint8 -> CHW BGR float tensor, unnormalized
    (0-255 range) — matches YOLOX's ``preproc()`` (no RGB swap, no /255)."""
    sample = np.ascontiguousarray(img_lb.transpose((2, 0, 1)))
    return sample.astype(np.float16 if half else np.float32)
