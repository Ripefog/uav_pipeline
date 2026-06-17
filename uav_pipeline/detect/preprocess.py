"""Shared letterbox preprocessing — thin shim over ``eval_yolo.utils.dataset``.

``eval_yolo`` has no ``__init__.py``; the package init already put its
directory on ``sys.path``, so ``from utils.dataset import resize`` resolves to
``eval_yolo/utils/dataset.py`` — the exact same ``resize`` the original infer
scripts use (letterbox to a square, scale-down only when ``augment=False``,
``BORDER_CONSTANT`` pad, returns ``(image, (r, r), (pad_w, pad_h))``).
"""
from typing import Tuple

import numpy as np

from utils.dataset import resize  # noqa: E402  (path set by uav_pipeline._paths)


def letterbox(img0: np.ndarray, imgsz: int) -> Tuple[np.ndarray, tuple, tuple]:
    """Letterbox ``img0`` (BGR uint8) to a square ``imgsz``.

    Returns the letterboxed image plus ``(ratio=(r,r), pad=(pad_w,pad_h))`` so
    detections in letterbox coords can be mapped back to original-image coords.
    """
    img, ratio, pad = resize(img0, imgsz, augment=False)
    return img, ratio, pad


def to_chw_rgb_float(img_lb: np.ndarray, half: bool = False) -> np.ndarray:
    """Letterboxed BGR HxWx3 uint8 -> CHW RGB float tensor (not yet batched)."""
    sample = np.ascontiguousarray(img_lb.transpose((2, 0, 1))[::-1])
    sample = sample.astype(np.float16 if half else np.float32) / 255.0
    return sample
