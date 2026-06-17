"""License-plate OCR — thin lazy wrapper over ``fast-plate-ocr`` (Keras CCT).

``decode_plate`` is a 4-line pure-numpy copy of the original plate-recognition
decode (kept inline so importing this module does NOT pull in
``fast_plate_ocr``/TensorFlow unless OCR is actually enabled). TF is forced to
CPU (TF 2.x has no GPU support on Jetson/Blackwell). The Keras model + plate
config ship in ``weights/`` (``plate_ocr.keras`` + ``plate_config.yaml``).
"""
import os
from collections import Counter, defaultdict, deque
from typing import Optional

import cv2
import numpy as np

try:
    import yaml
except ImportError as _e:  # pragma: no cover
    raise ImportError("PyYAML required: pip install pyyaml") from _e


def decode_plate(logits: np.ndarray, alphabet: str, pad_char: str) -> str:
    """logits (slots, vocab) -> decoded string with pad chars dropped."""
    idx = np.argmax(logits, axis=-1)
    chars = [alphabet[i] for i in idx]
    return "".join(c for c in chars if c != pad_char)


class PlateOCR:
    """Lazy Keras plate recognizer."""

    def __init__(self, keras_model: str, plate_config: str):
        # Force TF CPU + silence logs BEFORE importing fast_plate_ocr/Keras.
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

        from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
        from fast_plate_ocr.train.utilities.utils import load_keras_model

        with open(plate_config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self.alphabet = raw["alphabet"]
        self.pad_char = raw["pad_char"]
        self.H = int(raw["img_height"])
        self.W = int(raw["img_width"])

        cfg = load_plate_config_from_yaml(plate_config)
        self.model = load_keras_model(keras_model, cfg)

    def recognize(self, crop_bgr: Optional[np.ndarray]) -> Optional[str]:
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return None
        resized = cv2.resize(crop_bgr, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb[None, ...].astype(np.uint8)            # (1, H, W, 3) uint8
        out = self.model(x, training=False)
        if isinstance(out, dict):
            logits = np.asarray(out["plate"])[0]
        else:
            logits = np.asarray(out)[0]
        return decode_plate(logits, self.alphabet, self.pad_char)


class PlateVoter:
    """Per-track majority vote over a sliding window of plate readings."""

    def __init__(self, window: int = 5):
        self.window = max(1, window)
        self._buf = defaultdict(lambda: deque(maxlen=self.window))

    def add(self, track_id: int, text: Optional[str]):
        if text:
            self._buf[track_id].append(text)

    def majority(self, track_id: int) -> Optional[str]:
        d = self._buf.get(track_id)
        if not d:
            return None
        return Counter(d).most_common(1)[0][0]

    def drop(self, track_id: int):
        self._buf.pop(track_id, None)


def lower_third_crop(img: np.ndarray, bbox_xyxy) -> Optional[np.ndarray]:
    """Crop the bottom third of a vehicle bbox (where plates usually sit)."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    h = y2 - y1
    if h < 8 or (x2 - x1) < 8:
        return None
    y_start = y1 + int(h * 0.62)
    y_end = y2
    crop = img[max(0, y_start):y_end, max(0, x1):x2]
    return crop if crop.size else None
