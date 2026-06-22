"""UnifiedDetector — orchestrates preprocess -> backend -> shared postprocess.

Owns the primary multi-class detector and, optionally, a second backend for
license-plate detection (used by the OCR stage in ``plate_detection`` mode).
"""
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import yaml
except ImportError as _e:  # pragma: no cover
    raise ImportError("PyYAML required: pip install pyyaml") from _e

from .._paths import resolve
from ..config import DetectorCfg, PlateDetectorCfg
from ..contracts import Detection
from .base import build_backend
from .postprocess import postprocess, postprocess_batch, to_detections


def load_names(names_yaml: str) -> Dict[int, str]:
    """Load the ``names:`` map from a YOLO args.yaml (or a bare names dict)."""
    if not names_yaml:
        names_yaml = "weights/names.yaml"
    with open(resolve(names_yaml), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("names", data) if isinstance(data, dict) else data
    return {int(k): str(v) for k, v in raw.items()}


class UnifiedDetector:
    """Primary (multi-class) detector + optional plate detector."""

    def __init__(self, cfg: DetectorCfg):
        self.cfg = cfg
        p = cfg.primary
        raw = {"torch": p.pt, "onnx": p.onnx,
               "openvino": p.openvino, "trt": p.trt}.get(cfg.backend, "")
        if not raw:
            raise ValueError(
                f"No model path configured for backend '{cfg.backend}'. "
                f"Set detector.primary.{cfg.backend} in the config YAML.")
        model_path = resolve(raw)
        self.names = load_names(cfg.primary.names_yaml)
        self.coi = tuple(cfg.classes_of_interest)
        self.backend = build_backend(
            cfg.backend, model_path, imgsz=cfg.imgsz, device=cfg.device, fp16=cfg.fp16)
        self._plate_backend = None

    # -- primary detection --------------------------------------------------
    def detect(self, img0: np.ndarray, conf: Optional[float] = None,
               iou: Optional[float] = None) -> List[Detection]:
        conf = self.cfg.conf if conf is None else conf
        iou = self.cfg.iou if iou is None else iou
        inp, ratio, pad = self.backend._preprocess(img0)
        raw = self.backend._infer(inp)
        dets = postprocess(raw, conf, iou)
        return to_detections(dets, ratio, pad, self.names, img0.shape[:2], self.coi)

    def detect_batch(self, frames: List[np.ndarray], conf: Optional[float] = None,
                     iou: Optional[float] = None) -> List[List[Detection]]:
        """Detect on N frames with a single batched inference.

        Preprocess each frame, stack to ``[N,3,H,W]``, run the backend once,
        then map results back per frame. Returns one Detection list per frame.
        """
        conf = self.cfg.conf if conf is None else conf
        iou = self.cfg.iou if iou is None else iou
        inps, metas = [], []
        for f in frames:
            inp, ratio, pad = self.backend._preprocess(f)
            inps.append(inp)
            metas.append((ratio, pad, f.shape[:2]))
        batch = np.concatenate(inps, axis=0)
        raw = self.backend._infer(batch)
        per = postprocess_batch(raw, conf, iou)
        out: List[List[Detection]] = []
        for dets, (ratio, pad, hw) in zip(per, metas):
            out.append(to_detections(dets, ratio, pad, self.names, hw, self.coi))
        return out

    # -- optional plate detection ------------------------------------------
    def enable_plate(self, pcfg: PlateDetectorCfg):
        raw = {"onnx": pcfg.onnx, "openvino": pcfg.openvino,
               "pt": pcfg.pt, "trt": pcfg.trt}.get(pcfg.backend, "")
        if not raw:
            raise ValueError(f"No plate model path for backend '{pcfg.backend}'")
        self._plate_backend = build_backend(
            pcfg.backend, resolve(raw), imgsz=self.cfg.imgsz, device=self.cfg.device,
            fp16=self.cfg.fp16)
        self._plate_names = {0: "plate"}

    @property
    def plate_enabled(self) -> bool:
        return self._plate_backend is not None

    def detect_plates(self, img0: np.ndarray) -> List[Detection]:
        if self._plate_backend is None:
            return []
        inp, ratio, pad = self._plate_backend._preprocess(img0)
        raw = self._plate_backend._infer(inp)
        dets = postprocess(raw, self.cfg.conf, self.cfg.iou)
        return to_detections(dets, ratio, pad, self._plate_names, img0.shape[:2])
