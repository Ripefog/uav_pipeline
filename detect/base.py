"""Detector backend interface + factory.

Each backend implements two hooks; the ``UnifiedDetector`` orchestrates the
shared letterbox + NMS around them:

    _preprocess(img0) -> (model_input, ratio, pad)
    _infer(model_input) -> raw_model_output   # [1, ...]

Backend dtype/division differs (torch divides inside ``_infer`` to honor fp16;
ONNX/OpenVINO/TRT pre-divide in ``_preprocess``), but every backend returns a
raw output that ``postprocess`` can consume.
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple


class DetectorBackend(ABC):
    backend_name: str = "base"

    def __init__(self, model_path: str, imgsz: int = 640,
                 device: str = "", fp16: bool = False, preprocess: str = "ultralytics"):
        self.model_path = model_path
        self.imgsz = imgsz
        self.device = device
        self.fp16 = fp16
        self.preprocess = preprocess  # "ultralytics" | "yolox" -> which letterbox/normalize convention

    @abstractmethod
    def _preprocess(self, img0):
        """BGR uint8 HxWx3 -> (model_input, ratio, pad)."""

    @abstractmethod
    def _infer(self, model_input):
        """Run one forward pass; return the raw model output tensor/array."""


def build_backend(backend: str, model_path: str, imgsz: int = 640,
                  device: str = "", fp16: bool = False,
                  preprocess: str = "ultralytics") -> DetectorBackend:
    """Instantiate a backend by name. Imports are lazy so unused backends
    (e.g. TensorRT on a Windows dev box) don't drag in Jetson-only deps."""
    backend = backend.lower().strip()
    if backend == "openvino":
        from .backends.openvino_backend import OpenVINOBackend
        return OpenVINOBackend(model_path, imgsz, device, fp16, preprocess)
    if backend == "onnx":
        from .backends.onnx_backend import ONNXBackend
        return ONNXBackend(model_path, imgsz, device, fp16, preprocess)
    if backend == "torch":
        from .backends.torch_backend import TorchBackend
        return TorchBackend(model_path, imgsz, device, fp16, preprocess)
    if backend == "trt":
        from .backends.trt_backend import TensorRTBackend
        return TensorRTBackend(model_path, imgsz, device, fp16, preprocess)
    raise ValueError(
        f"Unknown detector backend '{backend}'. "
        f"Expected one of: openvino | onnx | torch | trt")
