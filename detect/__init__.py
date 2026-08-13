"""Detection layer.

YOLO backends (torch/onnx/openvino/trt) share letterbox + NMS. The optional
D-FINE-cpp adapter delegates preprocess/inference/decode to its native runtime
and returns the same pipeline Detection contract.
"""
from .base import DetectorBackend, build_backend
from .wrapper import UnifiedDetector

__all__ = ["DetectorBackend", "build_backend", "UnifiedDetector"]
