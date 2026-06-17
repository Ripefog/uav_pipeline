"""Detection layer — four backends (torch/onnx/openvino/trt). All backends share
one letterbox preprocess (``preprocess.letterbox``) and one NMS postprocess
(the vendored ``utils.util.non_max_suppression[_v26]`` under ``_vendor/``)."""
from .base import DetectorBackend, build_backend
from .wrapper import UnifiedDetector

__all__ = ["DetectorBackend", "build_backend", "UnifiedDetector"]
