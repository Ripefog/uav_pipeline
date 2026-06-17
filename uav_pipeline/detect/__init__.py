"""Detection layer — reuses eval_yolo backends (torch/onnx/openvino) verbatim
and adds a TensorRT backend for Jetson. All backends share one letterbox
preprocess (``eval_yolo.utils.dataset.resize``) and one NMS postprocess
(``eval_yolo.utils.util.non_max_suppression[_v26]``)."""
from .base import DetectorBackend, build_backend
from .wrapper import UnifiedDetector

__all__ = ["DetectorBackend", "build_backend", "UnifiedDetector"]
