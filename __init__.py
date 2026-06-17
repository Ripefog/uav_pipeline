"""UAV Edge Pipeline — FPT UAV AI Hackathon 2026.

A single-frame streaming pipeline for onboard UAV inference:

    FrameSource -> Detector -> Tracker -> TargetSelector -> FollowController -> Command
                     | (OCR, optional)                                   |
                     v                                                   v
                Track.plate_text                                   Sinks (HUD/telemetry/log)

Three competition pillars map onto the modules:
    * Phát hiện  (Detect)  -> ``detect/``   (vendored YOLO backends: torch/onnx/openvino/trt)
    * Theo dấu   (Track)   -> ``track/``    (faithful port of pratap424/visdrone_mot)
    * Bám đuổi   (Follow)  -> ``follow/``   (PID gimbal/body follow + mock controller)

The package is **self-contained**: the YOLO NMS/letterbox/model-defs helpers it
needs are vendored under ``_vendor/`` and the model weights live in ``weights/``.
Importing the package bootstraps those paths (see ``_paths``).
"""
from . import _paths  # noqa: F401  (side-effect: puts _vendor on sys.path)

__version__ = "0.1.0"
__all__ = ["__version__"]
