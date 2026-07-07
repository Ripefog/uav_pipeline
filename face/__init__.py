"""Face layer — detection (SCRFD) + recognition (MobileFaceNet + turbovec gallery).

Not yet wired into ``uav_pipeline.Pipeline`` / ``config.py`` / ``contracts.py``.
"""
from .detector import FaceDetector
from .recognizer import FaceDB, FaceRecognizer, FaceVoter

__all__ = ["FaceDetector", "FaceRecognizer", "FaceDB", "FaceVoter"]
