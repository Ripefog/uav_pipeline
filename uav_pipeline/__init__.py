"""UAV Edge Pipeline — FPT UAV AI Hackathon 2026.

A single-frame streaming pipeline for onboard UAV inference:

    FrameSource -> Detector -> Tracker -> TargetSelector -> FollowController -> Command
                     | (OCR, optional)                                   |
                     v                                                   v
                Track.plate_text                                   Sinks (HUD/telemetry/log)

Three competition pillars map onto the modules:
    * Phát hiện  (Detect)  -> ``detect/``   (reuses ``eval_yolo`` backends)
    * Theo dấu   (Track)   -> ``track/``    (faithful port of pratap424/visdrone_mot)
    * Bám đuổi   (Follow)  -> ``follow/``   (PID gimbal/body follow + mock controller)

Importing the package bootstraps the reuse paths (see ``_paths``).
"""
from . import _paths  # noqa: F401  (side-effect: puts eval_yolo on sys.path)

__version__ = "0.1.0"
__all__ = ["__version__"]
