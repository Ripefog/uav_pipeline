"""SOT (single object tracking) — MCITrack, bật/tắt bằng sot.enabled.

Loại trừ nhau với MOT (track/). Xem
docs/superpowers/specs/2026-08-12-sot-mcitrack-integration-design.md
"""
from .class_groups import accepted_ids
from .guard import GuardVerdict, LostGuard, iou_xywh
from .tracker import SotTracker

__all__ = ["SotTracker", "LostGuard", "GuardVerdict", "iou_xywh", "accepted_ids"]
