"""Tracking layer — faithful port of pratap424/visdrone_mot.

  * CameraMotionCompensator  — ORB+affine+EMAT severity (camera_motion.py)
  * DroneByteTracker         — constant-velocity + greedy ByteTrack 2-stage (drone_tracker.py)
"""
from .camera_motion import CameraMotionCompensator
from .drone_tracker import DroneByteTracker

__all__ = ["CameraMotionCompensator", "DroneByteTracker"]
