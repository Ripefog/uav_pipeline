"""Video file / GStreamer pipeline source (loops by default)."""
import time
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from ..contracts import FrameMeta
from .base import FrameSource


class VideoSource(FrameSource):
    """Reads a video file frame-by-frame. Optionally loops forever."""

    def __init__(self, path: str, loop: bool = True, fps_hint: float = 30.0,
                 max_frames: int = 0, gst: bool = False):
        self.path = path
        self.loop = loop
        self.max_frames = max_frames
        api = cv2.CAP_GSTREAMER if gst else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(path, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else fps_hint
        self.shape: Optional[Tuple[int, int]] = None

        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 0/-1 for live streams (gstreamer/RTSP)
        self.total_frames = int(count) if count and count > 0 else None
        if max_frames:
            self.total_frames = min(self.total_frames, max_frames) if self.total_frames else max_frames

    def __iter__(self) -> Iterator[Tuple[FrameMeta, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if not self.loop:
                    break
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    break
            if self.shape is None:
                self.shape = (frame.shape[0], frame.shape[1])
            yield FrameMeta(idx=idx, ts=time.monotonic(), shape_hw=self.shape), frame
            idx += 1
            if self.max_frames and idx >= self.max_frames:
                break

    def release(self):
        if self.cap is not None:
            self.cap.release()
