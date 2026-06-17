"""Live webcam / MIPI camera source."""
import time
from typing import Iterator, Optional, Tuple

import cv2

from ..contracts import FrameMeta
from .base import FrameSource


class WebcamSource(FrameSource):
    def __init__(self, index: int = 0, fps_hint: float = 30.0, max_frames: int = 0):
        self.index = index
        self.max_frames = max_frames
        self.fps = fps_hint
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {index}")
        self.shape: Optional[Tuple[int, int]] = None

    def __iter__(self) -> Iterator[Tuple[FrameMeta, object]]:
        idx = 0
        while True:
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
