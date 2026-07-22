"""Image-directory source (e.g. a VisDrone sequence folder)."""
import os
import time
from typing import Iterator, List, Optional, Tuple

import cv2

from ..contracts import FrameMeta
from .base import FrameSource

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class ImageDirSource(FrameSource):
    def __init__(self, path: str, fps_hint: float = 30.0, loop: bool = False,
                 max_frames: int = 0):
        self.path = path
        self.loop = loop
        self.max_frames = max_frames
        self.fps = fps_hint
        self.files: List[str] = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(_IMG_EXTS))
        if not self.files:
            raise RuntimeError(f"No images in {path}")
        self.shape: Optional[Tuple[int, int]] = None
        self.total_frames = min(len(self.files), max_frames) if max_frames else len(self.files)

    def __iter__(self) -> Iterator[Tuple[FrameMeta, object]]:
        idx = 0
        while True:
            for f in self.files:
                frame = cv2.imread(f)
                if frame is None:
                    continue
                if self.shape is None:
                    self.shape = (frame.shape[0], frame.shape[1])
                yield FrameMeta(idx=idx, ts=time.monotonic(),
                                shape_hw=self.shape), frame
                idx += 1
                if self.max_frames and idx >= self.max_frames:
                    return
            if not self.loop:
                break

    def release(self):
        pass
