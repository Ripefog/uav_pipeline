"""FrameSource interface. Implementations yield one BGR frame at a time."""
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

import numpy as np

from ..contracts import FrameMeta


class FrameSource(ABC):
    """Single-frame streaming source (contrast with the load-all-to-RAM infer scripts)."""

    fps: float = 30.0
    shape: Optional[Tuple[int, int]] = None   # (H, W) once known

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[FrameMeta, np.ndarray]]:
        ...

    def release(self):
        pass

    def reset(self):
        pass
