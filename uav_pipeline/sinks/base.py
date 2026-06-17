"""Sink interface. A sink consumes one FrameContext per frame."""
from abc import ABC, abstractmethod

from ..contracts import FrameContext


class Sink(ABC):
    @abstractmethod
    def write(self, ctx: FrameContext):
        ...

    def close(self):
        pass
