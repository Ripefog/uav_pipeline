"""Frame sources + factory."""
from ..config import SourceCfg
from .base import FrameSource
from .gstreamer import GStreamerSource
from .image_dir import ImageDirSource
from .video import VideoSource
from .webcam import WebcamSource


def make_source(cfg: SourceCfg) -> FrameSource:
    kind = cfg.type.lower().strip()
    if kind == "video":
        if not cfg.path:
            raise ValueError("source.path required for video source")
        return VideoSource(cfg.path, loop=cfg.loop, fps_hint=cfg.fps,
                           max_frames=cfg.max_frames)
    if kind == "webcam":
        return WebcamSource(cfg.index, fps_hint=cfg.fps, max_frames=cfg.max_frames)
    if kind == "image_dir":
        if not cfg.path:
            raise ValueError("source.path required for image_dir source")
        return ImageDirSource(cfg.path, fps_hint=cfg.fps, loop=cfg.loop,
                              max_frames=cfg.max_frames)
    if kind == "gstreamer":
        return GStreamerSource(cfg.gstreamer, fps_hint=cfg.fps, max_frames=cfg.max_frames)
    raise ValueError(f"Unknown source type '{cfg.type}' "
                     f"(expected video | webcam | image_dir | gstreamer)")


__all__ = [
    "FrameSource", "VideoSource", "WebcamSource", "ImageDirSource",
    "GStreamerSource", "make_source",
]
