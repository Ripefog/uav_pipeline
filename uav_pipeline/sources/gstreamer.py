"""GStreamer source (Jetson CSI / RTSP / nvargus). Thin wrapper around
``VideoSource`` with ``gst=True``; the pipeline string comes from config."""
from typing import Optional

from ..config import SourceCfg
from .video import VideoSource


class GStreamerSource(VideoSource):
    """Open a GStreamer pipeline string via OpenCV's GStreamer backend.

    Example pipeline (Jetson CSI cam):
      'nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,BGR ! appsink'
    Example RTSP:
      'rtspsrc location=rtsp://... latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,BGR ! appsink'
    """

    def __init__(self, pipeline: str, fps_hint: float = 30.0, max_frames: int = 0):
        super().__init__(path=pipeline, loop=False, fps_hint=fps_hint,
                         max_frames=max_frames, gst=True)


def build_gstreamer_from_cfg(cfg: SourceCfg) -> "GStreamerSource":
    if not cfg.gstreamer:
        raise ValueError("source.gstreamer pipeline string is empty")
    return GStreamerSource(cfg.gstreamer, fps_hint=cfg.fps, max_frames=cfg.max_frames)
