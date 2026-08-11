"""Sinks — HUD video, telemetry JSONL/CSV, control command log, SOT result txt."""
from .base import Sink
from .control_log import ControlLogSink
from .sot_result import SotResultSink
from .telemetry import TelemetrySink
from .video import HUDAnnotatedSink

__all__ = ["Sink", "HUDAnnotatedSink", "TelemetrySink", "ControlLogSink",
           "SotResultSink"]
