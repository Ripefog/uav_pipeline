"""Typed configuration loaded from a single YAML file.

Replaces the scattered ``.env`` files of the original infer scripts with one
declarative file (see ``configs/*.yaml``). Dataclasses are built tolerantly:
unknown keys are ignored, missing keys fall back to defaults.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar

try:
    import yaml  # PyYAML
except ImportError as _e:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from _e

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# generic tolerant dataclass builder
# --------------------------------------------------------------------------- #
def _build(cls: Type[T], data: Optional[Dict[str, Any]]) -> T:
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    data = data or {}
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        ftype = f.type
        if is_dataclass(ftype) and isinstance(raw, dict):
            kwargs[f.name] = _build(ftype, raw)
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)  # type: ignore[arg-type]


def _resolve_types(ns: Dict[str, Any]):
    """Resolve string type annotations (from `from __future__ import`) to classes
    in module scope so is_dataclass(f.type) works under dataclass defaults."""
    pass


# --------------------------------------------------------------------------- #
# config schema
# --------------------------------------------------------------------------- #
@dataclass
class SourceCfg:
    type: str = "video"                       # video | webcam | image_dir | gstreamer
    path: str = ""
    index: int = 0                            # webcam index
    gstreamer: str = ""                       # gst pipeline string
    loop: bool = True                         # loop video file
    fps: float = 30.0                         # fallback fps if undetectable
    max_frames: int = 0                       # 0 = all


@dataclass
class PrimaryModelCfg:
    onnx: str = ""
    openvino: str = ""
    pt: str = ""
    trt: str = ""
    names_yaml: str = ""


@dataclass
class DetectorCfg:
    backend: str = "openvino"                 # torch | onnx | openvino | trt
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    fp16: bool = False
    device: str = ""                          # "" = auto; cuda:0 / CPU / GPU
    classes_of_interest: List[int] = field(default_factory=list)  # [] = all
    primary: PrimaryModelCfg = field(default_factory=PrimaryModelCfg)


@dataclass
class PlateDetectorCfg:
    enabled: bool = False
    backend: str = "openvino"
    onnx: str = ""
    openvino: str = ""
    pt: str = ""
    trt: str = ""


@dataclass
class OCRCfg:
    enabled: bool = False
    keras_model: str = ""
    plate_config: str = ""
    crop_mode: str = "vehicle_lower_third"   # vehicle_lower_third | plate_detection
    vehicle_classes_for_lower_third: List[str] = field(
        default_factory=lambda: ["car", "van", "truck", "bus"])
    min_plate_area_px: int = 200
    every_n_frames: int = 5                   # throttle OCR per track
    vote_window: int = 5                      # majority-vote window for plate text
    plate_detector: PlateDetectorCfg = field(default_factory=PlateDetectorCfg)


@dataclass
class CMCCfg:
    enabled: bool = True
    method: str = "affine"                    # affine | homography
    downscale: float = 0.5
    n_features: int = 1000
    match_ratio: float = 0.75
    ransac_thresh: float = 5.0
    min_matches: int = 20


@dataclass
class TrackerCfg:
    high_conf: float = 0.4
    low_conf: float = 0.15
    iou: float = 0.3
    max_age: int = 50
    min_hits: int = 3
    cmc: CMCCfg = field(default_factory=CMCCfg)
    emat: bool = True
    interpolate_max_gap: int = 5
    same_class_gate: bool = False
    trajectory_len: int = 60


@dataclass
class PIDCfg:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    out_limit: float = 1.0


@dataclass
class PIDSetCfg:
    yaw: PIDCfg = field(default_factory=lambda: PIDCfg(0.08, 0.0, 0.02, 60.0))
    pitch: PIDCfg = field(default_factory=lambda: PIDCfg(0.06, 0.0, 0.02, 60.0))
    forward: PIDCfg = field(default_factory=lambda: PIDCfg(0.4, 0.0, 0.0, 3.0))
    vertical: PIDCfg = field(default_factory=lambda: PIDCfg(0.2, 0.0, 0.0, 2.0))


@dataclass
class FollowCfg:
    enabled: bool = True
    default_policy: str = "highest_score_area"  # highest_score_area | largest_area | nearest_center
    locked_id: Optional[int] = None
    preferred_classes: List[str] = field(default_factory=list)  # [] = all
    target_area_norm: float = 0.12              # desired target_area / frame_area
    deadzone_px: float = 12.0
    lost_recovery_frames: int = 15
    pid: PIDSetCfg = field(default_factory=PIDSetCfg)


@dataclass
class MAVLinkCfg:
    connection: str = "udpin:0.0.0.0:14550"
    system_id: int = 1
    component_id: int = 1


@dataclass
class ROS2Cfg:
    node: str = "uav_follow"
    cmd_vel_topic: str = "/cmd_vel"
    gimbal_topic: str = "/gimbal/cmd"


@dataclass
class ControllerCfg:
    backend: str = "mock"                      # mock | mavlink | ros2
    mavlink: MAVLinkCfg = field(default_factory=MAVLinkCfg)
    ros2: ROS2Cfg = field(default_factory=ROS2Cfg)


@dataclass
class VideoSinkCfg:
    enabled: bool = True
    path: str = "output/pipeline.mp4"
    codec: str = "mp4v"                        # mp4v (Win) | avc1 (Jetson)
    fps: float = 30.0
    draw: bool = True


@dataclass
class TelemetrySinkCfg:
    enabled: bool = True
    path: str = "output/telemetry.jsonl"
    csv_summary: str = "output/telemetry_summary.csv"


@dataclass
class ControlLogSinkCfg:
    enabled: bool = True
    path: str = "output/commands.jsonl"


@dataclass
class SinksCfg:
    video: VideoSinkCfg = field(default_factory=VideoSinkCfg)
    telemetry: TelemetrySinkCfg = field(default_factory=TelemetrySinkCfg)
    control_log: ControlLogSinkCfg = field(default_factory=ControlLogSinkCfg)


@dataclass
class Config:
    source: SourceCfg = field(default_factory=SourceCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    ocr: OCRCfg = field(default_factory=OCRCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    follow: FollowCfg = field(default_factory=FollowCfg)
    controller: ControllerCfg = field(default_factory=ControllerCfg)
    sinks: SinksCfg = field(default_factory=SinksCfg)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return _build(cls, data)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls.from_dict(data)
        cfg._source_path = path  # type: ignore[attr-defined]
        return cfg

    def model_path_for(self, backend: str) -> str:
        """Resolve the primary model path for the active backend."""
        p = self.detector.primary
        return {
            "torch": p.pt, "onnx": p.onnx, "openvino": p.openvino, "trt": p.trt,
        }.get(backend, "")

    def resolve_backend_device(self) -> str:
        return self.detector.device.strip()
