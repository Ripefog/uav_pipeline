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
    preprocess: str = "ultralytics"            # ultralytics (RGB/255/center-pad) | yolox (BGR/0-255/top-left-pad)
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    fp16: bool = False
    device: str = ""                          # "" = auto; cuda:0 / CPU / GPU
    batch: int = 1                            # >1 = buffer N frames, run detector once (fixed-batch IR)
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
    device: str = "gpu"                        # cpu | gpu — GPU by default; auto-falls-back
                                               #   to CPU where TF has no GPU (native Windows
                                               #   TF>=2.11, Jetson without NVTF, tensorflow-cpu)
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
    enabled: bool = True     # False = tắt MOT (dùng khi sot.enabled=true)
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
class SotJumpCfg:
    """Tầng 1 của guard: bắt cú nhảy bất khả thi về vật lý (chạy mỗi frame)."""
    enabled: bool = True
    px: float = 90.0          # ngưỡng dịch chuyển tâm/frame TẠI ref_width
    area: float = 2.5         # ngưỡng tỉ lệ diện tích/frame
    ref_width: float = 1904.0  # độ phân giải mà px được calibrate


@dataclass
class SotMotionCfg:
    """Tầng 2: bắt drift DẦN sang vật cùng class kề bên (dự đoán vận tốc không đổi)."""
    enabled: bool = True
    iou: float = 0.05
    k: int = 2                # số frame liên tiếp vi phạm -> LOST (cắt lui về đầu chuỗi)


@dataclass
class SotGuardCfg:
    enabled: bool = False     # OFF = hành vi MCITrack gốc, không bao giờ LOST
    gate: str = "class"       # class | family | presence
    verify_every: int = 10
    K: int = 3                # số lần verify MISS liên tiếp -> LOST
    iou_gate: float = 0.3
    jump: SotJumpCfg = field(default_factory=SotJumpCfg)
    motion: SotMotionCfg = field(default_factory=SotMotionCfg)


@dataclass
class SotCfg:
    enabled: bool = False
    mcitrack_root: str = "/home/anlnm/UAV/MCITrack"
    config: str = "mcitrack_l384"      # experiments/mcitrack/<config>.yaml
    dataset_preset: str = "uav"        # chọn preset UPT/UPH/INTER/MB
    device: str = "cuda:0"             # 'cuda' hoặc 'cuda:N'
    init_bbox: Optional[List[float]] = None   # [x,y,w,h]; None = detector tự lấy
    init_classes: List[int] = field(default_factory=list)  # [] = theo detector.classes_of_interest
    detect_every_frame: bool = False
    on_lost: str = "stop"              # stop | reacquire (chỉ có tác dụng khi guard bật)
    guard: SotGuardCfg = field(default_factory=SotGuardCfg)


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
class SotResultSinkCfg:
    enabled: bool = True
    path: str = "output/sot_result.txt"


@dataclass
class SinksCfg:
    video: VideoSinkCfg = field(default_factory=VideoSinkCfg)
    telemetry: TelemetrySinkCfg = field(default_factory=TelemetrySinkCfg)
    control_log: ControlLogSinkCfg = field(default_factory=ControlLogSinkCfg)
    sot_result: SotResultSinkCfg = field(default_factory=SotResultSinkCfg)


@dataclass
class Config:
    source: SourceCfg = field(default_factory=SourceCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    ocr: OCRCfg = field(default_factory=OCRCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    sot: SotCfg = field(default_factory=SotCfg)
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

    def validate(self) -> List[str]:
        """Lỗi cấu hình chí tử. Rỗng = ok. Caller tự sys.exit.

        Chạy TRƯỚC khi load checkpoint MCITrack 1.44GB (~20s) để chết sớm, chết rõ.
        """
        errs: List[str] = []
        if self.sot.enabled and self.tracker.enabled:
            errs.append(
                "sot.enabled và tracker.enabled không được bật cùng lúc — SOT và MOT "
                "loại trừ nhau. Đặt tracker.enabled: false để chạy SOT, hoặc "
                "sot.enabled: false để chạy MOT.")
        if not self.sot.enabled and not self.tracker.enabled:
            errs.append("cả sot.enabled và tracker.enabled đều false — "
                        "không có tracker nào chạy.")
        if self.sot.enabled:
            if self.sot.on_lost not in ("stop", "reacquire"):
                errs.append(f"sot.on_lost phải là 'stop' hoặc 'reacquire', "
                            f"đang là '{self.sot.on_lost}'")
            g = self.sot.guard
            if g.gate not in ("class", "family", "presence"):
                errs.append(f"sot.guard.gate phải là class|family|presence, "
                            f"đang là '{g.gate}'")
            if g.enabled:
                if g.verify_every < 1:
                    errs.append("sot.guard.verify_every phải >= 1")
                if g.K < 1:
                    errs.append("sot.guard.K phải >= 1")
                if g.motion.k < 1:
                    errs.append("sot.guard.motion.k phải >= 1")
                if g.jump.ref_width <= 0:
                    errs.append("sot.guard.jump.ref_width phải > 0")
            if self.sot.init_bbox is not None:
                b = self.sot.init_bbox
                if len(b) != 4 or float(b[2]) <= 0 or float(b[3]) <= 0:
                    errs.append(f"sot.init_bbox phải là [x,y,w,h] với w>0 và h>0, "
                                f"đang là {b}")
        return errs

    def warnings(self) -> List[str]:
        """Cảnh báo không chết, chỉ in ra."""
        w: List[str] = []
        if self.sot.enabled:
            if self.source.loop:
                w.append("source.loop=true + SOT: video chạy vòng lại nên guard sẽ "
                         "tính chuyển động sai ở chỗ nối.")
            if self.sot.on_lost == "reacquire" and not self.sot.guard.enabled:
                w.append("sot.on_lost=reacquire nhưng sot.guard.enabled=false -> "
                         "MCITrack không bao giờ tuyên bố LOST nên reacquire vô hiệu.")
        return w

    def model_path_for(self, backend: str) -> str:
        """Resolve the primary model path for the active backend."""
        p = self.detector.primary
        return {
            "torch": p.pt, "onnx": p.onnx, "openvino": p.openvino, "trt": p.trt,
        }.get(backend, "")

    def resolve_backend_device(self) -> str:
        return self.detector.device.strip()
