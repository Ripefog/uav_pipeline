"""SotTracker — state machine acquire -> tracking -> lost, bọc thành Track.

Cùng vai trò với DroneByteTracker.update() nên pipeline.py chỉ rẽ MỘT nhánh. Vì
kết quả là đúng type Track của contracts.py, follow/ + HUD + telemetry chạy nguyên
không phải sửa:
  * follow/selector.py:14 lọc age == 0 -> phải gọi Track.update() mỗi frame.
  * LOST -> trả [] -> follow._lost_step() tự coast rồi phanh.

Không biết MCITrack là gì: `model` chỉ cần 2 method initialize()/track() (xem
sot/mcitrack_wrapper.py), nhờ vậy test được bằng model giả, không cần GPU.
"""
from typing import Callable, Dict, List, Optional, Sequence

from ..config import SotCfg
from ..contracts import Detection, Track
from .guard import LostGuard


def _xywh_to_xyxy(b: Sequence[float]) -> List[float]:
    return [float(b[0]), float(b[1]), float(b[0]) + float(b[2]),
            float(b[1]) + float(b[3])]


class _DetectOnce:
    """Gọi detector nhiều nhất 1 lần cho 1 frame, giữ lại kết quả.

    Guard chỉ cần detection ở frame verify; acquire cần ở mọi frame. Lớp này để
    pipeline không detect 2 lần trên cùng frame, và để biết frame nào đã detect
    (ctx.detections).
    """

    def __init__(self, fn: Callable[[], List[Detection]],
                 prefetched: Optional[List[Detection]] = None):
        self._fn = fn
        self.result: Optional[List[Detection]] = prefetched

    def __call__(self) -> List[Detection]:
        if self.result is None:
            self.result = self._fn()
        return self.result


def _pick_init(dets: List[Detection],
               init_classes: Sequence[int]) -> Optional[Detection]:
    """Box conf cao nhất trong init_classes ([] = mọi class)."""
    cand = [d for d in dets if not init_classes or int(d.cls) in init_classes]
    if not cand:
        return None
    return max(cand, key=lambda d: d.score)


class SotTracker:
    def __init__(self, cfg: SotCfg, names: Dict[int, str], model):
        self.cfg = cfg
        self.names = names
        self.model = model
        self.mode = "acquire"          # acquire | tracking | lost
        self.status = "acquire"        # dòng cho HUD/telemetry
        self.last_detections: List[Detection] = []
        self.provisional = False       # frame đang bị motion gate nghi ngờ
        self.retract_from: Optional[int] = None   # set đúng 1 frame khi LOST
        self.lost_at: Optional[int] = None
        self.lost_reason = ""
        self.trk: Optional[Track] = None
        self.guard: Optional[LostGuard] = None
        self._next_id = 1
        self._acquire_frames = 0
        self._init_cls = -1

    @property
    def needs_deferral(self) -> bool:
        """True nếu guard có thể cắt lui -> pipeline phải hoãn ghi sink."""
        return bool(self.cfg.guard.enabled and self.cfg.guard.motion.enabled)

    # ------------------------------------------------------------------ #
    def update(self, frame, frame_idx: int,
               detect_fn: Callable[[], List[Detection]],
               prefetched: Optional[List[Detection]] = None) -> List[Track]:
        det = _DetectOnce(detect_fn, prefetched)
        self.provisional = False
        self.retract_from = None
        try:
            if self.mode == "acquire":
                return self._acquire(frame, frame_idx, det)
            if self.mode == "tracking":
                return self._tracking(frame, frame_idx, det)
            return []
        finally:
            self.last_detections = det.result or []

    # ------------------------------------------------------------------ #
    def _acquire(self, frame, frame_idx: int, det: _DetectOnce) -> List[Track]:
        # init_bbox chỉ dùng cho lần bám ĐẦU TIÊN: sau khi LOST nó đã lạc hậu.
        if self.cfg.init_bbox is not None and self._next_id == 1:
            box = [float(v) for v in self.cfg.init_bbox]
            score, cls, name = 1.0, -1, "init_bbox"
        else:
            self._acquire_frames += 1
            pick = _pick_init(det(), self.cfg.init_classes)
            if pick is None:
                self.status = f"acquire ({self._acquire_frames} frame, 0 box)"
                return []
            box = [pick.x1, pick.y1, pick.w, pick.h]
            score, cls, name = float(pick.score), int(pick.cls), pick.name

        h, w = frame.shape[:2]
        self.model.initialize(frame, box)
        self._init_cls = cls
        self.guard = LostGuard(self.cfg.guard, w, cls, self.names)
        self.trk = Track(track_id=self._next_id, bbox=_xywh_to_xyxy(box),
                         confidence=score, frame_id=frame_idx, cls=cls, name=name)
        self._next_id += 1
        self.mode = "tracking"
        self.status = f"tracking #{self.trk.track_id} init conf {score:.2f}"
        return [self.trk]

    def _tracking(self, frame, frame_idx: int, det: _DetectOnce) -> List[Track]:
        box, score = self.model.track(frame)
        v = self.guard.step(frame_idx, box, det)
        if not v.alive:
            self.lost_at, self.lost_reason = v.lost_at, v.reason
            self.retract_from = v.lost_at
            self.status = f"LOST @{v.lost_at} {v.reason}"
            self.trk = None
            if self.cfg.on_lost == "reacquire":
                self.mode = "acquire"
                self.guard = None
                self._acquire_frames = 0
            else:
                self.mode = "lost"
            return []
        self.trk.update(_xywh_to_xyxy(box), float(score), frame_idx,
                        cls=self._init_cls)
        self.provisional = v.provisional
        self.status = (f"tracking #{self.trk.track_id} conf {score:.2f}"
                       + (f"  [{v.reason}]" if v.provisional else ""))
        return [self.trk]
