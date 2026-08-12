"""Ba tầng chặn drift cho SOT. Hình học thuần — KHÔNG import torch/cv2/MCITrack.

Vì sao phải có: MCITrack không có khái niệm "mất target". track() luôn trả 1 box
mỗi frame; clip_box(margin=10) (lib/utils/box_ops.py:101) còn ép box nằm trong ảnh
nên nó dính vào biên rồi bám sang vật khác. Confidence KHÔNG dùng làm ngưỡng được:
đo trên truck của uav0000339_00001_v, còn target conf min 0.466, mất target conf
max 0.713 — chồng lấn nặng.

Ba tầng bù nhau, đo trên 5 đối tượng: jump cắt car (f56) và motor (f191);
class-gate verify cắt truck (f170) và bus (f70); motion gate bắt drift dần mà cả
hai tầng kia đều mù.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..config import SotGuardCfg
from ..contracts import Detection
from .class_groups import accepted_ids


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU của 2 box dạng [x, y, w, h] (góc trên-trái)."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / u) if u > 0 else 0.0


@dataclass
class GuardVerdict:
    alive: bool = True
    provisional: bool = False        # đang nghi ngờ, chưa đủ k -> hoãn ghi sink
    lost_at: Optional[int] = None    # frame ĐẦU chuỗi (cắt lui về đây)
    reason: str = ""


class LostGuard:
    def __init__(self, cfg: SotGuardCfg, frame_width: int, init_cls: int,
                 names: Dict[int, str]):
        self.cfg = cfg
        # jump.px calibrate ở 1904px. VisDrone trải 1344x756 -> 3840x2160 (2.86x):
        # cùng một chuyển động thật ra số px rất khác -> không scale là báo oan trên
        # video 4K và bỏ sót drift trên video nhỏ.
        self.jump_px = cfg.jump.px * (float(frame_width) / float(cfg.jump.ref_width))
        self.accept = accepted_ids(cfg.gate, init_cls, names)
        self.reset()

    def reset(self):
        # KHÔNG khởi tạo _prev bằng box của DETECTOR: box detector và box MCITrack là
        # 2 estimator khác nhau cho cùng một vật -> chênh scale ở frame đầu là bình
        # thường, không phải "nhảy". Trước khi sửa, uav0000117_02622_v_car LOST oan
        # ngay frame 2 (1/349 frame alive).
        self._prev: Optional[List[float]] = None
        self._motion_miss = 0
        self._streak_start: Optional[int] = None
        # Mốc dự đoán CHỈ cập nhật từ frame ĐÃ ĐƯỢC CHẤP NHẬN. Để frame giật làm mốc
        # thì metric tự đầu độc chính nó (uav0000086 person rank 4: f148 giật 35px,
        # f149-150 đã về đúng IoU_GT 0.66 nhưng IoU vs dự đoán vẫn 0.000).
        self._good_a: Optional[tuple] = None   # (box, frame_idx) cũ hơn
        self._good_b: Optional[tuple] = None   # (box, frame_idx) mới hơn
        self._verify_miss = 0
        # Chỉ kết án track mà detector ĐÃ TỪNG xác nhận. Vật 26x41 px của
        # uav0000339: detector 0 hit ở f10..f50 trong khi tracker vẫn đúng
        # (IoU 0.67-0.75) -> không latch thì cắt oan f30, và nâng K chỉ làm cắt muộn
        # hơn chứ không cứu được.
        self._verify_confirmed = False

    # ------------------------------------------------------------------ #
    def step(self, frame_idx: int, box_xywh: Sequence[float],
             detect_fn: Callable[[], List[Detection]]) -> GuardVerdict:
        """Chạy 3 tầng theo đúng thứ tự jump -> motion -> verify."""
        if not self.cfg.enabled:
            return GuardVerdict()
        box = [float(v) for v in box_xywh]

        v = self._jump(frame_idx, box)
        if v is not None:
            return v
        v = self._motion(frame_idx, box)
        if v is not None:
            return v

        # frame đã qua hết các gate -> mới được làm mốc
        self._prev = list(box)
        self._good_a, self._good_b = self._good_b, (list(box), frame_idx)
        self._motion_miss, self._streak_start = 0, None
        return self._verify(frame_idx, box, detect_fn)

    # ------------------------------------------------------------------ #
    def _jump(self, i: int, box: List[float]) -> Optional[GuardVerdict]:
        """Tầng 1, mỗi frame: cú nhảy bất khả thi về vật lý.

        Gate class MÙ HOÀN TOÀN khi drift sang vật CÙNG class — detection ở vị trí
        mới cũng là 'car' nên verify pass mọi lần. Tầng này không cần detector.
        """
        c = self.cfg.jump
        if not c.enabled or self._prev is None:
            return None
        p = self._prev
        dx = (box[0] + box[2] / 2.0) - (p[0] + p[2] / 2.0)
        dy = (box[1] + box[3] / 2.0) - (p[1] + p[3] / 2.0)
        disp = (dx * dx + dy * dy) ** 0.5
        pa, na = p[2] * p[3], box[2] * box[3]
        ratio = max(na / pa, pa / na) if pa > 0 and na > 0 else 1.0
        if disp > self.jump_px or ratio > c.area:
            # đang có frame bị hoãn (motion gate) thì cắt lui về đầu chuỗi luôn
            lost_at = self._streak_start if self._streak_start is not None else i
            return GuardVerdict(alive=False, lost_at=lost_at,
                                reason=f"jump {disp:.0f}px, area x{ratio:.1f}")
        return None

    def _motion(self, i: int, box: List[float]) -> Optional[GuardVerdict]:
        """Tầng 2, mỗi frame: drift DẦN sang vật cùng class kề bên.

        So box thật với dự đoán vận tốc không đổi từ CẶP FRAME TỐT cuối cùng
        (nhiều bước, không dùng frame đang bị nghi ngờ làm mốc).
        """
        c = self.cfg.motion
        if not c.enabled or self._good_a is None or self._good_b is None:
            return None
        (ba, fa), (bb, fb) = self._good_a, self._good_b
        dt = max(1, fb - fa)
        steps = i - fb
        pred = [bb[k] + (bb[k] - ba[k]) / dt * steps for k in range(4)]
        m_iou = iou_xywh(box, pred)
        if m_iou >= c.iou:
            return None
        self._motion_miss += 1
        if self._streak_start is None:
            self._streak_start = i
        # _prev VẪN cập nhật: tầng jump so với frame liền trước, không phải với mốc tốt
        self._prev = list(box)
        if self._motion_miss >= c.k:
            return GuardVerdict(
                alive=False, lost_at=self._streak_start,
                reason=(f"motion gate: {c.k} frame liên tiếp IoU với dự đoán < "
                        f"{c.iou} (cuối={m_iou:.3f}), cắt lui về frame "
                        f"{self._streak_start}"))
        return GuardVerdict(alive=True, provisional=True,
                            reason=(f"motion miss {self._motion_miss}/{c.k} "
                                    f"(IoU dự đoán {m_iou:.3f})"))

    def _verify(self, i: int, box: List[float],
                detect_fn: Callable[[], List[Detection]]) -> GuardVerdict:
        """Tầng 3, mỗi verify_every frame: chạy detector tại vị trí box tracker."""
        c = self.cfg
        if c.verify_every < 1 or i % c.verify_every != 0:
            return GuardVerdict()
        ok = self._verify_hit(detect_fn(), box)
        if ok:
            self._verify_confirmed = True
        self._verify_miss = 0 if ok else self._verify_miss + 1
        if self._verify_miss >= c.K and self._verify_confirmed:
            return GuardVerdict(alive=False, lost_at=i,
                                reason=(f"{c.K} verify MISS liên tiếp "
                                        f"(gate {c.gate})"))
        return GuardVerdict()

    def _verify_hit(self, dets: List[Detection], box: List[float]) -> bool:
        """HIT khi có detection thoả CẢ HAI: IoU > iou_gate và cùng nhóm class."""
        for d in dets:
            if self.accept is not None and int(d.cls) not in self.accept:
                continue
            if iou_xywh(box, [d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1]) > self.cfg.iou_gate:
                return True
        return False
