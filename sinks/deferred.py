"""DeferredSinkWriter — hoãn ghi vài frame để guard cắt lui được trong luồng stream.

Motion gate của SOT (sot/guard.py) có thể tuyên bố LOST rồi CẮT LUI về frame đầu
chuỗi nghi ngờ. Offline thì dễ: giữ frame trong buffer, chưa ghi video. Pipeline là
stream — frame đã ghi vào VideoWriter không lấy lại được.

Lớp này giữ tối đa (motion.k - 1) frame nghi ngờ. Khi guard cắt lui thì xoá tracks
của các frame đang giữ có frame_id >= lost_at rồi mới ghi -> 0 frame box sai lọt ra
output. Với motion.k=2 thì trễ đúng 1 frame (33ms @30fps).

Chỉ được dùng khi guard.enabled và guard.motion.enabled. Đường MOT và đường
guard-off không đi qua lớp này.
"""
import copy
from typing import Callable, List, Optional

from ..contracts import FrameContext


def _snapshot_tracks(tracks):
    """Copy nong (shallow) moi Track truoc khi giu lai trong _held.

    SotTracker chi giu 1 instance Track duy nhat va tra ve [self.trk] moi
    frame (sot/tracker.py) -> tracks[0] cua ctx dang giu va tracks[0] cua ctx
    frame SAU la CUNG MOT object Python. Track.update() GAN LAI (khong mutate
    in-place) cac thuoc tinh nhu bbox/confidence/age (contracts.py) nen
    copy.copy() la du cho chung -> gan lai o object goc khong lam thay doi
    ban copy. Nhung 'trajectory' la list bi mutate bang append()/pop() ngay
    tai cho, nen phai copy rieng list nay, khong thoi ban copy van thay doi
    theo object goc.
    """
    snap = []
    for t in tracks:
        t2 = copy.copy(t)
        traj = getattr(t2, "trajectory", None)
        if traj is not None:
            t2.trajectory = list(traj)
        snap.append(t2)
    return snap


class DeferredSinkWriter:
    def __init__(self, emit_fn: Callable[[FrameContext], None], max_hold: int):
        self._emit = emit_fn
        self.max_hold = max(0, int(max_hold))
        self._held: List[FrameContext] = []

    def write(self, ctx: FrameContext, provisional: bool = False,
              retract_from: Optional[int] = None):
        """provisional=True -> giữ lại. retract_from=<frame 1-indexed> -> cắt lui."""
        if retract_from is not None:
            # Cắt lui chỉ hợp lệ nếu MỌI frame có id >= retract_from vẫn còn
            # nắm được: đang trong _held, hoặc chính là ctx đang tới. Nếu frame
            # đang giữ SỚM NHẤT đã lớn hơn retract_from thì có khoảng hở
            # [retract_from, earliest-1] đã bị ghi ra ngoài từ trước (ví dụ
            # max_hold desync nhỏ hơn motion.k-1 khiến frame đầu chuỗi bị đẩy
            # ra trước khi verdict LOST kịp tới) -> box sai đã lọt ra output mà
            # không ai biết. earliest == retract_from là nhịp ĐÚNG, phải im.
            if self._held:
                earliest = self._held[0].meta.idx + 1
                gap = earliest > retract_from
            else:
                gap = ctx.meta.idx + 1 > retract_from
            if gap:
                # should-never-happen: chỉ log, không assert/raise vì đây là
                # hot path per-frame trong 1 lần chạy dài.
                held_ids = [h.meta.idx + 1 for h in self._held]
                print(f"[deferred] retract_from={retract_from} nam ngoai vung "
                      f"dang giu (held={held_ids}, ctx id={ctx.meta.idx + 1}) "
                      f"-> it nhat 1 frame tu {retract_from} da bi ghi ra truoc "
                      f"khi cat lui toi -> nghi desync max_hold/motion.k")
            for h in self._held:
                if h.meta.idx + 1 >= retract_from:
                    h.tracks = []
            self._flush()
            self._emit(ctx)
            return
        if provisional and self.max_hold > 0:
            # Chup nhanh tracks TAI THOI DIEM giu lai. Khong deep-copy ca ctx
            # (ctx.frame la anh goc, cop lang phi) -- chi thay ctx.tracks bang
            # list Track da copy, de tracker mutate object song ve sau khong
            # lam sai box da flush ra tu buffer nay.
            ctx.tracks = _snapshot_tracks(ctx.tracks)
            self._held.append(ctx)
            while len(self._held) > self.max_hold:
                self._emit(self._held.pop(0))
            return
        self._flush()
        self._emit(ctx)

    def _flush(self):
        while self._held:
            self._emit(self._held.pop(0))

    def close(self):
        # hết video mà còn frame đang giữ (chuỗi chưa đủ k) -> coi là hợp lệ
        self._flush()
