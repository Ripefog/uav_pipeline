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
from typing import Callable, List, Optional

from ..contracts import FrameContext


class DeferredSinkWriter:
    def __init__(self, emit_fn: Callable[[FrameContext], None], max_hold: int):
        self._emit = emit_fn
        self.max_hold = max(0, int(max_hold))
        self._held: List[FrameContext] = []

    def write(self, ctx: FrameContext, provisional: bool = False,
              retract_from: Optional[int] = None):
        """provisional=True -> giữ lại. retract_from=<frame 1-indexed> -> cắt lui."""
        if retract_from is not None:
            matched = any(h.meta.idx + 1 >= retract_from for h in self._held)
            if not matched and ctx.meta.idx + 1 < retract_from:
                # should-never-happen: guard tuyên bố cắt lui về 1 frame không
                # nằm trong buffer đang giữ (_held) và cũng không phải chính
                # frame hiện tại -> desync max_hold/motion.k hoặc thứ tự gọi
                # sai. Chỉ log, không assert/raise vì đây là hot path.
                held_ids = [h.meta.idx + 1 for h in self._held]
                print(f"[deferred] retract_from={retract_from} khong khop frame "
                      f"nao dang giu (held={held_ids}) hoac ctx hien tai "
                      f"(id={ctx.meta.idx + 1}) -> nghi desync max_hold/motion.k")
            for h in self._held:
                if h.meta.idx + 1 >= retract_from:
                    h.tracks = []
            self._flush()
            self._emit(ctx)
            return
        if provisional and self.max_hold > 0:
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
