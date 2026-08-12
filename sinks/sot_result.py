"""SOT result sink — txt 7 cột, đúng format của repo MCITrack (tracking/*.py).

    frame,x,y,w,h,conf,alive     # x,y,w,h = xywh góc trên-trái, pixel ảnh gốc
    3,949.90,576.50,93.50,126.90,0.9260,1
    56,-1,-1,-1,-1,-1,0          # LOST hoặc chưa acquire

Hai chi tiết bắt buộc:
  * frame là 1-INDEXED (meta.idx + 1) — khớp GT VisDrone và khớp writer MOT
    (scripts/eval_mot_visdrone.py:39).
  * số dòng luôn bằng số frame (kể cả frame acquire/lost) nên align theo frame id
    không bao giờ lệch.

Ghi + flush TỪNG FRAME thay vì buffer tới close(): DeferredSinkWriter
(sinks/deferred.py) đã cắt lui TRƯỚC khi sink thấy ctx, nên sink không bao giờ
phải rút lại dòng đã ghi -> crash giữa đường vẫn còn kết quả.
"""
import os
from typing import Optional

from ..config import SotResultSinkCfg
from ..contracts import FrameContext
from .base import Sink


class SotResultSink(Sink):
    def __init__(self, cfg: SotResultSinkCfg):
        self.cfg = cfg
        self._f = None
        if cfg.enabled and cfg.path:
            os.makedirs(os.path.dirname(cfg.path) or ".", exist_ok=True)
            self._f = open(cfg.path, "w", encoding="utf-8")

    def write(self, ctx: FrameContext):
        if self._f is None:
            return
        fid = ctx.meta.idx + 1
        if ctx.tracks:
            t = ctx.tracks[0]
            x1, y1, x2, y2 = [float(v) for v in t.bbox]
            self._f.write(f"{fid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                          f"{t.confidence:.4f},1\n")
        else:
            self._f.write(f"{fid},-1,-1,-1,-1,-1,0\n")
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
