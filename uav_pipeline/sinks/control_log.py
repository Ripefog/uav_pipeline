"""Control log sink — one JSON line per emitted Command (for replay/debug)."""
import json
import os

from ..config import ControlLogSinkCfg
from ..contracts import FrameContext
from .base import Sink


class ControlLogSink(Sink):
    def __init__(self, cfg: ControlLogSinkCfg):
        self.cfg = cfg
        self._f = None
        if cfg.enabled:
            os.makedirs(os.path.dirname(cfg.path) or ".", exist_ok=True)
            self._f = open(cfg.path, "w", encoding="utf-8")

    def write(self, ctx: FrameContext):
        if self._f is None or ctx.command is None:
            return
        cmd = ctx.command
        rec = {
            "frame": ctx.meta.idx,
            "ts": round(cmd.ts, 4),
            "yaw_rate": round(cmd.yaw_rate, 4),
            "pitch_rate": round(cmd.pitch_rate, 4),
            "forward_vel": round(cmd.forward_vel, 4),
            "vertical_vel": round(cmd.vertical_vel, 4),
            "target_id": cmd.target_id,
            "target_lost": cmd.target_lost,
            "mode": ctx.follow_state.mode,
            "errors": {k: round(float(v), 3) for k, v in cmd.raw_errors.items()},
        }
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
