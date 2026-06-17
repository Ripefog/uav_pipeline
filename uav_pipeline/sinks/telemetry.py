"""Telemetry sink — JSONL per frame + CSV summary."""
import csv
import json
import os
from typing import Any

from ..config import TelemetrySinkCfg
from ..contracts import FrameContext
from .base import Sink


def _jsonable(o: Any) -> Any:
    import numpy as np
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class TelemetrySink(Sink):
    def __init__(self, cfg: TelemetrySinkCfg):
        self.cfg = cfg
        self._fjson = None
        self._csv = None
        self._csv_writer = None
        if cfg.enabled:
            os.makedirs(os.path.dirname(cfg.path) or ".", exist_ok=True)
            self._fjson = open(cfg.path, "w", encoding="utf-8")
            if cfg.csv_summary:
                os.makedirs(os.path.dirname(cfg.csv_summary) or ".", exist_ok=True)
                self._csv = open(cfg.csv_summary, "w", newline="", encoding="utf-8")
                self._csv_writer = csv.writer(self._csv)
                self._csv_writer.writerow(
                    ["frame", "fps", "n_det", "n_trk", "mode", "target_id",
                     "yaw", "pitch", "forward", "vertical", "target_lost"])

    def write(self, ctx: FrameContext):
        if self._fjson is None:
            return
        cmd = ctx.command
        rec = {
            "frame": ctx.meta.idx,
            "ts": round(ctx.meta.ts, 4),
            "fps": round(ctx.fps, 2),
            "n_det": len(ctx.detections),
            "n_trk": len(ctx.tracks),
            "mode": ctx.follow_state.mode,
            "target_id": ctx.follow_state.target_id,
            "motion": ctx.extra_stats.get("motion", ""),
            "tracks": [t.as_dict() for t in ctx.tracks],
            "command": (None if cmd is None else {
                "yaw_rate": round(cmd.yaw_rate, 4),
                "pitch_rate": round(cmd.pitch_rate, 4),
                "forward_vel": round(cmd.forward_vel, 4),
                "vertical_vel": round(cmd.vertical_vel, 4),
                "target_id": cmd.target_id,
                "target_lost": cmd.target_lost,
            }),
        }
        self._fjson.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")
        self._fjson.flush()

        if self._csv_writer is not None:
            self._csv_writer.writerow([
                ctx.meta.idx, round(ctx.fps, 2), len(ctx.detections), len(ctx.tracks),
                ctx.follow_state.mode, ctx.follow_state.target_id,
                "" if cmd is None else round(cmd.yaw_rate, 3),
                "" if cmd is None else round(cmd.pitch_rate, 3),
                "" if cmd is None else round(cmd.forward_vel, 3),
                "" if cmd is None else round(cmd.vertical_vel, 3),
                "" if cmd is None else int(cmd.target_lost),
            ])

    def close(self):
        for f in (self._fjson, self._csv):
            if f is not None:
                f.close()
        self._fjson = self._csv = None
