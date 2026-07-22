"""HUD-annotated video sink — draws detections/tracks/plate/follow reticle."""
import os
from typing import Optional

import cv2
import numpy as np

from ..config import VideoSinkCfg
from ..contracts import Command, Detection, FrameContext, Track
from .base import Sink

# BGR palette
_C_DET = (60, 60, 60)        # faint detections
_C_TRK = (0, 200, 0)         # confirmed tracks
_C_TGT = (255, 255, 0)       # followed target (cyan)
_C_RET = (0, 165, 255)       # reticle (orange)
_C_HUD = (255, 255, 255)
_C_BAN = (0, 200, 255)


class HUDAnnotatedSink(Sink):
    def __init__(self, cfg: VideoSinkCfg, target_area_norm: float = 0.12):
        self.cfg = cfg
        self.target_area_norm = target_area_norm
        self.writer: Optional[cv2.VideoWriter] = None
        self._shape = None

    # ------------------------------------------------------------------ #
    def write(self, ctx: FrameContext):
        if not self.cfg.enabled or ctx.frame is None:
            return
        frame = self.render(ctx)
        if self.cfg.path:
            self._ensure_writer(frame.shape)
            if self.writer is not None:
                # Guard against mixed-size sources (e.g. an image dir): the
                # writer is fixed to the first frame's size, so resize later
                # frames to match instead of dropping them.
                if self._shape is not None and frame.shape[:2] != self._shape:
                    frame = cv2.resize(frame, (self._shape[1], self._shape[0]))
                self.writer.write(frame)

    def _ensure_writer(self, frame_shape):
        if self.writer is not None:
            return
        h, w = frame_shape[:2]
        self._shape = (h, w)
        os.makedirs(os.path.dirname(self.cfg.path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec)
        self.writer = cv2.VideoWriter(self.cfg.path, fourcc, self.cfg.fps, (w, h))
        if not self.writer.isOpened():
            print(f"[sink/video] WARNING: VideoWriter failed to open "
                  f"(codec='{self.cfg.codec}'). Try sinks.video.codec: avc1/x264.")
            self.writer = None

    # ------------------------------------------------------------------ #
    def render(self, ctx: FrameContext) -> np.ndarray:
        frame = ctx.frame.copy()
        h, w = frame.shape[:2]
        follow = ctx.follow_state

        # faint detection boxes
        if self.cfg.draw:
            for d in ctx.detections:
                self._box(frame, d, _C_DET, 1)
                self._label(frame, f"{d.name} {d.score:.2f}",
                            (int(d.x1), int(d.y1)), _C_DET, scale=0.4)

        target_id = follow.target_id if follow.locked else None
        # tracks + trajectory + labels
        for t in ctx.tracks:
            is_tgt = (target_id is not None and t.track_id == target_id)
            color = _C_TGT if is_tgt else _C_TRK
            thick = 3 if is_tgt else 2
            self._box_track(frame, t, color, thick)
            if len(t.trajectory) > 1:
                pts = np.array(t.trajectory, dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)
            plate = f" :{t.plate_text}" if t.plate_text else ""
            self._label(frame, f"#{t.track_id} {t.name}{plate}",
                        (int(t.bbox[0]), int(t.bbox[1]) - 6), color, scale=0.5)

        # reticle at frame center (where we want the target) + keep-in box
        if ctx.follow_state.mode in ("tracking", "recover", "acquire"):
            self._draw_reticle(frame, ctx.command, follow, target_id)

        # HUD (top-left) + banner (bottom-left)
        self._draw_hud(frame, ctx)
        self._draw_banner(frame, ctx.command)
        return frame

    # ---- drawing helpers -------------------------------------------------
    def _box(self, frame, d: Detection, color, thick):
        cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, thick)

    def _box_track(self, frame, t: Track, color, thick):
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    def _label(self, frame, text, org, color, scale=0.5):
        x, y = org
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        y = max(y, th)
        cv2.rectangle(frame, (x, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
        cv2.putText(frame, text, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    def _draw_reticle(self, frame, cmd: Optional[Command], follow, target_id):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        # crosshair
        cv2.drawMarker(frame, (cx, cy), _C_RET, cv2.MARKER_CROSS, 26, 1, cv2.LINE_AA)
        # keep-in box sized to desired target area
        side = int((self.target_area_norm ** 0.5) * min(w, h) * 2)
        side = max(24, side)
        cv2.rectangle(frame, (cx - side // 2, cy - side // 2),
                      (cx + side // 2, cy + side // 2), _C_RET, 1, cv2.LINE_AA)

    def _draw_hud(self, frame, ctx: FrameContext):
        lines = [
            f"FPS {ctx.fps:5.1f}  frame {ctx.meta.idx}",
            f"FPS det {ctx.fps_detect:5.1f}  pipe {ctx.fps_pipeline:5.1f}",
            f"Det {len(ctx.detections)}  Trk {len(ctx.tracks)}",
            f"Mode {ctx.follow_state.mode}  tgt {ctx.follow_state.target_id}",
        ]
        sev = ctx.extra_stats.get("motion", "")
        if sev:
            lines.append(f"Motion {sev}")
        y = 4
        for ln in lines:
            self._label(frame, ln, (6, y + 14), _C_HUD, scale=0.5)
            y += 18

    def _draw_banner(self, frame, cmd: Optional[Command]):
        if cmd is None:
            return
        lost = "LOST" if cmd.target_lost else "lock"
        text = (f"yaw {cmd.yaw_rate:+6.1f}  pitch {cmd.pitch_rate:+6.1f}  "
                f"fwd {cmd.forward_vel:+5.2f}  [{lost}]")
        h = frame.shape[0]
        self._label(frame, text, (6, h - 12), _C_BAN, scale=0.5)

    # ------------------------------------------------------------------ #
    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
