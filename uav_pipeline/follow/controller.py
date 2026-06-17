"""FollowController — turns a selected track into a gimbal/body Command.

Resolution-independent: errors are pixel offsets (ex, ey) normalized only for
the area term. Output sign convention (tune gains/signs to your airframe):

  * yaw_rate    > 0  => turn / pan right   (corrects target right of center, ex>0)
  * pitch_rate  > 0  => tilt gimbal down   (corrects target below center,  ey>0)
  * forward_vel < 0  => back up            (target larger than desired => too close)
  * vertical_vel      held at 0 (altitude hold) by default
"""
from dataclasses import replace
from typing import List, Optional, Tuple

from ..config import FollowCfg
from ..contracts import Command, FollowState, Track
from .pid import PID
from .selector import TargetSelector


class FollowController:
    def __init__(self, cfg: FollowCfg):
        self.cfg = cfg
        self.enabled = cfg.enabled
        self.selector = TargetSelector(cfg)
        p = cfg.pid
        self.pid_yaw = PID(p.yaw.kp, p.yaw.ki, p.yaw.kd, p.yaw.out_limit)
        self.pid_pitch = PID(p.pitch.kp, p.pitch.ki, p.pitch.kd, p.pitch.out_limit)
        self.pid_forward = PID(p.forward.kp, p.forward.ki, p.forward.kd, p.forward.out_limit)
        self.pid_vertical = PID(p.vertical.kp, p.vertical.ki, p.vertical.kd, p.vertical.out_limit)
        self._lost = 0
        self._last_cmd: Optional[Command] = None
        self.state = FollowState()

    def reset(self):
        for pid in (self.pid_yaw, self.pid_pitch, self.pid_forward, self.pid_vertical):
            pid.reset()
        self._lost = 0
        self._last_cmd = None
        self.state = FollowState()

    def step(self, tracks: List[Track], frame_hw: Tuple[int, int],
             ts: float) -> Tuple[FollowState, Command]:
        if not self.enabled:
            self.state = FollowState(mode="idle")
            return self.state, Command(ts=ts)

        sel = self.selector.select(tracks, frame_hw)
        if sel is not None:
            return self._track(sel, frame_hw, ts)
        return self._lost_step(ts)

    # ------------------------------------------------------------------ #
    def _track(self, sel: Track, frame_hw, ts) -> Tuple[FollowState, Command]:
        H, W = frame_hw
        self._lost = 0
        ex = sel.cx - W / 2.0
        ey = sel.cy - H / 2.0
        dz = self.cfg.deadzone_px
        ex_d = ex if abs(ex) > dz else 0.0
        ey_d = ey if abs(ey) > dz else 0.0

        area_norm = sel.area / float(W * H)
        escale = area_norm - self.cfg.target_area_norm

        yaw = self.pid_yaw.update(ex_d)
        pitch = self.pid_pitch.update(ey_d)
        forward = -self.pid_forward.update(escale)   # too close => back up
        vertical = 0.0                                # altitude hold

        cmd = Command(
            ts=ts, yaw_rate=yaw, pitch_rate=pitch,
            forward_vel=forward, vertical_vel=vertical,
            target_id=sel.track_id, target_lost=False,
            raw_errors={"ex": ex, "ey": ey, "escale": escale, "area_norm": area_norm},
        )
        self._last_cmd = cmd
        self.state = FollowState(target_id=sel.track_id, locked=True,
                                 target_lost=False, mode="tracking")
        return self.state, cmd

    def _lost_step(self, ts) -> Tuple[FollowState, Command]:
        self._lost += 1
        if self._last_cmd is not None and self._lost <= self.cfg.lost_recovery_frames:
            # Coast on the last command so brief occlusions don't yank the drone.
            coast = replace(self._last_cmd, ts=ts, target_lost=True)
            self.state = FollowState(target_id=self._last_cmd.target_id,
                                     locked=False, target_lost=True, mode="recover")
            return self.state, coast
        # Brake: zero all outputs and reset integrators.
        for pid in (self.pid_yaw, self.pid_pitch, self.pid_forward, self.pid_vertical):
            pid.reset()
        self.state = FollowState(target_id=None, locked=False,
                                 target_lost=True, mode="acquire")
        return self.state, Command(ts=ts, target_lost=True)
