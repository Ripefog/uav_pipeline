"""MAVLink controller — STUB.

This is a documented stub, NOT a working driver. It records commands so the
pipeline runs end-to-end with ``controller.backend: mavlink`` without crashing,
but it does NOT arm, connect, or fly anything. Implement the real OFFBOARD loop
where the TODOs are before any flight.

Command -> MAVLink mapping (PX4 / ArduPilot):

    yaw_rate     -> SET_ATTITUDE_TARGET.body_yaw_rate
                   (convert deg/s -> rad/s: /57.2958)
    pitch_rate   -> gimbal: MAV_CMD_DO_MOUNT_CONTROL (pitch) or
                   MAVLink Gimbal v2 (MAVLINK_MSG_ID_GIMBAL_DEVICE_SET_ATTITUDE)
    forward_vel  -> SET_POSITION_TARGET_LOCAL_NED with type_mask = velocity,
                   vx = forward_vel (body->NED rotation required)
    vertical_vel -> same message, vz = -vertical_vel (NED z is down)

Requirements before this can fly (out of band of this stub):
  * mavutil connection on cfg.connection (e.g. 'udpin:0.0.0.0:14550')
  * heartbeat loop, system/time sync
  * vehicle arming + OFFBOARD mode entry
  * a >=10 Hz setpoint stream (OFFBOARD exits if starved)

Install: ``pip install pymavlink``. Then replace the TODOs below.
"""
from typing import List, Optional

from ...config import MAVLinkCfg
from ...contracts import Command
from .base import Controller


class MAVLinkController(Controller):
    def __init__(self, cfg: MAVLinkCfg):
        self.cfg = cfg
        self.history: List[Command] = []
        self._last: Optional[Command] = None
        self._warned = False
        # TODO: self.master = mavutil.mavlink_connection(cfg.connection, ...)
        # TODO: wait for heartbeat, set target system/component, arm, enter OFFBOARD

    def _warn(self):
        if not self._warned:
            print("[controller] MAVLink STUB active — NO drone commands are sent. "
                  "Implement the connection/OFFBOARD loop in mavlink.py.")
            self._warned = True

    def send(self, cmd: Command):
        self._warn()
        # TODO: convert cmd -> SET_ATTITUDE_TARGET / SET_POSITION_TARGET_LOCAL_NED
        # TODO: self.master.mav.set_attitude_target_send(...)
        self.history.append(cmd)
        self._last = cmd

    @property
    def last_command(self) -> Optional[Command]:
        return self._last

    def close(self):
        # TODO: exit OFFBOARD, disarm if owned
        pass
