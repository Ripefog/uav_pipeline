"""Controller interface + Mock implementation + factory.

A Controller turns a ``Command`` (gimbal/body rates) into an actual actuation.
The competition requires the Follow pillar to command the UAV, but sending real
flight commands is dangerous and hardware-specific, so:

  * ``MockController`` (default) records commands and exposes the last one so
    the HUD can draw a gimbal reticle — no hardware touched.
  * ``MAVLinkController`` / ``ROS2Controller`` are documented stubs describing
    exactly how to wire a real drone later.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from ...config import ControllerCfg
from ...contracts import Command


class Controller(ABC):
    @abstractmethod
    def send(self, cmd: Command):
        """Transmit (or record) a command."""

    def reset(self):
        pass

    def close(self):
        pass

    @property
    def last_command(self) -> Optional[Command]:
        return None


class MockController(Controller):
    """Records every command; never touches hardware. Used for demos."""

    def __init__(self):
        self.history: List[Command] = []
        self._last: Optional[Command] = None

    def send(self, cmd: Command):
        self.history.append(cmd)
        self._last = cmd

    @property
    def last_command(self) -> Optional[Command]:
        return self._last

    def reset(self):
        self.history.clear()
        self._last = None


def make_controller(cfg: ControllerCfg) -> Controller:
    backend = cfg.backend.lower().strip()
    if backend == "mock":
        return MockController()
    if backend == "mavlink":
        from .mavlink import MAVLinkController
        return MAVLinkController(cfg.mavlink)
    if backend == "ros2":
        from .ros2 import ROS2Controller
        return ROS2Controller(cfg.ros2)
    raise ValueError(f"Unknown controller backend '{backend}' "
                     f"(expected mock | mavlink | ros2)")
