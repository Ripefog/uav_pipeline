"""ROS2 controller — STUB.

Documented stub (see mavlink.py for rationale). Records commands so the
pipeline runs with ``controller.backend: ros2`` without a live node.

Command -> ROS2 mapping:

    yaw_rate / forward_vel / vertical_vel -> geometry_msgs/Twist on
                                            cfg.cmd_vel_topic
        twist.linear.x  = forward_vel
        twist.linear.z  = vertical_vel
        twist.angular.z = yaw_rate (deg/s -> rad/s)
    pitch_rate -> sensor_msgs / custom gimbal msg on cfg.gimbal_topic
                  (e.g. a JointState or vendor gimbal command)

Requirements:
  * rclpy node spin thread, publisher creation on cfg.node
  * flight-control stack (px4_ros_com / ardupilot ros2) listening on cmd_vel
  * arming / mode set out of band

Install: ``pip install rclpy`` (from your ROS2 distro). Replace the TODOs.
"""
from typing import List, Optional

from ...config import ROS2Cfg
from ...contracts import Command
from .base import Controller


class ROS2Controller(Controller):
    def __init__(self, cfg: ROS2Cfg):
        self.cfg = cfg
        self.history: List[Command] = []
        self._last: Optional[Command] = None
        self._warned = False
        # TODO: rclpy.init(); self.node = rclpy.create_node(cfg.node)
        # TODO: self.cmd_vel = node.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        # TODO: self.gimbal = node.create_publisher(..., cfg.gimbal_topic, 10)
        # TODO: spin thread

    def _warn(self):
        if not self._warned:
            print("[controller] ROS2 STUB active — NO drone commands are sent. "
                  "Implement the node/publishers in ros2.py.")
            self._warned = True

    def send(self, cmd: Command):
        self._warn()
        # TODO: build Twist from cmd and self.cmd_vel.publish(twist)
        self.history.append(cmd)
        self._last = cmd

    @property
    def last_command(self) -> Optional[Command]:
        return self._last

    def close(self):
        # TODO: node.destroy(); rclpy.shutdown()
        pass
