"""Discrete PID with integral clamp, output clamp, and deadzone."""
from typing import Optional


class PID:
    def __init__(self, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0,
                 out_limit: float = 1.0, i_limit: Optional[float] = None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit = abs(out_limit)
        self.i_limit = abs(i_limit) if i_limit is not None else None
        self._i = 0.0
        self._prev = 0.0
        self._has_prev = False

    def reset(self):
        self._i = 0.0
        self._prev = 0.0
        self._has_prev = False

    def update(self, error: float, dt: float = 1.0) -> float:
        """One PID step. ``dt`` defaults to 1 (per-frame) so gains are tuned in
        per-frame units; pass a real dt (seconds) if you prefer SI units."""
        self._i += error * dt
        if self.i_limit is not None:
            self._i = max(-self.i_limit, min(self.i_limit, self._i))
        deriv = 0.0
        if self._has_prev and dt > 0:
            deriv = (error - self._prev) / dt
        self._prev = error
        self._has_prev = True

        out = self.kp * error + self.ki * self._i + self.kd * deriv
        if self.out_limit > 0:
            out = max(-self.out_limit, min(self.out_limit, out))
        return out
