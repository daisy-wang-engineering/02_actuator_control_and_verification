from dataclasses import dataclass
import numpy as np


@dataclass
class PIDGains:
    kp: float = 14.0
    ki: float = 1.0
    kd: float = 1.2
    derivative_filter_s: float = 0.003   # first-order low-pass time constant on the derivative term (0 = unfiltered)


class PIDController:
    """Position PID with derivative on measurement and conditional anti-windup."""

    def __init__(self, gains: PIDGains | None = None, voltage_limit_v: float = 12.0):
        self.gains = gains or PIDGains()
        self.voltage_limit_v = voltage_limit_v
        self.integral = 0.0
        self.previous_measurement = None
        self.derivative_filtered = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_measurement = None
        self.derivative_filtered = 0.0

    def update(self, target_rad, measured_rad, dt_s):
        g = self.gains
        error = target_rad - measured_rad
        derivative = 0.0
        if self.previous_measurement is not None:
            derivative = -(measured_rad - self.previous_measurement) / dt_s
        self.previous_measurement = measured_rad
        if g.derivative_filter_s > 0.0:
            alpha = dt_s / (g.derivative_filter_s + dt_s)
            self.derivative_filtered += alpha * (derivative - self.derivative_filtered)
            derivative = self.derivative_filtered

        proposed_integral = self.integral + error * dt_s
        raw = g.kp * error + g.ki * proposed_integral + g.kd * derivative
        output = max(-self.voltage_limit_v, min(self.voltage_limit_v, raw))

        # Integrate when unsaturated, or when the error would drive the output
        # back out of saturation.
        if raw == output or (raw > self.voltage_limit_v and error < 0) or (
            raw < -self.voltage_limit_v and error > 0
        ):
            self.integral = proposed_integral
        return output


@dataclass
class FaultMonitorConfig:
    """Plausibility monitor for a stuck/frozen position sensor.

    Trips when the controller has been commanding more than ``command_threshold_v``
    for ``window_s`` while the measured position has moved less than ``min_motion_deg``.
    A blocked rotor (stall) produces the same signature, so it also trips; in both cases
    removing drive is the correct response.
    """
    enabled: bool = True
    command_threshold_v: float = 1.0
    window_s: float = 0.05
    min_motion_deg: float = 0.5
    invalid_timeout_s: float = 0.02     # continuous loss/invalid feedback longer than this is latched as a fault


class FaultMonitor:
    def __init__(self, config: FaultMonitorConfig | None = None):
        self.c = config or FaultMonitorConfig()
        self.reset()

    def reset(self):
        self.latched = False
        self.fault_time_s = None
        self.reason = None
        self._invalid_since = None
        self._active_since = None
        self._ref = None

    def update(self, time_s, command_v, measured_rad):
        """Call every control sample (measured_rad is None/NaN when feedback is invalid).
        Returns True once a fault is latched."""
        c = self.c
        if not c.enabled or self.latched:
            return self.latched
        if measured_rad is None or not np.isfinite(measured_rad):
            if self._invalid_since is None:
                self._invalid_since = time_s
            elif time_s - self._invalid_since >= c.invalid_timeout_s:
                self.latched, self.fault_time_s, self.reason = True, time_s, "FEEDBACK_LOSS"
            return self.latched
        self._invalid_since = None
        if abs(command_v) <= c.command_threshold_v:
            self._active_since, self._ref = None, None
            return False
        if self._active_since is None:
            self._active_since, self._ref = time_s, measured_rad
        elif abs(measured_rad - self._ref) > np.deg2rad(c.min_motion_deg):
            self._active_since, self._ref = time_s, measured_rad      # motion seen: restart window
        elif time_s - self._active_since >= c.window_s:
            self.latched, self.fault_time_s, self.reason = True, time_s, "STUCK_OR_STALL"
        return self.latched
