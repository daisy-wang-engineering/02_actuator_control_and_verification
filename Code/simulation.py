from collections import deque
from dataclasses import dataclass
import numpy as np
from .models import DCActuator, MotorParameters, Encoder, EncoderParameters
from .control import PIDController, PIDGains, FaultMonitor, FaultMonitorConfig

# Iteration 0 = gains as submitted (14, 1.0, 1.2). Iteration 1 (24, 1.0, 1.6) failed SYS-006. Iteration 2 (19, 1.0, 1.6, 3 ms derivative filter) = final, after CR-1 and the SYS-016 noise finding.
V1_GAINS = PIDGains(kp=14.0, ki=1.0, kd=1.2, derivative_filter_s=0.0)          # iteration 0 (as submitted)
ITER1_GAINS = PIDGains(kp=24.0, ki=1.0, kd=1.6, derivative_filter_s=0.0)       # iteration 1 (fails SYS-006)
V2_GAINS = PIDGains(kp=19.0, ki=1.0, kd=1.6, derivative_filter_s=0.003)


@dataclass
class SimulationConfig:
    duration_s: float = 3.0
    dt_s: float = 0.001
    target_deg: float = 30.0
    disturbance_torque_nm: float = 0.0
    disturbance_start_s: float = 1.0
    disturbance_end_s: float = 2.0
    seed: int = 1
    loop_delay_samples: int = 0          # extra command latency (1 sample = dt_s)
    retarget_time_s: float | None = None  # optional second set-point (e.g. hard reversal test)
    retarget_deg: float = 0.0


def run_simulation(config=None, motor_parameters=None, encoder_parameters=None,
                   pid_gains=None, monitor_config=None):
    c = config or SimulationConfig()
    actuator = DCActuator(motor_parameters)
    encoder = Encoder(encoder_parameters, seed=c.seed)
    controller = PIDController(pid_gains or V2_GAINS, voltage_limit_v=actuator.p.max_voltage_v)
    monitor = FaultMonitor(monitor_config)
    actuator.reset()
    controller.reset()
    delay = deque([0.0] * c.loop_delay_samples)

    n = int(round(c.duration_s / c.dt_s)) + 1
    time = np.arange(n) * c.dt_s
    target = np.deg2rad(c.target_deg)
    true_angle = np.zeros(n)
    speed = np.zeros(n)
    current = np.zeros(n)
    measured = np.full(n, np.nan)
    voltage = np.zeros(n)
    applied = np.zeros(n)
    load = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    fault = np.zeros(n, dtype=bool)
    brake = np.zeros(n, dtype=bool)
    hw_trip_time = None
    target_arr = np.full(n, target)

    for i, t in enumerate(time):
        theta = actuator.state[0]
        cur_target = np.deg2rad(c.retarget_deg) if (c.retarget_time_s is not None and t >= c.retarget_time_s) else target
        target_arr[i] = cur_target
        lim = actuator.p.limit_switch_deg
        if lim is not None and hw_trip_time is None and abs(theta) >= np.deg2rad(lim):
            hw_trip_time = t                   # independent of encoder and controller
        reading = encoder.read(theta, t)
        if reading is None or not np.isfinite(reading):
            command = 0.0                      # fail safe: no feedback -> no drive
            monitor.update(t, command, None)
        else:
            valid[i] = True
            measured[i] = reading
            command = controller.update(cur_target, reading, c.dt_s)
            monitor.update(t, command, reading)
        trip = monitor.fault_time_s if monitor.latched else None
        if hw_trip_time is not None:
            trip = hw_trip_time if trip is None else min(trip, hw_trip_time)
        if trip is not None:
            command = 0.0                      # latched fault or limit switch -> de-energise
        fault[i] = trip is not None
        brake_on = trip is not None and (t - trip) >= actuator.p.brake_engage_delay_s
        brake[i] = brake_on

        delay.append(command)
        applied_cmd = delay.popleft()

        torque = (c.disturbance_torque_nm
                  if c.disturbance_start_s <= t <= c.disturbance_end_s else 0.0)
        true_angle[i], speed[i], current[i] = actuator.state
        voltage[i], applied[i], load[i] = command, applied_cmd, torque
        if i < n - 1:
            actuator.step(applied_cmd, torque, c.dt_s, brake_on)

    return {
        "time_s": time, "target_rad": target_arr,
        "true_angle_rad": true_angle, "measured_angle_rad": measured,
        "speed_rad_s": speed, "current_a": current, "voltage_v": voltage,
        "applied_voltage_v": applied, "load_torque_nm": load,
        "measurement_valid": valid, "fault_latched": fault, "brake_engaged": brake,
        "fault_time_s": monitor.fault_time_s, "fault_reason": monitor.reason,
        "limit_switch_time_s": hw_trip_time,
    }


def calculate_metrics(result, target_deg=None, settling_band_deg=1.0):
    t = result["time_s"]
    target = result["target_rad"][0]
    angle = result["true_angle_rad"]
    error_deg = np.rad2deg(target - angle)
    target_deg = float(np.rad2deg(target) if target_deg is None else target_deg)
    final_error_deg = abs(float(error_deg[-1]))
    overshoot_deg = max(0.0, float(np.max(np.rad2deg(angle)) - target_deg))
    band = np.deg2rad(settling_band_deg)
    outside = np.where(np.abs(target - angle) > band)[0]
    if not len(outside):
        settling_time = 0.0
    elif outside[-1] + 1 < len(t):
        settling_time = float(t[outside[-1] + 1])
    else:
        settling_time = float("nan")
    return {
        "final_error_deg": final_error_deg,
        "overshoot_deg": overshoot_deg,
        "settling_time_s": settling_time,
        "peak_speed_rad_s": float(np.max(np.abs(result["speed_rad_s"]))),
        "peak_current_a": float(np.max(np.abs(result["current_a"]))),
        "peak_voltage_v": float(np.max(np.abs(result["voltage_v"]))),
        "steady_command_rms_v": float(np.sqrt(np.mean(result["voltage_v"][t >= 1.0] ** 2))),
    }
