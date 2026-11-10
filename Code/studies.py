"""Parametric studies used for the report figures and for verification evidence."""
import numpy as np
from .models import EncoderParameters, MotorParameters
from .control import PIDGains, FaultMonitorConfig
from .simulation import SimulationConfig, run_simulation, calculate_metrics
from .analysis import loop_metrics


def design_space(kps, kds, ki=1.0, tau=0.003):
    """Grid of (Kp, Kd): nominal-sim metrics + linear-analysis metrics."""
    out = []
    for kp in kps:
        for kd in kds:
            g = PIDGains(kp, ki, kd, tau)
            m = calculate_metrics(run_simulation(SimulationConfig(), pid_gains=g))
            d = calculate_metrics(run_simulation(SimulationConfig(loop_delay_samples=20), pid_gains=g))
            r = run_simulation(SimulationConfig(disturbance_torque_nm=0.025), pid_gains=g)
            t, ang = r["time_s"], np.rad2deg(r["true_angle_rad"])
            peak = float(np.max(np.abs(30.0 - ang[(t >= 1.0) & (t <= 2.0)])))
            a = loop_metrics(g)
            out.append({"kp": kp, "kd": kd, **m, **a, "overshoot_20ms_deg": d["overshoot_deg"],
                        "settling_20ms_s": d["settling_time_s"], "dist_peak_deg": peak})
    return out


def latency_sweep(gains, delays_ms):
    out = []
    for d in delays_ms:
        res = run_simulation(SimulationConfig(loop_delay_samples=int(d)), pid_gains=gains)
        out.append({"delay_ms": d, **calculate_metrics(res)})
    return out


def stuck_onsets(gains, onsets, monitor=True, duration_s=1.0, motor_parameters=None):
    """Stuck-encoder injection at each onset time. Returns peak |angle| and detection latency."""
    out = []
    for on in onsets:
        res = run_simulation(
            SimulationConfig(duration_s=duration_s), motor_parameters=motor_parameters,
            encoder_parameters=EncoderParameters(stuck=True, stuck_after_s=float(on)), pid_gains=gains, monitor_config=FaultMonitorConfig(enabled=monitor))
        ft = res["fault_time_s"]
        out.append({"onset_s": float(on), "peak_deg": float(np.rad2deg(np.max(np.abs(res["true_angle_rad"])))),
                    "monitor_latency_ms": None if ft is None else 1e3 * (ft - float(on)),
                    "limit_switch": res["limit_switch_time_s"] is not None})
    return out


def nuisance_trip_load(gains, torques):
    """Smallest sustained holding load that trips the stuck/stall monitor (nuisance trip)."""
    for tq in torques:
        res = run_simulation(SimulationConfig(disturbance_torque_nm=float(tq), disturbance_start_s=1.0,
                                              disturbance_end_s=9.0), pid_gains=gains)
        if res["fault_time_s"] is not None:
            return float(tq)
    return None


# Protection-layer configurations for the defence-in-depth study (motor-side protections)
PROTECTION_LAYERS = {
    "none":                  (dict(brake_torque_nm=0.0, limit_switch_deg=None, current_limit_a=None), False),
    "monitor only":          (dict(brake_torque_nm=0.0, limit_switch_deg=None, current_limit_a=4.0), True),
    "monitor + brake":       (dict(brake_torque_nm=0.5, limit_switch_deg=None, current_limit_a=4.0), True),
    "monitor + brake + switch": (dict(brake_torque_nm=0.5, limit_switch_deg=60.0, current_limit_a=4.0), True),
}


def protection_study(gains, onsets):
    from dataclasses import replace
    out = {}
    for name, (kw, mon) in PROTECTION_LAYERS.items():
        out[name] = stuck_onsets(gains, onsets, monitor=mon, motor_parameters=replace(MotorParameters(), **kw))
    return out


def blocked_rotor_current(gains, **motor_kw):
    from dataclasses import replace
    res = run_simulation(SimulationConfig(duration_s=1.0, disturbance_torque_nm=1.0, disturbance_start_s=0.0, disturbance_end_s=9.0),
                         motor_parameters=replace(MotorParameters(limit_switch_deg=None), **motor_kw), pid_gains=gains,
                         monitor_config=FaultMonitorConfig(enabled=False))
    return float(np.max(np.abs(res["current_a"])))
