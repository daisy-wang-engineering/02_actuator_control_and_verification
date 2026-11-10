"""Requirement-based verification suite. Every case maps to a requirement in requirements.py."""
from dataclasses import dataclass
import numpy as np
from .simulation import run_simulation, SimulationConfig, calculate_metrics, V2_GAINS
from .models import EncoderParameters, MotorParameters
from .control import FaultMonitorConfig
from .analysis import loop_metrics
from .montecarlo import run_monte_carlo
from .studies import stuck_onsets
from .requirements import REQUIREMENTS, requirement_by_id


@dataclass
class VerificationResult:
    ver_id: str
    req_id: str
    measured: float
    limit: float
    unit: str
    sense: str
    passed: bool
    margin_pct: float
    method: str
    notes: str

    # backwards-compatible aliases used by older scripts
    @property
    def test_id(self): return self.ver_id


def _result(req_id, measured, notes="", extra_ok=True):
    r = requirement_by_id(req_id)
    ok = (measured <= r.limit) if r.sense == "<=" else (measured >= r.limit)
    margin = 100.0 * ((r.limit - measured) / r.limit if r.sense == "<=" else (measured - r.limit) / r.limit)
    return VerificationResult(r.ver_id, r.id, float(measured), r.limit, r.unit, r.sense,
                              bool(ok and extra_ok and np.isfinite(measured)), float(margin), r.method, notes)


def run_verification_suite(gains=None, mc_runs=200):
    g = gains or V2_GAINS
    out = []

    nominal = run_simulation(SimulationConfig(), pid_gains=g)
    m = calculate_metrics(nominal)
    out.append(_result("SYS-001", m["final_error_deg"], "30 deg step, error at t = 3 s"))
    out.append(_result("SYS-002", m["overshoot_deg"], "peak true angle minus target"))
    out.append(_result("SYS-003", m["settling_time_s"], "last exit from +/-1 deg band"))

    out.append(_result("SYS-016", m["steady_command_rms_v"], f"derivative filter {g.derivative_filter_s*1e3:.0f} ms; unfiltered gives ~1.3 V"))

    lm = loop_metrics(g)
    out.append(_result("SYS-004", lm["bandwidth_hz"], f"set-point response -3 dB; crossover {lm['crossover_hz']:.2f} Hz"))
    out.append(_result("SYS-005", lm["phase_margin_deg"], f"peak sensitivity Ms = {lm['peak_sensitivity']:.2f}"))

    delayed = run_simulation(SimulationConfig(loop_delay_samples=20), pid_gains=g)
    dm = calculate_metrics(delayed)
    delayed_ok = dm["overshoot_deg"] <= requirement_by_id("SYS-002").limit and dm["settling_time_s"] <= requirement_by_id("SYS-003").limit
    out.append(_result("SYS-006", lm["delay_margin_ms"],
                       f"with 20 ms delay: overshoot {dm['overshoot_deg']:.2f} deg, settling {dm['settling_time_s']:.2f} s",
                       extra_ok=delayed_ok))

    finals = [np.rad2deg(run_simulation(SimulationConfig(seed=s), pid_gains=g)["true_angle_rad"][-1]) for s in range(10)]
    out.append(_result("SYS-007", float(max(finals) - min(finals)), "ten deterministic noise seeds"))

    dist = run_simulation(SimulationConfig(disturbance_torque_nm=0.025), pid_gains=g)
    t, a = dist["time_s"], np.rad2deg(dist["true_angle_rad"])
    during = (t >= 1.0) & (t <= 2.0)
    out.append(_result("SYS-008", float(np.max(np.abs(30.0 - a[during]))), "0.025 N m applied 1-2 s"))
    out.append(_result("SYS-009", float(abs(30.0 - a[-1])), "error at t = 3 s, 1 s after load removal"))

    _, pass_rate = run_monte_carlo(g, n=mc_runs)
    out.append(_result("SYS-010", pass_rate, f"{mc_runs} randomised parameter sets"))

    drop = run_simulation(SimulationConfig(duration_s=1.0), encoder_parameters=EncoderParameters(dropout_probability=1.0), pid_gains=g)
    inv = run_simulation(SimulationConfig(duration_s=1.0), encoder_parameters=EncoderParameters(out_of_range_after_s=0.2), pid_gains=g)
    zero_ok = bool(np.all(np.abs(drop["voltage_v"]) < 1e-12)) and bool(np.all(np.abs(inv["voltage_v"][inv["time_s"] >= 0.2]) < 1e-12))
    def brake_latency(res, t0):
        idx = np.where(res["brake_engaged"])[0]
        return float("inf") if not len(idx) else 1e3 * (res["time_s"][idx[0]] - t0)
    worst = max(brake_latency(drop, 0.0), brake_latency(inv, 0.2))
    out.append(_result("SYS-011", worst, "100% dropout and invalid (NaN) reading; 0 V on every sample", extra_ok=zero_ok))

    onsets_move = np.arange(0.0, 0.121, 0.02)
    lat = [r["monitor_latency_ms"] for r in stuck_onsets(g, onsets_move)]
    out.append(_result("SYS-012", float(max(lat)) if all(v is not None for v in lat) else float("inf"),
                       "onsets 0-0.12 s while the axis is moving", extra_ok=all(v is not None for v in lat)))

    sweep = stuck_onsets(g, np.arange(0.0, 0.601, 0.01))
    peak = max(r["peak_deg"] for r in sweep)
    out.append(_result("SYS-013", peak, f"61 onsets 0-0.6 s; limit switch needed in {sum(r['limit_switch'] for r in sweep)} of them"))

    scenarios = [
        run_simulation(SimulationConfig(), pid_gains=g),
        dist,
        run_simulation(SimulationConfig(target_deg=45.0, retarget_time_s=0.12, retarget_deg=-45.0, duration_s=1.5), pid_gains=g),
        run_simulation(SimulationConfig(duration_s=1.0, disturbance_torque_nm=1.0, disturbance_start_s=0.0, disturbance_end_s=9.0),
                       motor_parameters=MotorParameters(limit_switch_deg=None), pid_gains=g,
                       monitor_config=FaultMonitorConfig(enabled=False)),   # blocked rotor, no protective layers
    ]
    out.append(_result("SYS-014", max(max(np.max(np.abs(s["voltage_v"])), np.max(np.abs(s["applied_voltage_v"]))) for s in scenarios),
                       "nominal, load, hard reversal, blocked rotor"))
    out.append(_result("SYS-015", max(np.max(np.abs(s["current_a"])) for s in scenarios),
                       "blocked rotor with monitor and limit switch disabled = worst case"))
    return sorted(out, key=lambda r: r.req_id)
