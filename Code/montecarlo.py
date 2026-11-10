"""Randomised parameter-tolerance analysis (Monte Carlo)."""
from dataclasses import replace
import numpy as np
from .models import MotorParameters, EncoderParameters
from .control import PIDGains
from .simulation import SimulationConfig, run_simulation, calculate_metrics
from .analysis import loop_metrics
from .requirements import requirement_by_id


def sample_parameters(rng):
    base = MotorParameters()
    kt = base.torque_constant_nm_per_a * rng.uniform(0.85, 1.15)
    motor = replace(
        base,
        resistance_ohm=base.resistance_ohm * rng.uniform(0.8, 1.2),
        torque_constant_nm_per_a=kt,
        back_emf_v_per_rad_s=kt,                       # Kt and Ke are physically tied
        inertia_kg_m2=base.inertia_kg_m2 * rng.uniform(0.6, 1.4),
        viscous_friction_nm_per_rad_s=base.viscous_friction_nm_per_rad_s * rng.uniform(0.5, 1.5),
        coulomb_friction_nm=base.coulomb_friction_nm * rng.uniform(0.5, 1.5),
    )
    enc = EncoderParameters()
    enc = replace(enc, noise_std_rad=enc.noise_std_rad * rng.uniform(0.5, 2.0))
    return motor, enc


def run_monte_carlo(gains: PIDGains, n=200, seed=2026):
    rng = np.random.default_rng(seed)
    limits = {k: requirement_by_id(k).limit for k in ("SYS-001", "SYS-002", "SYS-003", "SYS-004")}
    rows = []
    for i in range(n):
        motor, enc = sample_parameters(rng)
        res = run_simulation(SimulationConfig(seed=i), motor, enc, gains)
        m = calculate_metrics(res)
        lm = loop_metrics(gains, motor)
        ok = (m["final_error_deg"] <= limits["SYS-001"]
              and m["overshoot_deg"] <= limits["SYS-002"]
              and np.isfinite(m["settling_time_s"]) and m["settling_time_s"] <= limits["SYS-003"]
              and lm["bandwidth_hz"] >= limits["SYS-004"])
        rows.append({"run": i, "final_error_deg": m["final_error_deg"],
                     "overshoot_deg": m["overshoot_deg"], "settling_time_s": m["settling_time_s"],
                     "bandwidth_hz": lm["bandwidth_hz"], "phase_margin_deg": lm["phase_margin_deg"],
                     "delay_margin_ms": lm["delay_margin_ms"], "pass_all": bool(ok)})
    pass_rate = 100.0 * sum(r["pass_all"] for r in rows) / n
    return rows, pass_rate
