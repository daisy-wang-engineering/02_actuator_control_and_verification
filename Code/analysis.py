"""Linear (frequency-domain) analysis of the position loop.

Plant (voltage -> shaft angle), Coulomb friction and saturation neglected:

    P(s) = Kt / ( s [ (J s + b)(L s + R) + Kt Ke ] )

Loop transfer (filtered derivative on measurement): L(s) = (Kp + Ki/s + Kd s/(tau s + 1)) P(s)
Set-point response: T(s) = (Kp + Ki/s) P(s) / (1 + L(s))
"""
import numpy as np
from .models import MotorParameters
from .control import PIDGains


def plant(s, p: MotorParameters):
    return p.torque_constant_nm_per_a / (
        s * ((p.inertia_kg_m2 * s + p.viscous_friction_nm_per_rad_s)
             * (p.inductance_h * s + p.resistance_ohm)
             + p.torque_constant_nm_per_a * p.back_emf_v_per_rad_s))


def frequency_response(gains: PIDGains, p: MotorParameters | None = None, f_hz=None):
    p = p or MotorParameters()
    f = np.logspace(-2, 3, 6000) if f_hz is None else np.asarray(f_hz)
    s = 2j * np.pi * f
    P = plant(s, p)
    D = gains.kd * s / (gains.derivative_filter_s * s + 1.0)
    L = (gains.kp + gains.ki / s + D) * P
    T = (gains.kp + gains.ki / s) * P / (1 + L)
    S = 1 / (1 + L)
    return f, L, T, S


def loop_metrics(gains: PIDGains, p: MotorParameters | None = None):
    f, L, T, S = frequency_response(gains, p)
    mag, ph = np.abs(L), np.unwrap(np.angle(L))
    above = np.where(mag >= 1.0)[0]
    ic = above[-1]
    fc = f[ic]
    pm = 180.0 + np.degrees(ph[ic])
    bw = f[np.where(np.abs(T) >= 10 ** (-3 / 20))[0][-1]]
    return {
        "crossover_hz": float(fc),
        "phase_margin_deg": float(pm),
        "bandwidth_hz": float(bw),
        "peak_sensitivity": float(np.max(np.abs(S))),
        "delay_margin_ms": float(1e3 * np.radians(pm) / (2 * np.pi * fc)),
    }
