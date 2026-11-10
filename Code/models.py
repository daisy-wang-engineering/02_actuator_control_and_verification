from dataclasses import dataclass
import numpy as np


@dataclass
class MotorParameters:
    resistance_ohm: float = 2.0
    inductance_h: float = 0.02
    torque_constant_nm_per_a: float = 0.08
    back_emf_v_per_rad_s: float = 0.08
    inertia_kg_m2: float = 0.002
    viscous_friction_nm_per_rad_s: float = 0.004
    coulomb_friction_nm: float = 0.001
    max_voltage_v: float = 12.0
    brake_torque_nm: float = 0.5          # spring-applied fail-safe brake (~1.5x stall torque at the current limit)
    brake_engage_delay_s: float = 0.010   # mechanical engagement delay
    limit_switch_deg: float | None = 60.0  # independent hardware travel-limit switch (+/-), hard stops at +/-90 deg
    current_limit_a: float | None = 4.0   # driver current limit (80% derating of a notional 5 A driver)
    min_angle_rad: float = np.deg2rad(-90.0)
    max_angle_rad: float = np.deg2rad(90.0)


@dataclass
class EncoderParameters:
    counts_per_revolution: int = 4096
    noise_std_rad: float = np.deg2rad(0.03)
    dropout_probability: float = 0.0
    stuck: bool = False
    stuck_after_s: float = 0.0
    out_of_range_after_s: float | None = None

    @property
    def resolution_rad(self) -> float:
        return 2.0 * np.pi / self.counts_per_revolution


class DCActuator:
    """A simplified brushed DC motor driving a single rotary axis.

    State is [shaft angle (rad), shaft speed (rad/s), armature current (A)].
    A hard-stop model prevents travel beyond the configured angular limits.
    """

    def __init__(self, parameters: MotorParameters | None = None):
        self.p = parameters or MotorParameters()
        self.state = np.zeros(3, dtype=float)

    def reset(self, angle_rad: float = 0.0):
        self.state = np.array([angle_rad, 0.0, 0.0], dtype=float)

    def _derivative(self, state, voltage_v, load_torque_nm, brake_nm=0.0):
        theta, omega, current = state
        p = self.p
        dtheta = omega
        di = (voltage_v - p.resistance_ohm * current
              - p.back_emf_v_per_rad_s * omega) / p.inductance_h
        motor_torque = p.torque_constant_nm_per_a * current
        friction = (p.viscous_friction_nm_per_rad_s * omega
                    + p.coulomb_friction_nm * np.tanh(omega / 0.02)
                    + brake_nm * np.tanh(omega / 0.005))
        domega = (motor_torque - friction - load_torque_nm) / p.inertia_kg_m2
        if theta <= p.min_angle_rad and domega < 0:
            domega = 0.0
            dtheta = max(0.0, dtheta)
        if theta >= p.max_angle_rad and domega > 0:
            domega = 0.0
            dtheta = min(0.0, dtheta)
        return np.array([dtheta, domega, di])

    def step(self, voltage_v: float, load_torque_nm: float, dt_s: float, brake_on: bool = False):
        p = self.p
        b = p.brake_torque_nm if brake_on else 0.0
        u = float(np.clip(voltage_v, -p.max_voltage_v, p.max_voltage_v))
        if p.current_limit_a is not None:
            # Driver current limit: restrict the applied voltage to the value that
            # would drive at most +/- I_lim through the armature at the present speed.
            omega = self.state[1]
            u = float(np.clip(u, -p.resistance_ohm * p.current_limit_a + p.back_emf_v_per_rad_s * omega,
                                 p.resistance_ohm * p.current_limit_a + p.back_emf_v_per_rad_s * omega))
        x = self.state
        k1 = self._derivative(x, u, load_torque_nm, b)
        k2 = self._derivative(x + 0.5 * dt_s * k1, u, load_torque_nm, b)
        k3 = self._derivative(x + 0.5 * dt_s * k2, u, load_torque_nm, b)
        k4 = self._derivative(x + dt_s * k3, u, load_torque_nm, b)
        self.state = x + (dt_s / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.state[0] = np.clip(self.state[0], p.min_angle_rad, p.max_angle_rad)
        return self.state.copy()


class Encoder:
    def __init__(self, parameters: EncoderParameters | None = None, seed: int = 1):
        self.p = parameters or EncoderParameters()
        self.rng = np.random.default_rng(seed)
        self._stuck_value = None

    def read(self, true_angle_rad: float, time_s: float):
        p = self.p
        if p.stuck and time_s >= p.stuck_after_s:
            if self._stuck_value is None:
                self._stuck_value = self._quantize(true_angle_rad)
            return self._stuck_value
        if p.out_of_range_after_s is not None and time_s >= p.out_of_range_after_s:
            return float("nan")
        if self.rng.random() < p.dropout_probability:
            return None
        noisy = true_angle_rad + self.rng.normal(0.0, p.noise_std_rad)
        return self._quantize(noisy)

    def _quantize(self, angle_rad):
        step = self.p.resolution_rad
        return float(np.round(angle_rad / step) * step)
