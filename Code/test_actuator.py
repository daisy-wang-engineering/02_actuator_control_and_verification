import numpy as np
import pytest
from actuator_sim.models import DCActuator, Encoder, EncoderParameters, MotorParameters
from actuator_sim.control import PIDController, PIDGains, FaultMonitor, FaultMonitorConfig
from actuator_sim.simulation import SimulationConfig, run_simulation, calculate_metrics, V2_GAINS
from actuator_sim.analysis import loop_metrics
from actuator_sim.verification import run_verification_suite
from actuator_sim.requirements import REQUIREMENTS


def test_actuator_moves_positive_with_positive_voltage():
    a = DCActuator(); a.reset()
    for _ in range(200):
        a.step(6.0, 0.0, 0.001)
    assert a.state[0] > 0


def test_driver_current_limit_holds_in_blocked_rotor():
    res = run_simulation(SimulationConfig(duration_s=0.5, disturbance_torque_nm=1.0, disturbance_start_s=0.0, disturbance_end_s=9.0),
                         motor_parameters=MotorParameters(limit_switch_deg=None), monitor_config=FaultMonitorConfig(enabled=False))
    assert np.max(np.abs(res["current_a"])) <= 4.1


def test_encoder_quantisation_resolution():
    e = Encoder(EncoderParameters(noise_std_rad=0.0))
    step = 2 * np.pi / 4096
    assert abs(e.read(0.3, 0.0) - round(0.3 / step) * step) < 1e-12


def test_pid_output_is_voltage_limited():
    c = PIDController(PIDGains(1000, 0, 0), voltage_limit_v=12.0)
    assert abs(c.update(1.0, 0.0, 0.001)) <= 12.0


def test_nominal_step_meets_basic_performance():
    m = calculate_metrics(run_simulation(SimulationConfig(), pid_gains=V2_GAINS))
    assert m["final_error_deg"] <= 0.5 and m["overshoot_deg"] <= 3.0 and m["settling_time_s"] <= 1.0


def test_stuck_encoder_trips_monitor_and_brake_before_hard_stop():
    res = run_simulation(SimulationConfig(duration_s=1.0), encoder_parameters=EncoderParameters(stuck=True))
    assert res["fault_time_s"] is not None and res["fault_time_s"] <= 0.1
    assert np.max(np.abs(np.rad2deg(res["true_angle_rad"]))) < 85.0


def test_without_protection_stuck_encoder_hits_hard_stop():
    mp = MotorParameters(brake_torque_nm=0.0, limit_switch_deg=None)
    res = run_simulation(SimulationConfig(duration_s=1.0), motor_parameters=mp, encoder_parameters=EncoderParameters(stuck=True),
                         monitor_config=FaultMonitorConfig(enabled=False))
    assert np.max(np.rad2deg(res["true_angle_rad"])) >= 89.0


def test_feedback_loss_commands_zero_and_brakes():
    res = run_simulation(SimulationConfig(duration_s=0.5), encoder_parameters=EncoderParameters(dropout_probability=1.0))
    assert np.all(res["voltage_v"] == 0.0) and res["brake_engaged"].any()


def test_derivative_filter_reduces_chatter():
    raw = calculate_metrics(run_simulation(SimulationConfig(), pid_gains=PIDGains(19, 1, 1.6, 0.0)))
    filt = calculate_metrics(run_simulation(SimulationConfig(), pid_gains=V2_GAINS))
    assert filt["steady_command_rms_v"] < 0.5 * raw["steady_command_rms_v"]


def test_loop_margins_of_final_design():
    m = loop_metrics(V2_GAINS)
    assert m["phase_margin_deg"] >= 45 and m["bandwidth_hz"] >= 2.0 and m["delay_margin_ms"] >= 20


def test_verification_suite_covers_every_requirement_and_passes():
    results = run_verification_suite(V2_GAINS, mc_runs=20)
    assert [r.req_id for r in results] == sorted(r.id for r in REQUIREMENTS)
    assert all(r.passed for r in results), [r.req_id for r in results if not r.passed]
