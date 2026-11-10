"""Requirement registry: single source of truth for the requirement table, traceability
matrix and the pass/fail limits used by the verification suite.

All limits are NOTIONAL project values (not customer or program requirements). Each
carries a rationale so a reviewer can challenge it.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Need:
    id: str
    title: str
    statement: str


@dataclass(frozen=True)
class Requirement:
    id: str
    need: str
    text: str
    rationale: str
    method: str          # "Test" (simulated test case) or "Analysis"
    ver_id: str
    limit: float
    unit: str
    sense: str           # "<=" or ">="


NEEDS = [
    Need("N1", "Accuracy", "Place the load at the commanded angle."),
    Need("N2", "Response", "Respond fast enough for an outer guidance or attitude loop "
                           "(notional outer-loop bandwidth 0.5 Hz)."),
    Need("N3", "Robustness", "Keep performing with realistic sensor noise, load, latency "
                             "and part-to-part variation."),
    Need("N4", "Safety", "Fail to a safe state on sensor or driver faults and stay inside "
                         "hardware ratings."),
]

REQUIREMENTS = [
    Requirement("SYS-001", "N1", "Steady-state error after a 30 deg step shall be <= 0.5 deg.",
                "About 1.7% of the step; sets the accuracy the outer loop can assume.",
                "Test", "VER-001", 0.5, "deg", "<="),
    Requirement("SYS-002", "N2", "Step overshoot shall be <= 3 deg (10% of a 30 deg step).",
                "Overshoot consumes travel and mechanical clearance.",
                "Test", "VER-002", 3.0, "deg", "<="),
    Requirement("SYS-003", "N2", "The axis shall settle within +/-1 deg in <= 1.0 s.",
                "Settling several times faster than the outer-loop time constant (~0.3 s).",
                "Test", "VER-003", 1.0, "s", "<="),
    Requirement("SYS-004", "N2", "Closed-loop -3 dB bandwidth shall be >= 2.0 Hz.",
                "Flow-down: 4x separation from the 0.5 Hz outer loop (low end of the 4-10x rule of thumb). "
                "Relaxed from 2.5 Hz by CR-1 (see report, Section 6).",
                "Analysis", "VER-004", 2.0, "Hz", ">="),
    Requirement("SYS-005", "N3", "Loop phase margin shall be >= 45 deg.",
                "Standard minimum for well-damped response with margin for model error.",
                "Analysis", "VER-005", 45.0, "deg", ">="),
    Requirement("SYS-006", "N3", "Delay margin shall be >= 20 ms, and SYS-002/003 shall still hold "
                                 "with 20 ms of added loop latency.",
                "Covers 10 ms of driver/bus latency with a 2x allowance.",
                "Analysis + Test", "VER-006", 20.0, "ms", ">="),
    Requirement("SYS-007", "N3", "Final-position spread over ten noise realisations shall be <= 0.2 deg.",
                "Run-to-run repeatability under sensor noise.",
                "Test", "VER-007", 0.2, "deg", "<="),
    Requirement("SYS-008", "N3", "Peak deviation during a 0.025 N m, 1 s load disturbance shall be <= 2.0 deg.",
                "Load is ~8% of the torque available at the 4 A current limit.",
                "Test", "VER-008", 2.0, "deg", "<="),
    Requirement("SYS-009", "N3", "Error 1 s after the disturbance is removed shall be <= 0.75 deg.",
                "Recovery to near-nominal accuracy.",
                "Test", "VER-009", 0.75, "deg", "<="),
    Requirement("SYS-010", "N3", "SYS-001..004 shall be met in >= 95% of 200 randomised parameter sets "
                                 "(R +/-20%, Kt +/-15%, J +/-40%, friction +/-50%, noise 0.5-2x).",
                "Part-to-part and load-inertia uncertainty; nominal-only verification hides fragility.",
                "Analysis", "VER-010", 95.0, "% pass", ">="),
    Requirement("SYS-011", "N4", "On loss or invalidity of feedback the controller shall command 0 V on "
                                 "every sample and the brake shall engage within 50 ms.",
                "Safe state when the controller cannot trust its position.",
                "Test", "VER-011", 50.0, "ms", "<="),
    Requirement("SYS-012", "N4", "A stuck (frozen) encoder shall be detected and drive removed within 100 ms "
                                 "of the controller commanding > 1 V (tested for onsets during the move, 0-0.12 s).",
                "A frozen sensor looks healthy to the PID; plausibility monitoring is required.",
                "Test", "VER-012", 100.0, "ms", "<="),
    Requirement("SYS-013", "N4", "For a stuck encoder at any onset time (0-0.6 s) the axis shall not reach "
                                 "the +/-90 deg hard stops (peak |angle| <= 85 deg).",
                "Avoids hard-stop impact. Met by monitor + fail-safe brake + independent limit switch (defence in depth).",
                "Test", "VER-013", 85.0, "deg", "<="),
    Requirement("SYS-014", "N4", "Applied motor voltage shall not exceed 12 V in any scenario.",
                "Supply / driver rating.",
                "Test", "VER-014", 12.0, "V", "<="),
    Requirement("SYS-015", "N4", "Armature current shall not exceed 5 A in any scenario, "
                                 "including blocked rotor and hard reversal.",
                "Driver rating of a notional 5 A; the driver current limit is set at 4 A (80% derating).",
                "Test", "VER-015", 5.0, "A", "<="),
    Requirement("SYS-016", "N3", "Steady-state motor command activity (RMS, 1-3 s of the nominal run) shall be <= 0.5 V.",
                "Encoder quantisation amplified by the derivative term causes chatter: heating, audible noise, wear.",
                "Test", "VER-016", 0.5, "V rms", "<="),
]


def requirement_by_id(req_id):
    return next(r for r in REQUIREMENTS if r.id == req_id)
