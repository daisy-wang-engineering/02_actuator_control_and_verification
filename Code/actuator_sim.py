"""Run the nominal simulation, the full verification suite and write results/.

    python run_project.py            # ~1.5 min (includes 200-run Monte Carlo)
    python run_project.py --quick    # 20-run Monte Carlo for a fast smoke test
"""
import argparse
import csv
import json
from pathlib import Path
from actuator_sim.simulation import run_simulation, SimulationConfig, calculate_metrics, V2_GAINS
from actuator_sim.verification import run_verification_suite
from actuator_sim.requirements import REQUIREMENTS, NEEDS
from actuator_sim.plotting import save_plots


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    simulation = run_simulation(SimulationConfig(), pid_gains=V2_GAINS)
    metrics = calculate_metrics(simulation)
    save_plots(simulation, results_dir)

    tests = run_verification_suite(V2_GAINS, mc_runs=20 if args.quick else 200)
    req = {r.id: r for r in REQUIREMENTS}
    with (results_dir / "verification_report.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Requirement", "Need", "Method", "Verification", "Criterion", "Measured", "Margin %", "Result", "Notes"])
        for t in tests:
            w.writerow([t.req_id, req[t.req_id].need, t.method, t.ver_id, f"{t.sense} {t.limit:g} {t.unit}",
                        f"{t.measured:.4g} {t.unit}", f"{t.margin_pct:.0f}", "PASS" if t.passed else "FAIL", t.notes])
    with (results_dir / "traceability_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Need", "Need statement", "Requirement", "Requirement text", "Rationale", "Verification", "Method"])
        needs = {n.id: n for n in NEEDS}
        for r in REQUIREMENTS:
            w.writerow([r.need, needs[r.need].statement, r.id, r.text, r.rationale, r.ver_id, r.method])
    (results_dir / "verification_summary.json").write_text(json.dumps(
        [dict(req_id=t.req_id, ver_id=t.ver_id, measured=t.measured, limit=t.limit, unit=t.unit, sense=t.sense,
              passed=t.passed, margin_pct=t.margin_pct, notes=t.notes) for t in tests], indent=2))

    print("Nominal simulation metrics (final gains):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nVerification:")
    for t in tests:
        print(f"  {'PASS' if t.passed else 'FAIL'} {t.req_id} / {t.ver_id}: {t.measured:.4g} {t.unit} "
              f"({t.sense} {t.limit:g}), margin {t.margin_pct:.0f}% - {t.notes}")
    print(f"\n{sum(t.passed for t in tests)}/{len(tests)} requirements verified.")
    return 0 if all(t.passed for t in tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
