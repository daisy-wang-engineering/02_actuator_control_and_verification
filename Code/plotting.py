"""Plots: the basic response plots plus the report figure set."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NAVY, TEAL, AMBER, RED, GREY, LIGHT = "#1F3A5F", "#2A9D8F", "#E9A23B", "#C8423A", "#7A8794", "#E8EDF2"
NEED_COLORS = {"N1": NAVY, "N2": TEAL, "N3": AMBER, "N4": RED}
NEED_NAMES = {"N1": "Accuracy", "N2": "Response", "N3": "Robustness", "N4": "Safety"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=200); plt.close(fig)


def save_plots(result, output_dir="results"):
    """Basic position and controller-output plots (kept from the original project)."""
    t = result["time_s"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, np.rad2deg(result["target_rad"]), "--", color=GREY, label="Commanded position")
    ax.plot(t, np.rad2deg(result["true_angle_rad"]), color=NAVY, label="True simulated position")
    ax.plot(t, np.rad2deg(result["measured_angle_rad"]), color=TEAL, alpha=0.55, linewidth=0.8, label="Encoder measurement")
    ax.set(xlabel="Time (s)", ylabel="Position (deg)", title="Single axis actuator position response")
    ax.legend(); _save(fig, output_dir, "position_response.png")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, result["voltage_v"], color=NAVY, label="Motor command")
    ax.set(xlabel="Time (s)", ylabel="Voltage (V)", title="Controller output")
    ax.legend(); _save(fig, output_dir, "controller_output.png")


def fig_dashboard(results, out):
    """Requirement limit-utilisation chart: every bar left of 100% = requirement met."""
    from .requirements import requirement_by_id
    rows = []
    for r in results:
        util = 100.0 * (r.measured / r.limit if r.sense == "<=" else r.limit / r.measured)
        rows.append((r, util, requirement_by_id(r.req_id)))
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    y = np.arange(len(rows))[::-1]
    for yi, (r, util, req) in zip(y, rows):
        ax.barh(yi, util, color=NEED_COLORS[req.need], height=0.62, alpha=0.9 if r.passed else 1.0,
                edgecolor=RED if not r.passed else "none", linewidth=2)
        sym = "\u2264" if r.sense == "<=" else "\u2265"
        ax.text(104, yi, f"{r.measured:.3g} {r.unit}", va="center", ha="left", fontsize=7.8, color="#222", fontweight="bold")
        ax.text(148, yi, f"{sym} {r.limit:g}", va="center", ha="right", fontsize=7.8, color="#666")
    ax.axvline(100, color="black", lw=1.2)
    ax.text(104, len(rows) - 0.25, "measured", fontsize=7.5, color="#222"); ax.text(148, len(rows) - 0.25, "limit", fontsize=7.5, color="#666", ha="right")
    ax.text(50, len(rows) - 0.25, "within limit \u2192", fontsize=7.5, color="#666", ha="center")
    ax.set_yticks(y); ax.set_yticklabels([f"{r.req_id}  {_short(r.req_id)}" for r, _, _ in rows], fontsize=8)
    ax.set_xlim(0, 150); ax.set_ylim(-0.7, len(rows) + 0.1); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Share of allowed limit used (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=c, label=NEED_NAMES[k]) for k, c in NEED_COLORS.items()], loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=4, fontsize=8)
    ax.set_title("All 16 requirements verified against their limits", pad=14)
    _save(fig, out, "fig_dashboard.png")


SHORT = {"SYS-001": "Steady-state error", "SYS-002": "Overshoot", "SYS-003": "Settling time", "SYS-004": "Bandwidth",
         "SYS-005": "Phase margin", "SYS-006": "Latency tolerance", "SYS-007": "Repeatability", "SYS-008": "Load: peak deviation",
         "SYS-009": "Load: recovery", "SYS-010": "Tolerance Monte Carlo", "SYS-011": "Feedback-loss fail-safe",
         "SYS-012": "Stuck-sensor detection", "SYS-013": "Hard-stop avoidance", "SYS-014": "Voltage limit", "SYS-015": "Current limit", "SYS-016": "Command chatter (noise)"}


def _short(req_id):
    return SHORT[req_id]


def fig_step_and_disturbance(nominal, disturbed, out):
    t = nominal["time_s"]
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.0), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    ax = axs[0, 0]
    a = np.rad2deg(nominal["true_angle_rad"])
    ax.axhspan(29, 31, color=TEAL, alpha=0.15, label="\u00b11\u00b0 settling band")
    ax.plot(t, np.rad2deg(nominal["target_rad"]), "--", color=GREY, label="Command")
    ax.plot(t, a, color=NAVY, lw=1.8, label="True angle")
    ax.set_xlim(0, 1.0); ax.set_ylim(-2, 34); ax.set_ylabel("Angle (deg)")
    ts = t[np.where(np.abs(30 - a) > 1)[0][-1] + 1]
    ax.axvline(ts, color=TEAL, lw=1); ax.annotate(f"settled in {ts:.2f} s\n(limit 1.0 s)", (ts, 18), xytext=(ts + 0.12, 10), fontsize=8,
                                                 arrowprops=dict(arrowstyle="->", color=GREY))
    ax.legend(loc="lower right", fontsize=7.5); ax.set_title("Nominal 30\u00b0 step")
    ax = axs[1, 0]
    ax.plot(t, nominal["voltage_v"], color=NAVY, label="Command (V)")
    ax.plot(t, nominal["current_a"], color=AMBER, label="Current (A)")
    ax.axhline(4.0, color=AMBER, ls=":", lw=1); ax.text(0.98, 4.3, "4 A driver limit", ha="right", fontsize=7, color=AMBER)
    ax.set_ylim(-3, 14); ax.set_ylabel("V / A"); ax.set_xlabel("Time (s)"); ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    t2 = disturbed["time_s"]
    ax = axs[0, 1]
    ang = np.rad2deg(disturbed["true_angle_rad"])
    ax.axvspan(1, 2, color=RED, alpha=0.08); ax.text(1.5, 25.0, "0.025 N\u00b7m load", ha="center", fontsize=8, color=RED)
    ax.axhline(30 - 2.0, color=RED, ls=":", lw=1); ax.text(2.98, 28.15, "\u22122\u00b0 limit (SYS-008)", ha="right", fontsize=7, color=RED)
    ax.plot(t2, np.rad2deg(disturbed["target_rad"]), "--", color=GREY)
    ax.plot(t2, ang, color=NAVY, lw=1.8)
    ax.set_ylim(23, 32); ax.set_title("Load disturbance (1\u20132 s)")
    pk = np.max(np.abs(30 - ang[(t2 >= 1) & (t2 <= 2)]))
    ax.annotate(f"peak deviation {pk:.2f}\u00b0", (1.05, 30 - pk), xytext=(0.15, 24.2), fontsize=8, arrowprops=dict(arrowstyle="->", color=GREY))
    ax = axs[1, 1]
    ax.plot(t2, disturbed["voltage_v"], color=NAVY); ax.set_xlabel("Time (s)"); ax.set_ylabel("Command (V)")
    _save(fig, out, "fig_step_disturbance.png")


def fig_bode(gains, out, f_req=2.0):
    from .analysis import frequency_response, loop_metrics
    f, L, T, S = frequency_response(gains)
    m = loop_metrics(gains)
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
    ax = axs[0]
    ax.semilogx(f, 20 * np.log10(np.abs(L)), color=NAVY); ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(m["crossover_hz"], color=TEAL, ls="--", lw=1); ax.set_ylim(-60, 60)
    ax.set_title("Loop gain"); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.set_xlim(0.05, 200)
    ax.text(m["crossover_hz"] * 1.15, 25, f"crossover\n{m['crossover_hz']:.1f} Hz", fontsize=7.5, color=TEAL)
    ax = axs[1]
    ph = np.degrees(np.unwrap(np.angle(L)))
    ax.semilogx(f, ph, color=NAVY); ax.axhline(-180, color=RED, lw=0.8, ls=":")
    ax.axvline(m["crossover_hz"], color=TEAL, ls="--", lw=1); ax.set_xlim(0.05, 200); ax.set_ylim(-270, -60)
    pc = np.interp(m["crossover_hz"], f, ph)
    ax.annotate("", (m["crossover_hz"], -180), (m["crossover_hz"], pc), arrowprops=dict(arrowstyle="<->", color=AMBER))
    ax.text(m["crossover_hz"] * 1.2, (pc - 180) / 2, f"PM {m['phase_margin_deg']:.0f}\u00b0", fontsize=8, color=AMBER, va="center")
    ax.set_title("Loop phase"); ax.set_xlabel("Hz"); ax.set_ylabel("deg")
    ax = axs[2]
    ax.semilogx(f, 20 * np.log10(np.abs(T)), color=NAVY); ax.axhline(-3, color=GREY, ls=":", lw=1)
    ax.axvline(f_req, color=RED, ls="--", lw=1); ax.text(f_req * 1.1, -22, f"requirement\n{f_req:g} Hz", fontsize=7.5, color=RED)
    ax.plot([m["bandwidth_hz"]], [-3], "o", color=TEAL); ax.text(m["bandwidth_hz"] * 1.1, 1.0, f"{m['bandwidth_hz']:.2f} Hz", fontsize=8, color=TEAL)
    ax.set_xlim(0.05, 50); ax.set_ylim(-30, 6); ax.set_title("Closed loop (set-point)"); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    fig.tight_layout(); _save(fig, out, "fig_bode.png")


def fig_latency(rows, delay_margin_ms, out):
    d = [r["delay_ms"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
    ax = axs[0]
    ax.plot(d, [min(r["overshoot_deg"], 40) for r in rows], "o-", color=NAVY, ms=3)
    ax.axhline(3.0, color=RED, ls=":"); ax.text(1, 3.8, "SYS-002 limit 3\u00b0", fontsize=7.5, color=RED)
    ax.axvline(delay_margin_ms, color=TEAL, ls="--"); ax.text(delay_margin_ms + 0.8, 25, f"predicted delay\nmargin {delay_margin_ms:.0f} ms", fontsize=7.5, color=TEAL)
    ax.axvline(20, color=AMBER, ls="--", lw=1); ax.text(19.2, 33, "SYS-006: 20 ms", fontsize=7.5, color=AMBER, ha="right")
    ax.set_xlabel("Added loop latency (ms)"); ax.set_ylabel("Overshoot (deg)"); ax.set_title("Overshoot vs latency")
    ax = axs[1]
    ts = [r["settling_time_s"] if np.isfinite(r["settling_time_s"]) else np.nan for r in rows]
    ax.plot(d, ts, "o-", color=NAVY, ms=3); ax.axhline(1.0, color=RED, ls=":"); ax.text(1, 1.05, "SYS-003 limit 1.0 s", fontsize=7.5, color=RED)
    ax.axvline(delay_margin_ms, color=TEAL, ls="--"); ax.set_ylim(0, 3.1)
    ax.set_xlabel("Added loop latency (ms)"); ax.set_ylabel("Settling time (s)"); ax.set_title("Settling vs latency")
    fig.tight_layout(); _save(fig, out, "fig_latency.png")


def fig_design_space(grid, picks, out):
    kps = sorted({g["kp"] for g in grid}); kds = sorted({g["kd"] for g in grid})
    def field(key):
        z = np.zeros((len(kds), len(kps)))
        for g in grid:
            z[kds.index(g["kd"]), kps.index(g["kp"])] = g[key] if np.isfinite(g[key]) else 99
        return z
    X, Y = np.meshgrid(kps, kds)
    bw, dm, os20, dist = field("bandwidth_hz"), field("delay_margin_ms"), field("overshoot_20ms_deg"), field("dist_peak_deg")
    feas = ((bw >= 2.0) & (os20 <= 3.0) & (dist <= 2.0) & (dm >= 20) & (field("overshoot_deg") <= 3) & (field("phase_margin_deg") >= 45)).astype(float)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.contourf(X, Y, feas, levels=[0.5, 1.5], colors=[TEAL], alpha=0.25)
    ax.contour(X, Y, bw, levels=[2.5], colors=[GREY], linewidths=1.2, linestyles="--")
    ax.plot([], [], color=GREY, ls="--", label="original bandwidth \u2265 2.5 Hz (feasible: right/below)")
    for z, lv, c, lab, ls in [(bw, 2.0, NAVY, "bandwidth \u2265 2.0 Hz", "-"), (os20, 3.0, AMBER, "overshoot \u2264 3\u00b0 at 20 ms latency", "-"),
                              (dist, 2.0, RED, "load deviation \u2264 2\u00b0", "-")]:
        cs = ax.contour(X, Y, z, levels=[lv], colors=[c], linewidths=1.6, linestyles=ls)
    ax.plot([], [], color=NAVY, label="bandwidth \u2265 2.0 Hz (feasible: right/below)")
    ax.plot([], [], color=AMBER, label="overshoot \u2264 3\u00b0 at 20 ms latency (feasible: above)")
    ax.plot([], [], color=RED, label="load deviation \u2264 2\u00b0 (feasible: right)")
    ax.add_patch(Patch(color=TEAL, alpha=0.25, label="feasible region")) if False else None
    for name, (kp, kd), c, dx, dy in picks:
        ax.plot(kp, kd, "o", color=c, ms=8, mec="white", mew=1.2, zorder=5); ax.annotate(name, (kp, kd), xytext=(kp + dx, kd + dy), fontsize=8, color=c, fontweight="bold")
    ax.set_xlabel("$K_p$ (V/rad)"); ax.set_ylabel("$K_d$ (V s/rad)"); ax.set_title("Design space: where every requirement can be met  (teal)")
    ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.0, -0.18), ncol=1)
    _save(fig, out, "fig_design_space.png")


def fig_monte_carlo(mc, out):
    """mc: dict name -> (rows, pass_rate, color). Scatter of bandwidth vs delay margin."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axvspan(2.0, 6, ymin=0, ymax=1, color=TEAL, alpha=0.08)
    for name, (rows, pr, c) in mc.items():
        ax.scatter([r["bandwidth_hz"] for r in rows], [r["delay_margin_ms"] for r in rows], s=9, color=c, alpha=0.65, label=f"{name}  ({pr:.0f}% meet SYS-001\u2013004)")
    ax.axvline(2.0, color=NAVY, lw=1.4); ax.text(2.03, ax.get_ylim()[1] * 0.97, "SYS-004: 2.0 Hz", fontsize=7.5, color=NAVY, va="top")
    ax.axvline(2.5, color=GREY, lw=1, ls="--"); ax.text(2.53, 17, "original 2.5 Hz", fontsize=7.5, color=GREY)
    ax.axhline(20, color=AMBER, lw=1.2, ls="--"); ax.text(ax.get_xlim()[1] * 0.99, 20.6, "SYS-006: 20 ms", fontsize=7.5, color=AMBER, ha="right")
    ax.set_xlabel("Closed-loop bandwidth (Hz)"); ax.set_ylabel("Delay margin (ms)")
    ax.set_title("200 randomised hardware/load variations per design"); ax.legend(fontsize=7.5, loc="upper right")
    _save(fig, out, "fig_monte_carlo.png")


def fig_faults(runs, layers, onsets, drop, out):
    fig = plt.figure(figsize=(7.6, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 0.8], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    t = runs["none"]["time_s"]
    ax.plot(t, np.rad2deg(runs["none"]["true_angle_rad"]), color=RED, lw=1.8, label="no protection")
    ax.plot(t, np.rad2deg(runs["full"]["true_angle_rad"]), color=TEAL, lw=1.8, label="monitor + brake + switch")
    ax.axhline(90, color="black", lw=1); ax.text(0.98, 91.5, "hard stop 90\u00b0", ha="right", fontsize=7.5)
    ft = runs["full"]["fault_time_s"]; ax.axvline(ft, color=TEAL, ls=":", lw=1); ax.text(ft + 0.02, 55, f"fault latched\n{ft*1e3:.0f} ms", fontsize=7.5, color=TEAL)
    ax.set_xlim(0, 1); ax.set_ylim(-5, 100); ax.set_ylabel("Angle (deg)"); ax.set_xlabel("Time (s)")
    ax.set_title("Encoder freezes at t = 0 (cmd 30\u00b0)"); ax.legend(fontsize=7.5, loc="center right")
    ax = fig.add_subplot(gs[0, 1])
    for (name, rows), c, ls in zip(layers.items(), [RED, AMBER, NAVY, TEAL], ["-", "-", "-", "-"]):
        ax.plot([r["onset_s"] for r in rows], [r["peak_deg"] for r in rows], color=c, lw=1.6, label=name)
    ax.axhline(90, color="black", lw=1); ax.axhline(85, color=RED, ls=":", lw=1); ax.text(0.595, 77, "SYS-013 limit 85\u00b0", ha="right", fontsize=7.5, color=RED)
    ax.set_xlabel("Fault onset time (s)"); ax.set_ylabel("Peak excursion (deg)"); ax.set_ylim(0, 100)
    ax.set_title("Defence in depth, 61 onset times"); ax.legend(fontsize=6.8, loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax = fig.add_subplot(gs[1, :])
    t = drop["time_s"]
    ax.plot(t, drop["voltage_v"], color=NAVY, label="command (V)")
    ax.fill_between(t, 0, 12 * drop["brake_engaged"], color=AMBER, alpha=0.35, step="pre", label="brake engaged")
    ax.set_xlim(0, 0.12); ax.set_ylim(-1, 13); ax.set_xlabel("Time (s)"); ax.set_ylabel("V")
    ax.set_title("100% feedback loss: 0 V on every sample, brake engaged at 30 ms (20 ms timeout + 10 ms mechanical)"); ax.legend(fontsize=7.5, loc="center right")
    _save(fig, out, "fig_faults.png")
