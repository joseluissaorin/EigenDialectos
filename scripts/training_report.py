#!/usr/bin/env python3
"""training_report.py — Generate a training status report with figures and email it.

Usage: python3 training_report.py recipient@email.com
Designed to be run via cron at 9:00 AM daily during v3 training.
"""

import json
import sys
import subprocess
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
STEP_LOG = PROJECT_DIR / "outputs" / "eigen3" / "step_log.jsonl"
REPORT_DIR = PROJECT_DIR / "outputs" / "reports"
PID = 93558
V2_CON = 7.92

CLAUDE_PROMPT = (
    "You are monitoring an ML training run for the EigenDialectos v3 project "
    "(8 Spanish dialect embeddings, BETO+LoRA). "
    "Key architecture context: "
    "- Two-phase training: epochs 1-2 are pretrain (MLM+CLS only, con=0 by design), "
    "epochs 3+ activate SupCon contrastive with MoCo momentum queue. "
    "- MoCo replaces the broken XBM cross-batch memory. Momentum encoder (m=0.999) "
    "provides consistent queue entries. Queue of 4096 entries activates at epoch 4, "
    "ramps from 128 to 4096 over 5000 steps. "
    "- ArcFace classifier now operates on projected features (384-dim) instead of "
    "raw pooled (768-dim). Previous run had ArcFace collapse (all logits=-30). "
    "- ArcFace (s=30, m=0.3) means CLS starts high (~11) and should converge ~0.5-1.0. "
    "- Previous run (no MoCo) plateaued at contrastive=2.65 with only 24 in-batch negatives. "
    "MoCo should push contrastive significantly lower. "
    "- MLM floor is ~2.0 (shared capacity with other objectives + dialectal corpus noise). "
    "- Center loss is only active for the first 2000 contrastive steps, then disables (0.0 is normal). "
    "- Batch noise: MLM and CLS fluctuate ~0.05-0.10 between logged steps. Small deltas are noise. "
    "Given this report, write a brief (5-8 sentences) analysis: Is training healthy? "
    "Any real concerns (not noise)? What to look for next? Be specific and actionable."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_process(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime="],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"RUNNING (PID {pid}, elapsed {result.stdout.strip()})"
    except Exception:
        pass
    return f"NOT RUNNING (PID {pid} not found)"


def get_con(entry: dict) -> float:
    return entry["loss"].get("contrastive", entry["loss"].get("con", 0.0))


def get_ctr(entry: dict) -> float:
    return entry["loss"].get("center", entry["loss"].get("ctr", 0.0))


def smooth(y: np.ndarray, window: int = 20) -> np.ndarray:
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode="valid")


def smooth_steps(x: np.ndarray, window: int = 20) -> np.ndarray:
    if len(x) < window:
        return x
    return x[window - 1 :]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_step_log() -> list[dict]:
    with open(STEP_LOG) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def build_text_report(lines: list[dict], status: str) -> tuple[str, dict]:
    """Return (report_text, context_for_figures)."""
    d = lines[-1]
    now = datetime.now()

    steps = np.array([l["global_step"] for l in lines])
    mlm = np.array([l["loss"]["mlm"] for l in lines])
    cls_ = np.array([l["loss"]["cls"] for l in lines])
    con = np.array([get_con(l) for l in lines])
    ctr = np.array([get_ctr(l) for l in lines])
    sps = np.array([l["samples_per_sec"] for l in lines])
    mps = np.array([l["mps_mb"] for l in lines])

    steps_per_epoch = d["epoch_steps_total"]
    total_steps = d["global_steps_total"]
    ep3_step = 2 * steps_per_epoch
    ep4_step = 3 * steps_per_epoch
    current_step = d["global_step"]
    contrastive_active = current_step >= ep3_step
    moco_active = current_step >= ep4_step

    rate = current_step / d["run_elapsed_s"]
    remaining_s = (total_steps - current_step) / rate
    finish = now + timedelta(seconds=remaining_s)
    epoch_boundaries = [i * steps_per_epoch for i in range(1, 10)]

    out = []
    p = out.append

    p(f"=== EigenDialectos v3 Training Report ===")
    p(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"Process: {status}")
    p("")
    p(f"CURRENT STATE (as of {now.strftime('%Y-%m-%d %H:%M')})")
    p(f"  Epoch: {d['epoch']}/10")
    p(f"  Step: {current_step}/{total_steps} ({d['progress_pct']:.1f}%)")
    p(f"  Elapsed: {d['run_elapsed_s']/3600:.1f}h")
    p(f"  ETA: {remaining_s/3600:.1f}h ({finish.strftime('%A %B %d, %H:%M')})")
    p(f"  Throughput: {d['samples_per_sec']:.1f} sm/s (avg {sps.mean():.1f}, min {sps.min():.1f}, max {sps.max():.1f})")
    p(f"  MPS memory: {d['mps_mb']:.0f} MB")
    p("")
    p("LOSSES")
    p(f"  MLM:          {d['loss']['mlm']:.3f}")
    p(f"  CLS (ArcFace):{d['loss']['cls']:.3f}")
    cv = get_con(d)
    p(f"  Contrastive:  {cv:.4f}  {'(ACTIVE)' if contrastive_active else '(disabled — pretrain phase)'}")
    p(f"  Center:       {get_ctr(d):.4f}")
    p(f"  Total:        {d['loss']['total']:.3f}")
    p("")
    p("PHASE STATUS")
    p(f"  Contrastive: {'ACTIVE since step ' + str(ep3_step) if contrastive_active else 'activates at step ' + str(ep3_step)}")
    p(f"  MoCo queue:   {'ACTIVE since step ' + str(ep4_step) if moco_active else 'activates at step ' + str(ep4_step)}")
    if not moco_active and contrastive_active:
        moco_remaining = (ep4_step - current_step) / rate / 3600
        moco_time = now + timedelta(hours=moco_remaining)
        p(f"  MoCo activates in {moco_remaining:.1f}h ({moco_time.strftime('%a %b %d %H:%M')})")

    # Contrastive trajectory
    con_steps_arr = np.array([])
    con_vals = np.array([])
    last_con = 0.0

    if contrastive_active:
        con_mask = (steps >= ep3_step) & (con > 0.0)
        con_steps_arr = steps[con_mask]
        con_vals = con[con_mask]

        if len(con_vals) >= 2:
            first_con = con_vals[0]
            last_con = con_vals[-1]
            peak_con = con_vals.max()
            peak_step = con_steps_arr[con_vals.argmax()]
            n_active = int(con_steps_arr[-1] - con_steps_arr[0])

            quarter = max(2, len(con_vals) // 4)
            recent_delta = con_vals[-1] - con_vals[-quarter]
            recent_n = int(con_steps_arr[-1] - con_steps_arr[-quarter])

            p(f"\nCONTRASTIVE TRAJECTORY")
            p(f"  Initial value (step {int(con_steps_arr[0])}): {first_con:.4f}")
            p(f"  Peak (step {int(peak_step)}):          {peak_con:.4f}")
            p(f"  Current (step {int(con_steps_arr[-1])}):      {last_con:.4f}")
            p(f"  Total active steps:               {n_active}")
            p(f"  Drop from peak: {last_con - peak_con:+.4f}")
            p(f"  Recent trend (last {recent_n} steps): {recent_delta:+.4f}")

            if last_con < peak_con - 0.05:
                p(f"  Status: DESCENDING (from peak {peak_con:.3f} to {last_con:.3f})")
            elif abs(last_con - peak_con) < 0.05 and n_active > 2000:
                p(f"  Status: PLATEAU (near peak for {n_active} steps)")
            elif con_steps_arr[-1] - ep3_step < 500:
                p(f"  Status: INITIALIZING")
            elif recent_delta < -0.01:
                p(f"  Status: DESCENDING (recent trend negative)")
            elif abs(recent_delta) < 0.02:
                p(f"  Status: STABLE")
            else:
                p(f"  Status: SLOW DESCENT or PLATEAU")
        elif len(con_vals) == 0:
            p(f"\nCONTRASTIVE TRAJECTORY")
            p(f"  Contrastive just activated — no nonzero values logged yet")

    # V2 comparison
    if contrastive_active and len(con_vals) >= 1:
        p(f"\nV2 COMPARISON")
        p(f"  v2 contrastive at this stage: {V2_CON:.2f} (stuck, never descended)")
        p(f"  v3 contrastive now:           {last_con:.4f}")
        p(f"  v3 is {V2_CON - last_con:.2f} lower than v2 was")

    # Recent trend
    recent_mask = steps >= current_step - 1000
    if recent_mask.sum() >= 2:
        r_mlm = mlm[recent_mask]
        r_cls = cls_[recent_mask]
        r_steps = steps[recent_mask]
        n_recent = int(r_steps[-1] - r_steps[0])
        mlm_delta = r_mlm[-1] - r_mlm[0]
        cls_delta = r_cls[-1] - r_cls[0]
        mlm_noise = r_mlm.max() - r_mlm.min()
        cls_noise = r_cls.max() - r_cls.min()

        p(f"\nRECENT TREND (last {n_recent} steps)")
        p(f"  MLM: {r_mlm[-1]:.3f}  (delta {mlm_delta:+.4f}, range {r_mlm.min():.3f}\u2013{r_mlm.max():.3f})")
        p(f"  CLS: {r_cls[-1]:.3f}  (delta {cls_delta:+.4f}, range {r_cls.min():.3f}\u2013{r_cls.max():.3f})")
        if abs(mlm_delta) < mlm_noise * 0.5:
            p(f"  MLM delta is within normal batch noise ({mlm_noise:.3f} range)")
        if abs(cls_delta) < cls_noise * 0.5:
            p(f"  CLS delta is within normal batch noise ({cls_noise:.3f} range)")

    # --- Estimations ---
    p(f"\n{'='*60}")
    p("ESTIMATIONS & PROJECTIONS")
    p(f"{'='*60}")

    milestones = {
        "End epoch 3": 3 * steps_per_epoch,
        "End epoch 5": 5 * steps_per_epoch,
        "End epoch 7": 7 * steps_per_epoch,
        "End training": total_steps,
    }
    if not moco_active:
        milestones["MoCo activation (ep4)"] = ep4_step

    mlm_coeffs = cls_coeffs = lin_coeffs = None

    stable_mask = steps >= 8000
    if stable_mask.sum() >= 10:
        s_fit = steps[stable_mask].astype(float)
        m_fit = mlm[stable_mask]
        log_s = np.log(s_fit)
        A = np.vstack([log_s, np.ones(len(log_s))]).T
        mlm_coeffs, _, _, _ = np.linalg.lstsq(A, m_fit, rcond=None)

        p(f"\n  MLM projections (log-decay model):")
        for label, step in sorted(milestones.items(), key=lambda x: x[1]):
            pred = mlm_coeffs[0] * np.log(step) + mlm_coeffs[1]
            eta_time = now + timedelta(seconds=(step - current_step) / rate)
            p(f"    {label:25s} (step {step:>7d}, {eta_time.strftime('%a %b %d %H:%M')}): MLM \u2248 {pred:.3f}")

        c_fit = cls_[stable_mask]
        inv_sqrt = 1.0 / np.sqrt(s_fit)
        A2 = np.vstack([inv_sqrt, np.ones(len(inv_sqrt))]).T
        cls_coeffs, _, _, _ = np.linalg.lstsq(A2, c_fit, rcond=None)

        p(f"\n  CLS projections (inverse-sqrt model):")
        for label, step in sorted(milestones.items(), key=lambda x: x[1]):
            pred = cls_coeffs[0] / np.sqrt(step) + cls_coeffs[1]
            eta_time = now + timedelta(seconds=(step - current_step) / rate)
            p(f"    {label:25s} (step {step:>7d}, {eta_time.strftime('%a %b %d %H:%M')}): CLS \u2248 {pred:.3f}")

    if contrastive_active and len(con_vals) >= 20:
        con_s = con_steps_arr.astype(float)
        A_lin = np.vstack([con_s, np.ones(len(con_s))]).T
        lin_coeffs, _, _, _ = np.linalg.lstsq(A_lin, con_vals, rcond=None)
        rate_per_1k = lin_coeffs[0] * 1000

        p(f"\n  Contrastive projections (linear model, pre-MoCo):")
        p(f"    Current descent rate: {rate_per_1k:+.4f} per 1000 steps")
        if rate_per_1k < 0:
            p(f"    (Note: rate will likely accelerate when MoCo activates with 4096 negatives)")

        for label, step in sorted(milestones.items(), key=lambda x: x[1]):
            pred = lin_coeffs[0] * step + lin_coeffs[1]
            eta_time = now + timedelta(seconds=(step - current_step) / rate)
            p(f"    {label:25s} (step {step:>7d}, {eta_time.strftime('%a %b %d %H:%M')}): CON \u2248 {max(0, pred):.3f}")

        if lin_coeffs[0] < 0:
            p(f"\n  Contrastive milestone estimates (linear, conservative \u2014 MoCo will accelerate):")
            for target, label in [(2.5, "CON < 2.5"), (2.0, "CON < 2.0"), (1.0, "CON < 1.0"), (0.5, "CON < 0.5")]:
                step_at = (target - lin_coeffs[1]) / lin_coeffs[0]
                if current_step < step_at < total_steps * 2:
                    eta_time = now + timedelta(seconds=(step_at - current_step) / rate)
                    epoch_at = step_at / steps_per_epoch + 1
                    p(f"    {label}: step ~{int(step_at):,} (epoch ~{epoch_at:.1f}, {eta_time.strftime('%a %b %d %H:%M')})")
                elif step_at <= current_step:
                    p(f"    {label}: already reached")
                else:
                    p(f"    {label}: not reachable at current rate (MoCo should fix this)")

    p(f"\n  Throughput statistics:")
    p(f"    Current: {sps[-1]:.1f} sm/s")
    p(f"    Mean:    {sps.mean():.1f} sm/s")
    p(f"    Std:     {sps.std():.1f} sm/s")
    p(f"    Min:     {sps.min():.1f} sm/s (step {int(steps[sps.argmin()])})")
    p(f"    Max:     {sps.max():.1f} sm/s (step {int(steps[sps.argmax()])})")

    ctx = dict(
        steps=steps, mlm=mlm, cls_=cls_, con=con, ctr=ctr, sps=sps, mps=mps,
        steps_per_epoch=steps_per_epoch, total_steps=total_steps,
        ep3_step=ep3_step, ep4_step=ep4_step, current_step=current_step,
        contrastive_active=contrastive_active, moco_active=moco_active,
        epoch_boundaries=epoch_boundaries, d=d,
        con_steps_arr=con_steps_arr, con_vals=con_vals,
        mlm_coeffs=mlm_coeffs, cls_coeffs=cls_coeffs, lin_coeffs=lin_coeffs,
        stable_mask=stable_mask,
    )
    return "\n".join(out), ctx


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def generate_figures(ctx: dict, fig_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not available, skipping charts", file=sys.stderr)
        return []

    C_MLM, C_CLS, C_CON = "#3278C8", "#DC5028", "#28A03C"
    C_CTR, C_PHASE, C_MOCO = "#B464C8", "#CCCCCC", "#FF8C00"

    steps = ctx["steps"]
    mlm, cls_, con, ctr = ctx["mlm"], ctx["cls_"], ctx["con"], ctx["ctr"]
    sps, mps = ctx["sps"], ctx["mps"]
    steps_per_epoch = ctx["steps_per_epoch"]
    ep3_step, ep4_step = ctx["ep3_step"], ctx["ep4_step"]
    total_steps = ctx["total_steps"]
    current_step = ctx["current_step"]
    epoch_boundaries = ctx["epoch_boundaries"]
    contrastive_active = ctx["contrastive_active"]
    con_steps_arr, con_vals = ctx["con_steps_arr"], ctx["con_vals"]
    mlm_coeffs, cls_coeffs, lin_coeffs = ctx["mlm_coeffs"], ctx["cls_coeffs"], ctx["lin_coeffs"]
    stable_mask = ctx["stable_mask"]
    d = ctx["d"]
    saved: list[Path] = []

    def phase_decor(ax):
        ax.axvspan(0, ep3_step, alpha=0.06, color="blue")
        ax.axvline(ep3_step, color=C_PHASE, ls="--", lw=1, alpha=0.7)
        ax.axvline(ep4_step, color=C_MOCO, ls="--", lw=1, alpha=0.5)
        for eb in epoch_boundaries:
            if eb != ep3_step:
                ax.axvline(eb, color="#EEEEEE", ls=":", lw=0.5)
        ax.set_xlim(0, total_steps)
        ax.grid(True, alpha=0.15)

    # --- Figure 1: Overview ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"EigenDialectos v3 Training \u2014 Epoch {d['epoch']}/10, "
        f"Step {current_step:,} ({d['progress_pct']:.1f}%)",
        fontsize=14, fontweight="bold",
    )
    for ax in [ax1, ax2, ax3]:
        phase_decor(ax)

    s_sm, m_sm = smooth_steps(steps, 30), smooth(mlm, 30)
    ax1.plot(s_sm, m_sm, color=C_MLM, lw=1.5, label="MLM")
    ax1.set_ylabel("MLM Loss", fontweight="bold", color=C_MLM)
    ax1.set_ylim(max(0, mlm.min() - 0.3), min(mlm.max() + 0.5, 11))
    if mlm_coeffs is not None:
        proj_steps = np.linspace(current_step, total_steps, 100)
        ax1.plot(proj_steps, mlm_coeffs[0] * np.log(proj_steps) + mlm_coeffs[1],
                 color=C_MLM, ls="--", lw=1, alpha=0.5, label="MLM projected")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.text(ep3_step, ax1.get_ylim()[1] * 0.95, " SupCon ON", fontsize=8, color="gray", va="top")
    ax1.text(ep4_step, ax1.get_ylim()[1] * 0.95, " MoCo ON", fontsize=8, color=C_MOCO, va="top")

    s_sm2, c_sm = smooth_steps(steps, 30), smooth(cls_, 30)
    ax2.plot(s_sm2, c_sm, color=C_CLS, lw=1.5, label="CLS (ArcFace)")
    ax2.set_ylabel("CLS Loss", fontweight="bold", color=C_CLS)
    ax2.set_ylim(max(0, cls_.min() - 0.2), min(cls_.max() + 0.5, 12))
    if cls_coeffs is not None:
        proj_steps = np.linspace(current_step, total_steps, 100)
        ax2.plot(proj_steps, cls_coeffs[0] / np.sqrt(proj_steps) + cls_coeffs[1],
                 color=C_CLS, ls="--", lw=1, alpha=0.5, label="CLS projected")
    ax2.legend(loc="upper right", fontsize=9)

    if contrastive_active and len(con_vals) >= 2:
        con_plot_mask = con > 0
        ax3.plot(steps[con_plot_mask], con[con_plot_mask], color=C_CON, lw=1.5, label="Contrastive (SupCon)")
        if lin_coeffs is not None:
            proj_steps = np.linspace(current_step, total_steps, 100)
            proj_con = np.maximum(lin_coeffs[0] * proj_steps + lin_coeffs[1], 0)
            ax3.plot(proj_steps, proj_con, color=C_CON, ls="--", lw=1, alpha=0.5, label="CON projected")
        ax3.axhline(V2_CON, color="red", ls=":", lw=1, alpha=0.4, label=f"v2 stuck at {V2_CON}")
    ctr_mask = ctr > 0
    if ctr_mask.any():
        ax3.plot(steps[ctr_mask], ctr[ctr_mask], color=C_CTR, lw=1, alpha=0.7, label="Center loss")
    ax3.set_ylabel("Contrastive Loss", fontweight="bold", color=C_CON)
    ax3.set_xlabel("Global optimizer step")
    ax3.set_ylim(bottom=0)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.text(ep3_step, ax3.get_ylim()[1] * 0.95, " SupCon ON", fontsize=8, color="gray", va="top")
    ax3.text(ep4_step, ax3.get_ylim()[1] * 0.95, " MoCo ON", fontsize=8, color=C_MOCO, va="top")
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    plt.tight_layout()
    path = fig_dir / "01_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- Figure 2: Contrastive zoom ---
    if contrastive_active and len(con_vals) >= 5:
        fig2, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.set_title("Contrastive Loss Since Activation (Zoomed)", fontsize=13, fontweight="bold")
        ax.plot(con_steps_arr, con_vals, color=C_CON, lw=0.5, alpha=0.3, label="Raw")
        if len(con_vals) >= 10:
            w = min(10, len(con_vals) // 3)
            ax.plot(smooth_steps(con_steps_arr, w), smooth(con_vals, w),
                    color=C_CON, lw=2, label="Smoothed")
        if lin_coeffs is not None:
            future = np.linspace(con_steps_arr[-1], min(con_steps_arr[-1] + 30000, total_steps), 50)
            ax.plot(future, np.maximum(lin_coeffs[0] * future + lin_coeffs[1], 0),
                    color=C_CON, ls="--", lw=1.5, alpha=0.5, label="Linear projection")
        if not ctx["moco_active"]:
            ax.axvline(ep4_step, color=C_MOCO, ls="--", lw=2, label=f"MoCo activates (step {ep4_step:,})")
        ax.axhspan(0, 1.0, alpha=0.05, color="green")
        ax.axhspan(1.0, 2.0, alpha=0.03, color="yellow")
        ax.text(ax.get_xlim()[1] * 0.98, 0.5, "Excellent", ha="right", fontsize=9, color="green", alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.98, 1.5, "Good", ha="right", fontsize=9, color="olive", alpha=0.5)
        ax.axhline(V2_CON, color="red", ls=":", lw=1, alpha=0.3, label=f"v2 baseline ({V2_CON})")
        ax.set_xlabel("Global optimizer step")
        ax.set_ylabel("SupCon Loss")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        plt.tight_layout()
        path = fig_dir / "02_contrastive_zoom.png"
        fig2.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        saved.append(path)

    # --- Figure 3: System health ---
    fig3, (ax_t, ax_m) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig3.suptitle("System Health", fontsize=13, fontweight="bold")
    ax_t.plot(smooth_steps(steps, 30), smooth(sps, 30), color="#4488CC", lw=1.5)
    ax_t.axhline(sps.mean(), color="gray", ls=":", lw=1, alpha=0.5)
    ax_t.set_ylabel("Samples/sec", fontweight="bold")
    ax_t.set_ylim(0, sps.max() * 1.2)
    ax_t.grid(True, alpha=0.15)
    for eb in epoch_boundaries:
        ax_t.axvline(eb, color="#EEEEEE", ls=":", lw=0.5)
    ax_m.plot(smooth_steps(steps, 30), smooth(mps, 30), color="#CC6644", lw=1.5)
    ax_m.set_ylabel("MPS Memory (MB)", fontweight="bold")
    ax_m.set_xlabel("Global optimizer step")
    ax_m.grid(True, alpha=0.15)
    ax_m.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for eb in epoch_boundaries:
        ax_m.axvline(eb, color="#EEEEEE", ls=":", lw=0.5)
    plt.tight_layout()
    path = fig_dir / "03_system_health.png"
    fig3.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    saved.append(path)

    # --- Figure 4: Timeline ---
    fig4, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_title("Training Timeline & Milestones", fontsize=13, fontweight="bold")
    progress = current_step / total_steps
    ax.barh(0, progress, height=0.4, color=C_MLM, alpha=0.7, label=f"Completed ({progress*100:.1f}%)")
    ax.barh(0, 1 - progress, left=progress, height=0.4, color="#EEEEEE", alpha=0.7, label="Remaining")
    markers = [
        (ep3_step / total_steps, "SupCon ON", C_CON),
        (ep4_step / total_steps, "MoCo ON", C_MOCO),
    ]
    for i in range(1, 11):
        markers.append((i * steps_per_epoch / total_steps, f"Ep {i+1}", "#AAAAAA"))
    for pos, label, color in markers:
        if pos <= 1.0:
            ax.axvline(pos, color=color, ls="--", lw=1, alpha=0.6)
            ax.text(pos, 0.35, f" {label}", fontsize=7, color=color, rotation=45, va="bottom")
    ax.axvline(progress, color="black", lw=2)
    ax.text(progress, -0.35, f"NOW\nStep {current_step:,}", fontsize=8, ha="center", va="top", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Training progress")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(True, axis="x", alpha=0.15)
    plt.tight_layout()
    path = fig_dir / "04_timeline.png"
    fig4.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------
def get_claude_analysis(report_text: str) -> str:
    try:
        result = subprocess.run(
            ["claude", "-p", CLAUDE_PROMPT],
            input=report_text, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "Claude analysis unavailable."


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
def send_email(email: str, subject: str, body: str, attachments: list[Path]):
    import os
    env = {**os.environ, "GOG_ACCOUNT": "default"}

    cmd = [
        "gog", "gmail", "send",
        "--to", email,
        "--subject", subject,
        "--body", body,
        "--account", "default",
        "--force",
        "--no-input",
    ]
    for path in attachments:
        cmd.append(f"--attach={path}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        if result.returncode != 0:
            print(f"Email failed. stderr: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Email exception: {e}", file=sys.stderr)
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} recipient@email.com", file=sys.stderr)
        sys.exit(1)

    email = sys.argv[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORT_DIR / f"training_report_{timestamp}.txt"
    fig_dir = REPORT_DIR / f"figures_{timestamp}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process status
    status = check_process(PID)
    print(f"Process: {status}")

    # 2. Load data & build report
    lines = load_step_log()
    if not lines:
        print("No data in step log.", file=sys.stderr)
        sys.exit(0)

    report_text, ctx = build_text_report(lines, status)
    report_file.write_text(report_text)
    print(f"Report saved to {report_file}")

    # 3. Generate figures
    figures = generate_figures(ctx, fig_dir)
    print(f"Generated {len(figures)} figures in {fig_dir}")

    # 4. Claude analysis
    print("Generating Claude analysis...")
    analysis = get_claude_analysis(report_text)

    # 5. Compose & send email
    body = (
        f"{report_text}\n\n"
        f"--- Claude Analysis ---\n{analysis}\n\n"
        f"---\nAutomated report from training_report.py\n"
        f"See attached figures for visual trajectory and projections."
    )
    subject = f"[EigenDialectos] v3 Training Report \u2014 {datetime.now().strftime('%b %d %H:%M')}"

    if send_email(email, subject, body, figures):
        print(f"Report emailed to {email} (with {len(figures)} figures)")
    else:
        print(f"EMAIL FAILED — report saved locally at {report_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
