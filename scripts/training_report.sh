#!/bin/bash
# training_report.sh — Generate a training status report with figures and email it
# Usage: ./training_report.sh recipient@email.com
#
# Designed to be run via cron at 9:00 AM daily during v3 training.

set -euo pipefail

EMAIL="${1:?Usage: $0 recipient@email.com}"
PROJECT_DIR="/Users/joseluissaorin/Dropbox/Jose Luis Hijo/Dev/EigenDialectos"
STEP_LOG="$PROJECT_DIR/outputs/eigen3/step_log.jsonl"
REPORT_DIR="$PROJECT_DIR/outputs/reports"
mkdir -p "$REPORT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
REPORT_FILE="$REPORT_DIR/training_report_${TIMESTAMP}.txt"
FIGURE_DIR="$REPORT_DIR/figures_${TIMESTAMP}"
mkdir -p "$FIGURE_DIR"

# 1. Check if training is still running
PID=90158
if ps -p $PID > /dev/null 2>&1; then
    ELAPSED=$(ps -p $PID -o etime= | xargs)
    STATUS="RUNNING (PID $PID, elapsed $ELAPSED)"
else
    STATUS="NOT RUNNING (PID $PID not found)"
fi

# 2. Generate metrics, figures, and estimations in one Python pass
METRICS=$(python3 << PYEOF
import json, datetime, warnings
import numpy as np
warnings.filterwarnings("ignore")

# --- Load data ---
log_path = "$STEP_LOG"
with open(log_path) as f:
    lines = [json.loads(l) for l in f]

if not lines:
    print("No data in step log.")
    raise SystemExit(0)

d = lines[-1]
fig_dir = "$FIGURE_DIR"

def get_con(entry):
    return entry['loss'].get('contrastive', entry['loss'].get('con', 0.0))

def get_ctr(entry):
    return entry['loss'].get('center', entry['loss'].get('ctr', 0.0))

# --- Extract arrays ---
steps = np.array([l['global_step'] for l in lines])
mlm = np.array([l['loss']['mlm'] for l in lines])
cls = np.array([l['loss']['cls'] for l in lines])
con = np.array([get_con(l) for l in lines])
ctr = np.array([get_ctr(l) for l in lines])
total = np.array([l['loss']['total'] for l in lines])
sps = np.array([l['samples_per_sec'] for l in lines])
mps = np.array([l['mps_mb'] for l in lines])

# --- Key constants ---
steps_per_epoch = d['epoch_steps_total']
total_steps = d['global_steps_total']
ep3_step = 2 * steps_per_epoch
ep4_step = 3 * steps_per_epoch
current_step = d['global_step']
contrastive_active = current_step >= ep3_step
xbm_active = current_step >= ep4_step

rate = current_step / d['run_elapsed_s']
remaining_s = (total_steps - current_step) / rate
finish = datetime.datetime.now() + datetime.timedelta(seconds=remaining_s)

# Epoch boundaries for plotting
epoch_boundaries = [i * steps_per_epoch for i in range(1, 10)]

# ===========================================================================
# TEXT REPORT
# ===========================================================================
print(f"CURRENT STATE (as of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})")
print(f"  Epoch: {d['epoch']}/10")
print(f"  Step: {current_step}/{total_steps} ({d['progress_pct']:.1f}%)")
print(f"  Elapsed: {d['run_elapsed_s']/3600:.1f}h")
print(f"  ETA: {remaining_s/3600:.1f}h ({finish.strftime('%A %B %d, %H:%M')})")
print(f"  Throughput: {d['samples_per_sec']:.1f} sm/s (avg {sps.mean():.1f}, min {sps.min():.1f}, max {sps.max():.1f})")
print(f"  MPS memory: {d['mps_mb']:.0f} MB")
print()
print(f"LOSSES")
print(f"  MLM:          {d['loss']['mlm']:.3f}")
print(f"  CLS (ArcFace):{d['loss']['cls']:.3f}")
cv = get_con(d)
print(f"  Contrastive:  {cv:.4f}  {'(ACTIVE)' if contrastive_active else '(disabled — pretrain phase)'}")
print(f"  Center:       {get_ctr(d):.4f}")
print(f"  Total:        {d['loss']['total']:.3f}")
print()
print(f"PHASE STATUS")
print(f"  Contrastive: {'ACTIVE since step ' + str(ep3_step) if contrastive_active else 'activates at step ' + str(ep3_step)}")
print(f"  XBM queue:   {'ACTIVE since step ' + str(ep4_step) if xbm_active else 'activates at step ' + str(ep4_step)}")
if not xbm_active and contrastive_active:
    xbm_remaining = (ep4_step - current_step) / rate / 3600
    xbm_time = datetime.datetime.now() + datetime.timedelta(hours=xbm_remaining)
    print(f"  XBM activates in {xbm_remaining:.1f}h ({xbm_time.strftime('%a %b %d %H:%M')})")

# --- Contrastive trajectory ---
if contrastive_active:
    con_mask = (steps >= ep3_step) & (con > 0.0)
    con_steps_arr = steps[con_mask]
    con_vals = con[con_mask]

    if len(con_vals) >= 2:
        first_con = con_vals[0]
        last_con = con_vals[-1]
        peak_con = con_vals.max()
        peak_step = con_steps_arr[con_vals.argmax()]
        n_active = con_steps_arr[-1] - con_steps_arr[0]

        quarter = max(2, len(con_vals) // 4)
        recent_delta = con_vals[-1] - con_vals[-quarter]
        recent_n = con_steps_arr[-1] - con_steps_arr[-quarter]

        print(f"\nCONTRASTIVE TRAJECTORY")
        print(f"  Initial value (step {int(con_steps_arr[0])}): {first_con:.4f}")
        print(f"  Peak (step {int(peak_step)}):          {peak_con:.4f}")
        print(f"  Current (step {int(con_steps_arr[-1])}):      {last_con:.4f}")
        print(f"  Total active steps:               {n_active}")
        print(f"  Drop from peak: {last_con - peak_con:+.4f}")
        print(f"  Recent trend (last {recent_n} steps): {recent_delta:+.4f}")

        if last_con < peak_con - 0.05:
            print(f"  Status: DESCENDING (from peak {peak_con:.3f} to {last_con:.3f})")
        elif abs(last_con - peak_con) < 0.05 and n_active > 2000:
            print(f"  Status: PLATEAU (near peak for {n_active} steps)")
        elif con_steps_arr[-1] - ep3_step < 500:
            print(f"  Status: INITIALIZING")
        elif recent_delta < -0.01:
            print(f"  Status: DESCENDING (recent trend negative)")
        elif abs(recent_delta) < 0.02:
            print(f"  Status: STABLE")
        else:
            print(f"  Status: SLOW DESCENT or PLATEAU")
    elif len(con_vals) == 0:
        print(f"\nCONTRASTIVE TRAJECTORY")
        print(f"  Contrastive just activated — no nonzero values logged yet")
        con_vals = np.array([])
        con_steps_arr = np.array([])

# --- V2 comparison ---
v2_con = 7.92
if contrastive_active and len(con_vals) >= 1:
    print(f"\nV2 COMPARISON")
    print(f"  v2 contrastive at this stage: {v2_con:.2f} (stuck, never descended)")
    print(f"  v3 contrastive now:           {last_con:.4f}")
    print(f"  v3 is {v2_con - last_con:.2f} lower than v2 was")

# --- Recent trend (last 1000 steps) ---
recent_mask = steps >= current_step - 1000
if recent_mask.sum() >= 2:
    r_mlm = mlm[recent_mask]
    r_cls = cls[recent_mask]
    r_steps = steps[recent_mask]
    n_recent = r_steps[-1] - r_steps[0]
    mlm_delta = r_mlm[-1] - r_mlm[0]
    cls_delta = r_cls[-1] - r_cls[0]
    mlm_noise = r_mlm.max() - r_mlm.min()
    cls_noise = r_cls.max() - r_cls.min()

    print(f"\nRECENT TREND (last {n_recent} steps)")
    print(f"  MLM: {r_mlm[-1]:.3f}  (delta {mlm_delta:+.4f}, range {r_mlm.min():.3f}–{r_mlm.max():.3f})")
    print(f"  CLS: {r_cls[-1]:.3f}  (delta {cls_delta:+.4f}, range {r_cls.min():.3f}–{r_cls.max():.3f})")
    if abs(mlm_delta) < mlm_noise * 0.5:
        print(f"  MLM delta is within normal batch noise ({mlm_noise:.3f} range)")
    if abs(cls_delta) < cls_noise * 0.5:
        print(f"  CLS delta is within normal batch noise ({cls_noise:.3f} range)")

# ===========================================================================
# ESTIMATIONS
# ===========================================================================
print(f"\n{'='*60}")
print(f"ESTIMATIONS & PROJECTIONS")
print(f"{'='*60}")

# MLM projection using log-decay fit on stable regime (step >= 8000)
stable_mask = steps >= 8000
if stable_mask.sum() >= 10:
    s_fit = steps[stable_mask].astype(float)
    m_fit = mlm[stable_mask]
    log_s = np.log(s_fit)
    A = np.vstack([log_s, np.ones(len(log_s))]).T
    mlm_coeffs, _, _, _ = np.linalg.lstsq(A, m_fit, rcond=None)

    milestones = {
        'End epoch 3': 3 * steps_per_epoch,
        'End epoch 5': 5 * steps_per_epoch,
        'End epoch 7': 7 * steps_per_epoch,
        'End training': total_steps,
    }
    if not xbm_active:
        milestones['XBM activation (ep4)'] = ep4_step

    print(f"\n  MLM projections (log-decay model):")
    for label, step in sorted(milestones.items(), key=lambda x: x[1]):
        pred = mlm_coeffs[0] * np.log(step) + mlm_coeffs[1]
        eta_h = (step - current_step) / rate / 3600
        eta_time = datetime.datetime.now() + datetime.timedelta(hours=eta_h)
        print(f"    {label:25s} (step {step:>7d}, {eta_time.strftime('%a %b %d %H:%M')}): MLM ≈ {pred:.3f}")

# CLS projection
    c_fit = cls[stable_mask]
    inv_sqrt = 1.0 / np.sqrt(s_fit)
    A2 = np.vstack([inv_sqrt, np.ones(len(inv_sqrt))]).T
    cls_coeffs, _, _, _ = np.linalg.lstsq(A2, c_fit, rcond=None)

    print(f"\n  CLS projections (inverse-sqrt model):")
    for label, step in sorted(milestones.items(), key=lambda x: x[1]):
        pred = cls_coeffs[0] / np.sqrt(step) + cls_coeffs[1]
        eta_h = (step - current_step) / rate / 3600
        eta_time = datetime.datetime.now() + datetime.timedelta(hours=eta_h)
        print(f"    {label:25s} (step {step:>7d}, {eta_time.strftime('%a %b %d %H:%M')}): CLS ≈ {pred:.3f}")

# Contrastive projection if enough data
if contrastive_active and len(con_vals) >= 20:
    # Fit linear on log(con) for exponential decay estimate
    # Also fit simple linear for conservative estimate
    con_s = con_steps_arr.astype(float)
    con_v = con_vals

    # Linear fit
    A_lin = np.vstack([con_s, np.ones(len(con_s))]).T
    lin_coeffs, _, _, _ = np.linalg.lstsq(A_lin, con_v, rcond=None)

    # Rate of descent per 1000 steps
    rate_per_1k = lin_coeffs[0] * 1000

    print(f"\n  Contrastive projections (linear model, pre-XBM):")
    print(f"    Current descent rate: {rate_per_1k:+.4f} per 1000 steps")
    if rate_per_1k < 0:
        print(f"    (Note: rate will likely accelerate when XBM activates with 4096 negatives)")

    for label, step in sorted(milestones.items(), key=lambda x: x[1]):
        pred = lin_coeffs[0] * step + lin_coeffs[1]
        eta_h = (step - current_step) / rate / 3600
        eta_time = datetime.datetime.now() + datetime.timedelta(hours=eta_h)
        print(f"    {label:25s} (step {step:>7d}, {eta_time.strftime('%a %b %d %H:%M')}): CON ≈ {max(0, pred):.3f}")

    # Estimate when contrastive hits milestones
    if lin_coeffs[0] < 0:
        print(f"\n  Contrastive milestone estimates (linear, conservative — XBM will accelerate):")
        for target, label in [(2.5, "CON < 2.5"), (2.0, "CON < 2.0"), (1.0, "CON < 1.0"), (0.5, "CON < 0.5")]:
            step_at = (target - lin_coeffs[1]) / lin_coeffs[0]
            if step_at > current_step and step_at < total_steps * 2:
                eta_h = (step_at - current_step) / rate / 3600
                eta_time = datetime.datetime.now() + datetime.timedelta(hours=eta_h)
                epoch_at = step_at / steps_per_epoch + 1
                print(f"    {label}: step ~{int(step_at):,} (epoch ~{epoch_at:.1f}, {eta_time.strftime('%a %b %d %H:%M')})")
            elif step_at <= current_step:
                print(f"    {label}: already reached")
            else:
                print(f"    {label}: not reachable at current rate (XBM should fix this)")

# Throughput stats
print(f"\n  Throughput statistics:")
print(f"    Current: {sps[-1]:.1f} sm/s")
print(f"    Mean:    {sps.mean():.1f} sm/s")
print(f"    Std:     {sps.std():.1f} sm/s")
print(f"    Min:     {sps.min():.1f} sm/s (step {int(steps[sps.argmin()])})")
print(f"    Max:     {sps.max():.1f} sm/s (step {int(steps[sps.argmax()])})")

# ===========================================================================
# FIGURES
# ===========================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Smooth function for noisy data
    def smooth(y, window=20):
        if len(y) < window:
            return y
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode='valid')

    def smooth_steps(x, window=20):
        if len(x) < window:
            return x
        return x[window-1:]

    # Color scheme
    C_MLM = '#3278C8'
    C_CLS = '#DC5028'
    C_CON = '#28A03C'
    C_CTR = '#B464C8'
    C_PHASE = '#CCCCCC'
    C_XBM = '#FF8C00'

    # -----------------------------------------------------------------------
    # FIGURE 1: Full training overview (3-panel)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'EigenDialectos v3 Training — Epoch {d["epoch"]}/10, Step {current_step:,} ({d["progress_pct"]:.1f}%)',
                 fontsize=14, fontweight='bold')

    # Phase shading
    for ax in [ax1, ax2, ax3]:
        ax.axvspan(0, ep3_step, alpha=0.06, color='blue', label='_Phase 1')
        ax.axvline(ep3_step, color=C_PHASE, ls='--', lw=1, alpha=0.7)
        ax.axvline(ep4_step, color=C_XBM, ls='--', lw=1, alpha=0.5)
        for eb in epoch_boundaries:
            if eb != ep3_step:
                ax.axvline(eb, color='#EEEEEE', ls=':', lw=0.5)
        ax.set_xlim(0, total_steps)
        ax.grid(True, alpha=0.15)

    # Panel 1: MLM
    s_sm, m_sm = smooth_steps(steps, 30), smooth(mlm, 30)
    ax1.plot(s_sm, m_sm, color=C_MLM, lw=1.5, label='MLM')
    ax1.set_ylabel('MLM Loss', fontweight='bold', color=C_MLM)
    ax1.set_ylim(max(0, mlm.min() - 0.3), min(mlm.max() + 0.5, 11))
    # Projection
    if stable_mask.sum() >= 10:
        proj_steps = np.linspace(current_step, total_steps, 100)
        proj_mlm = mlm_coeffs[0] * np.log(proj_steps) + mlm_coeffs[1]
        ax1.plot(proj_steps, proj_mlm, color=C_MLM, ls='--', lw=1, alpha=0.5, label='MLM projected')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.text(ep3_step, ax1.get_ylim()[1]*0.95, ' SupCon ON', fontsize=8, color='gray', va='top')
    ax1.text(ep4_step, ax1.get_ylim()[1]*0.95, ' XBM ON', fontsize=8, color=C_XBM, va='top')

    # Panel 2: CLS
    s_sm2, c_sm = smooth_steps(steps, 30), smooth(cls, 30)
    ax2.plot(s_sm2, c_sm, color=C_CLS, lw=1.5, label='CLS (ArcFace)')
    ax2.set_ylabel('CLS Loss', fontweight='bold', color=C_CLS)
    ax2.set_ylim(max(0, cls.min() - 0.2), min(cls.max() + 0.5, 12))
    if stable_mask.sum() >= 10:
        proj_cls = cls_coeffs[0] / np.sqrt(proj_steps) + cls_coeffs[1]
        ax2.plot(proj_steps, proj_cls, color=C_CLS, ls='--', lw=1, alpha=0.5, label='CLS projected')
    ax2.legend(loc='upper right', fontsize=9)

    # Panel 3: Contrastive + Center
    if contrastive_active and len(con_vals) >= 2:
        # Plot full con array (zeros in phase 1, values in phase 2)
        con_plot_mask = con > 0
        ax3.plot(steps[con_plot_mask], con[con_plot_mask], color=C_CON, lw=1.5, label='Contrastive (SupCon)')
        # Projection
        if len(con_vals) >= 20:
            proj_con = lin_coeffs[0] * proj_steps + lin_coeffs[1]
            proj_con = np.maximum(proj_con, 0)
            ax3.plot(proj_steps, proj_con, color=C_CON, ls='--', lw=1, alpha=0.5, label='CON projected (linear)')
        # v2 baseline
        ax3.axhline(v2_con, color='red', ls=':', lw=1, alpha=0.4, label=f'v2 stuck at {v2_con}')
    ctr_plot_mask = ctr > 0
    if ctr_plot_mask.any():
        ax3.plot(steps[ctr_plot_mask], ctr[ctr_plot_mask], color=C_CTR, lw=1, alpha=0.7, label='Center loss')
    ax3.set_ylabel('Contrastive Loss', fontweight='bold', color=C_CON)
    ax3.set_xlabel('Global optimizer step')
    ax3.set_ylim(bottom=0)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.text(ep3_step, ax3.get_ylim()[1]*0.95, ' SupCon ON', fontsize=8, color='gray', va='top')
    ax3.text(ep4_step, ax3.get_ylim()[1]*0.95, ' XBM ON', fontsize=8, color=C_XBM, va='top')

    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    plt.tight_layout()
    fig.savefig(f'{fig_dir}/01_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # FIGURE 2: Contrastive zoom (if active)
    # -----------------------------------------------------------------------
    if contrastive_active and len(con_vals) >= 5:
        fig2, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.set_title('Contrastive Loss Since Activation (Zoomed)', fontsize=13, fontweight='bold')

        # Raw + smoothed
        ax.plot(con_steps_arr, con_vals, color=C_CON, lw=0.5, alpha=0.3, label='Raw')
        if len(con_vals) >= 10:
            cs_sm = smooth(con_vals, min(10, len(con_vals)//3))
            cs_st = smooth_steps(con_steps_arr, min(10, len(con_vals)//3))
            ax.plot(cs_st, cs_sm, color=C_CON, lw=2, label='Smoothed')

        # Projection
        if len(con_vals) >= 20:
            future_steps = np.linspace(con_steps_arr[-1], min(con_steps_arr[-1] + 30000, total_steps), 50)
            future_con = lin_coeffs[0] * future_steps + lin_coeffs[1]
            future_con = np.maximum(future_con, 0)
            ax.plot(future_steps, future_con, color=C_CON, ls='--', lw=1.5, alpha=0.5, label='Linear projection')

        # XBM activation line
        if not xbm_active:
            ax.axvline(ep4_step, color=C_XBM, ls='--', lw=2, label=f'XBM activates (step {ep4_step:,})')

        # Target zones
        ax.axhspan(0, 1.0, alpha=0.05, color='green')
        ax.axhspan(1.0, 2.0, alpha=0.03, color='yellow')
        ax.text(ax.get_xlim()[1]*0.98, 0.5, 'Excellent', ha='right', fontsize=9, color='green', alpha=0.5)
        ax.text(ax.get_xlim()[1]*0.98, 1.5, 'Good', ha='right', fontsize=9, color='olive', alpha=0.5)

        # v2 line
        ax.axhline(v2_con, color='red', ls=':', lw=1, alpha=0.3, label=f'v2 baseline ({v2_con})')

        ax.set_xlabel('Global optimizer step')
        ax.set_ylabel('SupCon Loss')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

        plt.tight_layout()
        fig2.savefig(f'{fig_dir}/02_contrastive_zoom.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)

    # -----------------------------------------------------------------------
    # FIGURE 3: Throughput and memory
    # -----------------------------------------------------------------------
    fig3, (ax_t, ax_m) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig3.suptitle('System Health', fontsize=13, fontweight='bold')

    s_sm3, sps_sm = smooth_steps(steps, 30), smooth(sps, 30)
    ax_t.plot(s_sm3, sps_sm, color='#4488CC', lw=1.5)
    ax_t.axhline(sps.mean(), color='gray', ls=':', lw=1, alpha=0.5)
    ax_t.set_ylabel('Samples/sec', fontweight='bold')
    ax_t.set_ylim(0, sps.max() * 1.2)
    ax_t.grid(True, alpha=0.15)
    for eb in epoch_boundaries:
        ax_t.axvline(eb, color='#EEEEEE', ls=':', lw=0.5)

    s_sm4, mps_sm = smooth_steps(steps, 30), smooth(mps, 30)
    ax_m.plot(s_sm4, mps_sm, color='#CC6644', lw=1.5)
    ax_m.set_ylabel('MPS Memory (MB)', fontweight='bold')
    ax_m.set_xlabel('Global optimizer step')
    ax_m.grid(True, alpha=0.15)
    ax_m.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))
    for eb in epoch_boundaries:
        ax_m.axvline(eb, color='#EEEEEE', ls=':', lw=0.5)

    plt.tight_layout()
    fig3.savefig(f'{fig_dir}/03_system_health.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # -----------------------------------------------------------------------
    # FIGURE 4: Estimation timeline
    # -----------------------------------------------------------------------
    fig4, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_title('Training Timeline & Milestones', fontsize=13, fontweight='bold')

    # Timeline bar
    progress = current_step / total_steps
    ax.barh(0, progress, height=0.4, color=C_MLM, alpha=0.7, label=f'Completed ({progress*100:.1f}%)')
    ax.barh(0, 1-progress, left=progress, height=0.4, color='#EEEEEE', alpha=0.7, label=f'Remaining')

    # Milestone markers
    markers = [
        (ep3_step/total_steps, 'SupCon ON', C_CON),
        (ep4_step/total_steps, 'XBM ON', C_XBM),
    ]
    for i in range(1, 11):
        markers.append((i*steps_per_epoch/total_steps, f'Ep {i+1}', '#AAAAAA'))

    for pos, label, color in markers:
        if pos <= 1.0:
            ax.axvline(pos, color=color, ls='--', lw=1, alpha=0.6)
            ax.text(pos, 0.35, f' {label}', fontsize=7, color=color, rotation=45, va='bottom')

    # Current position
    ax.axvline(progress, color='black', lw=2)
    ax.text(progress, -0.35, f'NOW\nStep {current_step:,}', fontsize=8, ha='center', va='top', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel('Training progress')
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(True, axis='x', alpha=0.15)

    plt.tight_layout()
    fig4.savefig(f'{fig_dir}/04_timeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)

    print(f"\nFIGURES: 4 charts saved to {fig_dir}/")

except ImportError:
    print(f"\nFIGURES: matplotlib not available, skipping charts")
except Exception as e:
    print(f"\nFIGURES: Error generating charts: {e}")
PYEOF
)

# 3. Build the report
cat > "$REPORT_FILE" << EOF
=== EigenDialectos v3 Training Report ===
Generated: $(date '+%Y-%m-%d %H:%M:%S')
Process: $STATUS

$METRICS
EOF

echo "Report saved to $REPORT_FILE"

# 4. Generate analysis with Claude
ANALYSIS=$(cat "$REPORT_FILE" | claude -p \
  "You are monitoring an ML training run for the EigenDialectos v3 project (8 Spanish dialect embeddings, BETO+LoRA). \
Key architecture context: \
- Two-phase training: epochs 1-2 are pretrain (MLM+CLS only, con=0 by design), epochs 3+ activate SupCon contrastive. \
- XBM (cross-batch memory, 4096 negatives) activates at epoch 4. Before XBM, contrastive only has 24 in-batch negatives, so descent is expected to be slow. \
- ArcFace classifier (s=30, m=0.3) means CLS starts high (~11) and converges around 0.5-1.0. \
- v2 baseline had contrastive stuck at 7.92 forever. Any descent below 3.5 means v3 is working. \
- MLM floor is ~2.0 (shared capacity with other objectives + dialectal corpus noise). \
- Center loss is only active for the first 2000 contrastive steps, then disables (0.0 is normal after that). \
- Batch noise: MLM and CLS fluctuate ~0.05-0.10 between logged steps. Small deltas within this range are noise, not trends. \
Given this report, write a brief (5-8 sentences) analysis: Is training healthy? Any real concerns (not noise)? What to look for next? Be specific and actionable." \
  2>/dev/null || echo "Claude analysis unavailable.")

# 5. Compose and send email with attachments
SUBJECT="[EigenDialectos] v3 Training Report — $(date '+%b %d %H:%M')"

# Write body to temp file to avoid quoting issues with spaces in paths
BODY_FILE=$(mktemp)
cat > "$BODY_FILE" << EOF
$(<"$REPORT_FILE")

--- Claude Analysis ---
$ANALYSIS

---
Automated report from training_report.sh
See attached figures for visual trajectory and projections.
EOF

# Build attachment flags as array (paths contain spaces)
ATTACH_ARGS=()
for fig in "$FIGURE_DIR"/*.png; do
    [ -f "$fig" ] && ATTACH_ARGS+=("--attach=$fig")
done

gog gmail send \
  --to="$EMAIL" \
  --subject="$SUBJECT" \
  --body="$(cat "$BODY_FILE")" \
  "${ATTACH_ARGS[@]}" \
  --force \
  --no-input

rm -f "$BODY_FILE"
echo "Report emailed to $EMAIL (with $(ls "$FIGURE_DIR"/*.png 2>/dev/null | wc -l | xargs) figures)"
