"""Main export module for EigenDialectos pipeline results.

Exports all pipeline artefacts in JSON, CSV, NumPy, and HTML formats so that
results are both machine-readable and human-inspectable.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
from html import escape as html_escape
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from eigendialectos.constants import DIALECT_NAMES, DialectCode
from eigendialectos.types import (
    DialectalSpectrum,
    EigenDecomposition,
    EmbeddingMatrix,
    ExperimentResult,
    TransformationMatrix,
)
from eigendialectos.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure(path: Path) -> Path:
    """Create parent directories and return *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _dialect_label(code: DialectCode) -> str:
    """Human-readable name for a dialect code, with fallback."""
    return DIALECT_NAMES.get(code, code.value)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that gracefully handles NumPy scalars and arrays."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            v = float(o)
            if math.isnan(v) or math.isinf(v):
                return str(v)
            return v
        if isinstance(o, np.complexfloating):
            return {"real": float(o.real), "imag": float(o.imag)}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, DialectCode):
            return o.value
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _save_json(data: Any, path: Path) -> Path:
    p = _ensure(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    return p


def _save_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write a list of dicts to CSV."""
    if not rows:
        return _ensure(path)
    p = _ensure(path)
    fieldnames = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p


def _save_npy(arr: npt.NDArray[Any], path: Path) -> Path:
    p = _ensure(path)
    np.save(p, arr)
    return p


# ---------------------------------------------------------------------------
# Per-type export helpers (public, usable standalone)
# ---------------------------------------------------------------------------

def export_transforms(
    transforms: dict[DialectCode, TransformationMatrix],
    output_dir: Path,
) -> list[Path]:
    """Export transformation matrices as .npy files."""
    created: list[Path] = []
    npy_dir = output_dir / "transforms"
    for code, tm in transforms.items():
        p = _save_npy(tm.data, npy_dir / f"{code.value}.npy")
        created.append(p)
        logger.info("Saved transform %s  ->  %s", code.value, p)
    return created


def export_eigendecomposition(
    eigendecompositions: dict[DialectCode, EigenDecomposition],
    output_dir: Path,
) -> list[Path]:
    """Export eigenvalues and eigenvectors as .npy, .json, and .csv."""
    created: list[Path] = []

    # --- .npy ---
    for code, ed in eigendecompositions.items():
        p_val = _save_npy(ed.eigenvalues, output_dir / "eigenvalues" / f"{code.value}.npy")
        p_vec = _save_npy(ed.eigenvectors, output_dir / "eigenvectors" / f"{code.value}.npy")
        created.extend([p_val, p_vec])

    # --- JSON ---
    json_data: dict[str, Any] = {}
    for code, ed in eigendecompositions.items():
        evs = ed.eigenvalues
        json_data[code.value] = {
            "dialect_name": _dialect_label(code),
            "count": len(evs),
            "eigenvalues": [
                {
                    "index": int(i),
                    "real": float(evs[i].real),
                    "imag": float(evs[i].imag),
                    "magnitude": float(np.abs(evs[i])),
                    "phase": float(np.angle(evs[i])),
                }
                for i in range(len(evs))
            ],
        }
    p = _save_json(json_data, output_dir / "eigenvalues.json")
    created.append(p)

    # --- CSV ---
    rows: list[dict[str, Any]] = []
    for code, ed in eigendecompositions.items():
        evs = ed.eigenvalues
        for i, ev in enumerate(evs):
            rows.append({
                "dialect": code.value,
                "dialect_name": _dialect_label(code),
                "index": i,
                "real_part": float(ev.real),
                "imag_part": float(ev.imag),
                "magnitude": float(np.abs(ev)),
                "phase": float(np.angle(ev)),
            })
    p = _save_csv(rows, output_dir / "eigenvalues.csv")
    created.append(p)

    return created


def export_spectra(
    spectra: dict[DialectCode, DialectalSpectrum],
    output_dir: Path,
) -> list[Path]:
    """Export spectral profiles as JSON and CSV."""
    created: list[Path] = []

    # --- JSON (spectral_results.json) ---
    json_data: dict[str, Any] = {}
    for code, sp in spectra.items():
        ce = sp.cumulative_energy
        json_data[code.value] = {
            "dialect_name": _dialect_label(code),
            "entropy": float(sp.entropy),
            "num_components": len(sp.eigenvalues_sorted),
            "eigenvalues_sorted": sp.eigenvalues_sorted.tolist(),
            "cumulative_energy": ce.tolist(),
        }
    p = _save_json(json_data, output_dir / "spectral_results.json")
    created.append(p)

    # --- CSV (spectra.csv) ---
    rows: list[dict[str, Any]] = []
    for code, sp in spectra.items():
        ce = sp.cumulative_energy
        for i, ev in enumerate(sp.eigenvalues_sorted):
            rows.append({
                "dialect": code.value,
                "dialect_name": _dialect_label(code),
                "index": i,
                "eigenvalue_magnitude": float(ev),
                "cumulative_energy": float(ce[i]) if i < len(ce) else "",
                "entropy": float(sp.entropy),
            })
    p = _save_csv(rows, output_dir / "spectra.csv")
    created.append(p)

    return created


def export_distances(
    distance_matrix: npt.NDArray[np.float64],
    dialect_order: list[DialectCode],
    output_dir: Path,
) -> list[Path]:
    """Export pairwise distance matrix as .npy, JSON, and CSV."""
    created: list[Path] = []

    # --- .npy ---
    p = _save_npy(distance_matrix, output_dir / "distance_matrix.npy")
    created.append(p)

    # --- JSON (dict-of-dicts) ---
    json_data: dict[str, dict[str, float]] = {}
    for i, src in enumerate(dialect_order):
        inner: dict[str, float] = {}
        for j, tgt in enumerate(dialect_order):
            inner[tgt.value] = float(distance_matrix[i, j])
        json_data[src.value] = inner
    p = _save_json(json_data, output_dir / "distance_matrix.json")
    created.append(p)

    # --- CSV (square matrix with headers) ---
    csv_path = _ensure(output_dir / "distance_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [""] + [c.value for c in dialect_order]
        writer.writerow(header)
        for i, src in enumerate(dialect_order):
            row = [src.value] + [f"{distance_matrix[i, j]:.6f}" for j in range(len(dialect_order))]
            writer.writerow(row)
    created.append(csv_path)

    return created


def export_experiment_results(
    experiment_results: dict[str, ExperimentResult],
    output_dir: Path,
) -> list[Path]:
    """Export experiment results as JSON and CSV."""
    created: list[Path] = []

    # --- JSON ---
    json_data: dict[str, Any] = {}
    for eid, er in experiment_results.items():
        json_data[eid] = {
            "experiment_id": er.experiment_id,
            "timestamp": er.timestamp,
            "metrics": er.metrics,
            "artifact_paths": er.artifact_paths,
            "config": er.config,
        }
    p = _save_json(json_data, output_dir / "experiment_results.json")
    created.append(p)

    # --- CSV ---
    rows: list[dict[str, Any]] = []
    for eid, er in experiment_results.items():
        row: dict[str, Any] = {
            "experiment_id": er.experiment_id,
            "timestamp": er.timestamp,
        }
        for mk, mv in er.metrics.items():
            row[f"metric_{mk}"] = mv
        rows.append(row)
    if rows:
        # Normalise keys across all rows so DictWriter has full fieldnames
        all_keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        for r in rows:
            for k in all_keys:
                r.setdefault(k, "")
        p = _save_csv(rows, output_dir / "experiment_summary.csv")
        created.append(p)

    return created


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def _build_html_report(
    *,
    spectra: dict[DialectCode, DialectalSpectrum] | None,
    eigendecompositions: dict[DialectCode, EigenDecomposition] | None,
    distance_matrix: npt.NDArray[np.float64] | None,
    dialect_order: list[DialectCode] | None,
    experiment_results: dict[str, ExperimentResult] | None,
    embeddings: dict[DialectCode, EmbeddingMatrix] | None,
    vocab: list[str] | None,
    transforms: dict[DialectCode, TransformationMatrix] | None,
) -> str:
    """Render a self-contained HTML report string."""

    now = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Collect section flags for TOC
    has_summary = True
    has_eigen = eigendecompositions is not None and len(eigendecompositions) > 0
    has_spectra = spectra is not None and len(spectra) > 0
    has_distances = distance_matrix is not None and dialect_order is not None
    has_experiments = experiment_results is not None and len(experiment_results) > 0
    has_transforms = transforms is not None and len(transforms) > 0
    has_embeddings = embeddings is not None and len(embeddings) > 0

    sections: list[str] = []

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------
    css = """\
    <style>
        :root {
            --bg: #fafbfc;
            --card-bg: #ffffff;
            --primary: #2c5282;
            --primary-light: #ebf4ff;
            --accent: #e53e3e;
            --text: #1a202c;
            --text-muted: #718096;
            --border: #e2e8f0;
            --success: #38a169;
            --warning: #d69e2e;
            --font-mono: 'SF Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html { font-size: 15px; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 0.5rem;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        h2 {
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--primary);
            margin-top: 2.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid var(--border);
        }
        h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text);
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .stat-box {
            background: var(--primary-light);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            text-align: center;
        }
        .stat-box .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
            font-family: var(--font-mono);
        }
        .stat-box .label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        th, td {
            padding: 0.5rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background: var(--primary);
            color: #fff;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        tr:nth-child(even) { background: #f7fafc; }
        tr:hover { background: #edf2f7; }
        td.num { text-align: right; font-family: var(--font-mono); font-size: 0.85rem; }
        .toc {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 2rem;
        }
        .toc h2 { margin-top: 0; border: none; font-size: 1rem; }
        .toc ol { padding-left: 1.5rem; }
        .toc li { margin: 0.3rem 0; }
        .toc a { color: var(--primary); text-decoration: none; }
        .toc a:hover { text-decoration: underline; }
        .dist-cell { padding: 0.35rem 0.5rem; text-align: center; font-family: var(--font-mono); font-size: 0.8rem; }
        .metric-badge {
            display: inline-block;
            background: var(--primary-light);
            color: var(--primary);
            border-radius: 4px;
            padding: 0.15rem 0.5rem;
            font-family: var(--font-mono);
            font-size: 0.85rem;
            margin: 0.15rem;
        }
        footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.8rem;
            text-align: center;
        }
        @media (max-width: 768px) {
            body { padding: 1rem; }
            .stat-grid { grid-template-columns: 1fr 1fr; }
        }
    </style>
"""

    # ------------------------------------------------------------------
    # Table of contents
    # ------------------------------------------------------------------
    toc_items: list[tuple[str, str]] = []
    if has_summary:
        toc_items.append(("summary", "Pipeline Summary"))
    if has_eigen:
        toc_items.append(("eigenvalues", "Eigenvalue Decomposition"))
    if has_spectra:
        toc_items.append(("spectra", "Spectral Entropy"))
    if has_distances:
        toc_items.append(("distances", "Distance Matrix"))
    if has_transforms:
        toc_items.append(("transforms", "Transformation Matrices"))
    if has_embeddings:
        toc_items.append(("embeddings", "Embedding Spaces"))
    if has_experiments:
        toc_items.append(("experiments", "Experiment Results"))

    toc_html = '<nav class="toc"><h2>Table of Contents</h2><ol>'
    for anchor, label in toc_items:
        toc_html += f'<li><a href="#{anchor}">{label}</a></li>'
    toc_html += "</ol></nav>"

    # ------------------------------------------------------------------
    # Summary section
    # ------------------------------------------------------------------
    num_dialects = 0
    if eigendecompositions:
        num_dialects = max(num_dialects, len(eigendecompositions))
    if spectra:
        num_dialects = max(num_dialects, len(spectra))
    if transforms:
        num_dialects = max(num_dialects, len(transforms))

    num_experiments = len(experiment_results) if experiment_results else 0
    vocab_size = len(vocab) if vocab else 0

    embed_dim = 0
    if embeddings:
        for em in embeddings.values():
            embed_dim = em.dim
            break

    summary_html = f"""
    <h2 id="summary">Pipeline Summary</h2>
    <div class="stat-grid">
        <div class="stat-box">
            <div class="value">{num_dialects}</div>
            <div class="label">Dialect Varieties</div>
        </div>
        <div class="stat-box">
            <div class="value">{vocab_size:,}</div>
            <div class="label">Vocabulary Size</div>
        </div>
        <div class="stat-box">
            <div class="value">{embed_dim}</div>
            <div class="label">Embedding Dim</div>
        </div>
        <div class="stat-box">
            <div class="value">{num_experiments}</div>
            <div class="label">Experiments</div>
        </div>
    </div>
"""
    sections.append(summary_html)

    # ------------------------------------------------------------------
    # Eigenvalue tables
    # ------------------------------------------------------------------
    if has_eigen:
        assert eigendecompositions is not None  # for type-checker
        eigen_html = '<h2 id="eigenvalues">Eigenvalue Decomposition</h2>'
        for code in sorted(eigendecompositions, key=lambda c: c.value):
            ed = eigendecompositions[code]
            evs = ed.eigenvalues
            top_n = min(20, len(evs))
            sorted_idx = np.argsort(-np.abs(evs))

            eigen_html += f'<h3>{html_escape(_dialect_label(code))} <span style="color:var(--text-muted);font-weight:400;">({code.value})</span></h3>'
            eigen_html += f'<p style="color:var(--text-muted);font-size:0.85rem;">Effective rank: {ed.rank} / {len(evs)} &mdash; showing top {top_n} by magnitude</p>'
            eigen_html += '<div class="card"><table>'
            eigen_html += "<tr><th>#</th><th>Real</th><th>Imag</th><th>Magnitude</th><th>Phase (rad)</th></tr>"
            for rank, idx in enumerate(sorted_idx[:top_n]):
                ev = evs[idx]
                eigen_html += (
                    f"<tr>"
                    f'<td class="num">{rank + 1}</td>'
                    f'<td class="num">{ev.real:+.6f}</td>'
                    f'<td class="num">{ev.imag:+.6f}</td>'
                    f'<td class="num">{np.abs(ev):.6f}</td>'
                    f'<td class="num">{np.angle(ev):+.4f}</td>'
                    f"</tr>"
                )
            eigen_html += "</table></div>"
        sections.append(eigen_html)

    # ------------------------------------------------------------------
    # Spectral entropy comparison
    # ------------------------------------------------------------------
    if has_spectra:
        assert spectra is not None
        spec_html = '<h2 id="spectra">Spectral Entropy</h2>'
        spec_html += '<div class="card"><table>'
        spec_html += "<tr><th>Dialect</th><th>Code</th><th>Entropy</th><th>Components</th><th>Energy at 50%</th><th>Energy at 90%</th></tr>"

        for code in sorted(spectra, key=lambda c: c.value):
            sp = spectra[code]
            ce = sp.cumulative_energy
            n = len(sp.eigenvalues_sorted)

            # Find k for 50% and 90% energy
            k50 = int(np.searchsorted(ce, 0.5)) + 1 if len(ce) > 0 else "-"
            k90 = int(np.searchsorted(ce, 0.9)) + 1 if len(ce) > 0 else "-"

            spec_html += (
                f"<tr>"
                f"<td>{html_escape(_dialect_label(code))}</td>"
                f'<td class="num">{code.value}</td>'
                f'<td class="num">{sp.entropy:.4f}</td>'
                f'<td class="num">{n}</td>'
                f'<td class="num">{k50}</td>'
                f'<td class="num">{k90}</td>'
                f"</tr>"
            )
        spec_html += "</table></div>"
        spec_html += '<p style="color:var(--text-muted);font-size:0.85rem;">"Energy at X%" shows how many leading eigenvalues are needed to capture X% of total spectral energy.</p>'
        sections.append(spec_html)

    # ------------------------------------------------------------------
    # Distance matrix (coloured)
    # ------------------------------------------------------------------
    if has_distances:
        assert distance_matrix is not None and dialect_order is not None
        dm = distance_matrix
        n = len(dialect_order)

        # Normalise for colour mapping
        flat = dm[np.triu_indices(n, k=1)]
        d_min = float(flat.min()) if len(flat) > 0 else 0.0
        d_max = float(flat.max()) if len(flat) > 0 else 1.0
        d_range = d_max - d_min if d_max > d_min else 1.0

        dist_html = '<h2 id="distances">Distance Matrix</h2>'
        dist_html += '<div class="card" style="overflow-x:auto;"><table>'
        dist_html += "<tr><th></th>"
        for c in dialect_order:
            dist_html += f"<th>{c.value}</th>"
        dist_html += "</tr>"

        for i, src in enumerate(dialect_order):
            dist_html += f"<tr><th style='text-align:left'>{src.value}</th>"
            for j, _tgt in enumerate(dialect_order):
                val = float(dm[i, j])
                if i == j:
                    bg = "#f0fff4"
                    txt = "&mdash;"
                else:
                    # Gradient: green (close) -> yellow -> red (far)
                    ratio = (val - d_min) / d_range
                    ratio = max(0.0, min(1.0, ratio))
                    if ratio < 0.5:
                        r = int(255 * (ratio * 2))
                        g = 200
                    else:
                        r = 255
                        g = int(200 * (1 - (ratio - 0.5) * 2))
                    bg = f"rgb({r},{g},80)"
                    txt = f"{val:.4f}"
                dist_html += f'<td class="dist-cell" style="background:{bg};">{txt}</td>'
            dist_html += "</tr>"
        dist_html += "</table></div>"

        # Legend
        dist_html += '<p style="color:var(--text-muted);font-size:0.85rem;">Colour scale: <span style="background:rgb(0,200,80);padding:0 6px;border-radius:3px;">low</span> &rarr; <span style="background:rgb(255,200,80);padding:0 6px;border-radius:3px;">mid</span> &rarr; <span style="background:rgb(255,0,80);padding:0 6px;border-radius:3px;color:#fff;">high</span></p>'
        sections.append(dist_html)

    # ------------------------------------------------------------------
    # Transformation matrices summary
    # ------------------------------------------------------------------
    if has_transforms:
        assert transforms is not None
        tf_html = '<h2 id="transforms">Transformation Matrices</h2>'
        tf_html += '<div class="card"><table>'
        tf_html += "<tr><th>Dialect</th><th>Source</th><th>Target</th><th>Shape</th><th>Regularization</th><th>Frobenius Norm</th></tr>"
        for code in sorted(transforms, key=lambda c: c.value):
            tm = transforms[code]
            frob = float(np.linalg.norm(tm.data, "fro"))
            tf_html += (
                f"<tr>"
                f"<td>{html_escape(_dialect_label(code))}</td>"
                f'<td class="num">{tm.source_dialect.value}</td>'
                f'<td class="num">{tm.target_dialect.value}</td>'
                f'<td class="num">{tm.shape[0]}x{tm.shape[1]}</td>'
                f'<td class="num">{tm.regularization:.4g}</td>'
                f'<td class="num">{frob:.4f}</td>'
                f"</tr>"
            )
        tf_html += "</table></div>"
        sections.append(tf_html)

    # ------------------------------------------------------------------
    # Embedding spaces summary
    # ------------------------------------------------------------------
    if has_embeddings:
        assert embeddings is not None
        emb_html = '<h2 id="embeddings">Embedding Spaces</h2>'
        emb_html += '<div class="card"><table>'
        emb_html += "<tr><th>Dialect</th><th>Vocabulary</th><th>Dimensions</th><th>L2 Mean</th><th>L2 Std</th></tr>"
        for code in sorted(embeddings, key=lambda c: c.value):
            em = embeddings[code]
            norms = np.linalg.norm(em.data, axis=1)
            emb_html += (
                f"<tr>"
                f"<td>{html_escape(_dialect_label(code))}</td>"
                f'<td class="num">{len(em.vocab):,}</td>'
                f'<td class="num">{em.dim}</td>'
                f'<td class="num">{float(norms.mean()):.4f}</td>'
                f'<td class="num">{float(norms.std()):.4f}</td>'
                f"</tr>"
            )
        emb_html += "</table></div>"
        sections.append(emb_html)

    # ------------------------------------------------------------------
    # Experiment results
    # ------------------------------------------------------------------
    if has_experiments:
        assert experiment_results is not None
        exp_html = '<h2 id="experiments">Experiment Results</h2>'
        for eid in sorted(experiment_results):
            er = experiment_results[eid]
            exp_html += f'<h3>{html_escape(er.experiment_id)}</h3>'
            exp_html += f'<p style="color:var(--text-muted);font-size:0.85rem;">Timestamp: {html_escape(er.timestamp)}</p>'
            exp_html += '<div class="card">'
            if er.metrics:
                exp_html += "<p><strong>Metrics:</strong></p><p>"
                for mk, mv in er.metrics.items():
                    if isinstance(mv, float):
                        disp = f"{mv:.6g}"
                    else:
                        disp = html_escape(str(mv))
                    exp_html += f'<span class="metric-badge">{html_escape(str(mk))}: {disp}</span> '
                exp_html += "</p>"
            if er.config:
                exp_html += "<p style='margin-top:0.75rem;'><strong>Configuration:</strong></p>"
                exp_html += '<table style="font-size:0.85rem;">'
                for ck, cv in er.config.items():
                    exp_html += f"<tr><td><strong>{html_escape(str(ck))}</strong></td><td>{html_escape(str(cv))}</td></tr>"
                exp_html += "</table>"
            if er.artifact_paths:
                exp_html += "<p style='margin-top:0.75rem;'><strong>Artifacts:</strong></p><ul>"
                for ap in er.artifact_paths:
                    exp_html += f"<li><code>{html_escape(ap)}</code></li>"
                exp_html += "</ul>"
            exp_html += "</div>"
        sections.append(exp_html)

    # ------------------------------------------------------------------
    # Assemble full page
    # ------------------------------------------------------------------
    body = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EigenDialectos &mdash; Pipeline Report</title>
{css}
</head>
<body>
    <h1>EigenDialectos &mdash; Pipeline Report</h1>
    <p class="subtitle">Generated {now}</p>
{toc_html}
{body}
    <footer>
        EigenDialectos pipeline report &bull; Generated automatically &bull; {now}
    </footer>
</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def _build_summary(
    *,
    spectra: dict[DialectCode, DialectalSpectrum] | None,
    eigendecompositions: dict[DialectCode, EigenDecomposition] | None,
    distance_matrix: npt.NDArray[np.float64] | None,
    dialect_order: list[DialectCode] | None,
    experiment_results: dict[str, ExperimentResult] | None,
    embeddings: dict[DialectCode, EmbeddingMatrix] | None,
    vocab: list[str] | None,
    transforms: dict[DialectCode, TransformationMatrix] | None,
) -> dict[str, Any]:
    """Build a summary dict capturing the high-level pipeline state."""
    summary: dict[str, Any] = {
        "generated_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "dialect_count": 0,
        "dialects": [],
    }

    # Collect all dialect codes seen
    all_codes: set[DialectCode] = set()
    if eigendecompositions:
        all_codes.update(eigendecompositions.keys())
    if spectra:
        all_codes.update(spectra.keys())
    if transforms:
        all_codes.update(transforms.keys())
    if embeddings:
        all_codes.update(embeddings.keys())

    sorted_codes = sorted(all_codes, key=lambda c: c.value)
    summary["dialect_count"] = len(sorted_codes)
    summary["dialects"] = [
        {"code": c.value, "name": _dialect_label(c)} for c in sorted_codes
    ]

    if vocab:
        summary["vocabulary_size"] = len(vocab)

    if embeddings:
        for em in embeddings.values():
            summary["embedding_dim"] = em.dim
            break

    if spectra:
        summary["entropy"] = {
            c.value: float(spectra[c].entropy) for c in sorted_codes if c in spectra
        }

    if distance_matrix is not None and dialect_order:
        n = len(dialect_order)
        triu = distance_matrix[np.triu_indices(n, k=1)]
        summary["distance_stats"] = {
            "mean": float(triu.mean()),
            "std": float(triu.std()),
            "min": float(triu.min()),
            "max": float(triu.max()),
        }

    if experiment_results:
        summary["experiments"] = list(experiment_results.keys())

    return summary


# ---------------------------------------------------------------------------
# Corpus stats CSV
# ---------------------------------------------------------------------------

def _export_corpus_stats(
    embeddings: dict[DialectCode, EmbeddingMatrix],
    output_dir: Path,
) -> Path:
    """Write corpus_stats.csv with per-dialect vocabulary and embedding info."""
    rows: list[dict[str, Any]] = []
    for code in sorted(embeddings, key=lambda c: c.value):
        em = embeddings[code]
        norms = np.linalg.norm(em.data, axis=1)
        rows.append({
            "dialect": code.value,
            "dialect_name": _dialect_label(code),
            "vocab_size": len(em.vocab),
            "embedding_dim": em.dim,
            "l2_norm_mean": f"{float(norms.mean()):.6f}",
            "l2_norm_std": f"{float(norms.std()):.6f}",
        })
    return _save_csv(rows, output_dir / "corpus_stats.csv")


# ---------------------------------------------------------------------------
# ExportManager class
# ---------------------------------------------------------------------------

class ExportManager:
    """Stateful manager that accumulates data and exports everything at once.

    Usage::

        mgr = ExportManager(output_dir)
        mgr.transforms = {...}
        mgr.spectra = {...}
        created = mgr.export_all()
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.transforms: dict[DialectCode, TransformationMatrix] | None = None
        self.eigendecompositions: dict[DialectCode, EigenDecomposition] | None = None
        self.spectra: dict[DialectCode, DialectalSpectrum] | None = None
        self.distance_matrix: npt.NDArray[np.float64] | None = None
        self.dialect_order: list[DialectCode] | None = None
        self.experiment_results: dict[str, ExperimentResult] | None = None
        self.embeddings: dict[DialectCode, EmbeddingMatrix] | None = None
        self.vocab: list[str] | None = None

    def export_all(self) -> dict[str, list[Path]]:
        """Export everything that has been set, delegating to :func:`export_all`."""
        return export_all(
            output_dir=self.output_dir,
            transforms=self.transforms,
            eigendecompositions=self.eigendecompositions,
            spectra=self.spectra,
            distance_matrix=self.distance_matrix,
            dialect_order=self.dialect_order,
            experiment_results=self.experiment_results,
            embeddings=self.embeddings,
            vocab=self.vocab,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def export_all(
    output_dir: Path,
    transforms: dict[DialectCode, TransformationMatrix] | None = None,
    eigendecompositions: dict[DialectCode, EigenDecomposition] | None = None,
    spectra: dict[DialectCode, DialectalSpectrum] | None = None,
    distance_matrix: npt.NDArray[np.float64] | None = None,
    dialect_order: list[DialectCode] | None = None,
    experiment_results: dict[str, ExperimentResult] | None = None,
    embeddings: dict[DialectCode, EmbeddingMatrix] | None = None,
    vocab: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Export all available pipeline results in JSON, CSV, NumPy, and HTML.

    Parameters
    ----------
    output_dir:
        Root directory for all exported files.  Sub-directories are created
        automatically (``transforms/``, ``eigenvalues/``, ``eigenvectors/``).
    transforms:
        Per-dialect transformation matrices.
    eigendecompositions:
        Per-dialect eigendecompositions.
    spectra:
        Per-dialect spectral profiles.
    distance_matrix:
        Pairwise spectral distance matrix.
    dialect_order:
        Ordered dialect codes corresponding to rows/columns of
        *distance_matrix*.
    experiment_results:
        Results from experiment runs.
    embeddings:
        Per-dialect embedding matrices.
    vocab:
        Shared vocabulary list.

    Returns
    -------
    dict[str, list[Path]]
        Mapping from category label (``"json"``, ``"csv"``, ``"npy"``,
        ``"html"``) to the list of files created in that category.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    created: dict[str, list[Path]] = {
        "json": [],
        "csv": [],
        "npy": [],
        "html": [],
    }

    logger.info("Beginning export to %s", out)

    # ------------------------------------------------------------------
    # Transforms (.npy)
    # ------------------------------------------------------------------
    if transforms:
        logger.info("Exporting %d transformation matrices", len(transforms))
        paths = export_transforms(transforms, out)
        created["npy"].extend(paths)

    # ------------------------------------------------------------------
    # Eigendecompositions (.npy + .json + .csv)
    # ------------------------------------------------------------------
    if eigendecompositions:
        logger.info("Exporting %d eigendecompositions", len(eigendecompositions))
        paths = export_eigendecomposition(eigendecompositions, out)
        for p in paths:
            if p.suffix == ".npy":
                created["npy"].append(p)
            elif p.suffix == ".json":
                created["json"].append(p)
            elif p.suffix == ".csv":
                created["csv"].append(p)

    # ------------------------------------------------------------------
    # Spectra (.json + .csv)
    # ------------------------------------------------------------------
    if spectra:
        logger.info("Exporting %d spectral profiles", len(spectra))
        paths = export_spectra(spectra, out)
        for p in paths:
            if p.suffix == ".json":
                created["json"].append(p)
            elif p.suffix == ".csv":
                created["csv"].append(p)

    # ------------------------------------------------------------------
    # Distance matrix (.npy + .json + .csv)
    # ------------------------------------------------------------------
    if distance_matrix is not None and dialect_order is not None:
        logger.info("Exporting distance matrix (%d dialects)", len(dialect_order))
        paths = export_distances(distance_matrix, dialect_order, out)
        for p in paths:
            if p.suffix == ".npy":
                created["npy"].append(p)
            elif p.suffix == ".json":
                created["json"].append(p)
            elif p.suffix == ".csv":
                created["csv"].append(p)

    # ------------------------------------------------------------------
    # Experiment results (.json + .csv)
    # ------------------------------------------------------------------
    if experiment_results:
        logger.info("Exporting %d experiment results", len(experiment_results))
        paths = export_experiment_results(experiment_results, out)
        for p in paths:
            if p.suffix == ".json":
                created["json"].append(p)
            elif p.suffix == ".csv":
                created["csv"].append(p)

    # ------------------------------------------------------------------
    # Corpus stats CSV (from embeddings)
    # ------------------------------------------------------------------
    if embeddings:
        logger.info("Exporting corpus stats for %d dialects", len(embeddings))
        p = _export_corpus_stats(embeddings, out)
        created["csv"].append(p)

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    summary = _build_summary(
        spectra=spectra,
        eigendecompositions=eigendecompositions,
        distance_matrix=distance_matrix,
        dialect_order=dialect_order,
        experiment_results=experiment_results,
        embeddings=embeddings,
        vocab=vocab,
        transforms=transforms,
    )
    p = _save_json(summary, out / "summary.json")
    created["json"].append(p)
    logger.info("Saved summary.json")

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------
    html = _build_html_report(
        spectra=spectra,
        eigendecompositions=eigendecompositions,
        distance_matrix=distance_matrix,
        dialect_order=dialect_order,
        experiment_results=experiment_results,
        embeddings=embeddings,
        vocab=vocab,
        transforms=transforms,
    )
    html_path = _ensure(out / "report.html")
    html_path.write_text(html, encoding="utf-8")
    created["html"].append(html_path)
    logger.info("Saved report.html")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total = sum(len(v) for v in created.values())
    logger.info(
        "Export complete: %d files (json=%d, csv=%d, npy=%d, html=%d)",
        total,
        len(created["json"]),
        len(created["csv"]),
        len(created["npy"]),
        len(created["html"]),
    )

    return created
