#!/usr/bin/env python3
"""Generate final research report from experiment outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def generate_report(output_dir: str = "outputs/reports") -> None:
    """Generate a summary report of all experiment results."""
    project_dir = Path(__file__).resolve().parents[1]
    experiments_dir = project_dir / "data" / "experiments"
    report_dir = project_dir / output_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# EigenDialectos: Experiment Report",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
    ]

    experiment_names = {
        "exp1_spectral_map": "Spectral Map of Spanish",
        "exp2_full_generation": "100% Generation",
        "exp3_dialectal_gradient": "Dialectal Gradient",
        "exp4_impossible_dialects": "Impossible Dialects",
        "exp5_archaeology": "Dialectal Archaeology",
        "exp6_evolution": "Eigenvalues as Evolution Proxy",
        "exp7_zeroshot": "Zero-shot Dialectal Transfer",
    }

    for exp_id, exp_name in experiment_names.items():
        result_file = experiments_dir / exp_id / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
            report_lines.append(f"### {exp_name}")
            report_lines.append(f"- Status: Completed")
            metrics = result.get("metrics", {})
            for k, v in metrics.items():
                report_lines.append(f"- {k}: {v}")
            report_lines.append("")
        else:
            report_lines.append(f"### {exp_name}")
            report_lines.append("- Status: Not yet run")
            report_lines.append("")

    report_text = "\n".join(report_lines)
    report_path = report_dir / "experiment_report.md"
    report_path.write_text(report_text)
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    generate_report()
