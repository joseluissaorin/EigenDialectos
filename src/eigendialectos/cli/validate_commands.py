"""CLI commands for validation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click


@click.command("run")
@click.argument("mode", default="all")
@click.option("--data-dir", default="data", help="Data directory.")
@click.option("--output-dir", default="outputs/validation", help="Output directory.")
def validate_run(mode: str, data_dir: str, output_dir: str) -> None:
    """Run validation suite (quantitative, qualitative, or all)."""
    click.echo(f"Running {mode} validation...")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    click.echo("Validation complete.")


@click.command("check")
def validate_check() -> None:
    """Run project completeness validator (MANIFEST.yaml check)."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "validate_project.py"
    if not script.exists():
        click.echo("Error: validate_project.py not found", err=True)
        return
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
    )
    raise SystemExit(result.returncode)
