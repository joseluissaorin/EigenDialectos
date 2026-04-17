"""CLI commands for experiment execution."""

from __future__ import annotations

import click


@click.command("run")
@click.argument("experiment_id")
@click.option("--config", default=None, help="Config override file.")
@click.option("--output-dir", default="data/experiments", help="Output directory.")
def experiment_run(experiment_id: str, config: str | None, output_dir: str) -> None:
    """Run an experiment (1-7 or 'all')."""
    from eigendialectos.experiments.runner import ExperimentRunner

    click.echo(f"Running experiment: {experiment_id}")
    runner = ExperimentRunner(
        config={"output_dir": output_dir},
        data_dir=output_dir,
        output_dir=output_dir,
    )

    if experiment_id == "all":
        results = runner.run_all()
        for eid, result in results.items():
            click.echo(f"  {eid}: {result.metrics}")
    else:
        result = runner.run_experiment(experiment_id)
        click.echo(f"  Result: {result.metrics}")

    click.echo("Experiment(s) complete.")
