"""
Module: eigendialectos.cli.main
Phase: P0_FOUNDATION
Component: P0.4
Status: DONE

CLI entry point for EigenDialectos.
"""

from __future__ import annotations

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="eigendialectos")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """EigenDialectos: Spectral Decomposition of Spanish Dialect Varieties."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.group()
def corpus() -> None:
    """Corpus management commands."""


@cli.group()
def embed() -> None:
    """Embedding training and alignment commands."""


@cli.group()
def spectral() -> None:
    """Spectral analysis commands."""


@cli.group()
def dial() -> None:
    """DIAL generative model commands."""


@cli.group()
def tensor() -> None:
    """Tensor dialectal commands."""


@cli.group()
def experiment() -> None:
    """Experiment execution commands."""


@cli.group()
def validate() -> None:
    """Validation commands."""


# Register subcommands
from eigendialectos.cli.corpus_commands import (
    corpus_download, corpus_preprocess, corpus_stats, corpus_generate_synthetic,
)
corpus.add_command(corpus_download, "download")
corpus.add_command(corpus_preprocess, "preprocess")
corpus.add_command(corpus_stats, "stats")
corpus.add_command(corpus_generate_synthetic, "generate-synthetic")

from eigendialectos.cli.embedding_commands import embed_train, embed_align
embed.add_command(embed_train, "train")
embed.add_command(embed_align, "align")

from eigendialectos.cli.spectral_commands import spectral_compute, spectral_analyze
spectral.add_command(spectral_compute, "compute")
spectral.add_command(spectral_analyze, "analyze")

from eigendialectos.cli.generate_commands import dial_generate, dial_mix
dial.add_command(dial_generate, "generate")
dial.add_command(dial_mix, "mix")

from eigendialectos.cli.experiment_commands import experiment_run
experiment.add_command(experiment_run, "run")

from eigendialectos.cli.validate_commands import validate_run, validate_check
validate.add_command(validate_run, "run")
validate.add_command(validate_check, "check")


if __name__ == "__main__":
    cli()
