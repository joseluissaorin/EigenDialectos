"""Configuration loading from YAML files.

Reads per-module YAML configs from the ``configs/`` directory and merges
them into a single dictionary.  Falls back to sensible defaults when
config files are absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults used when config files are missing
# ---------------------------------------------------------------------------

_DEFAULT_EMBEDDING_CONFIGS: dict[str, dict[str, Any]] = {
    "fasttext": {
        "model": "fasttext",
        "dim": 300,
        "window": 5,
        "min_count": 5,
        "epochs": 10,
        "sg": 1,
        "neg": 10,
        "min_n": 3,
        "max_n": 6,
        "per_variety": True,
    },
    "word2vec": {
        "model": "word2vec",
        "dim": 300,
        "window": 5,
        "min_count": 5,
        "epochs": 15,
        "sg": 1,
        "neg": 10,
        "per_variety": True,
    },
    "beto": {
        "model": "beto",
        "pretrained": "dccuchile/bert-base-spanish-wwm-cased",
        "max_length": 128,
        "batch_size": 32,
        "learning_rate": 2.0e-5,
        "epochs": 3,
        "warmup_steps": 100,
        "pooling": "mean",
        "per_variety": True,
    },
}

_DEFAULT_CONTRASTIVE_CONFIG: dict[str, Any] = {
    "method": "procrustes",
    "reference_dialect": "ES_PEN",
    "anchor_words": 5000,
    "normalize": True,
    "max_iterations": 10,
    "convergence_threshold": 1.0e-6,
}

_DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "output_dir": "outputs",
    "data_dir": "data",
    "model_dir": "models",
    "log_level": "INFO",
    "project": {
        "name": "EigenDialectos",
        "version": "0.1.0",
        "description": "Spectral Decomposition of Spanish Dialect Varieties in Embedding Spaces",
    },
    "dialects": {
        "reference": "ES_PEN",
        "targets": [
            "ES_AND",
            "ES_CAN",
            "ES_RIO",
            "ES_MEX",
            "ES_CAR",
            "ES_CHI",
            "ES_AND_BO",
        ],
    },
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Resolve the project root (directory containing ``configs/``)."""
    # Walk upward from this file until we find the configs directory
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "configs").is_dir():
            return current
        current = current.parent
    # Fallback: assume CWD
    return Path.cwd()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file, returning an empty dict on failure."""
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        logger.debug("Config file not found: %s", path)
        return {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load all YAML configs and merge them into a single dict.

    Parameters
    ----------
    config_dir : Path, optional
        Path to the ``configs/`` directory.  When *None*, auto-detected
        relative to the project root.

    Returns
    -------
    dict
        Merged configuration with keys for each subsystem:
        ``corpus``, ``embeddings``, ``spectral``, ``generative``,
        ``tensor``, ``validation``, ``experiments``, and top-level
        settings from ``config.yaml``.
    """
    if config_dir is None:
        config_dir = _project_root() / "configs"
    config_dir = Path(config_dir)

    if not config_dir.is_dir():
        logger.warning("Config directory not found: %s  -- using defaults", config_dir)
        return dict(_DEFAULT_CONFIG)

    # Start with top-level config.yaml
    cfg = _deep_merge(_DEFAULT_CONFIG, _load_yaml(config_dir / "config.yaml"))

    # Load per-subsystem configs.  Each subdirectory may contain multiple
    # YAML files; we store them keyed by stem under the subsystem name.
    subsystems = [
        "corpus",
        "embeddings",
        "spectral",
        "generative",
        "tensor",
        "validation",
        "experiments",
    ]
    for sub in subsystems:
        sub_dir = config_dir / sub
        if not sub_dir.is_dir():
            continue
        sub_cfg: dict[str, Any] = {}
        for yaml_file in sorted(sub_dir.glob("*.yaml")):
            sub_cfg[yaml_file.stem] = _load_yaml(yaml_file)
        if sub_cfg:
            cfg[sub] = _deep_merge(cfg.get(sub, {}), sub_cfg)

    return cfg


def get_embedding_config(model_type: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Return embedding config for a specific model backend.

    Parameters
    ----------
    model_type : str
        One of ``"fasttext"``, ``"word2vec"``, or ``"beto"``.
    config_dir : Path, optional
        Override for the configs directory.

    Returns
    -------
    dict
        Model-specific configuration dict.
    """
    cfg = load_config(config_dir)
    embeddings = cfg.get("embeddings", {})

    # Try the loaded config first
    if model_type in embeddings:
        return embeddings[model_type]

    # Fall back to built-in defaults
    if model_type in _DEFAULT_EMBEDDING_CONFIGS:
        logger.info("Using default config for embedding model %r", model_type)
        return dict(_DEFAULT_EMBEDDING_CONFIGS[model_type])

    raise KeyError(
        f"Unknown embedding model type {model_type!r}. "
        f"Available: {sorted(set(embeddings) | set(_DEFAULT_EMBEDDING_CONFIGS))}"
    )


def get_contrastive_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Return the cross-variety alignment (contrastive) configuration.

    Parameters
    ----------
    config_dir : Path, optional
        Override for the configs directory.

    Returns
    -------
    dict
        Alignment configuration.
    """
    cfg = load_config(config_dir)
    embeddings = cfg.get("embeddings", {})

    if "contrastive" in embeddings:
        return embeddings["contrastive"]

    logger.info("Using default contrastive config")
    return dict(_DEFAULT_CONTRASTIVE_CONFIG)
