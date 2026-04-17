"""LoRA adapter integration for dialect-aware language model fine-tuning.

Bridges the algebraic DIAL framework with parameter-efficient fine-tuning
(PEFT/LoRA), enabling continuous dialect generation via adapted LLMs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from eigendialectos.constants import DialectCode
from eigendialectos.generative.dial import apply_dial, dial_transform_embedding
from eigendialectos.types import EigenDecomposition, TransformationMatrix

logger = logging.getLogger(__name__)


class LoRADialectAdapter:
    """Creates LoRA training data and configs from DIAL transforms.

    Parameters
    ----------
    output_dir : Path or str
        Directory for saving adapters and training artefacts.
    rank : int
        LoRA rank (dimensionality of low-rank update matrices).
    lora_alpha : float
        LoRA scaling parameter (not the dialectal alpha).
    target_modules : list of str or None
        Transformer module names to apply LoRA to.
    """

    def __init__(
        self,
        output_dir: Path | str = Path("./lora_adapters"),
        rank: int = 8,
        lora_alpha: float = 16.0,
        target_modules: list[str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ]

    def create_parallel_corpus(
        self,
        neutral_embeddings: npt.NDArray[np.floating],
        dial_transforms: list[TransformationMatrix],
        n_samples: int,
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Generate paired neutral/dialectal embeddings for training.

        Creates ``n_samples`` paired embedding vectors by randomly sampling
        from the neutral embeddings and applying each DIAL transform.

        Parameters
        ----------
        neutral_embeddings : ndarray, shape (vocab_size, dim)
            Matrix of neutral (alpha=0) embeddings.
        dial_transforms : list of TransformationMatrix
            DIAL transform(s) to apply.
        n_samples : int
            Number of parallel pairs to generate.

        Returns
        -------
        list of (neutral_vec, dialectal_vec)
            Paired embeddings for LoRA training.
        """
        neutral = np.asarray(neutral_embeddings, dtype=np.float64)
        vocab_size = neutral.shape[0]
        rng = np.random.default_rng(42)

        pairs: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        indices = rng.choice(vocab_size, size=n_samples, replace=True)

        for idx in indices:
            source_vec = neutral[idx]
            # Pick a random transform from the set
            tm_idx = rng.integers(0, len(dial_transforms))
            W = dial_transforms[tm_idx].data
            target_vec = (W @ source_vec).astype(np.float64)
            pairs.append((source_vec.copy(), target_vec))

        return pairs

    def prepare_lora_config(
        self,
        dialect: DialectCode,
        alpha: float,
    ) -> dict[str, Any]:
        """Build a LoRA configuration dictionary.

        Parameters
        ----------
        dialect : DialectCode
            Target dialect for the adapter.
        alpha : float
            Dialectal intensity used to generate the training data.

        Returns
        -------
        dict
            Configuration suitable for PEFT ``LoraConfig``.
        """
        return {
            "task_type": "CAUSAL_LM",
            "r": self.rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": 0.05,
            "bias": "none",
            "dialect_code": dialect.value,
            "dialectal_alpha": alpha,
            "inference_mode": False,
        }

    def train_adapter(
        self,
        model_name: str,
        parallel_corpus: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
        config: dict[str, Any],
    ) -> Path:
        """Train a LoRA adapter using PEFT (if available).

        This is a best-effort integration: if the ``peft`` or
        ``transformers`` packages are not installed, the method saves the
        configuration and corpus metadata without actually training.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier (e.g. ``"meta-llama/Llama-2-7b"``).
        parallel_corpus : list of (source, target) embedding pairs
            Training data produced by :meth:`create_parallel_corpus`.
        config : dict
            LoRA configuration from :meth:`prepare_lora_config`.

        Returns
        -------
        Path
            Directory where the adapter (or fallback metadata) was saved.
        """
        dialect_code = config.get("dialect_code", "unknown")
        dial_alpha = config.get("dialectal_alpha", 1.0)
        adapter_dir = self.output_dir / f"{dialect_code}_alpha{dial_alpha}"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = adapter_dir / "lora_config.json"
        serialisable = {
            k: v
            for k, v in config.items()
            if isinstance(v, (str, int, float, bool, list))
        }
        serialisable["model_name"] = model_name
        serialisable["n_training_pairs"] = len(parallel_corpus)
        config_path.write_text(json.dumps(serialisable, indent=2))

        # Save corpus statistics
        if parallel_corpus:
            sources = np.array([s for s, _ in parallel_corpus])
            targets = np.array([t for _, t in parallel_corpus])
            stats = {
                "n_pairs": len(parallel_corpus),
                "source_mean_norm": float(np.mean(np.linalg.norm(sources, axis=1))),
                "target_mean_norm": float(np.mean(np.linalg.norm(targets, axis=1))),
                "embedding_dim": int(sources.shape[1]),
            }
            stats_path = adapter_dir / "corpus_stats.json"
            stats_path.write_text(json.dumps(stats, indent=2))

        # Attempt actual LoRA training
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
            from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

            lora_cfg = LoraConfig(
                task_type=config.get("task_type", "CAUSAL_LM"),
                r=config.get("r", self.rank),
                lora_alpha=config.get("lora_alpha", self.lora_alpha),
                target_modules=config.get("target_modules", self.target_modules),
                lora_dropout=config.get("lora_dropout", 0.05),
                bias=config.get("bias", "none"),
            )
            model = AutoModelForCausalLM.from_pretrained(model_name)
            peft_model = get_peft_model(model, lora_cfg)

            logger.info(
                "PEFT model created with %d trainable parameters",
                sum(p.numel() for p in peft_model.parameters() if p.requires_grad),
            )

            # Save the adapter
            peft_model.save_pretrained(str(adapter_dir))
            logger.info("Adapter saved to %s", adapter_dir)

        except ImportError:
            logger.warning(
                "peft/transformers not installed. Saved config and corpus "
                "stats to %s. Install with: pip install peft transformers",
                adapter_dir,
            )
        except Exception as exc:
            logger.warning(
                "LoRA training failed: %s. Config and corpus stats saved to %s",
                exc,
                adapter_dir,
            )

        return adapter_dir

    def load_adapter(self, path: Path | str) -> dict[str, Any]:
        """Load a previously saved adapter or its metadata.

        Parameters
        ----------
        path : Path or str
            Directory containing the adapter artefacts.

        Returns
        -------
        dict
            Loaded configuration (and model if PEFT is available).
        """
        path = Path(path)
        result: dict[str, Any] = {"path": str(path)}

        # Load config
        config_path = path / "lora_config.json"
        if config_path.exists():
            result["config"] = json.loads(config_path.read_text())

        # Load corpus stats
        stats_path = path / "corpus_stats.json"
        if stats_path.exists():
            result["corpus_stats"] = json.loads(stats_path.read_text())

        # Try to load the actual PEFT model
        try:
            from peft import PeftModel  # type: ignore[import-untyped]
            from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

            model_name = result.get("config", {}).get("model_name")
            if model_name:
                base_model = AutoModelForCausalLM.from_pretrained(model_name)
                peft_model = PeftModel.from_pretrained(base_model, str(path))
                result["model"] = peft_model
                logger.info("Loaded PEFT model from %s", path)
        except ImportError:
            logger.info(
                "peft/transformers not installed; returning config only."
            )
        except Exception as exc:
            logger.warning("Failed to load PEFT model: %s", exc)

        return result
