"""Contrastive alignment algorithms (Procrustes, VecMap, MUSE)."""

from eigendialectos.embeddings.contrastive.muse import MUSEAligner
from eigendialectos.embeddings.contrastive.procrustes import ProcrustesAligner
from eigendialectos.embeddings.contrastive.vecmap import VecMapAligner

__all__ = ["MUSEAligner", "ProcrustesAligner", "VecMapAligner"]
