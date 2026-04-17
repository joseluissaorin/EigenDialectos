"""Word-level embedding models (Word2Vec, GloVe)."""

from eigendialectos.embeddings.word.glove_model import GloVeModel
from eigendialectos.embeddings.word.word2vec_model import Word2VecModel

__all__ = ["GloVeModel", "Word2VecModel"]
