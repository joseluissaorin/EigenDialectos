"""SDC — Spectral Dialectal Compiler.

Transforms free-form Spanish text from one dialect variety into another by:

1. **Parsing** the text into tokens (preserving punctuation and whitespace).
2. **Finding replacements** via a dual-path kNN strategy:
   - *Spectral path*: transform the source word vector through W(alpha), then
     find nearest neighbours in the target embedding space.
   - *Direct path*: find nearest neighbours of the source word directly in the
     target space (acts as a sanity / fallback path).
   Candidates are scored by ``spectral_confidence * cosine_similarity``.
3. **Applying** the best replacement for each content word.
4. **Fixing** basic morpho-syntactic agreement (article gender).

The full pipeline is wrapped in :func:`compile`.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

import numpy as np

from eigen3.per_mode import compute_W_alpha
from eigen3.types import (
    AlphaVector,
    ChangeEntry,
    EigenDecomp,
    TransformResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stop words — Spanish function words that should never be replaced
# ---------------------------------------------------------------------------

_STOP_WORDS: set[str] = {
    # Articles
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # Prepositions
    "a", "ante", "bajo", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por",
    "según", "sin", "sobre", "tras",
    # Conjunctions
    "y", "e", "o", "u", "ni", "que", "si", "pero", "sino", "aunque",
    "porque", "como", "cuando", "donde", "mientras",
    # Pronouns & determiners
    "yo", "tú", "él", "ella", "ello", "nosotros", "nosotras",
    "vosotros", "vosotras", "ellos", "ellas", "usted", "ustedes",
    "me", "te", "se", "nos", "os", "le", "les", "lo", "la",
    "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra",
    "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras",
    # Demonstratives
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas",
    # Common verbs (copula, auxiliary)
    "es", "ser", "estar", "hay", "haber", "ha", "he", "son", "está",
    "están", "fue", "era", "sido", "siendo",
    # Adverbs
    "no", "sí", "ya", "más", "muy", "también", "bien", "mal",
    "solo", "así", "aquí", "allí", "ahí",
    # Other high-frequency function words
    "del", "al", "todo", "toda", "todos", "todas",
}


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def parse_text(text: str) -> list[dict]:
    """Tokenize *text* preserving word boundaries, punctuation and case.

    Each token is a dict with:
        - ``word``:     original surface form
        - ``lower``:    lowercased form
        - ``position``: 0-based index in the token list
        - ``is_punct``: True if the token is only punctuation/whitespace
        - ``is_stop``:  True if the lowered form is a stop word

    The tokeniser splits on word boundaries but keeps every character
    (including whitespace and punctuation) so that the original text can be
    perfectly reconstructed by concatenating ``token["word"]`` values.

    Returns
    -------
    list[dict]
        Ordered token list.
    """
    # Split into words and non-word chunks (whitespace + punctuation).
    raw_tokens = re.findall(r"\w+|[^\w]", text, re.UNICODE)

    tokens: list[dict] = []
    for i, raw in enumerate(raw_tokens):
        lower = raw.lower()
        is_punct = not raw.isalnum() and not any(c.isalpha() for c in raw)
        tokens.append({
            "word": raw,
            "lower": lower,
            "position": i,
            "is_punct": is_punct,
            "is_stop": lower in _STOP_WORDS,
        })
    return tokens


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def _normalise_rows(M: np.ndarray) -> np.ndarray:
    """L2-normalise every row; zero rows stay zero."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return M / norms


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _knn(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return *k* nearest neighbours of *query* in *matrix* (cosine).

    Parameters
    ----------
    query : np.ndarray (dim,)
    matrix : np.ndarray (vocab_size, dim)
    k : int

    Returns
    -------
    list of (index, cosine_similarity) sorted descending.
    """
    normed_matrix = _normalise_rows(matrix)
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return []
    normed_q = query / q_norm

    sims = normed_matrix @ normed_q  # (vocab_size,)
    # argpartition is O(n) instead of full sort
    if k >= len(sims):
        top_k = np.argsort(-sims)
    else:
        top_k_unsorted = np.argpartition(-sims, k)[:k]
        top_k = top_k_unsorted[np.argsort(-sims[top_k_unsorted])]

    return [(int(idx), float(sims[idx])) for idx in top_k]


# ---------------------------------------------------------------------------
# Replacement finding — dual-path kNN
# ---------------------------------------------------------------------------

def find_replacements(
    word: str,
    W_alpha: np.ndarray,
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    vocab: list[str],
    word_to_idx: dict[str, int],
    k: int = 10,
) -> list[dict]:
    """Find candidate replacements for *word* via dual-path kNN.

    Path 1 — **Spectral**: transform the source word vector through
    ``W_alpha``, then search for nearest neighbours in the target space.

    Path 2 — **Direct**: search for *word*'s source vector directly among
    target embeddings (no transformation).

    Candidates are scored as  ``spectral_confidence * cosine_similarity``
    for spectral hits, and ``direct_confidence * cosine_similarity * 0.5``
    for direct hits (halved to prefer the spectral path).

    Parameters
    ----------
    word : str
        Lowercased word to look up.
    W_alpha : np.ndarray
        (dim, dim) parametric transformation matrix.
    source_emb : np.ndarray
        (vocab_size, dim) source dialect embedding matrix.
    target_emb : np.ndarray
        (vocab_size, dim) target dialect embedding matrix.
    vocab : list[str]
        Ordered vocabulary (index ↔ word).
    word_to_idx : dict[str, int]
        Word → index mapping.
    k : int
        Number of nearest neighbours per path.

    Returns
    -------
    list[dict]
        Sorted (descending) by combined score.  Each dict has keys:
        ``candidate``, ``score``, ``path`` ("spectral" | "direct"),
        ``cosine``.
    """
    if word not in word_to_idx:
        return []

    idx = word_to_idx[word]
    source_vec = source_emb[idx]  # (dim,)

    candidates: dict[str, dict] = {}

    # --- Spectral path ---
    transformed = source_vec @ W_alpha.T  # (dim,)
    spectral_neighbours = _knn(transformed, target_emb, k)

    for nbr_idx, cosine in spectral_neighbours:
        cand_word = vocab[nbr_idx]
        if cand_word == word:
            continue
        spectral_conf = _cosine_similarity(transformed, target_emb[nbr_idx])
        score = spectral_conf * cosine
        if cand_word not in candidates or candidates[cand_word]["score"] < score:
            candidates[cand_word] = {
                "candidate": cand_word,
                "score": float(score),
                "path": "spectral",
                "cosine": float(cosine),
            }

    # --- Direct path ---
    direct_neighbours = _knn(source_vec, target_emb, k)

    for nbr_idx, cosine in direct_neighbours:
        cand_word = vocab[nbr_idx]
        if cand_word == word:
            continue
        direct_conf = cosine
        score = direct_conf * cosine * 0.5  # discount direct path
        if cand_word not in candidates or candidates[cand_word]["score"] < score:
            candidates[cand_word] = {
                "candidate": cand_word,
                "score": float(score),
                "path": "direct",
                "cosine": float(cosine),
            }

    return sorted(candidates.values(), key=lambda c: c["score"], reverse=True)


# ---------------------------------------------------------------------------
# Replacement application
# ---------------------------------------------------------------------------

def _match_case(original: str, replacement: str) -> str:
    """Reproduce the casing pattern of *original* on *replacement*.

    Rules:
        ALL UPPER  → replacement uppercased.
        Title Case → replacement title-cased.
        Otherwise  → replacement as-is (assumed lower).
    """
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply_replacements(
    tokens: list[dict],
    replacements: dict[int, str],
) -> str:
    """Rebuild text from *tokens*, substituting words at given positions.

    Parameters
    ----------
    tokens : list[dict]
        Token list from :func:`parse_text`.
    replacements : dict[int, str]
        ``{position: replacement_word}`` — positions that should be
        replaced. The case of the original token is applied to the
        replacement.

    Returns
    -------
    str
        Reconstructed text.
    """
    parts: list[str] = []
    for tok in tokens:
        pos = tok["position"]
        if pos in replacements:
            parts.append(_match_case(tok["word"], replacements[pos]))
        else:
            parts.append(tok["word"])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Agreement fixer
# ---------------------------------------------------------------------------

# Masculine / feminine article mapping
_ARTICLE_GENDER: dict[str, str] = {
    "el": "la", "la": "el",
    "los": "las", "las": "los",
    "un": "una", "una": "un",
    "unos": "unas", "unas": "unos",
}

_FEMININE_ENDINGS = ("a", "ión", "dad", "tad", "tud", "umbre", "ie", "ez")
_MASCULINE_ENDINGS = ("o", "or", "aje", "men")


def _guess_gender(word: str) -> Optional[str]:
    """Heuristic gender guess for a Spanish noun: 'm' or 'f' or None."""
    lower = word.lower()
    # Common exceptions
    if lower in ("mano", "foto", "moto", "radio"):
        return "f"
    if lower in ("día", "mapa", "problema", "sistema", "tema", "idioma", "programa"):
        return "m"
    for ending in _FEMININE_ENDINGS:
        if lower.endswith(ending):
            return "f"
    for ending in _MASCULINE_ENDINGS:
        if lower.endswith(ending):
            return "m"
    return None


def fix_agreement(text: str) -> str:
    """Basic article–noun gender agreement correction.

    Scans for article + word pairs and swaps the article if the heuristic
    gender of the noun disagrees.  This is intentionally conservative —
    it only handles the most common patterns.

    Parameters
    ----------
    text : str
        Input text (already with replacements applied).

    Returns
    -------
    str
        Text with corrected articles where possible.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text

    masc_articles = {"el", "los", "un", "unos"}
    fem_articles = {"la", "las", "una", "unas"}

    for i in range(len(tokens) - 1):
        art_lower = tokens[i].lower()
        if art_lower not in _ARTICLE_GENDER:
            continue

        noun = tokens[i + 1]
        gender = _guess_gender(noun)
        if gender is None:
            continue

        if gender == "f" and art_lower in masc_articles:
            replacement = _ARTICLE_GENDER[art_lower]
            tokens[i] = _match_case(tokens[i], replacement)
        elif gender == "m" and art_lower in fem_articles:
            replacement = _ARTICLE_GENDER[art_lower]
            tokens[i] = _match_case(tokens[i], replacement)

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def compile(
    text: str,
    source: str,
    target: str,
    embeddings: dict[str, np.ndarray],
    vocab: list[str],
    decomp: EigenDecomp,
    alpha: Optional[AlphaVector] = None,
    replacement_threshold: float = 0.3,
    k: int = 10,
) -> TransformResult:
    """Spectral Dialectal Compiler — full pipeline.

    Steps:
        1. Parse the input text into tokens.
        2. Build W(alpha) from the decomposition.
        3. For each content token, find replacement candidates.
        4. Apply the best replacements above the confidence threshold.
        5. Fix article–noun agreement.
        6. Package the result with full traceability.

    Parameters
    ----------
    text : str
        Input text in the *source* dialect.
    source : str
        Source variety code (e.g. ``"ES_PEN"``).
    target : str
        Target variety code (e.g. ``"ES_RIO"``).
    embeddings : dict[str, np.ndarray]
        ``{variety: (vocab_size, dim)}`` embedding matrices.
    vocab : list[str]
        Shared vocabulary.
    decomp : EigenDecomp
        Eigendecomposition of the source→target W matrix.
    alpha : AlphaVector, optional
        Per-mode intensity vector.  Defaults to all-ones (full transform).
    replacement_threshold : float
        Minimum candidate score to accept a replacement.
    k : int
        Number of kNN candidates per path.

    Returns
    -------
    TransformResult
        Transformed text together with a list of :class:`ChangeEntry`
        objects recording every substitution with its dominant eigenmode,
        eigenvalue and confidence.
    """
    if alpha is None:
        alpha = AlphaVector.ones(decomp.n_modes)

    # Build the parametric operator
    W_alpha = compute_W_alpha(decomp, alpha)

    source_emb = embeddings[source]
    target_emb = embeddings[target]
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # Tokenize
    tokens = parse_text(text)

    # Find replacements for each content token
    replacements: dict[int, str] = {}
    changes: list[ChangeEntry] = []

    for tok in tokens:
        if tok["is_punct"] or tok["is_stop"]:
            continue
        word = tok["lower"]

        candidates = find_replacements(
            word, W_alpha, source_emb, target_emb, vocab, word_to_idx, k=k,
        )
        if not candidates:
            continue

        best = candidates[0]
        if best["score"] < replacement_threshold:
            continue

        replacements[tok["position"]] = best["candidate"]

        # Determine dominant mode: the mode with the largest |lambda|
        # contribution (mode 0 by default since eigenvalues are sorted)
        dominant_mode = int(np.argmax(np.abs(decomp.eigenvalues) * alpha.values))
        dominant_eigenvalue = float(np.abs(decomp.eigenvalues[dominant_mode]))

        changes.append(ChangeEntry(
            position=tok["position"],
            original=tok["word"],
            replacement=_match_case(tok["word"], best["candidate"]),
            confidence=best["score"],
            mode_idx=dominant_mode,
            eigenvalue=dominant_eigenvalue,
        ))

    # Apply replacements
    output_text = apply_replacements(tokens, replacements)

    # Fix agreement
    output_text = fix_agreement(output_text)

    logger.info(
        "SDC %s→%s: %d tokens, %d replacements",
        source, target, len(tokens), len(changes),
    )

    return TransformResult(
        text=output_text,
        changes=changes,
        alpha=alpha,
        source=source,
        target=target,
    )
