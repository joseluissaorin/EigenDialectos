"""Corpus loading, quality filtering, balancing, blending, and vocabulary building.

Includes automatic scraping integration and quality-filtering pipeline
for OCR cleanup, language detection, deduplication, and authenticity scoring.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from scipy import stats

from eigen3.constants import (
    ALL_VARIETIES,
    BLEND_PAIRS,
    CANARIAN_LEXICON,
    PHONOLOGICAL_RULES,
    REGIONALISMS,
    USTEDES_MAP,
    _ENGLISH_MARKERS,
    _SPANISH_MARKERS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_corpus(data_dir: str | Path) -> dict[str, list[str]]:
    """Load JSONL corpus files -- one per variety or a single combined file.

    Each JSONL line must have at minimum ``{"text": "...", "variety": "ES_CAN"}``.
    """
    data_dir = Path(data_dir)
    corpus: dict[str, list[str]] = {v: [] for v in ALL_VARIETIES}

    # Try per-variety files first
    found_per_variety = False
    for variety in ALL_VARIETIES:
        path = data_dir / f"{variety}.jsonl"
        if path.exists():
            found_per_variety = True
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get("text", "").strip()
                if text:
                    corpus[variety].append(text)

    # Fallback: single combined file
    if not found_per_variety:
        combined = data_dir / "corpus.jsonl"
        if combined.exists():
            for line in combined.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                # Support both "variety" and "dialect" keys
                variety = doc.get("variety", "") or doc.get("dialect", "")
                text = doc.get("text", "").strip()
                if variety in corpus and text:
                    corpus[variety].append(text)

    total = sum(len(docs) for docs in corpus.values())
    logger.info("Loaded corpus: %d documents across %d varieties", total, len(corpus))
    return corpus


# ---------------------------------------------------------------------------
# Quality filtering pipeline
# ---------------------------------------------------------------------------

# SSA/ASS style overrides: {\an8}, {\pos(x,y)}, etc.
_SSA_RE = re.compile(r"\{\\[^}]*\}")
# Repeated punctuation beyond 3
_REPEATED_PUNCT_RE = re.compile(r"([!?.,:;])\1{3,}")
# Multiple spaces/tabs
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
# Lines that are purely numbers (subtitle indices, timestamps)
_PURE_NUMBER_RE = re.compile(r"^\d+$")
# Subtitle markers: [Music], (laughing), ♪, etc.
_SUBTITLE_MARKER_RE = re.compile(r"\[.*?\]|\(.*?\)|[♪♫]")


def _detect_spanish(text: str) -> bool:
    """Detect if text is Spanish using langdetect + word-marker fallback."""
    # Try langdetect first
    try:
        from langdetect import detect
        lang = detect(text)
        if lang in ("es", "ca", "gl", "pt"):
            return True
        if lang == "en":
            return False
    except Exception:
        pass

    # Fallback: word-marker heuristic for short or ambiguous texts
    words = set(text.lower().split())
    es_count = len(words & _SPANISH_MARKERS)
    en_count = len(words & _ENGLISH_MARKERS)
    return es_count >= en_count and es_count >= 2


def _clean_ocr_artifacts(text: str) -> str:
    """Remove common OCR and subtitle noise from text."""
    # Remove SSA style overrides
    text = _SSA_RE.sub("", text)
    # Remove subtitle markers
    text = _SUBTITLE_MARKER_RE.sub("", text)
    # Cap repeated punctuation
    text = _REPEATED_PUNCT_RE.sub(r"\1\1\1", text)
    # Collapse multiple spaces
    text = _MULTI_SPACE_RE.sub(" ", text)
    # Remove leading dialogue dashes
    text = re.sub(r"^-\s*", "", text, flags=re.MULTILINE)
    # Remove stray digits/symbols at start of lines
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    # Normalize line breaks
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text.strip()


def _compute_authenticity_score(text: str, dialect: str) -> float:
    """Score dialectal authenticity by counting known regionalisms."""
    words = set(text.lower().split())
    dialect_words = REGIONALISMS.get(dialect, set())
    if not dialect_words:
        return 0.0
    n_hits = len(words & dialect_words)
    return min(1.0, n_hits / 2.0)


def _dedup_texts(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate samples by content hash."""
    seen: set[str] = set()
    unique: list[dict] = []
    for sample in samples:
        text = sample.get("text", "")
        # Hash: first 50 chars + length + last 50 chars
        sig = f"{text[:50]}|{len(text)}|{text[-50:]}"
        h = hashlib.md5(sig.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(sample)
    return unique


def build_filtered_corpus(
    raw_samples: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    min_length: int = 30,
    max_length: int = 2000,
    min_confidence: float = 0.3,
    dedup: bool = True,
) -> dict[str, list[str]]:
    """Apply quality filters to raw scraper output and write JSONL files.

    Returns the cleaned corpus in load_corpus() format.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus: dict[str, list[str]] = {v: [] for v in ALL_VARIETIES}

    for dialect in ALL_VARIETIES:
        samples = raw_samples.get(dialect, [])

        # 1. Deduplicate
        if dedup:
            samples = _dedup_texts(samples)

        filtered: list[dict] = []
        for sample in samples:
            text = sample.get("text", "").strip()

            # 2. Clean OCR/subtitle artifacts
            text = _clean_ocr_artifacts(text)

            # 3. Length filter
            if len(text) < min_length or len(text) > max_length:
                continue

            # 4. Confidence filter
            conf = sample.get("confidence", 0.5)
            if conf < min_confidence:
                continue

            # 5. Language detection
            if not _detect_spanish(text):
                continue

            # 6. Boost with authenticity score
            auth = _compute_authenticity_score(text, dialect)
            sample_out = {
                "text": text,
                "dialect": dialect,
                "source": sample.get("source", "unknown"),
                "confidence": conf,
                "authenticity": auth,
            }
            filtered.append(sample_out)

        # Write JSONL
        jsonl_path = output_dir / f"{dialect}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for sample in filtered:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        corpus[dialect] = [s["text"] for s in filtered]
        logger.info("%s: %d -> %d samples (filtered)", dialect, len(samples), len(filtered))

    total = sum(len(docs) for docs in corpus.values())
    logger.info("Filtered corpus: %d total documents", total)
    return corpus


# ---------------------------------------------------------------------------
# Temperature-scaled balancing
# ---------------------------------------------------------------------------

def balance_corpus(
    corpus: dict[str, list[str]],
    temperature: float = 0.7,
    max_ratio: float = 3.0,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Temperature-scaled upsampling: n_target = n_max * (n_v / n_max)^T.

    T=1 -> no change, T=0 -> all equal, T=0.7 -> moderate upsampling.
    Never downsamples. Caps at max_ratio.
    """
    rng = random.Random(seed)
    sizes = {v: len(docs) for v, docs in corpus.items()}
    n_max = max(sizes.values()) if sizes else 0

    balanced: dict[str, list[str]] = {}
    for variety, docs in corpus.items():
        n_orig = len(docs)
        if n_orig == 0:
            balanced[variety] = []
            continue

        ratio = n_orig / n_max
        n_target = int(n_max * (ratio ** temperature))
        n_target = min(n_target, int(n_orig * max_ratio))
        n_target = max(n_target, n_orig)  # never downsample

        if n_target > n_orig:
            repeats = n_target // n_orig
            remainder = n_target % n_orig
            balanced_docs = docs * repeats + rng.sample(docs, remainder)
        else:
            balanced_docs = list(docs)

        balanced[variety] = balanced_docs
        if n_target != n_orig:
            logger.info(
                "%s: %d -> %d documents (%.1fx)",
                variety, n_orig, n_target, n_target / n_orig,
            )

    return balanced


# ---------------------------------------------------------------------------
# Affinity-based corpus blending
# ---------------------------------------------------------------------------

def blend_affine_varieties(
    corpus: dict[str, list[str]],
    pairs: Optional[list[tuple[str, str, float]]] = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Cross-pollinate documents between high-affinity variety pairs.

    For pair (A, B, frac): copy frac of A's docs into B and vice versa.
    Critical for CAN-CAR clustering.
    """
    if pairs is None:
        pairs = BLEND_PAIRS

    rng = random.Random(seed)
    blended = {v: list(docs) for v, docs in corpus.items()}

    for variety_a, variety_b, frac in pairs:
        docs_a = corpus.get(variety_a, [])
        docs_b = corpus.get(variety_b, [])
        if not docs_a or not docs_b:
            continue

        n_blend_a = max(1, int(len(docs_a) * frac))
        n_blend_b = max(1, int(len(docs_b) * frac))

        sample_a = rng.sample(docs_a, min(n_blend_a, len(docs_a)))
        sample_b = rng.sample(docs_b, min(n_blend_b, len(docs_b)))

        blended[variety_b].extend(sample_a)
        blended[variety_a].extend(sample_b)

        logger.info(
            "Blended %s <-> %s: %d + %d docs (%.0f%%)",
            variety_a, variety_b, n_blend_a, n_blend_b, frac * 100,
        )

    return blended


# ---------------------------------------------------------------------------
# Regionalism detection
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-záéíóúñüàèìòùâêîôûäëïöüç]+", re.IGNORECASE)


def _tokenize_simple(text: str) -> list[str]:
    """Extract lowercase alphabetic tokens."""
    return _WORD_RE.findall(text.lower())


def detect_regionalisms(
    corpus: dict[str, list[str]],
    method: str = "chi_squared",
    p_threshold: float = 0.01,
    min_count: int = 5,
) -> dict[str, set[str]]:
    """Detect regional words via chi-squared test + curated lists."""
    # Count word frequencies per variety
    variety_counts: dict[str, Counter] = {}
    for variety, docs in corpus.items():
        counter: Counter = Counter()
        for doc in docs:
            counter.update(_tokenize_simple(doc))
        variety_counts[variety] = counter

    # Start with curated lists
    result: dict[str, set[str]] = {v: set(REGIONALISMS.get(v, set())) for v in corpus}

    if method != "chi_squared":
        return result

    # Build vocabulary of words appearing in at least 2 varieties
    all_words: set[str] = set()
    for counter in variety_counts.values():
        all_words.update(counter.keys())

    varieties = list(corpus.keys())
    n_varieties = len(varieties)

    for word in all_words:
        counts = [variety_counts[v].get(word, 0) for v in varieties]
        total = sum(counts)
        if total < min_count * 2:
            continue

        expected = total / n_varieties
        if expected < 1:
            continue

        chi2_stat = sum((c - expected) ** 2 / expected for c in counts)
        p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=n_varieties - 1)

        if p_value < p_threshold:
            max_idx = max(range(n_varieties), key=lambda i: counts[i])
            result[varieties[max_idx]].add(word)

    for v in result:
        logger.info("%s: %d regionalisms detected", v, len(result[v]))

    return result


# ---------------------------------------------------------------------------
# Union vocabulary
# ---------------------------------------------------------------------------

def build_union_vocabulary(
    corpus: dict[str, list[str]],
    min_count: int = 3,
) -> list[str]:
    """Build sorted union vocabulary from all varieties."""
    counter: Counter = Counter()
    for docs in corpus.values():
        for doc in docs:
            counter.update(_tokenize_simple(doc))

    vocab = sorted(w for w, c in counter.items() if c >= min_count)
    logger.info("Union vocabulary: %d words (min_count=%d)", len(vocab), min_count)
    return vocab


# ---------------------------------------------------------------------------
# Synthetic phonological augmentation (PEN → CAN/AND)
# ---------------------------------------------------------------------------

def _apply_phonological_rules(text: str, dialect: str) -> str:
    """Apply phonological transformation rules to approximate a dialect.

    Transforms standard Peninsular text into a phonological approximation
    of the target dialect using regex-based sound change rules.
    """
    rules = PHONOLOGICAL_RULES.get(dialect, [])
    result = text
    for pattern, replacement, _desc in rules:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _apply_lexical_substitutions(text: str, dialect: str) -> str:
    """Apply lexical substitutions for dialect-specific vocabulary."""
    if dialect == "ES_CAN":
        lexicon = CANARIAN_LEXICON
    elif dialect == "ES_AND":
        lexicon = USTEDES_MAP
    else:
        return text

    words = text.split()
    result: list[str] = []
    for word in words:
        lower = word.lower()
        replacement = lexicon.get(lower)
        if replacement:
            # Preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(word)
    return " ".join(result)


def generate_synthetic_variants(
    corpus: dict[str, list[str]],
    target_dialects: list[str] | None = None,
    source_dialect: str = "ES_PEN",
    fraction: float = 0.3,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate synthetic dialect samples by transforming PEN texts.

    Takes Peninsular Spanish texts and applies phonological rules +
    lexical substitutions to approximate Canarian and Andalusian speech.
    This augments underrepresented dialects that share .es TLD and
    can't be easily sourced via web crawl.

    Parameters
    ----------
    corpus : dict
        Current corpus (modified in-place by appending synthetic texts).
    target_dialects : list[str] | None
        Which dialects to generate for. Defaults to ["ES_CAN", "ES_AND"].
    source_dialect : str
        Source dialect to transform from (default: ES_PEN).
    fraction : float
        Fraction of source docs to transform (0.3 = 30%).
    seed : int
        Random seed for sampling.

    Returns
    -------
    dict[str, list[str]]
        Corpus with synthetic variants appended.
    """
    if target_dialects is None:
        target_dialects = ["ES_CAN", "ES_AND"]

    rng = random.Random(seed)
    source_docs = corpus.get(source_dialect, [])
    if not source_docs:
        logger.warning("No source docs for synthetic augmentation from %s", source_dialect)
        return corpus

    n_sample = max(1, int(len(source_docs) * fraction))
    sampled = rng.sample(source_docs, min(n_sample, len(source_docs)))

    augmented = {v: list(docs) for v, docs in corpus.items()}

    for dialect in target_dialects:
        if dialect not in PHONOLOGICAL_RULES:
            continue

        synthetic: list[str] = []
        for text in sampled:
            # Apply phonological rules
            transformed = _apply_phonological_rules(text, dialect)
            # Apply lexical substitutions
            transformed = _apply_lexical_substitutions(transformed, dialect)
            # Only keep if meaningfully different from source
            if transformed != text and len(transformed) >= 20:
                synthetic.append(transformed)

        augmented[dialect].extend(synthetic)
        logger.info(
            "Synthetic %s: generated %d samples from %s (%.0f%% of source)",
            dialect, len(synthetic), source_dialect, fraction * 100,
        )

    return augmented


# ---------------------------------------------------------------------------
# One-call orchestrator
# ---------------------------------------------------------------------------

def _clean_tweet(text: str) -> str:
    """Clean a tweet for corpus inclusion: strip URLs, @mentions, RT prefix."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"^RT\s+@?\w+:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _compute_authenticity_score_enhanced(text: str, dialect: str) -> float:
    """Enhanced authenticity: regionalisms + phonological pattern detection."""
    words = set(text.lower().split())
    dialect_words = REGIONALISMS.get(dialect, set())
    if not dialect_words:
        return 0.0

    # Base: regionalism hits
    n_hits = len(words & dialect_words)
    score = n_hits / 2.0

    # Bonus: phonological patterns (for AND/CAN)
    if dialect in ("ES_AND", "ES_CAN"):
        text_lower = text.lower()
        # s-aspiration: words ending in -ao (cansao, helao)
        aspiration = len(re.findall(r"\b\w+ao\b", text_lower))
        # d-dropping: words ending in -á (verdá)
        d_drop = len(re.findall(r"\b\w+á\b", text_lower))
        score += (aspiration + d_drop) * 0.3

    return min(1.0, score)


def download_and_build(
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    sources: list[str] | None = None,
    max_per_dialect: int = 200_000,
    temperature: float = 0.7,
    seed: int = 42,
    hf_token: str | None = None,
    merge_existing: bool = True,
) -> dict[str, list[str]]:
    """One-call entry: download bulk datasets -> filter -> balance -> blend -> JSONL.

    Replaces scrape_and_build() with bulk dataset acquisition.
    Returns the final corpus dict ready for embedding training.
    """
    from eigen3.downloader import CorpusDownloader

    output_dir = Path(output_dir)
    if cache_dir is None:
        cache_dir = output_dir.parent / "raw_v4"
    cache_dir = Path(cache_dir)

    # Resolve HF token from param, env var, or .env file
    if hf_token is None:
        import os
        hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        env_file = Path(output_dir).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    logger.info("Starting bulk corpus download")

    # 1. Download
    downloader = CorpusDownloader(cache_dir=cache_dir, hf_token=hf_token)
    raw = downloader.download_all(sources=sources, max_per_dialect=max_per_dialect)

    # 1.5 Merge existing corpus if available
    if merge_existing:
        for existing_dir_name in ("processed_v3", "processed"):
            existing_dir = output_dir.parent / existing_dir_name
            if existing_dir.exists():
                logger.info("Merging existing corpus from %s", existing_dir)
                existing = load_corpus(existing_dir)
                for dialect, docs in existing.items():
                    for doc in docs:
                        raw.setdefault(dialect, []).append({
                            "text": doc,
                            "source": f"existing:{existing_dir_name}",
                            "confidence": 0.65,
                        })
                break  # Only merge from one source

    # 2. Filter
    corpus = build_filtered_corpus(raw, output_dir)

    # 3. Synthetic augmentation (CAN fraction boosted to 0.5)
    corpus = generate_synthetic_variants(
        corpus, target_dialects=["ES_CAN", "ES_AND"],
        fraction=0.5, seed=seed,
    )

    # 4. Balance
    corpus = balance_corpus(corpus, temperature=temperature, seed=seed)

    # 5. Blend affine varieties
    corpus = blend_affine_varieties(corpus, seed=seed)

    logger.info("Corpus build complete: %d total docs",
                sum(len(d) for d in corpus.values()))
    return corpus


def scrape_and_build(
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    sources: list[str] | None = None,
    temperature: float = 0.7,
    seed: int = 42,
) -> dict[str, list[str]]:
    """One-call entry point: scrape -> filter -> balance -> blend -> write JSONL.

    Returns the final corpus dict ready for embedding training.
    """
    from eigen3.scraper import CorpusScraper

    output_dir = Path(output_dir)
    if cache_dir is None:
        cache_dir = output_dir.parent / "raw"
    cache_dir = Path(cache_dir)

    logger.info("Starting automatic corpus build")

    # 1. Scrape
    scraper = CorpusScraper(cache_dir=cache_dir)
    raw = scraper.scrape_all(sources=sources)

    # 2. Filter
    corpus = build_filtered_corpus(raw, output_dir)

    # 3. Synthetic augmentation for underrepresented dialects (CAN, AND)
    corpus = generate_synthetic_variants(corpus, seed=seed)

    # 4. Balance
    corpus = balance_corpus(corpus, temperature=temperature, seed=seed)

    # 5. Blend affine varieties
    corpus = blend_affine_varieties(corpus, seed=seed)

    logger.info("Corpus build complete: %d total docs",
                sum(len(d) for d in corpus.values()))
    return corpus
