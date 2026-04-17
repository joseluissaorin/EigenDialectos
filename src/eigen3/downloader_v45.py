"""v4.5 high-quality source downloaders.

New sources with explicit region metadata:
- johnatanebonilla/tweet_hisp: 217M tweets with country+city fields
- marianbasti/preseea: 644 sociolinguistic interviews with city codes
- cladsu/COSER-2024: 149K rural interview turns with provincia
- StephannyPulido/corpus_registro_canarias: Lanzarote Cabildo transcripts
- projecte-aina/cv17_es_other_automatically_verified: Common Voice with accent field
- Parlamento de Canarias: PDF scrape of Diario de Sesiones

Each function is independently runnable so multiple sources can download in parallel.
Output format matches CorpusDownloader: dict[str, list[dict]] keyed by dialect code,
with each entry as {"text": str, "source": str, "confidence": float}.

Caches are written to data/raw_v4/{source}/{dialect}.jsonl (same convention as v4).
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from eigen3.constants import (
    ALL_VARIETIES,
    ANDALUSIAN_CITIES,
    CANARIAN_CITIES,
    CV_ACCENT_SUBSTRING_TO_DIALECT,
    HF_COSER_DATASET,
    HF_CV17_ES_DATASET,
    HF_LANZAROTE_DATASET,
    HF_PRESEEA_DATASET,
    HF_TWEET_HISP_DATASET,
    PARCAN_BASE_URL,
    PARCAN_MAX_LEGISLATURE,
    PARCAN_MAX_SESSIONS_PER_LEG,
    SOURCE_CONFIDENCE,
    SPAIN_PROVINCE_TO_DIALECT,
    TWEET_HISP_COUNTRY_TO_DIALECT,
)

logger = logging.getLogger(__name__)

# Tweet cleaning patterns (same as v4)
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_RT_RE = re.compile(r"^RT[\s:]*", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _clean_tweet(text: str) -> str:
    """Strip URLs, mentions, RT prefix; normalize whitespace."""
    text = _RT_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _HASHTAG_RE.sub(r"\1", text)  # Keep hashtag text, drop #
    text = _WS_RE.sub(" ", text).strip()
    return text


def _normalize_field(value: Any) -> str:
    """Normalize a metadata field value to lowercase stripped string."""
    if value is None:
        return ""
    return str(value).lower().strip()


def _save_cache(
    cache_dir: Path,
    source: str,
    data: dict[str, list[dict[str, Any]]],
) -> None:
    """Save downloaded data to cache as JSONL files per dialect."""
    cache_source_dir = cache_dir / source
    cache_source_dir.mkdir(parents=True, exist_ok=True)
    for dialect, samples in data.items():
        if not samples:
            continue
        cache_file = cache_source_dir / f"{dialect}.jsonl"
        with cache_file.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    total = sum(len(s) for s in data.values())
    logger.info("Cached %s: %d total docs", source, total)


def _load_cache(
    cache_dir: Path,
    source: str,
) -> dict[str, list[dict[str, Any]]] | None:
    """Load cached data if available."""
    cache_source_dir = cache_dir / source
    if not cache_source_dir.exists():
        return None
    result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}
    found_any = False
    for dialect in ALL_VARIETIES:
        cache_file = cache_source_dir / f"{dialect}.jsonl"
        if cache_file.exists():
            found_any = True
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        result[dialect].append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    if not found_any:
        return None
    total = sum(len(s) for s in result.values())
    if total == 0:
        return None
    logger.info("Cache hit: %s (%d docs)", source, total)
    return result


# ======================================================================
# 1. johnatanebonilla/tweet_hisp — 217M tweets with country+city
# ======================================================================

def _city_to_dialect(city: str, country_dialect: str | None) -> str | None:
    """Refine country-level dialect by city — upgrades PEN to CAN/AND when applicable."""
    if not city:
        return country_dialect
    city_lower = city.lower().strip()
    # Intra-Spain refinement
    if country_dialect == "ES_PEN":
        if city_lower in CANARIAN_CITIES:
            return "ES_CAN"
        if city_lower in ANDALUSIAN_CITIES:
            return "ES_AND"
        # Also check if city matches a province name
        prov_dialect = SPAIN_PROVINCE_TO_DIALECT.get(city_lower)
        if prov_dialect:
            return prov_dialect
    return country_dialect


def download_tweet_hisp(
    cache_dir: Path,
    hf_token: str | None = None,
    max_per_dialect: int = 200_000,
) -> dict[str, list[dict[str, Any]]]:
    """Download tweet_hisp — 217M geolocated Spanish tweets with country+city fields.

    The dataset is organized as 75 parquet files where each file contains a single
    country. Strategy: download each parquet, peek at its country, skip if that
    country's dialect is already capped, otherwise iterate it locally with pyarrow
    column projection (only tweet/country/city).
    """
    cached = _load_cache(cache_dir, "tweet_hisp")
    if cached:
        return cached

    try:
        from huggingface_hub import list_repo_files, snapshot_download
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("huggingface_hub + pyarrow required")
        return {v: [] for v in ALL_VARIETIES}

    logger.info("Listing tweet_hisp parquet files...")
    try:
        files = list_repo_files(HF_TWEET_HISP_DATASET, repo_type="dataset", token=hf_token)
    except Exception as e:
        logger.error("Failed to list tweet_hisp files: %s", e)
        return {v: [] for v in ALL_VARIETIES}

    parquet_files = sorted(
        [f for f in files if f.endswith(".parquet")],
        key=lambda f: int(re.search(r"_(\d+)\.parquet$", f).group(1)) if re.search(r"_(\d+)\.parquet$", f) else 0,
    )
    logger.info("tweet_hisp: %d parquet files; downloading in parallel...", len(parquet_files))

    # Parallel download all parquets at once (much faster than sequential)
    try:
        snapshot_dir = snapshot_download(
            repo_id=HF_TWEET_HISP_DATASET,
            repo_type="dataset",
            token=hf_token,
            allow_patterns=["*.parquet"],
            max_workers=8,
        )
        logger.info("tweet_hisp: snapshot_download complete -> %s", snapshot_dir)
    except Exception as e:
        logger.error("Snapshot download failed: %s", e)
        return {v: [] for v in ALL_VARIETIES}

    snapshot_path = Path(snapshot_dir)

    result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}
    counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
    total_processed = 0
    files_skipped = 0
    files_processed = 0

    for fname in parquet_files:
        # Stop if all dialects are full
        if all(counts[v] >= max_per_dialect for v in ALL_VARIETIES):
            logger.info("tweet_hisp: all dialects capped, stopping")
            break

        local_path = snapshot_path / fname
        if not local_path.exists():
            logger.warning("Missing parquet: %s", local_path)
            continue

        try:
            pf = pq.ParquetFile(str(local_path))
        except Exception as e:
            logger.warning("Failed to open %s: %s", fname, e)
            continue

        # Peek at first row to get country
        try:
            first_batch = next(pf.iter_batches(batch_size=1, columns=["country"]))
            peek_country = _normalize_field(first_batch.column("country")[0].as_py())
        except Exception:
            peek_country = ""

        peek_dialect = TWEET_HISP_COUNTRY_TO_DIALECT.get(peek_country)
        # For Spain, must process even if PEN is full (need CAN/AND via city refinement)
        if peek_country == "spain":
            spain_full = (
                counts["ES_PEN"] >= max_per_dialect
                and counts["ES_CAN"] >= max_per_dialect
                and counts["ES_AND"] >= max_per_dialect
            )
            if spain_full:
                logger.info("tweet_hisp: skip %s (Spain - all PEN/CAN/AND capped)", fname)
                files_skipped += 1
                continue
        elif not peek_dialect:
            logger.info("tweet_hisp: skip %s (country=%s unmapped)", fname, peek_country)
            files_skipped += 1
            continue
        elif counts[peek_dialect] >= max_per_dialect:
            logger.info("tweet_hisp: skip %s (%s -> %s, capped)", fname, peek_country, peek_dialect)
            files_skipped += 1
            continue

        files_processed += 1
        logger.info("tweet_hisp: processing %s (country=%s, %d rows)",
                    fname, peek_country, pf.metadata.num_rows)

        # Iterate in batches with column projection
        try:
            for batch in pf.iter_batches(batch_size=50_000, columns=["tweet", "country", "city"]):
                texts = batch.column("tweet")
                countries = batch.column("country")
                cities = batch.column("city")
                n_rows = len(texts)

                for i in range(n_rows):
                    total_processed += 1

                    country = _normalize_field(countries[i].as_py())
                    if not country:
                        continue
                    base_dialect = TWEET_HISP_COUNTRY_TO_DIALECT.get(country)
                    if not base_dialect:
                        continue

                    city = _normalize_field(cities[i].as_py())
                    dialect = _city_to_dialect(city, base_dialect) or base_dialect

                    if counts[dialect] >= max_per_dialect:
                        # If Spain (PEN base), still try city refinement to other dialects
                        if base_dialect == "ES_PEN" and dialect == "ES_PEN":
                            continue
                        # Otherwise stop processing this batch's country if all relevant capped
                        continue

                    text = texts[i].as_py()
                    if not text:
                        continue
                    text = text.strip()
                    if len(text) < 15:
                        continue

                    cleaned = _clean_tweet(text)
                    if len(cleaned) < 15:
                        continue

                    result[dialect].append({
                        "text": cleaned,
                        "source": f"tweet_hisp:{country}:{city}" if city else f"tweet_hisp:{country}",
                        "confidence": SOURCE_CONFIDENCE["tweet_hisp_city"],
                    })
                    counts[dialect] += 1

                # Per-batch progress
                if total_processed % 200_000 < 50_000:
                    logger.info("tweet_hisp [%s]: %d processed, counts: %s",
                                fname.replace("tweet_hisp_", "").replace(".parquet", ""),
                                total_processed,
                                {k: v for k, v in counts.items() if v > 0})

                # If all dialects this country could fill are full, break early
                if peek_country == "spain":
                    if (counts["ES_PEN"] >= max_per_dialect
                            and counts["ES_CAN"] >= max_per_dialect
                            and counts["ES_AND"] >= max_per_dialect):
                        break
                else:
                    if peek_dialect and counts[peek_dialect] >= max_per_dialect:
                        break
        except Exception as e:
            logger.warning("Iteration failed for %s: %s", fname, e)
            continue

    logger.info("tweet_hisp done: %d total (processed=%d rows, files=%d, skipped=%d)",
                sum(counts.values()), total_processed, files_processed, files_skipped)
    for d in sorted(ALL_VARIETIES):
        if counts[d] > 0:
            logger.info("  %s: %d", d, counts[d])

    _save_cache(cache_dir, "tweet_hisp", result)
    return result


# ======================================================================
# 2. marianbasti/preseea — 644 sociolinguistic interviews
# ======================================================================

# PRESEEA country (CSV column) → dialect
_PRESEEA_COUNTRY_TO_DIALECT: dict[str, str] = {
    "argentina": "ES_RIO",
    "uruguay": "ES_RIO",
    "bolivia": "ES_AND_BO",
    "perú": "ES_AND_BO",
    "peru": "ES_AND_BO",
    "ecuador": "ES_AND_BO",
    "chile": "ES_CHI",
    "méxico": "ES_MEX",
    "mexico": "ES_MEX",
    "guatemala": "ES_MEX",
    "cuba": "ES_CAR",
    "venezuela": "ES_CAR",
    "colombia": "ES_CAR",
    "puerto rico": "ES_CAR",
    "república dominicana": "ES_CAR",
}

# PRESEEA Spanish city (from XML ciudad attribute) → dialect
_PRESEEA_ES_CITY_TO_DIALECT: dict[str, str] = {
    "las palmas": "ES_CAN",
    "las palmas de gran canaria": "ES_CAN",
    "málaga": "ES_AND",
    "malaga": "ES_AND",
    "sevilla": "ES_AND",
    "granada": "ES_AND",
    "cádiz": "ES_AND",
    "cadiz": "ES_AND",
    "huelva": "ES_AND",
    "córdoba": "ES_AND",
    "cordoba": "ES_AND",
    "almería": "ES_AND",
    "almeria": "ES_AND",
    "jaén": "ES_AND",
    "jaen": "ES_AND",
    # Peninsular default
    "madrid": "ES_PEN",
    "alcalá": "ES_PEN",
    "alcalá de henares": "ES_PEN",
    "santander": "ES_PEN",
    "barcelona": "ES_PEN",
    "gijón": "ES_PEN",
    "gijon": "ES_PEN",
    "santiago de compostela": "ES_PEN",
    "valencia": "ES_PEN",
    "palma": "ES_PEN",
    "palma de mallorca": "ES_PEN",
}

# PRESEEA transcription cleanup patterns
_PRESEEA_XML_HEADER_RE = re.compile(r"<Trans[^>]*?>.*?</Trans>", re.DOTALL)
_PRESEEA_TAG_RE = re.compile(r"<[^>]+>")
_PRESEEA_TIEMPO_RE = re.compile(r"<tiempo[^/]*/>")
_PRESEEA_SPEAKER_RE = re.compile(r"(?:^|\s)([EIA]\d*):\s*")
_PRESEEA_PARENS_COMMENT_RE = re.compile(r"\(\(([^)]*)\)\)")  # (( ... )) unclear speech
_PRESEEA_SLASH_RE = re.compile(r"\s*/+\s*")  # PRESEEA pauses /, //, ///
_PRESEEA_MARKER_WORDS_RE = re.compile(
    r"<(?:vacilación|simultáneo|risa|risas|silencio|ininteligible|ruido|vacilacion)/?>",
    re.IGNORECASE,
)


def _clean_preseea_text(raw: str) -> tuple[str, list[str]]:
    """Strip PRESEEA XML header + transcription markers.

    Returns (city_hint, list_of_utterances).
    """
    # Extract ciudad from XML header
    ciudad_match = re.search(r'ciudad="([^"]+)"', raw)
    ciudad = ciudad_match.group(1).strip() if ciudad_match else ""

    # Remove XML header block
    body = re.sub(r"<Trans[^>]*>.*?</Hablantes>\s*</Trans>", "", raw, flags=re.DOTALL)
    if "</Trans>" in body:
        body = body.split("</Trans>", 1)[-1]

    # Remove tiempo markers
    body = _PRESEEA_TIEMPO_RE.sub(" ", body)
    # Remove PRESEEA marker tags
    body = _PRESEEA_MARKER_WORDS_RE.sub(" ", body)
    # Remove any remaining XML-ish tags
    body = _PRESEEA_TAG_RE.sub(" ", body)
    # Normalize ((comments))
    body = _PRESEEA_PARENS_COMMENT_RE.sub(r"\1", body)

    # Split on speaker tags "E:", "I:", "A:", etc.
    utterances: list[str] = []
    buf = ""
    for chunk in re.split(r"\s+([EIA]\d*):\s*", body):
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.fullmatch(r"[EIA]\d*", chunk):
            # Speaker tag — flush buf
            if buf:
                utterances.append(buf.strip())
                buf = ""
        else:
            buf += " " + chunk
    if buf:
        utterances.append(buf.strip())

    # Clean slashes and normalize whitespace
    cleaned: list[str] = []
    for u in utterances:
        u = _PRESEEA_SLASH_RE.sub(" ", u)
        u = _WS_RE.sub(" ", u).strip()
        if len(u) >= 20:
            cleaned.append(u)
    return ciudad, cleaned


def download_preseea(
    cache_dir: Path,
    hf_token: str | None = None,
    max_per_dialect: int = 50_000,
) -> dict[str, list[dict[str, Any]]]:
    """Download PRESEEA corpus — 627 sociolinguistic interviews.

    Reads metadata.csv directly from the HF repo, bypassing the audio column
    (which triggers torchcodec decoding in the datasets library).
    """
    cached = _load_cache(cache_dir, "preseea")
    if cached:
        return cached

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub library required")
        return {v: [] for v in ALL_VARIETIES}

    logger.info("Downloading PRESEEA metadata.csv...")
    try:
        csv_path = hf_hub_download(
            repo_id=HF_PRESEEA_DATASET,
            filename="metadata.csv",
            repo_type="dataset",
            token=hf_token,
        )
    except Exception as e:
        logger.error("Failed to download PRESEEA metadata: %s", e)
        return {v: [] for v in ALL_VARIETIES}

    import csv as _csv
    import sys as _sys
    _csv.field_size_limit(_sys.maxsize)

    result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}
    counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
    processed = 0
    unmatched = 0
    unmatched_countries: set[str] = set()

    with open(csv_path, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            processed += 1
            if processed % 50 == 0:
                logger.info("PRESEEA: %d interviews, counts: %s",
                            processed,
                            {k: v for k, v in counts.items() if v > 0})

            country = _normalize_field(row.get("country"))
            text = row.get("text", "") or ""
            file_name = row.get("file_name", "") or ""

            if not text or len(text) < 100:
                continue

            # Clean + segment the transcription
            ciudad, utterances = _clean_preseea_text(text)
            if not utterances:
                continue

            # Resolve dialect: city-level first (for Spain), then country-level
            dialect: str | None = None
            if country == "españa":
                ciudad_lower = ciudad.lower().strip()
                # Exact match first
                dialect = _PRESEEA_ES_CITY_TO_DIALECT.get(ciudad_lower)
                if not dialect:
                    # Substring match
                    for key, d in _PRESEEA_ES_CITY_TO_DIALECT.items():
                        if key in ciudad_lower:
                            dialect = d
                            break
                if not dialect:
                    dialect = "ES_PEN"  # Spain fallback
            else:
                dialect = _PRESEEA_COUNTRY_TO_DIALECT.get(country)

            if not dialect:
                unmatched += 1
                unmatched_countries.add(country)
                continue

            if counts[dialect] >= max_per_dialect:
                continue

            src_label = f"preseea:{country}:{ciudad}" if ciudad else f"preseea:{country}"
            # Emit utterances as separate segments, merging short ones into larger chunks
            chunk_buf = ""
            for utt in utterances:
                if counts[dialect] >= max_per_dialect:
                    break
                if len(chunk_buf) + len(utt) + 1 > 1500:
                    if chunk_buf:
                        result[dialect].append({
                            "text": chunk_buf.strip()[:2000],
                            "source": src_label,
                            "confidence": SOURCE_CONFIDENCE["preseea"],
                        })
                        counts[dialect] += 1
                    chunk_buf = utt
                else:
                    chunk_buf = f"{chunk_buf} {utt}".strip() if chunk_buf else utt
            if chunk_buf and counts[dialect] < max_per_dialect:
                result[dialect].append({
                    "text": chunk_buf.strip()[:2000],
                    "source": src_label,
                    "confidence": SOURCE_CONFIDENCE["preseea"],
                })
                counts[dialect] += 1

    if unmatched_countries:
        logger.info("PRESEEA unmatched countries: %s", sorted(unmatched_countries))
    logger.info("PRESEEA done: %d segments from %d interviews (unmatched=%d)",
                sum(counts.values()), processed, unmatched)
    for d in sorted(ALL_VARIETIES):
        if counts[d] > 0:
            logger.info("  %s: %d", d, counts[d])

    _save_cache(cache_dir, "preseea", result)
    return result


# ======================================================================
# 3. cladsu/COSER-2024 — 149K rural interview turns with provincia
# ======================================================================

def download_coser(
    cache_dir: Path,
    hf_token: str | None = None,
    max_per_dialect: int = 50_000,
) -> dict[str, list[dict[str, Any]]]:
    """Download COSER-2024 — 149K rural Spanish interview turns tagged by province."""
    cached = _load_cache(cache_dir, "coser")
    if cached:
        return cached

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library required")
        return {v: [] for v in ALL_VARIETIES}

    logger.info("Loading COSER-2024 from HuggingFace...")
    try:
        # Try loading all splits
        ds_dict = load_dataset(HF_COSER_DATASET, token=hf_token)
    except Exception as e:
        logger.error("Failed to load COSER: %s", e)
        return {v: [] for v in ALL_VARIETIES}

    result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}
    counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
    unmatched_provinces: set[str] = set()

    # Iterate over all splits (train/val/test)
    splits = list(ds_dict.keys()) if hasattr(ds_dict, "keys") else ["train"]
    for split_name in splits:
        ds = ds_dict[split_name] if hasattr(ds_dict, "__getitem__") else ds_dict
        logger.info("COSER split %s: %d rows", split_name, len(ds))

        for row in ds:
            # COSER fields: provincia, enclave, turno, texto (or similar)
            province = _normalize_field(
                row.get("provincia") or row.get("Provincia") or row.get("province"))
            text = ""
            for field in ("texto", "text", "turno", "turn", "content", "transcription"):
                val = row.get(field)
                if val:
                    text = str(val).strip()
                    break

            if not text or len(text) < 20:
                continue
            if not province:
                continue

            dialect = SPAIN_PROVINCE_TO_DIALECT.get(province)
            if not dialect:
                unmatched_provinces.add(province)
                continue

            if counts[dialect] >= max_per_dialect:
                continue

            # Strip COSER dialectal annotations like "ezo(=eso)" → keep both forms? Keep as-is.
            if len(text) > 2000:
                text = text[:2000]

            result[dialect].append({
                "text": text,
                "source": f"coser:{province}",
                "confidence": SOURCE_CONFIDENCE["coser"],
            })
            counts[dialect] += 1

    if unmatched_provinces:
        logger.info("COSER unmatched provinces: %s", sorted(unmatched_provinces)[:20])

    logger.info("COSER done: %d total", sum(counts.values()))
    for d in sorted(ALL_VARIETIES):
        if counts[d] > 0:
            logger.info("  %s: %d", d, counts[d])

    _save_cache(cache_dir, "coser", result)
    return result


# ======================================================================
# 4. StephannyPulido/corpus_registro_canarias — Lanzarote Cabildo
# ======================================================================

def download_lanzarote(
    cache_dir: Path,
    hf_token: str | None = None,
    max_per_dialect: int = 50_000,
) -> dict[str, list[dict[str, Any]]]:
    """Download Lanzarote Cabildo YouTube transcript corpus (pure ES_CAN)."""
    cached = _load_cache(cache_dir, "lanzarote")
    if cached:
        return cached

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library required")
        return {v: [] for v in ALL_VARIETIES}

    logger.info("Loading Lanzarote Cabildo corpus from HuggingFace...")
    try:
        ds_dict = load_dataset(HF_LANZAROTE_DATASET, token=hf_token)
    except Exception as e:
        logger.error("Failed to load Lanzarote corpus: %s", e)
        return {v: [] for v in ALL_VARIETIES}

    result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}
    count = 0

    splits = list(ds_dict.keys()) if hasattr(ds_dict, "keys") else ["train"]
    for split_name in splits:
        ds = ds_dict[split_name] if hasattr(ds_dict, "__getitem__") else ds_dict
        logger.info("Lanzarote split %s: %d rows", split_name, len(ds))

        for row in ds:
            # Text may be in Transcription_punct (punctuated) or Transcription
            text = ""
            for field in ("Transcription_punct", "transcription_punct",
                          "Transcription", "transcription", "text"):
                val = row.get(field)
                if val:
                    text = str(val).strip()
                    break

            if not text or len(text) < 50:
                continue

            # Segment into paragraphs (these are long transcripts)
            # Split by double newlines or long pauses
            segments = re.split(r"\n\n+|(?<=[.!?])\s{2,}", text)
            for seg in segments:
                seg = seg.strip()
                if len(seg) < 30 or len(seg) > 2000:
                    if len(seg) > 2000:
                        # Chunk further by sentences
                        sents = re.split(r"(?<=[.!?])\s+", seg)
                        buf = ""
                        for s in sents:
                            if len(buf) + len(s) > 1500:
                                if buf.strip():
                                    if count >= max_per_dialect:
                                        break
                                    result["ES_CAN"].append({
                                        "text": buf.strip(),
                                        "source": "lanzarote_cabildo",
                                        "confidence": SOURCE_CONFIDENCE["lanzarote_cabildo"],
                                    })
                                    count += 1
                                buf = s
                            else:
                                buf += " " + s
                        if buf.strip() and count < max_per_dialect:
                            result["ES_CAN"].append({
                                "text": buf.strip(),
                                "source": "lanzarote_cabildo",
                                "confidence": SOURCE_CONFIDENCE["lanzarote_cabildo"],
                            })
                            count += 1
                    continue

                if count >= max_per_dialect:
                    break

                result["ES_CAN"].append({
                    "text": seg,
                    "source": "lanzarote_cabildo",
                    "confidence": SOURCE_CONFIDENCE["lanzarote_cabildo"],
                })
                count += 1

            if count >= max_per_dialect:
                break

    logger.info("Lanzarote done: %d ES_CAN segments", count)
    _save_cache(cache_dir, "lanzarote", result)
    return result


# ======================================================================
# 5. projecte-aina/cv17_es_other_automatically_verified — Common Voice
# ======================================================================

def _cv_accent_to_dialect(accent: str) -> str | None:
    """Map a Common Voice accent label to a dialect code."""
    if not accent:
        return None
    accent_lower = accent.lower()
    for substring, dialect in CV_ACCENT_SUBSTRING_TO_DIALECT:
        if substring in accent_lower:
            return dialect
    return None


def download_cv17(
    cache_dir: Path,
    hf_token: str | None = None,
    max_per_dialect: int = 30_000,
) -> dict[str, list[dict[str, Any]]]:
    """Download Common Voice 17 Spanish (accent-tagged) from `other.tsv`.

    The HF dataset has a deprecated loader script, so we fetch corpus/files/other.tsv
    directly via huggingface_hub. This file already has all sentences + accents
    metadata (no audio needed).
    """
    cached = _load_cache(cache_dir, "cv17")
    if cached:
        return cached

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub library required")
        return {v: [] for v in ALL_VARIETIES}

    logger.info("Downloading CV17 other.tsv (accent-tagged sentences)...")
    try:
        tsv_path = hf_hub_download(
            repo_id=HF_CV17_ES_DATASET,
            filename="corpus/files/other.tsv",
            repo_type="dataset",
            token=hf_token,
        )
    except Exception as e:
        logger.error("Failed to download CV17 other.tsv: %s", e)
        return {v: [] for v in ALL_VARIETIES}

    result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}
    counts: dict[str, int] = {v: 0 for v in ALL_VARIETIES}
    processed = 0
    unmatched = 0
    seen_sentences: set[str] = set()  # In-source dedup (CV has many duplicate sentences)

    with open(tsv_path, encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        headers = header_line.split("\t")
        try:
            idx_sentence = headers.index("sentence")
            idx_accent = headers.index("accents")
        except ValueError as e:
            logger.error("CV17 unexpected headers: %s (%s)", headers, e)
            return result

        for line in f:
            processed += 1
            if processed % 50_000 == 0:
                logger.info("CV17: %d processed, counts: %s",
                            processed,
                            {k: v for k, v in counts.items() if v > 0})

            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(idx_sentence, idx_accent):
                continue

            sentence = parts[idx_sentence].strip()
            accent = parts[idx_accent].strip()

            if not sentence or len(sentence) < 15:
                continue

            dialect = _cv_accent_to_dialect(accent)
            if not dialect:
                unmatched += 1
                continue

            if counts[dialect] >= max_per_dialect:
                if all(c >= max_per_dialect for c in counts.values()):
                    break
                continue

            # Dedup
            sent_key = sentence[:120].lower()
            if sent_key in seen_sentences:
                continue
            seen_sentences.add(sent_key)

            result[dialect].append({
                "text": sentence,
                "source": f"cv17:{accent[:40]}",
                "confidence": SOURCE_CONFIDENCE["cv17_accent"],
            })
            counts[dialect] += 1

    logger.info("CV17 done: %d total (processed=%d, unmatched=%d)",
                sum(counts.values()), processed, unmatched)
    for d in sorted(ALL_VARIETIES):
        if counts[d] > 0:
            logger.info("  %s: %d", d, counts[d])

    _save_cache(cache_dir, "cv17", result)
    return result


# ======================================================================
# 6. Parlamento de Canarias — PDF Diario de Sesiones scrape
# ======================================================================

def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import pypdf
    except ImportError:
        try:
            import PyPDF2 as pypdf  # type: ignore
        except ImportError:
            logger.error("pypdf or PyPDF2 required for Parlamento de Canarias")
            return ""

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(pages)
    except Exception as e:
        logger.warning("PDF parse failed: %s", e)
        return ""


def _clean_parcan_text(text: str) -> list[str]:
    """Clean and segment a Parlamento de Canarias PDF text into paragraphs."""
    # Remove page headers/footers (session numbers, page refs)
    text = re.sub(r"Diario de Sesiones.*?\d+", " ", text)
    text = re.sub(r"Núm\.\s*\d+\s*/\s*\d+", " ", text)
    text = re.sub(r"Página\s*\d+", " ", text)
    # Remove speaker tags like "El señor XXX (socialista):" — keep content
    text = re.sub(r"^\s*(El señor|La señora|El\s|La\s)[^:]{2,80}:\s*", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Split into sentences/paragraphs
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])", text)
    # Buffer into paragraphs of ~500 chars
    paragraphs: list[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) > 800:
            if buf.strip() and len(buf.strip()) >= 50:
                paragraphs.append(buf.strip())
            buf = s
        else:
            buf += " " + s
    if buf.strip() and len(buf.strip()) >= 50:
        paragraphs.append(buf.strip())
    return paragraphs


def download_parcan(
    cache_dir: Path,
    hf_token: str | None = None,  # Unused, for API symmetry
    max_per_dialect: int = 100_000,
) -> dict[str, list[dict[str, Any]]]:
    """Scrape Parlamento de Canarias Diario de Sesiones PDFs.

    URL pattern: https://www.parcan.es/files/pub/diarios/{N}l/{NNN}/ds{NNN}.pdf
    N = legislature (1..11), NNN = session number (zero-padded to 3 digits).
    """
    cached = _load_cache(cache_dir, "parcan")
    if cached:
        return cached

    import requests

    pdf_cache = cache_dir / "parcan" / "pdfs"
    pdf_cache.mkdir(parents=True, exist_ok=True)

    result: dict[str, list[dict[str, Any]]] = {v: [] for v in ALL_VARIETIES}
    count = 0
    pdfs_downloaded = 0
    pdfs_failed = 0
    total_bytes = 0

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research/academic)"})

    for leg in range(1, PARCAN_MAX_LEGISLATURE + 1):
        if count >= max_per_dialect:
            break

        consecutive_failures = 0
        for session_num in range(1, PARCAN_MAX_SESSIONS_PER_LEG + 1):
            if count >= max_per_dialect:
                break
            # After 5 consecutive 404s, assume this legislature is exhausted
            if consecutive_failures >= 5:
                break

            nnn = f"{session_num:03d}"
            url = f"{PARCAN_BASE_URL}/{leg}l/{nnn}/ds{nnn}.pdf"
            local_pdf = pdf_cache / f"leg{leg:02d}_ds{nnn}.pdf"

            # Check cache first
            if local_pdf.exists() and local_pdf.stat().st_size > 1000:
                try:
                    pdf_bytes = local_pdf.read_bytes()
                    consecutive_failures = 0
                except Exception:
                    pdfs_failed += 1
                    consecutive_failures += 1
                    continue
            else:
                try:
                    resp = session.get(url, timeout=30)
                    if resp.status_code == 404:
                        consecutive_failures += 1
                        continue
                    if resp.status_code != 200:
                        consecutive_failures += 1
                        continue
                    pdf_bytes = resp.content
                    if len(pdf_bytes) < 1000:
                        consecutive_failures += 1
                        continue
                    local_pdf.write_bytes(pdf_bytes)
                    pdfs_downloaded += 1
                    total_bytes += len(pdf_bytes)
                    consecutive_failures = 0
                    # Polite rate limit
                    time.sleep(0.3)
                except Exception as e:
                    logger.debug("PDF download failed for %s: %s", url, e)
                    pdfs_failed += 1
                    consecutive_failures += 1
                    continue

            # Extract text
            text = _pdf_to_text(pdf_bytes)
            if not text or len(text) < 500:
                continue

            # Clean + segment
            paragraphs = _clean_parcan_text(text)
            for para in paragraphs:
                if count >= max_per_dialect:
                    break
                if len(para) > 2000:
                    para = para[:2000]
                result["ES_CAN"].append({
                    "text": para,
                    "source": f"parcan:leg{leg:02d}:ds{nnn}",
                    "confidence": SOURCE_CONFIDENCE["parcan"],
                })
                count += 1

            if pdfs_downloaded % 20 == 0 and pdfs_downloaded > 0:
                logger.info("parcan: %d PDFs downloaded (%.1f MB), %d segments",
                            pdfs_downloaded, total_bytes / 1e6, count)

    logger.info("parcan done: %d PDFs downloaded, %d failed, %d ES_CAN segments",
                pdfs_downloaded, pdfs_failed, count)
    _save_cache(cache_dir, "parcan", result)
    return result


# ======================================================================
# Source registry
# ======================================================================

V45_SOURCES: dict[str, Any] = {
    "tweet_hisp": download_tweet_hisp,
    "preseea": download_preseea,
    "coser": download_coser,
    "lanzarote": download_lanzarote,
    "cv17": download_cv17,
    "parcan": download_parcan,
}


def download_single_source(
    source: str,
    cache_dir: Path,
    hf_token: str | None = None,
    max_per_dialect: int = 50_000,
) -> dict[str, list[dict[str, Any]]]:
    """Dispatch to the correct download function by source name."""
    func = V45_SOURCES.get(source)
    if func is None:
        raise ValueError(f"Unknown v4.5 source: {source}")
    return func(cache_dir, hf_token, max_per_dialect)
