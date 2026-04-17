#!/usr/bin/env python3
"""Linguistic categorization of eigenvector axes for EigenDialectos.

Takes the eigenvector word projections and categorizes the top axes
into linguistic categories: lexical, morphosyntactic, phonological,
pragmatic, using curated word lists.

Outputs
-------
    outputs/analysis/linguistic_axes.json      — full categorization
    outputs/analysis/linguistic_axes_table.csv  — paper-ready table
    outputs/analysis/linguistic_axes.png        — visualization
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigendialectos.constants import DIALECT_NAMES, DialectCode

LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
logger = logging.getLogger("ling_cat")

ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"

DIALECT_CODES = [
    "ES_PEN", "ES_AND", "ES_CAN", "ES_RIO",
    "ES_MEX", "ES_CAR", "ES_CHI", "ES_AND_BO",
]

DIALECT_LABELS = {
    "ES_PEN": "Peninsular", "ES_AND": "Andaluz", "ES_CAN": "Canario",
    "ES_RIO": "Rioplatense", "ES_MEX": "Mexicano", "ES_CAR": "Caribeño",
    "ES_CHI": "Chileno", "ES_AND_BO": "Andino",
}

# ===================================================================
# Linguistic feature word lists
# ===================================================================

# Morphosyntactic markers: verb forms, pronouns, determiners, prepositions
MORPHOSYNTACTIC_MARKERS = {
    # Pronouns and address forms
    "vos", "tú", "usted", "ustedes", "vosotros", "vosotras",
    "os", "te", "le", "les", "lo", "la", "los", "las", "nos",
    "me", "se", "mí", "ti", "sí", "consigo", "conmigo", "contigo",
    # Verb endings and conjugated forms (high-frequency)
    "estamos", "estás", "estáis", "están", "estoy",
    "tenemos", "tenés", "tenéis", "tienen", "tienes", "tengo",
    "vamos", "vas", "vais", "van", "voy",
    "somos", "sois", "sos", "son", "eres", "soy",
    "hemos", "habéis", "han", "has", "he",
    "podemos", "podéis", "pueden", "puedes", "puedo",
    "queremos", "queréis", "quieren", "quieres", "quiero",
    "sabemos", "sabéis", "saben", "sabes", "sé",
    "decimos", "decís", "dicen", "dices", "digo",
    # Morphological elements
    "ido", "dado", "sido", "hecho", "dicho", "puesto", "visto",
    "ando", "iendo", "mente",
    # Articles / demonstratives
    "este", "esta", "esto", "estos", "estas",
    "ese", "esa", "eso", "esos", "esas",
    "aquel", "aquella", "aquello", "aquellos",
    "el", "la", "lo", "los", "las", "un", "una", "unos", "unas",
    # Relative / interrogative
    "que", "quien", "cual", "cuyo", "donde", "cuando", "como",
    "qué", "quién", "cuál",
    # Connectors / prepositions
    "de", "en", "por", "para", "con", "sin", "sobre", "entre",
    "desde", "hacia", "hasta", "según", "tras", "ante", "bajo",
}

# Phonological reflexes: words whose form differs across dialects
# due to seseo, ceceo, aspiración, yeísmo, etc.
PHONOLOGICAL_MARKERS = {
    # S/Z/C contrasts (seseo/ceceo/distinción)
    "corazón", "cabeza", "zapato", "cerveza", "cielo", "cien",
    "entonces", "necesario", "necesidad", "gracias", "servicio",
    "acción", "estación", "nación", "situación", "dirección",
    "educación", "generación", "relación", "posición",
    # Aspiration of /s/
    "mismo", "estos", "esos", "otros", "dos", "tres",
    "después", "más", "menos", "además",
    # Yeísmo (ll/y merger)
    "calle", "ella", "llegar", "llevar", "llamar", "lluvia",
    "allí", "aquello", "caballo", "pollo",
    # /d/ weakening (intervocalic)
    "todo", "nada", "cada", "lado", "pasado", "estado",
    "ciudad", "verdad", "libertad", "universidad",
    # Word-final consonant processes
    "ser", "hacer", "poder", "saber", "tener", "poner",
    "comer", "vivir", "salir", "venir",
    "general", "natural", "social", "cultural", "actual",
}

# Lexical markers: words with known dialectal alternates
LEXICAL_MARKERS = {
    # Transport
    "coche", "carro", "auto", "guagua", "bus", "colectivo", "micro",
    # Food
    "patata", "papa", "plátano", "banana", "judías", "frijoles",
    "melocotón", "durazno", "zumo", "jugo", "tortilla",
    # Everyday objects
    "ordenador", "computadora", "computador", "móvil", "celular",
    "piso", "apartamento", "departamento", "nevera", "heladera",
    # Actions
    "coger", "tomar", "agarrar", "conducir", "manejar",
    "enfadarse", "enojarse", "aparcar", "estacionar",
    # People
    "chaval", "chavo", "pibe", "niño", "muchacho", "chico",
    "guapa", "linda", "bonita", "hermosa",
    # Time/Frequency
    "vale", "bueno", "dale", "órale", "pues", "entonces",
}

# Pragmatic markers and discourse particles
PRAGMATIC_MARKERS = {
    # Interjections / fillers
    "bueno", "pues", "vale", "venga", "vamos", "dale", "órale",
    "mira", "oye", "oiga", "mijo", "mija", "che", "boludo",
    "güey", "compa", "viejo", "loco", "tío", "tía",
    # Discourse markers
    "entonces", "además", "también", "tampoco", "aunque",
    "claro", "exacto", "verdad", "cierto", "fíjate",
    # Hedges and softeners
    "quizás", "quizá", "tal vez", "acaso", "capaz",
    "parece", "creo", "pienso", "siento",
    # Address / politeness
    "señor", "señora", "don", "doña", "maestro",
    "Vamos", "Mira",
}

# All categories in one lookup
CATEGORY_LOOKUP: dict[str, str] = {}
for word in MORPHOSYNTACTIC_MARKERS:
    CATEGORY_LOOKUP[word.lower()] = "morphosyntactic"
for word in PHONOLOGICAL_MARKERS:
    if word.lower() not in CATEGORY_LOOKUP:
        CATEGORY_LOOKUP[word.lower()] = "phonological"
for word in LEXICAL_MARKERS:
    if word.lower() not in CATEGORY_LOOKUP:
        CATEGORY_LOOKUP[word.lower()] = "lexical"
for word in PRAGMATIC_MARKERS:
    if word.lower() not in CATEGORY_LOOKUP:
        CATEGORY_LOOKUP[word.lower()] = "pragmatic"

CATEGORY_COLORS = {
    "morphosyntactic": "#E74C3C",
    "phonological": "#3498DB",
    "lexical": "#2ECC71",
    "pragmatic": "#F39C12",
    "other": "#95A5A6",
}


def categorize_word(word: str) -> str:
    """Categorize a word into a linguistic category."""
    w = word.lower().strip()
    if w in CATEGORY_LOOKUP:
        return CATEGORY_LOOKUP[w]

    # Heuristic rules for uncategorized words
    # Words ending in -ción, -sión (phonological: s/z contrast)
    if w.endswith("ción") or w.endswith("sión"):
        return "phonological"
    # Words ending in -mente (morphosyntactic: adverb formation)
    if w.endswith("mente"):
        return "morphosyntactic"
    # Past participles (-ado, -ido)
    if w.endswith("ado") or w.endswith("ido") or w.endswith("ada") or w.endswith("ida"):
        return "morphosyntactic"
    # Gerunds (-ando, -iendo)
    if w.endswith("ando") or w.endswith("iendo"):
        return "morphosyntactic"
    # Infinitives (-ar, -er, -ir)
    if len(w) > 3 and (w.endswith("ar") or w.endswith("er") or w.endswith("ir")):
        return "morphosyntactic"
    # Diminutives (-ito, -ita, -ico, -ica)
    if w.endswith("ito") or w.endswith("ita") or w.endswith("ico") or w.endswith("ica"):
        return "lexical"  # diminutive usage is a lexical/dialectal choice

    return "other"


def load_eigenvector_words() -> dict[str, list[dict]]:
    """Load eigenvector word projection data for all dialects."""
    all_data = {}
    for code in DIALECT_CODES:
        path = ANALYSIS_DIR / f"eigenvector_words_{code}.json"
        if path.exists():
            with open(path) as f:
                all_data[code] = json.load(f)
    return all_data


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading eigenvector word projections...")
    eig_words = load_eigenvector_words()
    if not eig_words:
        logger.error("No eigenvector word data found in %s", ANALYSIS_DIR)
        return

    # ---------------------------------------------------------------
    # Categorize top words for each eigenvector of each dialect
    # ---------------------------------------------------------------
    results: dict[str, Any] = {}
    summary_rows = []

    for code in DIALECT_CODES:
        if code not in eig_words:
            continue

        dialect_data = eig_words[code]
        axes_info = []

        for axis in dialect_data[:10]:  # Top 10 eigenvectors
            rank = axis["rank"]
            eigenvalue_mag = axis["eigenvalue_magnitude"]
            top_words = axis["top_words"][:20]  # Top 20 words

            # Categorize each word
            categorized = []
            category_counts = {"morphosyntactic": 0, "phonological": 0, "lexical": 0, "pragmatic": 0, "other": 0}
            category_weight = {"morphosyntactic": 0.0, "phonological": 0.0, "lexical": 0.0, "pragmatic": 0.0, "other": 0.0}

            for w in top_words:
                cat = categorize_word(w["word"])
                categorized.append({
                    "word": w["word"],
                    "projection": w["projection"],
                    "category": cat,
                })
                category_counts[cat] += 1
                category_weight[cat] += w["projection"]

            # Dominant category
            total_weight = sum(category_weight.values())
            category_fractions = {
                k: v / total_weight if total_weight > 0 else 0
                for k, v in category_weight.items()
            }
            dominant = max(category_fractions, key=category_fractions.get)

            axis_info = {
                "rank": rank,
                "eigenvalue_magnitude": eigenvalue_mag,
                "dominant_category": dominant,
                "category_counts": category_counts,
                "category_weight_fractions": {k: round(v, 4) for k, v in category_fractions.items()},
                "top_words_categorized": categorized[:10],  # Save top 10 for brevity
            }
            axes_info.append(axis_info)

            # Summary row for CSV
            top3_words = ", ".join(w["word"] for w in top_words[:3])
            summary_rows.append({
                "dialect": code,
                "dialect_name": DIALECT_LABELS.get(code, code),
                "axis": rank,
                "|λ|": f"{eigenvalue_mag:.4f}",
                "dominant": dominant,
                "morph%": f"{category_fractions['morphosyntactic']:.0%}",
                "phon%": f"{category_fractions['phonological']:.0%}",
                "lex%": f"{category_fractions['lexical']:.0%}",
                "prag%": f"{category_fractions['pragmatic']:.0%}",
                "other%": f"{category_fractions['other']:.0%}",
                "top_3_words": top3_words,
            })

            if rank <= 5:
                logger.info(
                    "  %s axis %d (|λ|=%.4f): %s [%s] — %s",
                    code, rank, eigenvalue_mag, dominant,
                    ", ".join(f"{k}:{v}" for k, v in category_counts.items() if v > 0),
                    top3_words,
                )

        results[code] = axes_info

    # ---------------------------------------------------------------
    # Aggregate: what fraction of axes are morphosyntactic vs lexical?
    # ---------------------------------------------------------------
    aggregate = {}
    for code, axes in results.items():
        cats = [a["dominant_category"] for a in axes[:5]]
        from collections import Counter
        agg = Counter(cats)
        aggregate[code] = dict(agg)

    logger.info("\nAggregate dominant categories (top-5 axes):")
    for code, agg in sorted(aggregate.items()):
        logger.info("  %s: %s", code, agg)

    results["aggregate_top5"] = aggregate

    # ---------------------------------------------------------------
    # Cross-dialect patterns: are certain categories consistently dominant?
    # ---------------------------------------------------------------
    axis_categories_by_rank: dict[int, list[str]] = {}
    for code, axes in results.items():
        if code == "aggregate_top5":
            continue
        for axis in axes:
            rank = axis["rank"]
            axis_categories_by_rank.setdefault(rank, []).append(axis["dominant_category"])

    cross_dialect_patterns = {}
    for rank in sorted(axis_categories_by_rank.keys()):
        from collections import Counter
        counts = Counter(axis_categories_by_rank[rank])
        cross_dialect_patterns[rank] = dict(counts)

    results["cross_dialect_axis_patterns"] = cross_dialect_patterns

    logger.info("\nCross-dialect axis patterns:")
    for rank, counts in sorted(cross_dialect_patterns.items()):
        logger.info("  Axis %d: %s", rank, counts)

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------
    with open(ANALYSIS_DIR / "linguistic_axes.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved linguistic categorization to %s", ANALYSIS_DIR / "linguistic_axes.json")

    # ---------------------------------------------------------------
    # Save CSV for paper table
    # ---------------------------------------------------------------
    import csv
    csv_path = ANALYSIS_DIR / "linguistic_axes_table.csv"
    if summary_rows:
        fields = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("Saved paper table CSV to %s", csv_path)

    # ---------------------------------------------------------------
    # Plot: stacked bar chart of category fractions per dialect
    # ---------------------------------------------------------------
    fig, axes_plot = plt.subplots(1, 2, figsize=(16, 6))

    # Left: category distribution for top-5 axes per dialect
    ax = axes_plot[0]
    dialects_with_data = [c for c in DIALECT_CODES if c in results and c != "ES_PEN"]
    x = np.arange(len(dialects_with_data))

    bottom_morph = []
    bottom_phon = []
    bottom_lex = []
    bottom_prag = []
    bottom_other = []

    for code in dialects_with_data:
        axes_data = results[code][:5]
        fracs = {"morphosyntactic": 0, "phonological": 0, "lexical": 0, "pragmatic": 0, "other": 0}
        for ax_data in axes_data:
            for cat, frac in ax_data["category_weight_fractions"].items():
                fracs[cat] += frac
        total = sum(fracs.values())
        for cat in fracs:
            fracs[cat] /= total if total > 0 else 1
        bottom_morph.append(fracs["morphosyntactic"])
        bottom_phon.append(fracs["phonological"])
        bottom_lex.append(fracs["lexical"])
        bottom_prag.append(fracs["pragmatic"])
        bottom_other.append(fracs["other"])

    categories_data = [
        ("Morphosyntactic", bottom_morph, CATEGORY_COLORS["morphosyntactic"]),
        ("Phonological", bottom_phon, CATEGORY_COLORS["phonological"]),
        ("Lexical", bottom_lex, CATEGORY_COLORS["lexical"]),
        ("Pragmatic", bottom_prag, CATEGORY_COLORS["pragmatic"]),
        ("Other", bottom_other, CATEGORY_COLORS["other"]),
    ]

    cumulative = np.zeros(len(dialects_with_data))
    for name, vals, color in categories_data:
        ax.bar(x, vals, bottom=cumulative, label=name, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        cumulative += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([DIALECT_LABELS.get(c, c) for c in dialects_with_data], rotation=30, ha="right")
    ax.set_ylabel("Weighted Fraction")
    ax.set_title("Linguistic Category Distribution (Top-5 Axes)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)

    # Right: heatmap of dominant category per axis per dialect
    ax2 = axes_plot[1]
    cat_to_num = {"morphosyntactic": 0, "phonological": 1, "lexical": 2, "pragmatic": 3, "other": 4}
    n_axes = 10
    matrix = np.full((len(dialects_with_data), n_axes), 4)  # default "other"

    for i, code in enumerate(dialects_with_data):
        if code in results:
            for j, axis_data in enumerate(results[code][:n_axes]):
                matrix[i, j] = cat_to_num[axis_data["dominant_category"]]

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([
        CATEGORY_COLORS["morphosyntactic"],
        CATEGORY_COLORS["phonological"],
        CATEGORY_COLORS["lexical"],
        CATEGORY_COLORS["pragmatic"],
        CATEGORY_COLORS["other"],
    ])

    im = ax2.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=4)
    ax2.set_xticks(range(n_axes))
    ax2.set_xticklabels([f"Axis {i+1}" for i in range(n_axes)], rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(dialects_with_data)))
    ax2.set_yticklabels([DIALECT_LABELS.get(c, c) for c in dialects_with_data], fontsize=9)
    ax2.set_title("Dominant Category per Eigenvector Axis")

    # Legend
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=c.capitalize())
        for c in ["morphosyntactic", "phonological", "lexical", "pragmatic", "other"]
    ]
    ax2.legend(handles=legend_patches, loc="lower right", fontsize=7)

    fig.suptitle("Linguistic Interpretation of Eigenvector Axes", fontsize=14)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "linguistic_axes.png")
    plt.close(fig)
    logger.info("Saved linguistic axes visualization")


if __name__ == "__main__":
    main()
