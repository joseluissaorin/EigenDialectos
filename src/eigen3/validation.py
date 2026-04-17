"""Dialectometric validation against known linguistic relationships.

Compares computed spectral distances with expected dialectological
groupings from the literature (Moreno Fernández, Lipski, Penny).

Validates:
  1. Known close pairs rank correctly (CAN-CAR, AND-CAN, RIO-CHI)
  2. Expected dialect groupings have lower within-group distances
  3. Spectral distance correlates with geographic distance
  4. Overall distance ranking matches dialectological consensus
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from eigen3.constants import ALL_VARIETIES, REFERENCE_VARIETY, DIALECT_COORDINATES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected dialectological relationships (from literature)
# ---------------------------------------------------------------------------

# Similarity rankings from Moreno Fernández (2009), Lipski (1994), Penny (2000)
# Scale: 1.0 = essentially the same variety, 0.0 = maximally different
# These encode the consensus among Hispanic dialectologists.
EXPECTED_SIMILARITY: dict[tuple[str, str], float] = {
    # Atlantic cluster: CAN-CAR-AND share seseo, aspiration, ustedes
    ("ES_CAN", "ES_CAR"):    0.90,
    ("ES_AND", "ES_CAN"):    0.80,
    ("ES_AND", "ES_CAR"):    0.70,
    # Southern Cone: RIO-CHI share voseo, aspiration patterns
    ("ES_RIO", "ES_CHI"):    0.75,
    # Andean connection: AND_BO shares features with both Southern Cone and highlands
    ("ES_RIO", "ES_AND_BO"): 0.65,
    ("ES_CHI", "ES_AND_BO"): 0.60,
    # Iberian cluster: PEN and AND are geographically close but linguistically split
    ("ES_PEN", "ES_AND"):    0.55,
    # American general: all American varieties closer to each other than to PEN
    ("ES_MEX", "ES_CAR"):    0.55,
    ("ES_MEX", "ES_AND_BO"): 0.50,
    # PEN is most different from American varieties
    ("ES_PEN", "ES_MEX"):    0.35,
    ("ES_PEN", "ES_RIO"):    0.30,
    ("ES_PEN", "ES_CAR"):    0.30,
    ("ES_PEN", "ES_CHI"):    0.25,
    ("ES_PEN", "ES_AND_BO"): 0.25,
    ("ES_PEN", "ES_CAN"):    0.40,  # CAN closer to PEN than other American
}

# Expected groupings (dialect zones)
DIALECT_ZONES: dict[str, list[str]] = {
    "Atlantic":       ["ES_CAN", "ES_CAR", "ES_AND"],
    "Southern Cone":  ["ES_RIO", "ES_CHI"],
    "Andean":         ["ES_AND_BO"],
    "Mesoamerican":   ["ES_MEX"],
    "Peninsular":     ["ES_PEN"],
}

# Expected ranking constraints: (closer_pair, farther_pair)
# The first pair should have LOWER distance than the second pair
RANKING_CONSTRAINTS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    # CAN-CAR should be closer than CAN-MEX
    (("ES_CAN", "ES_CAR"), ("ES_CAN", "ES_MEX")),
    # RIO-CHI should be closer than RIO-MEX
    (("ES_RIO", "ES_CHI"), ("ES_RIO", "ES_MEX")),
    # AND-CAN should be closer than PEN-CAN
    (("ES_AND", "ES_CAN"), ("ES_PEN", "ES_CAN")),
    # CAN-CAR should be closer than PEN-CAR
    (("ES_CAN", "ES_CAR"), ("ES_PEN", "ES_CAR")),
    # RIO-AND_BO should be closer than PEN-AND_BO
    (("ES_RIO", "ES_AND_BO"), ("ES_PEN", "ES_AND_BO")),
    # CHI-RIO should be closer than CHI-PEN
    (("ES_CHI", "ES_RIO"), ("ES_CHI", "ES_PEN")),
    # AND-AND_BO should be closer than MEX-AND_BO (cultural/lexical affinity)
    (("ES_AND", "ES_AND_BO"), ("ES_MEX", "ES_AND_BO")),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Full dialectometric validation result."""
    # Distance matrix
    distance_matrix: np.ndarray
    varieties: list[str]

    # Spearman correlation with expected similarities
    similarity_correlation: float
    similarity_p_value: float

    # Geographic distance correlation
    geographic_correlation: float
    geographic_p_value: float

    # Ranking constraint satisfaction
    constraints_satisfied: int
    constraints_total: int
    constraint_details: list[tuple[str, bool]]

    # Within-group vs between-group distances
    zone_cohesion: dict[str, float]  # zone -> mean within-group distance
    zone_separation: float           # mean between-group distance
    cohesion_ratio: float            # separation / mean(cohesion) — higher is better

    # Closest and farthest pairs
    closest_pairs: list[tuple[str, str, float]]
    farthest_pairs: list[tuple[str, str, float]]


# ---------------------------------------------------------------------------
# Core validation functions
# ---------------------------------------------------------------------------

def validate_dialectometry(
    distance_matrix: np.ndarray,
    varieties: list[str],
) -> ValidationResult:
    """Run full dialectometric validation on a spectral distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        (n, n) symmetric distance matrix.
    varieties : list[str]
        Variety codes matching matrix rows/columns.

    Returns
    -------
    ValidationResult
    """
    n = len(varieties)
    var_to_idx = {v: i for i, v in enumerate(varieties)}

    def _get_dist(v1: str, v2: str) -> float:
        return float(distance_matrix[var_to_idx[v1], var_to_idx[v2]])

    # 1. Correlation with expected similarities
    expected_dists = []
    actual_dists = []
    for (v1, v2), sim in EXPECTED_SIMILARITY.items():
        if v1 in var_to_idx and v2 in var_to_idx:
            expected_dists.append(1.0 - sim)  # Convert similarity to distance
            actual_dists.append(_get_dist(v1, v2))

    if len(expected_dists) >= 3:
        sim_rho, sim_p = stats.spearmanr(expected_dists, actual_dists)
    else:
        sim_rho, sim_p = 0.0, 1.0

    # 2. Geographic distance correlation
    geo_dists = []
    spec_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = varieties[i], varieties[j]
            if v1 in DIALECT_COORDINATES and v2 in DIALECT_COORDINATES:
                lat1, lon1 = DIALECT_COORDINATES[v1]
                lat2, lon2 = DIALECT_COORDINATES[v2]
                geo_d = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
                geo_dists.append(geo_d)
                spec_dists.append(distance_matrix[i, j])

    if len(geo_dists) >= 3:
        geo_rho, geo_p = stats.spearmanr(geo_dists, spec_dists)
    else:
        geo_rho, geo_p = 0.0, 1.0

    # 3. Ranking constraints
    constraint_details = []
    satisfied = 0
    for (c1v1, c1v2), (c2v1, c2v2) in RANKING_CONSTRAINTS:
        if all(v in var_to_idx for v in [c1v1, c1v2, c2v1, c2v2]):
            d_close = _get_dist(c1v1, c1v2)
            d_far = _get_dist(c2v1, c2v2)
            ok = d_close < d_far
            label = f"{c1v1}-{c1v2} ({d_close:.2f}) < {c2v1}-{c2v2} ({d_far:.2f})"
            constraint_details.append((label, ok))
            if ok:
                satisfied += 1

    # 4. Within-group vs between-group distances
    zone_cohesion: dict[str, float] = {}
    within_dists = []
    between_dists = []

    # Map variety -> zone
    var_zone = {}
    for zone, members in DIALECT_ZONES.items():
        for v in members:
            var_zone[v] = zone

    for zone, members in DIALECT_ZONES.items():
        zone_members = [v for v in members if v in var_to_idx]
        if len(zone_members) >= 2:
            dists = []
            for i, v1 in enumerate(zone_members):
                for v2 in zone_members[i + 1:]:
                    d = _get_dist(v1, v2)
                    dists.append(d)
                    within_dists.append(d)
            zone_cohesion[zone] = float(np.mean(dists))
        elif len(zone_members) == 1:
            zone_cohesion[zone] = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = varieties[i], varieties[j]
            z1, z2 = var_zone.get(v1), var_zone.get(v2)
            if z1 and z2 and z1 != z2:
                between_dists.append(float(distance_matrix[i, j]))

    mean_within = float(np.mean(within_dists)) if within_dists else 0.0
    mean_between = float(np.mean(between_dists)) if between_dists else 0.0
    cohesion_ratio = mean_between / mean_within if mean_within > 0 else 0.0

    # 5. Closest and farthest pairs
    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((varieties[i], varieties[j], float(distance_matrix[i, j])))
    pairs.sort(key=lambda x: x[2])

    return ValidationResult(
        distance_matrix=distance_matrix,
        varieties=varieties,
        similarity_correlation=float(sim_rho),
        similarity_p_value=float(sim_p),
        geographic_correlation=float(geo_rho),
        geographic_p_value=float(geo_p),
        constraints_satisfied=satisfied,
        constraints_total=len(constraint_details),
        constraint_details=constraint_details,
        zone_cohesion=zone_cohesion,
        zone_separation=mean_between,
        cohesion_ratio=cohesion_ratio,
        closest_pairs=pairs[:5],
        farthest_pairs=pairs[-5:],
    )


def format_validation_report(result: ValidationResult) -> str:
    """Format validation results as a human-readable report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("DIALECTOMETRIC VALIDATION")
    lines.append("=" * 70)

    # Correlation with literature
    lines.append("\n--- Correlation with Known Dialectological Relationships ---\n")
    sim_grade = "EXCELLENT" if result.similarity_correlation > 0.7 else \
                "GOOD" if result.similarity_correlation > 0.5 else \
                "MODERATE" if result.similarity_correlation > 0.3 else "WEAK"
    lines.append(f"  Spearman rho with expected similarities: {result.similarity_correlation:.3f} "
                 f"(p={result.similarity_p_value:.4f}) [{sim_grade}]")

    geo_grade = "EXCELLENT" if result.geographic_correlation > 0.7 else \
                "GOOD" if result.geographic_correlation > 0.5 else \
                "MODERATE" if result.geographic_correlation > 0.3 else "WEAK"
    lines.append(f"  Spearman rho with geographic distance:   {result.geographic_correlation:.3f} "
                 f"(p={result.geographic_p_value:.4f}) [{geo_grade}]")

    # Ranking constraints
    lines.append(f"\n--- Ranking Constraints: {result.constraints_satisfied}/"
                 f"{result.constraints_total} satisfied ---\n")
    for label, ok in result.constraint_details:
        status = "PASS" if ok else "FAIL"
        lines.append(f"  [{status}] {label}")

    # Zone cohesion
    lines.append("\n--- Dialect Zone Cohesion ---\n")
    for zone, cohesion in sorted(result.zone_cohesion.items()):
        lines.append(f"  {zone:20s}: within-group distance = {cohesion:.4f}")
    lines.append(f"  {'Between-group':20s}: mean distance = {result.zone_separation:.4f}")
    lines.append(f"  Cohesion ratio (between/within): {result.cohesion_ratio:.2f} "
                 f"(higher is better, >1.5 expected)")

    # Closest / farthest pairs
    lines.append("\n--- Closest Dialect Pairs ---\n")
    for v1, v2, d in result.closest_pairs:
        lines.append(f"  {v1:10s} - {v2:10s}: {d:.4f}")

    lines.append("\n--- Most Distant Dialect Pairs ---\n")
    for v1, v2, d in result.farthest_pairs:
        lines.append(f"  {v1:10s} - {v2:10s}: {d:.4f}")

    # Distance matrix
    lines.append("\n--- Full Distance Matrix ---\n")
    header = "            " + "  ".join(f"{v:>10s}" for v in result.varieties)
    lines.append(header)
    for i, v in enumerate(result.varieties):
        row = f"  {v:10s}" + "  ".join(
            f"{result.distance_matrix[i, j]:10.4f}" for j in range(len(result.varieties))
        )
        lines.append(row)

    return "\n".join(lines)
