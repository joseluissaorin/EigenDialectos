"""Shared color palette for dialect visualizations."""

from __future__ import annotations

from eigendialectos.constants import DialectCode

# Consistent color mapping for all plots.
# Chosen for colour-blind friendliness (roughly based on the Okabe-Ito palette).
DIALECT_COLORS: dict[DialectCode, str] = {
    DialectCode.ES_PEN: "#E69F00",  # orange
    DialectCode.ES_AND: "#56B4E9",  # sky blue
    DialectCode.ES_CAN: "#009E73",  # bluish green
    DialectCode.ES_RIO: "#F0E442",  # yellow
    DialectCode.ES_MEX: "#0072B2",  # blue
    DialectCode.ES_CAR: "#D55E00",  # vermilion
    DialectCode.ES_CHI: "#CC79A7",  # reddish purple
    DialectCode.ES_AND_BO: "#999999",  # grey
}

DIALECT_MARKERS: dict[DialectCode, str] = {
    DialectCode.ES_PEN: "o",
    DialectCode.ES_AND: "s",
    DialectCode.ES_CAN: "^",
    DialectCode.ES_RIO: "D",
    DialectCode.ES_MEX: "v",
    DialectCode.ES_CAR: "P",
    DialectCode.ES_CHI: "X",
    DialectCode.ES_AND_BO: "*",
}


def dialect_label(code: DialectCode) -> str:
    """Short human-readable label for a dialect code."""
    from eigendialectos.constants import DIALECT_NAMES

    return DIALECT_NAMES.get(code, code.value)
