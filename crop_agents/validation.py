"""Input validation (safe response for invalid inputs, step 7)."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple

from .config import FEATURES, PLAUSIBLE_RANGES


def validate_measurements(raw: Any, stats=None) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Return (clean_values, errors, warnings)."""
    errors: List[str] = []
    warnings: List[str] = []
    clean: Dict[str, float] = {}

    if not isinstance(raw, dict) or not raw:
        return {}, ["No measurements were provided."], []

    for f in FEATURES:
        val = raw.get(f)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"Missing value for '{f}'.")
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            errors.append(f"'{f}' must be a number (got {val!r}).")
            continue
        if not math.isfinite(v):
            errors.append(f"'{f}' must be a finite number.")
            continue
        lo, hi = PLAUSIBLE_RANGES[f]
        if not lo <= v <= hi:
            errors.append(f"'{f}'={v:g} is outside the physically plausible range {lo}-{hi}.")
            continue
        clean[f] = v

    if stats is not None and not errors:
        for f, v in clean.items():
            lo, hi = stats.overall_range(f)
            if v < lo or v > hi:
                warnings.append(
                    f"'{f}'={v:g} is outside the range seen in the training data ({lo:g}-{hi:g}); "
                    "the model is extrapolating."
                )
    return clean, errors, warnings
