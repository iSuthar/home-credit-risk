"""Shared constants and chunk-aggregation helpers."""

from __future__ import annotations

from collections.abc import MutableMapping

import numpy as np
import pandas as pd

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    positive_denominator: bool = False,
) -> pd.Series:
    """Divide while treating invalid denominators and infinities as missing."""
    valid_denominator = denominator.gt(0) if positive_denominator else denominator.ne(0)
    result = numerator / denominator.where(valid_denominator)
    return result.replace([np.inf, -np.inf], np.nan)


def combine_metric(
    state: MutableMapping[str, pd.Series],
    name: str,
    values: pd.Series,
    operation: str = "sum",
) -> None:
    """Merge one chunk's applicant-level metric into an accumulated state."""
    if name not in state:
        state[name] = values
    elif operation == "sum":
        state[name] = state[name].add(values, fill_value=0)
    elif operation == "max":
        state[name] = pd.concat([state[name], values], axis=1).max(axis=1)
    else:
        raise ValueError(f"Unsupported combine operation: {operation}")


def assert_applicant_grain(frame: pd.DataFrame, name: str) -> None:
    """Fail early if an aggregate is not one row per applicant."""
    if frame.index.name != ID_COL:
        raise ValueError(f"{name} aggregate index must be named {ID_COL}")
    if not frame.index.is_unique:
        raise ValueError(f"{name} aggregate contains duplicate applicants")
