"""Chunked POS/CASH delinquency aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import ID_COL, assert_applicant_grain, combine_metric, safe_divide

POS_FEATURES = [
    "POS_RECORD_COUNT",
    "HAS_POS_HISTORY",
    "POS_MONTHS",
    "POS_DPD_MONTHS",
    "POS_DPD_MAX",
    "POS_DPD_DEF_MONTHS",
    "POS_DPD_DEF_MAX",
    "POS_DPD_SHARE",
    "POS_DPD_DEF_SHARE",
    "ANY_POS_DPD",
    "ANY_POS_DPD_DEF",
]


def aggregate_pos(path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Aggregate POS delinquency counts, maxima, shares, and flags."""
    state: dict[str, pd.Series] = {}
    for chunk in pd.read_csv(
        path,
        usecols=[ID_COL, "SK_DPD", "SK_DPD_DEF"],
        chunksize=chunksize,
    ):
        behavior = pd.DataFrame(
            {
                ID_COL: chunk[ID_COL],
                "POS_MONTHS": 1,
                "POS_DPD_MONTHS": chunk["SK_DPD"].gt(0).astype("int8"),
                "POS_DPD_MAX": chunk["SK_DPD"],
                "POS_DPD_DEF_MONTHS": chunk["SK_DPD_DEF"].gt(0).astype("int8"),
                "POS_DPD_DEF_MAX": chunk["SK_DPD_DEF"],
            }
        )
        grouped = behavior.groupby(ID_COL, sort=True)
        for metric in ["POS_MONTHS", "POS_DPD_MONTHS", "POS_DPD_DEF_MONTHS"]:
            combine_metric(state, metric, grouped[metric].sum())
        for metric in ["POS_DPD_MAX", "POS_DPD_DEF_MAX"]:
            combine_metric(state, metric, grouped[metric].max(), "max")

    raw = pd.DataFrame(state).sort_index()
    raw.index.name = ID_COL
    if raw.empty:
        result = pd.DataFrame(columns=POS_FEATURES)
        result.index.name = ID_COL
        return result

    result = pd.DataFrame(index=raw.index)
    result["POS_MONTHS"] = raw["POS_MONTHS"].astype("int32")
    result["POS_RECORD_COUNT"] = result["POS_MONTHS"]
    result["HAS_POS_HISTORY"] = result["POS_MONTHS"].gt(0).astype("int8")
    result["POS_DPD_MONTHS"] = raw["POS_DPD_MONTHS"].astype("int32")
    result["POS_DPD_MAX"] = raw["POS_DPD_MAX"]
    result["POS_DPD_DEF_MONTHS"] = raw["POS_DPD_DEF_MONTHS"].astype("int32")
    result["POS_DPD_DEF_MAX"] = raw["POS_DPD_DEF_MAX"]
    result["POS_DPD_SHARE"] = safe_divide(result["POS_DPD_MONTHS"], result["POS_MONTHS"])
    result["POS_DPD_DEF_SHARE"] = safe_divide(
        result["POS_DPD_DEF_MONTHS"], result["POS_MONTHS"]
    )
    result["ANY_POS_DPD"] = result["POS_DPD_MONTHS"].gt(0).astype("int8")
    result["ANY_POS_DPD_DEF"] = result["POS_DPD_DEF_MONTHS"].gt(0).astype("int8")
    result = result.replace([np.inf, -np.inf], np.nan)
    assert_applicant_grain(result, path.name)
    return result[POS_FEATURES]
