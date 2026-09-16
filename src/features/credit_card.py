"""Chunked credit-card utilization and delinquency aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import ID_COL, assert_applicant_grain, combine_metric, safe_divide

CARD_FEATURES = [
    "CARD_RECORD_COUNT",
    "CARD_MONTHS",
    "CARD_UTIL_MEAN",
    "CARD_UTIL_MAX",
    "CARD_OVER_LIMIT_MONTHS",
    "CARD_OVER_LIMIT_SHARE",
    "CARD_DPD_MONTHS",
    "CARD_DPD_MAX",
    "CARD_DPD_DEF_MAX",
    "CARD_DPD_SHARE",
    "HAS_CARD_HISTORY",
    "EVER_OVER_CARD_LIMIT",
    "ANY_CARD_DPD",
]


def aggregate_credit_cards(path: Path, chunksize: int = 750_000) -> pd.DataFrame:
    """Aggregate positive-limit utilization and delinquency per applicant."""
    state: dict[str, pd.Series] = {}
    columns = [
        ID_COL,
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        utilization = safe_divide(
            chunk["AMT_BALANCE"].clip(lower=0),
            chunk["AMT_CREDIT_LIMIT_ACTUAL"],
            positive_denominator=True,
        )
        behavior = pd.DataFrame(
            {
                ID_COL: chunk[ID_COL],
                "CARD_MONTHS": 1,
                "CARD_UTIL_VALID": utilization.notna().astype("int8"),
                "CARD_UTIL_SUM": utilization.fillna(0),
                "CARD_UTIL_MAX": utilization,
                "CARD_OVER_LIMIT_MONTHS": utilization.gt(1).astype("int8"),
                "CARD_DPD_MONTHS": chunk["SK_DPD"].gt(0).astype("int8"),
                "CARD_DPD_MAX": chunk["SK_DPD"],
                "CARD_DPD_DEF_MAX": chunk["SK_DPD_DEF"],
            }
        )
        grouped = behavior.groupby(ID_COL, sort=True)
        for metric in [
            "CARD_MONTHS",
            "CARD_UTIL_VALID",
            "CARD_UTIL_SUM",
            "CARD_OVER_LIMIT_MONTHS",
            "CARD_DPD_MONTHS",
        ]:
            combine_metric(state, metric, grouped[metric].sum())
        for metric in ["CARD_UTIL_MAX", "CARD_DPD_MAX", "CARD_DPD_DEF_MAX"]:
            combine_metric(state, metric, grouped[metric].max(), "max")

    raw = pd.DataFrame(state).sort_index()
    raw.index.name = ID_COL
    if raw.empty:
        result = pd.DataFrame(columns=CARD_FEATURES)
        result.index.name = ID_COL
        return result

    result = pd.DataFrame(index=raw.index)
    result["CARD_MONTHS"] = raw["CARD_MONTHS"].astype("int32")
    result["CARD_RECORD_COUNT"] = result["CARD_MONTHS"]
    result["CARD_UTIL_MEAN"] = safe_divide(raw["CARD_UTIL_SUM"], raw["CARD_UTIL_VALID"])
    result["CARD_UTIL_MAX"] = raw["CARD_UTIL_MAX"]
    result["CARD_OVER_LIMIT_MONTHS"] = raw["CARD_OVER_LIMIT_MONTHS"].astype("int32")
    result["CARD_OVER_LIMIT_SHARE"] = safe_divide(
        result["CARD_OVER_LIMIT_MONTHS"], raw["CARD_UTIL_VALID"]
    )
    result["CARD_DPD_MONTHS"] = raw["CARD_DPD_MONTHS"].astype("int32")
    result["CARD_DPD_MAX"] = raw["CARD_DPD_MAX"]
    result["CARD_DPD_DEF_MAX"] = raw["CARD_DPD_DEF_MAX"]
    result["CARD_DPD_SHARE"] = safe_divide(result["CARD_DPD_MONTHS"], result["CARD_MONTHS"])
    result["HAS_CARD_HISTORY"] = result["CARD_MONTHS"].gt(0).astype("int8")
    result["EVER_OVER_CARD_LIMIT"] = result["CARD_OVER_LIMIT_MONTHS"].gt(0).astype("int8")
    result["ANY_CARD_DPD"] = result["CARD_DPD_MONTHS"].gt(0).astype("int8")
    result = result.replace([np.inf, -np.inf], np.nan)
    assert_applicant_grain(result, path.name)
    return result[CARD_FEATURES]
