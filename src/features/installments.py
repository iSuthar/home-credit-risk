"""Chunked installment-payment behavior aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import ID_COL, assert_applicant_grain, combine_metric, safe_divide

INSTALLMENT_FEATURES = [
    "INSTALLMENT_RECORD_COUNT",
    "HAS_INSTALLMENT_HISTORY",
    "INSTALLMENT_MEAN_DELAY_DAYS",
    "INSTALLMENT_MAX_DELAY_DAYS",
    "INSTALLMENT_LATE_SHARE",
    "INSTALLMENT_MEAN_SHORTFALL_RATIO",
    "INSTALLMENT_UNDERPAID_SHARE",
    "ANY_LATE_INSTALLMENT",
    "ANY_INSTALLMENT_SHORTFALL",
]


def aggregate_installments(path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Aggregate signed delays and clipped payment shortfalls per applicant."""
    state: dict[str, pd.Series] = {}
    columns = [
        ID_COL,
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        delay = chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]
        payment_ratio = safe_divide(
            chunk["AMT_PAYMENT"],
            chunk["AMT_INSTALMENT"],
            positive_denominator=True,
        )
        shortfall = (1 - payment_ratio).clip(lower=0, upper=1)
        behavior = pd.DataFrame(
            {
                ID_COL: chunk[ID_COL],
                "INSTALLMENT_RECORD_COUNT": 1,
                "DELAY_VALID": delay.notna().astype("int8"),
                "DELAY_SUM": delay.fillna(0),
                "LATE_COUNT": delay.gt(0).astype("int8"),
                "MAX_DELAY": delay,
                "SHORTFALL_VALID": shortfall.notna().astype("int8"),
                "SHORTFALL_SUM": shortfall.fillna(0),
                "UNDERPAID_COUNT": shortfall.gt(0).astype("int8"),
            }
        )
        grouped = behavior.groupby(ID_COL, sort=True)
        for metric in [
            "INSTALLMENT_RECORD_COUNT",
            "DELAY_VALID",
            "DELAY_SUM",
            "LATE_COUNT",
            "SHORTFALL_VALID",
            "SHORTFALL_SUM",
            "UNDERPAID_COUNT",
        ]:
            combine_metric(state, metric, grouped[metric].sum())
        combine_metric(state, "MAX_DELAY", grouped["MAX_DELAY"].max(), "max")

    raw = pd.DataFrame(state).sort_index()
    raw.index.name = ID_COL
    if raw.empty:
        result = pd.DataFrame(columns=INSTALLMENT_FEATURES)
        result.index.name = ID_COL
        return result

    result = pd.DataFrame(index=raw.index)
    result["INSTALLMENT_RECORD_COUNT"] = raw["INSTALLMENT_RECORD_COUNT"].astype("int32")
    result["HAS_INSTALLMENT_HISTORY"] = result["INSTALLMENT_RECORD_COUNT"].gt(0).astype("int8")
    result["INSTALLMENT_MEAN_DELAY_DAYS"] = safe_divide(raw["DELAY_SUM"], raw["DELAY_VALID"])
    result["INSTALLMENT_MAX_DELAY_DAYS"] = raw["MAX_DELAY"]
    result["INSTALLMENT_LATE_SHARE"] = safe_divide(raw["LATE_COUNT"], raw["DELAY_VALID"])
    result["INSTALLMENT_MEAN_SHORTFALL_RATIO"] = safe_divide(
        raw["SHORTFALL_SUM"], raw["SHORTFALL_VALID"]
    )
    result["INSTALLMENT_UNDERPAID_SHARE"] = safe_divide(
        raw["UNDERPAID_COUNT"], raw["SHORTFALL_VALID"]
    )
    result["ANY_LATE_INSTALLMENT"] = raw["LATE_COUNT"].gt(0).astype("int8")
    result["ANY_INSTALLMENT_SHORTFALL"] = raw["UNDERPAID_COUNT"].gt(0).astype("int8")
    result = result.replace([np.inf, -np.inf], np.nan)
    assert_applicant_grain(result, path.name)
    return result[INSTALLMENT_FEATURES]
