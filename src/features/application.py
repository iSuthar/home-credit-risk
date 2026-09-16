"""Application-table features justified by the EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import TARGET_COL, safe_divide

EMPLOYMENT_SENTINEL = 365243
EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

APPLICATION_FEATURES = [
    "DAYS_EMPLOYED_ANOMALY",
    "DAYS_EMPLOYED_CLEAN",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "INCOME_CREDIT_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "CREDIT_GOODS_RATIO",
    "INCOME_PER_PERSON",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_COUNT",
    "EXT_SOURCE_MISSING_COUNT",
]

REQUIRED_COLUMNS = {
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_FAM_MEMBERS",
    *EXT_SOURCE_COLUMNS,
}


def engineer_application_features(application: pd.DataFrame) -> pd.DataFrame:
    """Preserve application columns and add deterministic row-wise features."""
    if TARGET_COL in application.columns:
        raise ValueError("TARGET must be removed before feature generation")
    missing = REQUIRED_COLUMNS.difference(application.columns)
    if missing:
        raise ValueError(f"Application table is missing columns: {sorted(missing)}")

    result = application.copy()
    anomaly = result["DAYS_EMPLOYED"].eq(EMPLOYMENT_SENTINEL)
    result["DAYS_EMPLOYED_ANOMALY"] = anomaly.astype("int8")
    result["DAYS_EMPLOYED_CLEAN"] = result["DAYS_EMPLOYED"].mask(anomaly, np.nan)
    result["AGE_YEARS"] = -result["DAYS_BIRTH"] / 365.25
    result["EMPLOYED_YEARS"] = -result["DAYS_EMPLOYED_CLEAN"] / 365.25
    result["INCOME_CREDIT_RATIO"] = safe_divide(
        result["AMT_INCOME_TOTAL"], result["AMT_CREDIT"]
    )
    result["ANNUITY_CREDIT_RATIO"] = safe_divide(
        result["AMT_ANNUITY"], result["AMT_CREDIT"]
    )
    result["CREDIT_GOODS_RATIO"] = safe_divide(
        result["AMT_CREDIT"], result["AMT_GOODS_PRICE"]
    )
    result["INCOME_PER_PERSON"] = safe_divide(
        result["AMT_INCOME_TOTAL"], result["CNT_FAM_MEMBERS"]
    )
    result["EXT_SOURCE_MEAN"] = result[EXT_SOURCE_COLUMNS].mean(axis=1)
    result["EXT_SOURCE_COUNT"] = (
        result[EXT_SOURCE_COLUMNS].notna().sum(axis=1).astype("int8")
    )
    result["EXT_SOURCE_MISSING_COUNT"] = (
        len(EXT_SOURCE_COLUMNS) - result["EXT_SOURCE_COUNT"]
    ).astype("int8")
    return result
