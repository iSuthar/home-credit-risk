"""Structural and leakage-oriented checks for final feature tables."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .application import EMPLOYMENT_SENTINEL
from .common import ID_COL, TARGET_COL

NO_HISTORY_RULES = {
    "INSTALLMENT_RECORD_COUNT": {
        "history": "HAS_INSTALLMENT_HISTORY",
        "zero": ["ANY_LATE_INSTALLMENT", "ANY_INSTALLMENT_SHORTFALL"],
        "nan": [
            "INSTALLMENT_MEAN_DELAY_DAYS",
            "INSTALLMENT_MAX_DELAY_DAYS",
            "INSTALLMENT_LATE_SHARE",
            "INSTALLMENT_MEAN_SHORTFALL_RATIO",
            "INSTALLMENT_UNDERPAID_SHARE",
        ],
    },
    "POS_RECORD_COUNT": {
        "history": "HAS_POS_HISTORY",
        "zero": [
            "POS_MONTHS",
            "POS_DPD_MONTHS",
            "POS_DPD_DEF_MONTHS",
            "ANY_POS_DPD",
            "ANY_POS_DPD_DEF",
        ],
        "nan": ["POS_DPD_MAX", "POS_DPD_DEF_MAX", "POS_DPD_SHARE", "POS_DPD_DEF_SHARE"],
    },
    "CARD_RECORD_COUNT": {
        "history": "HAS_CARD_HISTORY",
        "zero": [
            "CARD_MONTHS",
            "CARD_OVER_LIMIT_MONTHS",
            "CARD_DPD_MONTHS",
            "EVER_OVER_CARD_LIMIT",
            "ANY_CARD_DPD",
        ],
        "nan": [
            "CARD_UTIL_MEAN",
            "CARD_UTIL_MAX",
            "CARD_OVER_LIMIT_SHARE",
            "CARD_DPD_MAX",
            "CARD_DPD_DEF_MAX",
            "CARD_DPD_SHARE",
        ],
    },
}


def infinite_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Return numeric columns containing either positive or negative infinity."""
    failures = []
    for column in columns:
        if column in frame and pd.api.types.is_numeric_dtype(frame[column]):
            if np.isinf(frame[column].to_numpy(dtype="float64", na_value=np.nan)).any():
                failures.append(column)
    return failures


def _validate_one_table_no_history(frame: pd.DataFrame) -> None:
    history_pairs = [
        ("BUREAU_RECORD_COUNT", "HAS_BUREAU_HISTORY"),
        ("PREVIOUS_APPLICATION_RECORD_COUNT", "HAS_PREVIOUS_APPLICATION_HISTORY"),
        ("INSTALLMENT_RECORD_COUNT", "HAS_INSTALLMENT_HISTORY"),
        ("POS_RECORD_COUNT", "HAS_POS_HISTORY"),
        ("CARD_RECORD_COUNT", "HAS_CARD_HISTORY"),
    ]
    for count_column, flag_column in history_pairs:
        expected = frame[count_column].gt(0).astype("int8")
        if not expected.equals(frame[flag_column].astype("int8")):
            raise AssertionError(f"{flag_column} does not agree with {count_column}")

    for count_column, rules in NO_HISTORY_RULES.items():
        no_history = frame[count_column].eq(0)
        if not frame.loc[no_history, rules["history"]].eq(0).all():
            raise AssertionError(f"No-history rows have invalid {rules['history']}")
        for column in rules["zero"]:
            if not frame.loc[no_history, column].eq(0).all():
                raise AssertionError(f"No-history rows must have zero {column}")
        for column in rules["nan"]:
            if not frame.loc[no_history, column].isna().all():
                raise AssertionError(f"No-history rows must preserve missing {column}")


def validate_feature_tables(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    expected_train_rows: int,
    expected_test_rows: int,
    engineered_features: Sequence[str],
) -> dict[str, object]:
    """Run all required final-table validation checks."""
    if len(train) != expected_train_rows or len(test) != expected_test_rows:
        raise AssertionError("Application row counts changed")
    if not train[ID_COL].is_unique or not test[ID_COL].is_unique:
        raise AssertionError("SK_ID_CURR is not unique")
    if not set(train[ID_COL]).isdisjoint(set(test[ID_COL])):
        raise AssertionError("Train and test applicant IDs overlap")
    if TARGET_COL not in train or TARGET_COL in test:
        raise AssertionError("TARGET placement is invalid")
    train_schema = train.drop(columns=TARGET_COL).columns.tolist()
    if train_schema != test.columns.tolist():
        raise AssertionError("Train/test feature schemas are not aligned")

    expected_anomaly = train["DAYS_EMPLOYED"].eq(EMPLOYMENT_SENTINEL).astype("int8")
    if not expected_anomaly.equals(train["DAYS_EMPLOYED_ANOMALY"].astype("int8")):
        raise AssertionError("Training employment anomaly flag is incorrect")
    expected_anomaly = test["DAYS_EMPLOYED"].eq(EMPLOYMENT_SENTINEL).astype("int8")
    if not expected_anomaly.equals(test["DAYS_EMPLOYED_ANOMALY"].astype("int8")):
        raise AssertionError("Test employment anomaly flag is incorrect")

    inf_train = infinite_columns(train, engineered_features)
    inf_test = infinite_columns(test, engineered_features)
    if inf_train or inf_test:
        raise AssertionError(f"Engineered infinities found: train={inf_train}, test={inf_test}")

    _validate_one_table_no_history(train)
    _validate_one_table_no_history(test)

    return {
        "train_rows_match": True,
        "test_rows_match": True,
        "unique_applicant_keys": True,
        "disjoint_id_sets": True,
        "one_row_per_applicant_after_joins": True,
        "schemas_align_excluding_target": True,
        "target_only_in_train": True,
        "engineered_infinite_value_count": 0,
        "employment_anomaly_exact": True,
        "no_history_policy_valid": True,
        "auxiliary_features_at_applicant_grain": True,
    }


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Create a deterministic content/schema digest without retaining a second copy."""
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, frame.columns)).encode())
    digest.update("\x1f".join(map(str, frame.dtypes)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()
