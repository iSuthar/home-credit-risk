"""Build applicant-level Home Credit feature tables without learned preprocessing."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .application import APPLICATION_FEATURES, engineer_application_features
from .common import ID_COL, TARGET_COL, assert_applicant_grain
from .credit_card import CARD_FEATURES, aggregate_credit_cards
from .history import (
    BUREAU_FEATURES,
    PREVIOUS_APPLICATION_FEATURES,
    aggregate_bureau,
    aggregate_previous_applications,
)
from .installments import INSTALLMENT_FEATURES, aggregate_installments
from .pos import POS_FEATURES, aggregate_pos
from .validation import dataframe_fingerprint, validate_feature_tables

ENGINEERED_FEATURE_GROUPS = {
    "application": APPLICATION_FEATURES,
    "bureau": BUREAU_FEATURES,
    "previous_application": PREVIOUS_APPLICATION_FEATURES,
    "installments_payments": INSTALLMENT_FEATURES,
    "POS_CASH_balance": POS_FEATURES,
    "credit_card_balance": CARD_FEATURES,
}
ENGINEERED_FEATURES = [
    feature for features in ENGINEERED_FEATURE_GROUPS.values() for feature in features
]

COUNT_COLUMNS = [
    "BUREAU_RECORD_COUNT",
    "PREVIOUS_APPLICATION_RECORD_COUNT",
    "INSTALLMENT_RECORD_COUNT",
    "POS_RECORD_COUNT",
    "POS_MONTHS",
    "POS_DPD_MONTHS",
    "POS_DPD_DEF_MONTHS",
    "CARD_RECORD_COUNT",
    "CARD_MONTHS",
    "CARD_OVER_LIMIT_MONTHS",
    "CARD_DPD_MONTHS",
]

BINARY_COLUMNS = [
    "DAYS_EMPLOYED_ANOMALY",
    "HAS_BUREAU_HISTORY",
    "HAS_MICROLOAN",
    "HAS_PREVIOUS_APPLICATION_HISTORY",
    "HAS_REFUSED_PRIOR",
    "HAS_INSTALLMENT_HISTORY",
    "ANY_LATE_INSTALLMENT",
    "ANY_INSTALLMENT_SHORTFALL",
    "HAS_POS_HISTORY",
    "ANY_POS_DPD",
    "ANY_POS_DPD_DEF",
    "HAS_CARD_HISTORY",
    "EVER_OVER_CARD_LIMIT",
    "ANY_CARD_DPD",
]

REQUIRED_INPUT_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
]


def _load_application_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_raw = pd.read_csv(data_dir / "application_train.csv", low_memory=False)
    test_raw = pd.read_csv(data_dir / "application_test.csv", low_memory=False)
    if TARGET_COL not in train_raw or TARGET_COL in test_raw:
        raise ValueError("Expected TARGET only in application_train.csv")
    target = train_raw.pop(TARGET_COL)
    if train_raw.columns.tolist() != test_raw.columns.tolist():
        raise ValueError("Raw application train/test schemas do not align")
    return train_raw, target, test_raw


def _aggregate_auxiliary_tables(data_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    aggregates = [
        ("bureau", aggregate_bureau(data_dir / "bureau.csv")),
        (
            "previous_application",
            aggregate_previous_applications(data_dir / "previous_application.csv"),
        ),
        ("installments_payments", aggregate_installments(data_dir / "installments_payments.csv")),
        ("POS_CASH_balance", aggregate_pos(data_dir / "POS_CASH_balance.csv")),
        ("credit_card_balance", aggregate_credit_cards(data_dir / "credit_card_balance.csv")),
    ]
    for name, aggregate in aggregates:
        assert_applicant_grain(aggregate, name)
    return aggregates


def _join_aggregates(
    application: pd.DataFrame,
    aggregates: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    result = application
    original_rows = len(result)
    for name, aggregate in aggregates:
        result = result.merge(
            aggregate,
            how="left",
            left_on=ID_COL,
            right_index=True,
            sort=False,
            validate="one_to_one",
        )
        if len(result) != original_rows:
            raise AssertionError(f"{name} join changed the applicant row count")

    result[COUNT_COLUMNS] = result[COUNT_COLUMNS].fillna(0).astype("int32")
    result[BINARY_COLUMNS] = result[BINARY_COLUMNS].fillna(0).astype("int8")
    return result


def build_feature_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build target-independent train/test features entirely at applicant grain."""
    data_dir = Path(data_dir)
    missing = [name for name in REQUIRED_INPUT_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {', '.join(missing)}")

    train_raw, target, test_raw = _load_application_tables(data_dir)
    original_feature_count = len(test_raw.columns)
    expected_train_rows = len(train_raw)
    expected_test_rows = len(test_raw)

    train = engineer_application_features(train_raw)
    test = engineer_application_features(test_raw)
    aggregates = _aggregate_auxiliary_tables(data_dir)
    train = _join_aggregates(train, aggregates)
    test = _join_aggregates(test, aggregates)
    train.insert(1, TARGET_COL, target.to_numpy())

    checks = validate_feature_tables(
        train,
        test,
        expected_train_rows=expected_train_rows,
        expected_test_rows=expected_test_rows,
        engineered_features=ENGINEERED_FEATURES,
    )
    metadata: dict[str, Any] = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "original_feature_count_excluding_target": original_feature_count,
        "engineered_feature_count": len(ENGINEERED_FEATURES),
        "final_feature_count_excluding_target": test.shape[1],
        "engineered_feature_groups": ENGINEERED_FEATURE_GROUPS,
        "train_memory_mb": train.memory_usage(deep=True).sum() / 1024**2,
        "test_memory_mb": test.memory_usage(deep=True).sum() / 1024**2,
        "engineered_null_counts_train": {
            name: int(train[name].isna().sum()) for name in ENGINEERED_FEATURES
        },
        "engineered_null_counts_test": {
            name: int(test[name].isna().sum()) for name in ENGINEERED_FEATURES
        },
        "validation": checks,
    }
    return train, test, metadata


def write_feature_tables(
    train: pd.DataFrame,
    test: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "features_train.parquet", index=False)
    test.to_parquet(output_dir / "features_test.parquet", index=False)
    (output_dir / "feature_build_report.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    *,
    verify_determinism: bool = False,
) -> dict[str, Any]:
    train, test, metadata = build_feature_tables(data_dir)
    train_fingerprint = dataframe_fingerprint(train)
    test_fingerprint = dataframe_fingerprint(test)
    write_feature_tables(train, test, metadata, output_dir)

    if verify_determinism:
        del train, test
        gc.collect()
        repeat_train, repeat_test, _ = build_feature_tables(data_dir)
        deterministic = (
            dataframe_fingerprint(repeat_train) == train_fingerprint
            and dataframe_fingerprint(repeat_test) == test_fingerprint
        )
        if not deterministic:
            raise AssertionError("Two pipeline runs produced different feature tables")
        metadata["validation"]["two_runs_are_deterministic"] = True
        write_feature_tables(repeat_train, repeat_test, metadata, output_dir)
    else:
        metadata["validation"]["two_runs_are_deterministic"] = "not_requested"
        (output_dir / "feature_build_report.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Build twice and compare content/schema fingerprints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_pipeline(
        args.data_dir,
        args.output_dir,
        verify_determinism=args.verify_determinism,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
