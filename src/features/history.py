"""Applicant-grain bureau and previous-application features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import ID_COL, assert_applicant_grain, combine_metric

BUREAU_FEATURES = [
    "BUREAU_RECORD_COUNT",
    "HAS_BUREAU_HISTORY",
    "HAS_MICROLOAN",
]

PREVIOUS_APPLICATION_FEATURES = [
    "PREVIOUS_APPLICATION_RECORD_COUNT",
    "HAS_PREVIOUS_APPLICATION_HISTORY",
    "HAS_REFUSED_PRIOR",
]


def aggregate_category_presence(
    path: Path,
    *,
    category_column: str,
    category_value: str,
    count_name: str,
    history_name: str,
    flag_name: str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Count records and detect whether each applicant ever had a category."""
    state: dict[str, pd.Series] = {}
    for chunk in pd.read_csv(
        path,
        usecols=[ID_COL, category_column],
        chunksize=chunksize,
    ):
        behavior = pd.DataFrame(
            {
                ID_COL: chunk[ID_COL],
                count_name: 1,
                flag_name: chunk[category_column].eq(category_value).astype("int8"),
            }
        )
        grouped = behavior.groupby(ID_COL, sort=True)
        combine_metric(state, count_name, grouped[count_name].sum())
        combine_metric(state, flag_name, grouped[flag_name].max(), "max")

    result = pd.DataFrame(state).sort_index()
    result.index.name = ID_COL
    if not result.empty:
        result[count_name] = result[count_name].astype("int32")
        result[history_name] = result[count_name].gt(0).astype("int8")
        result[flag_name] = result[flag_name].astype("int8")
        result = result[[count_name, history_name, flag_name]]
    else:
        result = pd.DataFrame(columns=[count_name, history_name, flag_name])
        result.index.name = ID_COL
    assert_applicant_grain(result, path.name)
    return result


def aggregate_bureau(path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    return aggregate_category_presence(
        path,
        category_column="CREDIT_TYPE",
        category_value="Microloan",
        count_name="BUREAU_RECORD_COUNT",
        history_name="HAS_BUREAU_HISTORY",
        flag_name="HAS_MICROLOAN",
        chunksize=chunksize,
    )


def aggregate_previous_applications(
    path: Path, chunksize: int = 500_000
) -> pd.DataFrame:
    return aggregate_category_presence(
        path,
        category_column="NAME_CONTRACT_STATUS",
        category_value="Refused",
        count_name="PREVIOUS_APPLICATION_RECORD_COUNT",
        history_name="HAS_PREVIOUS_APPLICATION_HISTORY",
        flag_name="HAS_REFUSED_PRIOR",
        chunksize=chunksize,
    )
