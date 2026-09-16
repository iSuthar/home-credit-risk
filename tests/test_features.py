from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.application import engineer_application_features
from src.features.build_features import ENGINEERED_FEATURES, build_feature_tables
from src.features.credit_card import aggregate_credit_cards
from src.features.history import aggregate_bureau, aggregate_previous_applications
from src.features.installments import aggregate_installments
from src.features.pos import aggregate_pos


class ApplicationFeatureTests(unittest.TestCase):
    def test_sentinel_missingness_counts_and_zero_denominators(self) -> None:
        application = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "DAYS_EMPLOYED": [365243, -730.5],
                "DAYS_BIRTH": [-3652.5, -7305.0],
                "AMT_INCOME_TOTAL": [100.0, 200.0],
                "AMT_CREDIT": [0.0, 400.0],
                "AMT_ANNUITY": [10.0, 40.0],
                "AMT_GOODS_PRICE": [0.0, 320.0],
                "CNT_FAM_MEMBERS": [0.0, 2.0],
                "EXT_SOURCE_1": [0.1, np.nan],
                "EXT_SOURCE_2": [np.nan, np.nan],
                "EXT_SOURCE_3": [0.3, np.nan],
                "CATEGORY": ["A", "B"],
            }
        )
        result = engineer_application_features(application)

        self.assertEqual(result.loc[0, "DAYS_EMPLOYED_ANOMALY"], 1)
        self.assertTrue(pd.isna(result.loc[0, "DAYS_EMPLOYED_CLEAN"]))
        self.assertAlmostEqual(result.loc[1, "EMPLOYED_YEARS"], 2.0)
        self.assertAlmostEqual(result.loc[0, "AGE_YEARS"], 10.0)
        self.assertTrue(pd.isna(result.loc[0, "INCOME_CREDIT_RATIO"]))
        self.assertTrue(pd.isna(result.loc[0, "ANNUITY_CREDIT_RATIO"]))
        self.assertTrue(pd.isna(result.loc[0, "CREDIT_GOODS_RATIO"]))
        self.assertTrue(pd.isna(result.loc[0, "INCOME_PER_PERSON"]))
        self.assertAlmostEqual(result.loc[0, "EXT_SOURCE_MEAN"], 0.2)
        self.assertEqual(result.loc[0, "EXT_SOURCE_COUNT"], 2)
        self.assertEqual(result.loc[0, "EXT_SOURCE_MISSING_COUNT"], 1)
        self.assertTrue(pd.isna(result.loc[1, "EXT_SOURCE_MEAN"]))
        self.assertEqual(result.loc[1, "EXT_SOURCE_COUNT"], 0)
        self.assertIn("CATEGORY", result)
        numeric = result.select_dtypes(include="number").to_numpy()
        self.assertFalse(np.isinf(numeric).any())

    def test_target_is_rejected_by_feature_function(self) -> None:
        with self.assertRaisesRegex(ValueError, "TARGET"):
            engineer_application_features(pd.DataFrame({"TARGET": [0]}))


class HistoricalAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.root / name
        frame.to_csv(path, index=False)
        return path

    def test_installment_delay_shortfall_and_invalid_amount(self) -> None:
        path = self.write(
            "installments_payments.csv",
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1, 1, 1, 2],
                    "DAYS_INSTALMENT": [-10.0, -20.0, -30.0, -10.0],
                    "DAYS_ENTRY_PAYMENT": [-5.0, -22.0, np.nan, -10.0],
                    "AMT_INSTALMENT": [100.0, 100.0, 0.0, 0.0],
                    "AMT_PAYMENT": [50.0, 150.0, 100.0, 10.0],
                }
            ),
        )
        result = aggregate_installments(path, chunksize=2)

        self.assertEqual(result.loc[1, "INSTALLMENT_RECORD_COUNT"], 3)
        self.assertAlmostEqual(result.loc[1, "INSTALLMENT_MEAN_DELAY_DAYS"], 1.5)
        self.assertAlmostEqual(result.loc[1, "INSTALLMENT_MAX_DELAY_DAYS"], 5.0)
        self.assertAlmostEqual(result.loc[1, "INSTALLMENT_LATE_SHARE"], 0.5)
        self.assertAlmostEqual(result.loc[1, "INSTALLMENT_MEAN_SHORTFALL_RATIO"], 0.25)
        self.assertAlmostEqual(result.loc[1, "INSTALLMENT_UNDERPAID_SHARE"], 0.5)
        self.assertEqual(result.loc[1, "ANY_LATE_INSTALLMENT"], 1)
        self.assertEqual(result.loc[1, "ANY_INSTALLMENT_SHORTFALL"], 1)
        self.assertTrue(pd.isna(result.loc[2, "INSTALLMENT_MEAN_SHORTFALL_RATIO"]))

    def test_pos_delinquency_shares(self) -> None:
        path = self.write(
            "POS_CASH_balance.csv",
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1, 1, 1],
                    "SK_DPD": [0, 2, 0],
                    "SK_DPD_DEF": [0, 1, 0],
                }
            ),
        )
        result = aggregate_pos(path, chunksize=2)

        self.assertEqual(result.loc[1, "POS_MONTHS"], 3)
        self.assertEqual(result.loc[1, "POS_RECORD_COUNT"], 3)
        self.assertEqual(result.loc[1, "POS_DPD_MONTHS"], 1)
        self.assertEqual(result.loc[1, "POS_DPD_MAX"], 2)
        self.assertAlmostEqual(result.loc[1, "POS_DPD_SHARE"], 1 / 3)
        self.assertAlmostEqual(result.loc[1, "POS_DPD_DEF_SHARE"], 1 / 3)
        self.assertEqual(result.loc[1, "ANY_POS_DPD"], 1)
        self.assertEqual(result.loc[1, "ANY_POS_DPD_DEF"], 1)

    def test_card_utilization_over_limit_and_zero_limit(self) -> None:
        path = self.write(
            "credit_card_balance.csv",
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1, 1, 1, 2],
                    "AMT_BALANCE": [-50.0, 150.0, 100.0, 50.0],
                    "AMT_CREDIT_LIMIT_ACTUAL": [100.0, 100.0, 0.0, 0.0],
                    "SK_DPD": [0, 2, 0, 0],
                    "SK_DPD_DEF": [0, 1, 0, 0],
                }
            ),
        )
        result = aggregate_credit_cards(path, chunksize=2)

        self.assertEqual(result.loc[1, "CARD_MONTHS"], 3)
        self.assertAlmostEqual(result.loc[1, "CARD_UTIL_MEAN"], 0.75)
        self.assertAlmostEqual(result.loc[1, "CARD_UTIL_MAX"], 1.5)
        self.assertEqual(result.loc[1, "CARD_OVER_LIMIT_MONTHS"], 1)
        self.assertAlmostEqual(result.loc[1, "CARD_OVER_LIMIT_SHARE"], 0.5)
        self.assertAlmostEqual(result.loc[1, "CARD_DPD_SHARE"], 1 / 3)
        self.assertEqual(result.loc[1, "EVER_OVER_CARD_LIMIT"], 1)
        self.assertEqual(result.loc[1, "ANY_CARD_DPD"], 1)
        self.assertTrue(pd.isna(result.loc[2, "CARD_UTIL_MEAN"]))
        self.assertTrue(pd.isna(result.loc[2, "CARD_OVER_LIMIT_SHARE"]))

    def test_bureau_and_previous_presence_flags(self) -> None:
        bureau = self.write(
            "bureau.csv",
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1, 1, 2],
                    "CREDIT_TYPE": ["Consumer credit", "Microloan", "Car loan"],
                }
            ),
        )
        previous = self.write(
            "previous_application.csv",
            pd.DataFrame(
                {
                    "SK_ID_CURR": [1, 2, 2],
                    "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
                }
            ),
        )
        bureau_result = aggregate_bureau(bureau, chunksize=1)
        previous_result = aggregate_previous_applications(previous, chunksize=1)

        self.assertEqual(bureau_result.loc[1, "BUREAU_RECORD_COUNT"], 2)
        self.assertEqual(bureau_result.loc[1, "HAS_MICROLOAN"], 1)
        self.assertEqual(bureau_result.loc[2, "HAS_MICROLOAN"], 0)
        self.assertEqual(previous_result.loc[1, "HAS_REFUSED_PRIOR"], 0)
        self.assertEqual(previous_result.loc[2, "HAS_REFUSED_PRIOR"], 1)


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        base = {
            "DAYS_EMPLOYED": [-100.0, 365243],
            "DAYS_BIRTH": [-10000, -20000],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0],
            "AMT_CREDIT": [50000.0, 0.0],
            "AMT_ANNUITY": [5000.0, 10000.0],
            "AMT_GOODS_PRICE": [45000.0, 0.0],
            "CNT_FAM_MEMBERS": [2.0, 0.0],
            "EXT_SOURCE_1": [0.1, np.nan],
            "EXT_SOURCE_2": [0.2, np.nan],
            "EXT_SOURCE_3": [0.3, np.nan],
            "CATEGORY": ["A", "B"],
        }
        train = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1], **base})
        test_base = {key: [values[0]] for key, values in base.items()}
        test = pd.DataFrame({"SK_ID_CURR": [3], **test_base})
        train.to_csv(self.data_dir / "application_train.csv", index=False)
        test.to_csv(self.data_dir / "application_test.csv", index=False)
        pd.DataFrame({"SK_ID_CURR": [1], "CREDIT_TYPE": ["Microloan"]}).to_csv(
            self.data_dir / "bureau.csv", index=False
        )
        pd.DataFrame(
            {"SK_ID_CURR": [1], "NAME_CONTRACT_STATUS": ["Refused"]}
        ).to_csv(self.data_dir / "previous_application.csv", index=False)
        pd.DataFrame(
            {
                "SK_ID_CURR": [1],
                "DAYS_INSTALMENT": [-10],
                "DAYS_ENTRY_PAYMENT": [-8],
                "AMT_INSTALMENT": [100.0],
                "AMT_PAYMENT": [80.0],
            }
        ).to_csv(self.data_dir / "installments_payments.csv", index=False)
        pd.DataFrame({"SK_ID_CURR": [1], "SK_DPD": [1], "SK_DPD_DEF": [0]}).to_csv(
            self.data_dir / "POS_CASH_balance.csv", index=False
        )
        pd.DataFrame(
            {
                "SK_ID_CURR": [1],
                "AMT_BALANCE": [120.0],
                "AMT_CREDIT_LIMIT_ACTUAL": [100.0],
                "SK_DPD": [1],
                "SK_DPD_DEF": [0],
            }
        ).to_csv(self.data_dir / "credit_card_balance.csv", index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_no_history_schema_target_and_determinism(self) -> None:
        train_a, test_a, metadata_a = build_feature_tables(self.data_dir)
        train_b, test_b, metadata_b = build_feature_tables(self.data_dir)
        assert_frame_equal(train_a, train_b)
        assert_frame_equal(test_a, test_b)

        no_history_train = train_a.set_index("SK_ID_CURR").loc[2]
        no_history_test = test_a.set_index("SK_ID_CURR").loc[3]
        for row in [no_history_train, no_history_test]:
            self.assertEqual(row["INSTALLMENT_RECORD_COUNT"], 0)
            self.assertEqual(row["HAS_INSTALLMENT_HISTORY"], 0)
            self.assertEqual(row["ANY_LATE_INSTALLMENT"], 0)
            self.assertTrue(pd.isna(row["INSTALLMENT_LATE_SHARE"]))
            self.assertEqual(row["POS_RECORD_COUNT"], 0)
            self.assertEqual(row["HAS_POS_HISTORY"], 0)
            self.assertTrue(pd.isna(row["POS_DPD_SHARE"]))
            self.assertEqual(row["CARD_RECORD_COUNT"], 0)
            self.assertEqual(row["HAS_CARD_HISTORY"], 0)
            self.assertTrue(pd.isna(row["CARD_UTIL_MEAN"]))
            self.assertTrue(pd.isna(row["CARD_DPD_SHARE"]))

        self.assertIn("TARGET", train_a)
        self.assertNotIn("TARGET", test_a)
        self.assertEqual(
            train_a.drop(columns="TARGET").columns.tolist(), test_a.columns.tolist()
        )
        self.assertEqual(metadata_a["engineered_feature_count"], len(ENGINEERED_FEATURES))
        self.assertEqual(metadata_a["validation"], metadata_b["validation"])


if __name__ == "__main__":
    unittest.main()
