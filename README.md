# Home Credit Default Risk

Predicting which loan applicants will have repayment difficulty, using the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset: 307,511 training applicants, 48,744 test applicants, and seven linked tables of
credit-bureau, previous-application, installment, POS and credit-card history.

My goal here was not to chase the leaderboard with a large stack of automated features. I
wanted every feature in the final model to trace back to something I had actually measured
in the EDA, and I wanted the evaluation to be honest: all preprocessing fitted inside
cross-validation folds, no target leakage, no feature kept without evidence.

![Home Credit](data/home_credit.png)

## Results

5-fold stratified cross-validation, ROC-AUC on the full training set.

| Model | Features | Mean CV AUC | Std |
|---|---|---|---|
| Logistic regression (median/mode impute, scale, one-hot) | 201 | 0.7668 | 0.0034 |
| LightGBM, untuned | 201 | 0.7827 | 0.0036 |
| LightGBM, tuned | 201 | 0.7852 | 0.0036 |
| CatBoost, native categoricals | 201 | 0.7855 | 0.0036 |
| **Blend — 0.45 LightGBM / 0.55 CatBoost** | 201 | **0.7872** (OOF) | — |

The blend only buys about 0.0017 AUC over CatBoost alone. Worth taking, but it is not
where the gains came from.

## Where the gains actually came from

I added feature families one at a time and re-ran the same CV each time:

| Feature set | Features | Mean CV AUC | Gain |
|---|---|---|---|
| Raw application columns only | 120 | 0.7595 | — |
| + application-engineered | 131 | 0.7673 | +0.0078 |
| + bureau | 146 | 0.7728 | +0.0055 |
| + bureau balance | 151 | 0.7730 | +0.0002 |
| + previous applications | 165 | 0.7761 | +0.0032 |
| + installments | 176 | 0.7799 | +0.0038 |
| + POS/CASH | 186 | 0.7806 | +0.0007 |
| + credit card | 199 | 0.7830 | +0.0024 |
| + cross-table ratios | 201 | 0.7827 | −0.0004 |

Feature engineering was worth about +0.023 AUC over the raw application table, considerably
more than model choice or tuning. A leave-one-group-out ablation agrees on the ranking:
removing the engineered application features costs the most (−0.0035), then previous
applications (−0.0031) and bureau (−0.0024). Bureau-balance and the cross-table ratios do
not pay for themselves, which I have left in the notebook rather than quietly deleting.

By fold-wise permutation importance, `EXT_SOURCE_MEAN` dominates everything else. Permuting
it costs 0.060 AUC, an order of magnitude more than the next feature. After it come
`ANNUITY_CREDIT_RATIO` (0.0084), `AMT_ANNUITY` (0.0041), `EXT_SOURCE_1` (0.0035),
`CREDIT_GOODS_RATIO` (0.0033), `PREV_APPLICATION_CREDIT_RATIO_MEAN` (0.0028) and
`INSTALLMENT_LATE_SHARE` (0.0024).

## Notebooks

The three notebooks run in order and are committed with their outputs, so you can read the
whole analysis on GitHub without downloading the data.

**`notebooks/01_eda.ipynb`.** I work through the application table and then each relational
table with the same loop: question, action, finding, implication. Findings that drove later
decisions: an 8.07% default rate, 49 application columns at least 40% missing, 55,374
`DAYS_EMPLOYED` sentinel values (365243) that align exactly with `ORGANIZATION_TYPE == "XNA"`
and are *less* likely to default, 14 redundant AVG/MODE/MEDI housing triplets with pairwise
correlations above 0.963, and clear risk separation for late installments (+2.69pp), POS
delinquency (+2.41pp) and ever exceeding a card limit (+4.20pp). I also audited every
`DAYS_*` / `MONTHS_BALANCE` field to confirm nothing observed after the application date
leaks into the features.

**`notebooks/02_features.ipynb`.** Builds one applicant-level table: 121 original
application columns plus 81 engineered features (203 columns including `SK_ID_CURR` and
`TARGET`). Each history table is reduced to exactly one row per applicant and joined with a
validated one-to-one merge. The large event tables are streamed in chunks. Nothing here is
fitted: no imputers, encoders, scalers, bins or thresholds, and no target information. The
notebook builds the tables twice and compares SHA-256 fingerprints to prove the output is
deterministic.

**`notebooks/03_modeling.ipynb`.** Logistic-regression baseline, LightGBM, incremental
feature-group and ablation experiments, gain and permutation importance, tuning
(complexity, then sampling), CatBoost, and an OOF-weighted blend. Preprocessing is fitted
inside each fold and early stopping uses only the validation fold.

## Reusable pipeline

`src/features/` is the feature build extracted into a tested Python package:

```bash
python -m src.features.build_features \
  --data-dir data \
  --output-dir data/processed \
  --verify-determinism

python -m unittest discover -s tests -v
```

One caveat worth stating plainly: this package covers my **first-pass** feature set (50
engineered features), not the 81-feature set the final model uses. The full set currently
lives in `02_features.ipynb`. Porting the remaining features across is the next thing on my
list.

## Running it yourself

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ~688 MB archive, ~2.5 GB extracted; needs a Kaggle API token
kaggle competitions download -c home-credit-default-risk -p data
unzip data/home-credit-default-risk.zip -d data

jupyter lab
```

The raw CSVs and the generated Parquet tables are gitignored. They are too large for the
repo and are reproducible from the command above.

```

