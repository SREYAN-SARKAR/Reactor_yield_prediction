# Reactor Yield Prediction

A regression pipeline that predicts the **overall yield** of a tubular
chemical reactor from five operating conditions (flow rate, feed
concentration, inlet temperature, reactor length, and jacket temperature).
Built for a data-science hackathon; hold-out submission RMSE **18.71**.

## Overview

Given a small tabular dataset (150 training rows, 50 test rows) describing
reactor operating conditions, the goal is to predict `overall_yield` — the
percent conversion of the feed, in the range 0–100.

The target distribution is **bimodal**: a subset of operating points produce
(near) zero yield, while the rest cluster at high yield. The pipelines in this
repo are built around that observation — physics-motivated feature
engineering (residence time, temperature deltas/ratios) combined with
tree/ensemble models, validated with repeated k-fold cross-validation rather
than a single train/test split, since 150 rows is too small for one split to
be trustworthy.

## Repository Structure

```text
reactor-yield-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── train_dataset.csv          # 150 labelled rows
│   └── test_dataset.csv           # 50 unlabelled rows to predict
├── notebooks/
│   ├── 01_baseline_model.ipynb    # documented, linear walkthrough (reference: CV RMSE ~19.06)
│   └── 02_feature_selection_and_ensembling.ipynb  # automated feature-set search + wider model/ensemble sweep
├── outputs/
│   └── submission_baseline_RMSE_18_7105.csv  # tracked leaderboard submission from notebook 01
└── docs/
    ├── DATASET.md                 # column definitions and EDA findings
    └── MODEL.md                   # methodology detail for both notebooks
```

## How It Works

1. **Load & validate** the train/test CSVs, check for missing values and
   duplicates.
2. **Engineer features** from the five raw inputs — temperature deltas and
   ratios, a residence-time proxy (`length / flow_rate`), and interaction
   terms — using stateless, row-wise arithmetic (no data leakage).
3. **Cross-validate** a pool of linear, kernel, bagging, and boosting models
   with `RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)`, scored by
   RMSE.
4. **Tune** the strongest model families with a small, focused
   `RandomizedSearchCV` space.
5. **Ensemble** the top models (voting / stacking) and keep the ensemble only
   if it beats every individual model under the same CV scheme.
6. **Refit** the selected model on all training data, predict the test set,
   clip predictions to the valid `[0, 100]` range, and write a submission
   CSV.

See [`docs/MODEL.md`](docs/MODEL.md) for the full methodology of each
notebook, and [`docs/DATASET.md`](docs/DATASET.md) for column definitions and
EDA findings.

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data handling | pandas, NumPy |
| Modelling | scikit-learn (Ridge, SVR, Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting, Gaussian Process, Voting/Stacking ensembles) |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |

## Installation

### Prerequisites

- Python 3.10+
- `pip`

### Setup

```bash
git clone <repository-url>
cd reactor-yield-prediction

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Launch Jupyter and run either notebook from inside the `notebooks/` folder —
both already reference the dataset via `../data/...` and write their
submission file to `../outputs/...`:

```bash
jupyter notebook notebooks/01_baseline_model.ipynb
```

or

```bash
jupyter notebook notebooks/02_feature_selection_and_ensembling.ipynb
```

Run all cells top to bottom. Each notebook ends by writing its own
submission CSV into `outputs/`:

- `01_baseline_model.ipynb` → `outputs/submission.csv`
- `02_feature_selection_and_ensembling.ipynb` → `outputs/submission_optimized.csv`

## Dataset

| File | Rows | Description |
|---|---|---|
| `data/train_dataset.csv` | 150 | 5 features + `overall_yield` target |
| `data/test_dataset.csv` | 50 | 5 features, no target — this is what you predict |

Full column definitions and EDA findings are in
[`docs/DATASET.md`](docs/DATASET.md).

## Methodology & Results

| Notebook | Approach | Reference CV RMSE |
|---|---|---|
| `01_baseline_model.ipynb` | Hand-picked ~15 physics-informed features, documented model comparison, tuning, and ensembling | ≈ 19.06 (voting ensemble, top 2 models) |
| `02_feature_selection_and_ensembling.ipynb` | Automated search over 6 candidate feature sets, adds a Gaussian Process model, wider ensemble weight grid, seed-stability check | Exploratory — see notebook output for the current run's leaderboard |

The tracked submission file, `outputs/submission_baseline_RMSE_18_7105.csv`,
corresponds to the baseline notebook and scored **RMSE 18.71** on the
hackathon's evaluation.

Full methodology write-up: [`docs/MODEL.md`](docs/MODEL.md).

## Limitations

- The training set is very small (150 rows), so cross-validation scores carry
  meaningful variance (see the CV std columns in each notebook's leaderboard)
  — treat RMSE differences smaller than that variance as noise, not a
  meaningful ranking.
- Feature engineering is hand-designed from physical intuition about the
  reactor rather than learned; it may not generalize to reactor
  configurations very different from those in the training data.
- The second notebook (`02_...ipynb`) is an active experimentation track —
  its output has not been benchmarked against the hackathon leaderboard the
  way the baseline has.

## Future Improvements

- Benchmark `02_feature_selection_and_ensembling.ipynb`'s predictions against
  the hackathon leaderboard and record the result the same way the baseline's
  is tracked.
- Factor the duplicated feature-engineering and CV helper code in both
  notebooks into a shared, importable module if the project grows beyond
  notebook-based experimentation.
- Add a small unit test around the feature-engineering function(s) to guard
  against silent regressions (e.g. divide-by-zero guarding, expected output
  columns).

## Contributing

This started as a hackathon submission. If you'd like to
extend it, feel free to open an issue or pull request — please keep any new
model/feature experiments in a new notebook (or a shared module, per
"Future Improvements" above) rather than editing the tracked baseline, so the
reference RMSE stays reproducible.

## License

No license has been selected yet. Until one is added, all rights are
reserved by the author. If you intend to share this project publicly,
consider the MIT License for a permissive, low-friction choice common to
portfolio/hackathon projects.

## Author

Sreyan Sarkar.
