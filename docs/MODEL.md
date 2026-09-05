# Modelling Notes

This project contains two notebooks that approach the same problem at
increasing levels of sophistication. Both are self-contained and can be run
independently.

## 1. `notebooks/01_baseline_model.ipynb` — Baseline pipeline

A documented, linear walkthrough of the full workflow:

1. **EDA** — missingness/duplicate checks, target distribution, feature vs.
   target scatter plots, correlation matrix.
2. **Feature engineering** — a small, physics-motivated set of ~15 features
   (e.g. `delta_T`, `temp_ratio`, `residence_time`), computed with row-wise
   arithmetic only, so it introduces no leakage when applied before the
   train/validation split.
3. **Validation strategy** — `RepeatedKFold(n_splits=5, n_repeats=3,
   random_state=42)`, scored on negative RMSE, with 5-fold/10-fold used as
   sanity checks that the ranking of models doesn't depend on the CV scheme.
4. **Model comparison** — Ridge, SVR, Random Forest, Extra Trees, Gradient
   Boosting and Hist Gradient Boosting, each wrapped in a `Pipeline`
   (imputation + optional scaling) so preprocessing is refit per fold.
5. **Hyperparameter tuning** — `RandomizedSearchCV` over small, focused
   search spaces for the top model families.
6. **Ensembling** — voting and stacking ensembles built from the top
   individual/tuned models, accepted only if they beat every single model
   under the same repeated CV.
7. **Final fit & submission** — the selected model (a two-way voting
   ensemble in the reference run) is refit on all training rows, used to
   predict the test set, clipped to the valid `[0, 100]` yield range, and
   written to `submission.csv`.

Reference result from this notebook's own run: **CV RMSE ≈ 19.06** (voting
ensemble, top 2 models, equal weights), which produced the leaderboard
submission tracked in this repo as
`outputs/submission_baseline_RMSE_18_7105.csv`.

## 2. `notebooks/02_feature_selection_and_ensembling.ipynb` — Extended pipeline

A follow-up notebook that automates parts of the baseline's manual choices:

- Expands the feature set to ~20 engineered columns (thermal, residence-time,
  interaction, and squared terms) and evaluates **six candidate feature
  subsets** with an Extra Trees model under the same repeated CV, selecting
  the best-performing subset automatically rather than hand-picking features.
- Adds a **Gaussian Process Regressor** (Matern + white-noise kernel) to the
  candidate model pool alongside the tree/boosting/linear models from the
  baseline.
- Builds a larger grid of **weighted voting ensembles** (11 weight
  combinations) plus a stacking regressor, in addition to the two ensembles
  used in the baseline.
- Adds a **seed-stability check**: the best individual model, best ensemble,
  and overall winner are each re-scored under `RepeatedKFold` with 5
  different random seeds, to confirm the model ranking isn't an artifact of
  one particular CV split.
- Produces `submission_optimized.csv` following the same clip-and-round
  convention as the baseline.

This notebook is exploratory — its output filename is not the same as the
score-labelled file tracked under `outputs/`, so no verified leaderboard
score is currently associated with it. Treat it as the ongoing experimentation
track rather than the production baseline.

## Reproducing a run

Both notebooks expect `train_dataset.csv` and `test_dataset.csv` to be
readable via a **relative path** (`"train_dataset.csv"`, `"test_dataset.csv"`).
Since the data now lives in `data/`, either:

- run the notebook from inside `data/` (e.g. copy/symlink the notebook there
  temporarily), or
- update the two `pd.read_csv(...)` calls near the top of each notebook to
  point at `"../data/train_dataset.csv"` and `"../data/test_dataset.csv"`.

The second option is the recommended one and is called out in the main
`README.md` setup steps.

All randomness is seeded (`random_state=42` throughout), so results should be
reproducible on the same library versions.
