# Dataset

## Overview

The task is a supervised regression problem: predict `overall_yield` (percent
conversion of feed, in the range 0–100) for a tubular chemical reactor from
five operating conditions.

| File | Rows | Columns | Purpose |
|---|---|---|---|
| `data/train_dataset.csv` | 150 | 6 (5 features + target) | Model training / cross-validation |
| `data/test_dataset.csv` | 50 | 5 (features only) | Held-out rows to generate predictions for |

## Columns

| Column | Type | Description |
|---|---|---|
| `flow_rate_L_min` | float | Volumetric feed flow rate through the reactor (L/min) |
| `concentration_mol_L` | float | Inlet reactant concentration (mol/L) |
| `inlet_temperature_K` | float | Feed temperature at the reactor inlet (K) |
| `length_m` | float | Reactor tube length (m) |
| `jacket_temperature_K` | float | Temperature of the heating/cooling jacket (K) |
| `overall_yield` | float | **Target.** Percent conversion of the feed (0–100). Present only in `train_dataset.csv`. |

## Known characteristics (from EDA in `notebooks/01_baseline_model.ipynb`)

- No missing values and no duplicate rows in the training set.
- The target is **bimodal**: a meaningful subset of rows sit at (near) zero
  yield — operating regimes where the reaction effectively does not proceed —
  while the remaining rows cluster at high yield. This is why simple linear
  models underperform tree-based / ensemble models on this data.
- `jacket_temperature_K` and the gap between jacket and inlet temperature are
  the strongest visible drivers of yield.
- Flow rate and reactor length matter primarily through **residence time**
  (`length_m / flow_rate_L_min`), not in isolation.

## Obtaining the data

The CSV files are small (a few KB) and are committed directly under `data/`.
No external download step is required to reproduce the notebooks.
