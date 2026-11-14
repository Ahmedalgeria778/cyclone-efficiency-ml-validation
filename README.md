# Cyclone Collection Efficiency: Geometry-Grouped Validation, Physics-Inspired Feature Ablation, and Uncertainty Assessment

Repository accompanying the manuscript submitted to *Powder Technology* (POWTEC-D-26-03167, revised version R1).

## Authors

A. Hanifi, A. Beniaiche, A. Belkalouche, S. Ghodbane, C. Belkacemi, M. Belkadi

## Overview

This repository contains all code, data, and results needed to reproduce the experiments reported in the manuscript. The study examines machine-learning prediction of cyclone collection efficiency across geometries, with emphasis on:

1. **Geometry-grouped validation** — random splits vs. leave-one-geometry-out (LOGO)
2. **Physics-inspired feature ablation** — 17 input features including a vortical-capture group Pi_VC
3. **Uncertainty quantification** — heteroscedastic intervals, conformal prediction, Mahalanobis extrapolation indicator
4. **Cut-diameter analysis** — Stk_50 vs Re scaling with propagated uncertainty

## Repository structure

```
github_repo/
  code/          Python scripts (feature construction, models, protocols, analysis)
  data/          Efficiency database (977 points, 28 geometries)
  results/       CSV outputs, JSON summaries, bootstrap analyses
  figures/       Publication figures (PNG)
  README.md      This file
  requirements.txt
```

## Data

The efficiency database is from the supplementary material of Bregolin et al. (*Waste* 2, 2024, 240--257). It contains 977 unique measurements (after removing 3 duplicates) spanning 28 cyclone geometries, with 7 inputs per record:

| Variable | Symbol | Description |
|----------|--------|-------------|
| De/D | D_e/D | Vortex finder diameter ratio |
| a/D | a/D | Inlet height ratio |
| b/D | b/D | Inlet width ratio |
| Re | Re | Reynolds number |
| Stk | Stk | Stokes number |
| Dp/D | D_p/D | Particle diameter ratio |
| rr | rho_p/rho_g | Particle-to-gas density ratio |
| eta | eta | Collection efficiency |

## Code

| Script | Purpose |
|--------|---------|
| `common.py` | Data loading, Pi_VC definition, calibration, feature construction, metrics |
| `models.py` | Model definitions (GBR, XGB, LGBM, CatBoost, RF, HGB, MLP) |
| `run_logo.py` | LOGO validation (28 folds, 5 feature sets, 6 tree models + MLP) |
| `run_split.py` | Repeated grouped splits, random splits, L2GO, tuned protocol |
| `analysis_logo.py` | LOGO analysis: per-geometry metrics, Mahalanobis, intervals, Re bands |
| `bootstrap_macro_rms.py` | Paired bootstrap of macro-RMS difference (audit point 4) |
| `verify_table6.py` | Numerical verification of Table 6 values |
| `gen_tables.py` | LaTeX table generation |
| `figs.py` | Figure generation |
| `stk50.py` / `stk50_s.py` | Cut-diameter extraction and regression |
| `re_bands.py` | Reynolds-number band analysis |

## Requirements

```
numpy>=1.20
pandas>=1.3
scipy>=1.7
scikit-learn>=1.0
xgboost>=1.5
lightgbm>=3.0
catboost>=1.0
matplotlib>=3.3
```

## Reproduction

```bash
pip install -r requirements.txt

# Run LOGO validation (produces logo_preds.pkl, logo_pars.pkl)
cd code
python run_logo.py

# Run split protocols (fixed, tuned, random, L2GO)
python run_split.py

# Analyze LOGO results (produces CSV tables, JSON summaries)
python analysis_logo.py

# Bootstrap macro-RMS (audit correction)
python bootstrap_macro_rms.py

# Verify Table 6 values
python verify_table6.py
```

## Key results

| Metric | Random split | Grouped 23/5 | LOGO |
|--------|-------------|-------------|------|
| GBR R^2 (raw) | 0.948 | 0.727 | 0.731 |
| GBR R^2 (17 feat.) | 0.945 | 0.738 | 0.740 |
| GBR RMSE (LOGO, 17) | -- | -- | 0.173 |

The engineered features (17 inputs) provide small consistent gains under tuned hyperparameters (RMSE reduction 0.001--0.011) but **no statistically significant gain under LOGO** (bootstrap CI of Delta-RMSE_macro contains zero for all models).

## License

The code is provided for reproducibility of the manuscript. The dataset belongs to the original authors (Bregolin et al., 2024).
