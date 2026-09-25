# Optimization Process (Exact) and Why the Results Are Trustable

Date: 2026-03-28
Project root: /home/alrezshams/acs_tnout_ulsan

## 1) What we optimized, exactly

We optimized two different objectives, because the paper has two decision layers:

1. Point forecasting quality for TNout (MAE, RMSE, MASE1, MASE7).
2. Alarm utility under bounded operational capacity (top-k precision/recall at fixed budget r).

Forecast horizons used in the project:

- H = 1 day
- H = 3 days
- H = 5 days

Alarm operating points used in final reporting:

- Thresholds tau in {15, 16, 17} mg/L
- Budgets r in {0.05, 0.10}

## 2) Exact data and feature build we used

Primary raw study dataset:

- data/raw/Ulsan_Yongsan.csv

External SI validation dataset:

- data/raw/Seoul_Tancheon_1.csv

Feature build script and exact logic:

- src/build_ulsan_npz_v2.py

What this script does (exactly):

1. Reads Ulsan daily data and sorts by Date.
2. Builds leakage-safe target y = TNout shifted by -H.
3. Builds autoregressive memory features (lag1, lag3, lag5, lag7; rolling 7/14/30).
4. Builds load/weather features and interactions.
5. Adds seasonal encodings (sin_doy, cos_doy).
6. Drops rows with missing feature/target values after shifting/rolling.
7. Saves compressed NPZ per horizon:
   - features/ulsan_H1_features_v2.npz
   - features/ulsan_H3_features_v2.npz
   - features/ulsan_H5_features_v2.npz

Command pattern used:

```bash
/bin/python3 src/build_ulsan_npz_v2.py --h 1
/bin/python3 src/build_ulsan_npz_v2.py --h 3
/bin/python3 src/build_ulsan_npz_v2.py --h 5
```

## 3) Exact optimization procedures we ran

### 3.1 Linear model optimization (main linear benchmark)

Script:

- src/train_main_linear.py

Core settings:

- TimeSeriesSplit(n_splits=3)
- No shuffling (chronological blocked CV)
- Pipeline with StandardScaler fit on training fold only
- Ridge alpha grid: [0.01, 0.1, 1.0, 10.0, 100.0]
- ElasticNet grid:
  - (alpha, l1_ratio) in
  - (0.01,0.2), (0.1,0.2), (1.0,0.2), (0.01,0.5), (0.1,0.5), (1.0,0.5)
- Baselines included:
  - persistence_1
  - seasonal_7

Saved metrics files used in final tables:

- results/metrics/main_linear_metrics_H1.json
- results/metrics/main_linear_metrics_H3.json
- results/metrics/main_linear_metrics_H5.json

Final top linear point results used in paper context:

- H1 best MAE: enet_a0.1_l0.5 (1.5373)
- H3 best MAE: enet_a0.1_l0.5 (1.7191)
- H5 best MAE: enet_a1.0_l0.2 (1.7944)

These values are present in:

- results/final_tables/results_discussion_csvs/Table1_leakage_safe_point_forecasting.csv

### 3.2 HGBR optimization

Script:

- src/train_hgbr_optuna_v2.py

Exact optimization settings from script:

- Optuna study, direction=minimize MAE
- Default trials: 160
- TimeSeriesSplit(n_splits=3)
- HistGradientBoostingRegressor with tuned params:
  - learning_rate
  - max_iter
  - max_leaf_nodes
  - min_samples_leaf
  - l2_regularization
  - max_bins
- early_stopping=True
- validation_fraction=0.1
- random_state=42

Output artifact:

- results/hgbr/hgbr_optuna_H5_v2.json (and analogous per H when run)

### 3.3 BCR-TCN v1.1 optimization (deep model)

Script:

- src/train_bcr_tcn_v11.py

Exact optimization settings from script:

- Seed fixed to 42
- TimeSeriesSplit(n_splits=3)
- Inner validation split in each outer training fold:
  - val_frac = 0.15 (chronological tail split)
- Standardization fit on sub-train only
- Sequence length L default 60
- Model:
  - channels=48
  - blocks=5
  - kernel=3
  - dropout=0.25
- Optimizer: AdamW(lr=8e-4, weight_decay=1e-3)
- Loss = w_reg*Huber + w_bce*BCE + w_rank*pairwise_rank
  - w_reg=1.0
  - w_bce=0.6
  - w_rank=1.4
- Gradient clipping:
  - grad_clip=1.0
- Early stopping:
  - patience=25
- Model selection objective:
  - maximize validation Recall@5% for tau=16

Outputs:

- results/predictions/bcr_tcn_v11_H5_v2_preds.csv
- results/metrics/bcr_tcn_v11_H5_v2_meta.json

### 3.4 Alarm-budget optimization for baseline linear/persistence

Script:

- src/run_linear_alarm_v2.py

Exact behavior:

- Trains ElasticNet (alpha=0.1, l1_ratio=0.5) fold-wise
- Computes persistence_v2 fold-wise
- Evaluates alarm metrics at tau={15,16,17}, r={0.05,0.10}

Important currently-materialized prediction files:

- results/predictions/enet_a0.1_l0.5_v2_H5_preds.csv
- results/predictions/persistence_v2_H5_preds.csv

### 3.5 Hybrid rank ensemble optimization (final H5 policy run)

Script:

- src/make_hybrid_rank_ensemble_v2.py

Exact optimization logic:

1. Point hybrid weights optimized on training folds only (grid over convex weights).
2. Rank hybrid weights optimized fold-wise and tau-wise on training folds only.
3. rank_objective used in final selected run: policy_guarded.
4. Precision floor constraint vs paper-fixed baseline at key conservative point.
5. Guarded fallback for low budget (r <= 0.05) and policy-specific behavior for higher budget.

Critical run config that drives final guarded H5 tables:

- results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/run_config.json

Exact run settings in that config:

- horizon: 5
- taus: [15,16,17]
- budgets: [0.05, 0.1]
- weight_step: 0.1
- rank_objective: policy_guarded
- guard_budget_max: 0.05
- inputs:
  - results/predictions/bcr_tcn_v11_H5_v2_preds.csv
  - results/predictions/enet_a0.1_l0.5_v2_H5_preds.csv
  - results/predictions/persistence_v2_H5_preds.csv
  - results/predictions/hgbr_optuna_H5_v2_preds.csv

Major output folder:

- results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/

Contains exactly:

- run_config.json
- models/ (weights, model_registry_H5.csv)
- predictions/ (hybrid_point, enet, persistence, hgbr, rank score files)
- tables/ (point metrics, alarm metrics, key operating points)
- figures/ (learning curves and budget-recall visuals)

## 4) Exact final manuscript-level CSVs generated from this optimization process

Main final CSV handoff folder:

- results/final_tables/results_discussion_csvs/

Key files:

- Table1_leakage_safe_point_forecasting.csv
- Table2_exceedance_prevalence_random_baseline.csv
- Table3_alarm_budget_H5_core_operating_points.csv
- Table4_H5_normalized_context_comparison.csv
- Table5_key_operating_points_fold_uncertainty.csv
- Fig3_budget_recall_input.csv
- Fig4_uncertainty_bars_input.csv
- Fig5_decision_curve_input.csv
- Fig6_silence_safety_frontier_input.csv

## 5) Why these results are trustable

### 5.1 Leakage controls (hard safeguards)

1. Chronological blocked CV (TimeSeriesSplit), no shuffle.
2. Feature generation uses only information up to time t for predicting t+H.
3. Scaling fit only on training portion of each fold.
4. TCN inner validation split is chronological and inside outer train only.
5. Hybrid weight search is train-fold only, then applied to held-out fold.

### 5.2 Optimization stability controls

1. Fixed random seed (42) for deep model and deterministic framework settings where defined.
2. Gradient clipping at 1.0 to prevent exploding updates.
3. Early stopping with patience to avoid overtraining.
4. Guarded low-budget policy to avoid regressions at conservative alarm budgets.

### 5.3 Decision-level validity controls

1. Metrics aligned to operational constraint (top-k under fixed r).
2. Precision and recall both reported; not recall-only optimization.
3. Fold-level uncertainty tables available for key points.
4. SI failure diagnostics available (TP/FP/FN/TN, miss rate, false alarm rate).

Evidence files:

- results/tables/blocked_cv_split_summary_all_horizons.csv
- results/tables/si_gradient_ve_summary_H5.csv
- results/tables/si_overfit_underfit_counts_H5.csv
- results/tables/si_failure_bars_H5.csv
- results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table5_key_operating_points_fold_uncertainty.csv

## 6) Exact statement of what was selected for final claims

For final H5 guarded-comparison claims in manuscript-level tables, the selected run lineage is:

- results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/

For normalized shared-context comparison and uncertainty package generation, additional artifact pipeline references:

- results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/
- results/hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2/ (used by SI/support export scripts and normalization workflows)

This is why all main claim CSVs include explicit source file trace fields and can be audited back to run directories.

## 7) One-line trustability summary

The results are trustable because optimization, selection, and evaluation were all time-ordered and leakage-safe, and every key paper number is traceable to versioned run artifacts (with run config, prediction files, and fold-aware metrics preserved).