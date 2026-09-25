# ElasticNet Definition and Provenance

## 1. Scope and decision gate
Objective: recover the submitted ElasticNet baseline family definition for H1, H3, and H5, classify provenance confidence, and predeclare a leakage-safe corrected rerun strategy.

Classification: B

Meaning of classification B:
- Candidate hyperparameter grid is confirmed from released source code.
- Released artifacts show multiple held-out-informed variant/context switches (including H5 v1/v2 context mixing), so a single submitted fixed configuration for corrected evaluation cannot be used as authoritative.
- Corrected rerun must perform fold-local nested selection using only corrected purged inner partitions.

## 2. Provenance evidence searched
Legacy scripts:
- src/train_main_linear.py
- src/run_linear_alarm.py
- src/run_linear_alarm_v2.py
- src/save_fold_predictions.py
- src/build_ulsan_npz.py
- src/build_ulsan_npz_v2.py

Claim-verification mapping:
- src/verify_claimed_values_reproducibility.py

Metrics and summaries:
- results/metrics/main_linear_metrics_H1.json
- results/metrics/main_linear_metrics_H3.json
- results/metrics/main_linear_metrics_H5.json
- results/metrics/alarm_budget_H1.csv
- results/metrics/alarm_budget_H3.csv
- results/metrics/alarm_budget_H5.csv
- results/metrics/alarm_budget_v2_H5.csv
- reports/final_model_summary.md
- reports/model_performance_full_report.md
- reports/reproducibility_declaration_submission_ready_20260406.md

Protocol/rerun decisions:
- revision_2026/02_corrected_pipeline/model_rerun_decision.csv
- revision_2026/02_corrected_pipeline/temporal_leakage_audit_decision.md
- revision_2026/03_corrected_protocol/corrected_evaluation_protocol.md

## 3. Recovered submitted ElasticNet family definition
Model implementation:
- sklearn.linear_model.ElasticNet inside a sklearn Pipeline with StandardScaler.

Recovered candidate grid:
- (alpha=0.01, l1_ratio=0.2)
- (alpha=0.1, l1_ratio=0.2)
- (alpha=1.0, l1_ratio=0.2)
- (alpha=0.01, l1_ratio=0.5)
- (alpha=0.1, l1_ratio=0.5)
- (alpha=1.0, l1_ratio=0.5)

Recovered fixed/default arguments used in legacy scripts:
- fit_intercept=True (default)
- max_iter=20000
- tol not explicitly set in legacy scripts, so sklearn default applies (0.0001 in sklearn 1.3.2)
- random_state=42
- missing-value handling: submitted feature builders drop rows with missing required features/target before model fitting (no imputer in pipeline)

## 4. Recovered submitted feature schema by horizon
Authoritative source for recovered submitted main-linear feature schema:
- results/metrics/main_linear_metrics_H1.json
- results/metrics/main_linear_metrics_H3.json
- results/metrics/main_linear_metrics_H5.json

Recovered submitted main-linear feature set (same 18 columns for H1/H3/H5):
1. TNout_lag1
2. TNout_roll7
3. TNout_roll14
4. Inflow
5. Inflow_roll7
6. TNin
7. TNin_roll7
8. TOCin
9. TOCin_roll7
10. BODin
11. BODin_roll7
12. C_N
13. temp_mean_c
14. temp_roll7
15. precip_total_mm
16. precip_sum3
17. sin_doy
18. cos_doy

Target/date mapping:
- For each horizon H in {1,3,5}, y_true is TNout shifted by -H from feature_date.
- target_date = feature_date + H days.

## 5. Submitted final-configuration lineage recovered
Point-forecast summary lineage (main linear context):
- reports/final_model_summary.md lists:
  - H1 best MAE: enet_a0.1_l0.5
  - H3 best MAE: enet_a0.1_l0.5
  - H5 best MAE: enet_a1.0_l0.2

Representative alarm-claim lineage:
- src/verify_claimed_values_reproducibility.py maps representative-check rows to:
  - H1: enet_a0.1_l0.5_H1_preds.csv
  - H3: enet_a1.0_l0.2_H3_preds.csv and enet_a0.01_l0.2_H3_preds.csv
  - H5 (generic enet): enet_a0.1_l0.5_v2_H5_preds.csv

Interpretation:
- H1/H3/H5 reporting did not use one globally predeclared ElasticNet configuration.
- H3 and H5 claims include operating-point-specific or context-specific variant switches.

## 6. H5 v1/v2 lineage recovery
Recovered H5 v1 lineage:
- Predictions exist for all six grid variants as non-v2 files:
  - results/predictions/enet_a*_H5_preds.csv
- These artifacts have N=759 (3 folds x 253 rows).
- Feature lineage aligns with build_ulsan_npz.py (18-feature schema).

Recovered H5 v2 lineage:
- Fixed model run exists at:
  - results/predictions/enet_a0.1_l0.5_v2_H5_preds.csv
- This artifact has N=747 (3 folds x 249 rows).
- Script lineage is run_linear_alarm_v2.py with NPZ input ulsan_H5_features_v2.npz.
- build_ulsan_npz_v2.py defines expanded v2 features (34 columns).

Risk implication:
- H5 comparisons mix at least two ElasticNet contexts (v1 grid context and v2 fixed-config context).
- This context switch is consistent with held-out-informed selection/reporting risk flagged in temporal_leakage_audit_decision.md.

## 7. Corrected rerun authorization outcome
Status B rerun policy for this controlled rerun:
- Use confirmed submitted ElasticNet candidate grid (6 alpha/l1_ratio combinations).
- Use the recovered submitted main-linear feature schema by horizon from main_linear_metrics_H1/H3/H5 JSON files.
- Perform fold-local nested selection for each horizon and outer fold, using only corrected inner_train/inner_validation rows from corrected_split_assignment.csv (with purged_inner_boundary rows excluded from inner_train).
- Enforce strict outer-test isolation during selection.

Why fixed-config replay is not authorized as corrected selection authority:
- Released artifacts demonstrate post-hoc variant/context switching across horizons and operating points.
- H5 v2 is a single fixed-context run, not a complete predeclared nested selection result across the full submitted grid.
