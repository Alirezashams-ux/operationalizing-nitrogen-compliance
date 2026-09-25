# Ridge Definition and Provenance

## 1. Scope and decision gate
Objective: recover the submitted Ridge baseline definition and determine whether fixed-config replay is possible, or fold-local alpha selection is required.

Classification: B

Meaning of classification B:
- Submitted alpha candidate grid is confirmed.
- A single fixed submitted alpha for all H1/H3/H5 cannot be confirmed from released per-horizon Ridge prediction artifacts.
- Alpha must be selected inside each corrected purged outer-training fold using only inner-train and inner-validation rows from corrected_split_assignment.csv.

## 2. Provenance evidence searched
Legacy scripts:
- src/train_main_linear.py
- src/build_ulsan_npz.py
- src/build_ulsan_npz_H1.py
- src/run_linear_alarm.py
- src/run_linear_alarm_v2.py

Release package code mirrors:
- deliverables/public_release_split_20260406/code/src/train_main_linear.py

Tables and metrics artifacts:
- results/metrics/main_linear_metrics.json
- results/metrics/main_linear_metrics_H1.json
- results/metrics/main_linear_metrics_H3.json
- results/metrics/main_linear_metrics_H5.json
- results/tables/main_linear_leaderboard.csv
- results/tables/main_linear_leaderboard_by_rmse.csv

SI and methods materials:
- reports/Optimization_Process_Exact_Trustable_Results.md
- reports/methods and materials_SI.md
- reports/complete_methods_foundations_novelty_report.md

Protocol/rerun decisions:
- revision_2026/02_corrected_pipeline/model_rerun_decision.csv
- revision_2026/03_corrected_protocol/rerun_execution_plan.csv

## 3. Recovered submitted Ridge definition
Feature set by horizon:
- H1 features from results/metrics/main_linear_metrics_H1.json: 18 columns.
- H3 features from results/metrics/main_linear_metrics_H3.json: same 18-column set.
- H5 features from results/metrics/main_linear_metrics_H5.json: same 18-column set.

Recovered feature list (all horizons):
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

Target definition:
- For horizon H, target is TNout shifted by -H relative to feature_date.
- Equivalent date mapping: y_true at target_date equals TNout(feature_date + H days).

Preprocessing pipeline:
- StandardScaler is used inside sklearn Pipeline for Ridge in src/train_main_linear.py.
- Scaling is fit on each training fold in legacy CV loops.

Ridge implementation:
- sklearn.linear_model.Ridge(alpha=a, random_state=42) in src/train_main_linear.py.
- fit_intercept is default True.
- solver is default auto.

Alpha candidate grid:
- [0.01, 0.1, 1.0, 10.0, 100.0]
- Confirmed in src/train_main_linear.py and mirrored in deliverable code copy.

Missing-value handling:
- Feature-building path drops rows with NA in required features and target before training (src/build_ulsan_npz.py).
- No explicit imputer in submitted Ridge pipeline.

Random-state relevance:
- random_state=42 is passed in legacy Ridge constructor.
- With solver=auto on dense tabular arrays, Ridge solution is deterministic and random_state is not expected to alter the solution path.

## 4. Fixed versus tuned alpha recovery
Was alpha fixed in submission?
- Not confirmed as one fixed, horizon-agnostic alpha for H1/H3/H5 prediction artifacts.
- Released per-horizon point-metric files main_linear_metrics_H1/H3/H5 do not include Ridge entries.

Was Ridge tuned?
- Yes, candidate-grid benchmarking exists (main_linear_metrics.json and leaderboard tables).

How tuning was performed in legacy workflow:
- Grid candidates were evaluated on blocked CV folds and compared via fold-averaged summaries.
- Legacy process was not the corrected nested, purged inner-selection protocol required now.

Did outer-test performance influence legacy alpha selection?
- Yes in legacy benchmarking sense: fold held-out outcomes were compared to rank candidates.
- This is exactly why corrected rerun requires fold-local inner selection under purged splits.

Were H1, H3, H5 settings the same or separate?
- Feature schema is the same 18-column list across H1/H3/H5.
- No released evidence confirms one fixed submitted alpha for all three horizons under corrected isolation.
- Therefore alpha selection must be performed separately for each horizon and outer fold.

## 5. Authorization outcome
Classification B is reached and Ridge fitting is authorized only under this rule:
- Use the confirmed submitted alpha grid.
- Select alpha independently inside each corrected purged outer-training fold using only inner-train and inner-validation rows.
- Never use outer-test outcomes for alpha selection.
