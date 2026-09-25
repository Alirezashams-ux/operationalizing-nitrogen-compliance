# Corrected Ridge Baseline Completion Report

## 1. Input Verification
Input verification output: revision_2026/04_controlled_reruns/ridge/input_verification.json

Verified items:
- dataset SHA-256 expected and observed are identical:
  - 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb
- corrected split SHA-256:
  - f83baf5732ffbf88b9a448a3d9ab9fa270f7495fe44d1559c1ac49795a30881a
- corrected protocol checksum list validation:
  - all listed files passed
- branch gate:
  - controlled-reruns-v1
- git commit:
  - 41f184a29ff88aaf275f3365267b8c909ce7598a

Result: PASS.

## 2. Submitted Ridge Provenance
Provenance document: revision_2026/04_controlled_reruns/ridge/ridge_definition_and_provenance.md

Recovered Ridge definition highlights:
- model implementation: sklearn Ridge inside a fold-local StandardScaler pipeline
- candidate alpha grid: [0.01, 0.1, 1.0, 10.0, 100.0]
- fit_intercept: True (default)
- solver: auto (default)
- random_state argument: 42 in legacy script
- missing-value handling: submitted feature builder drops rows with missing required features/target before training

Recovered provenance classification:
- B

Implication:
- fixed submitted alpha across H1/H3/H5 could not be confirmed from released prediction artifacts
- alpha must be selected per horizon and per outer fold using inner-only corrected split partitions

## 3. Feature Sets
Feature source used for recovery:
- results/metrics/main_linear_metrics_H1.json
- results/metrics/main_linear_metrics_H3.json
- results/metrics/main_linear_metrics_H5.json

Recovered feature set (same 18 columns for H1/H3/H5):
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

## 4. Hyperparameter Strategy
Selection plan: revision_2026/04_controlled_reruns/ridge/ridge_selection_plan.json

Applied strategy:
- status B workflow
- for each horizon and outer fold, evaluate every alpha candidate using only:
  - inner_train rows labeled in corrected_split_assignment.csv with purged_inner_boundary excluded
  - inner_validation rows labeled in corrected_split_assignment.csv
- selection metric: inner_validation_MAE
- deterministic tie-break: smallest alpha among MAE ties within tolerance 1e-12
- outer-test isolation: enforced, no outer-test outcomes accessed during selection

## 5. Inner Selection Results
Selection results file:
- revision_2026/04_controlled_reruns/ridge/ridge_inner_selection_results.csv

Selected configurations file:
- revision_2026/04_controlled_reruns/ridge/ridge_selected_configurations.csv

Selected alpha by horizon and fold:
- H1: fold1=10.0, fold2=100.0, fold3=100.0
- H3: fold1=0.01, fold2=100.0, fold3=100.0
- H5: fold1=100.0, fold2=100.0, fold3=100.0

All rows in outer_test_accessed_during_selection are FALSE.

## 6. H1 Results
Files:
- revision_2026/04_controlled_reruns/ridge/ridge_predictions.csv
- revision_2026/04_controlled_reruns/ridge/ridge_metrics_by_fold.csv
- revision_2026/04_controlled_reruns/ridge/ridge_metrics_pooled.csv

H1 pooled corrected metrics (N=762):
- MAE: 1.5586750869880055
- MSE: 3.83447502156652
- RMSE: 1.958181559908713
- MASE: 1.248899881704779

## 7. H3 Results
H3 pooled corrected metrics (N=747):
- MAE: 1.7975041211450118
- MSE: 5.084399311840097
- RMSE: 2.2548612622155044
- MASE: 1.4452793653103087

## 8. H5 Results
H5 pooled corrected metrics (N=747):
- MAE: 1.7731884418843413
- MSE: 5.115733635474376
- RMSE: 2.2617987610471397
- MASE: 1.4184002030396226

## 9. Event Prevalence and Date Equality
Event prevalence file:
- revision_2026/04_controlled_reruns/ridge/ridge_event_prevalence.csv

Checks performed:
- event counts are derived from Ridge y_true only
- Ridge horizon-fold date sets match corrected locked outer_test assignments exactly
- Ridge event counts match Persistence event counts on identical horizon/fold/date sets

Result: PASS.

## 10. Comparison with Submitted References
Submitted Ridge prediction-level artifact status:
- No released submitted Ridge prediction CSV for H1/H3/H5 was found.

Consequence:
- direct prediction-level comparison is impossible.

Traceable submitted Ridge summaries used for comparison:
- results/metrics/main_linear_metrics.json
- results/tables/main_linear_leaderboard.csv
- results/tables/main_linear_leaderboard_by_rmse.csv

Observed summary-level context:
- submitted leaderboard includes Ridge alpha variants and reports ridge_alpha_100.0 as best Ridge summary in that legacy context
- this legacy comparison is not a corrected purged nested per-fold inner-selection workflow and is therefore not used as corrected selection authority

## 11. Automated Test Results
Test file:
- revision_2026/04_controlled_reruns/ridge/test_corrected_ridge.py

Execution status:
- Ran 15 tests; failures=0; errors=0

Assertions covered:
1. dataset checksum lock
2. split checksum lock
3. exact canonical test-date equality
4. horizon/fold count equality
5. duplicate-key absence
6. target-date assignment equality
7. training-only preprocessing partition boundaries
8. outer-test isolation during alpha selection
9. selected alpha in confirmed candidate grid
10. metric recomputation equality
11. pooled RMSE equals sqrt(pooled MSE)
12. event-count threshold equality
13. Ridge versus Persistence event-count equality on identical date sets
14. repeated execution determinism across output checksums
15. repeated prediction checksum identity

Result: PASS.

## 12. Deviations or Blockers
No locked-input failures occurred.
No temporal-isolation test failures occurred.
No blocker prevented corrected Ridge regeneration under status B.

Documented limitation:
- no submitted Ridge prediction CSV exists for direct row-level comparison.

## 13. Readiness Decision
A. Ridge regeneration passed; ElasticNet nested revalidation may begin.
