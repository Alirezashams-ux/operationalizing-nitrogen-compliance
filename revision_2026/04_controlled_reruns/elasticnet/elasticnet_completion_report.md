# Corrected ElasticNet Baseline Completion Report

## 1. Input Verification
Input verification output: revision_2026/04_controlled_reruns/elasticnet/input_verification.json

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
  - 69cfb2bf8bc181dc1de1e09f0c7477a1a5de7585

Result: PASS.

## 2. Submitted ElasticNet Provenance
Provenance document: revision_2026/04_controlled_reruns/elasticnet/elasticnet_definition_and_provenance.md

Recovered ElasticNet definition highlights:
- model implementation: sklearn ElasticNet inside a fold-local StandardScaler pipeline
- candidate grid: 6 configs over alpha in {0.01, 0.1, 1.0} and l1_ratio in {0.2, 0.5}
- fit_intercept: True
- max_iter: 20000
- tol: 0.0001 (sklearn default in submitted environment)
- missing-value handling: submitted-style feature assembly with dropna before training; no imputer

Recovered provenance classification:
- B

Implication:
- a single fixed submitted ElasticNet configuration cannot be used as corrected selection authority due held-out-informed variant/context switching
- corrected rerun requires fold-local nested selection on corrected purged inner partitions

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
Selection plan: revision_2026/04_controlled_reruns/elasticnet/elasticnet_selection_plan.json

Applied strategy:
- status B workflow
- for each horizon and outer fold, evaluate every (alpha, l1_ratio) candidate using only:
  - inner_train rows labeled in corrected_split_assignment.csv with purged_inner_boundary excluded
  - inner_validation rows labeled in corrected_split_assignment.csv
- selection metric: inner_validation_MAE
- deterministic tie-break: smallest alpha, then smallest l1_ratio among MAE ties within tolerance 1e-12
- outer-test isolation: enforced, no outer-test outcomes accessed during selection

## 5. Inner Selection Results
Selection results file:
- revision_2026/04_controlled_reruns/elasticnet/elasticnet_inner_selection_results.csv

Selected configurations file:
- revision_2026/04_controlled_reruns/elasticnet/elasticnet_selected_configurations.csv

Selected config by horizon and fold:
- H1: fold1=(0.01,0.2), fold2=(0.1,0.5), fold3=(1.0,0.2)
- H3: fold1=(0.01,0.2), fold2=(1.0,0.2), fold3=(1.0,0.5)
- H5: fold1=(0.1,0.5), fold2=(1.0,0.2), fold3=(1.0,0.5)

All rows in outer_test_accessed_during_selection are FALSE.

## 6. H1 Results
Files:
- revision_2026/04_controlled_reruns/elasticnet/elasticnet_predictions.csv
- revision_2026/04_controlled_reruns/elasticnet/elasticnet_metrics_by_fold.csv
- revision_2026/04_controlled_reruns/elasticnet/elasticnet_metrics_pooled.csv

H1 pooled corrected metrics (N=762):
- MAE: 1.5824313701860826
- MSE: 3.942231671688746
- RMSE: 1.985505394525219
- MASE: 1.2672482350774492

## 7. H3 Results
H3 pooled corrected metrics (N=747):
- MAE: 1.7768710675654538
- MSE: 4.991493489210709
- RMSE: 2.234165054155737
- MASE: 1.4279953850177276

## 8. H5 Results
H5 pooled corrected metrics (N=747):
- MAE: 1.787277086046654
- MSE: 5.167935931896362
- RMSE: 2.2733094668118463
- MASE: 1.4314435970081079

## 9. Event Prevalence and Date Equality
Event prevalence file:
- revision_2026/04_controlled_reruns/elasticnet/elasticnet_event_prevalence.csv

Checks performed:
- event counts are derived from ElasticNet y_true only
- ElasticNet horizon-fold date sets match corrected locked outer_test assignments exactly
- ElasticNet event counts match Persistence event counts on identical horizon/fold/date sets

Result: PASS.

## 10. Comparison with Submitted References
Submitted reference contexts recovered:
- point-summary lineage (reports/final_model_summary.md):
  - H1 enet_a0.1_l0.5, MAE 1.5373
  - H3 enet_a0.1_l0.5, MAE 1.7191
  - H5 enet_a1.0_l0.2, MAE 1.7944
- representative-claim lineage (src/verify_claimed_values_reproducibility.py):
  - H5 generic enet mapped to enet_a0.1_l0.5_v2_H5_preds.csv

Corrected-versus-submitted MAE deltas:
- H1: 1.5824313701860826 - 1.5373 = +0.0451313701860825
- H3: 1.7768710675654538 - 1.7191 = +0.0577710675654537
- H5: 1.787277086046654 - 1.7944 = -0.0071229139533461

Interpretation:
- corrected H3/H5 use canonical N=747 date sets from corrected protocol (submitted non-v2 H3/H5 main-linear artifacts used 759)
- submitted H5 reporting lineage mixes v1 grid context and v2 fixed-context artifact; corrected rerun resolves this by applying one predeclared nested strategy under locked corrected splits

## 11. Run Manifest and Checksums
Manifest:
- revision_2026/04_controlled_reruns/elasticnet/elasticnet_run_manifest.json

Recorded values:
- run_id: elasticnet_corrected_69cfb2bf8bc1_6ed147cd_f83baf57
- prediction_sha256: dab8dd118ac95fbdef4bac3ca1b763be93320ea0dac1573a8b3b269a15960edb
- metrics_sha256:
  - elasticnet_inner_selection_results.csv: c70fc22d13dbd27d67086916c9da8d8ce67138c13054952a9d10b246d25158e1
  - elasticnet_selected_configurations.csv: b966065cd9831d52130a2947a43c6c0aa3d3fe99a28c4d6653ca24f97ba93349
  - elasticnet_metrics_by_fold.csv: 7ef0cba62a430da9e3d99f54f94041f98350ec96e42231389fa95deb32a82c90
  - elasticnet_metrics_pooled.csv: 048c02ccb30746c3cbdd0d7cbee92b630a0512b8c1ea7798f4c38d30df24fbe6
  - elasticnet_event_prevalence.csv: 64f1b42b94f82ee5952b42018a2691ec74f025945187de98caa27585832ef38f

## 12. Automated Test Results
Test file:
- revision_2026/04_controlled_reruns/elasticnet/test_corrected_elasticnet.py

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
8. outer-test isolation and deterministic tie-break during config selection
9. selected config in confirmed candidate grid
10. metric recomputation equality
11. pooled RMSE equals sqrt(pooled MSE)
12. event-count threshold equality
13. ElasticNet versus Persistence event-count equality on identical date sets
14. repeated execution determinism across output checksums
15. repeated prediction checksum identity

Result: PASS.

## 13. Deviations or Blockers
No locked-input failures occurred.
No temporal-isolation test failures occurred.
No blocker prevented corrected ElasticNet regeneration under status B.

Documented limitation:
- submitted ElasticNet lineage includes context switching (including H5 v1/v2), so direct one-to-one row-level replay against a single submitted configuration is not a valid corrected authority.

## 14. Readiness Decision
A. ElasticNet nested revalidation passed; the next controlled-rerun model family may begin.
