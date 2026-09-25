# Persistence Controlled Rerun Completion Report

## 1. Objective and Scope
This run regenerated only the Persistence baseline for H1, H3, and H5 under the corrected protocol, using:
- locked canonical dataset
- locked corrected split assignment
- no fitted models
- no edits under revision_2026/03_corrected_protocol/locked_protocol_v1/

## 2. Locked Inputs and Integrity Status
- dataset path: /home/alrezshams/acs_tnout_ulsan_revision/revision_2026/00_provenance/recovered_inputs/figure2/Ulsan_Yongsan.csv
- dataset SHA-256 (expected): 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb
- dataset SHA-256 (observed): 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb
- split file: revision_2026/03_corrected_protocol/corrected_split_assignment.csv
- split SHA-256 (observed): f83baf5732ffbf88b9a448a3d9ab9fa270f7495fe44d1559c1ac49795a30881a
- protocol checksum list: revision_2026/03_corrected_protocol/corrected_protocol_v1.sha256
- protocol checksum verification: PASS for all listed files

## 3. Execution Record
- required branch gate enforced in runner: controlled-reruns-v1
- git commit recorded in artifacts: 2bbf5eba419b0650693f1e88326107bf9a346c68
- run_id: persistence_corrected_2bbf5eba419b_6ed147cd_f83baf57
- commands executed:
  - python revision_2026/04_controlled_reruns/persistence/run_corrected_persistence.py
  - python revision_2026/04_controlled_reruns/persistence/test_corrected_persistence.py

## 4. Persistence Definition Used
Definition was implemented as carry-forward at horizon lag:
- y_hat(t+H) = TNout(t)
- y_true(t+H) = TNout(t+H)

Supporting evidence is documented in revision_2026/04_controlled_reruns/persistence/persistence_definition.md.

## 5. Produced Artifacts
Generated files:
- revision_2026/04_controlled_reruns/persistence/input_verification.json
- revision_2026/04_controlled_reruns/persistence/persistence_predictions.csv
- revision_2026/04_controlled_reruns/persistence/persistence_metrics_by_fold.csv
- revision_2026/04_controlled_reruns/persistence/persistence_metrics_pooled.csv
- revision_2026/04_controlled_reruns/persistence/persistence_event_prevalence.csv
- revision_2026/04_controlled_reruns/persistence/persistence_run_manifest.json

Checksums recorded in manifest:
- predictions: 383014421fa4419e7ba626a9a0e0df61f1b302431bb546c9049ce9f8a4b51754
- metrics_by_fold: 6c716094f371283a2f9b77701b1e60db738e54e4bbc188d99976cc2d10c7dcb1
- metrics_pooled: 72ced96eaab2effa6614105eb79ac2dc0e64b466a41eb6babdc68a0e90a26dcf
- event_prevalence: ff3991e212e39f77af511a260d4f220cb74f28cce49ffaaa36252d1b4adb3737

## 6. Row-Count and Boundary Consistency
Observed outer-test counts from corrected split:
- H1: 762 total (254 per fold)
- H3: 747 total (249 per fold)
- H5: 747 total (249 per fold)

All prediction rows matched corrected split assignments exactly by (horizon, fold, feature_date, target_date). No duplicates were found.

## 7. Test Coverage and Outcome
Test file: revision_2026/04_controlled_reruns/persistence/test_corrected_persistence.py

Assertions completed:
1. dataset checksum matches lock
2. split checksum matches approved protocol list
3. test dates equal canonical corrected test dates
4. counts match corrected assignments
5. no duplicate (horizon, fold, feature_date)
6. target dates equal assigned target dates
7. y_pred uses feature_date TNout only
8. y_true is not used to form y_pred
9. saved fold metrics equal recomputed metrics
10. pooled metrics computed from pooled predictions directly
11. event counts and prevalence match threshold logic
12. repeated execution is deterministic by file checksums

Result: PASS (12/12).

## 8. Corrected Persistence Results (Pooled)
| Horizon | N | MAE | MSE | RMSE | MASE |
|---|---:|---:|---:|---:|---:|
| 1 | 762 | 1.5460530815485565 | 4.244749855878133 | 2.0602790723293127 | 1.2326583186056685 |
| 3 | 747 | 2.025676317670683 | 6.752268131053822 | 2.598512676716014 | 1.6150044626281836 |
| 5 | 747 | 2.074511378848728 | 7.108238333434664 | 2.66612796644022 | 1.64992835286952 |

## 9. Comparison to Submitted Persistence Values
Submitted references used:
- results/metrics/main_linear_metrics_H1.json
- results/metrics/main_linear_metrics_H3.json
- results/metrics/main_linear_metrics_H5.json
- results/final_tables/results_discussion_csvs/Table2_exceedance_prevalence_random_baseline.csv

Point-metric comparison (submitted values are fold means in those JSON files):

| Horizon | Metric | Corrected | Submitted | Delta (Corrected - Submitted) |
|---|---|---:|---:|---:|
| 1 | N | 762 | 762 | 0 |
| 1 | MAE | 1.5460530815485565 | 1.5460530816726485 | -0.0000000001240920 |
| 1 | RMSE | 2.0602790723293127 | 2.044486237973267 | +0.0157928343560457 |
| 1 | MASE | 1.2326583186056685 | 1.2315543026492117 | +0.0011040159564568 |
| 3 | N | 747 | 759 | -12 |
| 3 | MAE | 2.025676317670683 | 2.0250068772608736 | +0.0006694404098094 |
| 3 | RMSE | 2.598512676716014 | 2.578892839628263 | +0.0196198370877510 |
| 3 | MASE | 1.6150044626281836 | 1.6110314315369614 | +0.0039730310912222 |
| 5 | N | 747 | 759 | -12 |
| 5 | MAE | 2.074511378848728 | 2.0774835591573804 | -0.0029721803086524 |
| 5 | RMSE | 2.66612796644022 | 2.6533337137266915 | +0.0127942527135285 |
| 5 | MASE | 1.64992835286952 | 1.6664476520183547 | -0.0165192991488347 |

Event prevalence comparison vs Table2:
- H1: exact match (N=762; events 141/74/30 for tau 15/16/17)
- H3: corrected N=747 and events 134/71/29 vs submitted N=759 and events 140/73/29
- H5: corrected N=747 and events 134/71/29 vs submitted N=759 and events 140/73/29

## 10. Interpretation of Differences
Differences are expected and attributable to corrected split assignments and evaluation aggregation specifics:
- corrected split has fewer outer-test rows for H3/H5 (747 vs 759)
- pooled RMSE in corrected rerun is computed from pooled rows directly
- submitted JSON references are fold-level outputs from legacy pipeline context

No evidence was found of leakage-style prediction construction in this rerun. Predictions are date-mapped strictly from feature_date TNout.

### 10.1 Metric-Reconciliation Subsection
Reconciliation details are fully documented in:
- revision_2026/04_controlled_reruns/persistence/persistence_metric_reconciliation.csv
- revision_2026/04_controlled_reruns/persistence/persistence_metric_reconciliation.md

Resolved findings:
- H1 submitted RMSE (2.044486237973267) is the unweighted mean of fold RMSE values from main_linear_metrics_H1.json, not pooled RMSE.
- Corrected H1 pooled RMSE (2.0602790723293127) is mathematically correct for pooled held-out rows.
- H3/H5 differences are decomposed into:
  - aggregation effect (pooled RMSE vs fold-mean RMSE), and
  - canonical-date effect from removing 12 noncanonical dates (759 -> 747).
- MASE differences are decomposed into:
  - canonical-date effect, and
  - denominator effect caused by corrected outer-train purged denominators.
- Persistence implementation itself is unchanged in practice (overlapping-date prediction values match to sub-micro precision).

## 11. Final Decision (A/B/C)
Decision: A

Rationale:
- All integrity gates and all 12 required tests passed.
- Persistence implementation matches the approved definition.
- Artifacts are deterministic and reproducible.
- Metric provenance is now fully reconciled, including H1 RMSE aggregation method and H3/H5 N-effect decomposition.
- No calculation error was found in corrected Persistence outputs.
