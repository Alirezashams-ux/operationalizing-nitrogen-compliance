# Persistence Metric Reconciliation

## 1. Purpose
This report reconciles submitted Persistence metrics against the corrected canonical rerun using existing prediction files only. No prediction values, locked split files, or fold assignments were modified.

## 2. Submitted Metric Provenance
Submitted persistence-relevant sources and provenance checks are listed below.

| absolute_path | filename | generator | horizon | N | MAE | MSE | RMSE | MASE | date_range | fold_aggregation_method | pooled_or_fold_averaged | MSE/RMSE label check | confidence |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| /home/alrezshams/acs_tnout_ulsan_revision/results/metrics/main_linear_metrics_H1.json | main_linear_metrics_H1.json | src/run_linear_alarm.py:main -> point_metrics() + mase() | H1 | 762 | 1.5460530816726485 | 4.24474994612578 | 2.044486237973267 | 1.2315543026492117 | 2021-09-29..2023-10-30 | unweighted arithmetic mean over 3 fold metrics | fold-averaged | RMSE numerically correct as fold-mean RMSE; not pooled RMSE | High |
| /home/alrezshams/acs_tnout_ulsan_revision/results/metrics/main_linear_metrics_H3.json | main_linear_metrics_H3.json | src/run_linear_alarm.py:main -> point_metrics() + mase() | H3 | 759 | 2.0250068772608736 | 6.7293257705745395 | 2.578892839628263 | 1.6110314315369614 | 2021-09-30..2023-10-28 | unweighted arithmetic mean over 3 fold metrics | fold-averaged | RMSE numerically correct as fold-mean RMSE; not pooled RMSE | High |
| /home/alrezshams/acs_tnout_ulsan_revision/results/metrics/main_linear_metrics_H5.json | main_linear_metrics_H5.json | src/run_linear_alarm.py:main -> point_metrics() + mase() | H5 | 759 | 2.0774835591573804 | 7.086584237036678 | 2.6533337137266915 | 1.6664476520183547 | 2021-09-28..2023-10-26 | unweighted arithmetic mean over 3 fold metrics | fold-averaged | RMSE numerically correct as fold-mean RMSE; not pooled RMSE | High |
| /home/alrezshams/acs_tnout_ulsan_revision/results/metrics/main_linear_metrics.json | main_linear_metrics.json | src/train_main_linear.py:run -> eval_fold() + mase() | H1/H3/H5 variants | embedded per fold | contains | not explicit | contains | contains MASE1/MASE7 | per NPZ input | stores fold rows; aggregation external | fold rows | RMSE labels correct for per-fold values | Medium |
| /home/alrezshams/acs_tnout_ulsan_revision/results/final_tables/results_discussion_csvs/Table2_exceedance_prevalence_random_baseline.csv | Table2_exceedance_prevalence_random_baseline.csv | src/paper_make_uncertainty_decision_acs.py:build_exceedance_prevalence_table() | H1/H3/H5 | 762/759/759 | NA | NA | NA | NA | derived from persistence_H1/H3/H5 prediction files | none (row-wise prevalence over each horizon pooled set) | pooled | N/A (no MSE/RMSE labels) | High |
| /home/alrezshams/acs_tnout_ulsan_revision/results/paper_artifacts/20260303_164056_mainpaper_fig2_fig5_csv_handoff/Table2_exceedance_prevalence_random_baseline.csv | Table2_exceedance_prevalence_random_baseline.csv (handoff) | release handoff copy of Table2 prevalence output | H1/H3/H5 | 762/759/759 | NA | NA | NA | NA | same as source Table2 | none | pooled | N/A | High |
| /home/alrezshams/acs_tnout_ulsan_revision/results/final_tables/results_discussion_csvs/Table3_representative_operating_points_main.csv | Table3_representative_operating_points_main.csv | curated representative rows; audited by src/verify_claimed_values_reproducibility.py:verify_table3() | H1/H3/H5 | 762/759/747 in table rows | NA | NA | NA | NA | depends on referenced prediction files per row | global top-k over pooled horizon rows (k=ceil(r*N)) | pooled | N/A (no MSE/RMSE labels) | Medium-High |
| /home/alrezshams/acs_tnout_ulsan_revision/reports/claimed_values_reproducibility_audit_20260406.md | claimed_values_reproducibility_audit_20260406.md | src/verify_claimed_values_reproducibility.py | H1/H3/H5 | documents 762/759/759 prevalence and Table3 rows | documents best-model table values; not persistence MAE | NA | documents best-model table values; not persistence RMSE | documents best-model table values; not persistence MASE | inherits source artifacts | verification report (no new aggregation) | as source | as source | High |
| /home/alrezshams/acs_tnout_ulsan_revision/reports/final_submission_sanity_check_20260406.md | final_submission_sanity_check_20260406.md | src/check_submission_docx_sanity.py | H1/H3/H5 | confirms manuscript Table1 maps to Table2 prevalence CSV | confirms manuscript Table2 maps to point-summary CSV | NA | confirms point-summary CSV match | confirms point-summary CSV match | inherits source artifacts | verification report (no new aggregation) | as source | reveals table numbering swap in docx parsing flow | High |
| /home/alrezshams/acs_tnout_ulsan_revision/reports/_tmp_manuscript_docx_clean.txt | _tmp_manuscript_docx_clean.txt | docx text extraction artifact | H1/H3/H5 narrative mentions | textual prevalence ranges and persistence examples | textual mention for ENet; persistence MAE not tabulated | text definitions only | text definitions only | text definitions only | not explicit | narrative only | narrative only | N/A | Medium |
| /home/alrezshams/acs_tnout_ulsan_revision/reports/_tmp_manuscript_extracted_latest.txt | _tmp_manuscript_extracted_latest.txt | docx extraction artifact | H1/H3/H5 narrative mentions | textual prevalence ranges and persistence examples | textual mention for ENet; persistence MAE not tabulated | text definitions only | text definitions only | text definitions only | not explicit | narrative only | narrative only | N/A | Medium |

Exact manuscript and SI persistence values identified:
- Main manuscript prevalence baseline values are in Table2_exceedance_prevalence_random_baseline.csv (H1: N=762, H3/H5: N=759).
- Main manuscript representative operating points include persistence rows in Table3_representative_operating_points_main.csv.
- Point-forecast manuscript summary table does not report persistence as the best model; persistence point metrics are in main_linear_metrics_H*.json.
- SI split-count lineage reports canonical totals 762/747/747 in blocked_cv_split_summary_all_horizons.csv.

## 3. Metric Definitions
For each horizon, from corrected persistence predictions:
1. Pooled MAE = mean(|error|) over all held-out rows.
2. Pooled MSE = mean(error^2) over all held-out rows.
3. Pooled RMSE = sqrt(pooled MSE).
4. Unweighted mean fold MAE = arithmetic mean of fold MAE values.
5. Test-size-weighted mean fold MAE = weighted mean by fold N.
6. Unweighted mean fold MSE = arithmetic mean of fold MSE values.
7. Test-size-weighted mean fold MSE = weighted mean by fold N.
8. Unweighted mean fold RMSE = arithmetic mean of fold RMSE values.
9. Test-size-weighted mean fold RMSE = weighted mean by fold N.
10. sqrt(unweighted mean fold MSE).
11. sqrt(weighted mean fold MSE).
12. Submitted-script formula: unweighted mean over per-fold point_metrics in main_linear_metrics_H*.json.

RMSE identities used: pooled RMSE is not equivalent to mean fold RMSE.

## 4. H1 Reconciliation
- Submitted H1 RMSE: 2.044486237973267
- Corrected pooled H1 RMSE: 2.060279072329313
- Corrected fold RMSE values: fold1=1.775888487844843, fold2=1.971106448145625, fold3=2.386463705245336
- Aggregation reproducing submitted value: unweighted mean fold RMSE from main_linear_metrics_H1.json point_metrics.persistence.
- Absolute difference (submitted vs corrected pooled): 0.015792834356046.
- Submitted label correctness: RMSE value is correct for fold-mean RMSE, but not pooled RMSE; label should explicitly state fold-mean RMSE.
- Recommended revision: replace submitted H1 RMSE reference with canonical pooled RMSE in revision tables/claims that intend pooled held-out error.
- H1 MAE is numerically close because H1 test rows are identical (N=762 both lineages) and fold sizes are equal; tiny differences are floating-point representation only.
- H1 MASE differs mainly because corrected denominator uses purged outer-train rows, while submitted denominator uses non-purged TimeSeriesSplit training rows.

## 5. H3 Reconciliation
- Submitted H3 RMSE (fold-mean on N=759): 2.578892839628263
- Corrected H3 pooled RMSE (N=747): 2.598512676716014
- Aggregation effect at canonical N=747 (pooled RMSE - fold-mean RMSE): +0.013293874106372
- Canonical-date effect (fold-mean RMSE 747 - fold-mean RMSE 759): +0.006325993162443
- Total difference accounted for: +0.019619867268815
- Persistence implementation effect: none detected (overlapping-date y_pred values match to sub-micro precision).

## 6. H5 Reconciliation
- Submitted H5 RMSE (fold-mean on N=759): 2.653333713726691
- Corrected H5 pooled RMSE (N=747): 2.666127966440220
- Aggregation effect at canonical N=747 (pooled RMSE - fold-mean RMSE): +0.008677354401454
- Canonical-date effect (fold-mean RMSE 747 - fold-mean RMSE 759): +0.004116909980395
- Total difference accounted for: +0.012794264381849
- Persistence implementation effect: none detected (overlapping-date y_pred values match to sub-micro precision).

## 7. MASE Denominator Verification
Submitted MASE definition in source scripts: training-only m=1 naive denominator, computed per fold (mean absolute first-difference on y_train).
Corrected MASE definition: same formula, but training set is corrected outer-train with boundary purge applied.

| horizon | fold | submitted_train_denom_m1 | corrected_train_denom_m1 | submitted_train_scope | corrected_train_scope |
|---:|---:|---:|---:|---|---|
| 1 | 1 | 1.164308310555 | 1.160396825397 | non-v2 TimeSeriesSplit training rows only | corrected split outer_train rows only (purged) |
| 1 | 2 | 1.248560071992 | 1.249387351779 | non-v2 TimeSeriesSplit training rows only | corrected split outer_train rows only (purged) |
| 1 | 3 | 1.339894891785 | 1.339631578947 | non-v2 TimeSeriesSplit training rows only | corrected split outer_train rows only (purged) |
| 3 | 1 | 1.171338559197 | 1.153643724696 | non-v2 TimeSeriesSplit training rows only | corrected split outer_train rows only (purged) |
| 3 | 2 | 1.248619199799 | 1.257983870968 | non-v2 TimeSeriesSplit training rows only | corrected split outer_train rows only (purged) |
| 3 | 3 | 1.341631651925 | 1.343785234899 | non-v2 TimeSeriesSplit training rows only | corrected split outer_train rows only (purged) |
| 5 | 1 | 1.155793667840 | 1.160411522634 | non-v2 TimeSeriesSplit training rows only | corrected split outer_train rows only (purged) |
| 5 | 2 | 1.241168261574 | 1.260365853659 | non-v2 TimeSeriesSplit training rows only | corrected split outer_train rows only (purged) |
| 5 | 3 | 1.336912990616 | 1.344669365722 | non-v2 TimeSeriesSplit training rows only | corrected split outer_train rows only (purged) |

- H1 MASE delta decomposition: total +0.001104015956 = N/date effect -0.000000000567 + denominator effect +0.001104016524.
- H3 MASE delta decomposition: total +0.003973031091 = N/date effect +0.000614643114 + denominator effect +0.003358387977.
- H5 MASE delta decomposition: total -0.016519299149 = N/date effect -0.002830581985 + denominator effect -0.013688717164.
- Conclusion: corrected MASE implementation matches documented definition (training-only fold-specific denominator, m=1), with expected denominator changes from corrected split/training scope.

## 8. Effect of Corrected Test Dates
- H3 removed 12 noncanonical feature dates from legacy 759 set: 2021-09-30, 2021-10-01, 2021-10-02, 2021-10-03, 2021-10-04, 2021-10-05, 2021-10-06, 2021-10-07, 2021-10-08, 2021-10-09, 2021-10-10, 2021-10-11.
- H5 removed 12 noncanonical feature dates from legacy 759 set: 2021-09-28, 2021-09-29, 2021-09-30, 2021-10-01, 2021-10-02, 2021-10-03, 2021-10-04, 2021-10-05, 2021-10-06, 2021-10-07, 2021-10-08, 2021-10-09.
- H3 MAE shift is primarily date-set effect; RMSE shift is date-set effect plus pooled-vs-fold aggregation effect.
- H5 MAE shift is primarily date-set effect; RMSE shift is date-set effect plus pooled-vs-fold aggregation effect.

## 9. Submitted Labeling Issues
- Submitted persistence RMSE references in main_linear_metrics_H*.json are fold-mean RMSE values, not pooled RMSE values.
- If manuscript text intends pooled held-out RMSE, submitted labels are incomplete/misleading and should be revised to pooled RMSE values.
- MSE is not explicitly reported in submitted persistence metric JSON files; any MSE comparison there is implied from RMSE^2.
- check_submission_docx_sanity.py maps main DOCX Table 1 to prevalence CSV and Table 2 to point-forecast CSV, indicating a table-ordering/documentation quirk.

## 10. Recommended Revised Values
Recommended canonical persistence values (pooled over corrected held-out rows):

| horizon | N | MAE | MSE | RMSE | MASE |
|---:|---:|---:|---:|---:|---:|
| 1 | 762 | 1.546053081549 | 4.244749855878 | 2.060279072329 | 1.232658318606 |
| 3 | 747 | 2.025676317671 | 6.752268131054 | 2.598512676716 | 1.615004462628 |
| 5 | 747 | 2.074511378849 | 7.108238333435 | 2.666127966440 | 1.649928352870 |

## 11. Final Decision
Decision: A

Rationale: persistence metrics are fully reconciled. The H1 discrepancy is traced to fold-mean RMSE vs pooled RMSE aggregation, H3/H5 differences are decomposed into aggregation and canonical-date effects, MASE denominator behavior is fully traced and consistent with definitions, and no corrected prediction-row calculation error was found.