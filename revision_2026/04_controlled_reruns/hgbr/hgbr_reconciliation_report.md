# HGBR Result Reconciliation

## 1. Purpose
This report resolves why HGBR was initially assigned decision B and determines whether the corrected HGBR artifacts can be upgraded to decision A without retraining or post-hoc model selection.

Scope constraints applied in this reconciliation:
- No model retraining was performed.
- No HGBR predictions or selected configurations were modified.
- No other model family was executed.
- All conclusions are based on existing artifacts only.

## 2. Reason for Initial Decision B
The original decision B was driven by an unresolved submitted-provenance discrepancy documented in the completion report:
- Submitted H3/H5 hyperparameter summary values in deliverables/si_upload_packages_20260329/generated/final_chosen_hyperparameters_by_horizon_model.csv did not match values stored in results/hgbr/hgbr_optuna_H3.json and results/hgbr/hgbr_optuna_H5.json.

The issue registry in revision_2026/04_controlled_reruns/hgbr/hgbr_decision_b_issue_registry.csv now classifies this as a reporting provenance conflict, not a corrected-run calculation error.

## 3. Submitted HGBR Lineages
Submitted and manuscript-linked HGBR lineages are compiled in revision_2026/04_controlled_reruns/hgbr/hgbr_submitted_value_registry.csv.

Recovered lineages:
- H1 legacy prediction artifact: missing (no direct submitted prediction-level comparison possible).
- H3 ordinary prediction lineage: results/predictions/hgbr_optuna_H3_preds.csv, N=759.
- H5 ordinary prediction lineage: results/predictions/hgbr_optuna_H5_preds.csv, N=759.
- H5 v2 prediction lineage: results/predictions/hgbr_optuna_H5_v2_preds.csv, N=747.
- H5 shared-date manuscript-linked point summary: results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/tables/point_metrics_tnout_H5.csv, N=747.
- H5 normalized exploratory shared-date summary: results/hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2/tables/point_metrics_tnout_H5.csv, N=735.
- Manuscript table row: results/final_tables/results_discussion_csvs/Table1_leakage_safe_point_forecasting.csv (hgbr H5 entry, source-file-linked to the 747 shared-date run).

Aggregation contexts found in submitted ecosystem:
- Optuna JSON best_mae fields are inner-validation objective summaries.
- Prediction CSV metrics are pooled-over-rows outer-test metrics.
- Manuscript-linked H5 point metrics are pooled-overall shared-date summaries.

## 4. Metric Aggregation Reconciliation
Aggregation formulas were recalculated directly from submitted prediction rows for traceable horizons:

For each dataset, we computed:
- pooled MAE, pooled MSE, pooled RMSE;
- unweighted and weighted means of fold MAE;
- unweighted and weighted means of fold RMSE;
- sqrt(unweighted mean fold MSE) and sqrt(weighted mean fold MSE).

Findings:
- Submitted prediction-based comparisons are pooled metrics, not mean-fold RMSE primary metrics.
- Manuscript-linked H5 row values align with pooled overall metrics from the referenced shared-date source file.
- There is no evidence that manuscript H5 point metrics are reported as mean-fold RMSE.

## 5. H1 Reconciliation
H1 submitted prediction artifact is missing. This prevents direct submitted-versus-corrected prediction-level decomposition.

What can still be concluded:
- H1 candidate space is recoverable from legacy training scripts and was predeclared before corrected outer-test generation.
- Corrected H1 run uses fold-local nested selection with strict outer-test isolation.
- Missing legacy H1 prediction file is a reporting limitation, not a corrected-run validity failure.

Corrected H1 canonical pooled values remain:
- N=762
- MAE=1.6649183736062991
- MSE=4.351177237744344
- RMSE=2.08594756351744
- MASE=1.331702333745425

## 6. H3 Reconciliation
Submitted traceable baseline used for decomposition:
- H3 ordinary prediction file (N=759): MAE=1.7767021126165674, RMSE=2.254932564350989.

Canonical-date restricted H3 ordinary subset (N=747):
- MAE=1.7786895786681223
- RMSE=2.259191317068711

Corrected H3 canonical pooled metrics (N=747):
- MAE=1.763640238285536
- RMSE=2.256657700039158

Decomposition (corrected minus submitted-ordinary-native):
- Date-set effect: MAE +0.0019874660515548648, RMSE +0.004258752717722025
- Nested-selection effect: MAE -0.015049340382586207, RMSE -0.0025336170295533478
- Aggregation effect: 0.0 (same pooled basis)
- Lineage effect: 0.0 (ordinary lineage reference)
- Residual/unresolved effect: 0.0

## 7. H5 and v2 Lineage Reconciliation
Submitted traceable baselines:
- H5 ordinary native (N=759): MAE=1.754919394394562, RMSE=2.2888387104134282
- H5 ordinary restricted to canonical 747 dates: MAE=1.7610860363377878, RMSE=2.2971710581186544
- H5 v2 (N=747): MAE=1.7954476951540996, RMSE=2.2905010637462424

Corrected H5 canonical pooled metrics (N=747):
- MAE=1.8071915266682184
- RMSE=2.288659983853924

Decomposition from submitted H5 ordinary native to corrected H5:
- Date-set effect (759 to canonical 747): MAE +0.0061666419432258035, RMSE +0.008332347705226173
- Lineage effect (ordinary 747 to v2 747): MAE +0.03436165881631181, RMSE -0.0066699943724120025
- Nested-selection effect (submitted v2 to corrected): MAE +0.01174383151411873, RMSE -0.001841079892318298
- Aggregation effect: 0.0 (same pooled basis)
- Residual/unresolved effect: 0.0

H5 ordinary-versus-v2 conclusion:
- Corrected H5 use of v2 candidate space is valid because v2 authorization was predeclared in revision_2026/04_controlled_reruns/hgbr/hgbr_selection_plan.json before corrected outer-test generation.
- Evaluating alternative H5 candidate spaces post hoc on corrected outer-test results would be impermissible retrospective model selection.

## 8. Effects of Purging and Canonical Dates
Purging and canonical-date effects are reflected in the 759-to-747 transitions for H3/H5 ordinary lineages.

Operationally in available artifacts:
- Purge-related row exclusions and canonical-date alignment are represented by restricting submitted ordinary prediction rows to corrected canonical date sets.
- Separate standalone numeric attribution for "outer purge only" versus "inner purge only" is not identifiable from submitted prediction files alone.

Conclusion:
- Date/purge regime differences are expected protocol effects and are quantified where prediction artifacts permit.

## 9. Effects of Nested Selection
Nested fold-local selection effects were quantified by holding date sets/lineages fixed where possible:
- H3: corrected nested selection improves MAE relative to submitted ordinary-on-canonical subset.
- H5: corrected nested selection is slightly worse in MAE than submitted H5 v2, with slightly lower RMSE.

These differences are expected consequences of corrected leakage-safe selection and do not imply a corrected-run implementation error.

## 10. Comparison with Other Corrected Models
Canonical pooled descriptive metrics:

| Horizon | Model | N | MAE | MSE | RMSE | MASE |
|---|---|---:|---:|---:|---:|---:|
| H1 | Persistence | 762 | 1.5460530815485565 | 4.244749855878133 | 2.0602790723293127 | 1.2326583186056685 |
| H1 | Ridge | 762 | 1.5586750869880055 | 3.83447502156652 | 1.958181559908713 | 1.248899881704779 |
| H1 | ElasticNet | 762 | 1.5824313701860826 | 3.942231671688746 | 1.985505394525219 | 1.2672482350774492 |
| H1 | HGBR | 762 | 1.6649183736062991 | 4.351177237744344 | 2.08594756351744 | 1.331702333745425 |
| H3 | Persistence | 747 | 2.025676317670683 | 6.752268131053822 | 2.598512676716014 | 1.6150044626281836 |
| H3 | Ridge | 747 | 1.7975041211450118 | 5.084399311840097 | 2.2548612622155044 | 1.4452793653103087 |
| H3 | ElasticNet | 747 | 1.7768710675654538 | 4.991493489210709 | 2.234165054155737 | 1.4279953850177276 |
| H3 | HGBR | 747 | 1.763640238285536 | 5.0925039751460215 | 2.256657700039158 | 1.410724330361394 |
| H5 | Persistence | 747 | 2.074511378848728 | 7.108238333434664 | 2.66612796644022 | 1.64992835286952 |
| H5 | Ridge | 747 | 1.7731884418843413 | 5.115733635474376 | 2.2617987610471397 | 1.4184002030396226 |
| H5 | ElasticNet | 747 | 1.787277086046654 | 5.167935931896362 | 2.2733094668118463 | 1.4314435970081079 |
| H5 | HGBR | 747 | 1.8071915266682184 | 5.237964521694242 | 2.288659983853924 | 1.444501350881476 |

These outer-test comparisons are descriptive only and cannot be used to retroactively change predeclared HGBR candidate spaces.

## 11. Remaining Limitations
Remaining limitations are reporting/provenance limitations, not corrected-run validity failures:
- Missing submitted H1 prediction artifact prevents direct prediction-level back-comparison.
- Submitted H3/H5 hyperparameter summary table values conflict with referenced JSON files.
- Exact standalone numeric split between outer-purge-only and inner-purge-only effects is not recoverable from submitted prediction artifacts.

No unresolved validity-critical issue remains in the corrected HGBR artifacts.

## 12. Recommended Revised HGBR Values
Recommended revised values are the canonical corrected pooled metrics in revision_2026/04_controlled_reruns/hgbr/hgbr_metrics_pooled.csv:

- H1: N=762, MAE=1.6649183736062991, MSE=4.351177237744344, RMSE=2.08594756351744, MASE=1.331702333745425
- H3: N=747, MAE=1.763640238285536, MSE=5.0925039751460215, RMSE=2.256657700039158, MASE=1.410724330361394
- H5: N=747, MAE=1.8071915266682184, MSE=5.237964521694242, RMSE=2.288659983853924, MASE=1.444501350881476

## 13. Final Decision
Final decision: A.

Rationale:
- Corrected inputs, checksums, and temporal isolation already passed.
- Candidate spaces by horizon are defensibly documented and predeclared.
- Observed discrepancies are explained by date-set regime, lineage, aggregation context, and nested-selection differences.
- Missing H1 legacy predictions are documented but do not invalidate corrected H1 provenance.
- No corrected-run calculation error or unresolved validity-critical issue remains.

Therefore, HGBR nested revalidation passed and the completion report can be upgraded to decision A.