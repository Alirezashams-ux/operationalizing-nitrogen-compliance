# Corrected Retrospective H5 Alarm-Budget Evaluation

## 1. Purpose
- Perform corrected retrospective offline H5 top-k alarm-budget evaluation using frozen canonical base-model scores and frozen corrected fixed rank-ensemble scores.

## 2. Input Verification
- verification_pass: True
- canonical_assembly_id: canonical_h5_cross_model_6ed147cd_f83baf57_596f1d05ff47
- fixed_hybridrank_run_id: fixed_hybridrank_h5_ea738f9bca32_6ed147cd_f83baf57

## 3. Models and Risk Scores
- Models: Persistence, Ridge, ElasticNet, HGBR, BCR-TCN v1.1, HybridRank_fixed_documented.
- No point-blend result is retained.
- No tuned HybridRank result is retained.

## 4. Canonical Date and Event Equality
- All models are evaluated on the same 747 canonical keys and identical fold/date partitions.

## 5. Fold-Wise Alarm Budgets
- k is applied separately in each fold.
- H5 fold N is 249.
- k equals 13 and 25 at nominal budgets 0.05 and 0.10.
- pooled K equals 39 and 75.

## 6. Deterministic Top-k Rule
- Top-k sorting rule: descending score, ascending target_date, ascending feature_date, ascending canonical_row_id.
- Tie-audit deterministic pass rows: 108/108

## 7. Fold-Level Alarm Results
- Fold-level results are reported in retrospective_h5_alarm_metrics_by_fold.csv.

## 8. Pooled Alarm Results
- Pooled results are reported in retrospective_h5_alarm_metrics_pooled.csv.
- No model winner was selected automatically.

## 9. Exact Random Baseline
- exact random expected recall is K/N after ceiling.
- r=0.05 pooled exact random expected recall=0.0522088353
- r=0.1 pooled exact random expected recall=0.1004016064

## 10. Fixed Rank-Ensemble Results
- The fixed rank ensemble uses documented fixed weights from the corrected fixed-policy package.
- No corrected validation or outer-test outcome selected the weights.

## 11. Operating-Point Table Sources
- Full operating-point source: retrospective_h5_operating_point_table.csv.
- Compact prespecified table source: retrospective_h5_compact_table3_source.csv.

## 12. Fold Variability
- Descriptive fold variability is reported in retrospective_h5_fold_variability.csv; no narrow formal CI is claimed.

## 13. Label-Invariance Audit
- selection_label_invariance_pass: True

## 14. Retrospective-Only Interpretation
- Every result is labeled retrospective_offline_top_k.
- Complete held-out-fold ranking makes this retrospective only.
- No deployable sequential claim is made.
- A separate sequential simulation remains necessary.

## 15. Source Preservation and Determinism
- source_preservation_pass: True
- deterministic_result: pass

## 16. Limitations
1. This is an offline retrospective benchmark and not a deployable sequential policy.
2. Cutoffs are evaluated after complete held-out-fold ranking.
3. A separate sequential simulation remains required.

## 17. Readiness Decision
- run_id: retrospective_h5_alarm_budget_1a7fa526c263_6ed147cd_f83baf57

FINAL DECISION

<!-- AUTO_DECISION_START -->
A. Corrected retrospective H5 alarm-budget evaluation passed; sequential alarm-policy simulation may begin.
<!-- AUTO_DECISION_END -->

- test_result: pass_66_of_66
