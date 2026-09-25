# Corrected Fixed H5 Rank Ensemble

## 1. Purpose
- Construct corrected fixed H5 rank-ensemble scores from frozen canonical corrected component predictions only.

## 2. Reason the Tuned HybridRank Pathway Was Not Retained
- The tuned submitted pathway is not retained.
- The fully nested tuned reconstruction stopped with Decision C because Fold 1 component validation had zero tau = 16 events.
- The fixed documented vector was retained unchanged for this corrected fixed-policy stage.

## 3. Input Verification
- run_id: fixed_hybridrank_h5_ea738f9bca32_6ed147cd_f83baf57
- git_commit: ea738f9bca32b3bf5648a860c35cd84fbe4e205b
- canonical_assembly_id: canonical_h5_cross_model_6ed147cd_f83baf57_596f1d05ff47
- canonical_N: 747
- fully_nested_feasibility_decision: C
- input_verification_pass: True

## 4. Canonical Corrected Component Scores
- Score construction used only canonical_h5_predictions_wide.csv as corrected numerical input.
- Submitted predictions were inspected for provenance only and were not used as corrected model inputs.

## 5. Fixed Documented Weight Vector
- policy_name: HybridRank_fixed_documented
- submitted_alias: hybrid_paper_fixed
- BCR-TCN v1.1 = 0.50
- ElasticNet = 0.25
- Persistence = 0.25
- HGBR = 0.00
- No corrected validation or outer-test outcome selected these weights.

## 6. Threshold-Specific Component Inputs
- tau 15 uses bcr_tcn_v11_p_tau15
- tau 16 uses bcr_tcn_v11_p_tau16
- tau 17 uses bcr_tcn_v11_p_tau17
- ElasticNet, Persistence, and HGBR use their canonical point predictions for all thresholds.

## 7. Fold-Wise Rank Normalization
- Rank normalization is independent per outer fold, threshold, and component.
- Ascending average ties and N-1 denominator were used.
- Rank audit pass rate: 36/36 rows.

## 8. Corrected Fixed Hybrid Scores
- Output rows: 2241
- The fixed score is budget-independent.
- The same fixed score may later be evaluated at r = 0.05 and r = 0.10.

## 9. Label-Invariance Audit
- rank_values_identical: True
- hybrid_scores_identical: True
- maximum_absolute_score_difference: 0.0
- label_invariance_pass: True

## 10. Test-Outcome and Future-Fold Isolation
- No validation or outer-test outcomes were used for weight selection.
- No future-fold outcomes were used.
- Source-preservation pass: True

## 11. Difference from the Submitted Tuned Pathway
- The tuned submitted pathway is not retained in this stage.
- Evaluation at r = 0.10 is treated as corrected fixed-policy analysis, not the submitted tuned guarded policy.

## 12. Retrospective-Only Interpretation
- Within-test-fold rank normalization makes this score retrospective and non-deployable.
- A separate sequential policy will be evaluated later.

## 13. Determinism and Automated Tests
- Automated tests are provided in test_fixed_hybridrank_h5_scores.py.
- Determinism is validated by repeated generation in the test stage.

## 14. Limitations
1. The tuned submitted pathway is not retained after corrected chronology review.
2. This stage is score-construction only; no alarm metrics were calculated in this stage.
3. Fixed scores are retrospective and not a deployable sequential score.

## 15. Readiness Decision
- run_id: fixed_hybridrank_h5_ea738f9bca32_6ed147cd_f83baf57

FINAL DECISION

<!-- AUTO_DECISION_START -->
A. Corrected fixed H5 rank-ensemble scores passed; retrospective alarm-budget evaluation may begin.
<!-- AUTO_DECISION_END -->

