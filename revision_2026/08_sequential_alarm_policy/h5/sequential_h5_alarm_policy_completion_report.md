# Corrected Sequential H5 Alarm-Policy Simulation

## 1. Purpose
- Execute a strictly chronological historical simulation of a deployable H5 alarm-decision rule.

## 2. Input Verification
- verification_pass: True
- canonical_assembly_id: canonical_h5_cross_model_6ed147cd_f83baf57_596f1d05ff47
- fixed_hybridrank_run_id: fixed_hybridrank_h5_ea738f9bca32_6ed147cd_f83baf57
- retrospective_alarm_run_id: retrospective_h5_alarm_budget_1a7fa526c263_6ed147cd_f83baf57

## 3. Difference from Retrospective Top-k
- The retrospective benchmark ranks complete held-out folds and enforces exact foldwise k.
- This sequential simulation processes one date at a time and never forces exact alarm counts.

## 4. Authorized Models and Scores
- Models: Persistence, Ridge, ElasticNet, HGBR, BCR-TCN v1.1, HybridRank_fixed_documented.
- No point-blend result is included.
- No tuned HybridRank result is included.

## 5. Startup Calibration
- startup_days: 30
- First 30 dates in each fold are startup calibration with no alarms.
- Primary performance excludes startup dates.
- fold 1: total_N=249 startup_N=30 eligible_N=219 startup=2021-10-15..2021-11-13 eligible=2021-11-14..2022-06-20
- fold 2: total_N=249 startup_N=30 eligible_N=219 startup=2022-06-21..2022-07-20 eligible=2022-07-21..2023-02-24
- fold 3: total_N=249 startup_N=30 eligible_N=219 startup=2023-02-25..2023-03-26 eligible=2023-03-27..2023-10-31

## 6. Causal Fixed-Ensemble Score
- Causal ranks use only prior scores within fold and threshold.
- Weights: BCR-TCN 0.50, ElasticNet 0.25, Persistence 0.25, HGBR 0.00.
- The sequential ensemble differs from retrospective full-fold rank normalization.

## 7. Past-Only Cutoff Rule
- Cutoff uses the higher empirical quantile of strictly prior policy scores.
- Equality rule is conservative: alarm only when score > cutoff.
- No future score or label enters a decision.

## 8. Sequential Alarm Decisions
- Decisions are labeled historical_sequential_past_only.
- Simulation processes one date at a time.

## 9. Realized Alarm Burden
- Realized burden is observed output and not forced to equal nominal capacity.
- model=BCR-TCN v1.1 tau=15 r=0.05 mean_realized_alarm_fraction=0.159817 mean_abs_nominal_diff=0.109817
- model=BCR-TCN v1.1 tau=15 r=0.1 mean_realized_alarm_fraction=0.223744 mean_abs_nominal_diff=0.129528
- model=BCR-TCN v1.1 tau=16 r=0.05 mean_realized_alarm_fraction=0.132420 mean_abs_nominal_diff=0.106621
- model=BCR-TCN v1.1 tau=16 r=0.1 mean_realized_alarm_fraction=0.193303 mean_abs_nominal_diff=0.129528
- model=BCR-TCN v1.1 tau=17 r=0.05 mean_realized_alarm_fraction=0.086758 mean_abs_nominal_diff=0.051826
- model=BCR-TCN v1.1 tau=17 r=0.1 mean_realized_alarm_fraction=0.144597 mean_abs_nominal_diff=0.068645
- model=ElasticNet tau=15 r=0.05 mean_realized_alarm_fraction=0.108067 mean_abs_nominal_diff=0.091400
- model=ElasticNet tau=15 r=0.1 mean_realized_alarm_fraction=0.159817 mean_abs_nominal_diff=0.126484
- model=ElasticNet tau=16 r=0.05 mean_realized_alarm_fraction=0.108067 mean_abs_nominal_diff=0.091400
- model=ElasticNet tau=16 r=0.1 mean_realized_alarm_fraction=0.159817 mean_abs_nominal_diff=0.126484
- model=ElasticNet tau=17 r=0.05 mean_realized_alarm_fraction=0.108067 mean_abs_nominal_diff=0.091400
- model=ElasticNet tau=17 r=0.1 mean_realized_alarm_fraction=0.159817 mean_abs_nominal_diff=0.126484
- model=HGBR tau=15 r=0.05 mean_realized_alarm_fraction=0.073059 mean_abs_nominal_diff=0.050304
- model=HGBR tau=15 r=0.1 mean_realized_alarm_fraction=0.109589 mean_abs_nominal_diff=0.070167
- model=HGBR tau=16 r=0.05 mean_realized_alarm_fraction=0.073059 mean_abs_nominal_diff=0.050304
- model=HGBR tau=16 r=0.1 mean_realized_alarm_fraction=0.109589 mean_abs_nominal_diff=0.070167
- model=HGBR tau=17 r=0.05 mean_realized_alarm_fraction=0.073059 mean_abs_nominal_diff=0.050304
- model=HGBR tau=17 r=0.1 mean_realized_alarm_fraction=0.109589 mean_abs_nominal_diff=0.070167
- model=HybridRank_fixed_documented tau=15 r=0.05 mean_realized_alarm_fraction=0.095890 mean_abs_nominal_diff=0.067047
- model=HybridRank_fixed_documented tau=15 r=0.1 mean_realized_alarm_fraction=0.135464 mean_abs_nominal_diff=0.077778
- model=HybridRank_fixed_documented tau=16 r=0.05 mean_realized_alarm_fraction=0.088280 mean_abs_nominal_diff=0.062481
- model=HybridRank_fixed_documented tau=16 r=0.1 mean_realized_alarm_fraction=0.143075 mean_abs_nominal_diff=0.070167
- model=HybridRank_fixed_documented tau=17 r=0.05 mean_realized_alarm_fraction=0.092846 mean_abs_nominal_diff=0.054871
- model=HybridRank_fixed_documented tau=17 r=0.1 mean_realized_alarm_fraction=0.147641 mean_abs_nominal_diff=0.077778
- model=Persistence tau=15 r=0.05 mean_realized_alarm_fraction=0.082192 mean_abs_nominal_diff=0.041172
- model=Persistence tau=15 r=0.1 mean_realized_alarm_fraction=0.155251 mean_abs_nominal_diff=0.067123
- model=Persistence tau=16 r=0.05 mean_realized_alarm_fraction=0.082192 mean_abs_nominal_diff=0.041172
- model=Persistence tau=16 r=0.1 mean_realized_alarm_fraction=0.155251 mean_abs_nominal_diff=0.067123
- model=Persistence tau=17 r=0.05 mean_realized_alarm_fraction=0.082192 mean_abs_nominal_diff=0.041172
- model=Persistence tau=17 r=0.1 mean_realized_alarm_fraction=0.155251 mean_abs_nominal_diff=0.067123
- model=Ridge tau=15 r=0.05 mean_realized_alarm_fraction=0.109589 mean_abs_nominal_diff=0.092922
- model=Ridge tau=15 r=0.1 mean_realized_alarm_fraction=0.165906 mean_abs_nominal_diff=0.132572
- model=Ridge tau=16 r=0.05 mean_realized_alarm_fraction=0.109589 mean_abs_nominal_diff=0.092922
- model=Ridge tau=16 r=0.1 mean_realized_alarm_fraction=0.165906 mean_abs_nominal_diff=0.132572
- model=Ridge tau=17 r=0.05 mean_realized_alarm_fraction=0.109589 mean_abs_nominal_diff=0.092922
- model=Ridge tau=17 r=0.1 mean_realized_alarm_fraction=0.165906 mean_abs_nominal_diff=0.132572

## 10. Post-Startup Performance
- Primary post-startup pooled metrics are reported in sequential_h5_metrics_pooled.csv.
- model=BCR-TCN v1.1 tau=15 r=0.05 pooled_precision=0.24761904761904763 pooled_recall=0.2222222222222222 pooled_alarms=105
- model=BCR-TCN v1.1 tau=15 r=0.1 pooled_precision=0.23809523809523808 pooled_recall=0.29914529914529914 pooled_alarms=147
- model=BCR-TCN v1.1 tau=16 r=0.05 pooled_precision=0.08045977011494253 pooled_recall=0.1044776119402985 pooled_alarms=87
- model=BCR-TCN v1.1 tau=16 r=0.1 pooled_precision=0.11023622047244094 pooled_recall=0.208955223880597 pooled_alarms=127
- model=BCR-TCN v1.1 tau=17 r=0.05 pooled_precision=0.12280701754385964 pooled_recall=0.2692307692307692 pooled_alarms=57
- model=BCR-TCN v1.1 tau=17 r=0.1 pooled_precision=0.09473684210526316 pooled_recall=0.34615384615384615 pooled_alarms=95
- model=ElasticNet tau=15 r=0.05 pooled_precision=0.352112676056338 pooled_recall=0.21367521367521367 pooled_alarms=71
- model=ElasticNet tau=15 r=0.1 pooled_precision=0.29523809523809524 pooled_recall=0.26495726495726496 pooled_alarms=105
- model=ElasticNet tau=16 r=0.05 pooled_precision=0.23943661971830985 pooled_recall=0.2537313432835821 pooled_alarms=71
- model=ElasticNet tau=16 r=0.1 pooled_precision=0.19047619047619047 pooled_recall=0.29850746268656714 pooled_alarms=105
- model=ElasticNet tau=17 r=0.05 pooled_precision=0.11267605633802817 pooled_recall=0.3076923076923077 pooled_alarms=71
- model=ElasticNet tau=17 r=0.1 pooled_precision=0.09523809523809523 pooled_recall=0.38461538461538464 pooled_alarms=105
- model=HGBR tau=15 r=0.05 pooled_precision=0.3958333333333333 pooled_recall=0.1623931623931624 pooled_alarms=48
- model=HGBR tau=15 r=0.1 pooled_precision=0.3611111111111111 pooled_recall=0.2222222222222222 pooled_alarms=72
- model=HGBR tau=16 r=0.05 pooled_precision=0.2708333333333333 pooled_recall=0.19402985074626866 pooled_alarms=48
- model=HGBR tau=16 r=0.1 pooled_precision=0.2222222222222222 pooled_recall=0.23880597014925373 pooled_alarms=72
- model=HGBR tau=17 r=0.05 pooled_precision=0.10416666666666667 pooled_recall=0.19230769230769232 pooled_alarms=48
- model=HGBR tau=17 r=0.1 pooled_precision=0.08333333333333333 pooled_recall=0.23076923076923078 pooled_alarms=72
- model=HybridRank_fixed_documented tau=15 r=0.05 pooled_precision=0.36507936507936506 pooled_recall=0.19658119658119658 pooled_alarms=63
- model=HybridRank_fixed_documented tau=15 r=0.1 pooled_precision=0.34831460674157305 pooled_recall=0.26495726495726496 pooled_alarms=89
- model=HybridRank_fixed_documented tau=16 r=0.05 pooled_precision=0.3275862068965517 pooled_recall=0.2835820895522388 pooled_alarms=58
- model=HybridRank_fixed_documented tau=16 r=0.1 pooled_precision=0.26595744680851063 pooled_recall=0.373134328358209 pooled_alarms=94
- model=HybridRank_fixed_documented tau=17 r=0.05 pooled_precision=0.13114754098360656 pooled_recall=0.3076923076923077 pooled_alarms=61
- model=HybridRank_fixed_documented tau=17 r=0.1 pooled_precision=0.09278350515463918 pooled_recall=0.34615384615384615 pooled_alarms=97
- model=Persistence tau=15 r=0.05 pooled_precision=0.25925925925925924 pooled_recall=0.11965811965811966 pooled_alarms=54
- model=Persistence tau=15 r=0.1 pooled_precision=0.29411764705882354 pooled_recall=0.2564102564102564 pooled_alarms=102
- model=Persistence tau=16 r=0.05 pooled_precision=0.16666666666666666 pooled_recall=0.13432835820895522 pooled_alarms=54
- model=Persistence tau=16 r=0.1 pooled_precision=0.21568627450980393 pooled_recall=0.3283582089552239 pooled_alarms=102
- model=Persistence tau=17 r=0.05 pooled_precision=0.05555555555555555 pooled_recall=0.11538461538461539 pooled_alarms=54
- model=Persistence tau=17 r=0.1 pooled_precision=0.08823529411764706 pooled_recall=0.34615384615384615 pooled_alarms=102
- model=Ridge tau=15 r=0.05 pooled_precision=0.3472222222222222 pooled_recall=0.21367521367521367 pooled_alarms=72
- model=Ridge tau=15 r=0.1 pooled_precision=0.3119266055045872 pooled_recall=0.2905982905982906 pooled_alarms=109
- model=Ridge tau=16 r=0.05 pooled_precision=0.2638888888888889 pooled_recall=0.2835820895522388 pooled_alarms=72
- model=Ridge tau=16 r=0.1 pooled_precision=0.2018348623853211 pooled_recall=0.3283582089552239 pooled_alarms=109
- model=Ridge tau=17 r=0.05 pooled_precision=0.08333333333333333 pooled_recall=0.23076923076923078 pooled_alarms=72
- model=Ridge tau=17 r=0.1 pooled_precision=0.08256880733944955 pooled_recall=0.34615384615384615 pooled_alarms=109

## 11. Full-Fold Secondary Performance
- Secondary full-fold metrics treat startup dates as no-alarm dates and never replace the primary post-startup metrics.
- Label used: secondary_full_fold_with_startup_no_alarms.

## 12. Label and Future-Information Isolation
- label_invariance_pass: True
- future_score_invariance_pass: True

## 13. Prefix Causality
- prefix_causality_pass: True
- maximum_earlier_decision_difference: 0.0

## 14. Source Preservation and Determinism
- source_preservation_pass: True
- deterministic_result: pass

## 15. Operational Interpretation
- Results represent a historical simulation, not prospective field validation.
- A future 3-6-month shadow deployment remains necessary.
- No model or policy winner was automatically selected.

## 16. Limitations
1. This is not prospective field validation or real-time deployment.
2. Primary performance excludes startup dates by design.
3. Realized burden is unconstrained and may deviate from nominal r.
4. No point-blend or tuned HybridRank result is included.

## 17. Readiness Decision
- run_id: sequential_h5_alarm_policy_140b8193f314_6ed147cd_f83baf57

FINAL DECISION

A. Corrected sequential H5 alarm-policy simulation passed; H1/H3 canonical reconciliation may begin.

- test_result: pass_65_of_65
