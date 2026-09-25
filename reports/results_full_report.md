# Comprehensive Results Report (for Results + Discussion)

This report synthesizes project outputs across point forecasting and alarm-budget evaluation, with emphasis on operational interpretation.

## 1) Evaluation scope
- Horizons: H=1, H=3, H=5 days.
- Point metrics: MAE, RMSE, MASE1, MASE7.
- Alarm operating points: tau in {15,16} mg/L and budget r in {0.05,0.10} (overall scope).

## 2) Point forecasting results (main linear runs)
- H1: best MAE model = enet_a0.1_l0.5 (MAE=1.5373, RMSE=1.9287, MASE1=1.2295, MASE7=0.8116).
- H3: best MAE model = enet_a0.1_l0.5 (MAE=1.7191, RMSE=2.1631, MASE1=1.3744, MASE7=0.9070).
- H5: best MAE model = enet_a1.0_l0.2 (MAE=1.7944, RMSE=2.2665, MASE1=1.4486, MASE7=0.9522).

### Interpretation
- ElasticNet variants remain competitive across all horizons.
- Error increases with horizon, consistent with reduced short-term predictability at longer lead times.
- In hybrid-v2 runs, the hybrid point blend further improved H5 point metrics relative to individual ENet/HGBR/Persistence in that run context.

## 3) Alarm-budget results by horizon (winner by recall)
- H1, tau=15, r=0.05: winner=persistence (precision=0.7692, recall=0.2128, TP=30/141, k=39).
- H1, tau=15, r=0.10: winner=persistence (precision=0.6494, recall=0.3546, TP=50/141, k=77).
- H1, tau=16, r=0.05: winner=persistence (precision=0.5128, recall=0.2703, TP=20/74, k=39).
- H1, tau=16, r=0.10: winner=persistence (precision=0.4286, recall=0.4459, TP=33/74, k=77).
- H3, tau=15, r=0.05: winner=enet_a1.0_l0.2 (precision=0.6316, recall=0.1714, TP=24/140, k=38).
- H3, tau=15, r=0.10: winner=enet_a0.01_l0.2 (precision=0.5132, recall=0.2786, TP=39/140, k=76).
- H3, tau=16, r=0.05: winner=enet_a1.0_l0.2 (precision=0.4211, recall=0.2192, TP=16/73, k=38).
- H3, tau=16, r=0.10: winner=enet_a0.01_l0.2 (precision=0.3026, recall=0.3151, TP=23/73, k=76).
- H5, tau=15, r=0.05: winner=enet_a0.1_l0.5 (precision=0.5263, recall=0.1429, TP=20/140, k=38).
- H5, tau=15, r=0.10: winner=persistence (precision=0.4474, recall=0.2429, TP=34/140, k=76).
- H5, tau=16, r=0.05: winner=persistence (precision=0.3684, recall=0.1918, TP=14/73, k=38).
- H5, tau=16, r=0.10: winner=persistence (precision=0.3026, recall=0.3151, TP=23/73, k=76).

### Interpretation
- Alarm winners are operating-point dependent; there is no universal winner across all tau/r combinations.
- This supports the paper argument that model selection changes when viewed through operationally constrained ranking.

## 4) H5 v2 paper benchmark vs guarded hybrid policy
- Paper v2 benchmark (`hybrid_rank_ens`) key rows:
  - tau=15, r=0.05: precision=0.5526, recall=0.1567, TP=21/134, k=38.
  - tau=15, r=0.10: precision=0.3600, recall=0.2015, TP=27/134, k=75.
  - tau=16, r=0.05: precision=0.4474, recall=0.2394, TP=17/71, k=38.
  - tau=16, r=0.10: precision=0.2800, recall=0.2958, TP=21/71, k=75.
- Guarded hybrid (`hybrid_rank_v2`) key rows:
  - tau=15, r=0.05: precision=0.5526, recall=0.1567, TP=21/134, k=38.
  - tau=15, r=0.10: precision=0.4400, recall=0.2463, TP=33/134, k=75.
  - tau=16, r=0.05: precision=0.4474, recall=0.2394, TP=17/71, k=38.
  - tau=16, r=0.10: precision=0.3467, recall=0.3662, TP=26/71, k=75.

### Delta vs paper hybrid_rank_ens (H5 v2)
- tau=15, r=0.05: Delta precision=+0.0000, Delta recall=+0.0000.
- tau=15, r=0.10: Delta precision=+0.0800, Delta recall=+0.0448.
- tau=16, r=0.05: Delta precision=+0.0000, Delta recall=+0.0000.
- tau=16, r=0.10: Delta precision=+0.0667, Delta recall=+0.0704.

## 5) Discussion-oriented conclusions
1. **Point forecasting and alarm ranking answer different questions**: lower MAE/RMSE does not guarantee best top-k event capture.
2. **Operational constraints change model preference**: at fixed budget, rank quality near the alert cutoff dominates utility.
3. **Guarded policy is practical**: conservative low-budget behavior is preserved while higher-budget operating points can improve materially.
4. **Methodological implication**: papers should report both point metrics and alarm-budget metrics to avoid misleading model claims in decision settings.

## 6) Limitations and what to report transparently
- Small daily dataset size implies fold-level variance; avoid over-claiming universal superiority.
- Some comparisons span different run groups (main-linear vs v2 hybrid artifacts); mark this explicitly in manuscript tables.
- Report exact file/version provenance for each comparison row.

## 7) Artifact references for discussion section
- Point summaries: `/home/alrezshams/acs_tnout_ulsan/results/metrics/main_linear_metrics_H1.json`, `/home/alrezshams/acs_tnout_ulsan/results/metrics/main_linear_metrics_H3.json`, `/home/alrezshams/acs_tnout_ulsan/results/metrics/main_linear_metrics_H5.json`
- Alarm summaries: `/home/alrezshams/acs_tnout_ulsan/results/metrics/alarm_budget_H1.csv`, `/home/alrezshams/acs_tnout_ulsan/results/metrics/alarm_budget_H3.csv`, `/home/alrezshams/acs_tnout_ulsan/results/metrics/alarm_budget_H5.csv`
- Paper H5 v2 table: `/home/alrezshams/acs_tnout_ulsan/results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/tables/table3_alarm_budget_H5_v2_long.csv`
- Guarded comparison: `/home/alrezshams/acs_tnout_ulsan/results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/tables/key_operating_points_compare_H5.csv`
- Discussion key-results CSV: `/home/alrezshams/acs_tnout_ulsan/results/tables/discussion_key_results.csv`