# Model Performance Report (Project-wide)

- Total unique models: 15
- Point metrics table: results/tables/model_point_performance_full.csv
- Alarm metrics table: results/tables/model_alarm_performance_full.csv

## Model List (complete names)
- bcr_tcn_v11: BCR-TCN v1.1
- enet: ElasticNet (generic)
- enet_a0.01_l0.2: ElasticNet (alpha=0.01, l1_ratio=0.2)
- enet_a0.01_l0.5: ElasticNet (alpha=0.01, l1_ratio=0.5)
- enet_a0.1_l0.2: ElasticNet (alpha=0.1, l1_ratio=0.2)
- enet_a0.1_l0.5: ElasticNet (alpha=0.1, l1_ratio=0.5)
- enet_a1.0_l0.2: ElasticNet (alpha=1.0, l1_ratio=0.2)
- enet_a1.0_l0.5: ElasticNet (alpha=1.0, l1_ratio=0.5)
- hgbr: Histogram Gradient Boosting Regressor (Optuna-tuned)
- hybrid_paper_fixed: Hybrid rank ensemble (paper-fixed weights)
- hybrid_point: Hybrid point predictor (weighted regression blend)
- hybrid_rank_ens: Hybrid rank ensemble (paper v2)
- hybrid_rank_v2: Hybrid rank ensemble v2 (policy-based)
- persistence: Persistence baseline (carry-forward)
- tcn: TCN risk score model

## Point Forecast Highlights (best MAE by source/horizon)
- [hybrid_rank_v2_guarded] H5: hybrid_point (Hybrid point predictor (weighted regression blend)) -> MAE=1.7127, RMSE=2.2115, MASE1=1.1040, MASE7=0.8149
- [main_linear] H1: enet_a0.1_l0.5 (ElasticNet (alpha=0.1, l1_ratio=0.5)) -> MAE=1.5373, RMSE=1.9287, MASE1=1.2295, MASE7=0.8116
- [main_linear] H3: enet_a0.1_l0.5 (ElasticNet (alpha=0.1, l1_ratio=0.5)) -> MAE=1.7191, RMSE=2.1631, MASE1=1.3744, MASE7=0.9070
- [main_linear] H5: enet_a1.0_l0.2 (ElasticNet (alpha=1.0, l1_ratio=0.2)) -> MAE=1.7944, RMSE=2.2665, MASE1=1.4486, MASE7=0.9522

## Alarm Highlights (best recall by source/horizon/tau/budget)
- [hybrid_rank_v2_guarded] H5, tau=15, r=0.05: hybrid_paper_fixed (Hybrid rank ensemble (paper-fixed weights)) -> precision=0.5526, recall=0.1567, TP=21
- [hybrid_rank_v2_guarded] H5, tau=15, r=0.10: enet (ElasticNet (generic)) -> precision=0.4400, recall=0.2463, TP=33
- [hybrid_rank_v2_guarded] H5, tau=16, r=0.05: hybrid_paper_fixed (Hybrid rank ensemble (paper-fixed weights)) -> precision=0.4474, recall=0.2394, TP=17
- [hybrid_rank_v2_guarded] H5, tau=16, r=0.10: hybrid_rank_v2 (Hybrid rank ensemble v2 (policy-based)) -> precision=0.3467, recall=0.3662, TP=26
- [main_alarm] H1, tau=15, r=0.05: persistence (Persistence baseline (carry-forward)) -> precision=0.7692, recall=0.2128, TP=30
- [main_alarm] H1, tau=15, r=0.10: persistence (Persistence baseline (carry-forward)) -> precision=0.6494, recall=0.3546, TP=50
- [main_alarm] H1, tau=16, r=0.05: persistence (Persistence baseline (carry-forward)) -> precision=0.5128, recall=0.2703, TP=20
- [main_alarm] H1, tau=16, r=0.10: persistence (Persistence baseline (carry-forward)) -> precision=0.4286, recall=0.4459, TP=33
- [main_alarm] H1, tau=17, r=0.05: persistence (Persistence baseline (carry-forward)) -> precision=0.2308, recall=0.3000, TP=9
- [main_alarm] H1, tau=17, r=0.10: enet_a0.01_l0.2 (ElasticNet (alpha=0.01, l1_ratio=0.2)) -> precision=0.1818, recall=0.4667, TP=14
- [main_alarm] H3, tau=15, r=0.05: enet_a1.0_l0.2 (ElasticNet (alpha=1.0, l1_ratio=0.2)) -> precision=0.6316, recall=0.1714, TP=24
- [main_alarm] H3, tau=15, r=0.10: enet_a0.01_l0.2 (ElasticNet (alpha=0.01, l1_ratio=0.2)) -> precision=0.5132, recall=0.2786, TP=39
- [main_alarm] H3, tau=16, r=0.05: enet_a1.0_l0.2 (ElasticNet (alpha=1.0, l1_ratio=0.2)) -> precision=0.4211, recall=0.2192, TP=16
- [main_alarm] H3, tau=16, r=0.10: enet_a0.01_l0.2 (ElasticNet (alpha=0.01, l1_ratio=0.2)) -> precision=0.3026, recall=0.3151, TP=23
- [main_alarm] H3, tau=17, r=0.05: enet_a1.0_l0.2 (ElasticNet (alpha=1.0, l1_ratio=0.2)) -> precision=0.2105, recall=0.2759, TP=8
- [main_alarm] H3, tau=17, r=0.10: enet_a0.01_l0.2 (ElasticNet (alpha=0.01, l1_ratio=0.2)) -> precision=0.1579, recall=0.4138, TP=12
- [main_alarm] H5, tau=15, r=0.05: enet_a0.1_l0.5 (ElasticNet (alpha=0.1, l1_ratio=0.5)) -> precision=0.5263, recall=0.1429, TP=20
- [main_alarm] H5, tau=15, r=0.10: enet_a0.1_l0.2 (ElasticNet (alpha=0.1, l1_ratio=0.2)) -> precision=0.4474, recall=0.2429, TP=34
- [main_alarm] H5, tau=16, r=0.05: persistence (Persistence baseline (carry-forward)) -> precision=0.3684, recall=0.1918, TP=14
- [main_alarm] H5, tau=16, r=0.10: enet_a0.1_l0.2 (ElasticNet (alpha=0.1, l1_ratio=0.2)) -> precision=0.3026, recall=0.3151, TP=23
- [main_alarm] H5, tau=17, r=0.05: enet_a0.01_l0.2 (ElasticNet (alpha=0.01, l1_ratio=0.2)) -> precision=0.2105, recall=0.2759, TP=8
- [main_alarm] H5, tau=17, r=0.10: enet_a0.1_l0.2 (ElasticNet (alpha=0.1, l1_ratio=0.2)) -> precision=0.1447, recall=0.3793, TP=11
- [paper_v2_h5] H5, tau=15, r=0.05: hybrid_rank_ens (Hybrid rank ensemble (paper v2)) -> precision=0.5526, recall=0.1567, TP=21
- [paper_v2_h5] H5, tau=15, r=0.10: enet_a0.1_l0.5 (ElasticNet (alpha=0.1, l1_ratio=0.5)) -> precision=0.4400, recall=0.2463, TP=33
- [paper_v2_h5] H5, tau=16, r=0.05: hybrid_rank_ens (Hybrid rank ensemble (paper v2)) -> precision=0.4474, recall=0.2394, TP=17
- [paper_v2_h5] H5, tau=16, r=0.10: persistence (Persistence baseline (carry-forward)) -> precision=0.3067, recall=0.3239, TP=23