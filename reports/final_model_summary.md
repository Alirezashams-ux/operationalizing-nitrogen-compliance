# Final Model Performance Summary

## Best Point Forecast Model by Horizon (MAE)

- H1: enet_a0.1_l0.5 (MAE=1.5373, RMSE=1.9287, MASE1=1.2295, MASE7=0.8116)
- H3: enet_a0.1_l0.5 (MAE=1.7191, RMSE=2.1631, MASE1=1.3744, MASE7=0.9070)
- H5: enet_a1.0_l0.2 (MAE=1.7944, RMSE=2.2665, MASE1=1.4486, MASE7=0.9522)

## Best Alarm-Recall Model by Horizon / Operating Point

- H1, tau=15, r=0.05: persistence (precision=0.7692, recall=0.2128)
- H1, tau=15, r=0.10: persistence (precision=0.6494, recall=0.3546)
- H1, tau=16, r=0.05: persistence (precision=0.5128, recall=0.2703)
- H1, tau=16, r=0.10: persistence (precision=0.4286, recall=0.4459)
- H3, tau=15, r=0.05: enet_a1.0_l0.2 (precision=0.6316, recall=0.1714)
- H3, tau=15, r=0.10: enet_a0.01_l0.2 (precision=0.5132, recall=0.2786)
- H3, tau=16, r=0.05: enet_a1.0_l0.2 (precision=0.4211, recall=0.2192)
- H3, tau=16, r=0.10: enet_a0.01_l0.2 (precision=0.3026, recall=0.3151)
- H5, tau=15, r=0.05: enet_a0.1_l0.5 (precision=0.5263, recall=0.1429)
- H5, tau=15, r=0.10: persistence (precision=0.4474, recall=0.2429)
- H5, tau=16, r=0.05: persistence (precision=0.3684, recall=0.1918)
- H5, tau=16, r=0.10: persistence (precision=0.3026, recall=0.3151)

## H5 v2 Guarded Hybrid vs Baselines (overall)

- tau=15, r=0.05, enet: precision=0.4211, recall=0.1194
- tau=15, r=0.05, hgbr: precision=0.5000, recall=0.1418
- tau=15, r=0.05, hybrid_paper_fixed: precision=0.5526, recall=0.1567
- tau=15, r=0.05, hybrid_rank_v2: precision=0.5526, recall=0.1567
- tau=15, r=0.05, persistence: precision=0.5000, recall=0.1418
- tau=15, r=0.05, tcn: precision=0.5000, recall=0.1418
- tau=15, r=0.10, enet: precision=0.4400, recall=0.2463
- tau=15, r=0.10, hgbr: precision=0.4000, recall=0.2239
- tau=15, r=0.10, hybrid_paper_fixed: precision=0.3600, recall=0.2015
- tau=15, r=0.10, hybrid_rank_v2: precision=0.4400, recall=0.2463
- tau=15, r=0.10, persistence: precision=0.4267, recall=0.2388
- tau=15, r=0.10, tcn: precision=0.3867, recall=0.2164
- tau=16, r=0.05, enet: precision=0.2105, recall=0.1127
- tau=16, r=0.05, hgbr: precision=0.3684, recall=0.1972
- tau=16, r=0.05, hybrid_paper_fixed: precision=0.4474, recall=0.2394
- tau=16, r=0.05, hybrid_rank_v2: precision=0.4474, recall=0.2394
- tau=16, r=0.05, persistence: precision=0.3947, recall=0.2113
- tau=16, r=0.05, tcn: precision=0.3947, recall=0.2113
- tau=16, r=0.10, enet: precision=0.2533, recall=0.2676
- tau=16, r=0.10, hgbr: precision=0.2933, recall=0.3099
- tau=16, r=0.10, hybrid_paper_fixed: precision=0.2800, recall=0.2958
- tau=16, r=0.10, hybrid_rank_v2: precision=0.3467, recall=0.3662
- tau=16, r=0.10, persistence: precision=0.3067, recall=0.3239
- tau=16, r=0.10, tcn: precision=0.2400, recall=0.2535