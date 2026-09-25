# Initial Descriptive Analysis: Ulsan_Yongsan (Main + Climate Features)

- Dataset file: `/home/alrezshams/acs_tnout_ulsan/data/raw/Ulsan_Yongsan.csv`
- Rows: 1031
- Period: 2021-01-04 to 2023-10-31
- Selected variables (8): Inflow, TNin, TOCin, BODin, TNout, temp_mean_c, precip_total_mm, wet_hours

## Target (TNout) profile
- Mean ± SD: 12.690 ± 2.234 mg/L
- Median [Q1, Q3]: 12.600 [11.110, 14.065] mg/L
- Range: 6.310 to 19.600 mg/L
- Exceedance counts/rates: τ=15 (169, 16.39%), τ=16 (91, 8.83%), τ=17 (31, 3.01%)

## Temporal dependence (TNout autocorrelation)
- lag1=0.627, lag3=0.401, lag5=0.354, lag7=0.318, lag14=0.183

## Bivariate signal vs TNout (Pearson correlation)
- Inflow: -0.261
- TNin: 0.312
- TOCin: 0.065
- BODin: 0.155
- temp_mean_c: -0.190
- precip_total_mm: -0.083
- wet_hours: -0.109

## Methodology-oriented interpretation (initial)
- Strong short-lag autocorrelation supports leakage-safe lag/rolling memory features and persistence baselines.
- Rare high-threshold exceedances (especially τ=17) justify alarm-budget evaluation (top-k ranking) in addition to MAE/RMSE.
- Moderate influent signal (e.g., TNin/TOCin/BODin) supports inclusion of process covariates for horizon-dependent forecasting.
- Climate variables (temperature/precipitation/wet-hours) should be retained as exogenous drivers, but likely with nonlinear or interaction effects.
- For methodology reporting, present both point-forecast metrics (MAE/RMSE/MASE) and constrained alarm metrics (precision/recall at fixed r).

- Descriptive stats table saved to: `/home/alrezshams/acs_tnout_ulsan/results/tables/Ulsan_Yongsan_main_feature_descriptive_stats.csv`