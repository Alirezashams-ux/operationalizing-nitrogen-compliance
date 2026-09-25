# Methods and Materials for Main Manuscript

## 1) Study objective and decision framing

The methodological objective is to produce leakage-safe TNout forecasts that are decision-useful under bounded operator attention. The main outcome is not only lower average forecast error but higher exceedance capture under fixed alarm capacity.

## 2) Data, target, and feature engineering

Main data: data/raw/Ulsan_Yongsan.csv

Forecast horizons: H = 1, 3, 5 days.

Target equation:

$$
y_t^{(H)} = TNout_{t+H}
$$

Leakage-safe feature rule:

$$
\mathbf{x}_t = \phi(\mathcal{Z}_{\le t})
$$

Feature groups:

1. TNout lags and rolling memory.
2. Influent/load signals and rolling smoothers.
3. Weather and precipitation aggregates.
4. Stoichiometric proxies and interactions.
5. Seasonal harmonic terms.

Generated with src/build_ulsan_npz_v2.py into features/ulsan_H1/H3/H5_features_v2.npz.

## 3) Strict blocked leakage-safe validation

Chronological expanding blocked CV (n_splits = 3):

$$
\max(\mathcal{T}_i)<\min(\mathcal{E}_i),\quad \mathcal{T}_1\subset\mathcal{T}_2\subset\mathcal{T}_3
$$

No random shuffle is used.

Fold-safe scaling:

$$
\tilde{\mathbf{x}}_t = \frac{\mathbf{x}_t-\mu_i}{\sigma_i+10^{-6}},
\quad \mu_i,\sigma_i\text{ fit on train only}
$$

Why this matters: it eliminates look-ahead leakage and prevents fold contamination in preprocessing.

## 4) Model classes and selection logic

Models used:

1. Persistence baseline.
2. ElasticNet and Ridge.
3. HGBR (Optuna tuned).
4. BCR-TCN v1.1.
5. Hybrid point and hybrid rank ensembles.

Selection logic is policy-aware:

- Point layer: MAE, RMSE, MASE.
- Decision layer: top-k precision and recall under operational budgets.

## 5) Core equations for main manuscript

Point metrics:

$$
MAE = \frac{1}{n}\sum |y-\hat y|,
\quad
RMSE = \sqrt{\frac{1}{n}\sum (y-\hat y)^2}
$$

$$
MASE(m)=\frac{\frac{1}{n}\sum |y-\hat y|}{\frac{1}{n-m}\sum |y_t-y_{t-m}|+\epsilon}
$$

Alarm-budget policy:

$$
k=\lceil rn\rceil,
\quad
Precision@k=\frac{TP}{k},
\quad
Recall@k=\frac{TP}{TP+FN}
$$

Random baseline recall:

$$
\mathbb{E}[Recall@k]=r
$$

Hybrid rank equation:

$$
S_\tau(t)=w_{tcn}R(p_\tau^{tcn}(t)) + \sum_m w_m R(\hat y_m(t))
$$

with \(\sum_j w_j=1\) and \(w_j\ge0\).

Guarding constraint:

$$
Precision_{candidate}(\tau,r) \ge Precision_{paper-fixed}(\tau,r)
$$

Guarded policy form:

$$
S_\tau^{guarded}(t;r)=
\begin{cases}
S_\tau^{paper-fixed}(t), & r\le r_{guard}\\
S_\tau^{policy}(t;r), & r>r_{guard}
\end{cases}
$$

with \(r_{guard}=0.05\).

## 6) Optimization and training details to include briefly

BCR-TCN objective:

$$
\mathcal{L}_{total}=w_{reg}\mathcal{L}_{Huber}+w_{bce}\mathcal{L}_{BCE}+w_{rank}\mathcal{L}_{pair}
$$

Implemented weights: \(w_{reg}=1.0\), \(w_{bce}=0.6\), \(w_{rank}=1.4\).

Gradient clipping:

$$
\mathbf{g}_{clip}=\mathbf{g}\cdot\min\left(1,\frac{1.0}{\|\mathbf{g}\|_2+\epsilon}\right)
$$

Early stopping targets validation Recall@5% for tau = 16.

## 7) Methodological novelties to emphasize

1. Decision-first evaluation under fixed budget, not MAE-only ranking.
2. Guarded policy optimization to preserve conservative operational safety.
3. Explicitly leakage-safe split-transform-train chain.
4. Separate optimization of point and ranking layers, then controlled integration.

## 8) Strict-reviewer defense statements

1. No future information in features.
2. No fold leakage in scaling or tuning.
3. No test-fold tuning for hybrid weights.
4. No external hold-out retuning for SI transferability claims.

## 9) Counterintuitive Table 1 for main manuscript

Suggested title:

Methods and Materials Risk-Control Matrix (Counter to 2025-2026 Performance-Only Trend)

Suggested row topics:

1. Time split discipline.
2. Fold-local preprocessing.
3. Budget-constrained decision evaluation.
4. Guarded optimization constraint.
5. Uncertainty and decision-curve reporting.
6. External strict hold-out verification.
7. Stability diagnostics.
8. Artifact traceability and run reproducibility.

## 10) Main methods evidence files

- results/final_tables/results_discussion_csvs/Table1_leakage_safe_point_forecasting.csv
- results/final_tables/results_discussion_csvs/Table2_exceedance_prevalence_random_baseline.csv
- results/final_tables/results_discussion_csvs/Table3_alarm_budget_H5_core_operating_points.csv
- results/final_tables/results_discussion_csvs/Table4_H5_normalized_context_comparison.csv
- results/final_tables/results_discussion_csvs/Table5_key_operating_points_fold_uncertainty.csv