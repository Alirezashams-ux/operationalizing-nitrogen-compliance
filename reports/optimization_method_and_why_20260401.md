# Optimization Methods and Why These Methods Were Chosen

Date: 2026-04-01
Project: acs_tnout_ulsan

## 1) Optimization objective in this project

This project optimizes two layers, because operational early warning requires both accurate concentration prediction and good alarm prioritization.

1. Point forecasting objective: predict TNout at horizon H with low error.
2. Decision objective: maximize exceedance capture under a fixed alarm budget.

Horizon set used:
- H in {1, 3, 5} days.

Alarm operating points used:
- thresholds tau in {15, 16, 17} mg/L,
- budget r in {0.05, 0.10}.

## 2) Core optimization design principles

The same principles are applied across model families:

1. Chronological blocked optimization with TimeSeriesSplit (no shuffle).
2. Training-only fitting of preprocessing parameters inside each fold.
3. Hyperparameter/weight search only on training folds.
4. Held-out fold used only for final fold evaluation.

This prevents look-ahead leakage and keeps optimization aligned with forward-in-time deployment.

## 3) Optimization methods actually used

## 3.1 ElasticNet and linear baselines

Implemented in linear scripts with a sklearn Pipeline and StandardScaler.

### Method

For ElasticNet, the fold-level optimization solves:

$$
\min_{\beta} \frac{1}{n}\|y - X\beta\|_2^2 + \alpha\left(\rho\|\beta\|_1 + (1-\rho)\frac{1}{2}\|\beta\|_2^2\right)
$$

where:
- alpha = regularization strength,
- rho = l1_ratio.

Grid search is used over small transparent sets of (alpha, l1_ratio), evaluated by blocked CV metrics.

### Why this method

- Handles multicollinearity among lag/rolling engineered features.
- Balances sparsity and shrinkage, improving stability on modest-sized daily datasets.
- Transparent and reproducible baseline for manuscript reporting.

## 3.2 HGBR with Optuna

Implemented in the HGBR optimization script using Optuna.

### Method

The optimization problem is:

$$
\theta^{\ast} = \arg\min_{\theta\in\Theta}\; \frac{1}{F}\sum_{f=1}^{F} \mathrm{MAE}_f(\theta)
$$

with F = number of TimeSeriesSplit folds.

Optuna searches parameter space including:
- learning_rate,
- max_iter,
- max_leaf_nodes,
- min_samples_leaf,
- l2_regularization,
- max_bins.

Model is HistGradientBoostingRegressor with early_stopping enabled.

### Why this method

- Captures nonlinear interactions in engineered process and weather features.
- Histogram boosting is robust and efficient for tabular regression.
- Bayesian-style trial search (Optuna) is more sample-efficient than coarse exhaustive grids for this mixed parameter space.

## 3.3 BCR-TCN (deep sequence model)

Implemented in train_bcr_tcn_v11.py.

### Method

Optimization uses AdamW and a multi-objective surrogate loss:

$$
\mathcal{L} = w_{reg}\,\mathcal{L}_{Huber} + w_{bce}\,\mathcal{L}_{BCE} + w_{rank}\,\mathcal{L}_{rank}
$$

with:
- Huber for point regression,
- weighted BCE for exceedance-class imbalance at tau values,
- pairwise rank loss to improve ordering of high-risk days.

Validation model selection criterion is operational:

$$
\max\; \mathrm{Recall@}r\;\text{for }\tau=16,\; r=0.05
$$

Training controls include:
- chronological inner validation split (val tail inside outer-train),
- early stopping with patience,
- gradient norm clipping.

### Why this method

- Sequence structure of daily WWTP behavior is temporal and nonlinear; TCN is suitable for such dynamics.
- A pure MAE objective can miss alarm-ranking quality; adding BCE and rank terms aligns optimization with decision use.
- Selecting on Recall@5% directly matches the real bounded-attention operating condition.

## 3.4 Hybrid rank ensemble optimization

Implemented in make_hybrid_rank_ensemble_v2.py.

### Method

Two optimization stages are used:

1. Point hybrid weights for regression predictions:

$$
\hat{y}^{hyb} = \sum_{m=1}^{M} w_m \hat{y}^{(m)}, \quad w_m \ge 0,\; \sum_m w_m = 1
$$

Weights are selected on training folds by minimizing MAE.

2. Alarm rank hybrid weights for threshold-specific risk scores:

$$
s^{hyb}_{\tau} = \sum_{m=1}^{M} w_{m,\tau} \tilde{s}^{(m)}_{\tau}
$$

where each component score is foldwise rank-normalized to [0,1]:

$$
\tilde{s}_i = \frac{\mathrm{rank}(s_i)-1}{n-1}
$$

For guarded policy mode, low-budget constraints are enforced, including a precision floor versus baseline fixed weights.

### Why this method

- Different models capture different error structures; convex blending improves robustness.
- Rank normalization makes heterogeneous score scales comparable without leakage-prone global calibration.
- Guarded policy reduces unstable behavior in extreme top-k tails (small budget), which is critical for operator trust.

## 4) Why these optimization methods fit this project specifically

This project has constraints that favor these choices:

1. Small-to-moderate daily dataset with strong temporal dependence.
2. Main practical output is not only average error, but top-k alarm capture.
3. False confidence from leakage is a major risk in environmental time series.

Therefore the chosen stack is appropriate:
- ElasticNet for interpretable, stable linear regularization.
- HGBR for nonlinear tabular patterns.
- TCN for sequence dynamics and event ranking.
- Hybrid rank policy for decision-layer robustness under budget limits.

## 5) Optimization equations for manuscript-ready use

### 5.1 Point metrics used during model comparison

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

$$
\mathrm{MASE}(m) = \frac{\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|}{\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t-y_{t-m}|}
$$

### 5.2 Alarm-budget metrics

For score vector s and budget r:

$$
k = \lceil r n \rceil
$$

Top-k alarm set:

$$
\mathcal{A}_{k} = \text{indices of k largest } s_i
$$

Event indicator at threshold tau:

$$
e_i(\tau)=\mathbf{1}[y_i\ge\tau]
$$

$$
\mathrm{Precision@}k = \frac{\sum_{i\in\mathcal{A}_k} e_i}{k}, \quad
\mathrm{Recall@}k = \frac{\sum_{i\in\mathcal{A}_k} e_i}{\sum_{i=1}^{n} e_i}
$$

## 6) Trustability of optimization outcomes

Optimization outcomes are trustworthy because:

1. All searches are chronology-safe and fold-local.
2. Training, validation, and testing roles are strictly separated by time.
3. Objective functions are aligned to deployment goals (especially top-k recall under fixed budget).
4. Hyperparameters and run outputs are serialized in versioned artifacts and run configs.

## 7) Final concise answer: why this optimization method

The project uses a layered optimization strategy because no single method simultaneously guarantees temporal realism, strong point prediction, and robust low-budget alarm ranking.

- ElasticNet gives stable interpretable baselines.
- HGBR captures nonlinear tabular effects efficiently.
- BCR-TCN optimizes sequence-aware risk ordering directly.
- Guarded hybrid rank fusion improves operational reliability at conservative alarm budgets.

Together, these methods are the most suitable for this specific leakage-sensitive, decision-constrained wastewater forecasting task.
