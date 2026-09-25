# Complete Feature Engineering Evolution Report (Project-Wide)

Date: 2026-04-01  
Scope: Ulsan main pipeline + Seoul external transfer pipeline  
Project: TNout leakage-safe forecasting and alarm-budget evaluation

## 1. Purpose of this report

This report documents what was actually implemented for features across the project, how the feature method evolved scientifically over time, and the exact mathematical definitions used in the pipeline.

Your current manuscript paragraph is directionally correct, but it under-specifies major implemented elements:
- expanded lag structure and multi-scale windows,
- explicit interaction terms,
- horizon-specific artifact generation,
- fold-safe scaling and evaluation integration,
- sequence-model usage of the same engineered feature space,
- cross-site harmonization for Seoul transfer.

## 2. Primary evidence used

Code sources reviewed:
- src/build_ulsan_npz_H1.py
- src/build_ulsan_npz.py
- src/build_ulsan_npz_v2.py
- src/run_linear_alarm_v2.py
- src/train_hgbr_optuna_v2.py
- src/save_hgbr_predictions_v2.py
- src/train_bcr_tcn.py
- src/train_bcr_tcn_v11.py
- src/make_hybrid_rank_ensemble_v2.py
- src/si_make_seoul_external_eval.py

Method/report sources reviewed:
- reports/Optimization_Process_Exact_Trustable_Results.md
- reports/complete_methods_foundations_novelty_report.md
- reports/methodology_full_report.md

Data artifacts checked:
- features/ulsan_H1_features.npz
- features/ulsan_H3_features.npz
- features/ulsan_H5_features.npz
- features/ulsan_H3_features_v2.npz
- features/ulsan_H5_features_v2.npz

Observed artifact status:
- Legacy NPZ exists for H1/H3/H5 with 18 features.
- v2 NPZ exists for H3/H5 with 34 features.
- v2 NPZ for H1 is referenced by scripts but not currently present in features/.

## 3. Scientific objective that drove feature design

The feature system was designed to solve a practical WWTP decision problem under bounded attention, not to build a fully mechanistic ASM-style simulator.

Core objective:
- predict future TNout at horizon H,
- convert predictions into ranked alarms,
- preserve strict temporal deployability (no leakage),
- retain physically meaningful drivers with minimal complexity.

So feature engineering balances three constraints:
1. Temporal causality: only information available at day t is allowed to predict day t+H.
2. Process relevance: include known influent/load/weather/seasonality drivers.
3. Sample efficiency: compact engineered covariates for small daily datasets.

## 4. Formal forecasting setup

For daily index t and horizon H in {1, 3, 5}:

$$
y_t^{(H)} = \mathrm{TNout}_{t+H}
$$

In code, this is implemented as:

$$
y_t^{(H)} = \mathrm{shift}_{-H}(\mathrm{TNout})
$$

with rows dropped where feature or target values are undefined after shifting/rolling.

## 5. Exact implemented feature classes and equations

Let x_t denote value at day t.

### 5.1 Autoregressive memory features

Legacy set:
- TNout_lag1
- TNout_roll7
- TNout_roll14

Expanded v2 set:
- TNout_lag1, TNout_lag3, TNout_lag5, TNout_lag7
- TNout_roll7, TNout_roll14, TNout_roll30

Definitions:

$$
\mathrm{TNout\_lag}k_t = \mathrm{TNout}_{t-k}
$$

$$
\mathrm{TNout\_roll}w_t = \frac{1}{w} \sum_{i=1}^{w} \mathrm{TNout}_{t-i}
$$

Important leakage-safe detail:
- rolling TNout terms use shift(1) before rolling, so day t never uses TNout_t while predicting t+H.

### 5.2 Influent and load features

Base variables:
- Inflow
- TNin
- TOCin
- BODin (optional in raw, set to NaN if absent)

Legacy smoothers:
- 7-day means for each base variable.

v2 smoothers:
- 7-day and 14-day means for each base variable.

Definition:

$$
\mathrm{Var\_roll}w_t = \frac{1}{w} \sum_{i=0}^{w-1} \mathrm{Var}_{t-i}
$$

### 5.3 Stoichiometric proxy features

Base ratio:

$$
\mathrm{C\_N}_t = \frac{\mathrm{TOCin}_t}{\mathrm{TNin}_t + 10^{-6}}
$$

v2 adds:

$$
\mathrm{C\_N\_roll14}_t = \frac{1}{14} \sum_{i=0}^{13} \mathrm{C\_N}_{t-i}
$$

Scientific reason:
- approximates carbon-to-nitrogen availability balance that influences denitrification potential, but in a compact empirical proxy form.

### 5.4 Weather forcing features

Base variables:
- temp_mean_c
- precip_total_mm

Legacy aggregates:
- temp_roll7
- precip_sum3

v2 aggregates:
- temp_roll7, temp_roll14, temp_roll30
- precip_sum3, precip_sum7, precip_sum14

Definitions:

$$
\mathrm{temp\_roll}w_t = \frac{1}{w} \sum_{i=0}^{w-1} \mathrm{temp}_{t-i}
$$

$$
\mathrm{precip\_sum}w_t = \sum_{i=0}^{w-1} \mathrm{precip}_{t-i}
$$

### 5.5 Seasonality features

Cyclic day-of-year encoding:

$$
\mathrm{sin\_doy}_t = \sin\left(2\pi\frac{\mathrm{doy}_t}{365.25}\right), \quad
\mathrm{cos\_doy}_t = \cos\left(2\pi\frac{\mathrm{doy}_t}{365.25}\right)
$$

Scientific reason:
- encodes annual periodicity continuously, avoiding discontinuity between day 365 and day 1.

### 5.6 Interaction features (v2 expansion)

Implemented interaction terms:

$$
\mathrm{Inflow\_x\_precip3}_t = \mathrm{Inflow}_t \cdot \mathrm{precip\_sum3}_t
$$

$$
\mathrm{temp\_x\_CN}_t = \mathrm{temp\_mean\_c}_t \cdot \mathrm{C\_N}_t
$$

$$
\mathrm{Inflow\_x\_TNin}_t = \mathrm{Inflow}_t \cdot \mathrm{TNin}_t
$$

Scientific reason:
- captures coupled hydraulic-loading and substrate-weather effects that linear additivity can miss.

## 6. What changed over time (scientific evolution)

### Phase A. Initial leakage-safe compact design (18 features)

Implemented in early/legacy build scripts.

Characteristics:
- 1 lag + short rolling memory,
- core influent/load variables,
- minimal weather aggregates,
- C/N proxy,
- cyclic seasonality.

Scientific role:
- robust baseline under limited data, strong interpretability, low variance.

### Phase B. Generalized horizon build for H in {1,3,5}

The same compact feature template was applied to multiple horizons via build_ulsan_npz.py and related scripts.

Scientific implication:
- established cross-horizon comparability with stable feature semantics.

### Phase C. v2 expanded temporal/context representation (34 features)

Implemented in build_ulsan_npz_v2.py.

Added:
- richer lag spectrum (1/3/5/7),
- longer memory windows (30-day TNout and temperature, 14-day load/weather smoothers),
- interaction terms.

Scientific motivation:
- improve long-horizon ranking utility (especially H=5),
- provide nonlinear learners (HGBR/TCN/hybrids) richer but still structured covariates,
- preserve operational realism (all daily measurable features).

### Phase D. Sequence modeling over engineered tabular space

In train_bcr_tcn.py and train_bcr_tcn_v11.py:
- TCN consumes sequences of the engineered feature vectors, not raw unprocessed streams.
- Sequence length L used as temporal context (default 30 in one version, 60 in v11).

Thus, feature evolution includes both:
- per-day engineered covariates,
- and multi-day causal sequence context over those covariates.

### Phase E. Policy-aware rank fusion (feature usage at decision layer)

In make_hybrid_rank_ensemble_v2.py:
- prediction outputs from models become ranking signals,
- signals are rank-normalized and weighted leakage-safely by fold/training data,
- policy-specific and guarded objectives tune how features-to-risk mappings are combined.

This is not new raw feature creation, but it is a major semantic evolution in how feature-derived predictions are operationalized.

### Phase F. External semantic transfer to Seoul

In si_make_seoul_external_eval.py:
- Seoul columns are harmonized to Ulsan schema,
- feature engineering mirrors v2 equations,
- missing fields handled explicitly (for example BODin fallback, precipitation fallback).

Scientific role:
- tests transportability of feature semantics under domain shift without retuning main claims.

## 7. Exact feature inventory sizes and artifact reality

Observed from saved NPZ files:
- ulsan_H1_features.npz: 18 features
- ulsan_H3_features.npz: 18 features
- ulsan_H5_features.npz: 18 features
- ulsan_H3_features_v2.npz: 34 features
- ulsan_H5_features_v2.npz: 34 features

Practical note:
- code references ulsan_H1_features_v2.npz for some v2 workflows, but the file is not currently present in features/.

## 8. Leakage-safe preprocessing and validation mathematics

### 8.1 Fold-safe scaling for linear/TCN workflows

For training subset T within a fold:

$$
\mu_j = \frac{1}{|T|}\sum_{i \in T} x_{ij}, \quad
\sigma_j = \sqrt{\frac{1}{|T|}\sum_{i \in T}(x_{ij}-\mu_j)^2} + \epsilon
$$

Transform any split S (train/val/test) using training statistics only:

$$
\tilde{x}_{ij} = \frac{x_{ij}-\mu_j}{\sigma_j}
$$

No test/validation information enters scaler fitting.

### 8.2 Time-ordered cross-validation

Using blocked TimeSeriesSplit with no shuffle:
- train block always earlier than test block.
- in TCN, train block is further split chronologically into sub-train and validation tail.

This enforces:

$$
\max(\text{time in train}) < \min(\text{time in test})
$$

for each fold.

## 9. Essential evaluation equations tied to feature pipeline

### 9.1 Point metrics

$$
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

MASE with seasonal period m:

$$
\mathrm{MASE}_m = \frac{\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|}{\frac{1}{N-m}\sum_{t=m+1}^{N}|y_t-y_{t-m}|}
$$

### 9.2 Alarm-budget metrics

For threshold tau and budget fraction r:

$$
\text{event}_i = \mathbb{1}(y_i \ge \tau), \quad k=\lceil r n \rceil
$$

Rank scores descending, alarm top k samples, then:

$$
\mathrm{TP}=\sum_{i \in \text{top-}k} \text{event}_i
$$

$$
\mathrm{Precision}=\frac{\mathrm{TP}}{k}, \quad
\mathrm{Recall}=\frac{\mathrm{TP}}{\sum_i \text{event}_i}
$$

Used enrichment expression:

$$
\mathrm{Enrichment}=\frac{\mathrm{Recall}}{r}
$$

## 10. TCN objective terms connected to feature learning

From train_bcr_tcn variants, the multitask loss combines:
- regression loss (Smooth L1),
- weighted BCE for exceedance logits,
- pairwise ranking loss (primarily at tau=16, with auxiliary tau=15 in one version),
- optional budget prior term in one version.

Generic form:

$$
\mathcal{L}=\lambda_{reg}\mathcal{L}_{reg}+\lambda_{bce}\mathcal{L}_{bce}+\lambda_{rank}\mathcal{L}_{rank}+\lambda_{budget}\mathcal{L}_{budget}
$$

This means engineered features are optimized not only for point fit, but for thresholded ranking utility under alarm-budget constraints.

## 11. Scientific interpretation of why the feature method evolved

The evolution is scientifically coherent:

1. Start with parsimonious physically interpretable leakage-safe features.
2. Expand temporal windows and interactions when moving to harder horizons and nonlinear models.
3. Retain the same feature semantics across model families so gains reflect modeling/policy effects, not hidden data leakage.
4. Add decision-layer rank fusion because operational objective is top-k exceedance capture, not only global error minimization.
5. Validate semantic robustness via Seoul transfer under harmonized feature construction.

In short, this project evolved from compact leakage-safe forecasting features to a full decision-grade feature-to-risk pipeline while preserving chronological validity.

## 12. Direct upgrade for your manuscript feature paragraph

Your paragraph is good foundation but should explicitly add:
- the expanded lag set (1,3,5,7) and longer windows (up to 30 days),
- interaction terms (Inflow x precip_sum3, temp x C_N, Inflow x TNin),
- C_N rolling smoother,
- horizon-specific feature artifacts and row-drop policy after rolling/shift,
- fold-safe scaling and chronology-preserving CV,
- sequence usage (TCN with L up to 60) over engineered feature vectors,
- policy-aware rank-level usage of model outputs for alarm-budget decisions.

That revised description will match what the project actually implemented and published in artifacts.

## 13. Reproducibility command set (feature-focused)

Feature build:
- /bin/python3 src/build_ulsan_npz.py --h 1
- /bin/python3 src/build_ulsan_npz.py --h 3
- /bin/python3 src/build_ulsan_npz.py --h 5
- /bin/python3 src/build_ulsan_npz_v2.py --h 1
- /bin/python3 src/build_ulsan_npz_v2.py --h 3
- /bin/python3 src/build_ulsan_npz_v2.py --h 5

Linear alarm workflow on v2 features:
- /bin/python3 src/run_linear_alarm_v2.py --h 3
- /bin/python3 src/run_linear_alarm_v2.py --h 5

HGBR tuning/predictions:
- /bin/python3 src/train_hgbr_optuna_v2.py --h 3
- /bin/python3 src/train_hgbr_optuna_v2.py --h 5
- /bin/python3 src/save_hgbr_predictions_v2.py --h 3
- /bin/python3 src/save_hgbr_predictions_v2.py --h 5

TCN and hybrid rank:
- /bin/python3 src/train_bcr_tcn_v11.py --h 5 --device cpu
- /bin/python3 src/make_hybrid_rank_ensemble_v2.py --h 5 --rank_objective policy_guarded

Seoul external semantic transfer:
- /bin/python3 src/si_make_seoul_external_eval.py
