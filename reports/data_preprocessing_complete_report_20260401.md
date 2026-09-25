# Complete Data Preprocessing Report (Project-Wide)

Date: 2026-04-01
Project: acs_tnout_ulsan

## 1) Scope and evidence base

This report documents data preprocessing as implemented in project code for:
- feature construction and NPZ export,
- chronology-safe split design,
- normalization/standardization,
- handling of missing and extreme values,
- preprocessing-related equations used in modeling and diagnostics.

Primary implementation sources:
- src/build_ulsan_npz.py
- src/build_ulsan_npz_v2.py
- src/build_ulsan_npz_H1.py
- src/run_linear_alarm.py
- src/run_linear_alarm_v2.py
- src/train_main_linear.py
- src/train_bcr_tcn_v11.py
- src/train_hgbr_optuna_v2.py
- src/save_hgbr_predictions_v2.py
- src/make_hybrid_rank_ensemble_v2.py
- src/analyze_yongyeon_raw_si.py
- src/prepare_si_tables_s3_s8_and_insert_docx.py

## 2) Raw data preparation and base cleaning

Across NPZ builders, the following steps are implemented:

1. Read CSV from data/raw/Ulsan_Yongsan.csv.
2. Drop unnamed index-like columns: columns whose name starts with Unnamed.
3. Parse Date to datetime with coercion.
4. Drop rows with invalid Date.
5. Sort rows by Date in ascending order.
6. Reset index.

Required variables are checked (with minor variant-specific column mapping in build_ulsan_npz_H1.py):
- TNout, Inflow, TNin, TOCin, temp_mean_c, precip_total_mm.
- If BODin is absent, it is created as NaN.

No row shuffling is used at preprocessing stage.

## 3) Target definition and leakage-safe temporal feature construction

### 3.1 Forecast target shift

For horizon H in {1, 3, 5}, target is shifted forward:

$$
y_t^{(H)} = \mathrm{TNout}_{t+H}
$$

implemented as shift(-H).

### 3.2 Derived chemistry proxy

$$
C\_N_t = \frac{\mathrm{TOCin}_t}{\mathrm{TNin}_t + 10^{-6}}
$$

### 3.3 Lag and rolling features (past-only)

The feature design is leakage-safe in time: lag/rolling operations are built so each feature at index t uses information available at or before t.

Core examples:

$$
\mathrm{TNout\_lag1}_t = \mathrm{TNout}_{t-1}
$$

$$
\mathrm{TNout\_roll7}_t = \frac{1}{7}\sum_{i=1}^{7}\mathrm{TNout}_{t-i}
$$

$$
\mathrm{precip\_sum3}_t = \sum_{i=0}^{2}\mathrm{precip}_{t-i}
$$

v2 extends this set (for stronger multi-horizon context), including additional lags (3, 5, 7), longer windows (14, 30), and interaction terms such as:

$$
\mathrm{Inflow\_x\_precip3}_t = \mathrm{Inflow}_t \cdot \mathrm{precip\_sum3}_t
$$

$$
\mathrm{temp\_x\_CN}_t = \mathrm{temp}_t \cdot C\_N_t
$$

### 3.4 Seasonality encoding

Day-of-year (doy) cyclical encoding:

$$
\mathrm{sin\_doy}_t = \sin\left(2\pi\frac{\mathrm{doy}_t}{365.25}\right), \quad
\mathrm{cos\_doy}_t = \cos\left(2\pi\frac{\mathrm{doy}_t}{365.25}\right)
$$

### 3.5 Complete-case row filtering before NPZ export

Rows are dropped if any selected feature or target y is missing:

$$
\mathcal{D}_{\text{final}} = \{t: x_{t,j}\text{ is finite for all selected }j,\; y_t\text{ is finite}\}
$$

This is implemented via dropna on feature_cols + [y].

## 4) Split strategy: splitting before normalization

## 4.1 Outer split protocol

Model evaluation/training scripts use chronological blocked forward-chaining via TimeSeriesSplit (typically n_splits = 3):

$$
(\mathcal{T}_f, \mathcal{E}_f),\; f=1,\dots,F,
$$

where all training indices precede test indices in time:

$$
\max(\mathcal{T}_f) < \min(\mathcal{E}_f)
$$

No shuffle.

### 4.2 Inner split for TCN

In BCR-TCN, each outer-train block is further split chronologically:
- sub-train = first (1 - val_frac) portion,
- validation = last val_frac tail (val_frac = 0.15).

If train indices in fold f are T_f (ordered), cutoff is:

$$
\text{cut}_f = \left\lfloor |\mathcal{T}_f|(1-\text{val\_frac})\right\rfloor
$$

$$
\mathcal{T}^{\text{sub}}_f = \mathcal{T}_f[0:\text{cut}_f], \quad
\mathcal{V}_f = \mathcal{T}_f[\text{cut}_f:]
$$

## 5) Normalization and scaling methods used

This project uses more than one normalization type, each for a specific layer.

### 5.1 StandardScaler for linear tabular models (ElasticNet, Ridge)

Used in sklearn Pipeline in linear scripts. For each fold, scaling is fit only on training subset and then applied to test subset.

Per feature j in fold f:

$$
\mu_{f,j} = \frac{1}{|\mathcal{T}_f|}\sum_{t\in\mathcal{T}_f} x_{t,j}
$$

$$
\sigma_{f,j} = \sqrt{\frac{1}{|\mathcal{T}_f|}\sum_{t\in\mathcal{T}_f}(x_{t,j}-\mu_{f,j})^2}
$$

Transformation for any split S in the same fold (train/test):

$$
\tilde{x}_{t,j} = \frac{x_{t,j}-\mu_{f,j}}{\sigma_{f,j}+\epsilon}, \quad \epsilon \approx 10^{-6}\text{ to }10^{-9}
$$

### 5.2 TCN input standardization (manual z-score)

In train_bcr_tcn_v11.py, scaling is computed from sub-train only (not full outer-train):

$$
\mu^{\text{sub}}_{f,j} = \frac{1}{|\mathcal{T}^{\text{sub}}_f|}\sum_{t\in\mathcal{T}^{\text{sub}}_f} x_{t,j}
$$

$$
\sigma^{\text{sub}}_{f,j} = \mathrm{std}_{t\in\mathcal{T}^{\text{sub}}_f}(x_{t,j})
$$

Applied to all rows in that fold context:

$$
\tilde{x}_{t,j} = \frac{x_{t,j}-\mu^{\text{sub}}_{f,j}}{\sigma^{\text{sub}}_{f,j}+10^{-6}}
$$

Then the transformed matrix is used for sub-train, validation, and test sequence datasets.

### 5.3 No feature scaling for HGBR

HistGradientBoostingRegressor scripts consume NPZ features directly, with no StandardScaler/MinMax/RobustScaler step in code.

### 5.4 Rank normalization for alarm-score fusion (decision layer)

In hybrid rank ensemble, model scores are rank-normalized to [0,1] before weighted fusion:

For a score vector s of length n:

$$
r_i = \mathrm{rank}(s_i; \text{ascending, average ties})
$$

$$
\hat{s}_i = \frac{r_i - 1}{n - 1}
$$

This is score normalization, not feature normalization.

### 5.5 What normalization types were done in this project

Implemented:
- z-score standardization (StandardScaler / manual mean-std scaling), fold-local and train-only.
- rank normalization to [0,1] for ensemble score fusion.

Not implemented in modeling preprocessing:
- MinMaxScaler,
- RobustScaler,
- quantile transform,
- power transform,
- global dataset-level scaling before split.

## 6) Why splitting before normalization is critical

If normalization is fit on the full dataset before splitting, future test distribution statistics leak into train transforms.

Leakage-safe protocol used here:
1. split chronologically,
2. fit transform on training (or sub-train) only,
3. apply frozen transform to val/test.

This preserves causal ordering and realistic deployment behavior.

Mathematically, leakage-safe transform in fold f is:

$$
\mathcal{A}_f = \mathrm{fit\_transformer}(X_{\mathcal{T}_f})
$$

$$
\tilde{X}_{\mathcal{E}_f} = \mathcal{A}_f(X_{\mathcal{E}_f})
$$

and never:

$$
\mathcal{A}^{\text{leaky}} = \mathrm{fit\_transformer}(X_{\mathcal{T}_f\cup\mathcal{E}_f})
$$

The implemented approach avoids optimistic bias in CV metrics and avoids train/test contamination.

## 7) Missing data handling

### 7.1 During feature assembly

- Numeric coercion is used in H1 builder for robust column typing.
- BODin may be absent and is set to NaN.
- Rolling windows and shifted targets naturally create NaNs at boundaries.
- Final NPZ dataset uses complete-case filtering (dropna on selected features + target).

### 7.2 During model training

Training scripts assume NPZ is already cleaned and finite for chosen features/target after builder filtering.

No explicit imputation method (mean/median/KNN/etc.) is applied in model scripts.

## 8) Outlier handling methods

## 8.1 In modeling pipeline (actual training/evaluation)

Implemented policy:
- No explicit outlier removal,
- No winsorization/clipping of feature values,
- No robust scaler transform for feature preprocessing.

This is consistent with project SI table generation script labels (outlier clipping: none for model families).

### 8.2 Outlier diagnostics in raw-data SI analysis

The raw-data quality analysis script computes outlier fractions using two diagnostic methods:

1. IQR rule:

$$
\mathrm{IQR} = Q_3 - Q_1
$$

$$
\text{outlier if } x < Q_1 - 1.5\,\mathrm{IQR} \;\text{or}\; x > Q_3 + 1.5\,\mathrm{IQR}
$$

2. Robust z-score (MAD-based):

$$
\tilde{x} = \mathrm{median}(x), \quad \mathrm{MAD} = \mathrm{median}(|x-\tilde{x}|)
$$

$$
z^{\ast} = 0.6745\frac{x-\tilde{x}}{\mathrm{MAD}}
$$

$$
\text{outlier if } |z^{\ast}| > 3.5
$$

Important: in this repository, these are used for descriptive quality reporting, not as automatic row deletion in model-building scripts.

### 8.3 Clarification: gradient clipping is not data outlier handling

TCN training applies gradient norm clipping:

$$
\|g\|_2 \leftarrow \min(\|g\|_2, c), \; c=1.0
$$

This stabilizes optimization and is unrelated to input-data outlier removal.

## 9) End-to-end preprocessing workflow summary

For each horizon H:

1. Build target y(t+H) and engineered features from chronologically sorted daily records.
2. Keep leakage-safe temporal features (lags/rolling/seasonality/interactions).
3. Drop rows with missing required feature/target values.
4. Save NPZ arrays X, y, dates, feature_names.
5. Run blocked chronological CV.
6. Fit normalization only on train (or sub-train for TCN), then transform val/test.
7. Train model and evaluate on future block only.
8. For rank-hybrid decisions, rank-normalize model scores and combine with fold-safe tuned weights.

## 10) Practical implications for this project

- The preprocessing is chronology-consistent and explicitly leakage-safe.
- Normalization is model-specific and split-aware.
- Outlier treatment is conservative: diagnostic quantification is present; hard trimming/clipping is not part of default model training.
- Reported performance is therefore aligned with real forward-in-time deployment constraints.

## 11) Compact checklist requested

- Splitting before normalization: implemented and essential to prevent temporal leakage.
- Normalizations done:
  - Standard z-score scaling (train-only; fold-local) for linear and TCN pipelines.
  - Rank normalization [0,1] for hybrid score fusion.
- Outlier handling done:
  - No explicit outlier clipping/removal in training preprocessing.
  - IQR and robust z-score methods used for SI raw-data outlier diagnostics.
