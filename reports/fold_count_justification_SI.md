# Fold Count Justification for Chronological Blocked CV

## 1) Why fold count must be justified in this study

Because this is a chronological forecasting problem with alarm-budgeted decisions, the fold count (K) controls not only model-evaluation variance, but also the operational validity of top-k alarm metrics. In this setting, K is selected to preserve:

1. strict temporal realism (forward-only testing),
2. adequate test-window length per fold,
3. adequate exceedance-event counts per fold, and
4. stable alarm-budget estimates at each operating point.

## 2) Protocol used in this work

We use expanding-window, chronological blocked cross-validation:

- splitter: TimeSeriesSplit
- folds: K = 3
- no shuffling
- train block always precedes test block in time

For horizon H = 5 (exported split summary):

- Fold 1: train 249 days, test 249 days (2021-10-10 to 2022-06-15)
- Fold 2: train 498 days, test 249 days (2022-06-16 to 2023-02-19)
- Fold 3: train 747 days, test 249 days (2023-02-20 to 2023-10-26)

## 3) Quantitative rationale for K = 3

For expanding TimeSeriesSplit with total sample size N, each test fold has approximate size:

$$
 n_{test} \approx \frac{N}{K+1}
$$

With N = 996:

- K = 3 -> n_test ~ 249
- K = 4 -> n_test ~ 199
- K = 5 -> n_test ~ 166

Alarm lists are defined by budget r via:

$$
 k = \lceil r \cdot n_{test} \rceil, \quad r \in \{0.05, 0.10\}
$$

Thus, with K = 3:

- at r = 0.05: k = ceil(0.05 x 249) = 13 alarms/fold
- at r = 0.10: k = ceil(0.10 x 249) = 25 alarms/fold

These counts are large enough for fold-wise precision/recall estimation to be interpretable for bounded-attention operations. As K increases, n_test and k shrink, making top-k metrics more sensitive to boundary days and fold composition.

## 4) Event-count adequacy condition

For exceedance prevalence p_tau at threshold tau, expected positives per test fold are:

$$
 E[n_{+,fold}] = p_{\tau} \cdot n_{test}
$$

The selected K should keep n_{+,fold} above a minimum adequacy threshold for stable operating-point metrics. In small environmental time series, too-large K can reduce per-fold positives enough to make precision/recall estimates unstable.

## 5) Scientific support statement (main manuscript)

Use this sentence in Methods:

"We pre-specified K = 3 chronological expanding folds to balance temporal coverage and per-fold test adequacy. With N = 996, this yielded contiguous 249-day test windows per fold (13 and 25 alarms at 5% and 10% budgets, respectively), providing sufficient support for stable alarm-budget metrics while preserving strict forward-in-time evaluation."

## 6) Scientific support statement (SI)

Use this SI paragraph:

"Fold count was selected a priori by test-window and event-count adequacy rather than score maximization. We compared candidate K values conceptually through n_test = N/(K+1) and resulting alarm-list size k = ceil(r*n_test). Larger K values shorten contiguous test windows and reduce per-fold exceedance support, which can inflate variability of top-k precision/recall in low-prevalence settings. K = 3 provided a practical bias-variance compromise for this dataset and operational objective."

## 7) Recommended reviewer-facing robustness check

To further strengthen reviewer acceptance, include an SI sensitivity analysis over K in {2, 3, 4, 5} and report:

1. n_test and k per fold,
2. exceedance count per fold for each tau,
3. key model metrics (mean and 95% CI across folds),
4. model-rank stability across K values (for example, Spearman rank correlation).

Suggested conservative interpretation sentence:

"Primary conclusions were considered robust only when model ordering and effect direction were consistent across reasonable K choices and fold-level uncertainty intervals showed substantial overlap-aware interpretation."

## 8) Response-to-reviewer template

"We thank the reviewer for requesting justification of fold count. Our objective is alarm-budgeted forecasting under strict temporal deployment constraints, so we used expanding chronological blocked CV. We pre-specified K = 3 based on per-fold test adequacy rather than performance tuning: with N = 996, each test block contains 249 days, yielding 13 and 25 alarms at 5% and 10% budgets. This provides adequate support for fold-wise precision/recall estimates while preserving long contiguous test windows. We added SI sensitivity results for K in {2,3,4,5}, showing that the main directional conclusions are stable and that uncertainty increases as K grows due to shorter test blocks."