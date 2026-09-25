# Persistence Baseline Definition

## Exact Formula
For each horizon H in {1, 3, 5} and forecast issue date t:

- target is y at t + H
- persistence prediction is y_hat at t + H = observed TNout at t

Equivalent expression used in legacy scripts when arrays store horizon-shifted targets:

- y_pred[i] = y_all[i - H]

This array form is mathematically identical to carry-forward from feature_date t because y_all[i] represents TNout at t + H.

## Source Script and Function Evidence
1. src/run_linear_alarm.py
- persistence_pred_from_y function defines y_hat(t+H)=TNout(t) and computes y_all[i-lag]: lines 42-49.
- persistence branch uses that function with lag=args.h: lines 92-97.

2. src/run_linear_alarm_v2.py
- persistence_pred_from_y computes y_all[i-lag] for horizon-H: lines 26-32.
- persistence branch calls the function with lag=args.h: lines 63-66.

3. src/save_fold_predictions.py
- persistence_pred computes y_all[i-lag]: lines 15-20.
- persistence path for H1 calls lag=1: lines 48-53.

## Submitted Methods Consistency
reports/_tmp_manuscript_extracted_latest.txt line 230 states:
- persistence baseline is carry-forward at the horizon lag.

This is consistent with the legacy implementation above.

## Required Information at Forecast Issue Time
To compute persistence for feature_date t and horizon H:
- required value is TNout observed at feature_date t
- no future date observation is required

## Confirmation of No Future Target Use
The persistence prediction uses TNout on feature_date only. The target TNout at target_date t+H is used only for evaluation after prediction generation.

## Difference from Submitted Implementation
No conceptual difference is introduced.

- Submitted implementation uses index arithmetic on horizon-shifted y arrays.
- Corrected controlled rerun uses explicit date mapping from feature_date to TNout.

These are equivalent definitions.
