# Corrected Alarm Evaluation Protocol

## 1. Scope and Separation Rule
This document defines two strictly separated alarm evaluations:
- Retrospective benchmark (diagnostic only, non-deployable)
- Prospective online policy (deployable-style chronological simulation)

Neither evaluation may be used to select models, runs, score weights, or operating points from outer-test outcomes.

## 2. Retrospective Benchmark (Non-Deployable)
### 2.1 Purpose
Quantify how models rank events when full held-out blocks are available after the fact.

### 2.2 Rule
For budget r and test size N:
- k = ceil(r x N)
- Sort test-block risk scores descending.
- Alarm top-k rows.

### 2.3 Required Labeling
All outputs must be labeled:
- retrospective test-set benchmark
- non-deployable (uses full held-out score distribution)

### 2.4 Allowed Metrics
Report TP, FP, FN, precision, recall, realized alarm fraction, false-alarm rate, and miss rate, but do not use these metrics for final model lock decisions.

## 3. Prospective Online Alarm Policy (Deployable-Style)
### 3.1 Per-Outer-Fold Calibration
For each horizon and outer fold:
- Use only inner-train/inner-validation scores to calibrate risk cutoff(s).
- Freeze cutoff(s) before any outer-test outcomes are viewed.
- Record calibration window dates and the frozen cutoff version.

### 3.2 Chronological Test Execution
During outer-test execution:
- Process rows in ascending feature_date order.
- For each date, issue alarm if score >= frozen cutoff.
- Do not rank the complete test block retrospectively.
- Do not recompute cutoff using future outer-test information.

### 3.3 Required Fold Outputs
For each fold and horizon, produce:
- realized_alarm_rate
- TP, FP, FN
- precision, recall
- false_alarm_rate = FP / (TP + FP)
- miss_rate = FN / (TP + FN)
- dated alarm log with feature_date, target_date, score, cutoff, alarm_flag, and realized event label

### 3.4 Update Constraints
Model or cutoff updates are allowed only when information would have been available by that date in operational time. No backfill from future observations is permitted.

## 4. Governance Rules
- Retrospective outputs are diagnostic and must not override prospective policy evidence.
- Final manuscript claims must explicitly identify whether a metric is retrospective or prospective.
- Cross-model comparisons require identical canonical outer-test dates per horizon/fold from corrected_split_assignment.csv.
- Any missing model-date predictions on canonical test windows must be flagged before comparison.

## 5. Minimal Release Artifacts
When reruns are authorized, each model/horizon should emit:
- prospective_fold_metrics.csv
- retrospective_budget_metrics.csv
- prospective_alarm_log.csv
- calibration_manifest.json (cutoff source partition, dates, and seed/state)

These artifacts must be generated only after corrected split validation is complete.
