from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/home/alrezshams/acs_tnout_ulsan_revision')
OUT = ROOT / 'revision_2026' / '02_corrected_pipeline'
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 5]
FOLDS = [1, 2, 3]
VAL_FRAC = 0.15


def dt(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def dstr(x: pd.Timestamp | None) -> str:
    if x is None or pd.isna(x):
        return 'NA'
    return pd.Timestamp(x).strftime('%Y-%m-%d')


def serialize_dates(dates: List[pd.Timestamp]) -> str:
    if not dates:
        return 'NA'
    return ';'.join(dstr(x) for x in dates)


def serialize_pairs(feature_dates: List[pd.Timestamp], horizon: int) -> str:
    if not feature_dates:
        return 'NA'
    pairs = []
    for f in feature_dates:
        t = f + pd.Timedelta(days=horizon)
        pairs.append(f'{dstr(f)}->{dstr(t)}')
    return ';'.join(pairs)


def load_assignment(h: int) -> pd.DataFrame:
    p = ROOT / 'results' / 'tables' / f'blocked_cv_assignment_matrix_H{h}.csv'
    d = pd.read_csv(p)
    d['date'] = pd.to_datetime(d['date']).dt.normalize()
    return d


def outer_split_dates(assign: pd.DataFrame, fold: int) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    col = f'fold{fold}_set'
    tr = sorted(assign.loc[assign[col] == 'train', 'date'].tolist())
    te = sorted(assign.loc[assign[col] == 'test', 'date'].tolist())
    return tr, te


def split_inner_tail(train_dates: List[pd.Timestamp], frac: float = VAL_FRAC) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    cut = int(math.floor(len(train_dates) * (1.0 - frac)))
    sub_train = train_dates[:cut]
    val = train_dates[cut:]
    return sub_train, val


def reconstructed_random_val_split(train_dates: List[pd.Timestamp], frac: float = 0.10, seed: int = 42) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    n = len(train_dates)
    n_val = max(1, int(math.ceil(frac * n)))
    perm = np.random.RandomState(seed).permutation(n)
    val_idx = sorted(perm[:n_val].tolist())
    sub_idx = sorted(perm[n_val:].tolist())
    sub_train = [train_dates[i] for i in sub_idx]
    val = [train_dates[i] for i in val_idx]
    return sub_train, val


def boundary_check(train_feature: List[pd.Timestamp], holdout_feature: List[pd.Timestamp], horizon: int) -> Dict[str, object]:
    train_target = [d + pd.Timedelta(days=horizon) for d in train_feature]
    holdout_set = set(holdout_feature)
    overlap_target_dates = sorted(set(train_target).intersection(holdout_set))
    offending_feature_rows = [d for d in train_feature if (d + pd.Timedelta(days=horizon)) in holdout_set]

    max_train_target = max(train_target) if train_target else None
    min_holdout_feature = min(holdout_feature) if holdout_feature else None
    condition_pass = bool(max_train_target is not None and min_holdout_feature is not None and max_train_target < min_holdout_feature)

    purge_days = 0
    if not condition_pass and max_train_target is not None and min_holdout_feature is not None:
        purge_days = int((max_train_target - min_holdout_feature).days + 1)

    return {
        'pass': condition_pass,
        'overlap_target_dates': overlap_target_dates,
        'offending_feature_rows': offending_feature_rows,
        'rows_to_remove': len(offending_feature_rows),
        'effective_purge_days': max(0, purge_days),
        'max_train_target': max_train_target,
        'min_holdout_feature': min_holdout_feature,
    }


def build_fold_boundary_audit() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for h in HORIZONS:
        assign = load_assignment(h)

        for fold in FOLDS:
            train_dates, test_dates = outer_split_dates(assign, fold)
            sub_train_dates, val_dates = split_inner_tail(train_dates, frac=VAL_FRAC)

            # Part A canonical temporal index
            rows.append(
                {
                    'record_type': 'canonical_temporal_index',
                    'part': 'A',
                    'horizon': h,
                    'fold': fold,
                    'model_family': 'all_models',
                    'feature_date_train_start': dstr(min(sub_train_dates) if sub_train_dates else None),
                    'feature_date_train_end': dstr(max(sub_train_dates) if sub_train_dates else None),
                    'target_date_train_start': dstr((min(sub_train_dates) + pd.Timedelta(days=h)) if sub_train_dates else None),
                    'target_date_train_end': dstr((max(sub_train_dates) + pd.Timedelta(days=h)) if sub_train_dates else None),
                    'feature_date_validation_start': dstr(min(val_dates) if val_dates else None),
                    'feature_date_validation_end': dstr(max(val_dates) if val_dates else None),
                    'target_date_validation_start': dstr((min(val_dates) + pd.Timedelta(days=h)) if val_dates else None),
                    'target_date_validation_end': dstr((max(val_dates) + pd.Timedelta(days=h)) if val_dates else None),
                    'feature_date_test_start': dstr(min(test_dates) if test_dates else None),
                    'feature_date_test_end': dstr(max(test_dates) if test_dates else None),
                    'target_date_test_start': dstr((min(test_dates) + pd.Timedelta(days=h)) if test_dates else None),
                    'target_date_test_end': dstr((max(test_dates) + pd.Timedelta(days=h)) if test_dates else None),
                    'outer_train_rows': len(train_dates),
                    'training_rows': len(sub_train_dates),
                    'validation_rows': len(val_dates),
                    'test_rows': len(test_dates),
                    'boundary_condition': 'NA',
                    'boundary_result': 'NA',
                    'overlap_target_dates_count': 'NA',
                    'overlap_target_dates': 'NA',
                    'offending_training_rows': 'NA',
                    'boundary_reference_date': 'NA',
                    'rows_to_remove': 'NA',
                    'effective_purge_days': 'NA',
                    'validation_tail_size': len(val_dates),
                    'last_15_percent_rule': 'YES',
                    'h_day_purge_applied': 'NO',
                    'predictions_relative_to_required_purge': 'before_required_purge',
                    'evidence': 'results/tables/blocked_cv_assignment_matrix_H{h}.csv + src/export_methods_figure_cv_csvs.py_VAL_FRAC_0.15'.format(h=h),
                    'notes': 'target_date = feature_date + horizon verified in feature builders (y=TNout.shift(-h)).',
                }
            )

            # Part B outer train-test boundary
            outer = boundary_check(train_dates, test_dates, horizon=h)
            rows.append(
                {
                    'record_type': 'outer_train_test_purge',
                    'part': 'B',
                    'horizon': h,
                    'fold': fold,
                    'model_family': 'all_models_using_blocked_cv',
                    'feature_date_train_start': dstr(min(train_dates) if train_dates else None),
                    'feature_date_train_end': dstr(max(train_dates) if train_dates else None),
                    'target_date_train_start': dstr((min(train_dates) + pd.Timedelta(days=h)) if train_dates else None),
                    'target_date_train_end': dstr((max(train_dates) + pd.Timedelta(days=h)) if train_dates else None),
                    'feature_date_validation_start': 'NA',
                    'feature_date_validation_end': 'NA',
                    'target_date_validation_start': 'NA',
                    'target_date_validation_end': 'NA',
                    'feature_date_test_start': dstr(min(test_dates) if test_dates else None),
                    'feature_date_test_end': dstr(max(test_dates) if test_dates else None),
                    'target_date_test_start': dstr((min(test_dates) + pd.Timedelta(days=h)) if test_dates else None),
                    'target_date_test_end': dstr((max(test_dates) + pd.Timedelta(days=h)) if test_dates else None),
                    'outer_train_rows': len(train_dates),
                    'training_rows': len(train_dates),
                    'validation_rows': 0,
                    'test_rows': len(test_dates),
                    'boundary_condition': 'max(outer_training_target_date) < min(test_feature_date)',
                    'boundary_result': 'PASS' if outer['pass'] else 'FAIL',
                    'overlap_target_dates_count': len(outer['overlap_target_dates']),
                    'overlap_target_dates': serialize_dates(outer['overlap_target_dates']),
                    'offending_training_rows': serialize_pairs(outer['offending_feature_rows'], h),
                    'boundary_reference_date': dstr(outer['min_holdout_feature']),
                    'rows_to_remove': outer['rows_to_remove'],
                    'effective_purge_days': outer['effective_purge_days'],
                    'validation_tail_size': 'NA',
                    'last_15_percent_rule': 'NA',
                    'h_day_purge_applied': 'NO',
                    'predictions_relative_to_required_purge': 'before_required_purge',
                    'evidence': 'TimeSeriesSplit with no gap in src/save_fold_predictions.py, src/run_linear_alarm.py, src/save_hgbr_predictions_v2.py, src/train_bcr_tcn_v11.py',
                    'notes': 'No purge-gap parameter used in any blocked CV training/prediction script.',
                }
            )

            # Part C inner boundary for TCN/BCR-TCN (chronological 15% tail)
            inner_tcn = boundary_check(sub_train_dates, val_dates, horizon=h)
            rows.append(
                {
                    'record_type': 'inner_train_validation_purge',
                    'part': 'C',
                    'horizon': h,
                    'fold': fold,
                    'model_family': 'TCN/BCR-TCN',
                    'feature_date_train_start': dstr(min(sub_train_dates) if sub_train_dates else None),
                    'feature_date_train_end': dstr(max(sub_train_dates) if sub_train_dates else None),
                    'target_date_train_start': dstr((min(sub_train_dates) + pd.Timedelta(days=h)) if sub_train_dates else None),
                    'target_date_train_end': dstr((max(sub_train_dates) + pd.Timedelta(days=h)) if sub_train_dates else None),
                    'feature_date_validation_start': dstr(min(val_dates) if val_dates else None),
                    'feature_date_validation_end': dstr(max(val_dates) if val_dates else None),
                    'target_date_validation_start': dstr((min(val_dates) + pd.Timedelta(days=h)) if val_dates else None),
                    'target_date_validation_end': dstr((max(val_dates) + pd.Timedelta(days=h)) if val_dates else None),
                    'feature_date_test_start': 'NA',
                    'feature_date_test_end': 'NA',
                    'target_date_test_start': 'NA',
                    'target_date_test_end': 'NA',
                    'outer_train_rows': len(train_dates),
                    'training_rows': len(sub_train_dates),
                    'validation_rows': len(val_dates),
                    'test_rows': 0,
                    'boundary_condition': 'max(inner_subtrain_target_date) < min(validation_feature_date)',
                    'boundary_result': 'PASS' if inner_tcn['pass'] else 'FAIL',
                    'overlap_target_dates_count': len(inner_tcn['overlap_target_dates']),
                    'overlap_target_dates': serialize_dates(inner_tcn['overlap_target_dates']),
                    'offending_training_rows': serialize_pairs(inner_tcn['offending_feature_rows'], h),
                    'boundary_reference_date': dstr(inner_tcn['min_holdout_feature']),
                    'rows_to_remove': inner_tcn['rows_to_remove'],
                    'effective_purge_days': inner_tcn['effective_purge_days'],
                    'validation_tail_size': len(val_dates),
                    'last_15_percent_rule': 'YES',
                    'h_day_purge_applied': 'NO',
                    'predictions_relative_to_required_purge': 'before_required_purge',
                    'evidence': 'src/train_bcr_tcn_v11.py cut=floor(len(tr_idx)*(1-val_frac)); val_frac=0.15; no purge step',
                    'notes': 'Chronological validation tail exists, but contiguous split causes horizon-dependent leakage at boundary.',
                }
            )

            # Part C inner boundary for HGBR (reconstructed random validation_fraction=0.1)
            hg_sub, hg_val = reconstructed_random_val_split(train_dates, frac=0.10, seed=42)
            inner_hg = boundary_check(hg_sub, hg_val, horizon=h)
            rows.append(
                {
                    'record_type': 'inner_train_validation_purge',
                    'part': 'C',
                    'horizon': h,
                    'fold': fold,
                    'model_family': 'HGBR',
                    'feature_date_train_start': dstr(min(hg_sub) if hg_sub else None),
                    'feature_date_train_end': dstr(max(hg_sub) if hg_sub else None),
                    'target_date_train_start': dstr((min(hg_sub) + pd.Timedelta(days=h)) if hg_sub else None),
                    'target_date_train_end': dstr((max(hg_sub) + pd.Timedelta(days=h)) if hg_sub else None),
                    'feature_date_validation_start': dstr(min(hg_val) if hg_val else None),
                    'feature_date_validation_end': dstr(max(hg_val) if hg_val else None),
                    'target_date_validation_start': dstr((min(hg_val) + pd.Timedelta(days=h)) if hg_val else None),
                    'target_date_validation_end': dstr((max(hg_val) + pd.Timedelta(days=h)) if hg_val else None),
                    'feature_date_test_start': 'NA',
                    'feature_date_test_end': 'NA',
                    'target_date_test_start': 'NA',
                    'target_date_test_end': 'NA',
                    'outer_train_rows': len(train_dates),
                    'training_rows': len(hg_sub),
                    'validation_rows': len(hg_val),
                    'test_rows': 0,
                    'boundary_condition': 'max(inner_subtrain_target_date) < min(validation_feature_date)',
                    'boundary_result': 'PASS' if inner_hg['pass'] else 'FAIL',
                    'overlap_target_dates_count': len(inner_hg['overlap_target_dates']),
                    'overlap_target_dates': serialize_dates(inner_hg['overlap_target_dates']),
                    'offending_training_rows': serialize_pairs(inner_hg['offending_feature_rows'], h),
                    'boundary_reference_date': dstr(inner_hg['min_holdout_feature']),
                    'rows_to_remove': inner_hg['rows_to_remove'],
                    'effective_purge_days': inner_hg['effective_purge_days'],
                    'validation_tail_size': len(hg_val),
                    'last_15_percent_rule': 'NO',
                    'h_day_purge_applied': 'NO',
                    'predictions_relative_to_required_purge': 'before_required_purge',
                    'evidence': 'src/train_hgbr_optuna_v2.py and src/save_hgbr_predictions_v2.py use HistGradientBoostingRegressor(early_stopping=True, validation_fraction=0.1, random_state=42)',
                    'notes': 'Validation split reconstructed using sklearn random split assumption (shuffle=True, random_state=42).',
                }
            )

            # Part C rows for model families without explicit inner validation subset in final fold fit
            for model_family, evidence, note in [
                (
                    'ElasticNet',
                    'src/save_fold_predictions.py and src/run_linear_alarm.py fit Pipeline(StandardScaler, ElasticNet) directly on outer train folds.',
                    'No explicit inner validation tail in final fold fit; hyperparameter selection handled separately.',
                ),
                (
                    'Ridge',
                    'src/train_main_linear.py evaluates Ridge grid via TimeSeriesSplit; no saved ridge fold predictions.',
                    'No explicit inner validation tail in final fold fit artifacts.',
                ),
                (
                    'HybridRank',
                    'src/make_hybrid_rank_ensemble_v2.py uses tr = rows with fold != test_fold; no subtrain/validation split.',
                    'Cross-fold training rows used directly for objective evaluation; no explicit validation tail.',
                ),
                (
                    'point blend',
                    'src/make_hybrid_rank_ensemble_v2.py point-hybrid weights selected by train MAE on tr folds.',
                    'No explicit inner validation tail; objective evaluated on train folds only.',
                ),
                (
                    'risk-score blend',
                    'src/make_hybrid_rank_ensemble_v2.py rank weights selected on tr folds; budget-specific guarded objective optional.',
                    'No explicit inner validation tail; objective evaluated on train folds only.',
                ),
            ]:
                rows.append(
                    {
                        'record_type': 'inner_train_validation_purge',
                        'part': 'C',
                        'horizon': h,
                        'fold': fold,
                        'model_family': model_family,
                        'feature_date_train_start': 'NA',
                        'feature_date_train_end': 'NA',
                        'target_date_train_start': 'NA',
                        'target_date_train_end': 'NA',
                        'feature_date_validation_start': 'NA',
                        'feature_date_validation_end': 'NA',
                        'target_date_validation_start': 'NA',
                        'target_date_validation_end': 'NA',
                        'feature_date_test_start': 'NA',
                        'feature_date_test_end': 'NA',
                        'target_date_test_start': 'NA',
                        'target_date_test_end': 'NA',
                        'outer_train_rows': len(train_dates),
                        'training_rows': 'NA',
                        'validation_rows': 'NA',
                        'test_rows': 'NA',
                        'boundary_condition': 'max(inner_subtrain_target_date) < min(validation_feature_date)',
                        'boundary_result': 'N/A',
                        'overlap_target_dates_count': 'N/A',
                        'overlap_target_dates': 'N/A',
                        'offending_training_rows': 'N/A',
                        'boundary_reference_date': 'N/A',
                        'rows_to_remove': 'N/A',
                        'effective_purge_days': 'N/A',
                        'validation_tail_size': 'N/A',
                        'last_15_percent_rule': 'NO',
                        'h_day_purge_applied': 'NO',
                        'predictions_relative_to_required_purge': 'before_required_purge',
                        'evidence': evidence,
                        'notes': note,
                    }
                )

    df = pd.DataFrame(rows)
    return df


def build_feature_leakage_audit() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(
        feature_name: str,
        horizon_coverage: str,
        source_column: str,
        formula: str,
        shift_before_rolling: str,
        current_day_tnout_included: str,
        future_obs_possible: str,
        missing_handling: str,
        fold_boundary_handling: str,
        leakage_status: str,
        evidence_script: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                'feature_name': feature_name,
                'horizon_coverage': horizon_coverage,
                'source_column': source_column,
                'feature_formula': formula,
                'shift_before_rolling': shift_before_rolling,
                'current_day_tnout_included': current_day_tnout_included,
                'future_observation_possible': future_obs_possible,
                'missing_value_handling': missing_handling,
                'fold_boundary_handling': fold_boundary_handling,
                'leakage_status': leakage_status,
                'evidence_script': evidence_script,
                'notes': notes,
            }
        )

    # TNout lag and rolling features
    add('TNout_lag1', 'H1/H3/H5', 'TNout', 'TNout(t-1)', 'N/A', 'NO', 'NO', 'dropna after feature build', 'computed globally, uses past history across fold boundary', 'PASS_no_future_leakage', 'src/build_ulsan_npz.py + src/build_ulsan_npz_H1.py + src/build_ulsan_npz_v2.py', 'Lag uses only prior day TNout.')
    add('TNout_lag3', 'H3/H5_v2', 'TNout', 'TNout(t-3)', 'N/A', 'NO', 'NO', 'dropna after feature build', 'same', 'PASS_no_future_leakage', 'src/build_ulsan_npz_v2.py', 'Not present in legacy H1/H3 non-v2 build.')
    add('TNout_lag5', 'H3/H5_v2', 'TNout', 'TNout(t-5)', 'N/A', 'NO', 'NO', 'dropna after feature build', 'same', 'PASS_no_future_leakage', 'src/build_ulsan_npz_v2.py', 'Lagged strictly backward.')
    add('TNout_lag7', 'H3/H5_v2', 'TNout', 'TNout(t-7)', 'N/A', 'NO', 'NO', 'dropna after feature build', 'same', 'PASS_no_future_leakage', 'src/build_ulsan_npz_v2.py', 'Lagged strictly backward.')

    add('TNout_roll7', 'H1/H3/H5', 'TNout', 'mean(TNout.shift(1), window=7)', 'shift(1)', 'NO', 'NO', 'dropna after feature build', 'same', 'PASS_no_future_leakage', 'src/build_ulsan_npz.py + src/build_ulsan_npz_H1.py + src/build_ulsan_npz_v2.py', 'Explicit shift(1) prevents current-day TNout leakage.')
    add('TNout_roll14', 'H1/H3/H5', 'TNout', 'mean(TNout.shift(1), window=14)', 'shift(1)', 'NO', 'NO', 'dropna after feature build', 'same', 'PASS_no_future_leakage', 'src/build_ulsan_npz.py + src/build_ulsan_npz_H1.py + src/build_ulsan_npz_v2.py', 'Explicit shift(1) prevents current-day TNout leakage.')
    add('TNout_roll30', 'H3/H5_v2', 'TNout', 'mean(TNout.shift(1), window=30)', 'shift(1)', 'NO', 'NO', 'dropna after feature build', 'same', 'PASS_no_future_leakage', 'src/build_ulsan_npz_v2.py', 'Expanded memory in v2 build.')

    # Weather and precipitation
    add('temp_mean_c', 'H1/H3/H5', 'temp_mean_c', 'raw temp_mean_c(t)', 'N/A', 'N/A', 'NO', 'numeric coercion then dropna', 'same', 'PASS_no_future_leakage_but_requires_issue_time_availability', 'src/build_ulsan_npz.py + src/build_ulsan_npz_H1.py + src/build_ulsan_npz_v2.py', 'Uses current-day weather measurement.')
    add('temp_roll7/temp_roll14/temp_roll30', 'H1/H3/H5 (roll14/30 in v2)', 'temp_mean_c', 'rolling mean over recent days (includes day t)', 'no pre-shift', 'N/A', 'NO', 'dropna', 'same', 'PASS_no_future_leakage_but_current_day_dependency', 'src/build_ulsan_npz.py + src/build_ulsan_npz_v2.py', 'No forward-looking window; includes current day.')
    add('precip_total_mm', 'H1/H3/H5', 'precip_total_mm', 'raw precipitation(t)', 'N/A', 'N/A', 'NO', 'numeric coercion then dropna', 'same', 'PASS_no_future_leakage_but_requires_issue_time_availability', 'src/build_ulsan_npz.py + src/build_ulsan_npz_H1.py + src/build_ulsan_npz_v2.py', 'Current-day precipitation is used as predictor.')
    add('precip_sum3/7/14', 'H1/H3/H5 (7/14 in v2)', 'precip_total_mm', 'rolling sum over recent days (includes day t)', 'no pre-shift', 'N/A', 'NO', 'dropna', 'same', 'PASS_no_future_leakage_but_current_day_dependency', 'src/build_ulsan_npz.py + src/build_ulsan_npz_v2.py', 'No future values enter rolling sums.')

    # Process variables and ratios
    add('Inflow/TNin/TOCin/BODin raw + rolling', 'H1/H3/H5', 'Inflow,TNin,TOCin,BODin', 'raw(t) and rolling means over recent days', 'no pre-shift', 'N/A', 'NO', 'dropna; BODin may be synthetic NaN column when missing', 'same', 'PASS_no_future_leakage_but_current_day_dependency', 'src/build_ulsan_npz.py + src/build_ulsan_npz_H1.py + src/build_ulsan_npz_v2.py', 'Current-day process measurements included; no forward windows.')
    add('C_N and C_N_roll14', 'H1/H3/H5 (roll14 in v2)', 'TOCin,TNin', 'C_N = TOCin/(TNin+1e-6); optional rolling mean', 'no pre-shift', 'N/A', 'NO', 'dropna', 'same', 'PASS_no_future_leakage', 'src/build_ulsan_npz.py + src/build_ulsan_npz_H1.py + src/build_ulsan_npz_v2.py', 'Ratio built from same-day influent variables.')

    # Interactions and seasonality
    add('Inflow_x_precip3,temp_x_CN,Inflow_x_TNin', 'H3/H5_v2', 'derived from existing predictors', 'pairwise interactions of contemporaneous/rolling inputs', 'inherits parent feature shift', 'N/A', 'NO', 'dropna', 'same', 'PASS_no_future_leakage', 'src/build_ulsan_npz_v2.py', 'No interaction uses future columns.')
    add('sin_doy, cos_doy', 'H1/H3/H5', 'Date', 'sin/cos(day_of_year)', 'N/A', 'N/A', 'NO', 'always available from date', 'same', 'PASS_no_future_leakage', 'src/build_ulsan_npz.py + src/build_ulsan_npz_H1.py + src/build_ulsan_npz_v2.py', 'Calendar encoding only.')

    # Explicit checks requested by prompt
    add('forward_fill_or_interpolation', 'H1/H3/H5', 'N/A', 'not used in feature builders', 'N/A', 'N/A', 'NO', 'N/A', 'N/A', 'PASS_not_detected_in_feature_scripts', 'src/build_ulsan_npz.py + src/build_ulsan_npz_H1.py + src/build_ulsan_npz_v2.py', 'No .ffill(), .bfill(), or .interpolate() calls in feature engineering scripts.')

    return pd.DataFrame(rows)


def build_preprocessing_model_selection_audit() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(
        record_type: str,
        model_family: str,
        horizon_scope: str,
        fold_scope: str,
        item: str,
        implementation: str,
        fit_scope: str,
        validation_construction: str,
        uses_outer_test_information: str,
        uses_validation_outcomes: str,
        leakage_status: str,
        evidence_script: str,
        evidence_artifact: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                'record_type': record_type,
                'model_family': model_family,
                'horizon_scope': horizon_scope,
                'fold_scope': fold_scope,
                'item': item,
                'implementation': implementation,
                'fit_scope': fit_scope,
                'validation_construction': validation_construction,
                'uses_outer_test_information': uses_outer_test_information,
                'uses_validation_outcomes': uses_validation_outcomes,
                'leakage_status': leakage_status,
                'evidence_script': evidence_script,
                'evidence_artifact': evidence_artifact,
                'notes': notes,
            }
        )

    # Preprocessing isolation
    add(
        'preprocessing',
        'Persistence',
        'H1/H3/H5',
        'fold1-3',
        'baseline_prediction_rule',
        'persistence prediction uses lagged y (no parameter fitting)',
        'no_fitting_object',
        'none',
        'NO',
        'NO',
        'PASS',
        'src/run_linear_alarm.py + src/run_linear_alarm_v2.py',
        'results/predictions/persistence_H1_preds.csv;results/predictions/persistence_H3_preds.csv;results/predictions/persistence_H5_preds.csv',
        'No scaler/imputer/selector fitted.',
    )

    add(
        'preprocessing',
        'ElasticNet',
        'H1/H3/H5',
        'fold1-3',
        'scaling_and_fit',
        'Pipeline(StandardScaler, ElasticNet) fit inside outer fold loops',
        'outer_training_only_per_fold',
        'no_inner_tail_in_final_fit',
        'NO',
        'NO',
        'PASS_for_scaler_isolation',
        'src/save_fold_predictions.py + src/run_linear_alarm.py + src/run_linear_alarm_v2.py',
        'results/predictions/enet_a0.1_l0.5_H1_preds.csv;results/predictions/enet_a0.1_l0.5_H3_preds.csv;results/predictions/enet_a0.1_l0.5_v2_H5_preds.csv',
        'Scaler fit scope is leakage-safe per fold; outer purge issue handled separately.',
    )

    add(
        'preprocessing',
        'Ridge',
        'H1/H3/H5',
        'fold1-3 (evaluation only)',
        'scaling_and_fit',
        'Pipeline(StandardScaler, Ridge) in main linear trainer',
        'outer_training_only_per_fold_for_cv_rows',
        'no_saved_fold_predictions',
        'NO',
        'NO',
        'PASS_for_scaler_isolation',
        'src/train_main_linear.py',
        'results/metrics/main_linear_metrics.json',
        'Prediction artifacts missing; preprocessing logic itself is fold-local.',
    )

    add(
        'preprocessing',
        'HGBR',
        'H3/H5 (H1 artifact missing)',
        'fold1-3',
        'tree_fit_and_internal_validation',
        'HistGradientBoostingRegressor(... early_stopping=True, validation_fraction=0.1, random_state=42)',
        'outer_training_only_but_internal_random_validation_split',
        'random_holdout_from_outer_train_not_chronological',
        'NO',
        'YES_internal_validation_outcomes_used_for_early_stopping',
        'FAIL_temporal_validation_is_not_blocked',
        'src/train_hgbr_optuna_v2.py + src/save_hgbr_predictions_v2.py',
        'results/hgbr/hgbr_optuna_H3.json;results/hgbr/hgbr_optuna_H5_v2.json',
        'Chronological isolation of inner validation is not enforced for tree training.',
    )

    add(
        'preprocessing',
        'TCN/BCR-TCN',
        'H5_v2_artifact_present',
        'fold1-3',
        'standardization_and_class_weighting',
        'mu/sd and pos_weight computed from sub_train indices only',
        'inner_subtrain_only',
        'chronological_tail_val_frac_0.15_without_horizon_gap',
        'NO',
        'YES_validation_recall_used_for_epoch_selection',
        'FAIL_missing_horizon_purge_between_subtrain_and_validation',
        'src/train_bcr_tcn_v11.py',
        'results/metrics/bcr_tcn_v11_H5_v2_meta.json',
        'Validation tail is chronological but contiguous; boundary overlap exists for H>0.',
    )

    add(
        'preprocessing',
        'HybridRank',
        'H5_runs',
        'fold1-3',
        'rank_normalization_and_weighting',
        'rank01 normalization and weighted score blend per fold/tau',
        'train_rows_for_weight_tuning;test_rows_for_final_scoring',
        'no_explicit_inner_validation_split',
        'NO_for_weight_tuning',
        'YES_training_outcomes_used_for_weight_objective',
        'PASS_for_weight_tuning_scope_but_retrospective_scoring_context',
        'src/make_hybrid_rank_ensemble_v2.py',
        'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/run_config.json;results/hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2/run_config.json',
        'Fold-specific score ranking is computed over full held-out fold distributions.',
    )

    add(
        'preprocessing',
        'point blend',
        'H5_runs',
        'fold1-3',
        'weighted_point_blend',
        'weights selected by minimizing MAE on train folds only',
        'train_rows_only_for_weight_selection',
        'no_explicit_inner_validation_split',
        'NO',
        'NO',
        'PASS_for_fit_scope',
        'src/make_hybrid_rank_ensemble_v2.py',
        'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/models/point_hybrid_weights_H5.csv',
        'Blend depends on base predictions that inherit upstream boundary/selection issues.',
    )

    add(
        'preprocessing',
        'risk-score blend',
        'H5_runs',
        'fold1-3',
        'weighted_rank_blend',
        'weights selected on train folds with policy-specific/guarded objective',
        'train_rows_only_for_weight_selection',
        'no_explicit_inner_validation_split',
        'NO_for_weight_tuning',
        'YES_training_outcomes_used_for_weight_objective',
        'PASS_for_fit_scope_but_retrospective_policy_usage',
        'src/make_hybrid_rank_ensemble_v2.py',
        'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/models/rank_hybrid_weights_H5.csv',
        'Operational benchmarking still uses full held-out fold score ordering.',
    )

    # Model selection and hyperparameter provenance
    add(
        'model_selection',
        'ElasticNet',
        'H1/H3/H5',
        'global_blocked_cv',
        'hyperparameter_selection',
        'grid over alpha x l1_ratio; model variants selected by OOF metrics',
        'full_dataset_blocked_cv_not_nested_against_reporting',
        'TimeSeriesSplit_n_splits_3',
        'YES',
        'YES',
        'FAIL_outer_test_influenced_model_choice',
        'src/train_main_linear.py + src/run_linear_alarm.py + src/paper_make_fig2_point_forecast_generalization.py',
        'results/metrics/main_linear_metrics_H1.json;results/metrics/main_linear_metrics_H3.json;results/metrics/main_linear_metrics_H5.json;results/final_tables/results_discussion_csvs/Table1_point_forecast_summary_H1_H3_H5_main_linear.csv',
        'H5 best model differs by table context (main_linear best ENet vs guarded comparison ENet variant).',
    )

    add(
        'model_selection',
        'Ridge',
        'H1/H3/H5',
        'global_blocked_cv',
        'hyperparameter_selection',
        'alpha grid [0.01,0.1,1,10,100] evaluated via TimeSeriesSplit',
        'full_dataset_blocked_cv',
        'TimeSeriesSplit_n_splits_3',
        'YES',
        'NO_explicit_inner_tail',
        'FAIL_not_nested_and_predictions_missing',
        'src/train_main_linear.py',
        'results/metrics/main_linear_metrics.json',
        'No released ridge prediction files to verify corrected evaluation.',
    )

    add(
        'model_selection',
        'HGBR',
        'H3/H5 (H1 missing)',
        'global_blocked_cv_optuna',
        'hyperparameter_selection',
        'Optuna objective minimizes mean MAE across TimeSeriesSplit folds',
        'full_dataset_blocked_cv_not_nested_against_reporting',
        'TimeSeriesSplit_n_splits_3 + random_validation_fraction_0.1_in_each_fit',
        'YES',
        'YES',
        'FAIL_outer_test_influenced_tuning_and_nonchronological_internal_validation',
        'src/train_hgbr_optuna.py + src/train_hgbr_optuna_v2.py',
        'results/hgbr/hgbr_optuna_H3.json;results/hgbr/hgbr_optuna_H5.json;results/hgbr/hgbr_optuna_H5_v2.json',
        'Best params selected on same blocked folds later reused for headline comparisons.',
    )

    add(
        'model_selection',
        'TCN/BCR-TCN',
        'H5_v2_artifact_present',
        'outer_fold_specific',
        'epoch_selection_and_variant_choice',
        'best epoch selected by validation recall@r=0.05 tau16 within each fold; v11 variant used in claims',
        'inner_validation_tail_per_outer_fold',
        'chronological_tail_val_frac_0.15',
        'NO_for_epoch_selection',
        'YES',
        'UNCLEAR_variant_selection_history_not_fully_logged',
        'src/train_bcr_tcn.py + src/train_bcr_tcn_v11.py + src/verify_claimed_values_reproducibility.py',
        'results/metrics/bcr_tcn_H5_v2_meta.json;results/metrics/bcr_tcn_v11_H5_v2_meta.json',
        'Version transition from bcr_tcn to bcr_tcn_v11 is not fully provenance-locked by pre-registered criterion.',
    )

    add(
        'model_selection',
        'HybridRank',
        'H5_runs',
        'fold1-3',
        'weight_objective_and_run_choice',
        'policy_guarded objective in run config; manuscript lineage uses 20260219 while uncertainty package prefers 20260220',
        'train_rows_for_weight_tuning_but_posthoc_run_selection',
        'no_inner_validation_tail',
        'YES_run_variant_choice',
        'YES',
        'FAIL_posthoc_variant_selection_across_heldout_contexts',
        'src/make_hybrid_rank_ensemble_v2.py + src/paper_make_uncertainty_decision_acs.py + src/verify_claimed_values_reproducibility.py',
        'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/run_config.json;results/hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2/run_config.json',
        'Choice between guarded and normalized H5 contexts is not tied to a single predeclared decision rule.',
    )

    add(
        'model_selection',
        'point blend',
        'H5_runs',
        'fold1-3',
        'weight_selection',
        'weights selected by minimum train MAE across train folds',
        'train_rows_only',
        'no_inner_validation_tail',
        'NO',
        'NO',
        'PASS_for_selection_scope',
        'src/make_hybrid_rank_ensemble_v2.py',
        'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/models/point_hybrid_weights_H5.csv',
        'Still contingent on upstream base-model leakage corrections.',
    )

    add(
        'model_selection',
        'risk-score blend',
        'H5_runs',
        'fold1-3',
        'weight_selection_and_reporting_variant',
        'train-fold objective for weights, plus paper-fixed and guarded reported variants',
        'train_rows_only_for_weights_but_posthoc_reporting_variant_choice',
        'no_inner_validation_tail',
        'YES_reporting_variant_choice',
        'YES',
        'FAIL_posthoc_reporting_variant_selection',
        'src/make_hybrid_rank_ensemble_v2.py + src/paper_make_table3_fig3.py',
        'results/final_tables/results_discussion_csvs/Table3_representative_operating_points_main.csv;results/final_tables/results_discussion_csvs/Table3_alarm_budget_H5_core_operating_points.csv',
        'Multiple valid score definitions were reported across artifacts after held-out inspection.',
    )

    return pd.DataFrame(rows)


def build_alarm_policy_leakage_audit() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(
        operation: str,
        scope: str,
        uses_training_outcomes: str,
        uses_validation_outcomes: str,
        uses_outer_test_outcomes: str,
        uses_full_heldout_distribution: str,
        classification: str,
        evidence_script: str,
        evidence_artifact: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                'operation': operation,
                'scope': scope,
                'uses_training_outcomes': uses_training_outcomes,
                'uses_validation_outcomes': uses_validation_outcomes,
                'uses_outer_test_outcomes': uses_outer_test_outcomes,
                'uses_full_heldout_distribution': uses_full_heldout_distribution,
                'classification': classification,
                'evidence_script': evidence_script,
                'evidence_artifact': evidence_artifact,
                'notes': notes,
            }
        )

    add(
        'risk_score_weighting_hybridrank',
        'fold-wise weight tuning for hybrid score',
        'YES',
        'NO_explicit_val_tail',
        'NO_for_weight_fit',
        'NO',
        'training-only deployable',
        'src/make_hybrid_rank_ensemble_v2.py',
        'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/models/rank_hybrid_weights_H5.csv',
        'Weights optimized on train folds excluding the test fold.',
    )

    add(
        'score_normalization_rank01',
        'per-fold rank normalization of scores before top-k selection',
        'NO',
        'NO',
        'NO_outcomes_used',
        'YES',
        'retrospective test-set benchmark',
        'src/make_hybrid_rank_ensemble_v2.py',
        'results/hybrid_rank_v2_runs/*/predictions/rank_scores_tauwise_H5.csv',
        'Normalization uses full fold score distribution; not equivalent to single-step online deployment.',
    )

    add(
        'tau_threshold_definition',
        'tau in {15,16,17} event definition y_true >= tau',
        'NO',
        'NO',
        'NO',
        'NO',
        'training-only deployable',
        'src/run_linear_alarm.py + src/run_linear_alarm_v2.py',
        'results/hybrid_rank_v2_runs/*/tables/alarm_budget_metrics_H5.csv',
        'Threshold set is constant in code and can be declared ex ante.',
    )

    add(
        'global_top_k_cutoff',
        'retrospective concatenated k = ceil(r*N) over held-out block',
        'NO',
        'NO',
        'NO_outcomes_needed_for_k_but_full_holdout_size_needed',
        'YES',
        'retrospective test-set benchmark',
        'src/run_linear_alarm.py + src/run_linear_alarm_v2.py + src/paper_make_table3_fig3.py',
        'results/final_tables/results_discussion_csvs/Table3_representative_operating_points_main.csv',
        'k depends on full held-out N and assumes retrospective block availability.',
    )

    add(
        'representative_operating_point_selection',
        'selection of Table 3 rows and notes',
        'NO',
        'NO',
        'YES',
        'YES',
        'leakage',
        'src/paper_make_table3_fig3.py + manual final table curation',
        'results/final_tables/results_discussion_csvs/Table3_representative_operating_points_main.csv',
        'Operating points and model variants are selected after examining held-out performance summaries.',
    )

    add(
        'choice_between_H5_runs',
        'guarded 20260219 vs normalized 20260220 lineages',
        'NO',
        'NO',
        'YES_context_dependent',
        'YES',
        'leakage',
        'src/paper_make_uncertainty_decision_acs.py + src/verify_claimed_values_reproducibility.py',
        'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/run_config.json;results/hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2/run_config.json',
        'Different held-out run lineages are used in different manuscript artifacts.',
    )

    add(
        'choice_between_model_variants',
        'ElasticNet variants in H3/H5 and hybrid paper-fixed vs guarded rows',
        'NO',
        'NO',
        'YES',
        'YES',
        'leakage',
        'src/verify_claimed_values_reproducibility.py + final table assembly scripts',
        'results/final_tables/results_discussion_csvs/Table3_representative_operating_points_main.csv',
        'Variant choices vary by operating point and are not pre-registered by a single validation criterion.',
    )

    add(
        'common_date_intersection_merge',
        'fold/date inner merge for multi-model hybrid inputs',
        'NO',
        'NO',
        'NO_outcomes_used',
        'YES_full_holdout_prediction_inventory_used',
        'retrospective test-set benchmark',
        'src/make_hybrid_rank_ensemble_v2.py',
        'results/hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2/hybrid_rank_v2_H5_preds.csv',
        'Date inclusion is constrained by available model predictions; benchmark context is retrospective.',
    )

    add(
        'guard_policy_precision_floor',
        'baseline precision floor at r<=0.05 in policy_guarded objective',
        'YES',
        'NO',
        'NO_for_weight_fit',
        'NO',
        'training-only deployable',
        'src/make_hybrid_rank_ensemble_v2.py',
        'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/models/rank_hybrid_weights_H5.csv',
        'Constraint uses train-fold alarm metrics while tuning fold-specific weights.',
    )

    return pd.DataFrame(rows)


def build_model_rerun_decision() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(model_family: str, horizon: int, prediction_artifact_path: str, artifact_available: str, outer_purge_status: str, inner_purge_status: str, preprocessing_status: str, selection_status: str, date_equality_status: str, recommended_action: str, rationale: str) -> None:
        rows.append(
            {
                'model_family': model_family,
                'horizon': horizon,
                'prediction_artifact_path': prediction_artifact_path,
                'artifact_available': artifact_available,
                'outer_purge_status': outer_purge_status,
                'inner_purge_status': inner_purge_status,
                'preprocessing_status': preprocessing_status,
                'selection_status': selection_status,
                'date_equality_status': date_equality_status,
                'recommended_action': recommended_action,
                'rationale': rationale,
            }
        )

    # Persistence
    add('Persistence', 1, 'results/predictions/persistence_H1_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'N/A', 'PASS_no_fit_object', 'PASS_no_hyperparameter_search', 'H1_dates_match_762', 'recalculate metrics from existing predictions', 'No fitting leakage object, but purge and common-date corrections are needed.')
    add('Persistence', 3, 'results/predictions/persistence_H3_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'N/A', 'PASS_no_fit_object', 'PASS_no_hyperparameter_search', 'uses_759_not_canonical_747', 'recalculate metrics from existing predictions', 'Predictions exist; recompute metrics on corrected purged/canonical dates.')
    add('Persistence', 5, 'results/predictions/persistence_v2_H5_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'N/A', 'PASS_no_fit_object', 'PASS_no_hyperparameter_search', 'v2_747_available_and_legacy_759_exists', 'recalculate metrics from existing predictions', 'Use canonical 747 set for corrected benchmark comparisons.')

    # Ridge
    add('Ridge', 1, 'results/predictions/ridge_*_H1_preds.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'PASS_pipeline_defined_but_artifact_missing', 'FAIL_global_cv_model_selection_not_nested', 'missing', 'missing model artifact', 'No released ridge prediction CSV; rerun fixed published alpha grid on corrected folds is needed before comparison.')
    add('Ridge', 3, 'results/predictions/ridge_*_H3_preds.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'PASS_pipeline_defined_but_artifact_missing', 'FAIL_global_cv_model_selection_not_nested', 'missing', 'missing model artifact', 'No released ridge prediction CSV; rerun fixed published alpha grid on corrected folds is needed before comparison.')
    add('Ridge', 5, 'results/predictions/ridge_*_H5_preds.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'PASS_pipeline_defined_but_artifact_missing', 'FAIL_global_cv_model_selection_not_nested', 'missing', 'missing model artifact', 'No released ridge prediction CSV; rerun fixed published alpha grid on corrected folds is needed before comparison.')

    # ElasticNet
    add('ElasticNet', 1, 'results/predictions/enet_a0.1_l0.5_H1_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'N/A', 'PASS_scaler_fit_per_fold', 'FAIL_selected_from_same_oof_block', 'H1_dates_match_762', 'retrain and revalidate because leakage affected fitting', 'Hyperparameter/model selection used the same blocked folds later reported; nested revalidation required.')
    add('ElasticNet', 3, 'results/predictions/enet_a0.1_l0.5_H3_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'N/A', 'PASS_scaler_fit_per_fold', 'FAIL_selected_from_same_oof_block_and_variant_switch', 'uses_759_not_canonical_747', 'retrain and revalidate because leakage affected fitting', 'OOF-driven variant selection across operating points requires nested revalidation on corrected folds.')
    add('ElasticNet', 5, 'results/predictions/enet_a0.1_l0.5_v2_H5_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'N/A', 'PASS_scaler_fit_per_fold', 'FAIL_selected_from_same_oof_block_and_context_switch', 'v2_747_vs_legacy_759_contexts', 'retrain and revalidate because leakage affected fitting', 'Selection leakage and mixed reporting contexts require full nested revalidation.')

    # HGBR
    add('HGBR', 1, 'results/predictions/hgbr_optuna_H1_preds.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'FAIL_internal_random_validation_not_chronological', 'FAIL_optuna_on_same_blocked_data', 'missing', 'missing model artifact', 'H1 HGBR prediction and config artifacts are missing in workspace.')
    add('HGBR', 3, 'results/predictions/hgbr_optuna_H3_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'FAIL_inner_random_validation_split', 'FAIL_internal_random_validation_not_chronological', 'FAIL_optuna_on_same_blocked_data', 'uses_759_not_canonical_747', 'retrain and revalidate because leakage affected fitting', 'Chronology of inner validation and non-nested tuning both require revalidation.')
    add('HGBR', 5, 'results/predictions/hgbr_optuna_H5_v2_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'FAIL_inner_random_validation_split', 'FAIL_internal_random_validation_not_chronological', 'FAIL_optuna_on_same_blocked_data', 'v2_747_vs_legacy_759_contexts', 'retrain and revalidate because leakage affected fitting', 'Temporal validation and model-selection leakage affect fitted configuration credibility.')

    # TCN/BCR-TCN
    add('TCN/BCR-TCN', 1, 'results/predictions/bcr_tcn_v11_H1_v2_preds.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN_missing_artifact', 'UNKNOWN_missing_artifact', 'missing', 'missing model artifact', 'No H1 BCR-TCN prediction artifact available.')
    add('TCN/BCR-TCN', 3, 'results/predictions/bcr_tcn_v11_H3_v2_preds.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN_missing_artifact', 'UNKNOWN_missing_artifact', 'missing', 'missing model artifact', 'No H3 BCR-TCN prediction artifact available.')
    add('TCN/BCR-TCN', 5, 'results/predictions/bcr_tcn_v11_H5_v2_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'FAIL_no_h_day_gap_before_val_tail', 'FAIL_inner_tail_without_purge', 'UNCLEAR_variant_transition_bcr_tcn_to_v11', 'v2_747_dates', 'rerun using fixed existing hyperparameters on corrected folds', 'Keep current hyperparameters but enforce horizon purge before validation/test boundaries.')

    # HybridRank
    add('HybridRank', 1, 'results/hybrid_rank_v2_runs/*_H1_hybrid_rank_v2', 'NO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN_missing_artifact', 'UNKNOWN_missing_artifact', 'missing', 'missing model artifact', 'No H1 HybridRank run artifact present.')
    add('HybridRank', 3, 'results/hybrid_rank_v2_runs/*_H3_hybrid_rank_v2', 'NO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN_missing_artifact', 'UNKNOWN_missing_artifact', 'missing', 'missing model artifact', 'No H3 HybridRank run artifact present.')
    add('HybridRank', 5, 'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/hybrid_rank_v2_H5_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'N/A_no_inner_tail_constructed', 'PASS_weight_fit_on_train_folds_only', 'FAIL_posthoc_run_and_variant_selection', '747_vs_735_vs_759_context_mismatch', 'rerun using fixed existing hyperparameters on corrected folds', 'Rebuild on corrected base predictions with predeclared single run-selection rule.')

    # point blend
    add('point blend', 1, 'results/hybrid_rank_v2_runs/*_H1_hybrid_rank_v2/predictions/hybrid_point_H1_preds.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN_missing_artifact', 'UNKNOWN_missing_artifact', 'missing', 'missing model artifact', 'No H1 point blend artifact in hybrid run structure.')
    add('point blend', 3, 'results/hybrid_rank_v2_runs/*_H3_hybrid_rank_v2/predictions/hybrid_point_H3_preds.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN_missing_artifact', 'UNKNOWN_missing_artifact', 'missing', 'missing model artifact', 'No H3 point blend artifact in hybrid run structure.')
    add('point blend', 5, 'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/predictions/hybrid_point_H5_preds.csv', 'YES', 'FAIL_no_horizon_purge', 'N/A', 'PASS_weight_fit_on_train_folds_only', 'PASS_weight_selection_on_train_rows', 'depends_on_corrected_base_models', 'rerun using fixed existing hyperparameters on corrected folds', 'Blend can be rerun with same weight-grid logic after corrected base-model reruns.')

    # risk-score blend
    add('risk-score blend', 1, 'results/hybrid_rank_v2_runs/*_H1_hybrid_rank_v2/predictions/rank_scores_tauwise_H1.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN_missing_artifact', 'UNKNOWN_missing_artifact', 'missing', 'missing model artifact', 'No H1 risk-score blend artifact in hybrid run structure.')
    add('risk-score blend', 3, 'results/hybrid_rank_v2_runs/*_H3_hybrid_rank_v2/predictions/rank_scores_tauwise_H3.csv', 'NO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN_missing_artifact', 'UNKNOWN_missing_artifact', 'missing', 'missing model artifact', 'No H3 risk-score blend artifact in hybrid run structure.')
    add('risk-score blend', 5, 'results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/predictions/rank_scores_tauwise_H5.csv', 'YES', 'FAIL_no_horizon_purge', 'N/A_no_inner_tail_constructed', 'PASS_weight_fit_on_train_rows', 'FAIL_reporting_variant_selection_and_retro_global_k', 'retrospective_global_budget_context', 'rerun using fixed existing hyperparameters on corrected folds', 'Keep same weight objective family but re-evaluate with corrected folds and deployable policy calibration.',)

    return pd.DataFrame(rows)


def build_markdowns(
    fold_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    prep_df: pd.DataFrame,
    alarm_df: pd.DataFrame,
    rerun_df: pd.DataFrame,
) -> None:
    # Fold boundary markdown
    outer = fold_df[fold_df['record_type'] == 'outer_train_test_purge'].copy()
    inner_tcn = fold_df[(fold_df['record_type'] == 'inner_train_validation_purge') & (fold_df['model_family'] == 'TCN/BCR-TCN')].copy()
    inner_hg = fold_df[(fold_df['record_type'] == 'inner_train_validation_purge') & (fold_df['model_family'] == 'HGBR')].copy()

    lines = []
    lines.append('# Fold Boundary and Purge Audit')
    lines.append('')
    lines.append('## Part A: Canonical Temporal Index')
    lines.append('- Canonical split source: results/tables/blocked_cv_split_summary_all_horizons.csv')
    lines.append('- Assignment matrices used: blocked_cv_assignment_matrix_H1.csv, blocked_cv_assignment_matrix_H3.csv, blocked_cv_assignment_matrix_H5.csv')
    lines.append('- Target shift verified in feature builders as y = TNout.shift(-h), so target_date = feature_date + horizon.')
    lines.append('')
    lines.append('## Part B: Outer Train-Test Boundary')
    for h in HORIZONS:
        sub = outer[outer['horizon'] == h]
        fail_n = int((sub['boundary_result'] == 'FAIL').sum())
        avg_remove = float(pd.to_numeric(sub['rows_to_remove'], errors='coerce').mean())
        lines.append(f'- H={h}: FAIL folds={fail_n}/3, mean rows requiring purge={avg_remove:.1f}.')
    lines.append('- Existing predictions were generated before any explicit horizon purge-gap was applied.')
    lines.append('')
    lines.append('## Part C: Inner Train-Validation Boundary')
    lines.append('- TCN/BCR-TCN uses a chronological 15% validation tail, but no H-day purge between subtrain and validation.')
    for h in HORIZONS:
        sub = inner_tcn[inner_tcn['horizon'] == h]
        fail_n = int((sub['boundary_result'] == 'FAIL').sum())
        max_overlap = int(pd.to_numeric(sub['overlap_target_dates_count'], errors='coerce').max())
        lines.append(f'- TCN H={h}: FAIL folds={fail_n}/3, max overlap target dates={max_overlap}.')
    lines.append('- HGBR uses internal validation_fraction=0.1 random holdout (reconstructed), not chronological tail validation.')
    lines.append('')
    lines.append('## Notes')
    lines.append('- Full row-level evidence, offending rows, and purge-day calculations are in fold_boundary_purge_audit.csv.')

    (OUT / 'fold_boundary_purge_audit.md').write_text('\n'.join(lines), encoding='utf-8')

    # Feature leakage markdown
    pass_n = int((feat_df['leakage_status'].str.startswith('PASS')).sum())
    risk_n = int((~feat_df['leakage_status'].str.startswith('PASS')).sum())
    lines = []
    lines.append('# Feature Leakage Audit')
    lines.append('')
    lines.append(f'- PASS-like findings: {pass_n}')
    lines.append(f'- Non-pass findings: {risk_n}')
    lines.append('')
    lines.append('## Key Findings')
    lines.append('- TNout lag and rolling features explicitly apply shift(1) before rolling; current-day TNout is not used in those windows.')
    lines.append('- No forward-fill/interpolation operations were detected in primary feature-builder scripts.')
    lines.append('- Several process/weather rolling features include current-day (t) measurements; this is not future leakage but requires clear forecast issue-time assumptions.')
    lines.append('- Feature windows are computed globally before split, but all audited rolling formulas are backward-looking only.')

    (OUT / 'feature_leakage_audit.md').write_text('\n'.join(lines), encoding='utf-8')

    # Preprocessing/model-selection markdown
    prep_fail = int((prep_df['leakage_status'].str.startswith('FAIL')).sum())
    prep_unclear = int((prep_df['leakage_status'].str.startswith('UNCLEAR')).sum())
    prep_pass = int((prep_df['leakage_status'].str.startswith('PASS')).sum())

    lines = []
    lines.append('# Preprocessing and Model-Selection Audit')
    lines.append('')
    lines.append(f'- PASS rows: {prep_pass}')
    lines.append(f'- FAIL rows: {prep_fail}')
    lines.append(f'- UNCLEAR rows: {prep_unclear}')
    lines.append('')
    lines.append('## Preprocessing Isolation')
    lines.append('- ElasticNet/Ridge scaler fitting is fold-local in pipeline code.')
    lines.append('- HGBR uses random internal validation (validation_fraction=0.1), not chronological blocked validation.')
    lines.append('- TCN standardization uses subtrain-only statistics, but no horizon purge before validation tail.')
    lines.append('')
    lines.append('## Hyperparameter and Model Selection')
    lines.append('- ElasticNet and Ridge variant choice is based on the same blocked folds later used for reported performance (not nested).')
    lines.append('- HGBR Optuna tuning also uses the same blocked data context for selection and reporting.')
    lines.append('- HybridRank manuscript lineage mixes guarded and normalized H5 run contexts, indicating post-hoc variant selection risk.')

    (OUT / 'preprocessing_model_selection_audit.md').write_text('\n'.join(lines), encoding='utf-8')

    # Alarm policy markdown
    class_counts = alarm_df['classification'].value_counts().to_dict()
    lines = []
    lines.append('# Alarm Policy Leakage Audit')
    lines.append('')
    for k in [
        'training-only deployable',
        'validation-calibrated deployable',
        'retrospective test-set benchmark',
        'leakage',
        'unclear',
    ]:
        lines.append(f"- {k}: {int(class_counts.get(k, 0))}")
    lines.append('')
    lines.append('## Classification Notes')
    lines.append('- Global k = ceil(r*N) over complete held-out blocks is retrospective benchmarking, not an online deployable rule.')
    lines.append('- Rank normalization over full fold score distributions is retrospective block scoring.')
    lines.append('- Table3 representative operating-point and run-variant choices are classified as leakage due held-out-informed selection.')

    (OUT / 'alarm_policy_leakage_audit.md').write_text('\n'.join(lines), encoding='utf-8')

    # Final decision markdown with required structure
    outer_fail = outer[outer['boundary_result'] == 'FAIL']
    inner_tcn_fail = inner_tcn[inner_tcn['boundary_result'] == 'FAIL']
    inner_hg_fail = inner_hg[inner_hg['boundary_result'] == 'FAIL']

    retain = rerun_df[rerun_df['recommended_action'] == 'recalculate metrics from existing predictions']
    fixed_rerun = rerun_df[rerun_df['recommended_action'] == 'rerun using fixed existing hyperparameters on corrected folds']
    full_reval = rerun_df[rerun_df['recommended_action'] == 'retrain and revalidate because leakage affected fitting']
    missing = rerun_df[rerun_df['recommended_action'] == 'missing model artifact']

    # Cross-model date equality section inputs
    mda = pd.read_csv(ROOT / 'revision_2026' / '01_original_audit' / 'model_date_alignment.csv')
    mda_h1 = mda[(mda['record_type'] == 'max_common_primary') & (mda['horizon'] == 1)]
    mda_h3 = mda[(mda['record_type'] == 'max_common_primary') & (mda['horizon'] == 3)]
    mda_h5 = mda[(mda['record_type'] == 'max_common_primary') & (mda['horizon'] == 5)]

    lines = []
    lines.append('# Temporal Leakage Audit Decision')
    lines.append('')
    lines.append('## 1. Canonical Temporal Index')
    lines.append('- Canonical fold/date sources are blocked_cv_split_summary_all_horizons.csv and blocked_cv_assignment_matrix_H1/H3/H5.csv.')
    lines.append('- Feature-to-target mapping is confirmed as target_date = feature_date + horizon from y = TNout.shift(-h) in feature builders.')
    lines.append('- Inner validation tails are reconstructed at 15% of outer training length for chronological deep-model validation checks.')
    lines.append('')
    lines.append('## 2. Outer Train–Test Purge')
    lines.append(f'- Outer boundary failures: {len(outer_fail)}/9 fold-horizon combinations.')
    lines.append('- H=1 requires 1-row purge at each fold boundary; H=3 requires 3-row purge; H=5 requires 5-row purge.')
    lines.append('- Existing prediction artifacts were generated before applying these horizon purge removals.')
    lines.append('')
    lines.append('## 3. Inner Train–Validation Purge')
    lines.append(f'- TCN/BCR-TCN inner boundary failures: {len(inner_tcn_fail)}/9 (chronological tail present, horizon purge absent).')
    lines.append(f'- HGBR inner boundary failures (reconstructed random validation splits): {len(inner_hg_fail)}/9.')
    lines.append('- ElasticNet/Ridge/HybridRank blends have no explicit chronological validation tail in final fold-fit scripts.')
    lines.append('')
    lines.append('## 4. Feature Availability')
    lines.append('- TNout lag/rolling memory features are backward-looking and do not include future TNout values.')
    lines.append('- Process and weather rolling features include current-day values (t) but not future values; issue-time availability must be explicitly declared.')
    lines.append('- No forward-fill/interpolation feature operations were found in primary feature builders.')
    lines.append('')
    lines.append('## 5. Preprocessing Isolation')
    lines.append('- ElasticNet/Ridge scaling is fit on fold-local training data only.')
    lines.append('- HGBR uses random internal validation fraction during fitting, violating strict chronological validation isolation.')
    lines.append('- TCN normalization and class-weighting are subtrain-local, but validation-tail purge is missing.')
    lines.append('')
    lines.append('## 6. Hyperparameter and Model Selection')
    lines.append('- ElasticNet and Ridge variant choice relies on blocked OOF metrics from the same data context used for reporting, not nested revalidation.')
    lines.append('- HGBR Optuna tuning is similarly non-nested for final comparative reporting.')
    lines.append('- Table 3 mixes ElasticNet variants by operating point and mixes H5 run contexts, indicating post-hoc selection behavior.')
    lines.append('')
    lines.append('## 7. Retrospective Alarm-Budget Classification')
    lines.append('- k = ceil(r*N) with global held-out N is classified as retrospective fixed-budget benchmarking.')
    lines.append('- Rank normalization over complete fold score distributions is retrospective block scoring.')
    lines.append('- Risk-score weighting itself is train-fold-only, but representative reporting choices are held-out-informed.')
    lines.append('')
    lines.append('## 8. Cross-Model Date Equality')
    if not mda_h1.empty:
        lines.append(f"- H1 max common-date set: {int(mda_h1['unique_dates'].iloc[0])} (families incomplete: no H1 BCR-TCN/HGBR artifacts).")
    if not mda_h3.empty:
        lines.append(f"- H3 max common-date set among available families: {int(mda_h3['unique_dates'].iloc[0])}, but this is 759 and differs from canonical 747 split.")
    if not mda_h5.empty:
        lines.append(f"- H5 max common-date set across four families: {int(mda_h5['unique_dates'].iloc[0])}; manuscript artifacts still mix 747, 759, and 735 contexts.")
    lines.append('')
    lines.append('## 9. Confirmed Leakage Risks')
    lines.append('- Missing horizon purge at outer train-test boundaries for all horizons.')
    lines.append('- Missing horizon purge at chronological inner subtrain-validation boundary for TCN/BCR-TCN.')
    lines.append('- Non-chronological internal validation in HGBR fitting.')
    lines.append('- Held-out-informed model/run/operating-point selection in final manuscript lineage.')
    lines.append('')
    lines.append('## 10. Models and Horizons That Can Be Retained')
    if retain.empty:
        lines.append('- None are fully retainable unchanged under strict leakage-safe pipeline criteria; persistence can be reused only for recalculated metrics on corrected dates.')
    else:
        for _, r in retain.iterrows():
            lines.append(f"- {r['model_family']} H{int(r['horizon'])}: {r['recommended_action']}.")
    lines.append('')
    lines.append('## 11. Models and Horizons That Must Be Rerun')
    for _, r in fixed_rerun.iterrows():
        lines.append(f"- {r['model_family']} H{int(r['horizon'])}: rerun fixed existing hyperparameters on corrected folds.")
    for _, r in full_reval.iterrows():
        lines.append(f"- {r['model_family']} H{int(r['horizon'])}: full nested revalidation required.")
    if not missing.empty:
        lines.append('- Missing artifacts requiring regeneration before fair comparison:')
        for _, r in missing.iterrows():
            lines.append(f"  - {r['model_family']} H{int(r['horizon'])}")
    lines.append('')
    lines.append('## 12. Corrected Evaluation Requirements')
    lines.append('- Enforce H-day purge between outer training targets and test feature windows for every horizon/fold.')
    lines.append('- For models with inner validation, enforce H-day purge between subtrain targets and validation feature windows.')
    lines.append('- Re-run model selection in nested fashion when reporting comparative performance tables.')
    lines.append('- Separate retrospective fixed-budget benchmark reporting from deployable online policy claims.')
    lines.append('')
    lines.append('## 13. Final Decision')
    lines.append('D. Hyperparameter or model selection used held-out test information; affected configurations require nested revalidation.')

    (OUT / 'temporal_leakage_audit_decision.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    fold_df = build_fold_boundary_audit()
    feat_df = build_feature_leakage_audit()
    prep_df = build_preprocessing_model_selection_audit()
    alarm_df = build_alarm_policy_leakage_audit()
    rerun_df = build_model_rerun_decision()

    fold_df.to_csv(OUT / 'fold_boundary_purge_audit.csv', index=False)
    feat_df.to_csv(OUT / 'feature_leakage_audit.csv', index=False)
    prep_df.to_csv(OUT / 'preprocessing_model_selection_audit.csv', index=False)
    alarm_df.to_csv(OUT / 'alarm_policy_leakage_audit.csv', index=False)
    rerun_df.to_csv(OUT / 'model_rerun_decision.csv', index=False)

    build_markdowns(fold_df, feat_df, prep_df, alarm_df, rerun_df)

    print('Wrote:', OUT / 'fold_boundary_purge_audit.csv')
    print('Wrote:', OUT / 'fold_boundary_purge_audit.md')
    print('Wrote:', OUT / 'feature_leakage_audit.csv')
    print('Wrote:', OUT / 'feature_leakage_audit.md')
    print('Wrote:', OUT / 'preprocessing_model_selection_audit.csv')
    print('Wrote:', OUT / 'preprocessing_model_selection_audit.md')
    print('Wrote:', OUT / 'alarm_policy_leakage_audit.csv')
    print('Wrote:', OUT / 'alarm_policy_leakage_audit.md')
    print('Wrote:', OUT / 'model_rerun_decision.csv')
    print('Wrote:', OUT / 'temporal_leakage_audit_decision.md')


if __name__ == '__main__':
    main()
