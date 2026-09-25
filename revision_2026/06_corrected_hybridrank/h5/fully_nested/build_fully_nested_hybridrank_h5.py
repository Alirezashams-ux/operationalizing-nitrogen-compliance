from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
AUTHORIZED_REL_DIR = "revision_2026/06_corrected_hybridrank/h5/fully_nested"
EXPECTED_BRANCH = "controlled-reruns-v1"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CONTROLLED_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions" / "h5_cross_model"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

CANONICAL_CHECKSUMS_PATH = CANONICAL_DIR / "canonical_h5_checksums.sha256"
CANONICAL_REPORT_PATH = CANONICAL_DIR / "canonical_h5_assembly_report.md"
CANONICAL_MANIFEST_PATH = CANONICAL_DIR / "canonical_h5_assembly_manifest.json"
CANONICAL_WIDE_PATH = CANONICAL_DIR / "canonical_h5_predictions_wide.csv"

ENET_SELECTION_PLAN_PATH = CONTROLLED_DIR / "elasticnet" / "elasticnet_selection_plan.json"
HGBR_SELECTION_PLAN_PATH = CONTROLLED_DIR / "hgbr" / "hgbr_selection_plan.json"
BCR_FIXED_CONFIG_PATH = CONTROLLED_DIR / "bcr_tcn_v11_h5" / "bcr_tcn_v11_fixed_configuration.json"

PERSISTENCE_COMPLETION_REPORT_PATH = CONTROLLED_DIR / "persistence" / "persistence_completion_report.md"
ENET_COMPLETION_REPORT_PATH = CONTROLLED_DIR / "elasticnet" / "elasticnet_completion_report.md"
HGBR_COMPLETION_REPORT_PATH = CONTROLLED_DIR / "hgbr" / "hgbr_completion_report.md"
BCR_COMPLETION_REPORT_PATH = CONTROLLED_DIR / "bcr_tcn_v11_h5" / "bcr_tcn_v11_h5_completion_report.md"

PROTECTED_DIRS = [PROTOCOL_DIR, CONTROLLED_DIR, CANONICAL_DIR]

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
NESTED_ASSIGNMENT_PATH = OUT_DIR / "fully_nested_h5_split_assignment.csv"
NESTED_SUMMARY_PATH = OUT_DIR / "fully_nested_h5_split_summary.csv"
COMPONENT_SELECTION_PATH = OUT_DIR / "component_selection_results.csv"
BCR_GATE_PATH = OUT_DIR / "bcr_component_validation_gate.csv"
FOLD1_TAU16_DIAGNOSTIC_PATH = OUT_DIR / "fold1_component_validation_tau16_diagnostic.csv"
HV_COMPONENT_PRED_PATH = OUT_DIR / "hybrid_validation_component_predictions.csv"
HV_RANK_PATH = OUT_DIR / "hybrid_validation_rank_scores.csv"
HV_SUPPORT_PATH = OUT_DIR / "hybrid_validation_event_support.csv"
WEIGHTS_PATH = OUT_DIR / "fully_nested_hybridrank_weights.csv"
OUTER_COMPONENT_PRED_PATH = OUT_DIR / "fully_nested_outer_test_component_predictions.csv"
OUTER_SCORES_PATH = OUT_DIR / "fully_nested_hybridrank_outer_test_scores.csv"
ISOLATION_AUDIT_PATH = OUT_DIR / "fully_nested_hybridrank_isolation_audit.csv"
MANIFEST_PATH = OUT_DIR / "fully_nested_hybridrank_h5_manifest.json"
COMPLETION_REPORT_PATH = OUT_DIR / "fully_nested_hybridrank_h5_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "fully_nested_hybridrank_h5_checksums.sha256"

H = 5
OUTER_FOLDS = (1, 2, 3)
TAUS = (15.0, 16.0, 17.0)
BUDGETS = (0.05, 0.10)
COMPONENT_ORDER = ("BCR-TCN", "ElasticNet", "Persistence", "HGBR")

EMBARGO_DAYS = 5
HYBRID_VALIDATION_FRACTION = 0.15
COMPONENT_VALIDATION_FRACTION = 0.15

SUPPORT_MIN_EVENTS = 5
SUPPORT_MIN_ALARM_SLOTS = 3

FIXED_WEIGHTS = {
    "BCR-TCN": 0.50,
    "ElasticNet": 0.25,
    "Persistence": 0.25,
    "HGBR": 0.00,
}

MODEL_NAME = "HybridRank"
MODEL_VERSION = "fully_nested_h5_v1"
SEED = 42
DEVICE = "cpu"
TIE_TOL = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_output(args: List[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    return mapped.fillna(False).astype(bool)


def to_date_str(x: Any) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_inside_authorized(path: Path) -> None:
    rp = path.resolve()
    auth = (ROOT / AUTHORIZED_REL_DIR).resolve()
    if str(rp).startswith(str(auth)):
        return
    raise RuntimeError("Attempted write outside authorized directory: {}".format(rp))


def to_csv_with_dates(df: pd.DataFrame, path: Path, date_cols: List[str]) -> None:
    ensure_inside_authorized(path)
    out = df.copy()
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def parse_checksum_manifest(path: Path) -> List[Tuple[str, str]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: List[Tuple[str, str]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            raise RuntimeError("Malformed checksum line: {}".format(line))
        out.append((parts[0], parts[-1]))
    return out


def verify_checksum_manifest(path: Path, base_dir: Path) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for expected, rel_name in parse_checksum_manifest(path):
        target = base_dir / rel_name
        if not target.exists():
            raise RuntimeError("Checksum target missing: {}".format(target))
        observed = sha256_file(target)
        ok = observed == expected
        checks[rel_name] = {
            "expected": expected,
            "observed": observed,
            "pass": bool(ok),
        }
        if not ok:
            raise RuntimeError(
                "Checksum mismatch for {}: expected {}, observed {}".format(rel_name, expected, observed)
            )
    return checks


def parse_git_status_paths() -> List[str]:
    raw = git_output(["git", "status", "--porcelain"]) if (ROOT / ".git").exists() else ""
    paths: List[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        payload = line[3:]
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        paths.append(payload.strip())
    return paths


def is_authorized_git_status_path(path: str, authorized_rel_dir: str) -> bool:
    p = str(path).strip().rstrip("/")
    a = str(authorized_rel_dir).strip().rstrip("/")
    if not p:
        return False
    # Accept both leaf-file paths under authorized dir and parent-dir aggregate rows.
    return p.startswith(a) or a.startswith(p)


def snapshot_tree_checksums(dirs: List[Path]) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            snap[rel] = sha256_file(p)
    return snap


def infer_decision_from_report(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")
    m = re.search(r"Decision:\s*([ABCD])", txt)
    if m:
        return m.group(1)

    lines = [ln.strip() for ln in txt.splitlines()]
    final_block_idx = -1
    for i, ln in enumerate(lines):
        if "FINAL DECISION" in ln.upper() or "FINAL DECISION" in ln:
            final_block_idx = i
            break
    if final_block_idx >= 0:
        for ln in lines[final_block_idx + 1 : final_block_idx + 20]:
            if re.match(r"^[ABCD]\.\s", ln):
                return ln[0]

    for ln in lines[-30:]:
        if re.match(r"^[ABCD]\.\s", ln):
            return ln[0]

    raise RuntimeError("Could not infer final decision from report: {}".format(path))


def load_lock(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_verify_dataset(lock: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    dataset_path = Path(lock["absolute_path"])
    if not dataset_path.exists():
        raise RuntimeError("Locked dataset does not exist: {}".format(dataset_path))

    observed_sha = sha256_file(dataset_path)
    expected_sha = lock["sha256"]
    if observed_sha != expected_sha:
        raise RuntimeError(
            "Dataset checksum mismatch: expected {}, observed {}".format(expected_sha, observed_sha)
        )

    df = pd.read_csv(dataset_path)
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    date_col = lock["date_column"]
    target_col = lock["target_column"]
    if date_col not in df.columns:
        raise RuntimeError("Dataset missing date column: {}".format(date_col))
    if target_col not in df.columns:
        raise RuntimeError("Dataset missing target column: {}".format(target_col))

    df[date_col] = normalize_date_col(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if len(df) != int(lock["row_count"]):
        raise RuntimeError("Dataset row_count mismatch")
    if int(df[date_col].nunique()) != int(lock["unique_date_count"]):
        raise RuntimeError("Dataset unique_date_count mismatch")
    if int(df.duplicated(subset=[date_col]).sum()) != int(lock["duplicate_date_count"]):
        raise RuntimeError("Dataset duplicate_date_count mismatch")

    min_date = to_date_str(df[date_col].min())
    max_date = to_date_str(df[date_col].max())
    if min_date != lock["minimum_date"] or max_date != lock["maximum_date"]:
        raise RuntimeError("Dataset date range mismatch")

    all_days = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    missing_days = int(len(all_days) - int(df[date_col].nunique()))
    if missing_days != int(lock["missing_calendar_date_count"]):
        raise RuntimeError("Dataset missing_calendar_date_count mismatch")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if df[target_col].isna().any():
        raise RuntimeError("Target column contains NaN values")

    return df, observed_sha


def load_split_assignment(path: Path) -> pd.DataFrame:
    split = pd.read_csv(path)
    required = {
        "horizon",
        "outer_fold",
        "feature_date",
        "target_date",
        "outer_role",
        "inner_fold",
        "inner_role",
        "original_row_index",
        "purged_outer_boundary",
        "purged_inner_boundary",
        "purge_reason",
    }
    missing = required.difference(split.columns)
    if missing:
        raise RuntimeError("Split assignment missing columns: {}".format(sorted(missing)))

    split["feature_date"] = normalize_date_col(split["feature_date"])
    split["target_date"] = normalize_date_col(split["target_date"])
    split["purged_outer_boundary"] = as_bool(split["purged_outer_boundary"])
    split["purged_inner_boundary"] = as_bool(split["purged_inner_boundary"])

    if split["feature_date"].isna().any() or split["target_date"].isna().any():
        raise RuntimeError("Split assignment has unparsable dates")

    return split


def deterministic_tail_count(n_rows: int, frac: float) -> int:
    if n_rows <= 1:
        return 1
    n = int(math.floor(float(n_rows) * float(frac)))
    if n < 1:
        n = 1
    if n >= n_rows:
        n = n_rows - 1
    return int(n)


def add_assignment_rows(
    rows: List[Dict[str, Any]],
    frame: pd.DataFrame,
    outer_fold: int,
    nested_role: str,
    included: bool,
    exclusion_reason: str,
    component_validation_start: pd.Timestamp,
    hybrid_validation_start: pd.Timestamp,
    outer_test_start: pd.Timestamp,
    dataset_sha: str,
    split_sha: str,
    nested_split_id: str,
) -> None:
    if frame.empty:
        return
    for r in frame.itertuples(index=False):
        rows.append(
            {
                "outer_fold": int(outer_fold),
                "feature_date": r.feature_date,
                "target_date": r.target_date,
                "nested_role": nested_role,
                "included": bool(included),
                "exclusion_reason": exclusion_reason,
                "component_validation_start": component_validation_start,
                "hybrid_validation_start": hybrid_validation_start,
                "outer_test_start": outer_test_start,
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "nested_split_id": nested_split_id,
            }
        )


def build_nested_split(
    split_h5: pd.DataFrame,
    dataset_sha: str,
    split_sha: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
    assignment_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    fold_data: Dict[int, Dict[str, Any]] = {}

    for fold in OUTER_FOLDS:
        sf = split_h5[split_h5["outer_fold"] == fold].copy().sort_values("feature_date")
        if sf.empty:
            raise RuntimeError("No split rows found for H5 fold {}".format(fold))

        outer_test = sf[sf["outer_role"] == "outer_test"].copy().sort_values("feature_date")
        outer_train_all = sf[sf["outer_role"] == "outer_train"].copy().sort_values("feature_date")
        outer_embargo_excluded = outer_train_all[outer_train_all["purged_outer_boundary"]].copy()
        outer_train = outer_train_all[~outer_train_all["purged_outer_boundary"]].copy().sort_values("feature_date")

        if outer_test.empty or outer_train.empty:
            raise RuntimeError("Missing outer_train or outer_test for fold {}".format(fold))

        hv_n = deterministic_tail_count(len(outer_train), HYBRID_VALIDATION_FRACTION)
        hybrid_validation = outer_train.iloc[len(outer_train) - hv_n :].copy()
        hybrid_validation_start = hybrid_validation["feature_date"].min()

        before_hybrid = outer_train[outer_train["feature_date"] < hybrid_validation_start].copy()
        component_development = before_hybrid[
            before_hybrid["target_date"] < hybrid_validation_start
        ].copy()
        hybrid_embargo_excluded = before_hybrid[
            ~(before_hybrid["target_date"] < hybrid_validation_start)
        ].copy()

        if component_development.empty:
            raise RuntimeError("Component-development segment is empty for fold {}".format(fold))

        cv_n = deterministic_tail_count(len(component_development), COMPONENT_VALIDATION_FRACTION)
        component_validation = component_development.iloc[len(component_development) - cv_n :].copy()
        component_validation_start = component_validation["feature_date"].min()

        subtrain_before_embargo = component_development[
            component_development["feature_date"] < component_validation_start
        ].copy()
        component_subtrain = subtrain_before_embargo[
            subtrain_before_embargo["target_date"] < component_validation_start
        ].copy()
        inner_embargo_excluded = subtrain_before_embargo[
            ~(subtrain_before_embargo["target_date"] < component_validation_start)
        ].copy()

        if component_subtrain.empty:
            raise RuntimeError("Component-subtrain segment is empty for fold {}".format(fold))
        if component_validation.empty:
            raise RuntimeError("Component-validation segment is empty for fold {}".format(fold))
        if hybrid_validation.empty:
            raise RuntimeError("Hybrid-validation segment is empty for fold {}".format(fold))

        outer_test_start = outer_test["feature_date"].min()

        inner_pass = bool(component_subtrain["target_date"].max() < component_validation["feature_date"].min())
        hybrid_pass = bool(component_development["target_date"].max() < hybrid_validation["feature_date"].min())
        outer_pass = bool(outer_train["target_date"].max() < outer_test["feature_date"].min())
        future_fold_pass = bool(outer_train["feature_date"].max() < outer_test["feature_date"].min())

        if not inner_pass or not hybrid_pass or not outer_pass:
            raise RuntimeError("Embargo boundary violation in fold {}".format(fold))

        nested_split_id = "h5_fold{}_fully_nested_v1".format(fold)

        add_assignment_rows(
            assignment_rows,
            component_subtrain,
            fold,
            "component_subtrain",
            True,
            "",
            component_validation_start,
            hybrid_validation_start,
            outer_test_start,
            dataset_sha,
            split_sha,
            nested_split_id,
        )
        add_assignment_rows(
            assignment_rows,
            inner_embargo_excluded,
            fold,
            "inner_embargo_excluded",
            False,
            "target_date_not_before_component_validation_start",
            component_validation_start,
            hybrid_validation_start,
            outer_test_start,
            dataset_sha,
            split_sha,
            nested_split_id,
        )
        add_assignment_rows(
            assignment_rows,
            component_validation,
            fold,
            "component_validation",
            True,
            "",
            component_validation_start,
            hybrid_validation_start,
            outer_test_start,
            dataset_sha,
            split_sha,
            nested_split_id,
        )
        add_assignment_rows(
            assignment_rows,
            hybrid_embargo_excluded,
            fold,
            "hybrid_embargo_excluded",
            False,
            "target_date_not_before_hybrid_validation_start",
            component_validation_start,
            hybrid_validation_start,
            outer_test_start,
            dataset_sha,
            split_sha,
            nested_split_id,
        )
        add_assignment_rows(
            assignment_rows,
            hybrid_validation,
            fold,
            "hybrid_validation",
            True,
            "",
            component_validation_start,
            hybrid_validation_start,
            outer_test_start,
            dataset_sha,
            split_sha,
            nested_split_id,
        )
        add_assignment_rows(
            assignment_rows,
            outer_embargo_excluded,
            fold,
            "outer_embargo_excluded",
            False,
            "outer_target_date_not_before_outer_test_start",
            component_validation_start,
            hybrid_validation_start,
            outer_test_start,
            dataset_sha,
            split_sha,
            nested_split_id,
        )
        add_assignment_rows(
            assignment_rows,
            outer_test,
            fold,
            "outer_test",
            True,
            "",
            component_validation_start,
            hybrid_validation_start,
            outer_test_start,
            dataset_sha,
            split_sha,
            nested_split_id,
        )

        total_expected = len(sf)
        total_assigned = (
            len(component_subtrain)
            + len(inner_embargo_excluded)
            + len(component_validation)
            + len(hybrid_embargo_excluded)
            + len(hybrid_validation)
            + len(outer_embargo_excluded)
            + len(outer_test)
        )
        if total_assigned != total_expected:
            raise RuntimeError(
                "Nested assignment count mismatch for fold {}: assigned {}, expected {}".format(
                    fold, total_assigned, total_expected
                )
            )

        summary_rows.append(
            {
                "outer_fold": int(fold),
                "outer_train_N": int(len(outer_train)),
                "component_subtrain_N_before_embargo": int(len(subtrain_before_embargo)),
                "inner_embargo_excluded_N": int(len(inner_embargo_excluded)),
                "component_subtrain_N_after_embargo": int(len(component_subtrain)),
                "component_validation_N": int(len(component_validation)),
                "hybrid_embargo_excluded_N": int(len(hybrid_embargo_excluded)),
                "hybrid_validation_N": int(len(hybrid_validation)),
                "outer_test_N": int(len(outer_test)),
                "component_subtrain_start": component_subtrain["feature_date"].min(),
                "component_subtrain_end": component_subtrain["feature_date"].max(),
                "component_validation_start": component_validation["feature_date"].min(),
                "component_validation_end": component_validation["feature_date"].max(),
                "hybrid_validation_start": hybrid_validation["feature_date"].min(),
                "hybrid_validation_end": hybrid_validation["feature_date"].max(),
                "outer_test_start": outer_test["feature_date"].min(),
                "outer_test_end": outer_test["feature_date"].max(),
                "inner_target_separation_pass": bool(inner_pass),
                "hybrid_target_separation_pass": bool(hybrid_pass),
                "outer_target_separation_pass": bool(outer_pass),
                "future_fold_exclusion_pass": bool(future_fold_pass),
            }
        )

        fold_data[fold] = {
            "outer_train": outer_train,
            "outer_test": outer_test,
            "component_development": component_development,
            "component_subtrain": component_subtrain,
            "component_validation": component_validation,
            "hybrid_validation": hybrid_validation,
            "inner_embargo_excluded": inner_embargo_excluded,
            "hybrid_embargo_excluded": hybrid_embargo_excluded,
            "outer_embargo_excluded": outer_embargo_excluded,
            "component_validation_start": component_validation_start,
            "hybrid_validation_start": hybrid_validation_start,
            "outer_test_start": outer_test_start,
            "nested_split_id": nested_split_id,
        }

    assignment_df = pd.DataFrame(assignment_rows).sort_values(
        ["outer_fold", "feature_date", "target_date", "nested_role"]
    )
    summary_df = pd.DataFrame(summary_rows).sort_values("outer_fold")

    return assignment_df, summary_df, fold_data


def date_to_target_map(dataset_df: pd.DataFrame, date_col: str, target_col: str) -> Dict[pd.Timestamp, float]:
    mapping: Dict[pd.Timestamp, float] = {}
    for r in dataset_df[[date_col, target_col]].itertuples(index=False):
        d = pd.Timestamp(r[0]).normalize()
        if d in mapping:
            raise RuntimeError("Duplicate date in canonical dataset: {}".format(d))
        mapping[d] = float(r[1])
    return mapping


def build_fold_tau_support_diagnostic(
    fold: int,
    fold_info: Dict[str, Any],
    date_to_target: Dict[pd.Timestamp, float],
    dataset_sha: str,
    split_sha: str,
    run_id: str,
    tau: float,
) -> pd.DataFrame:
    partition_specs = [
        ("component_boundary", "component_subtrain", True, "", fold_info["component_subtrain"]),
        (
            "component_boundary",
            "inner_embargo_excluded",
            False,
            "target_date_not_before_component_validation_start",
            fold_info["inner_embargo_excluded"],
        ),
        ("component_boundary", "component_validation", True, "", fold_info["component_validation"]),
        (
            "hybrid_boundary",
            "hybrid_embargo_excluded",
            False,
            "target_date_not_before_hybrid_validation_start",
            fold_info["hybrid_embargo_excluded"],
        ),
        ("hybrid_boundary", "hybrid_validation", True, "", fold_info["hybrid_validation"]),
        (
            "outer_boundary",
            "outer_embargo_excluded",
            False,
            "outer_target_date_not_before_outer_test_start",
            fold_info["outer_embargo_excluded"],
        ),
        ("outer_boundary", "outer_test", True, "", fold_info["outer_test"]),
    ]

    rows: List[Dict[str, Any]] = []
    for split_boundary, nested_role, included, exclusion_reason, frame in partition_specs:
        part = frame.copy().sort_values("feature_date")
        n = int(len(part))

        event_count = 0
        prevalence = 0.0
        if n > 0:
            mapped_y = part["target_date"].map(date_to_target)
            if mapped_y.isna().any():
                raise RuntimeError(
                    "Fold {} {} has target dates missing in canonical dataset".format(fold, nested_role)
                )
            y = mapped_y.to_numpy(dtype=float)
            event_count = int((y >= float(tau)).sum())
            prevalence = float(event_count / float(n))

        note = ""
        if nested_role == "component_validation" and event_count <= 0:
            note = "component_validation_tau_support_zero"

        rows.append(
            {
                "outer_fold": int(fold),
                "split_boundary": split_boundary,
                "nested_role": nested_role,
                "included": bool(included),
                "exclusion_reason": exclusion_reason,
                "rows_N": int(n),
                "tau_threshold": float(tau),
                "tau_event_count": int(event_count),
                "tau_event_prevalence": float(prevalence),
                "feature_date_start": part["feature_date"].min() if n > 0 else pd.NaT,
                "feature_date_end": part["feature_date"].max() if n > 0 else pd.NaT,
                "target_date_start": part["target_date"].min() if n > 0 else pd.NaT,
                "target_date_end": part["target_date"].max() if n > 0 else pd.NaT,
                "component_validation_start": fold_info["component_validation_start"],
                "hybrid_validation_start": fold_info["hybrid_validation_start"],
                "outer_test_start": fold_info["outer_test_start"],
                "nested_split_id": fold_info["nested_split_id"],
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "run_id": run_id,
                "diagnostic_note": note,
            }
        )

    return pd.DataFrame(rows).sort_values(["split_boundary", "nested_role"]).reset_index(drop=True)


def extract_partition(
    feature_frame: pd.DataFrame,
    split_rows: pd.DataFrame,
    label: str,
    fold: int,
) -> pd.DataFrame:
    keys = split_rows[["feature_date", "target_date"]].drop_duplicates().copy()
    merged = keys.merge(
        feature_frame,
        on=["feature_date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    if "row_index" not in merged.columns:
        raise RuntimeError("Feature frame missing row_index in {} fold {}".format(label, fold))

    if merged["row_index"].isna().any():
        miss = merged[merged["row_index"].isna()][["feature_date", "target_date"]].head(5)
        pairs = [
            "{}->{}".format(to_date_str(r.feature_date), to_date_str(r.target_date))
            for r in miss.itertuples(index=False)
        ]
        raise RuntimeError("Missing feature rows for {} fold {}: {}".format(label, fold, pairs))

    merged = merged.sort_values("feature_date").reset_index(drop=True)
    return merged


def build_elasticnet_feature_frame(
    dataset_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    feature_cols: List[str],
) -> pd.DataFrame:
    df = dataset_df.copy()

    needed = ["Inflow", "TNin", "TOCin", "temp_mean_c", "precip_total_mm", target_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError("Canonical dataset missing required columns for ElasticNet: {}".format(missing))

    if "BODin" not in df.columns:
        df["BODin"] = np.nan

    for col in [target_col, "Inflow", "TNin", "TOCin", "BODin", "temp_mean_c", "precip_total_mm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["C_N"] = df["TOCin"] / (df["TNin"] + 1e-6)
    df["feature_date"] = df[date_col]
    df["target_date"] = df[date_col] + pd.to_timedelta(horizon, unit="D")
    df["y_true"] = df[target_col].shift(-horizon)

    df["TNout_lag1"] = df[target_col].shift(1)
    df["TNout_roll7"] = df[target_col].shift(1).rolling(7).mean()
    df["TNout_roll14"] = df[target_col].shift(1).rolling(14).mean()

    for col in ["Inflow", "TNin", "TOCin", "BODin"]:
        df["{}_roll7".format(col)] = pd.to_numeric(df[col], errors="coerce").rolling(7).mean()

    df["temp_roll7"] = pd.to_numeric(df["temp_mean_c"], errors="coerce").rolling(7).mean()
    df["precip_sum3"] = pd.to_numeric(df["precip_total_mm"], errors="coerce").rolling(3).sum()

    doy = df[date_col].dt.dayofyear.values
    df["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    unavailable = [c for c in feature_cols if c not in df.columns]
    if unavailable:
        raise RuntimeError("ElasticNet feature columns unavailable: {}".format(unavailable))

    out_cols = ["feature_date", "target_date", "y_true"] + feature_cols
    out = df[out_cols].copy().dropna(subset=feature_cols + ["y_true"]).sort_values("feature_date")
    out = out.reset_index(drop=True)
    out.insert(0, "row_index", np.arange(len(out), dtype=int))

    if out.duplicated(subset=["feature_date", "target_date"]).any():
        raise RuntimeError("ElasticNet feature frame has duplicate keys")

    return out


def load_hgbr_feature_frame(
    npz_path: Path,
    horizon: int,
    date_to_target: Dict[pd.Timestamp, float],
    expected_feature_count: int,
) -> Tuple[pd.DataFrame, List[str]]:
    if not npz_path.exists():
        raise RuntimeError("HGBR feature NPZ missing: {}".format(npz_path))

    z = np.load(npz_path, allow_pickle=True)
    required = {"X", "dates", "feature_names"}
    if not required.issubset(set(z.files)):
        raise RuntimeError("HGBR NPZ missing required arrays: {}".format(sorted(required)))

    X = np.asarray(z["X"], dtype=float)
    dates = normalize_date_col(pd.Series(z["dates"]))
    feature_names = [str(c) for c in list(z["feature_names"])]

    if X.ndim != 2:
        raise RuntimeError("HGBR feature matrix must be 2D")
    if len(dates) != X.shape[0]:
        raise RuntimeError("HGBR date length mismatch")
    if len(feature_names) != X.shape[1]:
        raise RuntimeError("HGBR feature-name count mismatch")
    if expected_feature_count and len(feature_names) != int(expected_feature_count):
        raise RuntimeError("HGBR expected feature count mismatch")

    frame = pd.DataFrame(X, columns=feature_names)
    frame.insert(0, "feature_date", dates)
    frame["target_date"] = frame["feature_date"] + pd.to_timedelta(horizon, unit="D")
    frame["y_true"] = frame["target_date"].map(date_to_target)

    if frame[feature_names].isna().any().any():
        raise RuntimeError("HGBR NPZ contains NaN feature values")

    frame = frame.dropna(subset=["y_true"]).sort_values("feature_date").reset_index(drop=True)
    frame.insert(0, "row_index", np.arange(len(frame), dtype=int))

    if frame.duplicated(subset=["feature_date", "target_date"]).any():
        raise RuntimeError("HGBR feature frame has duplicate keys")

    return frame, feature_names


def load_bcr_feature_bundle(
    npz_path: Path,
    horizon: int,
    date_to_target: Dict[pd.Timestamp, float],
) -> Dict[str, Any]:
    if not npz_path.exists():
        raise RuntimeError("BCR feature NPZ missing: {}".format(npz_path))

    z = np.load(npz_path, allow_pickle=True)
    required = {"X", "y", "dates"}
    if not required.issubset(set(z.files)):
        raise RuntimeError("BCR NPZ missing required arrays: {}".format(sorted(required)))

    X = np.asarray(z["X"], dtype=np.float32)
    y = np.asarray(z["y"], dtype=np.float32)
    dates = normalize_date_col(pd.Series(z["dates"]))

    if X.ndim != 2:
        raise RuntimeError("BCR feature matrix must be 2D")
    if len(dates) != X.shape[0] or len(y) != X.shape[0]:
        raise RuntimeError("BCR NPZ array length mismatch")

    target_dates = dates + pd.to_timedelta(horizon, unit="D")
    y_from_dataset = target_dates.map(date_to_target)
    keep = ~y_from_dataset.isna()

    X = X[keep.to_numpy()]
    y = y[keep.to_numpy()]
    dates_kept = dates[keep].reset_index(drop=True)
    target_dates_kept = target_dates[keep].reset_index(drop=True)
    y_map_vals = y_from_dataset[keep].to_numpy(dtype=float)

    max_abs = float(np.max(np.abs(y.astype(float) - y_map_vals))) if len(y) else 0.0
    if max_abs > 1e-5:
        raise RuntimeError("BCR NPZ target mismatch against canonical dataset; max_abs_diff={}".format(max_abs))

    feature_frame = pd.DataFrame(
        {
            "row_index": np.arange(len(dates_kept), dtype=int),
            "feature_date": dates_kept,
            "target_date": target_dates_kept,
            "y_true": y.astype(float),
        }
    )

    if feature_frame.duplicated(subset=["feature_date", "target_date"]).any():
        raise RuntimeError("BCR feature frame has duplicate keys")

    feature_names: List[str] = []
    if "feature_names" in z.files:
        feature_names = [str(c) for c in list(z["feature_names"])]

    return {
        "X": X,
        "y": y,
        "feature_dates": dates_kept.to_numpy(),
        "target_dates": target_dates_kept.to_numpy(),
        "feature_frame": feature_frame,
        "feature_names": feature_names,
    }


def rank01(series: pd.Series) -> np.ndarray:
    r = series.rank(method="average", ascending=True).to_numpy(dtype=float)
    if len(r) <= 1:
        return np.zeros_like(r)
    return (r - 1.0) / (len(r) - 1.0)


def build_weight_grid(n_models: int, step: float) -> List[Tuple[float, ...]]:
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    out: List[Tuple[float, ...]] = []
    for tup in product(vals, repeat=n_models):
        s = float(np.sum(tup))
        if s <= 0.0:
            continue
        w = tuple(float(v / s) for v in tup)
        out.append(w)
    return out


def alarm_metrics_from_scores(y_true: np.ndarray, score: np.ndarray, tau: float, budget_r: float) -> Dict[str, float]:
    n = int(len(y_true))
    if n <= 0:
        return {"n": 0, "k": 0, "events": 0, "TP": 0, "precision": 0.0, "recall": 0.0}

    k = max(1, int(math.ceil(float(budget_r) * float(n))))
    event = (y_true >= float(tau)).astype(int)
    order = np.argsort(-score)
    top_idx = order[:k]
    tp = int(event[top_idx].sum())
    events = int(event.sum())
    precision = float(tp / k) if k > 0 else 0.0
    recall = float(tp / events) if events > 0 else 0.0

    return {
        "n": n,
        "k": int(k),
        "events": events,
        "TP": int(tp),
        "precision": precision,
        "recall": recall,
    }


def param_to_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        torch.use_deterministic_algorithms(True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


class SeqIndexDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_dates: np.ndarray,
        target_dates: np.ndarray,
        indices: np.ndarray,
        L: int,
        taus: Tuple[float, float, float],
    ):
        self.X = X
        self.y = y
        self.feature_dates = feature_dates
        self.target_dates = target_dates
        self.indices = np.array(indices, dtype=int)
        self.L = int(L)
        self.taus = taus

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        s = max(0, i - self.L + 1)
        seq = self.X[s : i + 1]
        if len(seq) < self.L:
            pad = np.repeat(seq[:1], self.L - len(seq), axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        yt = float(self.y[i])
        events = np.array([1.0 if yt >= t else 0.0 for t in self.taus], dtype=np.float32)
        return (
            torch.from_numpy(seq).float(),
            torch.tensor(yt, dtype=torch.float32),
            torch.from_numpy(events).float(),
            torch.tensor(i, dtype=torch.int64),
        )


class CausalTCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1, dropout: float = 0.2):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=d)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.pad(x, (self.pad, 0))
        h = F.gelu(self.conv1(x1))
        h = self.dropout(h)
        h1 = F.pad(h, (self.pad, 0))
        h = F.gelu(self.conv2(h1))
        h = self.dropout(h)
        return h + self.res(x)


class BCRTCN(nn.Module):
    def __init__(
        self,
        n_feat: int,
        channels: int,
        blocks: int,
        kernel: int,
        dropout: float,
        n_taus: int,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = n_feat
        for b in range(blocks):
            d = 2 ** b
            layers.append(CausalTCNBlock(in_ch, channels, k=kernel, d=d, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.reg_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, n_taus),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h_last = h[:, :, -1]
        yhat = self.reg_head(h_last).squeeze(-1)
        logits = self.cls_head(h_last)
        return yhat, logits


def pairwise_rank_loss(scores: torch.Tensor, labels: torch.Tensor, rng: torch.Generator, max_pairs: int = 256) -> torch.Tensor:
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return scores.new_tensor(0.0)

    m = min(int(len(pos) * len(neg)), int(max_pairs))
    pos_s = pos[torch.randint(0, len(pos), (m,), device=scores.device, generator=rng)]
    neg_s = neg[torch.randint(0, len(neg), (m,), device=scores.device, generator=rng)]
    return F.softplus(-(pos_s - neg_s)).mean()


def run_bcr_component_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_dates: np.ndarray,
    target_dates: np.ndarray,
    subtrain_idx: np.ndarray,
    validation_idx: np.ndarray,
    cfg: Dict[str, Any],
    fold: int,
) -> Dict[str, Any]:
    set_deterministic(int(cfg["seed"]) + fold * 100 + 1)

    mu = X[subtrain_idx].mean(axis=0, keepdims=True)
    sd = X[subtrain_idx].std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd

    ds_tr = SeqIndexDataset(Xs, y, feature_dates, target_dates, subtrain_idx, int(cfg["L"]), TAUS)
    ds_va = SeqIndexDataset(Xs, y, feature_dates, target_dates, validation_idx, int(cfg["L"]), TAUS)

    dl_gen = torch.Generator(device="cpu")
    dl_gen.manual_seed(int(cfg["seed"]) + fold * 1000 + 11)
    rank_gen = torch.Generator(device="cpu")
    rank_gen.manual_seed(int(cfg["seed"]) + fold * 1000 + 123)

    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0, generator=dl_gen)
    dl_va = DataLoader(ds_va, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    model = BCRTCN(
        n_feat=X.shape[1],
        channels=int(cfg["channels"]),
        blocks=int(cfg["blocks"]),
        kernel=int(cfg["kernel"]),
        dropout=float(cfg["dropout"]),
        n_taus=len(TAUS),
    ).to(torch.device(DEVICE))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    y_sub = y[subtrain_idx]
    pos = np.array([(y_sub >= t).mean() for t in TAUS], dtype=float)
    pos_weight = torch.tensor([(1.0 - p) / (p + 1e-6) for p in pos], dtype=torch.float32, device=torch.device(DEVICE))

    huber = nn.SmoothL1Loss(beta=float(cfg["huber_beta"]))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_recall = -1.0
    best_epoch = -1
    best_mae = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = int(cfg["patience"])

    history_rows: List[Dict[str, Any]] = []

    for epoch in range(int(cfg["epochs_max"])):
        model.train()
        tr_losses = []
        for xb, yb, eb, _ in dl_tr:
            xb = xb.to(torch.device(DEVICE))
            yb = yb.to(torch.device(DEVICE))
            eb = eb.to(torch.device(DEVICE))

            yhat, logits = model(xb)
            loss_reg = huber(yhat, yb)
            loss_bce = bce(logits, eb)
            s16 = logits[:, 1]
            s15 = logits[:, 0]
            loss_rank = pairwise_rank_loss(s16, eb[:, 1], rng=rank_gen) + 0.3 * pairwise_rank_loss(s15, eb[:, 0], rng=rank_gen)

            loss = float(cfg["w_reg"]) * loss_reg + float(cfg["w_bce"]) * loss_bce + float(cfg["w_rank"]) * loss_rank

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()

            tr_losses.append(float(loss.detach().item()))

        model.eval()
        s16_all: List[np.ndarray] = []
        e16_all: List[np.ndarray] = []
        mae_vals: List[float] = []

        with torch.no_grad():
            for xb, yb, eb, _ in dl_va:
                xb = xb.to(torch.device(DEVICE))
                yb = yb.to(torch.device(DEVICE))
                eb = eb.to(torch.device(DEVICE))

                yhat, logits = model(xb)
                mae_vals.append(float(torch.mean(torch.abs(yhat - yb)).item()))
                s16_all.append(torch.sigmoid(logits[:, 1]).cpu().numpy())
                e16_all.append(eb[:, 1].cpu().numpy())

        if len(s16_all):
            s16 = np.concatenate(s16_all)
            e16 = np.concatenate(e16_all)
            rec_obj = alarm_metrics_from_scores(
                y_true=np.where(e16 > 0.5, 1.0, 0.0),
                score=s16,
                tau=0.5,
                budget_r=float(cfg["r_budget"]),
            )
            val_recall = float(rec_obj["recall"])
        else:
            val_recall = 0.0

        val_mae = float(np.mean(mae_vals)) if mae_vals else float("inf")

        history_rows.append(
            {
                "epoch": int(epoch),
                "training_loss": float(np.mean(tr_losses)) if tr_losses else float("nan"),
                "validation_recall_tau16_r05": val_recall,
                "validation_mae": val_mae,
            }
        )

        improved = val_recall > (best_recall + 1e-12)
        if improved:
            best_recall = val_recall
            best_mae = val_mae
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(cfg["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None or best_epoch < 0:
        raise RuntimeError("BCR component selection did not capture a checkpoint for fold {}".format(fold))

    model.load_state_dict(best_state)

    return {
        "selected_epoch": int(best_epoch),
        "best_validation_recall_tau16_r05": float(best_recall),
        "best_validation_mae": float(best_mae),
        "training_epochs_completed": int(len(history_rows)),
        "history": history_rows,
        "pos_weight_tau15": float(pos_weight[0].cpu().item()),
        "pos_weight_tau16": float(pos_weight[1].cpu().item()),
        "pos_weight_tau17": float(pos_weight[2].cpu().item()),
    }


def run_bcr_fixed_epochs_predict(
    X: np.ndarray,
    y: np.ndarray,
    feature_dates: np.ndarray,
    target_dates: np.ndarray,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    epochs_to_train: int,
    cfg: Dict[str, Any],
    fold: int,
    stage_offset: int,
) -> pd.DataFrame:
    set_deterministic(int(cfg["seed"]) + fold * 100 + stage_offset)

    if epochs_to_train <= 0:
        raise RuntimeError("epochs_to_train must be positive for BCR refit")

    mu = X[train_idx].mean(axis=0, keepdims=True)
    sd = X[train_idx].std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd

    ds_tr = SeqIndexDataset(Xs, y, feature_dates, target_dates, train_idx, int(cfg["L"]), TAUS)
    ds_pr = SeqIndexDataset(Xs, y, feature_dates, target_dates, predict_idx, int(cfg["L"]), TAUS)

    dl_gen = torch.Generator(device="cpu")
    dl_gen.manual_seed(int(cfg["seed"]) + fold * 1000 + stage_offset + 11)
    rank_gen = torch.Generator(device="cpu")
    rank_gen.manual_seed(int(cfg["seed"]) + fold * 1000 + stage_offset + 123)

    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0, generator=dl_gen)
    dl_pr = DataLoader(ds_pr, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    model = BCRTCN(
        n_feat=X.shape[1],
        channels=int(cfg["channels"]),
        blocks=int(cfg["blocks"]),
        kernel=int(cfg["kernel"]),
        dropout=float(cfg["dropout"]),
        n_taus=len(TAUS),
    ).to(torch.device(DEVICE))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    y_sub = y[train_idx]
    pos = np.array([(y_sub >= t).mean() for t in TAUS], dtype=float)
    pos_weight = torch.tensor([(1.0 - p) / (p + 1e-6) for p in pos], dtype=torch.float32, device=torch.device(DEVICE))

    huber = nn.SmoothL1Loss(beta=float(cfg["huber_beta"]))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(int(epochs_to_train)):
        model.train()
        for xb, yb, eb, _ in dl_tr:
            xb = xb.to(torch.device(DEVICE))
            yb = yb.to(torch.device(DEVICE))
            eb = eb.to(torch.device(DEVICE))

            yhat, logits = model(xb)
            loss_reg = huber(yhat, yb)
            loss_bce = bce(logits, eb)
            s16 = logits[:, 1]
            s15 = logits[:, 0]
            loss_rank = pairwise_rank_loss(s16, eb[:, 1], rng=rank_gen) + 0.3 * pairwise_rank_loss(s15, eb[:, 0], rng=rank_gen)
            loss = float(cfg["w_reg"]) * loss_reg + float(cfg["w_bce"]) * loss_bce + float(cfg["w_rank"]) * loss_rank

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()

    pred_rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for xb, yb, _, idxb in dl_pr:
            xb = xb.to(torch.device(DEVICE))
            yhat, logits = model(xb)
            probs = torch.sigmoid(logits)
            yhat_np = yhat.cpu().numpy()
            probs_np = probs.cpu().numpy()
            idx_np = idxb.cpu().numpy().astype(int)
            yb_np = yb.cpu().numpy()

            for j, ridx in enumerate(idx_np.tolist()):
                pred_rows.append(
                    {
                        "row_index": int(ridx),
                        "feature_date": pd.Timestamp(feature_dates[ridx]),
                        "target_date": pd.Timestamp(target_dates[ridx]),
                        "y_true": float(yb_np[j]),
                        "bcr_tcn_y_pred": float(yhat_np[j]),
                        "bcr_tcn_p_tau15": float(probs_np[j, 0]),
                        "bcr_tcn_p_tau16": float(probs_np[j, 1]),
                        "bcr_tcn_p_tau17": float(probs_np[j, 2]),
                    }
                )

    pred_df = pd.DataFrame(pred_rows).sort_values("feature_date").reset_index(drop=True)
    return pred_df


def build_hgbr_candidates(
    candidate_space: Dict[str, Any],
    budget: int,
    seed: int,
    horizon: int,
) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(int(seed))
    param_space = candidate_space["parameter_space"]
    fixed = dict(candidate_space.get("fixed_parameters", {}))

    required = [
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "max_bins",
    ]
    missing = [p for p in required if p not in param_space]
    if missing:
        raise RuntimeError("HGBR candidate space missing parameters: {}".format(missing))

    def sample_param(spec: Dict[str, Any]) -> Any:
        dist = str(spec["distribution"])
        low = spec["low"]
        high = spec["high"]
        if dist == "int_uniform":
            return int(rng.randint(int(low), int(high) + 1))
        if dist == "log_uniform":
            return float(np.exp(rng.uniform(np.log(float(low)), np.log(float(high)))))
        raise RuntimeError("Unsupported HGBR parameter distribution: {}".format(dist))

    out: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    max_attempts = 50000

    while len(out) < int(budget):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Could not sample enough unique HGBR candidates")

        params: Dict[str, Any] = {}
        for name in required:
            params[name] = sample_param(param_space[name])

        full = dict(fixed)
        full.update(params)

        key = param_to_json(full)
        if key in seen:
            continue
        seen.add(key)

        cfg_id = "h{}_cfg_{:03d}".format(horizon, len(out) + 1)
        out.append(
            {
                "candidate_configuration_id": cfg_id,
                "candidate_parameters": full,
                "candidate_parameters_json": key,
            }
        )

    return out


def compute_output_checksums(out_dir: Path) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        checksums[p.name] = sha256_file(p)
    return checksums


def write_checksums_file(out_dir: Path, checksum_path: Path) -> None:
    targets = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == checksum_path.name:
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".py"}:
            continue
        targets.append(p)

    lines = []
    for p in targets:
        lines.append("{}  {}".format(sha256_file(p), p.name))

    ensure_inside_authorized(checksum_path)
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_completion_report(
    run_id: str,
    dataset_sha: str,
    split_sha: str,
    git_commit: str,
    summary_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    support_df: pd.DataFrame,
    decision: str,
    bcr_gate_df: Optional[pd.DataFrame] = None,
    fold1_tau16_diag_summary: Optional[Dict[str, Any]] = None,
    decision_note: str = "",
) -> str:
    lines: List[str] = []
    lines.append("# Fully Nested Corrected H5 HybridRank")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("- Build a fully chronological and fully nested H5 HybridRank pathway with strict temporal isolation.")
    lines.append("")
    lines.append("## 2. Submitted Temporal-Selection Problem")
    lines.append("- Submitted tuned HybridRank pooled non-held-out folds, allowing later chronology to influence earlier-fold weight tuning.")
    lines.append("")
    lines.append("## 3. Input Verification")
    lines.append("- dataset_sha256: {}".format(dataset_sha))
    lines.append("- split_sha256: {}".format(split_sha))
    lines.append("- git_commit: {}".format(git_commit))
    lines.append("")
    lines.append("## 4. Fully Nested Split Design")
    for r in summary_df.itertuples(index=False):
        lines.append(
            "- Fold {}: outer_train_N={}, component_subtrain_N_after_embargo={}, component_validation_N={}, hybrid_validation_N={}, outer_test_N={}".format(
                int(r.outer_fold),
                int(r.outer_train_N),
                int(r.component_subtrain_N_after_embargo),
                int(r.component_validation_N),
                int(r.hybrid_validation_N),
                int(r.outer_test_N),
            )
        )
    lines.append("")
    lines.append("## 5. Five-Day Embargoes")
    lines.append("- inner_target_separation_pass: {}".format(bool(summary_df["inner_target_separation_pass"].all())))
    lines.append("- hybrid_target_separation_pass: {}".format(bool(summary_df["hybrid_target_separation_pass"].all())))
    lines.append("- outer_target_separation_pass: {}".format(bool(summary_df["outer_target_separation_pass"].all())))
    lines.append("")
    lines.append("## 6. Component-Model Selection")
    for r in selection_df.itertuples(index=False):
        lines.append(
            "- Fold {} {} selected_configuration={} objective={}".format(
                int(r.outer_fold), r.component, r.selected_configuration, r.selection_objective
            )
        )
    if bcr_gate_df is not None and not bcr_gate_df.empty:
        lines.append("- BCR checkpoint gate (component-validation tau16 events and k@5%):")
        for r in bcr_gate_df.itertuples(index=False):
            lines.append(
                "  fold{} N={} tau16_events={} k_r005={} recall_defined={} blocked={}".format(
                    int(r.outer_fold),
                    int(r.component_validation_N),
                    int(r.component_validation_events_tau16),
                    int(r.component_validation_k_r005),
                    bool(r.recall_at_5pct_defined),
                    bool(r.blocked_for_bcr_checkpoint_selection),
                )
            )
    if fold1_tau16_diag_summary is not None:
        lines.append(
            "- Fold 1 tau16 support diagnostic artifact: {}".format(
                fold1_tau16_diag_summary.get("file", FOLD1_TAU16_DIAGNOSTIC_PATH.name)
            )
        )
        lines.append(
            "  component_validation feature_span={}..{} target_span={}..{} N={} tau16_events={} prevalence={:.6f}".format(
                fold1_tau16_diag_summary.get("component_validation_feature_date_start", ""),
                fold1_tau16_diag_summary.get("component_validation_feature_date_end", ""),
                fold1_tau16_diag_summary.get("component_validation_target_date_start", ""),
                fold1_tau16_diag_summary.get("component_validation_target_date_end", ""),
                int(fold1_tau16_diag_summary.get("component_validation_N", 0)),
                int(fold1_tau16_diag_summary.get("component_validation_tau16_events", 0)),
                float(fold1_tau16_diag_summary.get("component_validation_tau16_prevalence", 0.0)),
            )
        )
        lines.append("  counts by split boundary are listed per nested role in the diagnostic CSV.")
    if decision_note:
        lines.append("- Decision note: {}".format(decision_note))
    lines.append("")
    lines.append("## 7. HybridRank Validation Predictions")
    lines.append("- Predictions generated from refit component models on component-development only.")
    lines.append("")
    lines.append("## 8. Event-Support Safeguard")
    lines.append("- minimum events required: {}".format(SUPPORT_MIN_EVENTS))
    lines.append("- minimum alarm slots required: {}".format(SUPPORT_MIN_ALARM_SLOTS))
    lines.append("- Guard uses fixed policy at r <= 0.05.")
    lines.append("- This sparse-support rule is a conservative revised safeguard, not an exact submitted feature.")
    lines.append("")
    lines.append("## 9. HybridRank Weight Selection")
    lines.append("- Candidate grid: 0.0..1.0 step 0.1; nonzero vectors normalized to sum 1.")
    lines.append("- Objective order: recall, precision, TP.")
    lines.append("- Precision floor: fixed-policy precision on HybridRank validation.")
    lines.append("")
    lines.append("## 10. Fixed and Guarded Policies")
    lines.append("- Fixed weights: BCR-TCN=0.50, ElasticNet=0.25, Persistence=0.25, HGBR=0.00.")
    lines.append("- Guard condition: r <= 0.05.")
    lines.append("")
    lines.append("## 11. Final Hybrid-Specific Component Refits")
    lines.append("- Final component fits use complete corrected outer-training blocks and produce hybrid-specific outer-test predictions.")
    lines.append("")
    lines.append("## 12. Outer-Test Hybrid Scores")
    lines.append("- Outer-test scores are rank-only retrospective outputs; no outer-test alarm decisions were computed.")
    lines.append("")
    lines.append("## 13. Test and Future-Fold Isolation")
    lines.append("- No outer-test outcomes were used for component selection or weight selection.")
    lines.append("- No future folds were used in earlier-fold model/weight selection.")
    lines.append("")
    lines.append("## 14. Difference from Canonical Point Models")
    lines.append("- Canonical base-model package remains unchanged; this stage adds hybrid-specific nested predictions and scores only.")
    lines.append("")
    lines.append("## 15. Determinism and Automated Tests")
    lines.append("- Build completed deterministically under fixed seeds; 62-assertion test script is generated separately.")
    lines.append("")
    lines.append("## 16. Limitations")
    lines.append("1. Submitted tuned HybridRank pooled non-held-out folds, allowing future chronology to influence earlier folds.")
    lines.append("2. The revised procedure separates component selection and ensemble selection.")
    lines.append("3. Five-day target-date embargoes separate every stage.")
    lines.append("4. The fixed policy is used at r <= 0.05.")
    lines.append("5. The fixed policy is also used when HybridRank-validation event support is insufficient.")
    lines.append("6. Full outer-test fold rank normalization remains retrospective and non-deployable.")
    lines.append("7. A separate sequential policy will be evaluated later.")
    lines.append("8. Hybrid-specific component predictions are not substituted for the canonical point-forecast package.")
    lines.append("9. No outer-test alarm metrics were calculated in this stage.")
    lines.append("")
    lines.append("## 17. Readiness Decision")
    lines.append("- run_id: {}".format(run_id))
    lines.append("")
    lines.append("FINAL DECISION")
    lines.append("")
    lines.append("<!-- AUTO_DECISION_START -->")
    if decision == "A":
        lines.append("A. Fully nested corrected H5 HybridRank scores passed; retrospective alarm-budget evaluation may begin.")
    elif decision == "B":
        lines.append("B. Pipeline completed, but one or more component or validation-support issues require investigation.")
    elif decision == "C":
        lines.append("C. Fully nested component or ensemble selection could not be completed safely with the available data.")
    else:
        lines.append("D. Temporal isolation, source preservation, or deterministic tests failed.")
    lines.append("<!-- AUTO_DECISION_END -->")

    lines.append("")
    lines.append("TERMINAL SUMMARY")
    lines.append("- event-support rows: {}".format(int(len(support_df))))

    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("Run this script from workspace root: {}".format(ROOT))

    active_branch = git_output(["git", "branch", "--show-current"])
    if active_branch != EXPECTED_BRANCH:
        raise RuntimeError("Active branch must remain {} (observed {})".format(EXPECTED_BRANCH, active_branch))

    changed_paths = parse_git_status_paths()
    outside_changes = [
        p for p in changed_paths if not is_authorized_git_status_path(p, AUTHORIZED_REL_DIR)
    ]
    if outside_changes:
        raise RuntimeError(
            "Working tree has changes outside authorized directory: {}".format(sorted(outside_changes)[:10])
        )

    protected_before = snapshot_tree_checksums(PROTECTED_DIRS)

    lock = load_lock(LOCK_PATH)
    dataset_df, dataset_sha = load_and_verify_dataset(lock)

    protocol_checks = verify_checksum_manifest(PROTOCOL_SHA_PATH, PROTOCOL_DIR)

    split_sha = sha256_file(SPLIT_PATH)
    expected_split_sha = protocol_checks["corrected_split_assignment.csv"]["expected"]
    if split_sha != expected_split_sha:
        raise RuntimeError("Corrected split checksum mismatch against protocol checksum manifest")

    canonical_checks = verify_checksum_manifest(CANONICAL_CHECKSUMS_PATH, CANONICAL_DIR)

    canonical_manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    canonical_report_decision = infer_decision_from_report(CANONICAL_REPORT_PATH)

    if canonical_report_decision != "A":
        raise RuntimeError("Canonical assembly decision is not A")
    if int(canonical_manifest.get("canonical_N", -1)) != 747:
        raise RuntimeError("Canonical H5 N must be 747")
    if int(canonical_manifest.get("horizon", -1)) != H:
        raise RuntimeError("Canonical assembly horizon must be H5")

    persistence_decision = infer_decision_from_report(PERSISTENCE_COMPLETION_REPORT_PATH)
    enet_decision = infer_decision_from_report(ENET_COMPLETION_REPORT_PATH)
    hgbr_decision = infer_decision_from_report(HGBR_COMPLETION_REPORT_PATH)
    bcr_decision = infer_decision_from_report(BCR_COMPLETION_REPORT_PATH)

    component_decisions = {
        "Persistence": persistence_decision,
        "ElasticNet": enet_decision,
        "HGBR": hgbr_decision,
        "BCR-TCN v1.1": bcr_decision,
    }
    if any(v != "A" for v in component_decisions.values()):
        raise RuntimeError("One or more controlled-rerun component decisions are not A: {}".format(component_decisions))

    git_commit = git_output(["git", "rev-parse", "HEAD"])
    python_executable = sys.executable
    python_version = platform.python_version()

    split = load_split_assignment(SPLIT_PATH)
    split_h5 = split[split["horizon"].astype(int) == H].copy()
    if split_h5.empty:
        raise RuntimeError("No H5 rows found in corrected split assignment")
    if sorted(split_h5["horizon"].astype(int).unique().tolist()) != [H]:
        raise RuntimeError("Split selection is not H5-only")

    assignment_df, summary_df, fold_data = build_nested_split(split_h5, dataset_sha, split_sha)

    # Preflight summary
    print("[Preflight] working_directory={}".format(Path.cwd()))
    print("[Preflight] branch={} commit={}".format(active_branch, git_commit))
    print("[Preflight] python_executable={} python_version={}".format(python_executable, python_version))

    date_col = lock["date_column"]
    target_col = lock["target_column"]
    date_to_target = date_to_target_map(dataset_df, date_col, target_col)

    for r in summary_df.itertuples(index=False):
        fold = int(r.outer_fold)
        hv = fold_data[fold]["hybrid_validation"].copy()
        hv_y = hv["target_date"].map(date_to_target).to_numpy(dtype=float)
        e15 = int((hv_y >= 15.0).sum())
        e16 = int((hv_y >= 16.0).sum())
        e17 = int((hv_y >= 17.0).sum())

        print(
            "[Preflight][Fold {}] outer_train_N={} component_subtrain_before={} inner_embargo_excluded={} "
            "component_subtrain_after={} component_validation_N={} hybrid_embargo_excluded={} "
            "hybrid_validation_N={} outer_test_N={}"
            .format(
                fold,
                int(r.outer_train_N),
                int(r.component_subtrain_N_before_embargo),
                int(r.inner_embargo_excluded_N),
                int(r.component_subtrain_N_after_embargo),
                int(r.component_validation_N),
                int(r.hybrid_embargo_excluded_N),
                int(r.hybrid_validation_N),
                int(r.outer_test_N),
            )
        )
        print(
            "[Preflight][Fold {}] hv_events_tau15={} hv_events_tau16={} hv_events_tau17={} bcr_sequence_count={}".format(
                fold,
                e15,
                e16,
                e17,
                int(r.component_subtrain_N_after_embargo),
            )
        )

        if int(r.component_subtrain_N_after_embargo) <= 0 or int(r.component_validation_N) <= 0 or int(r.hybrid_validation_N) <= 0:
            raise RuntimeError("Fold {} has an empty required segment".format(fold))

    run_id = "fully_nested_hybridrank_h5_{}_{}_{}".format(git_commit[:12], dataset_sha[:8], split_sha[:8])

    input_verification = {
        "working_directory": str(Path.cwd().resolve()),
        "expected_working_directory": str(ROOT.resolve()),
        "working_directory_pass": str(Path.cwd().resolve()) == str(ROOT.resolve()),
        "git_branch": active_branch,
        "git_branch_pass": active_branch == EXPECTED_BRANCH,
        "git_commit": git_commit,
        "working_tree_changes": changed_paths,
        "working_tree_clean_outside_authorized_directory": len(outside_changes) == 0,
        "authorized_output_directory": str((ROOT / AUTHORIZED_REL_DIR).resolve()),
        "dataset_path": lock["absolute_path"],
        "dataset_sha256_expected": lock["sha256"],
        "dataset_sha256_observed": dataset_sha,
        "split_file": str(SPLIT_PATH.resolve()),
        "split_sha256_expected": expected_split_sha,
        "split_sha256_observed": split_sha,
        "protocol_checks": protocol_checks,
        "canonical_checks": canonical_checks,
        "canonical_decision": canonical_report_decision,
        "canonical_h5_N": int(canonical_manifest["canonical_N"]),
        "canonical_assembly_id": canonical_manifest["assembly_id"],
        "horizon": H,
        "h5_only_pass": True,
        "component_decisions": component_decisions,
        "python_executable": python_executable,
        "python_version": python_version,
        "software_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "verification_pass": True,
        "timestamp": utc_now_iso(),
        "run_id": run_id,
    }
    write_json(INPUT_VERIFICATION_PATH, input_verification)

    # Write nested split outputs.
    to_csv_with_dates(
        assignment_df,
        NESTED_ASSIGNMENT_PATH,
        ["feature_date", "target_date", "component_validation_start", "hybrid_validation_start", "outer_test_start"],
    )
    to_csv_with_dates(
        summary_df,
        NESTED_SUMMARY_PATH,
        [
            "component_subtrain_start",
            "component_subtrain_end",
            "component_validation_start",
            "component_validation_end",
            "hybrid_validation_start",
            "hybrid_validation_end",
            "outer_test_start",
            "outer_test_end",
        ],
    )

    fold1_tau16_diag_df = build_fold_tau_support_diagnostic(
        fold=1,
        fold_info=fold_data[1],
        date_to_target=date_to_target,
        dataset_sha=dataset_sha,
        split_sha=split_sha,
        run_id=run_id,
        tau=16.0,
    )
    to_csv_with_dates(
        fold1_tau16_diag_df,
        FOLD1_TAU16_DIAGNOSTIC_PATH,
        [
            "feature_date_start",
            "feature_date_end",
            "target_date_start",
            "target_date_end",
            "component_validation_start",
            "hybrid_validation_start",
            "outer_test_start",
        ],
    )

    fold1_cv_diag = fold1_tau16_diag_df[
        fold1_tau16_diag_df["nested_role"].astype(str) == "component_validation"
    ].reset_index(drop=True)
    if len(fold1_cv_diag) != 1:
        raise RuntimeError("Fold 1 tau16 diagnostic must include exactly one component_validation row")

    cv_row = fold1_cv_diag.iloc[0]
    fold1_tau16_diag_summary = {
        "file": FOLD1_TAU16_DIAGNOSTIC_PATH.name,
        "component_validation_N": int(cv_row["rows_N"]),
        "component_validation_tau16_events": int(cv_row["tau_event_count"]),
        "component_validation_tau16_prevalence": float(cv_row["tau_event_prevalence"]),
        "component_validation_feature_date_start": (
            to_date_str(cv_row["feature_date_start"]) if pd.notna(cv_row["feature_date_start"]) else ""
        ),
        "component_validation_feature_date_end": (
            to_date_str(cv_row["feature_date_end"]) if pd.notna(cv_row["feature_date_end"]) else ""
        ),
        "component_validation_target_date_start": (
            to_date_str(cv_row["target_date_start"]) if pd.notna(cv_row["target_date_start"]) else ""
        ),
        "component_validation_target_date_end": (
            to_date_str(cv_row["target_date_end"]) if pd.notna(cv_row["target_date_end"]) else ""
        ),
    }

    # Load component plans/configs.
    enet_plan = json.loads(ENET_SELECTION_PLAN_PATH.read_text(encoding="utf-8"))
    enet_candidates = enet_plan["candidate_configs"]
    enet_feature_cols = [str(c) for c in enet_plan["feature_set_by_horizon"][str(H)]]

    hgbr_plan = json.loads(HGBR_SELECTION_PLAN_PATH.read_text(encoding="utf-8"))
    hgbr_space = hgbr_plan["candidate_space_by_horizon"][str(H)]
    hgbr_budget = int(hgbr_plan["search_budget"][str(H)])
    hgbr_seed = int(hgbr_plan["random_seed"]["candidate_sampling_seed_by_horizon"][str(H)])
    hgbr_model_seed = int(hgbr_plan["random_seed"]["model_fit_random_state"])
    hgbr_feature_info = hgbr_plan["feature_set_by_horizon"][str(H)]
    hgbr_npz_path = ROOT / str(hgbr_feature_info["feature_npz_path"])
    hgbr_expected_feature_count = int(hgbr_feature_info.get("expected_feature_count", 0))

    bcr_cfg = json.loads(BCR_FIXED_CONFIG_PATH.read_text(encoding="utf-8"))
    if int(bcr_cfg["horizon"]) != H:
        raise RuntimeError("BCR fixed config horizon mismatch")

    # Build feature frames.
    enet_frame = build_elasticnet_feature_frame(dataset_df, date_col, target_col, H, enet_feature_cols)
    hgbr_frame, hgbr_feature_cols = load_hgbr_feature_frame(
        npz_path=hgbr_npz_path,
        horizon=H,
        date_to_target=date_to_target,
        expected_feature_count=hgbr_expected_feature_count,
    )
    bcr_bundle = load_bcr_feature_bundle(ROOT / "features" / "ulsan_H5_features_v2.npz", H, date_to_target)

    bcr_X = np.asarray(bcr_bundle["X"], dtype=np.float32)
    bcr_y = np.asarray(bcr_bundle["y"], dtype=np.float32)
    bcr_feature_dates = np.asarray(bcr_bundle["feature_dates"])
    bcr_target_dates = np.asarray(bcr_bundle["target_dates"])
    bcr_frame = bcr_bundle["feature_frame"].copy()

    # Component selection stage.
    selection_rows: List[Dict[str, Any]] = []
    selected_enet: Dict[int, Dict[str, Any]] = {}
    selected_hgbr: Dict[int, Dict[str, Any]] = {}
    selected_bcr: Dict[int, Dict[str, Any]] = {}
    bcr_gate_rows: List[Dict[str, Any]] = []
    bcr_gate_failure: Optional[Dict[str, Any]] = None

    for fold in OUTER_FOLDS:
        fold_info = fold_data[fold]
        sub_keys = fold_info["component_subtrain"]
        val_keys = fold_info["component_validation"]

        # Events in component validation.
        val_y = val_keys["target_date"].map(date_to_target).to_numpy(dtype=float)
        ev15 = int((val_y >= 15.0).sum())
        ev16 = int((val_y >= 16.0).sum())
        ev17 = int((val_y >= 17.0).sum())

        selection_start = to_date_str(sub_keys["feature_date"].min())
        selection_end = to_date_str(val_keys["feature_date"].max())

        # ElasticNet selection.
        enet_sub = extract_partition(enet_frame, sub_keys, "enet_subtrain", fold)
        enet_val = extract_partition(enet_frame, val_keys, "enet_validation", fold)

        X_sub = enet_sub[enet_feature_cols].to_numpy(dtype=float)
        y_sub = enet_sub["y_true"].to_numpy(dtype=float)
        X_val = enet_val[enet_feature_cols].to_numpy(dtype=float)
        y_val = enet_val["y_true"].to_numpy(dtype=float)

        enet_eval_rows: List[Dict[str, Any]] = []
        for cfg in enet_candidates:
            alpha = float(cfg["alpha"])
            l1_ratio = float(cfg["l1_ratio"])
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "elasticnet",
                        ElasticNet(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            fit_intercept=True,
                            max_iter=20000,
                            tol=1e-4,
                            random_state=SEED,
                        ),
                    ),
                ]
            )
            pipe.fit(X_sub, y_sub)
            pred_val = pipe.predict(X_val)
            mae = float(np.mean(np.abs(y_val - pred_val)))
            enet_eval_rows.append(
                {
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "mae": mae,
                    "candidate_name": "enet_a{}_l{}".format(alpha, l1_ratio),
                }
            )

        best_mae = min(r["mae"] for r in enet_eval_rows)
        tied = [r for r in enet_eval_rows if abs(r["mae"] - best_mae) <= TIE_TOL]
        enet_selected = sorted(tied, key=lambda r: (float(r["alpha"]), float(r["l1_ratio"])))[0]

        selected_enet[fold] = {
            "alpha": float(enet_selected["alpha"]),
            "l1_ratio": float(enet_selected["l1_ratio"]),
            "selection_metric": "inner_validation_MAE",
            "selection_value": float(enet_selected["mae"]),
        }

        selection_rows.append(
            {
                "outer_fold": int(fold),
                "component": "ElasticNet",
                "candidate_space": param_to_json(enet_candidates),
                "selected_configuration": param_to_json(selected_enet[fold]),
                "selection_objective": "minimize_inner_validation_MAE_tie_smallest_alpha_then_l1_ratio",
                "component_subtrain_N": int(len(enet_sub)),
                "component_validation_N": int(len(enet_val)),
                "component_validation_events_tau15": int(ev15),
                "component_validation_events_tau16": int(ev16),
                "component_validation_events_tau17": int(ev17),
                "selection_start_date": selection_start,
                "selection_end_date": selection_end,
                "hybrid_validation_used": False,
                "outer_test_used": False,
                "future_fold_used": False,
                "selection_decision": "selected_min_inner_validation_MAE",
                "run_id": run_id,
                "checkpoint_selection_source": "not_applicable",
            }
        )

        # HGBR selection.
        hgbr_sub = extract_partition(hgbr_frame, sub_keys, "hgbr_subtrain", fold)
        hgbr_val = extract_partition(hgbr_frame, val_keys, "hgbr_validation", fold)

        X_sub_h = hgbr_sub[hgbr_feature_cols].to_numpy(dtype=float)
        y_sub_h = hgbr_sub["y_true"].to_numpy(dtype=float)
        X_val_h = hgbr_val[hgbr_feature_cols].to_numpy(dtype=float)
        y_val_h = hgbr_val["y_true"].to_numpy(dtype=float)

        candidates = build_hgbr_candidates(hgbr_space, hgbr_budget, hgbr_seed, H)
        hgbr_eval_rows: List[Dict[str, Any]] = []

        for c in candidates:
            params = dict(c["candidate_parameters"])
            if "random_state" in params:
                params["random_state"] = int(hgbr_model_seed)
            model = HistGradientBoostingRegressor(**params)
            model.fit(X_sub_h, y_sub_h)
            pred_val = model.predict(X_val_h)
            mae = float(np.mean(np.abs(y_val_h - pred_val)))
            hgbr_eval_rows.append(
                {
                    "candidate_configuration_id": str(c["candidate_configuration_id"]),
                    "candidate_parameters": params,
                    "mae": mae,
                }
            )

        best_mae_h = min(r["mae"] for r in hgbr_eval_rows)
        tied_h = [r for r in hgbr_eval_rows if abs(r["mae"] - best_mae_h) <= TIE_TOL]
        selected_h = sorted(tied_h, key=lambda r: str(r["candidate_configuration_id"]))[0]

        selected_hgbr[fold] = {
            "candidate_configuration_id": str(selected_h["candidate_configuration_id"]),
            "parameters": selected_h["candidate_parameters"],
            "selection_metric": "inner_validation_MAE",
            "selection_value": float(selected_h["mae"]),
        }

        selection_rows.append(
            {
                "outer_fold": int(fold),
                "component": "HGBR",
                "candidate_space": param_to_json(
                    {
                        "candidate_space": hgbr_space,
                        "search_budget": hgbr_budget,
                        "search_seed": hgbr_seed,
                        "model_fit_random_state": hgbr_model_seed,
                    }
                ),
                "selected_configuration": param_to_json(selected_hgbr[fold]),
                "selection_objective": "minimize_inner_validation_MAE_tie_smallest_candidate_configuration_id",
                "component_subtrain_N": int(len(hgbr_sub)),
                "component_validation_N": int(len(hgbr_val)),
                "component_validation_events_tau15": int(ev15),
                "component_validation_events_tau16": int(ev16),
                "component_validation_events_tau17": int(ev17),
                "selection_start_date": selection_start,
                "selection_end_date": selection_end,
                "hybrid_validation_used": False,
                "outer_test_used": False,
                "future_fold_used": False,
                "selection_decision": "selected_min_inner_validation_MAE",
                "run_id": run_id,
                "checkpoint_selection_source": "not_applicable",
            }
        )

        # BCR selection.
        bcr_sub = extract_partition(bcr_frame, sub_keys, "bcr_subtrain", fold)
        bcr_val = extract_partition(bcr_frame, val_keys, "bcr_validation", fold)

        bcr_sub_idx = bcr_sub["row_index"].astype(int).to_numpy()
        bcr_val_idx = bcr_val["row_index"].astype(int).to_numpy()

        bcr_val_n = int(len(bcr_val_idx))
        bcr_k_r005 = int(math.ceil(0.05 * float(bcr_val_n))) if bcr_val_n > 0 else 0
        bcr_tau16_events = int((bcr_y[bcr_val_idx] >= 16.0).sum()) if bcr_val_n > 0 else 0
        bcr_recall_defined = bool(bcr_tau16_events > 0 and bcr_k_r005 > 0)
        bcr_blocked = not bcr_recall_defined
        gate_notes = "ok"
        if bcr_tau16_events <= 0:
            gate_notes = "tau16_event_count_zero; recall_at_5pct_undefined; no_checkpoint_selection_permitted"
        elif bcr_k_r005 <= 0:
            gate_notes = "k_r005_zero; recall_at_5pct_undefined; no_checkpoint_selection_permitted"

        print(
            "[BCR-Gate][Fold {}] component_validation_N={} tau16_events={} k_r005={} recall_defined={}".format(
                fold,
                bcr_val_n,
                bcr_tau16_events,
                bcr_k_r005,
                bcr_recall_defined,
            )
        )

        bcr_gate_rows.append(
            {
                "outer_fold": int(fold),
                "component_validation_N": int(bcr_val_n),
                "component_validation_events_tau16": int(bcr_tau16_events),
                "component_validation_k_r005": int(bcr_k_r005),
                "recall_at_5pct_defined": bool(bcr_recall_defined),
                "blocked_for_bcr_checkpoint_selection": bool(bcr_blocked),
                "gate_notes": gate_notes,
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "nested_split_id": fold_info["nested_split_id"],
                "run_id": run_id,
            }
        )

        if bcr_blocked:
            selection_rows.append(
                {
                    "outer_fold": int(fold),
                    "component": "BCR-TCN",
                    "candidate_space": param_to_json({"fixed_configuration": bcr_cfg}),
                    "selected_configuration": "not_selected_guard_block",
                    "selection_objective": "checkpoint_selection_blocked_recall_at_5pct_undefined",
                    "component_subtrain_N": int(len(bcr_sub)),
                    "component_validation_N": int(len(bcr_val)),
                    "component_validation_events_tau15": int(ev15),
                    "component_validation_events_tau16": int(ev16),
                    "component_validation_events_tau17": int(ev17),
                    "selection_start_date": selection_start,
                    "selection_end_date": selection_end,
                    "hybrid_validation_used": False,
                    "outer_test_used": False,
                    "future_fold_used": False,
                    "selection_decision": "blocked_no_tau16_event_support_for_recall_at_5pct",
                    "run_id": run_id,
                    "checkpoint_selection_source": "blocked_no_positive_events_component_validation",
                }
            )
            bcr_gate_failure = {
                "outer_fold": int(fold),
                "component_validation_N": int(bcr_val_n),
                "component_validation_events_tau16": int(bcr_tau16_events),
                "component_validation_k_r005": int(bcr_k_r005),
                "reason": gate_notes,
            }
            break

        bcr_sel = run_bcr_component_selection(
            X=bcr_X,
            y=bcr_y,
            feature_dates=bcr_feature_dates,
            target_dates=bcr_target_dates,
            subtrain_idx=bcr_sub_idx,
            validation_idx=bcr_val_idx,
            cfg=bcr_cfg,
            fold=fold,
        )

        selected_bcr[fold] = {
            "selected_epoch": int(bcr_sel["selected_epoch"]),
            "training_epochs_completed": int(bcr_sel["training_epochs_completed"]),
            "best_validation_recall_tau16_r05": float(bcr_sel["best_validation_recall_tau16_r05"]),
            "best_validation_mae": float(bcr_sel["best_validation_mae"]),
        }

        selection_rows.append(
            {
                "outer_fold": int(fold),
                "component": "BCR-TCN",
                "candidate_space": param_to_json({"fixed_configuration": bcr_cfg}),
                "selected_configuration": param_to_json(selected_bcr[fold]),
                "selection_objective": "maximize_validation_recall_tau16_r05_tie_earliest_epoch",
                "component_subtrain_N": int(len(bcr_sub)),
                "component_validation_N": int(len(bcr_val)),
                "component_validation_events_tau15": int(ev15),
                "component_validation_events_tau16": int(ev16),
                "component_validation_events_tau17": int(ev17),
                "selection_start_date": selection_start,
                "selection_end_date": selection_end,
                "hybrid_validation_used": False,
                "outer_test_used": False,
                "future_fold_used": False,
                "selection_decision": "selected_fixed_configuration_with_component_validation_checkpoint",
                "run_id": run_id,
                "checkpoint_selection_source": "component_validation_only",
            }
        )

        # Persistence.
        selection_rows.append(
            {
                "outer_fold": int(fold),
                "component": "Persistence",
                "candidate_space": param_to_json({"rule": "y_pred = TNout at feature_date"}),
                "selected_configuration": param_to_json({"rule": "feature_date_target_lag_h5"}),
                "selection_objective": "no_fit_rule",
                "component_subtrain_N": int(len(sub_keys)),
                "component_validation_N": int(len(val_keys)),
                "component_validation_events_tau15": int(ev15),
                "component_validation_events_tau16": int(ev16),
                "component_validation_events_tau17": int(ev17),
                "selection_start_date": selection_start,
                "selection_end_date": selection_end,
                "hybrid_validation_used": False,
                "outer_test_used": False,
                "future_fold_used": False,
                "selection_decision": "deterministic_no_fit",
                "run_id": run_id,
                "checkpoint_selection_source": "not_applicable",
            }
        )

    selection_df = pd.DataFrame(selection_rows).sort_values(["outer_fold", "component"]).reset_index(drop=True)
    to_csv_with_dates(selection_df, COMPONENT_SELECTION_PATH, ["selection_start_date", "selection_end_date"])

    bcr_gate_df = pd.DataFrame(bcr_gate_rows).sort_values("outer_fold").reset_index(drop=True)
    to_csv_with_dates(bcr_gate_df, BCR_GATE_PATH, [])

    if bcr_gate_failure is not None:
        protected_after = snapshot_tree_checksums(PROTECTED_DIRS)
        protected_changed = []
        for rel, before_sha in protected_before.items():
            after_sha = protected_after.get(rel)
            if after_sha != before_sha:
                protected_changed.append(rel)

        if protected_changed:
            raise RuntimeError("Protected source/canonical artifacts changed: {}".format(protected_changed[:10]))

        out_files = [p for p in OUT_DIR.iterdir() if p.is_file()]

        limitations = [
            "Submitted tuned HybridRank pooled non-held-out folds, allowing future chronology to influence earlier folds.",
            "The revised procedure separates component selection and ensemble selection.",
            "Five-day target-date embargoes separate every stage.",
            "The fixed policy is used at r <= 0.05.",
            "The fixed policy is also used when HybridRank-validation event support is insufficient.",
            "Full outer-test fold rank normalization remains retrospective and non-deployable.",
            "A separate sequential policy will be evaluated later.",
            "Hybrid-specific component predictions are not substituted for the canonical point-forecast package.",
            "No outer-test alarm metrics were calculated in this stage.",
            "BCR-TCN component-validation Recall@5% checkpoint selection was blocked because tau16 event support was insufficient.",
        ]

        manifest = {
            "run_id": run_id,
            "purpose": "Fully chronological and fully nested H5 HybridRank pathway",
            "horizon": int(H),
            "nested_split_definition": {
                "component_validation_fraction": float(COMPONENT_VALIDATION_FRACTION),
                "hybrid_validation_fraction": float(HYBRID_VALIDATION_FRACTION),
                "deterministic_tail_rounding": "floor_with_min1_and_nonempty_guard",
                "embargo_days": int(EMBARGO_DAYS),
            },
            "component_validation_fraction": float(COMPONENT_VALIDATION_FRACTION),
            "hybrid_validation_fraction": float(HYBRID_VALIDATION_FRACTION),
            "embargo_days": int(EMBARGO_DAYS),
            "event_support_rule": {
                "minimum_events_required": int(SUPPORT_MIN_EVENTS),
                "minimum_alarm_slots_required": int(SUPPORT_MIN_ALARM_SLOTS),
                "guard_fixed_policy_condition": "r <= 0.05",
            },
            "guard_rule": "fixed_policy_when_r_le_0.05",
            "component_models": list(COMPONENT_ORDER),
            "candidate_spaces": {
                "ElasticNet": enet_candidates,
                "HGBR": {
                    "candidate_space_h5": hgbr_space,
                    "search_budget_h5": hgbr_budget,
                    "candidate_sampling_seed_h5": hgbr_seed,
                    "model_fit_random_state": hgbr_model_seed,
                },
                "BCR-TCN": bcr_cfg,
                "Persistence": {"rule": "y_pred = TNout at feature_date"},
            },
            "selection_objectives": {
                "ElasticNet": "min_inner_validation_MAE_tie_smallest_alpha_then_l1_ratio",
                "HGBR": "min_inner_validation_MAE_tie_smallest_candidate_configuration_id",
                "BCR-TCN": "max_validation_recall_tau16_r05_tie_earliest_epoch",
                "HybridRank": "precision_floor_then_lexicographic_recall_precision_tp",
            },
            "dataset_sha256": dataset_sha,
            "split_sha256": split_sha,
            "canonical_assembly_id": canonical_manifest["assembly_id"],
            "canonical_checksums": canonical_checks,
            "git_branch": active_branch,
            "git_commit": git_commit,
            "python_executable": python_executable,
            "python_version": python_version,
            "execution_command": "{} {}".format(sys.executable, Path(__file__).name),
            "timestamp": utc_now_iso(),
            "software_versions": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
            "output_files": sorted([p.name for p in out_files]),
            "output_checksums": compute_output_checksums(OUT_DIR),
            "test_result": "not_run",
            "deterministic_result": "not_applicable_blocked_before_full_pipeline",
            "files_modified_outside_authorized_directory": outside_changes,
            "limitations": limitations,
            "protected_source_checksum_snapshot": {
                "before": protected_before,
                "after": protected_after,
                "changed": protected_changed,
            },
            "bcr_checkpoint_gate": {
                "rows": bcr_gate_df.to_dict("records"),
                "blocked_folds": [
                    int(x)
                    for x in bcr_gate_df[
                        bcr_gate_df["blocked_for_bcr_checkpoint_selection"].astype(bool)
                    ]["outer_fold"].tolist()
                ],
                "policy": "if tau16_events==0 or k_r005==0 then stop fold and do not select checkpoint",
                "failure": bcr_gate_failure,
            },
            "fold1_tau16_support_diagnostic": fold1_tau16_diag_summary,
        }

        write_json(MANIFEST_PATH, manifest)

        report_text = build_completion_report(
            run_id=run_id,
            dataset_sha=dataset_sha,
            split_sha=split_sha,
            git_commit=git_commit,
            summary_df=summary_df,
            selection_df=selection_df,
            support_df=pd.DataFrame(),
            decision="C",
            bcr_gate_df=bcr_gate_df,
            fold1_tau16_diag_summary=fold1_tau16_diag_summary,
            decision_note="Fold {} blocked before BCR training: {}".format(
                int(bcr_gate_failure["outer_fold"]), bcr_gate_failure["reason"]
            ),
        )
        ensure_inside_authorized(COMPLETION_REPORT_PATH)
        COMPLETION_REPORT_PATH.write_text(report_text, encoding="utf-8")

        write_checksums_file(OUT_DIR, CHECKSUMS_PATH)

        print("1. working directory: {}".format(Path.cwd().resolve()))
        print("2. branch and Git commit: {} {}".format(active_branch, git_commit))
        print("3. Python executable and version: {} {}".format(python_executable, python_version))
        print("4. input-verification result: {}".format(input_verification["verification_pass"]))
        print("5. nested split counts by fold: see {}".format(NESTED_SUMMARY_PATH.name))
        print("6. embargo exclusions by boundary and fold: see {}".format(NESTED_SUMMARY_PATH.name))
        print("7. component configurations selected: partial, see {}".format(COMPONENT_SELECTION_PATH.name))
        print("8. BCR-TCN selected epoch by fold: blocked before selection for fold {}".format(int(bcr_gate_failure["outer_fold"])))
        print("   Fold 1 tau16 support diagnostic artifact: see {}".format(FOLD1_TAU16_DIAGNOSTIC_PATH.name))
        print("9. HybridRank-validation event support: not reached")
        print("10. tuned versus fixed decisions by fold, threshold, and budget: not reached")
        print("11. frozen weights: not produced")
        print("12. hybrid-validation prediction counts: not produced")
        print("13. outer-test prediction count: not produced")
        print("14. test and future-fold isolation: preselection checks passed")
        print("15. source preservation: {}".format(len(protected_changed) == 0))
        print("16. deterministic result: not_applicable_blocked_before_full_pipeline")
        print("17. passed assertions: pending_test_script")
        print("18. final decision: C")
        print("19. exactly one next action: Investigate component-validation tau16 event support and rerun")
        return

    # Hybrid-validation component predictions (refit on complete component-development).
    hv_rows: List[pd.DataFrame] = []

    for fold in OUTER_FOLDS:
        fold_info = fold_data[fold]
        dev_keys = fold_info["component_development"]
        hv_keys = fold_info["hybrid_validation"]

        base_hv = hv_keys[["feature_date", "target_date"]].drop_duplicates().copy().sort_values("feature_date")
        base_hv["outer_fold"] = int(fold)
        base_hv["y_true"] = base_hv["target_date"].map(date_to_target)

        if base_hv["y_true"].isna().any():
            raise RuntimeError("Hybrid-validation y_true missing for fold {}".format(fold))

        base_hv["event_tau15"] = (base_hv["y_true"] >= 15.0).astype(int)
        base_hv["event_tau16"] = (base_hv["y_true"] >= 16.0).astype(int)
        base_hv["event_tau17"] = (base_hv["y_true"] >= 17.0).astype(int)

        # ElasticNet refit and predict HV.
        enet_dev = extract_partition(enet_frame, dev_keys, "enet_dev", fold)
        enet_hv = extract_partition(enet_frame, hv_keys, "enet_hv", fold)

        enet_cfg = selected_enet[fold]
        enet_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNet(
                        alpha=float(enet_cfg["alpha"]),
                        l1_ratio=float(enet_cfg["l1_ratio"]),
                        fit_intercept=True,
                        max_iter=20000,
                        tol=1e-4,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        enet_pipe.fit(enet_dev[enet_feature_cols].to_numpy(dtype=float), enet_dev["y_true"].to_numpy(dtype=float))
        enet_pred = enet_pipe.predict(enet_hv[enet_feature_cols].to_numpy(dtype=float))
        enet_hv_pred = enet_hv[["feature_date", "target_date"]].copy()
        enet_hv_pred["elasticnet_y_pred"] = enet_pred

        # HGBR refit and predict HV.
        hgbr_dev = extract_partition(hgbr_frame, dev_keys, "hgbr_dev", fold)
        hgbr_hv = extract_partition(hgbr_frame, hv_keys, "hgbr_hv", fold)

        hgbr_cfg = selected_hgbr[fold]
        hg_model = HistGradientBoostingRegressor(**dict(hgbr_cfg["parameters"]))
        hg_model.fit(hgbr_dev[hgbr_feature_cols].to_numpy(dtype=float), hgbr_dev["y_true"].to_numpy(dtype=float))
        hg_pred = hg_model.predict(hgbr_hv[hgbr_feature_cols].to_numpy(dtype=float))
        hg_hv_pred = hgbr_hv[["feature_date", "target_date"]].copy()
        hg_hv_pred["hgbr_y_pred"] = hg_pred

        # BCR refit fixed epoch and predict HV.
        bcr_dev = extract_partition(bcr_frame, dev_keys, "bcr_dev", fold)
        bcr_hv = extract_partition(bcr_frame, hv_keys, "bcr_hv", fold)
        bcr_epochs = int(selected_bcr[fold]["selected_epoch"]) + 1

        bcr_hv_pred = run_bcr_fixed_epochs_predict(
            X=bcr_X,
            y=bcr_y,
            feature_dates=bcr_feature_dates,
            target_dates=bcr_target_dates,
            train_idx=bcr_dev["row_index"].astype(int).to_numpy(),
            predict_idx=bcr_hv["row_index"].astype(int).to_numpy(),
            epochs_to_train=bcr_epochs,
            cfg=bcr_cfg,
            fold=fold,
            stage_offset=300,
        )

        # Persistence predictions.
        per_hv_pred = base_hv[["feature_date", "target_date"]].copy()
        per_hv_pred["persistence_y_pred"] = per_hv_pred["feature_date"].map(date_to_target)
        if per_hv_pred["persistence_y_pred"].isna().any():
            raise RuntimeError("Persistence HV prediction missing feature_date map for fold {}".format(fold))

        merged = base_hv.merge(bcr_hv_pred.drop(columns=["row_index", "y_true"]), on=["feature_date", "target_date"], how="left")
        merged = merged.merge(enet_hv_pred, on=["feature_date", "target_date"], how="left")
        merged = merged.merge(per_hv_pred, on=["feature_date", "target_date"], how="left")
        merged = merged.merge(hg_hv_pred, on=["feature_date", "target_date"], how="left")

        if merged[["bcr_tcn_y_pred", "bcr_tcn_p_tau15", "bcr_tcn_p_tau16", "bcr_tcn_p_tau17", "elasticnet_y_pred", "persistence_y_pred", "hgbr_y_pred"]].isna().any().any():
            raise RuntimeError("Hybrid-validation component prediction merge produced missing values for fold {}".format(fold))

        merged["prediction_context"] = "independent_hybrid_weight_validation"
        merged["component_development_start"] = fold_info["component_development"]["feature_date"].min()
        merged["component_development_end"] = fold_info["component_development"]["feature_date"].max()
        merged["hybrid_validation_start"] = fold_info["hybrid_validation"]["feature_date"].min()
        merged["hybrid_validation_end"] = fold_info["hybrid_validation"]["feature_date"].max()
        merged["dataset_sha256"] = dataset_sha
        merged["split_sha256"] = split_sha
        merged["nested_split_id"] = fold_info["nested_split_id"]
        merged["run_id"] = run_id
        merged["bcr_refit_early_stop_used"] = False
        merged["weight_selection_partition"] = "hybrid_validation_only"

        hv_rows.append(merged)

    hv_pred_df = pd.concat(hv_rows, ignore_index=True).sort_values(["outer_fold", "feature_date"]).reset_index(drop=True)

    if hv_pred_df.duplicated(subset=["outer_fold", "feature_date", "target_date"]).any():
        raise RuntimeError("Duplicate keys detected in hybrid-validation component predictions")

    to_csv_with_dates(
        hv_pred_df,
        HV_COMPONENT_PRED_PATH,
        [
            "feature_date",
            "target_date",
            "component_development_start",
            "component_development_end",
            "hybrid_validation_start",
            "hybrid_validation_end",
        ],
    )

    # Hybrid-validation rank scores.
    rank_rows: List[Dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        fold_hv = hv_pred_df[hv_pred_df["outer_fold"] == fold].copy().sort_values("feature_date")
        if fold_hv.empty:
            raise RuntimeError("No hybrid-validation predictions for fold {}".format(fold))

        for tau in TAUS:
            bcr_raw_col = "bcr_tcn_p_tau{}".format(int(tau))
            bcr_rank = rank01(fold_hv[bcr_raw_col])
            enet_rank = rank01(fold_hv["elasticnet_y_pred"])
            per_rank = rank01(fold_hv["persistence_y_pred"])
            hg_rank = rank01(fold_hv["hgbr_y_pred"])

            for idx, row in fold_hv.reset_index(drop=True).iterrows():
                rank_rows.append(
                    {
                        "outer_fold": int(fold),
                        "feature_date": row["feature_date"],
                        "target_date": row["target_date"],
                        "y_true": float(row["y_true"]),
                        "event_tau15": int(row["event_tau15"]),
                        "event_tau16": int(row["event_tau16"]),
                        "event_tau17": int(row["event_tau17"]),
                        "threshold": int(tau),
                        "bcr_tcn_raw_score": float(row[bcr_raw_col]),
                        "bcr_tcn_rank_score": float(bcr_rank[idx]),
                        "elasticnet_raw_score": float(row["elasticnet_y_pred"]),
                        "elasticnet_rank_score": float(enet_rank[idx]),
                        "persistence_raw_score": float(row["persistence_y_pred"]),
                        "persistence_rank_score": float(per_rank[idx]),
                        "hgbr_raw_score": float(row["hgbr_y_pred"]),
                        "hgbr_rank_score": float(hg_rank[idx]),
                        "rank_method": "average_ties",
                        "rank_denominator": "N-1",
                        "rank_scope": "component_wise_within_outer_fold",
                        "used_outcomes": False,
                        "dataset_sha256": dataset_sha,
                        "split_sha256": split_sha,
                        "nested_split_id": fold_data[fold]["nested_split_id"],
                        "run_id": run_id,
                    }
                )

    hv_rank_df = pd.DataFrame(rank_rows).sort_values(["outer_fold", "threshold", "feature_date"]).reset_index(drop=True)
    to_csv_with_dates(hv_rank_df, HV_RANK_PATH, ["feature_date", "target_date"])

    # Event-support table.
    support_rows: List[Dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        for tau in TAUS:
            hv_tau = hv_rank_df[(hv_rank_df["outer_fold"] == fold) & (hv_rank_df["threshold"] == int(tau))].copy()
            if hv_tau.empty:
                raise RuntimeError("Missing HV rank rows for fold {} tau {}".format(fold, int(tau)))
            n = int(len(hv_tau))
            event = (hv_tau["y_true"].to_numpy(dtype=float) >= float(tau)).astype(int)
            event_count = int(event.sum())
            prevalence = float(event_count / n) if n > 0 else 0.0

            for r in BUDGETS:
                k = max(1, int(math.ceil(float(r) * float(n))))
                support_pass = bool(event_count >= SUPPORT_MIN_EVENTS and k >= SUPPORT_MIN_ALARM_SLOTS)
                tuned_permitted = bool((r > 0.05) and support_pass)
                reason = ""
                if float(r) <= 0.05:
                    reason = "guard_r_le_0.05_fixed_policy"
                elif not support_pass:
                    reason = "insufficient_events_or_alarm_slots_fixed_policy"

                support_rows.append(
                    {
                        "outer_fold": int(fold),
                        "threshold": int(tau),
                        "alarm_budget": float(r),
                        "hybrid_validation_N": int(n),
                        "event_count": int(event_count),
                        "prevalence": float(prevalence),
                        "k": int(k),
                        "minimum_events_required": int(SUPPORT_MIN_EVENTS),
                        "minimum_alarm_slots_required": int(SUPPORT_MIN_ALARM_SLOTS),
                        "tuned_weight_selection_permitted": bool(tuned_permitted),
                        "fallback_reason": reason,
                        "dataset_sha256": dataset_sha,
                        "split_sha256": split_sha,
                        "nested_split_id": fold_data[fold]["nested_split_id"],
                        "run_id": run_id,
                    }
                )

    support_df = pd.DataFrame(support_rows).sort_values(["outer_fold", "threshold", "alarm_budget"]).reset_index(drop=True)
    to_csv_with_dates(support_df, HV_SUPPORT_PATH, [])

    # Weight selection.
    weight_grid = build_weight_grid(len(COMPONENT_ORDER), step=0.1)
    candidate_grid_text = "0.0_to_1.0_step_0.1"
    enumeration_text = "product(values,repeat=4)->skip_zero->normalize_sum1"
    tie_rule = "lexicographic_recall_precision_tp_first_encountered_on_exact_tie"

    weight_rows: List[Dict[str, Any]] = []
    weight_lookup: Dict[Tuple[int, int, float, str], Dict[str, Any]] = {}

    for fold in OUTER_FOLDS:
        for tau in TAUS:
            hv_tau = hv_rank_df[(hv_rank_df["outer_fold"] == fold) & (hv_rank_df["threshold"] == int(tau))].copy()
            hv_tau = hv_tau.sort_values("feature_date").reset_index(drop=True)

            yv = hv_tau["y_true"].to_numpy(dtype=float)
            event = (yv >= float(tau)).astype(int)

            comp_rank_arrays = {
                "BCR-TCN": hv_tau["bcr_tcn_rank_score"].to_numpy(dtype=float),
                "ElasticNet": hv_tau["elasticnet_rank_score"].to_numpy(dtype=float),
                "Persistence": hv_tau["persistence_rank_score"].to_numpy(dtype=float),
                "HGBR": hv_tau["hgbr_rank_score"].to_numpy(dtype=float),
            }

            fixed_vector = tuple(float(FIXED_WEIGHTS[c]) for c in COMPONENT_ORDER)

            for r in BUDGETS:
                support_row = support_df[
                    (support_df["outer_fold"] == fold)
                    & (support_df["threshold"] == int(tau))
                    & (np.isclose(support_df["alarm_budget"].astype(float), float(r)))
                ]
                if support_row.empty:
                    raise RuntimeError("Missing support row for fold {} tau {} r {}".format(fold, int(tau), r))
                support_row = support_row.iloc[0]

                fixed_score = np.zeros(len(hv_tau), dtype=float)
                for wi, comp in zip(fixed_vector, COMPONENT_ORDER):
                    fixed_score += float(wi) * comp_rank_arrays[comp]
                fixed_metric = alarm_metrics_from_scores(yv, fixed_score, tau=tau, budget_r=float(r))
                precision_floor = float(fixed_metric["precision"])

                tuned_vector = fixed_vector
                tuned_metric = dict(fixed_metric)
                tuned_weight_source = "fixed_policy_due_guard_or_support"
                fallback_applied = True
                fallback_reason = str(support_row["fallback_reason"]) if str(support_row["fallback_reason"]) else "guard_or_support_rule"

                can_tune = bool(support_row["tuned_weight_selection_permitted"])
                if can_tune:
                    best_obj: Optional[Tuple[float, float, float]] = None
                    best_vec: Optional[Tuple[float, ...]] = None
                    best_metric: Optional[Dict[str, float]] = None

                    for vec in weight_grid:
                        score = np.zeros(len(hv_tau), dtype=float)
                        for wi, comp in zip(vec, COMPONENT_ORDER):
                            score += float(wi) * comp_rank_arrays[comp]
                        m = alarm_metrics_from_scores(yv, score, tau=tau, budget_r=float(r))
                        if float(m["precision"]) + 1e-12 < precision_floor:
                            continue

                        obj = (float(m["recall"]), float(m["precision"]), float(m["TP"]))
                        if best_obj is None or obj > best_obj:
                            best_obj = obj
                            best_vec = vec
                            best_metric = m

                    if best_vec is not None and best_metric is not None:
                        tuned_vector = best_vec
                        tuned_metric = best_metric
                        tuned_weight_source = "nested_hybrid_validation_tuned"
                        fallback_applied = False
                        fallback_reason = ""
                    else:
                        tuned_vector = fixed_vector
                        tuned_metric = fixed_metric
                        tuned_weight_source = "fixed_policy_due_precision_floor_no_candidate"
                        fallback_applied = True
                        fallback_reason = "precision_floor_no_candidate_passed"

                policy_vectors: Dict[str, Tuple[float, ...]] = {
                    "HybridRank_fixed": fixed_vector,
                    "HybridRank_nested_tuned": tuned_vector,
                    "HybridRank_nested_guarded": tuned_vector,
                }
                policy_sources: Dict[str, str] = {
                    "HybridRank_fixed": "fixed_policy",
                    "HybridRank_nested_tuned": tuned_weight_source,
                    "HybridRank_nested_guarded": tuned_weight_source,
                }
                policy_metrics: Dict[str, Dict[str, float]] = {
                    "HybridRank_fixed": fixed_metric,
                    "HybridRank_nested_tuned": tuned_metric,
                    "HybridRank_nested_guarded": tuned_metric,
                }
                policy_guard: Dict[str, bool] = {
                    "HybridRank_fixed": False,
                    "HybridRank_nested_tuned": False,
                    "HybridRank_nested_guarded": False,
                }
                policy_fallback: Dict[str, bool] = {
                    "HybridRank_fixed": False,
                    "HybridRank_nested_tuned": bool(fallback_applied),
                    "HybridRank_nested_guarded": bool(fallback_applied),
                }
                policy_reason: Dict[str, str] = {
                    "HybridRank_fixed": "",
                    "HybridRank_nested_tuned": fallback_reason,
                    "HybridRank_nested_guarded": fallback_reason,
                }

                if float(r) <= 0.05:
                    policy_vectors["HybridRank_nested_tuned"] = fixed_vector
                    policy_sources["HybridRank_nested_tuned"] = "fixed_policy_due_guard_r_le_0.05"
                    policy_metrics["HybridRank_nested_tuned"] = fixed_metric
                    policy_fallback["HybridRank_nested_tuned"] = True
                    policy_reason["HybridRank_nested_tuned"] = "guard_r_le_0.05_fixed_policy"

                    policy_vectors["HybridRank_nested_guarded"] = fixed_vector
                    policy_sources["HybridRank_nested_guarded"] = "guard_fixed_policy_r_le_0.05"
                    policy_metrics["HybridRank_nested_guarded"] = fixed_metric
                    policy_guard["HybridRank_nested_guarded"] = True
                    policy_fallback["HybridRank_nested_guarded"] = True
                    policy_reason["HybridRank_nested_guarded"] = "guard_r_le_0.05_fixed_policy"

                for policy_name in ["HybridRank_fixed", "HybridRank_nested_tuned", "HybridRank_nested_guarded"]:
                    vec = policy_vectors[policy_name]
                    m = policy_metrics[policy_name]
                    source = policy_sources[policy_name]
                    guard_applied = policy_guard[policy_name]
                    fb_applied = policy_fallback[policy_name]
                    fb_reason = policy_reason[policy_name]

                    key = (int(fold), int(tau), float(r), policy_name)
                    weight_lookup[key] = {
                        "weights": {comp: float(w) for comp, w in zip(COMPONENT_ORDER, vec)},
                        "weight_source": source,
                        "guard_applied": bool(guard_applied),
                        "fallback_applied": bool(fb_applied),
                        "fallback_reason": fb_reason,
                    }

                    for comp, w in zip(COMPONENT_ORDER, vec):
                        weight_rows.append(
                            {
                                "outer_fold": int(fold),
                                "threshold": int(tau),
                                "alarm_budget": float(r),
                                "policy_name": policy_name,
                                "component": comp,
                                "weight": float(w),
                                "weight_source": source,
                                "hybrid_validation_N": int(m["n"]),
                                "hybrid_validation_events": int(m["events"]),
                                "hybrid_validation_k": int(m["k"]),
                                "event_support_pass": bool(
                                    int(support_row["event_count"]) >= SUPPORT_MIN_EVENTS
                                    and int(support_row["k"]) >= SUPPORT_MIN_ALARM_SLOTS
                                ),
                                "precision_floor": float(precision_floor),
                                "objective_recall": float(m["recall"]),
                                "objective_precision": float(m["precision"]),
                                "objective_tp": int(m["TP"]),
                                "candidate_grid": candidate_grid_text,
                                "candidate_enumeration_order": enumeration_text,
                                "tie_rule": tie_rule,
                                "guard_applied": bool(guard_applied),
                                "fallback_applied": bool(fb_applied),
                                "fallback_reason": fb_reason,
                                "outer_test_outcomes_used": False,
                                "future_fold_outcomes_used": False,
                                "weight_selection_partition": "hybrid_validation_only",
                                "component_selection_frozen": True,
                                "dataset_sha256": dataset_sha,
                                "split_sha256": split_sha,
                                "nested_split_id": fold_data[fold]["nested_split_id"],
                                "run_id": run_id,
                            }
                        )

    weights_df = pd.DataFrame(weight_rows).sort_values(
        ["outer_fold", "threshold", "alarm_budget", "policy_name", "component"]
    ).reset_index(drop=True)

    # Weight sanity checks.
    grp = weights_df.groupby(["outer_fold", "threshold", "alarm_budget", "policy_name"], sort=True)["weight"].sum()
    if not np.allclose(grp.to_numpy(dtype=float), np.ones(len(grp)), atol=1e-9):
        raise RuntimeError("Selected weight vectors do not sum to one")
    if (weights_df["weight"].astype(float) < -1e-12).any():
        raise RuntimeError("Selected weights contain negative values")

    to_csv_with_dates(weights_df, WEIGHTS_PATH, [])

    # Final component refits on full corrected outer-train and predict outer-test.
    outer_rows: List[pd.DataFrame] = []
    for fold in OUTER_FOLDS:
        fold_info = fold_data[fold]
        outer_train_keys = fold_info["outer_train"]
        outer_test_keys = fold_info["outer_test"]

        base_out = outer_test_keys[["feature_date", "target_date"]].drop_duplicates().copy().sort_values("feature_date")
        base_out["outer_fold"] = int(fold)
        base_out["y_true"] = base_out["target_date"].map(date_to_target)
        if base_out["y_true"].isna().any():
            raise RuntimeError("Outer-test y_true missing for fold {}".format(fold))

        base_out["event_tau15"] = (base_out["y_true"] >= 15.0).astype(int)
        base_out["event_tau16"] = (base_out["y_true"] >= 16.0).astype(int)
        base_out["event_tau17"] = (base_out["y_true"] >= 17.0).astype(int)

        # ElasticNet.
        enet_tr = extract_partition(enet_frame, outer_train_keys, "enet_outer_train", fold)
        enet_te = extract_partition(enet_frame, outer_test_keys, "enet_outer_test", fold)

        enet_cfg = selected_enet[fold]
        enet_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNet(
                        alpha=float(enet_cfg["alpha"]),
                        l1_ratio=float(enet_cfg["l1_ratio"]),
                        fit_intercept=True,
                        max_iter=20000,
                        tol=1e-4,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        enet_pipe.fit(enet_tr[enet_feature_cols].to_numpy(dtype=float), enet_tr["y_true"].to_numpy(dtype=float))
        enet_outer_pred = enet_pipe.predict(enet_te[enet_feature_cols].to_numpy(dtype=float))
        enet_out_df = enet_te[["feature_date", "target_date"]].copy()
        enet_out_df["elasticnet_y_pred"] = enet_outer_pred

        # HGBR.
        hg_tr = extract_partition(hgbr_frame, outer_train_keys, "hgbr_outer_train", fold)
        hg_te = extract_partition(hgbr_frame, outer_test_keys, "hgbr_outer_test", fold)

        hg_cfg = selected_hgbr[fold]
        hg_model = HistGradientBoostingRegressor(**dict(hg_cfg["parameters"]))
        hg_model.fit(hg_tr[hgbr_feature_cols].to_numpy(dtype=float), hg_tr["y_true"].to_numpy(dtype=float))
        hg_outer_pred = hg_model.predict(hg_te[hgbr_feature_cols].to_numpy(dtype=float))
        hg_out_df = hg_te[["feature_date", "target_date"]].copy()
        hg_out_df["hgbr_y_pred"] = hg_outer_pred

        # BCR.
        bcr_tr = extract_partition(bcr_frame, outer_train_keys, "bcr_outer_train", fold)
        bcr_te = extract_partition(bcr_frame, outer_test_keys, "bcr_outer_test", fold)
        bcr_epochs = int(selected_bcr[fold]["selected_epoch"]) + 1

        bcr_out_df = run_bcr_fixed_epochs_predict(
            X=bcr_X,
            y=bcr_y,
            feature_dates=bcr_feature_dates,
            target_dates=bcr_target_dates,
            train_idx=bcr_tr["row_index"].astype(int).to_numpy(),
            predict_idx=bcr_te["row_index"].astype(int).to_numpy(),
            epochs_to_train=bcr_epochs,
            cfg=bcr_cfg,
            fold=fold,
            stage_offset=600,
        )

        # Persistence.
        per_out_df = base_out[["feature_date", "target_date"]].copy()
        per_out_df["persistence_y_pred"] = per_out_df["feature_date"].map(date_to_target)
        if per_out_df["persistence_y_pred"].isna().any():
            raise RuntimeError("Persistence outer-test prediction missing feature_date map for fold {}".format(fold))

        merged = base_out.merge(bcr_out_df.drop(columns=["row_index", "y_true"]), on=["feature_date", "target_date"], how="left")
        merged = merged.merge(enet_out_df, on=["feature_date", "target_date"], how="left")
        merged = merged.merge(per_out_df, on=["feature_date", "target_date"], how="left")
        merged = merged.merge(hg_out_df, on=["feature_date", "target_date"], how="left")

        if merged[["bcr_tcn_y_pred", "bcr_tcn_p_tau15", "bcr_tcn_p_tau16", "bcr_tcn_p_tau17", "elasticnet_y_pred", "persistence_y_pred", "hgbr_y_pred"]].isna().any().any():
            raise RuntimeError("Outer-test component prediction merge produced missing values for fold {}".format(fold))

        merged["prediction_context"] = "fully_nested_hybrid_specific_outer_test"
        merged["outer_train_start"] = fold_info["outer_train"]["feature_date"].min()
        merged["outer_train_end"] = fold_info["outer_train"]["feature_date"].max()
        merged["outer_test_start"] = fold_info["outer_test"]["feature_date"].min()
        merged["outer_test_end"] = fold_info["outer_test"]["feature_date"].max()
        merged["dataset_sha256"] = dataset_sha
        merged["split_sha256"] = split_sha
        merged["nested_split_id"] = fold_info["nested_split_id"]
        merged["run_id"] = run_id
        merged["bcr_outer_fit_early_stop_used"] = False

        outer_rows.append(merged)

    outer_pred_df = pd.concat(outer_rows, ignore_index=True).sort_values(["outer_fold", "feature_date"]).reset_index(drop=True)

    if len(outer_pred_df) != 747:
        raise RuntimeError("Outer-test prediction count must be 747; observed {}".format(len(outer_pred_df)))

    if outer_pred_df.duplicated(subset=["outer_fold", "feature_date", "target_date"]).any():
        raise RuntimeError("Duplicate keys in outer-test component predictions")

    # Canonical key and y_true alignment.
    canonical_wide = pd.read_csv(CANONICAL_WIDE_PATH)
    canonical_wide["feature_date"] = normalize_date_col(canonical_wide["feature_date"])
    canonical_wide["target_date"] = normalize_date_col(canonical_wide["target_date"])

    canon_key = canonical_wide[["outer_fold", "feature_date", "target_date", "y_true", "event_tau15", "event_tau16", "event_tau17"]].copy()
    merged_chk = outer_pred_df.merge(
        canon_key,
        on=["outer_fold", "feature_date", "target_date"],
        how="outer",
        indicator=True,
        suffixes=("", "_canonical"),
    )
    if not (merged_chk["_merge"] == "both").all():
        missing = merged_chk[merged_chk["_merge"] != "both"][["outer_fold", "feature_date", "target_date", "_merge"]].head(5)
        raise RuntimeError("Outer-test key mismatch against canonical H5 key set: {}".format(missing.to_dict("records")))

    max_abs_y = float(np.max(np.abs(merged_chk["y_true"].to_numpy(dtype=float) - merged_chk["y_true_canonical"].to_numpy(dtype=float))))
    if max_abs_y > 1e-6:
        raise RuntimeError("Outer-test y_true mismatch against canonical values")

    for col in ["event_tau15", "event_tau16", "event_tau17"]:
        if not (merged_chk[col].astype(int).to_numpy() == merged_chk[col + "_canonical"].astype(int).to_numpy()).all():
            raise RuntimeError("Outer-test event label mismatch against canonical values")

    to_csv_with_dates(
        outer_pred_df,
        OUTER_COMPONENT_PRED_PATH,
        ["feature_date", "target_date", "outer_train_start", "outer_train_end", "outer_test_start", "outer_test_end"],
    )

    # Outer-test hybrid scores from frozen weights.
    score_rows: List[Dict[str, Any]] = []

    for fold in OUTER_FOLDS:
        fold_pred = outer_pred_df[outer_pred_df["outer_fold"] == fold].copy().sort_values("feature_date")

        for tau in TAUS:
            bcr_raw_col = "bcr_tcn_p_tau{}".format(int(tau))
            bcr_rank = rank01(fold_pred[bcr_raw_col])
            enet_rank = rank01(fold_pred["elasticnet_y_pred"])
            per_rank = rank01(fold_pred["persistence_y_pred"])
            hg_rank = rank01(fold_pred["hgbr_y_pred"])

            for r in BUDGETS:
                for policy in ["HybridRank_fixed", "HybridRank_nested_tuned", "HybridRank_nested_guarded"]:
                    key = (int(fold), int(tau), float(r), policy)
                    if key not in weight_lookup:
                        raise RuntimeError("Missing frozen weight lookup for {}".format(key))
                    meta = weight_lookup[key]
                    w = meta["weights"]

                    hybrid = (
                        float(w["BCR-TCN"]) * bcr_rank
                        + float(w["ElasticNet"]) * enet_rank
                        + float(w["Persistence"]) * per_rank
                        + float(w["HGBR"]) * hg_rank
                    )

                    for idx, row in fold_pred.reset_index(drop=True).iterrows():
                        score_rows.append(
                            {
                                "horizon": int(H),
                                "outer_fold": int(fold),
                                "feature_date": row["feature_date"],
                                "target_date": row["target_date"],
                                "y_true": float(row["y_true"]),
                                "event_tau15": int(row["event_tau15"]),
                                "event_tau16": int(row["event_tau16"]),
                                "event_tau17": int(row["event_tau17"]),
                                "threshold": int(tau),
                                "alarm_budget": float(r),
                                "policy_name": policy,
                                "score_context": "retrospective_offline_only",
                                "hybrid_score": float(hybrid[idx]),
                                "bcr_tcn_raw_score": float(row[bcr_raw_col]),
                                "bcr_tcn_rank_score": float(bcr_rank[idx]),
                                "elasticnet_raw_score": float(row["elasticnet_y_pred"]),
                                "elasticnet_rank_score": float(enet_rank[idx]),
                                "persistence_raw_score": float(row["persistence_y_pred"]),
                                "persistence_rank_score": float(per_rank[idx]),
                                "hgbr_raw_score": float(row["hgbr_y_pred"]),
                                "hgbr_rank_score": float(hg_rank[idx]),
                                "component_weight_summary": json.dumps(w, sort_keys=True),
                                "guard_applied": bool(meta["guard_applied"]),
                                "fallback_applied": bool(meta["fallback_applied"]),
                                "weight_source": str(meta["weight_source"]),
                                "dataset_sha256": dataset_sha,
                                "split_sha256": split_sha,
                                "nested_split_id": fold_data[fold]["nested_split_id"],
                                "run_id": run_id,
                                "git_commit": git_commit,
                            }
                        )

    score_df = pd.DataFrame(score_rows).sort_values(
        ["outer_fold", "threshold", "alarm_budget", "policy_name", "feature_date"]
    ).reset_index(drop=True)

    if score_df.duplicated(subset=["outer_fold", "feature_date", "target_date", "threshold", "alarm_budget", "policy_name"]).any():
        raise RuntimeError("Duplicate keys in outer-test hybrid score output")

    to_csv_with_dates(score_df, OUTER_SCORES_PATH, ["feature_date", "target_date"])

    # Isolation audit.
    isolation_rows: List[Dict[str, Any]] = []
    summary_by_fold = summary_df.set_index("outer_fold")

    for fold in OUTER_FOLDS:
        s = summary_by_fold.loc[fold]
        fd = fold_data[fold]

        max_subtrain_target = fd["component_subtrain"]["target_date"].max()
        min_comp_val_feature = fd["component_validation"]["feature_date"].min()
        max_comp_dev_target = fd["component_development"]["target_date"].max()
        min_hv_feature = fd["hybrid_validation"]["feature_date"].min()
        max_outer_train_target = fd["outer_train"]["target_date"].max()
        min_outer_test_feature = fd["outer_test"]["feature_date"].min()

        isolation_pass = bool(
            bool(s["inner_target_separation_pass"])
            and bool(s["hybrid_target_separation_pass"])
            and bool(s["outer_target_separation_pass"])
            and bool(s["future_fold_exclusion_pass"])
        )

        isolation_rows.append(
            {
                "outer_fold": int(fold),
                "scope": "fold_level",
                "component_subtrain_rows": int(len(fd["component_subtrain"])),
                "component_validation_rows": int(len(fd["component_validation"])),
                "hybrid_validation_rows": int(len(fd["hybrid_validation"])),
                "outer_train_rows": int(len(fd["outer_train"])),
                "outer_test_rows": int(len(fd["outer_test"])),
                "max_component_subtrain_target": max_subtrain_target,
                "min_component_validation_feature": min_comp_val_feature,
                "max_component_development_target": max_comp_dev_target,
                "min_hybrid_validation_feature": min_hv_feature,
                "max_outer_train_target": max_outer_train_target,
                "min_outer_test_feature": min_outer_test_feature,
                "component_validation_used_for_component_selection": True,
                "hybrid_validation_used_for_component_selection": False,
                "hybrid_validation_used_for_weight_selection": True,
                "outer_test_used_for_component_selection": False,
                "outer_test_used_for_weight_selection": False,
                "future_fold_used": False,
                "weights_frozen_before_outer_test": True,
                "inner_embargo_pass": bool(s["inner_target_separation_pass"]),
                "hybrid_embargo_pass": bool(s["hybrid_target_separation_pass"]),
                "outer_embargo_pass": bool(s["outer_target_separation_pass"]),
                "isolation_pass": isolation_pass,
                "notes": "no_future_fold_no_outer_test_outcomes_in_selection",
            }
        )

    isolation_df = pd.DataFrame(isolation_rows).sort_values("outer_fold")
    to_csv_with_dates(
        isolation_df,
        ISOLATION_AUDIT_PATH,
        [
            "max_component_subtrain_target",
            "min_component_validation_feature",
            "max_component_development_target",
            "min_hybrid_validation_feature",
            "max_outer_train_target",
            "min_outer_test_feature",
        ],
    )

    # Verify protected directories unchanged.
    protected_after = snapshot_tree_checksums(PROTECTED_DIRS)
    protected_changed = []
    for rel, before_sha in protected_before.items():
        after_sha = protected_after.get(rel)
        if after_sha != before_sha:
            protected_changed.append(rel)

    if protected_changed:
        raise RuntimeError("Protected source/canonical artifacts changed: {}".format(protected_changed[:10]))

    # Ensure outputs only inside authorized directory.
    out_files = [p for p in OUT_DIR.iterdir() if p.is_file()]
    outside_output = []
    for p in out_files:
        rp = p.resolve()
        if not str(rp).startswith(str((ROOT / AUTHORIZED_REL_DIR).resolve())):
            outside_output.append(str(rp))

    if outside_output:
        raise RuntimeError("Detected output files outside authorized directory: {}".format(outside_output))

    limitations = [
        "Submitted tuned HybridRank pooled non-held-out folds, allowing future chronology to influence earlier folds.",
        "The revised procedure separates component selection and ensemble selection.",
        "Five-day target-date embargoes separate every stage.",
        "The fixed policy is used at r <= 0.05.",
        "The fixed policy is also used when HybridRank-validation event support is insufficient.",
        "Full outer-test fold rank normalization remains retrospective and non-deployable.",
        "A separate sequential policy will be evaluated later.",
        "Hybrid-specific component predictions are not substituted for the canonical point-forecast package.",
        "No outer-test alarm metrics were calculated in this stage.",
    ]

    manifest = {
        "run_id": run_id,
        "purpose": "Fully chronological and fully nested H5 HybridRank pathway",
        "horizon": int(H),
        "nested_split_definition": {
            "component_validation_fraction": float(COMPONENT_VALIDATION_FRACTION),
            "hybrid_validation_fraction": float(HYBRID_VALIDATION_FRACTION),
            "deterministic_tail_rounding": "floor_with_min1_and_nonempty_guard",
            "embargo_days": int(EMBARGO_DAYS),
            "strict_inequalities": [
                "max(component_subtrain_target_date) < min(component_validation_feature_date)",
                "max(component_development_target_date) < min(hybrid_validation_feature_date)",
                "max(outer_train_target_date) < min(outer_test_feature_date)",
            ],
        },
        "component_validation_fraction": float(COMPONENT_VALIDATION_FRACTION),
        "hybrid_validation_fraction": float(HYBRID_VALIDATION_FRACTION),
        "embargo_days": int(EMBARGO_DAYS),
        "event_support_rule": {
            "minimum_events_required": int(SUPPORT_MIN_EVENTS),
            "minimum_alarm_slots_required": int(SUPPORT_MIN_ALARM_SLOTS),
            "guard_fixed_policy_condition": "r <= 0.05",
        },
        "guard_rule": "fixed_policy_when_r_le_0.05",
        "component_models": list(COMPONENT_ORDER),
        "candidate_spaces": {
            "ElasticNet": enet_candidates,
            "HGBR": {
                "candidate_space_h5": hgbr_space,
                "search_budget_h5": hgbr_budget,
                "candidate_sampling_seed_h5": hgbr_seed,
                "model_fit_random_state": hgbr_model_seed,
            },
            "BCR-TCN": bcr_cfg,
            "Persistence": {"rule": "y_pred = TNout at feature_date"},
        },
        "selection_objectives": {
            "ElasticNet": "min_inner_validation_MAE_tie_smallest_alpha_then_l1_ratio",
            "HGBR": "min_inner_validation_MAE_tie_smallest_candidate_configuration_id",
            "BCR-TCN": "max_validation_recall_tau16_r05_tie_earliest_epoch",
            "HybridRank": "precision_floor_then_lexicographic_recall_precision_tp",
        },
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "canonical_assembly_id": canonical_manifest["assembly_id"],
        "canonical_checksums": canonical_checks,
        "git_branch": active_branch,
        "git_commit": git_commit,
        "python_executable": python_executable,
        "python_version": python_version,
        "execution_command": "{} {}".format(sys.executable, Path(__file__).name),
        "timestamp": utc_now_iso(),
        "software_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "output_files": sorted([p.name for p in out_files]),
        "output_checksums": compute_output_checksums(OUT_DIR),
        "test_result": "not_run",
        "deterministic_result": "pending_test_script",
        "files_modified_outside_authorized_directory": outside_changes,
        "limitations": limitations,
        "protected_source_checksum_snapshot": {
            "before": protected_before,
            "after": protected_after,
            "changed": protected_changed,
        },
        "bcr_checkpoint_gate": {
            "rows": bcr_gate_df.to_dict("records"),
            "blocked_folds": [
                int(x)
                for x in bcr_gate_df[
                    bcr_gate_df["blocked_for_bcr_checkpoint_selection"].astype(bool)
                ]["outer_fold"].tolist()
            ],
            "policy": "if tau16_events==0 or k_r005==0 then stop fold and do not select checkpoint",
        },
        "fold1_tau16_support_diagnostic": fold1_tau16_diag_summary,
    }

    write_json(MANIFEST_PATH, manifest)

    report_text = build_completion_report(
        run_id=run_id,
        dataset_sha=dataset_sha,
        split_sha=split_sha,
        git_commit=git_commit,
        summary_df=summary_df,
        selection_df=selection_df,
        support_df=support_df,
        decision="B",
        bcr_gate_df=bcr_gate_df,
        fold1_tau16_diag_summary=fold1_tau16_diag_summary,
    )
    ensure_inside_authorized(COMPLETION_REPORT_PATH)
    COMPLETION_REPORT_PATH.write_text(report_text, encoding="utf-8")

    write_checksums_file(OUT_DIR, CHECKSUMS_PATH)

    # Final summary (Part P; test-related fields pending until test script runs).
    print("1. working directory: {}".format(Path.cwd().resolve()))
    print("2. branch and Git commit: {} {}".format(active_branch, git_commit))
    print("3. Python executable and version: {} {}".format(python_executable, python_version))
    print("4. input-verification result: {}".format(input_verification["verification_pass"]))
    print("5. nested split counts by fold:")
    for r in summary_df.itertuples(index=False):
        print(
            "   fold{} outer_train={} component_subtrain={} component_validation={} hybrid_validation={} outer_test={}".format(
                int(r.outer_fold),
                int(r.outer_train_N),
                int(r.component_subtrain_N_after_embargo),
                int(r.component_validation_N),
                int(r.hybrid_validation_N),
                int(r.outer_test_N),
            )
        )
    print("6. embargo exclusions by boundary and fold:")
    for r in summary_df.itertuples(index=False):
        print(
            "   fold{} inner_excluded={} hybrid_excluded={}".format(
                int(r.outer_fold), int(r.inner_embargo_excluded_N), int(r.hybrid_embargo_excluded_N)
            )
        )
    print("7. component configurations selected:")
    for fold in OUTER_FOLDS:
        print("   fold{} ElasticNet={} HGBR={} BCR_epoch={}".format(
            fold,
            selected_enet[fold],
            selected_hgbr[fold]["candidate_configuration_id"],
            selected_bcr[fold]["selected_epoch"],
        ))
    print("8. BCR-TCN selected epoch by fold:")
    for fold in OUTER_FOLDS:
        print("   fold{} epoch={}".format(fold, int(selected_bcr[fold]["selected_epoch"])))
    print("   Fold 1 tau16 support diagnostic artifact: see {}".format(FOLD1_TAU16_DIAGNOSTIC_PATH.name))
    print("9. HybridRank-validation event support: see {}".format(HV_SUPPORT_PATH.name))
    print("10. tuned versus fixed decisions by fold/threshold/budget: see {}".format(WEIGHTS_PATH.name))
    print("11. frozen weights: see {}".format(WEIGHTS_PATH.name))
    print("12. hybrid-validation prediction counts: {}".format(len(hv_pred_df)))
    print("13. outer-test prediction count: {}".format(len(outer_pred_df)))
    print("14. test and future-fold isolation: {}".format(bool(isolation_df["isolation_pass"].all())))
    print("15. source preservation: {}".format(len(protected_changed) == 0))
    print("16. deterministic result: pending_test_script")
    print("17. passed assertions: pending_test_script")
    print("18. final decision: B")
    print("19. exactly one next action: Run test_fully_nested_hybridrank_h5.py")


if __name__ == "__main__":
    main()
