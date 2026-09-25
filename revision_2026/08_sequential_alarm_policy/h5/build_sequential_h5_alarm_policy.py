from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
AUTHORIZED_REL_DIR = "revision_2026/08_sequential_alarm_policy/h5"
EXPECTED_BRANCH = "controlled-reruns-v1"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CONTROLLED_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions" / "h5_cross_model"
FULLY_NESTED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fully_nested"
FIXED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fixed_policy"
RETRO_DIR = ROOT / "revision_2026" / "07_retrospective_alarm_budget" / "h5"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

CANONICAL_WIDE_PATH = CANONICAL_DIR / "canonical_h5_predictions_wide.csv"
CANONICAL_LONG_PATH = CANONICAL_DIR / "canonical_h5_predictions_long.csv"
CANONICAL_MANIFEST_PATH = CANONICAL_DIR / "canonical_h5_assembly_manifest.json"
CANONICAL_REPORT_PATH = CANONICAL_DIR / "canonical_h5_assembly_report.md"
CANONICAL_CHECKSUMS_PATH = CANONICAL_DIR / "canonical_h5_checksums.sha256"

FIXED_MANIFEST_PATH = FIXED_DIR / "fixed_hybridrank_h5_manifest.json"
FIXED_REPORT_PATH = FIXED_DIR / "fixed_hybridrank_h5_completion_report.md"
FIXED_WEIGHTS_PATH = FIXED_DIR / "fixed_hybridrank_weights.csv"
FIXED_POLICY_DEFINITION_PATH = FIXED_DIR / "fixed_hybridrank_policy_definition.md"
FIXED_CHECKSUMS_PATH = FIXED_DIR / "fixed_hybridrank_h5_checksums.sha256"

RETRO_MANIFEST_PATH = RETRO_DIR / "retrospective_h5_alarm_budget_manifest.json"
RETRO_REPORT_PATH = RETRO_DIR / "retrospective_h5_alarm_budget_completion_report.md"
RETRO_CHECKSUMS_PATH = RETRO_DIR / "retrospective_h5_alarm_budget_checksums.sha256"

FULLY_NESTED_REPORT_PATH = FULLY_NESTED_DIR / "fully_nested_hybridrank_h5_completion_report.md"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
STARTUP_REGISTRY_PATH = OUT_DIR / "sequential_h5_startup_registry.csv"
CAUSAL_COMPONENT_RANKS_PATH = OUT_DIR / "sequential_h5_causal_component_ranks.csv"
CAUSAL_FIXED_ENSEMBLE_PATH = OUT_DIR / "sequential_h5_causal_fixed_ensemble_scores.csv"
SEQUENTIAL_DECISIONS_PATH = OUT_DIR / "sequential_h5_alarm_decisions.csv"
CUTOFF_AUDIT_PATH = OUT_DIR / "sequential_h5_cutoff_audit.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "sequential_h5_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "sequential_h5_metrics_pooled.csv"
FULL_FOLD_SECONDARY_PATH = OUT_DIR / "sequential_h5_full_fold_secondary_metrics.csv"
BURDEN_AUDIT_PATH = OUT_DIR / "sequential_h5_alarm_burden_audit.csv"
SEQ_VS_RETRO_PATH = OUT_DIR / "sequential_vs_retrospective_definition.md"
INFO_ISOLATION_PATH = OUT_DIR / "sequential_h5_information_isolation_audit.json"
DATE_COVERAGE_AUDIT_PATH = OUT_DIR / "sequential_h5_date_coverage_audit.csv"
MANIFEST_PATH = OUT_DIR / "sequential_h5_alarm_policy_manifest.json"
COMPLETION_REPORT_PATH = OUT_DIR / "sequential_h5_alarm_policy_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "sequential_h5_alarm_policy_checksums.sha256"

H = 5
THRESHOLDS = (15, 16, 17)
BUDGETS = (0.05, 0.10)
STARTUP_DAYS = 30
MODELS = (
    "Persistence",
    "Ridge",
    "ElasticNet",
    "HGBR",
    "BCR-TCN v1.1",
    "HybridRank_fixed_documented",
)
SCORE_CONTEXT = "historical_sequential_past_only"

BASE_MODEL_COLUMN_MAP = {
    "Persistence": "persistence_y_pred",
    "Ridge": "ridge_y_pred",
    "ElasticNet": "elasticnet_y_pred",
    "HGBR": "hgbr_y_pred",
}

FIXED_WEIGHTS = {
    "BCR-TCN v1.1": 0.50,
    "ElasticNet": 0.25,
    "Persistence": 0.25,
    "HGBR": 0.00,
}

PROTECTED_DIRS = [
    PROTOCOL_DIR,
    CONTROLLED_DIR,
    CANONICAL_DIR,
    FULLY_NESTED_DIR,
    FIXED_DIR,
    RETRO_DIR,
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: Sequence[str]) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def parse_checksum_manifest(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            raise RuntimeError("Malformed checksum line in {}: {}".format(path, line))
        rows.append((parts[0], parts[-1]))
    return rows


def expected_checksum_for_file(path: Path, filename: str) -> str:
    for expected, rel_name in parse_checksum_manifest(path):
        rel_norm = str(rel_name).replace("\\", "/")
        if rel_norm == filename or rel_norm.endswith("/" + filename):
            return expected
    raise KeyError("Checksum entry not found for {} in {}".format(filename, path))


def verify_checksum_manifest(path: Path, base_dir: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for expected, rel_name in parse_checksum_manifest(path):
        target = base_dir / rel_name
        if not target.exists():
            raise RuntimeError("Checksum target missing: {}".format(target))
        observed = sha256_file(target)
        ok = observed == expected
        results[str(rel_name)] = {
            "expected": expected,
            "observed": observed,
            "pass": bool(ok),
        }
        if not ok:
            raise RuntimeError(
                "Checksum mismatch for {} in {}: expected {}, observed {}".format(
                    rel_name, path, expected, observed
                )
            )
    return results


def infer_decision_from_report(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")

    m = re.search(r"FINAL DECISION\s*(?:\n|\r\n)+(?:<!--.*?-->\s*)?([ABCD])\.\s", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()

    m2 = re.search(r"Decision:\s*([ABCD])", txt)
    if m2:
        return m2.group(1).upper()

    lines = [ln.strip() for ln in txt.splitlines()]
    for ln in reversed(lines[-40:]):
        if re.match(r"^[ABCD]\.\s", ln):
            return ln[0].upper()

    raise RuntimeError("Could not infer final decision from {}".format(path))


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


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
    return p.startswith(a) or a.startswith(p)


def snapshot_tree_checksums(dirs: Sequence[Path], rel_base: Path) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for d in dirs:
        if not d.exists():
            raise RuntimeError("Protected directory missing: {}".format(d))
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(rel_base)).replace("\\", "/")
            snap[rel] = sha256_file(p)
    return snap


def ensure_inside_authorized(path: Path) -> None:
    rp = path.resolve()
    auth = (ROOT / AUTHORIZED_REL_DIR).resolve()
    if not str(rp).startswith(str(auth)):
        raise RuntimeError("Attempted write outside authorized directory: {}".format(rp))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_inside_authorized(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def to_csv_with_dates(df: pd.DataFrame, path: Path, date_cols: Sequence[str]) -> None:
    out = df.copy()
    for c in date_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    ensure_inside_authorized(path)
    out.to_csv(path, index=False)


def compute_output_checksums(out_dir: Path, exclude_name: Optional[str]) -> Dict[str, str]:
    checks: Dict[str, str] = {}
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if exclude_name and p.name == exclude_name:
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".py"}:
            continue
        checks[p.name] = sha256_file(p)
    return checks


def write_checksums_file(out_dir: Path, checksum_path: Path) -> None:
    lines: List[str] = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == checksum_path.name:
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".py"}:
            continue
        lines.append("{}  {}".format(sha256_file(p), p.name))
    ensure_inside_authorized(checksum_path)
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num) / float(den)


def quantile_higher(values, probability):
    try:
        return float(
            np.quantile(values, probability, method="higher")
        )
    except TypeError:
        return float(
            np.quantile(
                values,
                probability,
                interpolation="higher",
            )
        )


def add_canonical_row_id(wide_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["horizon", "outer_fold", "feature_date", "target_date"]
    keys = wide_df[key_cols].drop_duplicates().sort_values(
        ["outer_fold", "target_date", "feature_date"]
    ).reset_index(drop=True)
    if int(len(keys)) != 747:
        raise RuntimeError("Expected 747 canonical keys, observed {}".format(len(keys)))

    keys["canonical_row_id"] = np.arange(1, len(keys) + 1, dtype=int)
    out = wide_df.merge(keys, on=key_cols, how="left", validate="one_to_one")
    if out["canonical_row_id"].isna().any():
        raise RuntimeError("Failed to assign canonical_row_id for all rows")

    out = out.sort_values(["outer_fold", "target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
    return out


def build_startup_registry(wide_df: pd.DataFrame, startup_days: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for fold in sorted(wide_df["outer_fold"].astype(int).unique().tolist()):
        f = wide_df[wide_df["outer_fold"].astype(int) == int(fold)].copy()
        f = f.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

        n = int(len(f))
        if n < startup_days:
            raise RuntimeError("Fold {} has only {} rows; startup_days is {}".format(fold, n, startup_days))

        startup_part = f.iloc[:startup_days].copy()
        eligible_part = f.iloc[startup_days:].copy()

        rows.append(
            {
                "horizon": int(H),
                "outer_fold": int(fold),
                "total_N": int(n),
                "startup_N": int(len(startup_part)),
                "eligible_N": int(len(eligible_part)),
                "startup_start": startup_part["target_date"].min(),
                "startup_end": startup_part["target_date"].max(),
                "eligible_start": eligible_part["target_date"].min(),
                "eligible_end": eligible_part["target_date"].max(),
                "startup_rule": "first_30_chronological_dates_no_alarm",
                "future_scores_used": False,
                "labels_used": False,
            }
        )

    out = pd.DataFrame(rows).sort_values(["outer_fold"]).reset_index(drop=True)
    return out


def compute_causal_component_ranks_and_ensemble(
    wide_df: pd.DataFrame,
    run_id: str,
    git_commit: str,
    dataset_sha: str,
    split_sha: str,
    canonical_assembly_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    component_rows: List[Dict[str, Any]] = []
    ensemble_rows: List[Dict[str, Any]] = []

    for tau in THRESHOLDS:
        bcr_col = "bcr_tcn_v11_p_tau{}".format(int(tau))
        components = [
            ("BCR-TCN v1.1", bcr_col, FIXED_WEIGHTS["BCR-TCN v1.1"]),
            ("ElasticNet", "elasticnet_y_pred", FIXED_WEIGHTS["ElasticNet"]),
            ("Persistence", "persistence_y_pred", FIXED_WEIGHTS["Persistence"]),
            ("HGBR", "hgbr_y_pred", FIXED_WEIGHTS["HGBR"]),
        ]

        for fold in sorted(wide_df["outer_fold"].astype(int).unique().tolist()):
            fold_df = wide_df[wide_df["outer_fold"].astype(int) == int(fold)].copy()
            fold_df = fold_df.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

            comp_ranks: Dict[str, np.ndarray] = {}

            for component_name, raw_col, weight in components:
                values = pd.to_numeric(fold_df[raw_col], errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        "Non-finite values in {} fold {} threshold {}".format(component_name, fold, tau)
                    )

                ranks = np.zeros(len(values), dtype=float)
                history_values: List[float] = []
                history_dates: List[pd.Timestamp] = []

                for i, current in enumerate(values):
                    n_past = int(len(history_values))
                    if n_past == 0:
                        smaller = 0
                        equal = 0
                        rank = 0.5
                    else:
                        past_arr = np.asarray(history_values, dtype=float)
                        smaller = int((past_arr < float(current)).sum())
                        equal = int((past_arr == float(current)).sum())
                        rank = (float(smaller) + 0.5 * float(equal)) / float(n_past)

                    hist_start = history_dates[0] if history_dates else pd.NaT
                    hist_end = history_dates[-1] if history_dates else pd.NaT

                    component_rows.append(
                        {
                            "horizon": int(H),
                            "outer_fold": int(fold),
                            "feature_date": fold_df.loc[i, "feature_date"],
                            "target_date": fold_df.loc[i, "target_date"],
                            "canonical_row_id": int(fold_df.loc[i, "canonical_row_id"]),
                            "y_true": float(fold_df.loc[i, "y_true"]),
                            "event_tau15": int(fold_df.loc[i, "event_tau15"]),
                            "event_tau16": int(fold_df.loc[i, "event_tau16"]),
                            "event_tau17": int(fold_df.loc[i, "event_tau17"]),
                            "threshold": int(tau),
                            "component": component_name,
                            "component_raw_score": float(current),
                            "past_score_count": int(n_past),
                            "smaller_prior_count": int(smaller),
                            "equal_prior_count": int(equal),
                            "causal_rank": float(rank),
                            "component_weight": float(weight),
                            "rank_history_start": hist_start,
                            "rank_history_end": hist_end,
                            "y_true_used_for_rank": False,
                            "event_labels_used_for_rank": False,
                            "future_scores_used": False,
                            "score_context": SCORE_CONTEXT,
                            "dataset_sha256": dataset_sha,
                            "split_sha256": split_sha,
                            "canonical_assembly_id": canonical_assembly_id,
                            "sequential_run_id": run_id,
                            "git_commit": git_commit,
                        }
                    )

                    ranks[i] = float(rank)
                    history_values.append(float(current))
                    history_dates.append(pd.Timestamp(fold_df.loc[i, "target_date"]))

                comp_ranks[component_name] = ranks

            ensemble_score = (
                FIXED_WEIGHTS["BCR-TCN v1.1"] * comp_ranks["BCR-TCN v1.1"]
                + FIXED_WEIGHTS["ElasticNet"] * comp_ranks["ElasticNet"]
                + FIXED_WEIGHTS["Persistence"] * comp_ranks["Persistence"]
                + FIXED_WEIGHTS["HGBR"] * comp_ranks["HGBR"]
            )

            if (not np.isfinite(ensemble_score).all()) or (ensemble_score.min() < 0.0) or (ensemble_score.max() > 1.0):
                raise RuntimeError("Causal ensemble score out of [0,1] bounds or non-finite")

            for i in range(len(fold_df)):
                ensemble_rows.append(
                    {
                        "horizon": int(H),
                        "outer_fold": int(fold),
                        "feature_date": fold_df.loc[i, "feature_date"],
                        "target_date": fold_df.loc[i, "target_date"],
                        "canonical_row_id": int(fold_df.loc[i, "canonical_row_id"]),
                        "y_true": float(fold_df.loc[i, "y_true"]),
                        "event_tau15": int(fold_df.loc[i, "event_tau15"]),
                        "event_tau16": int(fold_df.loc[i, "event_tau16"]),
                        "event_tau17": int(fold_df.loc[i, "event_tau17"]),
                        "threshold": int(tau),
                        "model": "HybridRank_fixed_documented",
                        "raw_score": float(ensemble_score[i]),
                        "policy_score": float(ensemble_score[i]),
                        "policy_score_source": "causal_fixed_ensemble_from_past_only_component_ranks",
                        "bcr_tcn_rank": float(comp_ranks["BCR-TCN v1.1"][i]),
                        "elasticnet_rank": float(comp_ranks["ElasticNet"][i]),
                        "persistence_rank": float(comp_ranks["Persistence"][i]),
                        "hgbr_rank": float(comp_ranks["HGBR"][i]),
                        "bcr_tcn_weight": float(FIXED_WEIGHTS["BCR-TCN v1.1"]),
                        "elasticnet_weight": float(FIXED_WEIGHTS["ElasticNet"]),
                        "persistence_weight": float(FIXED_WEIGHTS["Persistence"]),
                        "hgbr_weight": float(FIXED_WEIGHTS["HGBR"]),
                        "rank_rule": "strictly_prior_empirical_rank_with_half_tie",
                        "y_true_used_for_score": False,
                        "event_labels_used_for_score": False,
                        "future_scores_used": False,
                        "score_context": SCORE_CONTEXT,
                        "dataset_sha256": dataset_sha,
                        "split_sha256": split_sha,
                        "canonical_assembly_id": canonical_assembly_id,
                        "sequential_run_id": run_id,
                        "git_commit": git_commit,
                    }
                )

    comp_df = pd.DataFrame(component_rows).sort_values(
        ["component", "outer_fold", "threshold", "target_date", "feature_date", "canonical_row_id"]
    ).reset_index(drop=True)

    ensemble_df = pd.DataFrame(ensemble_rows).sort_values(
        ["outer_fold", "threshold", "target_date", "feature_date", "canonical_row_id"]
    ).reset_index(drop=True)

    expected_component_rows = 747 * len(THRESHOLDS) * 4
    if int(len(comp_df)) != int(expected_component_rows):
        raise RuntimeError(
            "Unexpected component rank row count: observed {}, expected {}".format(
                len(comp_df), expected_component_rows
            )
        )

    expected_ensemble_rows = 747 * len(THRESHOLDS)
    if int(len(ensemble_df)) != int(expected_ensemble_rows):
        raise RuntimeError(
            "Unexpected causal ensemble row count: observed {}, expected {}".format(
                len(ensemble_df), expected_ensemble_rows
            )
        )

    return comp_df, ensemble_df


def build_policy_score_panel(
    wide_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    run_id: str,
    git_commit: str,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    base_common_cols = [
        "horizon",
        "outer_fold",
        "feature_date",
        "target_date",
        "canonical_row_id",
        "y_true",
        "event_tau15",
        "event_tau16",
        "event_tau17",
        "dataset_sha256",
        "split_sha256",
        "canonical_assembly_id",
    ]

    for tau in THRESHOLDS:
        b = wide_df[base_common_cols].copy()
        b["threshold"] = int(tau)

        model_defs = [
            (
                "Persistence",
                "persistence_y_pred",
                "canonical_h5_predictions_wide.csv:persistence_y_pred",
            ),
            (
                "Ridge",
                "ridge_y_pred",
                "canonical_h5_predictions_wide.csv:ridge_y_pred",
            ),
            (
                "ElasticNet",
                "elasticnet_y_pred",
                "canonical_h5_predictions_wide.csv:elasticnet_y_pred",
            ),
            (
                "HGBR",
                "hgbr_y_pred",
                "canonical_h5_predictions_wide.csv:hgbr_y_pred",
            ),
            (
                "BCR-TCN v1.1",
                "bcr_tcn_v11_p_tau{}".format(int(tau)),
                "canonical_h5_predictions_wide.csv:bcr_tcn_v11_p_tau{}".format(int(tau)),
            ),
        ]

        for model_name, col_name, source_label in model_defs:
            s = b.copy()
            s["model"] = model_name
            s["raw_score"] = pd.to_numeric(wide_df[col_name], errors="coerce").to_numpy(dtype=float)
            s["policy_score"] = s["raw_score"].to_numpy(dtype=float)
            s["policy_score_source"] = source_label
            s["score_context"] = SCORE_CONTEXT
            s["sequential_run_id"] = run_id
            s["git_commit"] = git_commit
            rows.append(s)

        ef = ensemble_df[ensemble_df["threshold"].astype(int) == int(tau)].copy()
        ef = ef[
            [
                "horizon",
                "outer_fold",
                "feature_date",
                "target_date",
                "canonical_row_id",
                "y_true",
                "event_tau15",
                "event_tau16",
                "event_tau17",
                "threshold",
                "model",
                "raw_score",
                "policy_score",
                "policy_score_source",
                "score_context",
                "dataset_sha256",
                "split_sha256",
                "canonical_assembly_id",
                "sequential_run_id",
                "git_commit",
            ]
        ].copy()
        rows.append(ef)

    panel = pd.concat(rows, axis=0, ignore_index=True)

    expected_rows = 747 * len(THRESHOLDS) * len(MODELS)
    if int(len(panel)) != int(expected_rows):
        raise RuntimeError(
            "Unexpected score panel row count: observed {}, expected {}".format(
                len(panel), expected_rows
            )
        )

    if not np.isfinite(pd.to_numeric(panel["raw_score"], errors="coerce").to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite raw_score detected in score panel")

    panel = panel.sort_values(
        ["model", "outer_fold", "threshold", "target_date", "feature_date", "canonical_row_id"]
    ).reset_index(drop=True)

    return panel


def run_group_sequential_decisions(
    group_df: pd.DataFrame,
    nominal_budget_r: float,
    startup_days: int,
) -> pd.DataFrame:
    g = group_df.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

    past_scores: List[float] = []
    past_dates: List[pd.Timestamp] = []
    rows: List[Dict[str, Any]] = []

    q = float(1.0 - nominal_budget_r)

    for i, row in g.iterrows():
        current_score = float(row["policy_score"])
        past_count = int(len(past_scores))

        startup = bool(i < startup_days)
        eligible = not startup

        cutoff_score = float("nan")
        alarm_flag = 0
        equal_to_cutoff = False

        if past_dates:
            hist_start = past_dates[0]
            hist_end = past_dates[-1]
        else:
            hist_start = pd.NaT
            hist_end = pd.NaT

        if eligible:
            if past_count == 0:
                raise RuntimeError("Eligible row encountered with no past history")
            history = np.asarray(past_scores, dtype=float)
            # Python 3.8 / NumPy 1.24-compatible 'higher' quantile implementation.
            cutoff_score = quantile_higher(history, q)
            equal_to_cutoff = bool(np.isclose(current_score, cutoff_score, rtol=0.0, atol=0.0))
            alarm_flag = int(current_score > cutoff_score)
        else:
            alarm_flag = 0
            equal_to_cutoff = False

        rec = row.to_dict()
        rec.update(
            {
                "nominal_budget_r": float(nominal_budget_r),
                "past_score_count": int(past_count),
                "cutoff_probability": float(q),
                "cutoff_score": cutoff_score,
                "cutoff_history_start": hist_start,
                "cutoff_history_end": hist_end,
                "startup_calibration": bool(startup),
                "evaluation_eligible": bool(eligible),
                "alarm_flag": int(alarm_flag),
                "equal_to_cutoff": bool(equal_to_cutoff),
                "decision_rule": "alarm_if_policy_score_gt_past_only_higher_quantile_cutoff_with_strict_equality_no_alarm",
                "future_scores_used": False,
                "labels_used_for_score": False,
                "labels_used_for_cutoff": False,
            }
        )
        rows.append(rec)

        # Append only after decision to preserve strict past-only cutoff history.
        past_scores.append(current_score)
        past_dates.append(pd.Timestamp(row["target_date"]))

    return pd.DataFrame(rows)


def build_sequential_decisions(
    score_panel: pd.DataFrame,
    budgets: Sequence[float],
    startup_days: int,
    run_id: str,
    git_commit: str,
) -> pd.DataFrame:
    decision_parts: List[pd.DataFrame] = []

    group_cols = ["model", "outer_fold", "threshold"]

    for _, group_df in score_panel.groupby(group_cols, sort=True):
        for r in budgets:
            part = run_group_sequential_decisions(group_df=group_df, nominal_budget_r=float(r), startup_days=startup_days)
            part["sequential_run_id"] = run_id
            part["git_commit"] = git_commit
            part["score_context"] = SCORE_CONTEXT
            decision_parts.append(part)

    decisions = pd.concat(decision_parts, axis=0, ignore_index=True)

    expected_rows = 747 * len(THRESHOLDS) * len(BUDGETS) * len(MODELS)
    if int(len(decisions)) != int(expected_rows):
        raise RuntimeError(
            "Unexpected decision row count: observed {}, expected {}".format(len(decisions), expected_rows)
        )

    if decisions[decisions["startup_calibration"].astype(bool)]["alarm_flag"].astype(int).sum() != 0:
        raise RuntimeError("Startup rows must never emit alarms")

    decisions = decisions.sort_values(
        ["model", "threshold", "nominal_budget_r", "outer_fold", "target_date", "feature_date", "canonical_row_id"]
    ).reset_index(drop=True)

    required_cols = [
        "horizon",
        "outer_fold",
        "feature_date",
        "target_date",
        "canonical_row_id",
        "y_true",
        "event_tau15",
        "event_tau16",
        "event_tau17",
        "threshold",
        "nominal_budget_r",
        "model",
        "raw_score",
        "policy_score",
        "policy_score_source",
        "past_score_count",
        "cutoff_probability",
        "cutoff_score",
        "cutoff_history_start",
        "cutoff_history_end",
        "startup_calibration",
        "evaluation_eligible",
        "alarm_flag",
        "equal_to_cutoff",
        "decision_rule",
        "score_context",
        "future_scores_used",
        "labels_used_for_score",
        "labels_used_for_cutoff",
        "dataset_sha256",
        "split_sha256",
        "canonical_assembly_id",
        "sequential_run_id",
        "git_commit",
    ]

    missing = [c for c in required_cols if c not in decisions.columns]
    if missing:
        raise RuntimeError("Decision file missing required columns: {}".format(missing))

    return decisions[required_cols].copy()


def build_cutoff_audit(decisions_df: pd.DataFrame, startup_days: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    gcols = ["model", "outer_fold", "threshold", "nominal_budget_r"]
    for gkey, grp in decisions_df.groupby(gcols, sort=True):
        model_name, fold, tau, budget = gkey
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

        startup_rows = int(g["startup_calibration"].astype(bool).sum())
        eligible_rows = int(g["evaluation_eligible"].astype(bool).sum())

        eligible = g[g["evaluation_eligible"].astype(bool)].copy()
        min_past = int(eligible["past_score_count"].astype(int).min()) if not eligible.empty else 0
        max_past = int(eligible["past_score_count"].astype(int).max()) if not eligible.empty else 0
        cutoff_missing = int(eligible["cutoff_score"].isna().sum())

        row = {
            "model": str(model_name),
            "outer_fold": int(fold),
            "threshold": int(tau),
            "nominal_budget_r": float(budget),
            "total_rows": int(len(g)),
            "startup_rows": int(startup_rows),
            "eligible_rows": int(eligible_rows),
            "minimum_past_score_count": int(min_past),
            "maximum_past_score_count": int(max_past),
            "first_eligible_date": eligible["target_date"].min() if not eligible.empty else pd.NaT,
            "last_eligible_date": eligible["target_date"].max() if not eligible.empty else pd.NaT,
            "cutoff_min": float(pd.to_numeric(eligible["cutoff_score"], errors="coerce").min()) if not eligible.empty else float("nan"),
            "cutoff_max": float(pd.to_numeric(eligible["cutoff_score"], errors="coerce").max()) if not eligible.empty else float("nan"),
            "cutoff_missing_count_on_eligible_dates": int(cutoff_missing),
            "future_score_access": False,
            "current_score_in_cutoff_history": False,
            "labels_used": False,
        }

        rule_pass = bool(
            row["startup_rows"] == startup_days
            and row["eligible_rows"] == (249 - startup_days)
            and row["minimum_past_score_count"] >= startup_days
            and row["cutoff_missing_count_on_eligible_dates"] == 0
            and (row["future_score_access"] is False)
            and (row["current_score_in_cutoff_history"] is False)
            and (row["labels_used"] is False)
        )

        row["cutoff_rule_pass"] = rule_pass
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["model", "outer_fold", "threshold", "nominal_budget_r"]).reset_index(drop=True)


def confusion_counts(alarm: np.ndarray, event: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((alarm == 1) & (event == 1)).sum())
    fp = int(((alarm == 1) & (event == 0)).sum())
    fn = int(((alarm == 0) & (event == 1)).sum())
    tn = int(((alarm == 0) & (event == 0)).sum())
    return tp, fp, fn, tn


def build_metrics_by_fold(decisions_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    gcols = ["model", "horizon", "outer_fold", "threshold", "nominal_budget_r"]
    for gkey, grp in decisions_df.groupby(gcols, sort=True):
        model_name, horizon, fold, tau, budget = gkey
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

        startup_n = int(g["startup_calibration"].astype(bool).sum())
        eligible = g[g["evaluation_eligible"].astype(bool)].copy()

        event_col = "event_tau{}".format(int(tau))
        alarm = eligible["alarm_flag"].astype(int).to_numpy()
        event = eligible[event_col].astype(int).to_numpy()

        tp, fp, fn, tn = confusion_counts(alarm, event)

        eligible_n = int(len(eligible))
        eligible_events = int(event.sum())
        alarms_issued = int(alarm.sum())

        precision = safe_div(float(tp), float(alarms_issued))
        recall = safe_div(float(tp), float(eligible_events))
        specificity = safe_div(float(tn), float(tn + fp))
        fpr = safe_div(float(fp), float(fp + tn))
        false_alarm_fraction = safe_div(float(fp), float(alarms_issued))
        missed_event_fraction = safe_div(float(fn), float(eligible_events))
        realized_alarm_fraction = safe_div(float(alarms_issued), float(eligible_n))

        rows.append(
            {
                "model": str(model_name),
                "horizon": int(horizon),
                "outer_fold": int(fold),
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "eligible_N": int(eligible_n),
                "eligible_events": int(eligible_events),
                "alarms_issued": int(alarms_issued),
                "realized_alarm_fraction": realized_alarm_fraction,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "false_positive_rate": fpr,
                "false_alarm_fraction_of_alarms": false_alarm_fraction,
                "missed_event_fraction": missed_event_fraction,
                "nominal_budget_difference": float(realized_alarm_fraction - float(budget)),
                "startup_N": int(startup_n),
                "score_context": SCORE_CONTEXT,
                "sequential_run_id": run_id,
            }
        )

    out = pd.DataFrame(rows).sort_values(["model", "outer_fold", "threshold", "nominal_budget_r"]).reset_index(drop=True)
    return out


def build_metrics_pooled(fold_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    gcols = ["model", "horizon", "threshold", "nominal_budget_r"]
    for gkey, grp in fold_df.groupby(gcols, sort=True):
        model_name, horizon, tau, budget = gkey

        pooled_eligible_n = int(grp["eligible_N"].astype(int).sum())
        pooled_events = int(grp["eligible_events"].astype(int).sum())
        pooled_alarms = int(grp["alarms_issued"].astype(int).sum())
        pooled_tp = int(grp["TP"].astype(int).sum())
        pooled_fp = int(grp["FP"].astype(int).sum())
        pooled_fn = int(grp["FN"].astype(int).sum())
        pooled_tn = int(grp["TN"].astype(int).sum())

        pooled_precision = safe_div(float(pooled_tp), float(pooled_alarms))
        pooled_recall = safe_div(float(pooled_tp), float(pooled_events))
        pooled_specificity = safe_div(float(pooled_tn), float(pooled_tn + pooled_fp))
        pooled_fpr = safe_div(float(pooled_fp), float(pooled_fp + pooled_tn))
        pooled_false_alarm_fraction = safe_div(float(pooled_fp), float(pooled_alarms))
        pooled_missed_event_fraction = safe_div(float(pooled_fn), float(pooled_events))
        pooled_realized_alarm_fraction = safe_div(float(pooled_alarms), float(pooled_eligible_n))

        rows.append(
            {
                "model": str(model_name),
                "horizon": int(horizon),
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "pooled_eligible_N": int(pooled_eligible_n),
                "pooled_eligible_events": int(pooled_events),
                "pooled_alarms": int(pooled_alarms),
                "pooled_realized_alarm_fraction": pooled_realized_alarm_fraction,
                "pooled_TP": int(pooled_tp),
                "pooled_FP": int(pooled_fp),
                "pooled_FN": int(pooled_fn),
                "pooled_TN": int(pooled_tn),
                "pooled_precision": pooled_precision,
                "pooled_recall": pooled_recall,
                "pooled_specificity": pooled_specificity,
                "pooled_false_positive_rate": pooled_fpr,
                "pooled_false_alarm_fraction_of_alarms": pooled_false_alarm_fraction,
                "pooled_missed_event_fraction": pooled_missed_event_fraction,
                "nominal_budget_difference": float(pooled_realized_alarm_fraction - float(budget)),
                "score_context": SCORE_CONTEXT,
                "sequential_run_id": run_id,
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "threshold", "nominal_budget_r"]).reset_index(drop=True)


def build_full_fold_secondary_metrics(decisions_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    gcols = ["model", "horizon", "outer_fold", "threshold", "nominal_budget_r"]
    for gkey, grp in decisions_df.groupby(gcols, sort=True):
        model_name, horizon, fold, tau, budget = gkey
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

        event_col = "event_tau{}".format(int(tau))
        alarm = g["alarm_flag"].astype(int).to_numpy()
        event = g[event_col].astype(int).to_numpy()

        tp, fp, fn, tn = confusion_counts(alarm, event)

        total_n = int(len(g))
        total_events = int(event.sum())
        alarms_issued = int(alarm.sum())

        precision = safe_div(float(tp), float(alarms_issued))
        recall = safe_div(float(tp), float(total_events))
        specificity = safe_div(float(tn), float(tn + fp))
        fpr = safe_div(float(fp), float(fp + tn))
        false_alarm_fraction = safe_div(float(fp), float(alarms_issued))
        missed_event_fraction = safe_div(float(fn), float(total_events))

        rows.append(
            {
                "model": str(model_name),
                "horizon": int(horizon),
                "outer_fold": int(fold),
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "metric_scope": "secondary_full_fold_with_startup_no_alarms",
                "primary_reference": "primary_post_startup",
                "total_N": int(total_n),
                "total_events": int(total_events),
                "alarms_issued": int(alarms_issued),
                "realized_alarm_fraction": safe_div(float(alarms_issued), float(total_n)),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "false_positive_rate": fpr,
                "false_alarm_fraction_of_alarms": false_alarm_fraction,
                "missed_event_fraction": missed_event_fraction,
                "startup_N": int(g["startup_calibration"].astype(bool).sum()),
                "eligible_N": int(g["evaluation_eligible"].astype(bool).sum()),
                "score_context": SCORE_CONTEXT,
                "sequential_run_id": run_id,
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "outer_fold", "threshold", "nominal_budget_r"]).reset_index(drop=True)


def longest_alarm_streak(values: Sequence[int]) -> int:
    best = 0
    cur = 0
    for v in values:
        if int(v) == 1:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def max_alarms_in_7_day_window(dates: Sequence[pd.Timestamp], alarms: Sequence[int]) -> int:
    if len(dates) == 0:
        return 0

    n = len(dates)
    best = 0
    for i in range(n):
        end = pd.Timestamp(dates[i]) + pd.Timedelta(days=6)
        count = 0
        for j in range(i, n):
            if pd.Timestamp(dates[j]) > end:
                break
            count += int(alarms[j])
        if count > best:
            best = count
    return int(best)


def build_burden_audit(decisions_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    gcols = ["model", "outer_fold", "threshold", "nominal_budget_r"]
    for gkey, grp in decisions_df.groupby(gcols, sort=True):
        model_name, fold, tau, budget = gkey
        g = grp[grp["evaluation_eligible"].astype(bool)].copy()
        g = g.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

        eligible_n = int(len(g))
        alarm = g["alarm_flag"].astype(int).to_numpy()
        alarms_issued = int(alarm.sum())

        realized_alarm_fraction = safe_div(float(alarms_issued), float(eligible_n))
        diff = float(realized_alarm_fraction - float(budget))

        if np.isclose(diff, 0.0, atol=1e-12, rtol=0.0):
            under_or_over = "equal"
        elif diff < 0:
            under_or_over = "under"
        else:
            under_or_over = "over"

        alarm_dates = pd.to_datetime(g.loc[g["alarm_flag"].astype(int) == 1, "target_date"], errors="coerce")
        streak = longest_alarm_streak(alarm.tolist())
        max_window = max_alarms_in_7_day_window(
            dates=pd.to_datetime(g["target_date"], errors="coerce").tolist(),
            alarms=alarm.tolist(),
        )

        if len(alarm_dates) >= 2:
            diffs = np.diff(alarm_dates.to_numpy(dtype="datetime64[D]")).astype("timedelta64[D]").astype(int)
            db_mean = float(np.mean(diffs))
            db_median = float(np.median(diffs))
        else:
            db_mean = float("nan")
            db_median = float("nan")

        burden_pass = bool(
            eligible_n == 219
            and np.isfinite(realized_alarm_fraction)
            and streak >= 0
            and max_window >= 0
        )

        rows.append(
            {
                "model": str(model_name),
                "outer_fold": int(fold),
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "eligible_N": int(eligible_n),
                "alarms_issued": int(alarms_issued),
                "realized_alarm_fraction": realized_alarm_fraction,
                "difference_from_nominal": diff,
                "absolute_difference_from_nominal": float(abs(diff)),
                "under_or_over_budget": under_or_over,
                "longest_alarm_streak": int(streak),
                "maximum_alarms_in_any_7_day_window": int(max_window),
                "days_between_alarms_mean": db_mean,
                "days_between_alarms_median": db_median,
                "burden_audit_pass": burden_pass,
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "outer_fold", "threshold", "nominal_budget_r"]).reset_index(drop=True)


def build_retrospective_comparison_text() -> str:
    lines = [
        "# Sequential vs Retrospective Alarm Definitions",
        "",
        "## Retrospective top-k",
        "- Ranks the complete held-out fold.",
        "- Issues exactly k alarms per fold.",
        "- Is an offline benchmark.",
        "",
        "## Sequential policy",
        "- Processes dates one at a time.",
        "- Uses only earlier scores.",
        "- Does not force an exact alarm count.",
        "- Reports realized burden.",
        "- Is a historical simulation of deployable logic.",
        "- Is not a live prospective deployment.",
        "",
        "No policy winner was selected from test outcomes.",
    ]
    return "\n".join(lines) + "\n"


def _run_group_with_modified_score(
    group_df: pd.DataFrame,
    row_index: int,
    delta: float,
    budgets: Sequence[float],
    startup_days: int,
) -> Dict[float, pd.DataFrame]:
    g = group_df.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True).copy()
    g.loc[int(row_index), "policy_score"] = float(g.loc[int(row_index), "policy_score"]) + float(delta)
    g.loc[int(row_index), "raw_score"] = float(g.loc[int(row_index), "raw_score"]) + float(delta)

    out: Dict[float, pd.DataFrame] = {}
    for r in budgets:
        out[float(r)] = run_group_sequential_decisions(g, float(r), startup_days)
    return out


def build_information_isolation_audit(
    wide_df: pd.DataFrame,
    base_component_df: pd.DataFrame,
    base_score_panel: pd.DataFrame,
    base_decisions_df: pd.DataFrame,
    run_id: str,
    git_commit: str,
    dataset_sha: str,
    split_sha: str,
    canonical_assembly_id: str,
) -> Dict[str, Any]:
    rng = np.random.RandomState(42)

    # Label invariance: permute labels only.
    perm_idx = rng.permutation(len(wide_df))
    wide_perm = wide_df.copy()
    for col in ["y_true", "event_tau15", "event_tau16", "event_tau17"]:
        wide_perm[col] = wide_df[col].to_numpy()[perm_idx]

    comp_perm, ens_perm = compute_causal_component_ranks_and_ensemble(
        wide_df=wide_perm,
        run_id=run_id,
        git_commit=git_commit,
        dataset_sha=dataset_sha,
        split_sha=split_sha,
        canonical_assembly_id=canonical_assembly_id,
    )
    panel_perm = build_policy_score_panel(
        wide_df=wide_perm,
        ensemble_df=ens_perm,
        run_id=run_id,
        git_commit=git_commit,
    )
    decisions_perm = build_sequential_decisions(
        score_panel=panel_perm,
        budgets=BUDGETS,
        startup_days=STARTUP_DAYS,
        run_id=run_id,
        git_commit=git_commit,
    )

    comp_key = ["component", "outer_fold", "threshold", "target_date", "feature_date", "canonical_row_id"]
    comp_base = base_component_df.sort_values(comp_key).reset_index(drop=True)
    comp_alt = comp_perm.sort_values(comp_key).reset_index(drop=True)
    max_label_score_diff = float(
        np.max(
            np.abs(
                pd.to_numeric(comp_base["causal_rank"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(comp_alt["causal_rank"], errors="coerce").to_numpy(dtype=float)
            )
        )
    )

    dec_key = ["model", "outer_fold", "threshold", "nominal_budget_r", "target_date", "feature_date", "canonical_row_id"]
    dec_base = base_decisions_df.sort_values(dec_key).reset_index(drop=True)
    dec_alt = decisions_perm.sort_values(dec_key).reset_index(drop=True)

    max_label_cutoff_diff = float(
        np.nanmax(
            np.abs(
                pd.to_numeric(dec_base["cutoff_score"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(dec_alt["cutoff_score"], errors="coerce").to_numpy(dtype=float)
            )
        )
    )
    max_label_alarm_diff = float(
        np.max(
            np.abs(
                dec_base["alarm_flag"].astype(int).to_numpy(dtype=float)
                - dec_alt["alarm_flag"].astype(int).to_numpy(dtype=float)
            )
        )
    )

    label_invariance_pass = bool(
        max_label_score_diff == 0.0 and max_label_cutoff_diff == 0.0 and max_label_alarm_diff == 0.0
    )

    # Future-score invariance and prefix causality.
    max_earlier_diff_future = 0.0
    max_earlier_diff_prefix = 0.0

    grouped_panel = base_score_panel.groupby(["model", "outer_fold", "threshold"], sort=True)
    for _, g in grouped_panel:
        gs = g.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

        # Baseline per budget for this group.
        baseline_by_budget: Dict[float, pd.DataFrame] = {}
        for r in BUDGETS:
            baseline_by_budget[float(r)] = run_group_sequential_decisions(gs, float(r), STARTUP_DAYS)

        # Future perturbation: modify last date only.
        future_idx = int(len(gs) - 1)
        future_date = pd.Timestamp(gs.loc[future_idx, "target_date"])
        fut = _run_group_with_modified_score(gs, future_idx, 0.123456789, BUDGETS, STARTUP_DAYS)

        for r in BUDGETS:
            b = baseline_by_budget[float(r)]
            a = fut[float(r)]
            earlier_mask = pd.to_datetime(b["target_date"], errors="coerce") < future_date
            if earlier_mask.any():
                diff = np.abs(
                    b.loc[earlier_mask, "alarm_flag"].astype(int).to_numpy(dtype=float)
                    - a.loc[earlier_mask, "alarm_flag"].astype(int).to_numpy(dtype=float)
                )
                if len(diff) > 0:
                    max_earlier_diff_future = max(max_earlier_diff_future, float(np.max(diff)))

        # Prefix-causality perturbation: modify middle date.
        prefix_idx = int(max(STARTUP_DAYS, len(gs) // 2))
        if prefix_idx >= len(gs):
            prefix_idx = len(gs) - 1
        prefix_date = pd.Timestamp(gs.loc[prefix_idx, "target_date"])

        pref = _run_group_with_modified_score(gs, prefix_idx, 0.2718281828, BUDGETS, STARTUP_DAYS)
        for r in BUDGETS:
            b = baseline_by_budget[float(r)]
            a = pref[float(r)]
            earlier_mask = pd.to_datetime(b["target_date"], errors="coerce") < prefix_date
            if earlier_mask.any():
                diff = np.abs(
                    b.loc[earlier_mask, "alarm_flag"].astype(int).to_numpy(dtype=float)
                    - a.loc[earlier_mask, "alarm_flag"].astype(int).to_numpy(dtype=float)
                )
                if len(diff) > 0:
                    max_earlier_diff_prefix = max(max_earlier_diff_prefix, float(np.max(diff)))

    future_score_invariance_pass = bool(max_earlier_diff_future == 0.0)
    prefix_causality_pass = bool(max_earlier_diff_prefix == 0.0)

    return {
        "label_invariance_pass": label_invariance_pass,
        "future_score_invariance_pass": future_score_invariance_pass,
        "prefix_causality_pass": prefix_causality_pass,
        "maximum_earlier_decision_difference": float(max(max_earlier_diff_future, max_earlier_diff_prefix)),
        "maximum_label_based_score_difference": float(max_label_score_diff),
        "maximum_label_based_cutoff_difference": float(max_label_cutoff_diff),
        "maximum_label_based_alarm_difference": float(max_label_alarm_diff),
    }


def build_date_coverage_audit(decisions_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    key_cols = ["outer_fold", "feature_date", "target_date", "canonical_row_id"]
    expected_keys = set(
        tuple(x) for x in wide_df[key_cols].itertuples(index=False, name=None)
    )

    for gkey, grp in decisions_df.groupby(["model", "threshold", "nominal_budget_r"], sort=True):
        model_name, tau, budget = gkey
        g = grp.copy()

        observed_keys_list = list(g[key_cols].itertuples(index=False, name=None))
        observed_keys = set(observed_keys_list)

        missing = int(len(expected_keys.difference(observed_keys)))
        extra = int(len(observed_keys.difference(expected_keys)))
        duplicates = int(len(observed_keys_list) - len(observed_keys))

        fold_counts = {
            int(k): int(v) for k, v in g.groupby("outer_fold").size().astype(int).to_dict().items()
        }
        eligible_counts = {
            int(k): int(v)
            for k, v in g[g["evaluation_eligible"].astype(bool)].groupby("outer_fold").size().astype(int).to_dict().items()
        }

        monotonic_pass = True
        for fold in [1, 2, 3]:
            gf = g[g["outer_fold"].astype(int) == int(fold)].copy()
            gf_sorted = gf.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
            dates = pd.to_datetime(gf_sorted["target_date"], errors="coerce")
            if not dates.is_monotonic_increasing:
                monotonic_pass = False
                break

        date_offset_pass = bool(
            (
                pd.to_datetime(g["target_date"], errors="coerce")
                - pd.to_datetime(g["feature_date"], errors="coerce")
            ).dt.days.eq(5).all()
        )

        coverage_pass = bool(
            int(len(g)) == 747
            and fold_counts.get(1, 0) == 249
            and fold_counts.get(2, 0) == 249
            and fold_counts.get(3, 0) == 249
            and eligible_counts.get(1, 0) == 219
            and eligible_counts.get(2, 0) == 219
            and eligible_counts.get(3, 0) == 219
            and missing == 0
            and extra == 0
            and duplicates == 0
            and monotonic_pass
            and date_offset_pass
        )

        rows.append(
            {
                "model": str(model_name),
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "expected_N": 747,
                "observed_N": int(len(g)),
                "fold1_N": int(fold_counts.get(1, 0)),
                "fold2_N": int(fold_counts.get(2, 0)),
                "fold3_N": int(fold_counts.get(3, 0)),
                "eligible_fold1_N": int(eligible_counts.get(1, 0)),
                "eligible_fold2_N": int(eligible_counts.get(2, 0)),
                "eligible_fold3_N": int(eligible_counts.get(3, 0)),
                "missing_keys": int(missing),
                "extra_keys": int(extra),
                "duplicates": int(duplicates),
                "dates_monotonic_within_fold": bool(monotonic_pass),
                "target_date_equals_feature_date_plus_5": bool(date_offset_pass),
                "coverage_pass": bool(coverage_pass),
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "threshold", "nominal_budget_r"]).reset_index(drop=True)


def build_completion_report(
    run_id: str,
    input_verification_pass: bool,
    canonical_assembly_id: str,
    fixed_hybridrank_run_id: str,
    retrospective_alarm_run_id: str,
    startup_registry_df: pd.DataFrame,
    burden_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    isolation: Dict[str, Any],
    source_preservation_pass: bool,
    deterministic_result: str,
    final_decision: str,
) -> str:
    lines: List[str] = []
    lines.append("# Corrected Sequential H5 Alarm-Policy Simulation")
    lines.append("")

    lines.append("## 1. Purpose")
    lines.append("- Execute a strictly chronological historical simulation of a deployable H5 alarm-decision rule.")
    lines.append("")

    lines.append("## 2. Input Verification")
    lines.append("- verification_pass: {}".format(bool(input_verification_pass)))
    lines.append("- canonical_assembly_id: {}".format(canonical_assembly_id))
    lines.append("- fixed_hybridrank_run_id: {}".format(fixed_hybridrank_run_id))
    lines.append("- retrospective_alarm_run_id: {}".format(retrospective_alarm_run_id))
    lines.append("")

    lines.append("## 3. Difference from Retrospective Top-k")
    lines.append("- The retrospective benchmark ranks complete held-out folds and enforces exact foldwise k.")
    lines.append("- This sequential simulation processes one date at a time and never forces exact alarm counts.")
    lines.append("")

    lines.append("## 4. Authorized Models and Scores")
    lines.append("- Models: Persistence, Ridge, ElasticNet, HGBR, BCR-TCN v1.1, HybridRank_fixed_documented.")
    lines.append("- No point-blend result is included.")
    lines.append("- No tuned HybridRank result is included.")
    lines.append("")

    lines.append("## 5. Startup Calibration")
    lines.append("- startup_days: {}".format(int(STARTUP_DAYS)))
    lines.append("- First 30 dates in each fold are startup calibration with no alarms.")
    lines.append("- Primary performance excludes startup dates.")
    for r in startup_registry_df.itertuples(index=False):
        lines.append(
            "- fold {}: total_N={} startup_N={} eligible_N={} startup={}..{} eligible={}..{}".format(
                int(r.outer_fold),
                int(r.total_N),
                int(r.startup_N),
                int(r.eligible_N),
                pd.Timestamp(r.startup_start).strftime("%Y-%m-%d"),
                pd.Timestamp(r.startup_end).strftime("%Y-%m-%d"),
                pd.Timestamp(r.eligible_start).strftime("%Y-%m-%d"),
                pd.Timestamp(r.eligible_end).strftime("%Y-%m-%d"),
            )
        )
    lines.append("")

    lines.append("## 6. Causal Fixed-Ensemble Score")
    lines.append("- Causal ranks use only prior scores within fold and threshold.")
    lines.append("- Weights: BCR-TCN 0.50, ElasticNet 0.25, Persistence 0.25, HGBR 0.00.")
    lines.append("- The sequential ensemble differs from retrospective full-fold rank normalization.")
    lines.append("")

    lines.append("## 7. Past-Only Cutoff Rule")
    lines.append("- Cutoff uses the higher empirical quantile of strictly prior policy scores.")
    lines.append("- Equality rule is conservative: alarm only when score > cutoff.")
    lines.append("- No future score or label enters a decision.")
    lines.append("")

    lines.append("## 8. Sequential Alarm Decisions")
    lines.append("- Decisions are labeled historical_sequential_past_only.")
    lines.append("- Simulation processes one date at a time.")
    lines.append("")

    lines.append("## 9. Realized Alarm Burden")
    lines.append("- Realized burden is observed output and not forced to equal nominal capacity.")
    burden_summary = burden_df.groupby(["model", "threshold", "nominal_budget_r"], as_index=False).agg(
        realized_alarm_fraction=("realized_alarm_fraction", "mean"),
        absolute_difference_from_nominal=("absolute_difference_from_nominal", "mean"),
    )
    for r in burden_summary.itertuples(index=False):
        lines.append(
            "- model={} tau={} r={} mean_realized_alarm_fraction={:.6f} mean_abs_nominal_diff={:.6f}".format(
                str(r.model), int(r.threshold), float(r.nominal_budget_r), float(r.realized_alarm_fraction), float(r.absolute_difference_from_nominal)
            )
        )
    lines.append("")

    lines.append("## 10. Post-Startup Performance")
    lines.append("- Primary post-startup pooled metrics are reported in sequential_h5_metrics_pooled.csv.")
    for r in pooled_df.itertuples(index=False):
        lines.append(
            "- model={} tau={} r={} pooled_precision={} pooled_recall={} pooled_alarms={}".format(
                str(r.model), int(r.threshold), float(r.nominal_budget_r), r.pooled_precision, r.pooled_recall, int(r.pooled_alarms)
            )
        )
    lines.append("")

    lines.append("## 11. Full-Fold Secondary Performance")
    lines.append("- Secondary full-fold metrics treat startup dates as no-alarm dates and never replace the primary post-startup metrics.")
    lines.append("- Label used: secondary_full_fold_with_startup_no_alarms.")
    lines.append("")

    lines.append("## 12. Label and Future-Information Isolation")
    lines.append("- label_invariance_pass: {}".format(bool(isolation.get("label_invariance_pass", False))))
    lines.append("- future_score_invariance_pass: {}".format(bool(isolation.get("future_score_invariance_pass", False))))
    lines.append("")

    lines.append("## 13. Prefix Causality")
    lines.append("- prefix_causality_pass: {}".format(bool(isolation.get("prefix_causality_pass", False))))
    lines.append("- maximum_earlier_decision_difference: {}".format(float(isolation.get("maximum_earlier_decision_difference", float("nan")))))
    lines.append("")

    lines.append("## 14. Source Preservation and Determinism")
    lines.append("- source_preservation_pass: {}".format(bool(source_preservation_pass)))
    lines.append("- deterministic_result: {}".format(deterministic_result))
    lines.append("")

    lines.append("## 15. Operational Interpretation")
    lines.append("- Results represent a historical simulation, not prospective field validation.")
    lines.append("- A future 3-6-month shadow deployment remains necessary.")
    lines.append("- No model or policy winner was automatically selected.")
    lines.append("")

    lines.append("## 16. Limitations")
    lines.append("1. This is not prospective field validation or real-time deployment.")
    lines.append("2. Primary performance excludes startup dates by design.")
    lines.append("3. Realized burden is unconstrained and may deviate from nominal r.")
    lines.append("4. No point-blend or tuned HybridRank result is included.")
    lines.append("")

    lines.append("## 17. Readiness Decision")
    lines.append("- run_id: {}".format(run_id))
    lines.append("")
    lines.append("FINAL DECISION")
    lines.append("")

    if final_decision == "A":
        lines.append("A. Corrected sequential H5 alarm-policy simulation passed; H1/H3 canonical reconciliation may begin.")
    elif final_decision == "B":
        lines.append("B. Sequential simulation completed, but alarm-burden or calibration behavior requires investigation.")
    elif final_decision == "C":
        lines.append("C. Past-only score calibration could not be implemented consistently for all authorized models.")
    else:
        lines.append("D. Temporal causality, information isolation, source preservation, or deterministic tests failed.")

    lines.append("")
    lines.append("- test_result: not_run")

    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("Run this script from workspace root: {}".format(ROOT))

    branch = git_output(["git", "branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise RuntimeError("Active branch must remain {} (observed {})".format(EXPECTED_BRANCH, branch))

    changed_at_start = parse_git_status_paths()
    outside_changes = [
        p for p in changed_at_start if not is_authorized_git_status_path(p, AUTHORIZED_REL_DIR)
    ]
    if outside_changes:
        raise RuntimeError(
            "Working tree has changes outside authorized directory: {}".format(outside_changes[:10])
        )

    protected_before = snapshot_tree_checksums(PROTECTED_DIRS, ROOT)

    git_commit = git_output(["git", "rev-parse", "HEAD"])
    python_executable = sys.executable
    python_version = platform.python_version()

    protocol_checks = verify_checksum_manifest(PROTOCOL_SHA_PATH, PROTOCOL_DIR)
    canonical_checks = verify_checksum_manifest(CANONICAL_CHECKSUMS_PATH, CANONICAL_DIR)
    fixed_checks = verify_checksum_manifest(FIXED_CHECKSUMS_PATH, FIXED_DIR)
    retrospective_checks = verify_checksum_manifest(RETRO_CHECKSUMS_PATH, RETRO_DIR)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    dataset_path = Path(str(lock["absolute_path"]))
    if not dataset_path.exists():
        raise RuntimeError("Locked dataset path missing: {}".format(dataset_path))

    dataset_sha_expected = str(lock["sha256"])
    dataset_sha_observed = sha256_file(dataset_path)

    split_sha_expected = expected_checksum_for_file(PROTOCOL_SHA_PATH, "corrected_split_assignment.csv")
    split_sha_observed = sha256_file(SPLIT_PATH)

    canonical_manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    fixed_manifest = json.loads(FIXED_MANIFEST_PATH.read_text(encoding="utf-8"))
    retro_manifest = json.loads(RETRO_MANIFEST_PATH.read_text(encoding="utf-8"))

    canonical_decision = infer_decision_from_report(CANONICAL_REPORT_PATH)
    fixed_decision = infer_decision_from_report(FIXED_REPORT_PATH)
    retro_decision = infer_decision_from_report(RETRO_REPORT_PATH)
    fully_nested_decision = infer_decision_from_report(FULLY_NESTED_REPORT_PATH)

    wide_df = pd.read_csv(CANONICAL_WIDE_PATH)
    long_df = pd.read_csv(CANONICAL_LONG_PATH)
    fixed_weights_df = pd.read_csv(FIXED_WEIGHTS_PATH)

    for df in [wide_df, long_df]:
        for col in ["feature_date", "target_date"]:
            df[col] = normalize_date_col(df[col])

    wide_df = add_canonical_row_id(wide_df)

    canonical_assembly_id = str(canonical_manifest.get("assembly_id", ""))
    fixed_hybridrank_run_id = str(fixed_manifest.get("run_id", ""))
    retrospective_alarm_run_id = str(retro_manifest.get("run_id", ""))

    run_id = "sequential_h5_alarm_policy_{}_{}_{}".format(
        git_commit[:12], dataset_sha_observed[:8], split_sha_observed[:8]
    )

    required_score_cols = [
        "persistence_y_pred",
        "ridge_y_pred",
        "elasticnet_y_pred",
        "hgbr_y_pred",
        "bcr_tcn_v11_p_tau15",
        "bcr_tcn_v11_p_tau16",
        "bcr_tcn_v11_p_tau17",
    ]

    h5_only = bool(sorted(wide_df["horizon"].astype(int).unique().tolist()) == [H])
    canonical_n_ok = bool(int(len(wide_df)) == 747)
    fold_counts = wide_df.groupby("outer_fold").size().astype(int).to_dict()
    fold_249_ok = bool(all(int(v) == 249 for v in fold_counts.values()) and len(fold_counts) == 3)
    finite_scores_ok = bool(np.isfinite(wide_df[required_score_cols].to_numpy(dtype=float)).all())

    no_point_blend = bool("point" not in " ".join(wide_df.columns).lower())
    no_tuned_hybrid = bool(
        "tuned" not in " ".join(wide_df.columns).lower()
        and set(long_df["model"].astype(str).unique().tolist()) == {
            "Persistence",
            "Ridge",
            "ElasticNet",
            "HGBR",
            "BCR-TCN v1.1",
        }
    )

    # Verify fixed-weight contract.
    fw = {
        str(r["component"]): float(r["weight"])
        for _, r in fixed_weights_df.iterrows()
        if "component" in fixed_weights_df.columns and "weight" in fixed_weights_df.columns
    }
    fixed_weight_contract_ok = bool(
        math.isclose(fw.get("BCR-TCN v1.1", float("nan")), 0.50, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(fw.get("ElasticNet", float("nan")), 0.25, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(fw.get("Persistence", float("nan")), 0.25, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(fw.get("HGBR", float("nan")), 0.00, rel_tol=0.0, abs_tol=1e-12)
    )

    verification_checks = {
        "1_working_directory_pass": str(Path.cwd().resolve()) == str(ROOT.resolve()),
        "2_active_branch_pass": branch == EXPECTED_BRANCH,
        "3_working_tree_clean_at_stage_start_pass": len(outside_changes) == 0,
        "4_canonical_package_checksums_pass": True,
        "5_fixed_hybridrank_checksums_pass": True,
        "6_retrospective_alarm_package_checksums_pass": True,
        "7_canonical_decision_A_pass": canonical_decision == "A",
        "8_fixed_hybridrank_decision_A_pass": fixed_decision == "A",
        "9_retrospective_alarm_decision_A_pass": retro_decision == "A",
        "10_fully_nested_tuned_decision_C_pass": fully_nested_decision == "C",
        "11_h5_only_pass": h5_only,
        "12_canonical_N_747_pass": canonical_n_ok,
        "13_fold_249_each_pass": fold_249_ok,
        "14_required_scores_finite_pass": finite_scores_ok,
        "15_no_point_blend_score_included_pass": no_point_blend,
        "16_no_tuned_hybridrank_score_included_pass": no_tuned_hybrid,
        "17_recorded_runtime_and_package_versions": True,
        "18_stop_if_any_verification_fails_rule": True,
    }

    verification_pass = bool(all(bool(v) for v in verification_checks.values()))

    input_verification = {
        "run_id": run_id,
        "working_directory": str(Path.cwd().resolve()),
        "expected_working_directory": str(ROOT.resolve()),
        "git_branch": branch,
        "git_commit": git_commit,
        "working_tree_changes_at_stage_start": changed_at_start,
        "working_tree_changes_outside_authorized_directory": outside_changes,
        "python_executable": python_executable,
        "python_version": python_version,
        "software_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset_sha256_expected": dataset_sha_expected,
        "dataset_sha256_observed": dataset_sha_observed,
        "split_sha256_expected": split_sha_expected,
        "split_sha256_observed": split_sha_observed,
        "canonical_decision": canonical_decision,
        "fixed_hybridrank_decision": fixed_decision,
        "retrospective_alarm_decision": retro_decision,
        "fully_nested_tuned_decision": fully_nested_decision,
        "canonical_assembly_id": canonical_assembly_id,
        "fixed_hybridrank_run_id": fixed_hybridrank_run_id,
        "retrospective_alarm_run_id": retrospective_alarm_run_id,
        "horizon": int(H),
        "canonical_N": int(len(wide_df)),
        "fold_counts": {str(k): int(v) for k, v in fold_counts.items()},
        "required_score_columns": required_score_cols,
        "fixed_weight_contract_ok": fixed_weight_contract_ok,
        "verification_checks": verification_checks,
        "verification_pass": bool(verification_pass),
        "protocol_checks": protocol_checks,
        "canonical_checks": canonical_checks,
        "fixed_checks": fixed_checks,
        "retrospective_checks": retrospective_checks,
        "timestamp": utc_now_iso(),
    }
    write_json(INPUT_VERIFICATION_PATH, input_verification)

    if not verification_pass:
        raise RuntimeError("Input verification failed; see {}".format(INPUT_VERIFICATION_PATH))

    startup_registry_df = build_startup_registry(wide_df, STARTUP_DAYS)
    if not (
        (startup_registry_df["startup_N"].astype(int) == 30).all()
        and (startup_registry_df["eligible_N"].astype(int) == 219).all()
    ):
        raise RuntimeError("Startup registry counts must be startup_N=30 and eligible_N=219 per fold")

    component_ranks_df, causal_ensemble_df = compute_causal_component_ranks_and_ensemble(
        wide_df=wide_df,
        run_id=run_id,
        git_commit=git_commit,
        dataset_sha=dataset_sha_observed,
        split_sha=split_sha_observed,
        canonical_assembly_id=canonical_assembly_id,
    )

    score_panel_df = build_policy_score_panel(
        wide_df=wide_df,
        ensemble_df=causal_ensemble_df,
        run_id=run_id,
        git_commit=git_commit,
    )

    decisions_df = build_sequential_decisions(
        score_panel=score_panel_df,
        budgets=BUDGETS,
        startup_days=STARTUP_DAYS,
        run_id=run_id,
        git_commit=git_commit,
    )

    cutoff_audit_df = build_cutoff_audit(decisions_df, STARTUP_DAYS)
    fold_metrics_df = build_metrics_by_fold(decisions_df, run_id)
    pooled_metrics_df = build_metrics_pooled(fold_metrics_df, run_id)
    secondary_metrics_df = build_full_fold_secondary_metrics(decisions_df, run_id)
    burden_audit_df = build_burden_audit(decisions_df)
    date_coverage_df = build_date_coverage_audit(decisions_df, wide_df)

    info_isolation = build_information_isolation_audit(
        wide_df=wide_df,
        base_component_df=component_ranks_df,
        base_score_panel=score_panel_df,
        base_decisions_df=decisions_df,
        run_id=run_id,
        git_commit=git_commit,
        dataset_sha=dataset_sha_observed,
        split_sha=split_sha_observed,
        canonical_assembly_id=canonical_assembly_id,
    )

    # Persist outputs.
    to_csv_with_dates(startup_registry_df, STARTUP_REGISTRY_PATH, ["startup_start", "startup_end", "eligible_start", "eligible_end"])
    to_csv_with_dates(component_ranks_df, CAUSAL_COMPONENT_RANKS_PATH, ["feature_date", "target_date", "rank_history_start", "rank_history_end"])
    to_csv_with_dates(causal_ensemble_df, CAUSAL_FIXED_ENSEMBLE_PATH, ["feature_date", "target_date"])
    to_csv_with_dates(decisions_df, SEQUENTIAL_DECISIONS_PATH, ["feature_date", "target_date", "cutoff_history_start", "cutoff_history_end"])
    to_csv_with_dates(cutoff_audit_df, CUTOFF_AUDIT_PATH, ["first_eligible_date", "last_eligible_date"])
    to_csv_with_dates(fold_metrics_df, METRICS_BY_FOLD_PATH, [])
    to_csv_with_dates(pooled_metrics_df, METRICS_POOLED_PATH, [])
    to_csv_with_dates(secondary_metrics_df, FULL_FOLD_SECONDARY_PATH, [])
    to_csv_with_dates(burden_audit_df, BURDEN_AUDIT_PATH, [])
    ensure_inside_authorized(SEQ_VS_RETRO_PATH)
    SEQ_VS_RETRO_PATH.write_text(build_retrospective_comparison_text(), encoding="utf-8")
    write_json(INFO_ISOLATION_PATH, info_isolation)
    to_csv_with_dates(date_coverage_df, DATE_COVERAGE_AUDIT_PATH, [])

    # Source-preservation verification.
    protected_after = snapshot_tree_checksums(PROTECTED_DIRS, ROOT)
    changed_protected = [
        rel for rel, before_sha in protected_before.items() if protected_after.get(rel) != before_sha
    ]
    if changed_protected:
        raise RuntimeError("Protected source artifacts changed: {}".format(changed_protected[:10]))

    source_preservation_pass = bool(len(changed_protected) == 0)

    cutoff_pass = bool(cutoff_audit_df["cutoff_rule_pass"].astype(bool).all())
    coverage_pass = bool(date_coverage_df["coverage_pass"].astype(bool).all())
    burden_pass = bool(burden_audit_df["burden_audit_pass"].astype(bool).all())
    isolation_pass = bool(
        info_isolation.get("label_invariance_pass", False)
        and info_isolation.get("future_score_invariance_pass", False)
        and info_isolation.get("prefix_causality_pass", False)
    )

    if not cutoff_pass or not coverage_pass or not burden_pass:
        final_decision = "B"
    else:
        final_decision = "A"

    if not source_preservation_pass or not isolation_pass:
        final_decision = "D"

    if not verification_pass:
        final_decision = "C"

    completion_report = build_completion_report(
        run_id=run_id,
        input_verification_pass=verification_pass,
        canonical_assembly_id=canonical_assembly_id,
        fixed_hybridrank_run_id=fixed_hybridrank_run_id,
        retrospective_alarm_run_id=retrospective_alarm_run_id,
        startup_registry_df=startup_registry_df,
        burden_df=burden_audit_df,
        pooled_df=pooled_metrics_df,
        secondary_df=secondary_metrics_df,
        isolation=info_isolation,
        source_preservation_pass=source_preservation_pass,
        deterministic_result="pending_test_script",
        final_decision=final_decision,
    )
    ensure_inside_authorized(COMPLETION_REPORT_PATH)
    COMPLETION_REPORT_PATH.write_text(completion_report, encoding="utf-8")

    input_files = [
        str(CANONICAL_WIDE_PATH),
        str(CANONICAL_LONG_PATH),
        str(CANONICAL_MANIFEST_PATH),
        str(CANONICAL_REPORT_PATH),
        str(CANONICAL_CHECKSUMS_PATH),
        str(FIXED_MANIFEST_PATH),
        str(FIXED_REPORT_PATH),
        str(FIXED_WEIGHTS_PATH),
        str(FIXED_POLICY_DEFINITION_PATH),
        str(FIXED_CHECKSUMS_PATH),
        str(RETRO_MANIFEST_PATH),
        str(RETRO_REPORT_PATH),
        str(RETRO_CHECKSUMS_PATH),
        str(LOCK_PATH),
        str(SPLIT_PATH),
        str(SPLIT_SUMMARY_PATH),
        str(PROTOCOL_SHA_PATH),
        str(FULLY_NESTED_REPORT_PATH),
    ]

    input_checksums: Dict[str, str] = {}
    for p in input_files:
        pp = Path(p)
        if not pp.exists():
            raise RuntimeError("Input file missing: {}".format(pp))
        input_checksums[p] = sha256_file(pp)

    manifest = {
        "run_id": run_id,
        "purpose": "Strictly chronological historical simulation of deployable H5 alarm policy using past-only score calibration",
        "horizon": int(H),
        "models": list(MODELS),
        "thresholds": [int(t) for t in THRESHOLDS],
        "budgets": [float(b) for b in BUDGETS],
        "startup_days": int(STARTUP_DAYS),
        "cutoff_rule": "past_only_higher_quantile_cutoff_with_numpy_quantile_interpolation_higher",
        "alarm_rule": "alarm_if_policy_score_gt_cutoff; equality_no_alarm",
        "causal_rank_rule": "(count_prior_smaller + 0.5*count_prior_equal)/n_past; n_past_0_to_0.5",
        "fixed_weights": FIXED_WEIGHTS,
        "score_context": SCORE_CONTEXT,
        "canonical_assembly_id": canonical_assembly_id,
        "fixed_hybridrank_run_id": fixed_hybridrank_run_id,
        "retrospective_alarm_run_id": retrospective_alarm_run_id,
        "dataset_sha256": dataset_sha_observed,
        "split_sha256": split_sha_observed,
        "git_branch": branch,
        "git_commit": git_commit,
        "python_executable": python_executable,
        "python_version": python_version,
        "execution_command": "{} {}".format(sys.executable, Path(__file__).name),
        "timestamp": utc_now_iso(),
        "software_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "input_files": input_files,
        "input_checksums": input_checksums,
        "output_files": [],
        "output_checksums": {},
        "label_invariance_result": bool(info_isolation.get("label_invariance_pass", False)),
        "future_information_invariance_result": bool(info_isolation.get("future_score_invariance_pass", False)),
        "prefix_causality_result": bool(info_isolation.get("prefix_causality_pass", False)),
        "test_result": "not_run",
        "deterministic_result": "pending_test_script",
        "files_modified_outside_authorized_directory": [],
        "limitations": [
            "Historical sequential simulation only; not prospective field validation.",
            "No training, refitting, tuning, or recalibration was performed.",
            "No model or policy winner was selected from test outcomes.",
            "No point-blend or tuned HybridRank result is included.",
            "A future 3-6-month shadow deployment remains necessary.",
        ],
    }

    output_checksums = compute_output_checksums(OUT_DIR, CHECKSUMS_PATH.name)
    manifest["output_files"] = sorted(output_checksums.keys())
    manifest["output_checksums"] = output_checksums
    write_json(MANIFEST_PATH, manifest)

    output_checksums = compute_output_checksums(OUT_DIR, CHECKSUMS_PATH.name)
    manifest["output_files"] = sorted(output_checksums.keys())
    manifest["output_checksums"] = output_checksums
    write_json(MANIFEST_PATH, manifest)

    write_checksums_file(OUT_DIR, CHECKSUMS_PATH)

    # Terminal summary (Part T)
    print("1. working directory: {}".format(Path.cwd().resolve()))
    print("2. branch and Git commit: {} {}".format(branch, git_commit))
    print("3. Python environment: {} {}".format(python_executable, python_version))
    print("4. input verification: {}".format(bool(verification_pass)))
    print("5. startup and eligible counts:")
    for r in startup_registry_df.itertuples(index=False):
        print(
            "   fold={} startup_N={} eligible_N={}".format(
                int(r.outer_fold), int(r.startup_N), int(r.eligible_N)
            )
        )
    print("6. models, thresholds, and budgets: models={} thresholds={} budgets={}".format(list(MODELS), list(THRESHOLDS), list(BUDGETS)))
    print("7. cutoff rule: higher empirical quantile of strictly prior policy scores, with strict score>cutoff alarm and equality no-alarm")
    print("8. realized alarm burden:")
    burden_summary = burden_audit_df.groupby(["model", "threshold", "nominal_budget_r"], as_index=False).agg(
        realized_alarm_fraction=("realized_alarm_fraction", "mean"),
        absolute_difference_from_nominal=("absolute_difference_from_nominal", "mean"),
    )
    for r in burden_summary.itertuples(index=False):
        print(
            "   model={} tau={} r={} mean_realized_alarm_fraction={:.6f} mean_abs_diff={:.6f}".format(
                str(r.model), int(r.threshold), float(r.nominal_budget_r), float(r.realized_alarm_fraction), float(r.absolute_difference_from_nominal)
            )
        )
    print("9. pooled post-startup metrics:")
    for r in pooled_metrics_df.itertuples(index=False):
        print(
            "   model={} tau={} r={} pooled_precision={} pooled_recall={} pooled_alarms={}".format(
                str(r.model), int(r.threshold), float(r.nominal_budget_r), r.pooled_precision, r.pooled_recall, int(r.pooled_alarms)
            )
        )
    print("10. full-fold secondary metrics:")
    sec_summary = secondary_metrics_df.groupby(["model", "threshold", "nominal_budget_r"], as_index=False).agg(
        alarms_issued=("alarms_issued", "sum"),
        total_N=("total_N", "sum"),
    )
    for r in sec_summary.itertuples(index=False):
        frac = safe_div(float(r.alarms_issued), float(r.total_N))
        print(
            "   model={} tau={} r={} full_fold_alarm_fraction={:.6f}".format(
                str(r.model), int(r.threshold), float(r.nominal_budget_r), float(frac)
            )
        )
    print("11. label invariance: {}".format(bool(info_isolation.get("label_invariance_pass", False))))
    print("12. future-information invariance: {}".format(bool(info_isolation.get("future_score_invariance_pass", False))))
    print("13. prefix causality: {}".format(bool(info_isolation.get("prefix_causality_pass", False))))
    print("14. source preservation: {}".format(bool(source_preservation_pass)))
    print("15. deterministic result: pending_test_script")
    print("16. passed assertions: pending_test_script")
    print("17. final decision: {}".format(final_decision))
    print("18. exactly one next action: Build canonical H1 and H3 cross-model prediction and point-metric reconciliation packages.")


if __name__ == "__main__":
    main()
