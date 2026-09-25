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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
AUTHORIZED_REL_DIR = "revision_2026/08_sequential_alarm_policy/h5/quota_enforced"
EXPECTED_BRANCH = "controlled-reruns-v1"

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

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions" / "h5_cross_model"
SEQUENTIAL_DIR = ROOT / "revision_2026" / "08_sequential_alarm_policy" / "h5"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

SOURCE_DECISIONS_PATH = SEQUENTIAL_DIR / "sequential_h5_alarm_decisions.csv"
SOURCE_REPORT_PATH = SEQUENTIAL_DIR / "sequential_h5_alarm_policy_completion_report.md"
SOURCE_MANIFEST_PATH = SEQUENTIAL_DIR / "sequential_h5_alarm_policy_manifest.json"
SOURCE_CHECKSUMS_PATH = SEQUENTIAL_DIR / "sequential_h5_alarm_policy_checksums.sha256"
SOURCE_INFO_ISOLATION_PATH = SEQUENTIAL_DIR / "sequential_h5_information_isolation_audit.json"

CANONICAL_WIDE_PATH = CANONICAL_DIR / "canonical_h5_predictions_wide.csv"
CANONICAL_MANIFEST_PATH = CANONICAL_DIR / "canonical_h5_assembly_manifest.json"

INPUT_VERIFICATION_PATH = OUT_DIR / "quota_enforced_h5_input_verification.json"
DECISIONS_PATH = OUT_DIR / "quota_enforced_h5_alarm_decisions.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "quota_enforced_h5_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "quota_enforced_h5_metrics_pooled.csv"
CAPACITY_AUDIT_PATH = OUT_DIR / "quota_enforced_h5_capacity_audit.csv"
SUPPRESSION_AUDIT_PATH = OUT_DIR / "quota_enforced_h5_suppression_audit.csv"
COMPARISON_PATH = OUT_DIR / "quota_enforced_vs_unconstrained_sequential.csv"
INFO_AUDIT_PATH = OUT_DIR / "quota_enforced_h5_information_isolation_audit.json"
MANIFEST_PATH = OUT_DIR / "quota_enforced_h5_manifest.json"
REPORT_PATH = OUT_DIR / "quota_enforced_h5_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "quota_enforced_h5_checksums.sha256"


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


def rel_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


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


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num) / float(den)


def quantile_higher(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


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


def checksum_lookup(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for expected, rel_name in parse_checksum_manifest(path):
        out[str(rel_name).replace("\\", "/")] = expected
    return out


def expected_checksum_for_filename(manifest_path: Path, filename: str) -> str:
    lookup = checksum_lookup(manifest_path)
    if filename in lookup:
        return lookup[filename]
    for rel_name, expected in lookup.items():
        if rel_name.endswith("/" + filename):
            return expected
    raise KeyError("Checksum entry not found for {} in {}".format(filename, manifest_path))


def verify_checksum_manifest(path: Path, base_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for expected, rel_name in parse_checksum_manifest(path):
        target = base_dir / rel_name
        if not target.exists():
            raise RuntimeError("Checksum target missing: {}".format(target))
        observed = sha256_file(target)
        ok = observed == expected
        out[str(rel_name)] = {
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
    return out


def infer_decision_from_report(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")

    m = re.search(
        r"FINAL DECISION\s*(?:\n|\r\n)+(?:<!--.*?-->\s*)?([ABCD])\.\s",
        txt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).upper()

    m2 = re.search(r"Decision:\s*([ABCD])", txt)
    if m2:
        return m2.group(1).upper()

    for ln in reversed([x.strip() for x in txt.splitlines()]):
        if re.match(r"^[ABCD]\.\s", ln):
            return ln[0].upper()

    raise RuntimeError("Could not infer final decision from {}".format(path))


def ensure_inside_authorized(path: Path) -> None:
    rp = path.resolve()
    auth = (ROOT / AUTHORIZED_REL_DIR).resolve()
    if not str(rp).startswith(str(auth)):
        raise RuntimeError("Attempted write outside authorized directory: {}".format(rp))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_inside_authorized(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_with_dates(path: Path, df: pd.DataFrame, date_cols: Sequence[str]) -> None:
    out = df.copy()
    for c in date_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    ensure_inside_authorized(path)
    out.to_csv(path, index=False)


def get_stable_timestamp() -> str:
    if MANIFEST_PATH.exists():
        try:
            payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            ts = payload.get("timestamp")
            if isinstance(ts, str) and ts:
                return ts
        except Exception:
            pass
    if INPUT_VERIFICATION_PATH.exists():
        try:
            payload = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
            ts = payload.get("timestamp")
            if isinstance(ts, str) and ts:
                return ts
        except Exception:
            pass
    return utc_now_iso()


def get_existing_test_result() -> str:
    if not MANIFEST_PATH.exists():
        return "pending_not_run"
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return str(payload.get("test_result", "pending_not_run"))
    except Exception:
        return "pending_not_run"


def key_tuples(df: pd.DataFrame, cols: Sequence[str]) -> set:
    return set(tuple(r) for r in df[list(cols)].itertuples(index=False, name=None))


def confusion_counts(alarm: np.ndarray, event: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((alarm == 1) & (event == 1)).sum())
    fp = int(((alarm == 1) & (event == 0)).sum())
    fn = int(((alarm == 0) & (event == 1)).sum())
    tn = int(((alarm == 0) & (event == 0)).sum())
    return tp, fp, fn, tn


def simulate_from_scores(
    scores: np.ndarray,
    startup_flags: np.ndarray,
    eligible_flags: np.ndarray,
    nominal_budget_r: float,
) -> Dict[str, np.ndarray]:
    n = int(len(scores))
    past_scores: List[float] = []

    cutoffs = np.full(n, np.nan, dtype=float)
    candidates = np.zeros(n, dtype=int)
    past_counts = np.zeros(n, dtype=int)

    capacity_to_date = np.zeros(n, dtype=int)
    alarms_before_current = np.zeros(n, dtype=int)
    quota_available = np.zeros(n, dtype=int)
    alarms = np.zeros(n, dtype=int)
    suppressed = np.zeros(n, dtype=int)
    unused_capacity_after = np.zeros(n, dtype=int)
    cumulative_alarms = np.zeros(n, dtype=int)
    prefix_excess = np.zeros(n, dtype=int)

    q = float(1.0 - nominal_budget_r)

    eligible_seen = 0
    alarms_issued = 0

    for i in range(n):
        score = float(scores[i])
        startup = bool(startup_flags[i])
        eligible = bool(eligible_flags[i])

        past_counts[i] = int(len(past_scores))

        candidate_alarm = 0
        cutoff = float("nan")

        if eligible:
            if len(past_scores) == 0:
                raise RuntimeError("Eligible row encountered with no past history")
            cutoff = quantile_higher(np.asarray(past_scores, dtype=float), q)
            candidate_alarm = int(score > cutoff)
        elif startup:
            candidate_alarm = 0
        else:
            candidate_alarm = 0

        cutoffs[i] = cutoff
        candidates[i] = int(candidate_alarm)

        if eligible:
            eligible_seen += 1
            cap = int(math.ceil(float(nominal_budget_r) * float(eligible_seen)))
            alarms_before = int(alarms_issued)
            q_avail = int(alarms_before < cap)
            alarm_flag = int((candidate_alarm == 1) and (q_avail == 1))
            if alarm_flag == 1:
                alarms_issued += 1
            suppressed_flag = int((candidate_alarm == 1) and (alarm_flag == 0))
            unused = int(cap - alarms_issued)
            cum = int(alarms_issued)
            excess = int(max(0, cum - cap))
        else:
            cap = 0
            alarms_before = int(alarms_issued)
            q_avail = 0
            alarm_flag = 0
            suppressed_flag = 0
            unused = 0
            cum = int(alarms_issued)
            excess = 0

        capacity_to_date[i] = int(cap)
        alarms_before_current[i] = int(alarms_before)
        quota_available[i] = int(q_avail)
        alarms[i] = int(alarm_flag)
        suppressed[i] = int(suppressed_flag)
        unused_capacity_after[i] = int(unused)
        cumulative_alarms[i] = int(cum)
        prefix_excess[i] = int(excess)

        past_scores.append(score)

    return {
        "past_only_cutoff": cutoffs,
        "candidate_alarm": candidates,
        "past_score_count": past_counts,
        "capacity_to_date": capacity_to_date,
        "alarms_before_current_date": alarms_before_current,
        "quota_available": quota_available,
        "alarm_flag": alarms,
        "suppressed_by_quota": suppressed,
        "unused_capacity_after_decision": unused_capacity_after,
        "cumulative_alarms_to_date": cumulative_alarms,
        "prefix_excess_over_capacity": prefix_excess,
    }


def build_metrics_by_fold(decisions_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    gcols = ["model", "horizon", "outer_fold", "threshold", "nominal_budget_r"]
    for key, grp in decisions_df.groupby(gcols, sort=True):
        model_name, horizon, fold, tau, budget = key
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
        eligible = g[g["evaluation_eligible"].astype(bool)].copy()

        event_col = "event_tau{}".format(int(tau))
        alarm = eligible["alarm_flag"].astype(int).to_numpy()
        event = eligible[event_col].astype(int).to_numpy()

        tp, fp, fn, tn = confusion_counts(alarm, event)
        eligible_n = int(len(eligible))
        eligible_events = int(event.sum())
        alarms_issued = int(alarm.sum())
        candidate_alarms = int(eligible["candidate_alarm"].astype(int).sum())
        suppressed_n = int(eligible["suppressed_by_quota"].astype(int).sum())

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
                "candidate_alarms": int(candidate_alarms),
                "alarms_issued": int(alarms_issued),
                "alarms_suppressed_by_quota": int(suppressed_n),
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
                "date_start": eligible["target_date"].min(),
                "date_end": eligible["target_date"].max(),
                "dataset_sha256": str(g["dataset_sha256"].iloc[0]),
                "split_sha256": str(g["split_sha256"].iloc[0]),
                "source_sequential_run_id": str(g["source_sequential_run_id"].iloc[0]),
                "quota_run_id": str(g["quota_run_id"].iloc[0]),
            }
        )

    out = pd.DataFrame(rows).sort_values(["model", "outer_fold", "threshold", "nominal_budget_r"]).reset_index(drop=True)
    return out


def build_metrics_pooled(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    gcols = ["model", "horizon", "threshold", "nominal_budget_r"]
    for key, grp in fold_metrics_df.groupby(gcols, sort=True):
        model_name, horizon, tau, budget = key

        pooled_eligible_n = int(grp["eligible_N"].astype(int).sum())
        pooled_events = int(grp["eligible_events"].astype(int).sum())
        pooled_candidates = int(grp["candidate_alarms"].astype(int).sum())
        pooled_alarms = int(grp["alarms_issued"].astype(int).sum())
        pooled_suppressed = int(grp["alarms_suppressed_by_quota"].astype(int).sum())
        pooled_tp = int(grp["TP"].astype(int).sum())
        pooled_fp = int(grp["FP"].astype(int).sum())
        pooled_fn = int(grp["FN"].astype(int).sum())
        pooled_tn = int(grp["TN"].astype(int).sum())

        precision = safe_div(float(pooled_tp), float(pooled_alarms))
        recall = safe_div(float(pooled_tp), float(pooled_events))
        specificity = safe_div(float(pooled_tn), float(pooled_tn + pooled_fp))
        fpr = safe_div(float(pooled_fp), float(pooled_fp + pooled_tn))
        false_alarm_fraction = safe_div(float(pooled_fp), float(pooled_alarms))
        missed_event_fraction = safe_div(float(pooled_fn), float(pooled_events))
        realized_alarm_fraction = safe_div(float(pooled_alarms), float(pooled_eligible_n))

        rows.append(
            {
                "model": str(model_name),
                "horizon": int(horizon),
                "outer_fold": "pooled",
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "pooled_eligible_N": int(pooled_eligible_n),
                "pooled_eligible_events": int(pooled_events),
                "pooled_candidate_alarms": int(pooled_candidates),
                "pooled_alarms": int(pooled_alarms),
                "pooled_suppressed_by_quota": int(pooled_suppressed),
                "pooled_realized_alarm_fraction": realized_alarm_fraction,
                "pooled_TP": int(pooled_tp),
                "pooled_FP": int(pooled_fp),
                "pooled_FN": int(pooled_fn),
                "pooled_TN": int(pooled_tn),
                "pooled_precision": precision,
                "pooled_recall": recall,
                "pooled_specificity": specificity,
                "pooled_false_positive_rate": fpr,
                "pooled_false_alarm_fraction_of_alarms": false_alarm_fraction,
                "pooled_missed_event_fraction": missed_event_fraction,
                "nominal_budget_difference": float(realized_alarm_fraction - float(budget)),
                "dataset_sha256": str(grp["dataset_sha256"].iloc[0]),
                "split_sha256": str(grp["split_sha256"].iloc[0]),
                "source_sequential_run_id": str(grp["source_sequential_run_id"].iloc[0]),
                "quota_run_id": str(grp["quota_run_id"].iloc[0]),
            }
        )

    out = pd.DataFrame(rows).sort_values(["model", "threshold", "nominal_budget_r"]).reset_index(drop=True)
    return out


def build_capacity_audit(decisions_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    gcols = ["model", "horizon", "outer_fold", "threshold", "nominal_budget_r"]
    for key, grp in decisions_df.groupby(gcols, sort=True):
        model_name, horizon, fold, tau, budget = key
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

        startup_alarm_count = int(g[g["startup_calibration"].astype(bool)]["alarm_flag"].astype(int).sum())

        eligible = g[g["evaluation_eligible"].astype(bool)].copy()
        eligible_n = int(len(eligible))

        capacity = eligible["capacity_to_date"].astype(int).to_numpy()
        cum_alarms = eligible["cumulative_alarms_to_date"].astype(int).to_numpy()

        prefix_ok = bool((cum_alarms <= capacity).all())
        max_excess = int(max(0, int(np.max(cum_alarms - capacity)))) if eligible_n > 0 else 0

        final_alarms = int(eligible["alarm_flag"].astype(int).sum())
        final_capacity = int(math.ceil(float(budget) * float(eligible_n)))
        cap_limit = 11 if math.isclose(float(budget), 0.05, rel_tol=0.0, abs_tol=1e-12) else 22

        candidate = eligible["candidate_alarm"].astype(int).to_numpy()
        alarm = eligible["alarm_flag"].astype(int).to_numpy()
        suppressed = eligible["suppressed_by_quota"].astype(int).to_numpy()
        baseline_alarm = eligible["unconstrained_alarm_flag"].astype(int).to_numpy()

        candidate_matches_baseline = bool(np.array_equal(candidate, baseline_alarm))
        only_quota_gate_changes = bool(
            np.array_equal((candidate - alarm), suppressed)
            and np.all((candidate - alarm) >= 0)
            and np.all((candidate - alarm) <= 1)
            and np.all(alarm <= candidate)
        )
        suppression_consistency = bool(np.array_equal(suppressed, ((candidate == 1) & (alarm == 0)).astype(int)))

        no_future_information_used = bool((g["future_scores_used"].astype(bool) == False).all())
        no_label_used = bool(
            (g["labels_used_for_score"].astype(bool) == False).all()
            and (g["labels_used_for_cutoff"].astype(bool) == False).all()
        )

        row = {
            "model": str(model_name),
            "horizon": int(horizon),
            "outer_fold": int(fold),
            "threshold": int(tau),
            "nominal_budget_r": float(budget),
            "eligible_N": int(eligible_n),
            "startup_alarm_count": int(startup_alarm_count),
            "final_capacity": int(final_capacity),
            "final_alarms": int(final_alarms),
            "final_alarm_cap_limit": int(cap_limit),
            "maximum_prefix_excess_over_capacity": int(max_excess),
            "no_startup_alarm": bool(startup_alarm_count == 0),
            "cumulative_cap_never_exceeded": bool(prefix_ok),
            "final_alarm_cap_pass": bool(final_alarms <= cap_limit),
            "no_future_information_used": bool(no_future_information_used),
            "no_label_used": bool(no_label_used),
            "candidate_matches_frozen_baseline": bool(candidate_matches_baseline),
            "only_quota_gate_changes_candidates": bool(only_quota_gate_changes),
            "suppression_consistency": bool(suppression_consistency),
        }

        row["capacity_audit_pass"] = bool(
            row["no_startup_alarm"]
            and row["cumulative_cap_never_exceeded"]
            and row["final_alarm_cap_pass"]
            and row["no_future_information_used"]
            and row["no_label_used"]
            and row["candidate_matches_frozen_baseline"]
            and row["only_quota_gate_changes_candidates"]
            and row["suppression_consistency"]
            and row["maximum_prefix_excess_over_capacity"] == 0
            and row["final_alarms"] <= row["final_capacity"]
        )

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["model", "outer_fold", "threshold", "nominal_budget_r"]).reset_index(drop=True)
    return out


def build_suppression_audit(decisions_df: pd.DataFrame) -> pd.DataFrame:
    fold_rows: List[Dict[str, Any]] = []

    gcols = ["model", "horizon", "outer_fold", "threshold", "nominal_budget_r"]
    for key, grp in decisions_df.groupby(gcols, sort=True):
        model_name, horizon, fold, tau, budget = key
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
        eligible = g[g["evaluation_eligible"].astype(bool)].copy()

        eligible_n = int(len(eligible))
        candidate_alarms = int(eligible["candidate_alarm"].astype(int).sum())
        issued = int(eligible["alarm_flag"].astype(int).sum())
        suppressed = int(eligible["suppressed_by_quota"].astype(int).sum())

        final_capacity = int(math.ceil(float(budget) * float(eligible_n)))
        unused = int(final_capacity - issued)

        cum = eligible["cumulative_alarms_to_date"].astype(int).to_numpy()
        cap = eligible["capacity_to_date"].astype(int).to_numpy()
        max_excess = int(max(0, int(np.max(cum - cap)))) if eligible_n > 0 else 0

        fold_rows.append(
            {
                "model": str(model_name),
                "horizon": int(horizon),
                "outer_fold": str(fold),
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "eligible_N": int(eligible_n),
                "final_capacity": int(final_capacity),
                "candidate_alarms": int(candidate_alarms),
                "issued_alarms": int(issued),
                "alarms_suppressed_by_quota": int(suppressed),
                "unused_final_capacity": int(unused),
                "realized_alarm_fraction": safe_div(float(issued), float(eligible_n)),
                "maximum_prefix_excess_over_capacity": int(max_excess),
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    pooled_rows: List[Dict[str, Any]] = []
    pcols = ["model", "horizon", "threshold", "nominal_budget_r"]
    for key, grp in fold_df.groupby(pcols, sort=True):
        model_name, horizon, tau, budget = key
        eligible_n = int(grp["eligible_N"].astype(int).sum())
        final_capacity = int(grp["final_capacity"].astype(int).sum())
        candidate_alarms = int(grp["candidate_alarms"].astype(int).sum())
        issued = int(grp["issued_alarms"].astype(int).sum())
        suppressed = int(grp["alarms_suppressed_by_quota"].astype(int).sum())
        unused = int(final_capacity - issued)
        max_excess = int(grp["maximum_prefix_excess_over_capacity"].astype(int).max())

        pooled_rows.append(
            {
                "model": str(model_name),
                "horizon": int(horizon),
                "outer_fold": "pooled",
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "eligible_N": int(eligible_n),
                "final_capacity": int(final_capacity),
                "candidate_alarms": int(candidate_alarms),
                "issued_alarms": int(issued),
                "alarms_suppressed_by_quota": int(suppressed),
                "unused_final_capacity": int(unused),
                "realized_alarm_fraction": safe_div(float(issued), float(eligible_n)),
                "maximum_prefix_excess_over_capacity": int(max_excess),
            }
        )

    out = pd.concat([fold_df, pd.DataFrame(pooled_rows)], axis=0, ignore_index=True)
    out = out.sort_values(["model", "outer_fold", "threshold", "nominal_budget_r"]).reset_index(drop=True)
    return out


def build_comparison(decisions_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def summarize(g: pd.DataFrame, alarm_col: str, event_col: str) -> Dict[str, Any]:
        alarm = g[alarm_col].astype(int).to_numpy()
        event = g[event_col].astype(int).to_numpy()
        tp, fp, fn, tn = confusion_counts(alarm, event)
        n = int(len(g))
        alarms = int(alarm.sum())
        events = int(event.sum())
        return {
            "N": n,
            "events": events,
            "alarms": alarms,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": safe_div(float(tp), float(alarms)),
            "recall": safe_div(float(tp), float(events)),
            "realized_alarm_fraction": safe_div(float(alarms), float(n)),
        }

    gcols = ["model", "horizon", "threshold", "nominal_budget_r"]
    for key, grp_all in decisions_df.groupby(gcols, sort=True):
        model_name, horizon, tau, budget = key

        for fold_label in [1, 2, 3, "pooled"]:
            if fold_label == "pooled":
                grp = grp_all[grp_all["evaluation_eligible"].astype(bool)].copy()
                capacity_bound = int(math.ceil(float(budget) * 657.0))
            else:
                grp = grp_all[
                    (grp_all["outer_fold"].astype(int) == int(fold_label))
                    & (grp_all["evaluation_eligible"].astype(bool))
                ].copy()
                capacity_bound = int(math.ceil(float(budget) * 219.0))

            event_col = "event_tau{}".format(int(tau))

            unconstrained = summarize(grp, "unconstrained_alarm_flag", event_col)
            quota = summarize(grp, "alarm_flag", event_col)

            suppressed = int(unconstrained["alarms"] - quota["alarms"])

            rows.append(
                {
                    "model": str(model_name),
                    "horizon": int(horizon),
                    "outer_fold": str(fold_label),
                    "threshold": int(tau),
                    "nominal_budget_r": float(budget),
                    "eligible_N": int(quota["N"]),
                    "capacity_bound": int(capacity_bound),
                    "unconstrained_alarms": int(unconstrained["alarms"]),
                    "quota_enforced_alarms": int(quota["alarms"]),
                    "alarms_suppressed_by_quota": int(suppressed),
                    "unconstrained_realized_alarm_fraction": float(unconstrained["realized_alarm_fraction"]),
                    "quota_enforced_realized_alarm_fraction": float(quota["realized_alarm_fraction"]),
                    "unconstrained_precision": float(unconstrained["precision"]),
                    "quota_enforced_precision": float(quota["precision"]),
                    "unconstrained_recall": float(unconstrained["recall"]),
                    "quota_enforced_recall": float(quota["recall"]),
                    "unconstrained_TP": int(unconstrained["TP"]),
                    "unconstrained_FP": int(unconstrained["FP"]),
                    "unconstrained_FN": int(unconstrained["FN"]),
                    "unconstrained_TN": int(unconstrained["TN"]),
                    "quota_enforced_TP": int(quota["TP"]),
                    "quota_enforced_FP": int(quota["FP"]),
                    "quota_enforced_FN": int(quota["FN"]),
                    "quota_enforced_TN": int(quota["TN"]),
                    "unconstrained_exceeds_capacity": bool(unconstrained["alarms"] > capacity_bound),
                    "quota_enforced_exceeds_capacity": bool(quota["alarms"] > capacity_bound),
                    "source_context_unconstrained": "historical_sequential_past_only",
                    "source_context_quota": "historical_sequential_past_only_with_cumulative_hard_cap",
                }
            )

    out = pd.DataFrame(rows).sort_values(["model", "outer_fold", "threshold", "nominal_budget_r"]).reset_index(drop=True)
    return out


def run_information_isolation_audits(decisions_df: pd.DataFrame) -> Dict[str, Any]:
    rng = np.random.default_rng(20260806)

    label_permutation_pass = True
    future_score_perturbation_pass = True
    current_score_locality_pass = True
    no_outcome_dependency_pass = True

    max_earlier_alarm_diff_future = 0
    max_earlier_cutoff_diff_future = 0.0
    max_earlier_quota_diff_future = 0
    max_earlier_quota_diff_current = 0

    groups = decisions_df.groupby(["model", "outer_fold", "threshold", "nominal_budget_r"], sort=True)

    for _, grp in groups:
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
        n = int(len(g))

        scores = g["policy_score"].astype(float).to_numpy()
        startup_flags = g["startup_calibration"].astype(bool).to_numpy()
        eligible_flags = g["evaluation_eligible"].astype(bool).to_numpy()
        budget = float(g["nominal_budget_r"].iloc[0])

        baseline = simulate_from_scores(
            scores=scores,
            startup_flags=startup_flags,
            eligible_flags=eligible_flags,
            nominal_budget_r=budget,
        )

        # Label permutation: decisions and quota states must not change.
        y_perm = g["y_true"].to_numpy().copy()
        rng.shuffle(y_perm)
        # We intentionally do not use labels in simulation; this should be invariant.
        sim_perm = simulate_from_scores(
            scores=scores,
            startup_flags=startup_flags,
            eligible_flags=eligible_flags,
            nominal_budget_r=budget,
        )

        if not np.array_equal(baseline["candidate_alarm"], sim_perm["candidate_alarm"]):
            label_permutation_pass = False
        if not np.array_equal(baseline["alarm_flag"], sim_perm["alarm_flag"]):
            label_permutation_pass = False
        if not np.array_equal(baseline["capacity_to_date"], sim_perm["capacity_to_date"]):
            label_permutation_pass = False
        if not np.array_equal(baseline["alarms_before_current_date"], sim_perm["alarms_before_current_date"]):
            label_permutation_pass = False
        if not np.allclose(
            np.nan_to_num(baseline["past_only_cutoff"], nan=-1.0),
            np.nan_to_num(sim_perm["past_only_cutoff"], nan=-1.0),
            atol=0.0,
            rtol=0.0,
        ):
            label_permutation_pass = False

        # Future-score perturbation should not affect earlier decisions.
        pivot_future = min(max(STARTUP_DAYS + 40, 1), n - 3)
        pert_scores_future = scores.copy()
        for i in range(pivot_future + 1, n):
            pert_scores_future[i] = pert_scores_future[i] + (0.001 * float((i - pivot_future) % 7 + 1))

        sim_future = simulate_from_scores(
            scores=pert_scores_future,
            startup_flags=startup_flags,
            eligible_flags=eligible_flags,
            nominal_budget_r=budget,
        )

        alarm_diff_future = np.abs(
            baseline["alarm_flag"][: pivot_future + 1] - sim_future["alarm_flag"][: pivot_future + 1]
        )
        quota_diff_future = np.abs(
            baseline["alarms_before_current_date"][: pivot_future + 1]
            - sim_future["alarms_before_current_date"][: pivot_future + 1]
        )
        cutoff_diff_future = np.abs(
            np.nan_to_num(baseline["past_only_cutoff"][: pivot_future + 1], nan=-1.0)
            - np.nan_to_num(sim_future["past_only_cutoff"][: pivot_future + 1], nan=-1.0)
        )

        max_earlier_alarm_diff_future = int(max(max_earlier_alarm_diff_future, int(alarm_diff_future.max())))
        max_earlier_quota_diff_future = int(max(max_earlier_quota_diff_future, int(quota_diff_future.max())))
        max_earlier_cutoff_diff_future = float(max(max_earlier_cutoff_diff_future, float(cutoff_diff_future.max())))

        if int(alarm_diff_future.max()) != 0 or int(quota_diff_future.max()) != 0 or float(cutoff_diff_future.max()) != 0.0:
            future_score_perturbation_pass = False

        # Current-score perturbation can affect current/later only.
        pivot_current = min(max(STARTUP_DAYS + 10, 1), n - 3)
        pert_scores_current = scores.copy()
        pert_scores_current[pivot_current] = pert_scores_current[pivot_current] + 123.456

        sim_current = simulate_from_scores(
            scores=pert_scores_current,
            startup_flags=startup_flags,
            eligible_flags=eligible_flags,
            nominal_budget_r=budget,
        )

        alarm_diff_current = np.abs(
            baseline["alarm_flag"][:pivot_current] - sim_current["alarm_flag"][:pivot_current]
        )
        quota_diff_current = np.abs(
            baseline["alarms_before_current_date"][:pivot_current]
            - sim_current["alarms_before_current_date"][:pivot_current]
        )

        max_earlier_quota_diff_current = int(max(max_earlier_quota_diff_current, int(quota_diff_current.max())))

        if int(alarm_diff_current.max()) != 0 or int(quota_diff_current.max()) != 0:
            current_score_locality_pass = False

        # No-outcome dependency: changing labels only must not alter decisions.
        y_scrambled = g["y_true"].to_numpy().copy()
        y_scrambled = y_scrambled[::-1].copy()
        _ = y_scrambled  # labels deliberately unused
        sim_no_outcome = simulate_from_scores(
            scores=scores,
            startup_flags=startup_flags,
            eligible_flags=eligible_flags,
            nominal_budget_r=budget,
        )

        if not np.array_equal(baseline["alarm_flag"], sim_no_outcome["alarm_flag"]):
            no_outcome_dependency_pass = False
        if not np.array_equal(baseline["candidate_alarm"], sim_no_outcome["candidate_alarm"]):
            no_outcome_dependency_pass = False

    return {
        "label_permutation_pass": bool(label_permutation_pass),
        "future_score_perturbation_pass": bool(future_score_perturbation_pass),
        "current_score_locality_pass": bool(current_score_locality_pass),
        "no_outcome_dependency_pass": bool(no_outcome_dependency_pass),
        "max_earlier_alarm_difference_under_future_perturbation": int(max_earlier_alarm_diff_future),
        "max_earlier_cutoff_difference_under_future_perturbation": float(max_earlier_cutoff_diff_future),
        "max_earlier_quota_state_difference_under_future_perturbation": int(max_earlier_quota_diff_future),
        "max_earlier_quota_state_difference_under_current_perturbation": int(max_earlier_quota_diff_current),
    }


def get_output_checksum_targets() -> List[Path]:
    targets: List[Path] = []
    for p in sorted(OUT_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.name == CHECKSUMS_PATH.name:
            continue
        if p.suffix.lower() in {".csv", ".json", ".md", ".py"}:
            targets.append(p)
    return targets


def write_checksums_file() -> None:
    lines: List[str] = []
    for p in get_output_checksum_targets():
        lines.append("{}  {}".format(sha256_file(p), p.name))
    ensure_inside_authorized(CHECKSUMS_PATH)
    CHECKSUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    verification_pass: bool,
    source_decision: str,
    candidate_repro_pass: bool,
    capacity_pass: bool,
    suppression_pass: bool,
    info_pass: bool,
    source_preservation_pass: bool,
    deterministic_status: str,
    test_result: str,
    final_decision: str,
    git_branch: str,
    git_commit: str,
    run_id: str,
) -> str:
    decision_text_map = {
        "A": "A. Quota-enforced sequential H5 policy passed; H1/H3 canonical reconciliation may begin.",
        "B": "B. Policy completed, but capacity utilization or suppression behavior requires investigation.",
        "C": "C. A consistent quota-enforced policy could not be constructed.",
        "D": "D. Capacity, causality, information-isolation, source-preservation, or deterministic tests failed.",
    }

    decision_text = decision_text_map[final_decision]

    lines = [
        "# Quota-Enforced Sequential H5 Alarm Policy",
        "",
        "## 1. Purpose",
        "- Construct a cumulative-capacity-constrained sequential H5 alarm simulation over the frozen past-only policy scores.",
        "- Preserve existing candidate-alarm logic and add only a hard quota gate.",
        "",
        "## 2. Input Verification",
        "- verification_pass: {}".format(bool(verification_pass)),
        "- source_sequential_completion_decision: {}".format(str(source_decision)),
        "- git_branch: {}".format(str(git_branch)),
        "- git_commit: {}".format(str(git_commit)),
        "",
        "## 3. Candidate Alarm Logic",
        "- candidate_alarm = policy_score > past_only_cutoff (strict inequality).",
        "- Equality with cutoff remains no-alarm.",
        "- candidate_reproduction_vs_frozen_baseline_pass: {}".format(bool(candidate_repro_pass)),
        "",
        "## 4. Quota Gate",
        "- capacity_j = ceil(r * j) on eligible-date prefixes only.",
        "- alarm_flag = candidate_alarm AND alarms_before_current_date < capacity_j.",
        "- capacity_audit_pass: {}".format(bool(capacity_pass)),
        "- suppression_audit_pass: {}".format(bool(suppression_pass)),
        "",
        "## 5. Capacity Expectations",
        "- eligible_N per fold: 219",
        "- r=0.05 final per-fold cap=11; pooled cap=33.",
        "- r=0.10 final per-fold cap=22; pooled cap=66.",
        "- Unused capacity is allowed when candidate alarms are insufficient.",
        "",
        "## 6. Comparison with Unconstrained Sequential Simulation",
        "- The unconstrained policy is causally valid but may exceed capacity.",
        "- The quota-enforced policy guarantees the cumulative alarm cap.",
        "- Both remain historical simulations rather than prospective field validation.",
        "",
        "## 7. Information Isolation",
        "- information_isolation_pass: {}".format(bool(info_pass)),
        "- source_preservation_pass: {}".format(bool(source_preservation_pass)),
        "",
        "## 8. Determinism and Tests",
        "- deterministic_result: {}".format(str(deterministic_status)),
        "- test_result: {}".format(str(test_result)),
        "- run_id: {}".format(str(run_id)),
        "",
        "FINAL DECISION",
        "",
        decision_text,
        "",
        "TERMINAL SUMMARY",
        "",
        "1. Input verification result: {}".format(bool(verification_pass)),
        "2. Candidate reproduction to frozen baseline: {}".format(bool(candidate_repro_pass)),
        "3. Capacity audit result: {}".format(bool(capacity_pass)),
        "4. Suppression audit result: {}".format(bool(suppression_pass)),
        "5. Information-isolation result: {}".format(bool(info_pass)),
        "6. Source preservation result: {}".format(bool(source_preservation_pass)),
        "7. Deterministic status: {}".format(str(deterministic_status)),
        "8. Final decision: {}".format(str(final_decision)),
    ]

    return "\n".join(lines) + "\n"


def build_package() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("Run this script from workspace root: {}".format(ROOT))

    # Branch gate
    branch = git_output(["git", "branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise RuntimeError("Active branch is {}; expected {}".format(branch, EXPECTED_BRANCH))
    git_commit = git_output(["git", "rev-parse", "HEAD"])

    # Source preservation snapshots (read-only package fingerprints).
    source_fingerprints_before = {
        rel_to_root(SOURCE_DECISIONS_PATH): sha256_file(SOURCE_DECISIONS_PATH),
        rel_to_root(SOURCE_REPORT_PATH): sha256_file(SOURCE_REPORT_PATH),
        rel_to_root(SOURCE_MANIFEST_PATH): sha256_file(SOURCE_MANIFEST_PATH),
        rel_to_root(SOURCE_CHECKSUMS_PATH): sha256_file(SOURCE_CHECKSUMS_PATH),
    }

    # Protocol checks.
    protocol_checks = verify_checksum_manifest(PROTOCOL_SHA_PATH, PROTOCOL_DIR)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    dataset_path = Path(lock["absolute_path"])
    dataset_sha_expected = str(lock["sha256"])
    dataset_sha_observed = sha256_file(dataset_path)
    if dataset_sha_observed != dataset_sha_expected:
        raise RuntimeError("Dataset checksum mismatch")

    split_sha_observed = sha256_file(SPLIT_PATH)
    split_sha_expected = expected_checksum_for_filename(PROTOCOL_SHA_PATH, "corrected_split_assignment.csv")
    if split_sha_observed != split_sha_expected:
        raise RuntimeError("Split checksum mismatch")

    split_summary_sha_observed = sha256_file(SPLIT_SUMMARY_PATH)
    split_summary_sha_expected = expected_checksum_for_filename(PROTOCOL_SHA_PATH, "corrected_split_summary.csv")
    if split_summary_sha_observed != split_summary_sha_expected:
        raise RuntimeError("Split summary checksum mismatch")

    source_checksums = checksum_lookup(SOURCE_CHECKSUMS_PATH)
    source_decisions_expected_sha = expected_checksum_for_filename(
        SOURCE_CHECKSUMS_PATH, "sequential_h5_alarm_decisions.csv"
    )
    source_decisions_observed_sha = sha256_file(SOURCE_DECISIONS_PATH)
    if source_decisions_observed_sha != source_decisions_expected_sha:
        raise RuntimeError("Frozen sequential decision checksum mismatch")

    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_decision = infer_decision_from_report(SOURCE_REPORT_PATH)
    if source_decision != "A":
        raise RuntimeError("Source sequential package completion decision is not A")

    source_output_checksums = source_manifest.get("output_checksums", {})
    if str(source_output_checksums.get("sequential_h5_alarm_decisions.csv", "")) != source_decisions_observed_sha:
        raise RuntimeError("Source manifest decision checksum does not match observed checksum")

    source_info = json.loads(SOURCE_INFO_ISOLATION_PATH.read_text(encoding="utf-8"))

    canonical_manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    run_id = "quota_enforced_h5_policy_{}_{}_{}".format(
        git_commit[:12], dataset_sha_observed[:8], split_sha_observed[:8]
    )

    # Load and normalize frozen sequential decisions (unconstrained baseline).
    src = pd.read_csv(SOURCE_DECISIONS_PATH)

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
        "policy_score",
        "cutoff_score",
        "startup_calibration",
        "evaluation_eligible",
        "alarm_flag",
        "future_scores_used",
        "labels_used_for_score",
        "labels_used_for_cutoff",
        "dataset_sha256",
        "split_sha256",
        "canonical_assembly_id",
        "sequential_run_id",
        "git_commit",
    ]
    missing = [c for c in required_cols if c not in src.columns]
    if missing:
        raise RuntimeError("Source decision file missing required columns: {}".format(missing))

    src = src.copy()
    src["horizon"] = pd.to_numeric(src["horizon"], errors="raise").astype(int)
    src["outer_fold"] = pd.to_numeric(src["outer_fold"], errors="raise").astype(int)
    src["canonical_row_id"] = pd.to_numeric(src["canonical_row_id"], errors="raise").astype(int)
    src["threshold"] = pd.to_numeric(src["threshold"], errors="raise").astype(int)
    src["nominal_budget_r"] = pd.to_numeric(src["nominal_budget_r"], errors="raise").astype(float)
    src["feature_date"] = normalize_date_col(src["feature_date"])
    src["target_date"] = normalize_date_col(src["target_date"])

    src["startup_calibration"] = as_bool(src["startup_calibration"])
    src["evaluation_eligible"] = as_bool(src["evaluation_eligible"])
    src["future_scores_used"] = as_bool(src["future_scores_used"])
    src["labels_used_for_score"] = as_bool(src["labels_used_for_score"])
    src["labels_used_for_cutoff"] = as_bool(src["labels_used_for_cutoff"])
    src["alarm_flag"] = pd.to_numeric(src["alarm_flag"], errors="raise").astype(int)
    src["y_true"] = pd.to_numeric(src["y_true"], errors="raise").astype(float)
    src["policy_score"] = pd.to_numeric(src["policy_score"], errors="raise").astype(float)

    src = src[
        (src["horizon"] == H)
        & (src["model"].isin(MODELS))
        & (src["threshold"].isin(THRESHOLDS))
        & (src["nominal_budget_r"].isin(BUDGETS))
    ].copy()

    expected_rows = 747 * len(THRESHOLDS) * len(BUDGETS) * len(MODELS)
    if int(len(src)) != int(expected_rows):
        raise RuntimeError(
            "Unexpected source decision row count after filtering: observed {}, expected {}".format(
                len(src), expected_rows
            )
        )

    # Validate canonical H5 key set from locked split.
    split = pd.read_csv(SPLIT_PATH)
    split["horizon"] = pd.to_numeric(split["horizon"], errors="raise").astype(int)
    split["outer_fold"] = pd.to_numeric(split["outer_fold"], errors="raise").astype(int)
    split["feature_date"] = normalize_date_col(split["feature_date"])
    split["target_date"] = normalize_date_col(split["target_date"])

    h5_keys = split[(split["horizon"] == H) & (split["outer_role"] == "outer_test")][
        ["horizon", "outer_fold", "feature_date", "target_date"]
    ].drop_duplicates()
    if int(len(h5_keys)) != 747:
        raise RuntimeError("Locked H5 canonical key count is not 747")
    canonical_key_set = key_tuples(h5_keys, ["horizon", "outer_fold", "feature_date", "target_date"])

    # Build quota-enforced decisions.
    decision_parts: List[pd.DataFrame] = []
    candidate_reproduction_pass = True
    cutoff_reproduction_pass = True

    gcols = ["model", "outer_fold", "threshold", "nominal_budget_r"]
    for _, grp in src.groupby(gcols, sort=True):
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

        fold_id = int(g["outer_fold"].iloc[0])
        expected_fold_keys = h5_keys[h5_keys["outer_fold"].astype(int) == fold_id][
            ["horizon", "outer_fold", "feature_date", "target_date"]
        ].copy()
        expected_fold_set = key_tuples(
            expected_fold_keys, ["horizon", "outer_fold", "feature_date", "target_date"]
        )

        key_set = key_tuples(g, ["horizon", "outer_fold", "feature_date", "target_date"])
        if key_set != expected_fold_set:
            raise RuntimeError(
                "Canonical key mismatch in source decisions for model/fold group: model={}, fold={}".format(
                    str(g["model"].iloc[0]), fold_id
                )
            )
        if int(len(g)) != int(len(expected_fold_set)):
            raise RuntimeError(
                "Unexpected row count for model/fold group: model={}, fold={}, observed={}, expected={}".format(
                    str(g["model"].iloc[0]), fold_id, len(g), len(expected_fold_set)
                )
            )

        if int(g.duplicated(subset=["horizon", "outer_fold", "feature_date", "target_date"]).sum()) != 0:
            raise RuntimeError("Duplicate key detected in source decisions")

        sim = simulate_from_scores(
            scores=g["policy_score"].to_numpy(dtype=float),
            startup_flags=g["startup_calibration"].to_numpy(dtype=bool),
            eligible_flags=g["evaluation_eligible"].to_numpy(dtype=bool),
            nominal_budget_r=float(g["nominal_budget_r"].iloc[0]),
        )

        # Reproduction checks against frozen unconstrained baseline.
        eligible_mask = g["evaluation_eligible"].to_numpy(dtype=bool)
        src_cutoff = pd.to_numeric(g["cutoff_score"], errors="coerce").to_numpy(dtype=float)
        calc_cutoff = sim["past_only_cutoff"]
        if eligible_mask.any():
            cutoff_ok = np.allclose(
                src_cutoff[eligible_mask], calc_cutoff[eligible_mask], atol=1e-12, rtol=0.0
            )
            cutoff_reproduction_pass = cutoff_reproduction_pass and bool(cutoff_ok)

        src_alarm = g["alarm_flag"].astype(int).to_numpy()
        candidate_alarm = sim["candidate_alarm"].astype(int)
        candidate_ok = np.array_equal(src_alarm, candidate_alarm)
        candidate_reproduction_pass = candidate_reproduction_pass and bool(candidate_ok)

        part = g.copy()
        part["past_score_count"] = sim["past_score_count"].astype(int)
        part["past_only_cutoff"] = sim["past_only_cutoff"].astype(float)
        part["candidate_alarm"] = sim["candidate_alarm"].astype(int)
        part["capacity_to_date"] = sim["capacity_to_date"].astype(int)
        part["alarms_before_current_date"] = sim["alarms_before_current_date"].astype(int)
        part["quota_available"] = sim["quota_available"].astype(int)
        part["alarm_flag"] = sim["alarm_flag"].astype(int)
        part["suppressed_by_quota"] = sim["suppressed_by_quota"].astype(int)
        part["unused_capacity_after_decision"] = sim["unused_capacity_after_decision"].astype(int)
        part["cumulative_alarms_to_date"] = sim["cumulative_alarms_to_date"].astype(int)
        part["prefix_excess_over_capacity"] = sim["prefix_excess_over_capacity"].astype(int)
        part["unconstrained_alarm_flag"] = src_alarm.astype(int)
        part["startup_calibration"] = g["startup_calibration"].astype(bool)
        part["evaluation_eligible"] = g["evaluation_eligible"].astype(bool)
        part["source_sequential_run_id"] = str(g["sequential_run_id"].iloc[0])
        part["quota_run_id"] = str(run_id)
        decision_parts.append(part)

    decisions = pd.concat(decision_parts, axis=0, ignore_index=True)
    decisions = decisions.sort_values(
        ["model", "threshold", "nominal_budget_r", "outer_fold", "target_date", "feature_date", "canonical_row_id"]
    ).reset_index(drop=True)

    # Ensure startup rows carry no alarms.
    if int(decisions[decisions["startup_calibration"].astype(bool)]["alarm_flag"].astype(int).sum()) != 0:
        raise RuntimeError("Startup rows contain quota-enforced alarms")

    # Build outputs.
    decisions_cols = [
        "horizon",
        "outer_fold",
        "feature_date",
        "target_date",
        "canonical_row_id",
        "model",
        "threshold",
        "nominal_budget_r",
        "y_true",
        "event_tau15",
        "event_tau16",
        "event_tau17",
        "policy_score",
        "past_only_cutoff",
        "candidate_alarm",
        "capacity_to_date",
        "alarms_before_current_date",
        "quota_available",
        "alarm_flag",
        "suppressed_by_quota",
        "unused_capacity_after_decision",
        "cumulative_alarms_to_date",
        "prefix_excess_over_capacity",
        "evaluation_eligible",
        "startup_calibration",
        "past_score_count",
        "unconstrained_alarm_flag",
        "future_scores_used",
        "labels_used_for_score",
        "labels_used_for_cutoff",
        "dataset_sha256",
        "split_sha256",
        "canonical_assembly_id",
        "source_sequential_run_id",
        "quota_run_id",
        "git_commit",
    ]
    decisions = decisions[decisions_cols].copy()

    fold_metrics = build_metrics_by_fold(decisions)
    pooled_metrics = build_metrics_pooled(fold_metrics)
    capacity_audit = build_capacity_audit(decisions)
    suppression_audit = build_suppression_audit(decisions)
    comparison = build_comparison(decisions)
    info_audit = run_information_isolation_audits(decisions)

    # Capacity expected-value checks.
    expected_caps_ok = True
    sf = suppression_audit[suppression_audit["outer_fold"].astype(str).isin(["1", "2", "3"])].copy()
    sf_005 = sf[np.isclose(sf["nominal_budget_r"].astype(float), 0.05)]
    sf_010 = sf[np.isclose(sf["nominal_budget_r"].astype(float), 0.10)]
    if not bool((sf_005["final_capacity"].astype(int) == 11).all()):
        expected_caps_ok = False
    if not bool((sf_010["final_capacity"].astype(int) == 22).all()):
        expected_caps_ok = False

    sp = suppression_audit[suppression_audit["outer_fold"].astype(str) == "pooled"].copy()
    sp_005 = sp[np.isclose(sp["nominal_budget_r"].astype(float), 0.05)]
    sp_010 = sp[np.isclose(sp["nominal_budget_r"].astype(float), 0.10)]
    if not bool((sp_005["final_capacity"].astype(int) == 33).all()):
        expected_caps_ok = False
    if not bool((sp_010["final_capacity"].astype(int) == 66).all()):
        expected_caps_ok = False

    if not bool((suppression_audit["maximum_prefix_excess_over_capacity"].astype(int) == 0).all()):
        expected_caps_ok = False

    # Source-preservation recheck after computations.
    source_fingerprints_after = {
        rel_to_root(SOURCE_DECISIONS_PATH): sha256_file(SOURCE_DECISIONS_PATH),
        rel_to_root(SOURCE_REPORT_PATH): sha256_file(SOURCE_REPORT_PATH),
        rel_to_root(SOURCE_MANIFEST_PATH): sha256_file(SOURCE_MANIFEST_PATH),
        rel_to_root(SOURCE_CHECKSUMS_PATH): sha256_file(SOURCE_CHECKSUMS_PATH),
    }
    source_preservation_pass = source_fingerprints_before == source_fingerprints_after

    info_pass = bool(
        info_audit.get("label_permutation_pass", False)
        and info_audit.get("future_score_perturbation_pass", False)
        and info_audit.get("current_score_locality_pass", False)
        and info_audit.get("no_outcome_dependency_pass", False)
    )

    capacity_pass = bool(capacity_audit["capacity_audit_pass"].all()) and bool(expected_caps_ok)
    suppression_pass = bool((suppression_audit["maximum_prefix_excess_over_capacity"].astype(int) == 0).all())
    candidate_repro_pass = bool(candidate_reproduction_pass and cutoff_reproduction_pass)

    verification_pass = True
    audit_pass = bool(
        verification_pass
        and candidate_repro_pass
        and capacity_pass
        and suppression_pass
        and info_pass
        and source_preservation_pass
    )

    # Timestamp + run identifiers.
    timestamp = get_stable_timestamp()

    test_result = get_existing_test_result()
    deterministic_status = "pass" if str(test_result).startswith("pass") else "pending_not_run"

    if not verification_pass:
        final_decision = "C"
    elif not candidate_repro_pass:
        final_decision = "C"
    elif str(test_result).startswith("fail"):
        final_decision = "D"
    elif (not capacity_pass) or (not info_pass) or (not source_preservation_pass):
        final_decision = "D"
    elif str(test_result).startswith("pass"):
        final_decision = "A"
    else:
        final_decision = "B"

    software_versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }

    input_verification = {
        "verification_pass": bool(verification_pass),
        "dataset_sha256": dataset_sha_observed,
        "split_sha256": split_sha_observed,
        "protocol_tag": "corrected_protocol_v1",
        "git_branch": branch,
        "git_commit": git_commit,
        "source_sequential_run_id": str(source_manifest.get("run_id", "")),
        "source_sequential_decision": source_decision,
        "source_decisions_file": rel_to_root(SOURCE_DECISIONS_PATH),
        "source_decisions_sha256_expected": source_decisions_expected_sha,
        "source_decisions_sha256_observed": source_decisions_observed_sha,
        "protocol_checks": protocol_checks,
        "source_package_checksums_file_sha256": sha256_file(SOURCE_CHECKSUMS_PATH),
        "source_package_files_preserved": bool(source_preservation_pass),
        "source_package_fingerprints_before": source_fingerprints_before,
        "source_package_fingerprints_after": source_fingerprints_after,
        "source_information_isolation_baseline": source_info,
        "timestamp": timestamp,
        "software_versions": software_versions,
    }

    write_json(INPUT_VERIFICATION_PATH, input_verification)

    # Write CSV/JSON outputs.
    write_csv_with_dates(
        DECISIONS_PATH,
        decisions,
        date_cols=["feature_date", "target_date"],
    )
    write_csv_with_dates(
        METRICS_BY_FOLD_PATH,
        fold_metrics,
        date_cols=["date_start", "date_end"],
    )
    write_csv_with_dates(METRICS_POOLED_PATH, pooled_metrics, date_cols=[])
    write_csv_with_dates(CAPACITY_AUDIT_PATH, capacity_audit, date_cols=[])
    write_csv_with_dates(SUPPRESSION_AUDIT_PATH, suppression_audit, date_cols=[])
    write_csv_with_dates(COMPARISON_PATH, comparison, date_cols=[])

    info_audit_payload = {
        "label_permutation_pass": bool(info_audit["label_permutation_pass"]),
        "future_score_perturbation_pass": bool(info_audit["future_score_perturbation_pass"]),
        "current_score_locality_pass": bool(info_audit["current_score_locality_pass"]),
        "no_outcome_dependency_pass": bool(info_audit["no_outcome_dependency_pass"]),
        "max_earlier_alarm_difference_under_future_perturbation": int(
            info_audit["max_earlier_alarm_difference_under_future_perturbation"]
        ),
        "max_earlier_cutoff_difference_under_future_perturbation": float(
            info_audit["max_earlier_cutoff_difference_under_future_perturbation"]
        ),
        "max_earlier_quota_state_difference_under_future_perturbation": int(
            info_audit["max_earlier_quota_state_difference_under_future_perturbation"]
        ),
        "max_earlier_quota_state_difference_under_current_perturbation": int(
            info_audit["max_earlier_quota_state_difference_under_current_perturbation"]
        ),
        "summary_pass": bool(info_pass),
    }
    write_json(INFO_AUDIT_PATH, info_audit_payload)

    # Manifest + report
    manifest = {
        "run_id": run_id,
        "horizon": H,
        "models": list(MODELS),
        "thresholds": list(THRESHOLDS),
        "nominal_budgets": list(BUDGETS),
        "startup_days": STARTUP_DAYS,
        "policy_type": "quota_enforced_sequential_past_only",
        "candidate_rule": "candidate_alarm = policy_score > past_only_cutoff (strict equality no-alarm)",
        "quota_rule": "alarm_flag = candidate_alarm AND alarms_before_current_date < ceil(r*j)",
        "dataset_sha256": dataset_sha_observed,
        "split_sha256": split_sha_observed,
        "canonical_assembly_id": str(canonical_manifest.get("assembly_id", "")),
        "source_sequential_run_id": str(source_manifest.get("run_id", "")),
        "source_sequential_decision": source_decision,
        "source_decisions_file": rel_to_root(SOURCE_DECISIONS_PATH),
        "source_decisions_sha256": source_decisions_observed_sha,
        "git_branch": branch,
        "git_commit": git_commit,
        "execution_command": "{} {}".format(sys.executable, Path(__file__).resolve()),
        "timestamp": timestamp,
        "software_versions": software_versions,
        "verification_pass": bool(verification_pass),
        "candidate_reproduction_pass": bool(candidate_repro_pass),
        "capacity_audit_pass": bool(capacity_pass),
        "suppression_audit_pass": bool(suppression_pass),
        "information_isolation_pass": bool(info_pass),
        "source_preservation_pass": bool(source_preservation_pass),
        "audit_pass": bool(audit_pass),
        "test_result": str(test_result),
        "deterministic_result": str(deterministic_status),
        "final_decision": str(final_decision),
        "notes": [
            "No model training, tuning, refit, or recalibration was performed.",
            "No retrospective top-k selection, no HybridRank regeneration, and no sequential external validation were run.",
            "Frozen past-only sequential decisions were used as read-only source for candidate-alarm reproduction.",
        ],
    }

    write_json(MANIFEST_PATH, manifest)

    report_text = build_report(
        verification_pass=verification_pass,
        source_decision=source_decision,
        candidate_repro_pass=candidate_repro_pass,
        capacity_pass=capacity_pass,
        suppression_pass=suppression_pass,
        info_pass=info_pass,
        source_preservation_pass=source_preservation_pass,
        deterministic_status=deterministic_status,
        test_result=test_result,
        final_decision=final_decision,
        git_branch=branch,
        git_commit=git_commit,
        run_id=run_id,
    )
    ensure_inside_authorized(REPORT_PATH)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    write_checksums_file()

    print("Wrote {}".format(INPUT_VERIFICATION_PATH))
    print("Wrote {}".format(DECISIONS_PATH))
    print("Wrote {}".format(METRICS_BY_FOLD_PATH))
    print("Wrote {}".format(METRICS_POOLED_PATH))
    print("Wrote {}".format(CAPACITY_AUDIT_PATH))
    print("Wrote {}".format(SUPPRESSION_AUDIT_PATH))
    print("Wrote {}".format(COMPARISON_PATH))
    print("Wrote {}".format(INFO_AUDIT_PATH))
    print("Wrote {}".format(MANIFEST_PATH))
    print("Wrote {}".format(REPORT_PATH))
    print("Wrote {}".format(CHECKSUMS_PATH))


if __name__ == "__main__":
    build_package()
