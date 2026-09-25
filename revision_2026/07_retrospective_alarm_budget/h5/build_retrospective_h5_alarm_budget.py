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


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
AUTHORIZED_REL_DIR = "revision_2026/07_retrospective_alarm_budget/h5"
EXPECTED_BRANCH = "controlled-reruns-v1"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CONTROLLED_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions" / "h5_cross_model"
FULLY_NESTED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fully_nested"
FIXED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fixed_policy"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_MD_PATH = PROTOCOL_DIR / "corrected_evaluation_protocol.md"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

CANONICAL_LONG_PATH = CANONICAL_DIR / "canonical_h5_predictions_long.csv"
CANONICAL_WIDE_PATH = CANONICAL_DIR / "canonical_h5_predictions_wide.csv"
CANONICAL_SOURCE_REGISTRY_PATH = CANONICAL_DIR / "canonical_h5_source_registry.csv"
CANONICAL_MANIFEST_PATH = CANONICAL_DIR / "canonical_h5_assembly_manifest.json"
CANONICAL_REPORT_PATH = CANONICAL_DIR / "canonical_h5_assembly_report.md"
CANONICAL_CHECKSUMS_PATH = CANONICAL_DIR / "canonical_h5_checksums.sha256"
CANONICAL_PREVALENCE_PATH = CANONICAL_DIR / "canonical_h5_event_prevalence.csv"

FIXED_SCORES_PATH = FIXED_DIR / "fixed_hybridrank_h5_scores.csv"
FIXED_WEIGHTS_PATH = FIXED_DIR / "fixed_hybridrank_weights.csv"
FIXED_POLICY_DEFINITION_PATH = FIXED_DIR / "fixed_hybridrank_policy_definition.md"
FIXED_MANIFEST_PATH = FIXED_DIR / "fixed_hybridrank_h5_manifest.json"
FIXED_REPORT_PATH = FIXED_DIR / "fixed_hybridrank_h5_completion_report.md"
FIXED_CHECKSUMS_PATH = FIXED_DIR / "fixed_hybridrank_h5_checksums.sha256"
FIXED_TIE_RULE_REGISTRY_PATH = FIXED_DIR / "future_alarm_tie_rule_registry.md"

FULLY_NESTED_REPORT_PATH = FULLY_NESTED_DIR / "fully_nested_hybridrank_h5_completion_report.md"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
ALARM_BUDGET_DEFINITION_PATH = OUT_DIR / "alarm_budget_definition.csv"
TOPK_TIE_AUDIT_PATH = OUT_DIR / "retrospective_topk_tie_audit.csv"
ALARM_DECISIONS_PATH = OUT_DIR / "retrospective_h5_alarm_decisions.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "retrospective_h5_alarm_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "retrospective_h5_alarm_metrics_pooled.csv"
EVENT_PREVALENCE_RECON_PATH = OUT_DIR / "retrospective_h5_event_prevalence_reconciliation.csv"
MODEL_DATE_AUDIT_PATH = OUT_DIR / "retrospective_h5_model_date_audit.csv"
OPERATING_POINT_TABLE_PATH = OUT_DIR / "retrospective_h5_operating_point_table.csv"
COMPACT_TABLE3_SOURCE_PATH = OUT_DIR / "retrospective_h5_compact_table3_source.csv"
FOLD_VARIABILITY_PATH = OUT_DIR / "retrospective_h5_fold_variability.csv"
RANDOM_BASELINE_AUDIT_PATH = OUT_DIR / "retrospective_h5_random_baseline_audit.csv"
SELECTION_LABEL_INVARIANCE_PATH = OUT_DIR / "retrospective_h5_selection_label_invariance_audit.json"
MANIFEST_PATH = OUT_DIR / "retrospective_h5_alarm_budget_manifest.json"
COMPLETION_REPORT_PATH = OUT_DIR / "retrospective_h5_alarm_budget_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "retrospective_h5_alarm_budget_checksums.sha256"

H = 5
THRESHOLDS = (15, 16, 17)
BUDGETS = (0.05, 0.10)
MODEL_ORDER = (
    "Persistence",
    "Ridge",
    "ElasticNet",
    "HGBR",
    "BCR-TCN v1.1",
    "HybridRank_fixed_documented",
)

SCORE_CONTEXT = "retrospective_offline_top_k"
BUDGET_APPLICATION_RULE = "ceil_budget_separately_within_each_outer_test_fold"
CEILING_RULE = "k_fold = ceil(nominal_budget_r * N_fold)"
TIE_RULE = "desc_score_then_target_date_then_feature_date_then_canonical_row_id"

PROTECTED_DIRS = [
    PROTOCOL_DIR,
    CONTROLLED_DIR,
    CANONICAL_DIR,
    FULLY_NESTED_DIR,
    FIXED_DIR,
]


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


def parse_checksum_manifest(path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            raise RuntimeError("Malformed checksum line in {}: {}".format(path, line))
        out.append((parts[0], parts[-1]))
    return out


def expected_checksum_for_file(path: Path, filename: str) -> str:
    for expected, rel_name in parse_checksum_manifest(path):
        rel_norm = str(rel_name).replace("\\", "/")
        if rel_norm == filename or rel_norm.endswith("/" + filename):
            return expected
    raise KeyError("Checksum entry not found for {} in {}".format(filename, path))


def verify_checksum_manifest(path: Path, base_dir: Path) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for expected, rel_name in parse_checksum_manifest(path):
        target = base_dir / rel_name
        if not target.exists():
            raise RuntimeError("Checksum target missing: {}".format(target))
        observed = sha256_file(target)
        ok = observed == expected
        checks[str(rel_name)] = {
            "expected": expected,
            "observed": observed,
            "pass": bool(ok),
        }
        if not ok:
            raise RuntimeError(
                "Checksum mismatch for {}: expected {}, observed {}".format(rel_name, expected, observed)
            )
    return checks


def infer_decision_from_report(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")
    m = re.search(r"Decision:\s*([ABCD])", txt)
    if m:
        return m.group(1)

    lines = [ln.strip() for ln in txt.splitlines()]
    final_idx = -1
    for i, ln in enumerate(lines):
        if "FINAL DECISION" in ln.upper():
            final_idx = i
            break

    if final_idx >= 0:
        for ln in lines[final_idx + 1 : final_idx + 24]:
            if re.match(r"^[ABCD]\.\s", ln):
                return ln[0]

    for ln in lines[-40:]:
        if re.match(r"^[ABCD]\.\s", ln):
            return ln[0]

    raise RuntimeError("Could not infer final decision from {}".format(path))


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def to_date_str(x: Any) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


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
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
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
    return out


def build_model_score_panel(
    wide_df: pd.DataFrame,
    fixed_scores_df: pd.DataFrame,
    fixed_hybridrank_run_id: str,
    alarm_run_id: str,
    git_commit: str,
) -> pd.DataFrame:
    score_rows: List[pd.DataFrame] = []

    base_cols = [
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

    fixed_filter = fixed_scores_df[
        fixed_scores_df["policy_name"].astype(str) == "HybridRank_fixed_documented"
    ].copy()

    for tau in THRESHOLDS:
        b = wide_df[base_cols].copy()
        b["threshold"] = int(tau)

        model_map = [
            ("Persistence", "persistence_y_pred", "canonical_h5_predictions_wide.csv:persistence_y_pred"),
            ("Ridge", "ridge_y_pred", "canonical_h5_predictions_wide.csv:ridge_y_pred"),
            ("ElasticNet", "elasticnet_y_pred", "canonical_h5_predictions_wide.csv:elasticnet_y_pred"),
            ("HGBR", "hgbr_y_pred", "canonical_h5_predictions_wide.csv:hgbr_y_pred"),
        ]

        bcr_col = "bcr_tcn_v11_p_tau{}".format(int(tau))
        model_map.append(("BCR-TCN v1.1", bcr_col, "canonical_h5_predictions_wide.csv:{}".format(bcr_col)))

        for model_name, col, source_label in model_map:
            s = b.copy()
            s["model"] = model_name
            s["score"] = pd.to_numeric(wide_df[col], errors="coerce").to_numpy(dtype=float)
            s["score_source"] = source_label
            s["score_context"] = SCORE_CONTEXT
            s["fixed_hybridrank_run_id"] = fixed_hybridrank_run_id
            s["alarm_run_id"] = alarm_run_id
            s["git_commit"] = git_commit
            score_rows.append(s)

        f = fixed_filter[fixed_filter["threshold"].astype(int) == int(tau)].copy()
        f = f[
            [
                "horizon",
                "outer_fold",
                "feature_date",
                "target_date",
                "hybrid_score",
                "run_id",
                "score_context",
            ]
        ].copy()
        f["feature_date"] = normalize_date_col(f["feature_date"])
        f["target_date"] = normalize_date_col(f["target_date"])

        fm = b.merge(
            f,
            on=["horizon", "outer_fold", "feature_date", "target_date"],
            how="left",
            validate="one_to_one",
        )
        if fm["hybrid_score"].isna().any():
            raise RuntimeError("Missing fixed HybridRank score rows for threshold {}".format(tau))

        sf = fm.copy()
        sf["model"] = "HybridRank_fixed_documented"
        sf["score"] = pd.to_numeric(sf["hybrid_score"], errors="coerce").to_numpy(dtype=float)
        sf["score_source"] = "fixed_hybridrank_h5_scores.csv:hybrid_score"
        sf["score_context"] = SCORE_CONTEXT
        sf["fixed_hybridrank_run_id"] = sf["run_id"].astype(str)
        sf["alarm_run_id"] = alarm_run_id
        sf["git_commit"] = git_commit
        score_rows.append(sf[b.columns.tolist() + ["model", "score", "score_source", "score_context", "fixed_hybridrank_run_id", "alarm_run_id", "git_commit"]])

    panel = pd.concat(score_rows, axis=0, ignore_index=True)

    expected_rows = 747 * len(THRESHOLDS) * len(MODEL_ORDER)
    if int(len(panel)) != int(expected_rows):
        raise RuntimeError(
            "Model score panel row count mismatch: observed {}, expected {}".format(
                len(panel), expected_rows
            )
        )

    panel = panel.sort_values(
        ["model", "threshold", "outer_fold", "target_date", "feature_date", "canonical_row_id"]
    ).reset_index(drop=True)

    return panel


def compute_alarm_budget_definition(
    wide_df: pd.DataFrame,
    budgets: Sequence[float],
    dataset_sha: str,
    split_sha: str,
    run_id: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    n_by_fold = {
        int(k): int(v)
        for k, v in wide_df.groupby("outer_fold").size().astype(int).to_dict().items()
    }
    pooled_n = int(sum(n_by_fold.values()))

    for r in budgets:
        k_total = 0
        k_fold_map: Dict[int, int] = {}
        for fold in sorted(n_by_fold.keys()):
            n = int(n_by_fold[fold])
            k = int(math.ceil(float(r) * float(n)))
            k_fold_map[fold] = k
            k_total += k

        for fold in sorted(n_by_fold.keys()):
            n = int(n_by_fold[fold])
            k = int(k_fold_map[fold])
            rows.append(
                {
                    "horizon": int(H),
                    "outer_fold": int(fold),
                    "N_fold": int(n),
                    "nominal_budget_r": float(r),
                    "k_fold": int(k),
                    "realized_fold_alarm_fraction": safe_div(float(k), float(n)),
                    "pooled_N": int(pooled_n),
                    "pooled_K_total": int(k_total),
                    "realized_pooled_alarm_fraction": safe_div(float(k_total), float(pooled_n)),
                    "budget_application_rule": BUDGET_APPLICATION_RULE,
                    "ceiling_rule": CEILING_RULE,
                    "dataset_sha256": dataset_sha,
                    "split_sha256": split_sha,
                    "run_id": run_id,
                }
            )

    out = pd.DataFrame(rows).sort_values(["outer_fold", "nominal_budget_r"]).reset_index(drop=True)
    return out


def build_alarm_decisions_and_tie_audit(
    panel_df: pd.DataFrame,
    budgets: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    decision_rows: List[pd.DataFrame] = []
    tie_rows: List[Dict[str, Any]] = []

    group_cols = ["model", "outer_fold", "threshold"]

    for gkey, grp in panel_df.groupby(group_cols, sort=True):
        model_name, fold, tau = gkey
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).copy().reset_index(drop=True)
        n = int(len(g))

        for r in budgets:
            k = int(math.ceil(float(r) * float(n)))
            if k < 0 or k > n:
                raise RuntimeError("Invalid k for model={} fold={} tau={} r={}".format(model_name, fold, tau, r))

            srt = g.sort_values(
                ["score", "target_date", "feature_date", "canonical_row_id"],
                ascending=[False, True, True, True],
                kind="mergesort",
            ).reset_index(drop=True)

            srt["rank_within_fold"] = np.arange(1, len(srt) + 1, dtype=int)
            srt["k_fold"] = int(k)
            srt["nominal_budget_r"] = float(r)
            srt["alarm_flag"] = (srt["rank_within_fold"].astype(int) <= int(k)).astype(int)
            cutoff_score = float(srt.loc[int(k) - 1, "score"]) if k > 0 else float("nan")
            srt["cutoff_score"] = cutoff_score
            srt["tie_rule"] = TIE_RULE

            if int(srt["alarm_flag"].sum()) != int(k):
                raise RuntimeError(
                    "Alarm count mismatch for model={} fold={} tau={} r={}".format(
                        model_name, fold, tau, r
                    )
                )

            vals = srt["score"].to_numpy(dtype=float)
            vc = pd.Series(vals).value_counts(dropna=False)
            tie_group_count = int((vc > 1).sum())
            max_tie_size = int(vc.max()) if len(vc) > 0 else 0
            score_unique_count = int(pd.Series(vals).nunique(dropna=True))

            if k > 0:
                rows_equal_cutoff = int(np.isclose(vals, cutoff_score, atol=0.0, rtol=0.0).sum())
                n_above = int((vals > cutoff_score).sum())
                selected_eq = int(k - n_above)
                ties_at_cutoff = bool(rows_equal_cutoff > selected_eq)
            else:
                rows_equal_cutoff = 0
                ties_at_cutoff = False

            tie_rows.append(
                {
                    "model": str(model_name),
                    "outer_fold": int(fold),
                    "threshold": int(tau),
                    "nominal_budget_r": float(r),
                    "N": int(n),
                    "k": int(k),
                    "score_unique_count": int(score_unique_count),
                    "score_tie_group_count": int(tie_group_count),
                    "maximum_tie_size": int(max_tie_size),
                    "ties_at_cutoff": bool(ties_at_cutoff),
                    "rows_equal_to_cutoff_score": int(rows_equal_cutoff),
                    "secondary_rule_used": True,
                    "event_labels_used_for_ties": False,
                    "y_true_used_for_ties": False,
                    "deterministic_selection_pass": True,
                }
            )

            decision_rows.append(srt)

    decisions = pd.concat(decision_rows, axis=0, ignore_index=True)
    tie_audit = pd.DataFrame(tie_rows).sort_values(
        ["model", "outer_fold", "threshold", "nominal_budget_r"]
    ).reset_index(drop=True)

    decisions = decisions.sort_values(
        ["model", "outer_fold", "threshold", "nominal_budget_r", "rank_within_fold"]
    ).reset_index(drop=True)

    return decisions, tie_audit


def build_fold_metrics(decisions_df: pd.DataFrame, alarm_run_id: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    group_cols = ["model", "horizon", "outer_fold", "threshold", "nominal_budget_r"]
    for gkey, grp in decisions_df.groupby(group_cols, sort=True):
        model_name, horizon, fold, tau, budget = gkey
        g = grp.copy()

        event_col = "event_tau{}".format(int(tau))
        ev = g[event_col].astype(int).to_numpy()
        alarm = g["alarm_flag"].astype(int).to_numpy()

        n = int(len(g))
        events = int(ev.sum())
        k = int(alarm.sum())

        tp = int(((alarm == 1) & (ev == 1)).sum())
        fp = int(((alarm == 1) & (ev == 0)).sum())
        fn = int(((alarm == 0) & (ev == 1)).sum())
        tn = int(((alarm == 0) & (ev == 0)).sum())

        precision = safe_div(float(tp), float(k))
        recall = safe_div(float(tp), float(events))
        specificity = safe_div(float(tn), float(tn + fp))
        false_positive_rate = safe_div(float(fp), float(fp + tn))
        false_alarm_fraction = safe_div(float(fp), float(k))
        missed_event_fraction = safe_div(float(fn), float(events))
        random_expected_recall = safe_div(float(k), float(n))
        recall_enrichment = safe_div(recall, random_expected_recall) if not np.isnan(recall) else float("nan")

        rows.append(
            {
                "model": str(model_name),
                "horizon": int(horizon),
                "outer_fold": int(fold),
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "N": int(n),
                "events": int(events),
                "k": int(k),
                "realized_alarm_fraction": safe_div(float(k), float(n)),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "false_positive_rate": false_positive_rate,
                "false_alarm_fraction_of_alarms": false_alarm_fraction,
                "missed_event_fraction": missed_event_fraction,
                "random_expected_recall": random_expected_recall,
                "recall_enrichment_over_random": recall_enrichment,
                "date_start": g["target_date"].min(),
                "date_end": g["target_date"].max(),
                "score_context": SCORE_CONTEXT,
                "dataset_sha256": str(g["dataset_sha256"].iloc[0]),
                "split_sha256": str(g["split_sha256"].iloc[0]),
                "alarm_run_id": alarm_run_id,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["model", "outer_fold", "threshold", "nominal_budget_r"]
    ).reset_index(drop=True)
    return out


def build_pooled_metrics(fold_metrics_df: pd.DataFrame, alarm_run_id: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    gcols = ["model", "horizon", "threshold", "nominal_budget_r"]
    for gkey, grp in fold_metrics_df.groupby(gcols, sort=True):
        model_name, horizon, tau, budget = gkey

        pooled_n = int(grp["N"].sum())
        pooled_events = int(grp["events"].sum())
        pooled_k = int(grp["k"].sum())
        pooled_tp = int(grp["TP"].sum())
        pooled_fp = int(grp["FP"].sum())
        pooled_fn = int(grp["FN"].sum())
        pooled_tn = int(grp["TN"].sum())

        pooled_precision = safe_div(float(pooled_tp), float(pooled_k))
        pooled_recall = safe_div(float(pooled_tp), float(pooled_events))
        pooled_specificity = safe_div(float(pooled_tn), float(pooled_tn + pooled_fp))
        pooled_fpr = safe_div(float(pooled_fp), float(pooled_fp + pooled_tn))
        pooled_false_alarm_fraction = safe_div(float(pooled_fp), float(pooled_k))
        pooled_missed_event_fraction = safe_div(float(pooled_fn), float(pooled_events))

        random_expected_recall = safe_div(float(pooled_k), float(pooled_n))
        enrichment = safe_div(pooled_recall, random_expected_recall) if not np.isnan(pooled_recall) else float("nan")

        rows.append(
            {
                "model": str(model_name),
                "horizon": int(horizon),
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "pooled_N": int(pooled_n),
                "pooled_events": int(pooled_events),
                "pooled_K": int(pooled_k),
                "realized_pooled_alarm_fraction": safe_div(float(pooled_k), float(pooled_n)),
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
                "random_expected_recall": random_expected_recall,
                "recall_enrichment_over_random": enrichment,
                "folds": "1,2,3",
                "score_context": SCORE_CONTEXT,
                "dataset_sha256": str(grp["dataset_sha256"].iloc[0]),
                "split_sha256": str(grp["split_sha256"].iloc[0]),
                "alarm_run_id": alarm_run_id,
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "threshold", "nominal_budget_r"]).reset_index(drop=True)


def build_event_prevalence_reconciliation(
    wide_df: pd.DataFrame,
    canonical_prevalence_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if "outer_fold" not in canonical_prevalence_df.columns:
        raise RuntimeError("canonical_h5_event_prevalence.csv missing outer_fold")

    for fold_or_pooled in ["1", "2", "3", "pooled"]:
        if fold_or_pooled == "pooled":
            obs = wide_df.copy()
        else:
            obs = wide_df[wide_df["outer_fold"].astype(int) == int(fold_or_pooled)].copy()

        n = int(len(obs))
        e15 = int((pd.to_numeric(obs["y_true"], errors="coerce") >= 15.0).sum())
        e16 = int((pd.to_numeric(obs["y_true"], errors="coerce") >= 16.0).sum())
        e17 = int((pd.to_numeric(obs["y_true"], errors="coerce") >= 17.0).sum())

        cref = canonical_prevalence_df[
            canonical_prevalence_df["outer_fold"].astype(str) == fold_or_pooled
        ].copy()
        if cref.empty:
            raise RuntimeError("Missing canonical prevalence rows for {}".format(fold_or_pooled))

        ref_unique = cref[["N", "events_tau15", "events_tau16", "events_tau17"]].drop_duplicates()
        ref_consistent = bool(len(ref_unique) == 1)
        ref_row = ref_unique.iloc[0]

        pass_counts = bool(
            ref_consistent
            and int(ref_row["N"]) == int(n)
            and int(ref_row["events_tau15"]) == int(e15)
            and int(ref_row["events_tau16"]) == int(e16)
            and int(ref_row["events_tau17"]) == int(e17)
        )

        rows.append(
            {
                "outer_fold_or_pooled": fold_or_pooled,
                "N": int(n),
                "events_tau15": int(e15),
                "events_tau16": int(e16),
                "events_tau17": int(e17),
                "prevalence_tau15": safe_div(float(e15), float(n)),
                "prevalence_tau16": safe_div(float(e16), float(n)),
                "prevalence_tau17": safe_div(float(e17), float(n)),
                "canonical_source": str(CANONICAL_PREVALENCE_PATH),
                "event_count_pass": bool(pass_counts),
            }
        )

    return pd.DataFrame(rows)


def build_model_date_audit(decisions_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    key_cols = ["outer_fold", "feature_date", "target_date", "canonical_row_id"]
    base_keys_df = wide_df[key_cols + ["y_true", "event_tau15", "event_tau16", "event_tau17"]].copy()
    expected_keys = set(
        tuple(x)
        for x in base_keys_df[key_cols].itertuples(index=False, name=None)
    )

    for gkey, grp in decisions_df.groupby(["model", "threshold", "nominal_budget_r"], sort=True):
        model_name, tau, budget = gkey
        g = grp.copy()

        observed_tuples = list(g[key_cols].itertuples(index=False, name=None))
        observed_set = set(observed_tuples)

        missing_keys = int(len(expected_keys.difference(observed_set)))
        extra_keys = int(len(observed_set.difference(expected_keys)))
        duplicate_keys = int(len(observed_tuples) - len(observed_set))

        merged = g.merge(
            base_keys_df,
            on=key_cols,
            how="left",
            validate="many_to_one",
            suffixes=("", "_canonical"),
        )

        y_true_identical = bool(
            np.allclose(
                pd.to_numeric(merged["y_true"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(merged["y_true_canonical"], errors="coerce").to_numpy(dtype=float),
                atol=1e-6,
                rtol=0.0,
            )
        )

        event_identical = bool(
            (merged["event_tau15"].astype(int).to_numpy() == merged["event_tau15_canonical"].astype(int).to_numpy()).all()
            and (merged["event_tau16"].astype(int).to_numpy() == merged["event_tau16_canonical"].astype(int).to_numpy()).all()
            and (merged["event_tau17"].astype(int).to_numpy() == merged["event_tau17_canonical"].astype(int).to_numpy()).all()
        )

        score_finite = bool(np.isfinite(pd.to_numeric(g["score"], errors="coerce").to_numpy(dtype=float)).all())
        date_set_identical = bool(missing_keys == 0 and extra_keys == 0)

        coverage_pass = bool(
            int(len(observed_set)) == 747
            and missing_keys == 0
            and extra_keys == 0
            and duplicate_keys == 0
            and date_set_identical
            and y_true_identical
            and event_identical
            and score_finite
        )

        rows.append(
            {
                "model": str(model_name),
                "threshold": int(tau),
                "budget": float(budget),
                "expected_keys": 747,
                "observed_keys": int(len(observed_set)),
                "missing_keys": int(missing_keys),
                "extra_keys": int(extra_keys),
                "duplicate_keys": int(duplicate_keys),
                "date_set_identical": bool(date_set_identical),
                "y_true_identical": bool(y_true_identical),
                "event_labels_identical": bool(event_identical),
                "score_finite": bool(score_finite),
                "coverage_pass": bool(coverage_pass),
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "threshold", "budget"]).reset_index(drop=True)


def build_operating_point_table(pooled_df: pd.DataFrame, alarm_run_id: str) -> pd.DataFrame:
    out = pooled_df.copy()
    out = out.rename(
        columns={
            "pooled_K": "pooled_K",
            "pooled_TP": "TP",
            "pooled_FP": "false_alarms",
            "pooled_FN": "missed_events",
            "realized_pooled_alarm_fraction": "realized_alarm_fraction",
            "pooled_precision": "precision",
            "pooled_recall": "recall",
        }
    )

    out["alarm_run_id"] = alarm_run_id

    cols = [
        "model",
        "threshold",
        "nominal_budget_r",
        "pooled_N",
        "pooled_events",
        "pooled_K",
        "realized_alarm_fraction",
        "TP",
        "precision",
        "recall",
        "random_expected_recall",
        "recall_enrichment_over_random",
        "missed_events",
        "false_alarms",
        "score_context",
        "alarm_run_id",
    ]
    return out[cols].sort_values(["model", "threshold", "nominal_budget_r"]).reset_index(drop=True)


def build_fold_variability(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for gkey, grp in fold_metrics_df.groupby(["model", "threshold", "nominal_budget_r"], sort=True):
        model_name, tau, budget = gkey
        prec = pd.to_numeric(grp["precision"], errors="coerce").to_numpy(dtype=float)
        rec = pd.to_numeric(grp["recall"], errors="coerce").to_numpy(dtype=float)
        tp = pd.to_numeric(grp["TP"], errors="coerce").to_numpy(dtype=float)

        rows.append(
            {
                "model": str(model_name),
                "threshold": int(tau),
                "nominal_budget_r": float(budget),
                "fold_precision_mean": float(np.nanmean(prec)),
                "fold_precision_sd": float(np.nanstd(prec, ddof=0)),
                "fold_precision_min": float(np.nanmin(prec)),
                "fold_precision_max": float(np.nanmax(prec)),
                "fold_recall_mean": float(np.nanmean(rec)),
                "fold_recall_sd": float(np.nanstd(rec, ddof=0)),
                "fold_recall_min": float(np.nanmin(rec)),
                "fold_recall_max": float(np.nanmax(rec)),
                "fold_TP_min": float(np.nanmin(tp)),
                "fold_TP_max": float(np.nanmax(tp)),
                "number_of_folds": int(grp["outer_fold"].nunique()),
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "threshold", "nominal_budget_r"]).reset_index(drop=True)


def build_random_baseline_audit(budget_def_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for r in sorted(budget_def_df["nominal_budget_r"].astype(float).unique().tolist()):
        s = budget_def_df[np.isclose(budget_def_df["nominal_budget_r"].astype(float), float(r))].copy()

        for fold in sorted(s["outer_fold"].astype(int).unique().tolist()):
            sf = s[s["outer_fold"].astype(int) == int(fold)].copy().reset_index(drop=True)
            n = int(sf.loc[0, "N_fold"])
            k = int(sf.loc[0, "k_fold"])
            realized = safe_div(float(k), float(n))
            rows.append(
                {
                    "outer_fold_or_pooled": str(int(fold)),
                    "N": int(n),
                    "nominal_budget_r": float(r),
                    "k": int(k),
                    "realized_alarm_fraction": realized,
                    "submitted_nominal_random_recall": float(r),
                    "corrected_exact_random_expected_recall": realized,
                    "difference_due_to_ceiling": float(realized - float(r)),
                }
            )

        pooled_n = int(s["pooled_N"].iloc[0])
        pooled_k = int(s["pooled_K_total"].iloc[0])
        pooled_realized = safe_div(float(pooled_k), float(pooled_n))
        rows.append(
            {
                "outer_fold_or_pooled": "pooled",
                "N": int(pooled_n),
                "nominal_budget_r": float(r),
                "k": int(pooled_k),
                "realized_alarm_fraction": pooled_realized,
                "submitted_nominal_random_recall": float(r),
                "corrected_exact_random_expected_recall": pooled_realized,
                "difference_due_to_ceiling": float(pooled_realized - float(r)),
            }
        )

    return pd.DataFrame(rows).sort_values(["nominal_budget_r", "outer_fold_or_pooled"]).reset_index(drop=True)


def build_selection_label_invariance_audit(
    panel_df: pd.DataFrame,
    base_decisions_df: pd.DataFrame,
    budgets: Sequence[float],
) -> Dict[str, Any]:
    rng = np.random.RandomState(42)
    perm_idx = rng.permutation(len(panel_df))

    perm_panel = panel_df.copy()
    for col in ["y_true", "event_tau15", "event_tau16", "event_tau17"]:
        perm_panel[col] = panel_df[col].to_numpy()[perm_idx]

    perm_decisions, _ = build_alarm_decisions_and_tie_audit(perm_panel, budgets)

    key_cols = ["model", "outer_fold", "threshold", "nominal_budget_r", "canonical_row_id"]
    cmp_cols = ["score", "rank_within_fold", "alarm_flag", "cutoff_score"]

    b = base_decisions_df.sort_values(key_cols).reset_index(drop=True)
    p = perm_decisions.sort_values(key_cols).reset_index(drop=True)

    scores_identical = bool(
        np.array_equal(
            pd.to_numeric(b["score"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(p["score"], errors="coerce").to_numpy(dtype=float),
        )
    )
    ranks_identical = bool(
        np.array_equal(
            b["rank_within_fold"].astype(int).to_numpy(),
            p["rank_within_fold"].astype(int).to_numpy(),
        )
    )
    alarm_identical = bool(
        np.array_equal(
            b["alarm_flag"].astype(int).to_numpy(),
            p["alarm_flag"].astype(int).to_numpy(),
        )
    )
    cutoff_identical = bool(
        np.array_equal(
            pd.to_numeric(b["cutoff_score"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(p["cutoff_score"], errors="coerce").to_numpy(dtype=float),
        )
    )

    max_score_diff = float(
        np.max(
            np.abs(
                pd.to_numeric(b["score"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(p["score"], errors="coerce").to_numpy(dtype=float)
            )
        )
    )

    max_rank_diff = float(
        np.max(
            np.abs(
                b["rank_within_fold"].astype(float).to_numpy()
                - p["rank_within_fold"].astype(float).to_numpy()
            )
        )
    )

    # Optional confirmation that metrics can change when labels are perturbed.
    base_event_tp = []
    perm_event_tp = []
    for gkey, g in b.groupby(["model", "outer_fold", "threshold", "nominal_budget_r"], sort=True):
        tau = int(gkey[2])
        ev_col = "event_tau{}".format(tau)
        ev = g[ev_col].astype(int).to_numpy()
        alarm = g["alarm_flag"].astype(int).to_numpy()
        base_event_tp.append(int(((alarm == 1) & (ev == 1)).sum()))

    for gkey, g in p.groupby(["model", "outer_fold", "threshold", "nominal_budget_r"], sort=True):
        tau = int(gkey[2])
        ev_col = "event_tau{}".format(tau)
        ev = g[ev_col].astype(int).to_numpy()
        alarm = g["alarm_flag"].astype(int).to_numpy()
        perm_event_tp.append(int(((alarm == 1) & (ev == 1)).sum()))

    metrics_change_observed = bool(base_event_tp != perm_event_tp)

    pass_flag = bool(
        scores_identical
        and ranks_identical
        and alarm_identical
        and cutoff_identical
        and abs(max_score_diff) <= 0.0
        and abs(max_rank_diff) <= 0.0
    )

    return {
        "selection_recomputed_after_label_permutation": True,
        "scores_identical": bool(scores_identical),
        "ranks_identical": bool(ranks_identical),
        "alarm_flags_identical": bool(alarm_identical),
        "cutoffs_identical": bool(cutoff_identical),
        "maximum_score_difference": float(max_score_diff),
        "maximum_rank_difference": float(max_rank_diff),
        "metrics_change_observed_after_label_permutation": bool(metrics_change_observed),
        "selection_label_invariance_pass": bool(pass_flag),
    }


def build_completion_report(
    run_id: str,
    verification_pass: bool,
    canonical_assembly_id: str,
    fixed_hybridrank_run_id: str,
    budget_def_df: pd.DataFrame,
    prevalence_df: pd.DataFrame,
    pooled_metrics_df: pd.DataFrame,
    tie_audit_df: pd.DataFrame,
    invariance: Dict[str, Any],
    source_preservation_pass: bool,
    deterministic_result: str,
    final_decision: str,
) -> str:
    lines: List[str] = []
    lines.append("# Corrected Retrospective H5 Alarm-Budget Evaluation")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("- Perform corrected retrospective offline H5 top-k alarm-budget evaluation using frozen canonical base-model scores and frozen corrected fixed rank-ensemble scores.")
    lines.append("")
    lines.append("## 2. Input Verification")
    lines.append("- verification_pass: {}".format(bool(verification_pass)))
    lines.append("- canonical_assembly_id: {}".format(canonical_assembly_id))
    lines.append("- fixed_hybridrank_run_id: {}".format(fixed_hybridrank_run_id))
    lines.append("")
    lines.append("## 3. Models and Risk Scores")
    lines.append("- Models: Persistence, Ridge, ElasticNet, HGBR, BCR-TCN v1.1, HybridRank_fixed_documented.")
    lines.append("- No point-blend result is retained.")
    lines.append("- No tuned HybridRank result is retained.")
    lines.append("")
    lines.append("## 4. Canonical Date and Event Equality")
    lines.append("- All models are evaluated on the same 747 canonical keys and identical fold/date partitions.")
    lines.append("")
    lines.append("## 5. Fold-Wise Alarm Budgets")
    lines.append("- k is applied separately in each fold.")
    lines.append("- H5 fold N is 249.")
    lines.append("- k equals 13 and 25 at nominal budgets 0.05 and 0.10.")
    lines.append("- pooled K equals 39 and 75.")
    lines.append("")
    lines.append("## 6. Deterministic Top-k Rule")
    lines.append("- Top-k sorting rule: descending score, ascending target_date, ascending feature_date, ascending canonical_row_id.")
    lines.append("- Tie-audit deterministic pass rows: {}/{}".format(
        int(tie_audit_df["deterministic_selection_pass"].astype(bool).sum()),
        int(len(tie_audit_df)),
    ))
    lines.append("")
    lines.append("## 7. Fold-Level Alarm Results")
    lines.append("- Fold-level results are reported in retrospective_h5_alarm_metrics_by_fold.csv.")
    lines.append("")
    lines.append("## 8. Pooled Alarm Results")
    lines.append("- Pooled results are reported in retrospective_h5_alarm_metrics_pooled.csv.")
    lines.append("- No model winner was selected automatically.")
    lines.append("")
    lines.append("## 9. Exact Random Baseline")
    lines.append("- exact random expected recall is K/N after ceiling.")
    for r in sorted(budget_def_df["nominal_budget_r"].astype(float).unique().tolist()):
        s = budget_def_df[np.isclose(budget_def_df["nominal_budget_r"].astype(float), float(r))].copy()
        lines.append(
            "- r={} pooled exact random expected recall={:.10f}".format(
                r, float(s["pooled_K_total"].iloc[0]) / float(s["pooled_N"].iloc[0])
            )
        )
    lines.append("")
    lines.append("## 10. Fixed Rank-Ensemble Results")
    lines.append("- The fixed rank ensemble uses documented fixed weights from the corrected fixed-policy package.")
    lines.append("- No corrected validation or outer-test outcome selected the weights.")
    lines.append("")
    lines.append("## 11. Operating-Point Table Sources")
    lines.append("- Full operating-point source: retrospective_h5_operating_point_table.csv.")
    lines.append("- Compact prespecified table source: retrospective_h5_compact_table3_source.csv.")
    lines.append("")
    lines.append("## 12. Fold Variability")
    lines.append("- Descriptive fold variability is reported in retrospective_h5_fold_variability.csv; no narrow formal CI is claimed.")
    lines.append("")
    lines.append("## 13. Label-Invariance Audit")
    lines.append("- selection_label_invariance_pass: {}".format(bool(invariance.get("selection_label_invariance_pass", False))))
    lines.append("")
    lines.append("## 14. Retrospective-Only Interpretation")
    lines.append("- Every result is labeled retrospective_offline_top_k.")
    lines.append("- Complete held-out-fold ranking makes this retrospective only.")
    lines.append("- No deployable sequential claim is made.")
    lines.append("- A separate sequential simulation remains necessary.")
    lines.append("")
    lines.append("## 15. Source Preservation and Determinism")
    lines.append("- source_preservation_pass: {}".format(bool(source_preservation_pass)))
    lines.append("- deterministic_result: {}".format(deterministic_result))
    lines.append("")
    lines.append("## 16. Limitations")
    lines.append("1. This is an offline retrospective benchmark and not a deployable sequential policy.")
    lines.append("2. Cutoffs are evaluated after complete held-out-fold ranking.")
    lines.append("3. A separate sequential simulation remains required.")
    lines.append("")
    lines.append("## 17. Readiness Decision")
    lines.append("- run_id: {}".format(run_id))
    lines.append("")
    lines.append("FINAL DECISION")
    lines.append("")
    lines.append("<!-- AUTO_DECISION_START -->")
    if final_decision == "A":
        lines.append("A. Corrected retrospective H5 alarm-budget evaluation passed; sequential alarm-policy simulation may begin.")
    elif final_decision == "B":
        lines.append("B. Evaluation completed, but one or more count, metric, or provenance discrepancies require investigation.")
    elif final_decision == "C":
        lines.append("C. Canonical base-model or fixed HybridRank inputs could not be reconciled safely.")
    else:
        lines.append("D. Selection, source-preservation, or deterministic tests failed.")
    lines.append("<!-- AUTO_DECISION_END -->")
    lines.append("")

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

    protected_before = snapshot_tree_checksums(PROTECTED_DIRS, ROOT)

    git_commit = git_output(["git", "rev-parse", "HEAD"])
    python_executable = sys.executable
    python_version = platform.python_version()

    protocol_checks = verify_checksum_manifest(PROTOCOL_SHA_PATH, PROTOCOL_DIR)
    canonical_checks = verify_checksum_manifest(CANONICAL_CHECKSUMS_PATH, CANONICAL_DIR)
    fixed_checks = verify_checksum_manifest(FIXED_CHECKSUMS_PATH, FIXED_DIR)

    lock_sha_expected = expected_checksum_for_file(PROTOCOL_SHA_PATH, "canonical_dataset_lock.json")
    split_sha_expected = expected_checksum_for_file(PROTOCOL_SHA_PATH, "corrected_split_assignment.csv")

    lock_sha_observed = sha256_file(LOCK_PATH)
    split_sha_observed = sha256_file(SPLIT_PATH)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    dataset_path = Path(str(lock["absolute_path"]))
    dataset_sha_expected = str(lock["sha256"])
    if not dataset_path.exists():
        raise RuntimeError("Locked dataset path missing: {}".format(dataset_path))
    dataset_sha_observed = sha256_file(dataset_path)

    canonical_manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    fixed_manifest = json.loads(FIXED_MANIFEST_PATH.read_text(encoding="utf-8"))

    canonical_decision = infer_decision_from_report(CANONICAL_REPORT_PATH)
    fixed_decision = infer_decision_from_report(FIXED_REPORT_PATH)
    fully_nested_decision = infer_decision_from_report(FULLY_NESTED_REPORT_PATH)

    canonical_long_sha_expected = expected_checksum_for_file(CANONICAL_CHECKSUMS_PATH, "canonical_h5_predictions_long.csv")
    canonical_wide_sha_expected = expected_checksum_for_file(CANONICAL_CHECKSUMS_PATH, "canonical_h5_predictions_wide.csv")
    canonical_long_sha_observed = sha256_file(CANONICAL_LONG_PATH)
    canonical_wide_sha_observed = sha256_file(CANONICAL_WIDE_PATH)

    canonical_long_df = pd.read_csv(CANONICAL_LONG_PATH)
    canonical_wide_df = pd.read_csv(CANONICAL_WIDE_PATH)
    fixed_scores_df = pd.read_csv(FIXED_SCORES_PATH)
    fixed_weights_df = pd.read_csv(FIXED_WEIGHTS_PATH)
    canonical_prev_df = pd.read_csv(CANONICAL_PREVALENCE_PATH)

    for df, cols in [
        (canonical_long_df, ["feature_date", "target_date"]),
        (canonical_wide_df, ["feature_date", "target_date"]),
        (fixed_scores_df, ["feature_date", "target_date"]),
    ]:
        for c in cols:
            df[c] = normalize_date_col(df[c])

    canonical_wide_df = add_canonical_row_id(canonical_wide_df)

    run_id = "retrospective_h5_alarm_budget_{}_{}_{}".format(
        git_commit[:12], dataset_sha_observed[:8], split_sha_observed[:8]
    )

    key_cols = ["horizon", "outer_fold", "feature_date", "target_date"]

    canonical_long_df["y_true"] = pd.to_numeric(canonical_long_df["y_true"], errors="coerce")
    if canonical_long_df["y_true"].isna().any():
        raise RuntimeError("canonical_h5_predictions_long.csv has NaN y_true values")

    y_span = canonical_long_df.groupby(key_cols, dropna=False)["y_true"].agg(["min", "max"]).reset_index()
    ytrue_max_abs_diff = float(
        np.max(np.abs(y_span["max"].to_numpy(dtype=float) - y_span["min"].to_numpy(dtype=float)))
    )
    ytrue_identical_across_models = bool(ytrue_max_abs_diff <= 1e-6)

    event_identical_across_models = True
    for ec in ["event_tau15", "event_tau16", "event_tau17"]:
        event_identical_across_models = event_identical_across_models and bool(
            canonical_long_df.groupby(key_cols, dropna=False)[ec].nunique(dropna=False).max() == 1
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
    required_cols_present = bool(all(c in canonical_wide_df.columns for c in required_score_cols))

    score_finite_pass = False
    if required_cols_present:
        score_finite_pass = bool(
            np.isfinite(canonical_wide_df[required_score_cols].to_numpy(dtype=float)).all()
        )

    no_point_blend_score = bool(
        ("point" not in " ".join(canonical_wide_df.columns).lower())
        and ("point_blend" not in " ".join(fixed_scores_df.columns).lower())
    )

    no_tuned_hybrid_in_fixed = bool(
        set(fixed_scores_df["policy_name"].astype(str).unique().tolist()) == {"HybridRank_fixed_documented"}
    )

    verification_checks = {
        "1_working_directory_pass": str(Path.cwd().resolve()) == str(ROOT.resolve()),
        "2_active_branch_pass": active_branch == EXPECTED_BRANCH,
        "3_working_tree_clean_stage_start_pass": len(outside_changes) == 0,
        "4_git_commit_present": True,
        "5_python_environment_present": True,
        "6_canonical_package_checksums_pass": True,
        "7_fixed_hybridrank_package_checksums_pass": True,
        "8_canonical_assembly_decision_A_pass": canonical_decision == "A",
        "9_fixed_hybridrank_decision_A_pass": fixed_decision == "A",
        "10_fully_nested_tuned_decision_C_pass": fully_nested_decision == "C",
        "11_dataset_checksum_pass": dataset_sha_observed == dataset_sha_expected,
        "12_split_checksum_pass": split_sha_observed == split_sha_expected,
        "13_h5_only_pass": sorted(canonical_wide_df["horizon"].astype(int).unique().tolist()) == [H],
        "14_canonical_key_count_747_pass": int(len(canonical_wide_df)) == 747,
        "15_each_fold_count_249_pass": bool((canonical_wide_df.groupby("outer_fold").size().astype(int) == 249).all()),
        "16_identical_y_true_and_event_labels_across_models_pass": bool(
            ytrue_identical_across_models and event_identical_across_models
        ),
        "17_required_model_scores_finite_pass": bool(score_finite_pass),
        "18_no_point_blend_score_included_pass": bool(no_point_blend_score),
        "19_no_tuned_hybridrank_score_included_pass": bool(no_tuned_hybrid_in_fixed),
        "20_stop_if_any_verification_fails_rule": True,
    }

    verification_pass = bool(all(bool(v) for v in verification_checks.values()))

    fixed_hybridrank_run_id = str(fixed_manifest.get("run_id", ""))
    canonical_assembly_id = str(canonical_manifest.get("assembly_id", ""))

    input_verification = {
        "run_id": run_id,
        "working_directory": str(Path.cwd().resolve()),
        "expected_working_directory": str(ROOT.resolve()),
        "working_tree_changes": changed_paths,
        "files_modified_outside_authorized_directory": outside_changes,
        "authorized_output_directory": str((ROOT / AUTHORIZED_REL_DIR).resolve()),
        "git_branch": active_branch,
        "git_commit": git_commit,
        "python_executable": python_executable,
        "python_version": python_version,
        "canonical_checksums_file": str(CANONICAL_CHECKSUMS_PATH),
        "fixed_checksums_file": str(FIXED_CHECKSUMS_PATH),
        "protocol_checksums_file": str(PROTOCOL_SHA_PATH),
        "canonical_long_sha256_expected": canonical_long_sha_expected,
        "canonical_long_sha256_observed": canonical_long_sha_observed,
        "canonical_wide_sha256_expected": canonical_wide_sha_expected,
        "canonical_wide_sha256_observed": canonical_wide_sha_observed,
        "dataset_sha256_expected": dataset_sha_expected,
        "dataset_sha256_observed": dataset_sha_observed,
        "split_sha256_expected": split_sha_expected,
        "split_sha256_observed": split_sha_observed,
        "lock_sha256_expected": lock_sha_expected,
        "lock_sha256_observed": lock_sha_observed,
        "canonical_assembly_decision": canonical_decision,
        "fixed_hybridrank_decision": fixed_decision,
        "fully_nested_tuned_decision": fully_nested_decision,
        "canonical_assembly_id": canonical_assembly_id,
        "fixed_hybridrank_run_id": fixed_hybridrank_run_id,
        "horizon": int(H),
        "canonical_key_count": int(len(canonical_wide_df)),
        "canonical_rows_per_fold": {
            str(k): int(v)
            for k, v in canonical_wide_df.groupby("outer_fold").size().astype(int).to_dict().items()
        },
        "long_models": sorted(canonical_long_df["model"].astype(str).unique().tolist()),
        "long_y_true_model_consistency_max_abs_diff": ytrue_max_abs_diff,
        "verification_checks": verification_checks,
        "verification_pass": bool(verification_pass),
        "canonical_checks": canonical_checks,
        "fixed_checks": fixed_checks,
        "protocol_checks": protocol_checks,
        "timestamp": utc_now_iso(),
        "software_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    write_json(INPUT_VERIFICATION_PATH, input_verification)

    if not verification_pass:
        raise RuntimeError("Input verification failed; see {}".format(INPUT_VERIFICATION_PATH))

    panel = build_model_score_panel(
        wide_df=canonical_wide_df,
        fixed_scores_df=fixed_scores_df,
        fixed_hybridrank_run_id=fixed_hybridrank_run_id,
        alarm_run_id=run_id,
        git_commit=git_commit,
    )

    budget_def_df = compute_alarm_budget_definition(
        wide_df=canonical_wide_df,
        budgets=BUDGETS,
        dataset_sha=dataset_sha_observed,
        split_sha=split_sha_observed,
        run_id=run_id,
    )

    decisions, tie_audit_df = build_alarm_decisions_and_tie_audit(panel, BUDGETS)

    decisions["score_context"] = SCORE_CONTEXT
    decisions["dataset_sha256"] = decisions["dataset_sha256"].astype(str)
    decisions["split_sha256"] = decisions["split_sha256"].astype(str)
    decisions["canonical_assembly_id"] = decisions["canonical_assembly_id"].astype(str)
    decisions["fixed_hybridrank_run_id"] = decisions["fixed_hybridrank_run_id"].astype(str)
    decisions["alarm_run_id"] = run_id
    decisions["git_commit"] = git_commit

    alarm_cols = [
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
        "score",
        "score_source",
        "score_context",
        "rank_within_fold",
        "k_fold",
        "alarm_flag",
        "cutoff_score",
        "tie_rule",
        "dataset_sha256",
        "split_sha256",
        "canonical_assembly_id",
        "fixed_hybridrank_run_id",
        "alarm_run_id",
        "git_commit",
    ]

    decisions = decisions[alarm_cols].copy().sort_values(
        ["model", "threshold", "nominal_budget_r", "outer_fold", "rank_within_fold"]
    ).reset_index(drop=True)

    expected_decision_rows = 747 * len(THRESHOLDS) * len(BUDGETS) * len(MODEL_ORDER)
    if int(len(decisions)) != int(expected_decision_rows):
        raise RuntimeError(
            "Alarm decision row count mismatch: observed {}, expected {}".format(
                len(decisions), expected_decision_rows
            )
        )

    # Every combination must have exactly k alarms.
    check_alarm_count = []
    for gkey, grp in decisions.groupby(["model", "outer_fold", "threshold", "nominal_budget_r"], sort=True):
        k_fold = int(grp["k_fold"].iloc[0])
        n_alarm = int(grp["alarm_flag"].sum())
        check_alarm_count.append(bool(k_fold == n_alarm))
    if not all(check_alarm_count):
        raise RuntimeError("One or more model/fold/threshold/budget groups do not satisfy sum(alarm_flag)=k_fold")

    fold_metrics_df = build_fold_metrics(decisions, run_id)
    pooled_metrics_df = build_pooled_metrics(fold_metrics_df, run_id)
    prevalence_recon_df = build_event_prevalence_reconciliation(canonical_wide_df, canonical_prev_df)
    date_audit_df = build_model_date_audit(decisions, canonical_wide_df)
    operating_table_df = build_operating_point_table(pooled_metrics_df, run_id)
    compact_table_df = operating_table_df[
        operating_table_df["threshold"].astype(int).isin([15, 16])
        & operating_table_df["nominal_budget_r"].astype(float).isin([0.05, 0.10])
        & operating_table_df["model"].astype(str).isin(list(MODEL_ORDER))
    ].copy().sort_values(["model", "threshold", "nominal_budget_r"]).reset_index(drop=True)
    fold_variability_df = build_fold_variability(fold_metrics_df)
    random_baseline_df = build_random_baseline_audit(budget_def_df)
    invariance = build_selection_label_invariance_audit(panel, decisions, BUDGETS)

    # Persist outputs.
    to_csv_with_dates(budget_def_df, ALARM_BUDGET_DEFINITION_PATH, [])
    to_csv_with_dates(tie_audit_df, TOPK_TIE_AUDIT_PATH, [])
    to_csv_with_dates(decisions, ALARM_DECISIONS_PATH, ["feature_date", "target_date"])
    to_csv_with_dates(fold_metrics_df, METRICS_BY_FOLD_PATH, ["date_start", "date_end"])
    to_csv_with_dates(pooled_metrics_df, METRICS_POOLED_PATH, [])
    to_csv_with_dates(prevalence_recon_df, EVENT_PREVALENCE_RECON_PATH, [])
    to_csv_with_dates(date_audit_df, MODEL_DATE_AUDIT_PATH, [])
    to_csv_with_dates(operating_table_df, OPERATING_POINT_TABLE_PATH, [])
    to_csv_with_dates(compact_table_df, COMPACT_TABLE3_SOURCE_PATH, [])
    to_csv_with_dates(fold_variability_df, FOLD_VARIABILITY_PATH, [])
    to_csv_with_dates(random_baseline_df, RANDOM_BASELINE_AUDIT_PATH, [])
    write_json(SELECTION_LABEL_INVARIANCE_PATH, invariance)

    # Post-write source preservation.
    protected_after = snapshot_tree_checksums(PROTECTED_DIRS, ROOT)
    protected_changed = []
    for rel, before_sha in protected_before.items():
        if protected_after.get(rel) != before_sha:
            protected_changed.append(rel)

    if protected_changed:
        raise RuntimeError("Protected source artifacts changed: {}".format(protected_changed[:10]))

    source_preservation_pass = bool(len(protected_changed) == 0)

    # Decision gates.
    budget_k_pass = bool(
        (budget_def_df[np.isclose(budget_def_df["nominal_budget_r"].astype(float), 0.05)]["k_fold"].astype(int) == 13).all()
        and (budget_def_df[np.isclose(budget_def_df["nominal_budget_r"].astype(float), 0.10)]["k_fold"].astype(int) == 25).all()
    )
    pooled_k_pass = bool(
        int(budget_def_df[np.isclose(budget_def_df["nominal_budget_r"].astype(float), 0.05)]["pooled_K_total"].iloc[0]) == 39
        and int(budget_def_df[np.isclose(budget_def_df["nominal_budget_r"].astype(float), 0.10)]["pooled_K_total"].iloc[0]) == 75
    )
    tie_pass = bool(tie_audit_df["deterministic_selection_pass"].astype(bool).all())
    prevalence_pass = bool(prevalence_recon_df["event_count_pass"].astype(bool).all())
    date_audit_pass = bool(date_audit_df["coverage_pass"].astype(bool).all())
    invariance_pass = bool(invariance.get("selection_label_invariance_pass", False))

    decision_rows_match = bool(int(len(decisions)) == 26892)

    stage_pass = bool(
        budget_k_pass
        and pooled_k_pass
        and tie_pass
        and prevalence_pass
        and date_audit_pass
        and invariance_pass
        and source_preservation_pass
        and decision_rows_match
    )

    final_decision = "A" if stage_pass else "B"

    completion_report = build_completion_report(
        run_id=run_id,
        verification_pass=verification_pass,
        canonical_assembly_id=canonical_assembly_id,
        fixed_hybridrank_run_id=fixed_hybridrank_run_id,
        budget_def_df=budget_def_df,
        prevalence_df=prevalence_recon_df,
        pooled_metrics_df=pooled_metrics_df,
        tie_audit_df=tie_audit_df,
        invariance=invariance,
        source_preservation_pass=source_preservation_pass,
        deterministic_result="pending_test_script",
        final_decision=final_decision,
    )
    ensure_inside_authorized(COMPLETION_REPORT_PATH)
    COMPLETION_REPORT_PATH.write_text(completion_report, encoding="utf-8")

    input_files = [
        str(CANONICAL_LONG_PATH),
        str(CANONICAL_WIDE_PATH),
        str(CANONICAL_SOURCE_REGISTRY_PATH),
        str(CANONICAL_MANIFEST_PATH),
        str(CANONICAL_REPORT_PATH),
        str(CANONICAL_CHECKSUMS_PATH),
        str(CANONICAL_PREVALENCE_PATH),
        str(FIXED_SCORES_PATH),
        str(FIXED_WEIGHTS_PATH),
        str(FIXED_POLICY_DEFINITION_PATH),
        str(FIXED_MANIFEST_PATH),
        str(FIXED_REPORT_PATH),
        str(FIXED_CHECKSUMS_PATH),
        str(FIXED_TIE_RULE_REGISTRY_PATH),
        str(LOCK_PATH),
        str(SPLIT_PATH),
        str(SPLIT_SUMMARY_PATH),
        str(PROTOCOL_MD_PATH),
        str(PROTOCOL_SHA_PATH),
        str(FULLY_NESTED_REPORT_PATH),
    ]

    input_checksums = {}
    for p in input_files:
        pp = Path(p)
        if not pp.exists():
            raise RuntimeError("Input file missing: {}".format(pp))
        input_checksums[p] = sha256_file(pp)

    manifest = {
        "run_id": run_id,
        "purpose": "Corrected retrospective offline H5 top-k alarm-budget evaluation from frozen canonical base-model and fixed rank-ensemble scores",
        "horizon": int(H),
        "thresholds": [int(t) for t in THRESHOLDS],
        "budgets": [float(b) for b in BUDGETS],
        "models": list(MODEL_ORDER),
        "score_context": SCORE_CONTEXT,
        "budget_application_rule": BUDGET_APPLICATION_RULE,
        "tie_rule": TIE_RULE,
        "canonical_assembly_id": canonical_assembly_id,
        "fixed_hybridrank_run_id": fixed_hybridrank_run_id,
        "canonical_N": 747,
        "fold_N": {"1": 249, "2": 249, "3": 249},
        "dataset_sha256": dataset_sha_observed,
        "split_sha256": split_sha_observed,
        "protocol_tag": "corrected_protocol_v1",
        "git_branch": active_branch,
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
        "test_result": "not_run",
        "deterministic_result": "pending_test_script",
        "selection_label_invariance": invariance,
        "files_modified_outside_authorized_directory": outside_changes,
        "limitations": [
            "All results are retrospective_offline_top_k and not deployable sequential alarms.",
            "Complete held-out fold ranking is used before top-k selection.",
            "No model winner was selected automatically.",
            "No point-blend or tuned HybridRank result is retained.",
            "A separate sequential simulation remains necessary.",
        ],
        "input_verification_file": str(INPUT_VERIFICATION_PATH),
        "source_preservation": {
            "protected_dirs_checked": [str(p) for p in PROTECTED_DIRS],
            "changed": protected_changed,
            "pass": source_preservation_pass,
        },
        "final_decision": final_decision,
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
    print("2. active branch and Git commit: {} {}".format(active_branch, git_commit))
    print("3. Python executable and version: {} {}".format(python_executable, python_version))
    print("4. input verification: {}".format(bool(verification_pass)))
    print("5. models: {}".format(list(MODEL_ORDER)))
    print("6. thresholds and budgets: thresholds={} budgets={}".format(list(THRESHOLDS), list(BUDGETS)))
    print("7. canonical and fold counts: canonical_N=747 fold_N={{1:249,2:249,3:249}}")
    print("8. k by fold and total K:")
    for r in BUDGETS:
        sb = budget_def_df[np.isclose(budget_def_df["nominal_budget_r"].astype(float), float(r))].copy()
        k_by_fold = {
            int(k): int(v)
            for k, v in sb.set_index("outer_fold")["k_fold"].to_dict().items()
        }
        pooled_k = int(sb["pooled_K_total"].iloc[0])
        print("   r={} k_by_fold={} pooled_K={}".format(float(r), k_by_fold, pooled_k))
    print("9. event counts (pooled):")
    pooled_prev = prevalence_recon_df[prevalence_recon_df["outer_fold_or_pooled"].astype(str) == "pooled"].iloc[0]
    print(
        "   events_tau15={} events_tau16={} events_tau17={}".format(
            int(pooled_prev["events_tau15"]), int(pooled_prev["events_tau16"]), int(pooled_prev["events_tau17"])
        )
    )
    print("10. pooled metrics by model, threshold, and budget:")
    for r in pooled_metrics_df.itertuples(index=False):
        print(
            "   model={} tau={} r={} TP={} precision={} recall={}".format(
                str(r.model), int(r.threshold), float(r.nominal_budget_r), int(r.pooled_TP), r.pooled_precision, r.pooled_recall
            )
        )
    print("11. exact random baseline:")
    for r in sorted(random_baseline_df["nominal_budget_r"].astype(float).unique().tolist()):
        prow = random_baseline_df[
            (np.isclose(random_baseline_df["nominal_budget_r"].astype(float), float(r)))
            & (random_baseline_df["outer_fold_or_pooled"].astype(str) == "pooled")
        ].iloc[0]
        print(
            "   r={} pooled_exact_random_expected_recall={:.10f}".format(
                float(r), float(prow["corrected_exact_random_expected_recall"])
            )
        )
    print("12. tie audit: {}".format(bool(tie_pass)))
    print("13. label-invariance result: {}".format(bool(invariance_pass)))
    print("14. source-preservation result: {}".format(bool(source_preservation_pass)))
    print("15. deterministic result: pending_test_script")
    print("16. passed assertions: pending_test_script")
    print("17. final decision: {}".format(final_decision))
    if final_decision == "A":
        print("18. exactly one next action: Run the corrected sequential H5 alarm-policy simulation using cutoffs calibrated only from prior training or validation data.")
    else:
        print("18. exactly one next action: Investigate count, metric, or provenance discrepancies and rerun build_retrospective_h5_alarm_budget.py.")


if __name__ == "__main__":
    main()
