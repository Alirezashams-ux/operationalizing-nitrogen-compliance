from __future__ import annotations

import hashlib
import json
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
AUTHORIZED_REL_DIR = "revision_2026/06_corrected_hybridrank/h5/fixed_policy"
EXPECTED_BRANCH = "controlled-reruns-v1"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CONTROLLED_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions" / "h5_cross_model"
FULLY_NESTED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fully_nested"

ORIGINAL_ROOT = Path("/home/alrezshams/acs_tnout_ulsan")
ORIGINAL_SCRIPT_PATH = ORIGINAL_ROOT / "src" / "make_hybrid_rank_ensemble_v2.py"
ORIGINAL_RUN_DIR = ORIGINAL_ROOT / "results" / "hybrid_rank_v2_runs" / "20260220_131820_H5_hybrid_rank_v2"
ORIGINAL_RUN_CONFIG_PATH = ORIGINAL_RUN_DIR / "run_config.json"
ORIGINAL_FIXED_WEIGHTS_PATH = ORIGINAL_RUN_DIR / "models" / "paper_fixed_rank_weights_H5.csv"
ORIGINAL_RANK_SCORES_PATH = ORIGINAL_RUN_DIR / "predictions" / "rank_scores_tauwise_H5.csv"
ORIGINAL_HYBRID_PREDS_PATH = ORIGINAL_RUN_DIR / "hybrid_rank_v2_H5_preds.csv"
ORIGINAL_MODEL_REGISTRY_PATH = ORIGINAL_RUN_DIR / "models" / "model_registry_H5.csv"

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

FULLY_NESTED_MANIFEST_PATH = FULLY_NESTED_DIR / "fully_nested_hybridrank_h5_manifest.json"
FULLY_NESTED_REPORT_PATH = FULLY_NESTED_DIR / "fully_nested_hybridrank_h5_completion_report.md"
FULLY_NESTED_FOLD1_DIAG_PATH = FULLY_NESTED_DIR / "fold1_component_validation_tau16_diagnostic.csv"
FULLY_NESTED_CHECKSUMS_PATH = FULLY_NESTED_DIR / "fully_nested_hybridrank_h5_checksums.sha256"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
POLICY_DEFINITION_PATH = OUT_DIR / "fixed_hybridrank_policy_definition.md"
WEIGHTS_PATH = OUT_DIR / "fixed_hybridrank_weights.csv"
SCORES_PATH = OUT_DIR / "fixed_hybridrank_h5_scores.csv"
RANK_AUDIT_PATH = OUT_DIR / "fixed_hybridrank_rank_audit.csv"
ISOLATION_AUDIT_PATH = OUT_DIR / "fixed_hybridrank_isolation_audit.csv"
LABEL_INVARIANCE_PATH = OUT_DIR / "fixed_hybridrank_label_invariance_audit.json"
SCORE_SUMMARY_PATH = OUT_DIR / "fixed_hybridrank_score_summary.csv"
TIE_RULE_PATH = OUT_DIR / "future_alarm_tie_rule_registry.md"
SOURCE_REGISTRY_PATH = OUT_DIR / "fixed_hybridrank_source_registry.csv"
MANIFEST_PATH = OUT_DIR / "fixed_hybridrank_h5_manifest.json"
COMPLETION_REPORT_PATH = OUT_DIR / "fixed_hybridrank_h5_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "fixed_hybridrank_h5_checksums.sha256"

H = 5
THRESHOLDS = (15, 16, 17)
POLICY_NAME = "HybridRank_fixed_documented"
SUBMITTED_ALIAS = "hybrid_paper_fixed"
SCORE_CONTEXT = "retrospective_offline_only"
RANK_POPULATION = "within_outer_test_fold"
RANK_TIE_METHOD = "average"
RANK_DENOMINATOR = "N_minus_1"
FIXED_WEIGHT_SOURCE = str(ORIGINAL_FIXED_WEIGHTS_PATH)
FIXED_WEIGHTS = {
    "BCR-TCN v1.1": 0.50,
    "ElasticNet": 0.25,
    "Persistence": 0.25,
    "HGBR": 0.00,
}

PROTECTED_DIRS = [PROTOCOL_DIR, CONTROLLED_DIR, CANONICAL_DIR, FULLY_NESTED_DIR]
EXTERNAL_PROTECTED_FILES = [
    ORIGINAL_SCRIPT_PATH,
    ORIGINAL_RUN_CONFIG_PATH,
    ORIGINAL_FIXED_WEIGHTS_PATH,
    ORIGINAL_RANK_SCORES_PATH,
    ORIGINAL_HYBRID_PREDS_PATH,
    ORIGINAL_MODEL_REGISTRY_PATH,
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


def bool_to_str(v: bool) -> str:
    return "TRUE" if bool(v) else "FALSE"


def param_to_json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


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


def snapshot_file_checksums(paths: Sequence[Path]) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for p in paths:
        if not p.exists():
            raise RuntimeError("Protected file missing: {}".format(p))
        snap[str(p)] = sha256_file(p)
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


def rank_normalize_values(x: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    if np.isnan(vals).any():
        raise RuntimeError("Non-numeric or NaN raw scores encountered during rank normalization")
    n = len(vals)
    if n <= 1:
        return np.zeros(n, dtype=float)
    r = pd.Series(vals).rank(method="average", ascending=True).to_numpy(dtype=float)
    return (r - 1.0) / float(n - 1)


def compute_core_scores_from_raw_components(
    wide_df: pd.DataFrame,
    thresholds: Sequence[int],
    fixed_weights: Dict[str, float],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    bcr_col_map = {
        15: "bcr_tcn_v11_p_tau15",
        16: "bcr_tcn_v11_p_tau16",
        17: "bcr_tcn_v11_p_tau17",
    }

    for tau in thresholds:
        bcr_col = bcr_col_map[int(tau)]
        for fold, grp in wide_df.groupby("outer_fold", sort=True):
            g = grp.sort_values(["feature_date", "target_date"]).copy().reset_index(drop=True)

            bcr_rank = rank_normalize_values(g[bcr_col])
            enet_rank = rank_normalize_values(g["elasticnet_y_pred"])
            per_rank = rank_normalize_values(g["persistence_y_pred"])
            hgbr_rank = rank_normalize_values(g["hgbr_y_pred"])

            score = (
                float(fixed_weights["BCR-TCN v1.1"]) * bcr_rank
                + float(fixed_weights["ElasticNet"]) * enet_rank
                + float(fixed_weights["Persistence"]) * per_rank
                + float(fixed_weights["HGBR"]) * hgbr_rank
            )

            out = pd.DataFrame(
                {
                    "horizon": g["horizon"].astype(int),
                    "outer_fold": g["outer_fold"].astype(int),
                    "feature_date": g["feature_date"],
                    "target_date": g["target_date"],
                    "threshold": int(tau),
                    "bcr_tcn_raw_score": pd.to_numeric(g[bcr_col], errors="coerce").to_numpy(dtype=float),
                    "bcr_tcn_rank_score": bcr_rank,
                    "elasticnet_raw_score": pd.to_numeric(g["elasticnet_y_pred"], errors="coerce").to_numpy(dtype=float),
                    "elasticnet_rank_score": enet_rank,
                    "persistence_raw_score": pd.to_numeric(g["persistence_y_pred"], errors="coerce").to_numpy(dtype=float),
                    "persistence_rank_score": per_rank,
                    "hgbr_raw_score": pd.to_numeric(g["hgbr_y_pred"], errors="coerce").to_numpy(dtype=float),
                    "hgbr_rank_score": hgbr_rank,
                    "hybrid_score": score,
                }
            )
            rows.append(out)

    return pd.concat(rows, axis=0, ignore_index=True).sort_values(
        ["threshold", "outer_fold", "feature_date", "target_date"]
    ).reset_index(drop=True)


def compute_rank_audit(score_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    components = [
        ("BCR-TCN v1.1", "bcr_tcn_raw_score", "bcr_tcn_rank_score"),
        ("ElasticNet", "elasticnet_raw_score", "elasticnet_rank_score"),
        ("Persistence", "persistence_raw_score", "persistence_rank_score"),
        ("HGBR", "hgbr_raw_score", "hgbr_rank_score"),
    ]

    for fold in sorted(score_df["outer_fold"].astype(int).unique().tolist()):
        for tau in sorted(score_df["threshold"].astype(int).unique().tolist()):
            sf = score_df[
                (score_df["outer_fold"].astype(int) == int(fold))
                & (score_df["threshold"].astype(int) == int(tau))
            ].copy()
            for comp, raw_col, rank_col in components:
                raw_vals = pd.to_numeric(sf[raw_col], errors="coerce").to_numpy(dtype=float)
                rank_vals = pd.to_numeric(sf[rank_col], errors="coerce").to_numpy(dtype=float)

                n = int(len(raw_vals))
                raw_missing_count = int(np.isnan(raw_vals).sum())
                rank_missing_count = int(np.isnan(rank_vals).sum())

                raw_series = pd.Series(raw_vals)
                raw_counts = raw_series.value_counts(dropna=False)
                tie_group_count = int((raw_counts > 1).sum())
                maximum_tie_size = int(raw_counts.max()) if len(raw_counts) > 0 else 0

                order = np.argsort(raw_vals, kind="mergesort")
                rank_direction_pass = True
                if n > 1:
                    rank_direction_pass = bool(np.all(np.diff(rank_vals[order]) >= -1e-12))

                rank_range_pass = True
                if n > 0:
                    rank_range_pass = bool(
                        np.nanmin(rank_vals) >= -1e-12 and np.nanmax(rank_vals) <= 1.0 + 1e-12
                    )

                expected_rank = rank_normalize_values(pd.Series(raw_vals))
                rank_formula_pass = bool(np.allclose(rank_vals, expected_rank, atol=1e-12, rtol=0.0))

                audit_pass = bool(
                    n == 249
                    and raw_missing_count == 0
                    and rank_missing_count == 0
                    and rank_direction_pass
                    and rank_range_pass
                    and rank_formula_pass
                )

                rows.append(
                    {
                        "outer_fold": int(fold),
                        "threshold": int(tau),
                        "component": comp,
                        "N": int(n),
                        "raw_min": float(np.nanmin(raw_vals)) if n > 0 else np.nan,
                        "raw_max": float(np.nanmax(raw_vals)) if n > 0 else np.nan,
                        "raw_missing_count": int(raw_missing_count),
                        "raw_unique_count": int(pd.Series(raw_vals).nunique(dropna=True)),
                        "tie_group_count": int(tie_group_count),
                        "maximum_tie_size": int(maximum_tie_size),
                        "rank_min": float(np.nanmin(rank_vals)) if n > 0 else np.nan,
                        "rank_max": float(np.nanmax(rank_vals)) if n > 0 else np.nan,
                        "rank_missing_count": int(rank_missing_count),
                        "rank_direction_pass": bool(rank_direction_pass),
                        "rank_range_pass": bool(rank_range_pass),
                        "rank_formula_pass": bool(rank_formula_pass),
                        "y_true_dependency": False,
                        "event_label_dependency": False,
                        "audit_pass": bool(audit_pass),
                    }
                )

    return pd.DataFrame(rows).sort_values(["outer_fold", "threshold", "component"]).reset_index(drop=True)


def compute_label_invariance_audit(
    canonical_wide_df: pd.DataFrame,
    thresholds: Sequence[int],
    fixed_weights: Dict[str, float],
) -> Dict[str, Any]:
    base_core = compute_core_scores_from_raw_components(canonical_wide_df, thresholds, fixed_weights)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(canonical_wide_df))

    perm_df = canonical_wide_df.copy()
    for col in ["y_true", "event_tau15", "event_tau16", "event_tau17"]:
        perm_df[col] = canonical_wide_df[col].to_numpy()[perm]

    perm_core = compute_core_scores_from_raw_components(perm_df, thresholds, fixed_weights)

    key_cols = ["horizon", "outer_fold", "feature_date", "target_date", "threshold"]
    compare_cols = [
        "bcr_tcn_rank_score",
        "elasticnet_rank_score",
        "persistence_rank_score",
        "hgbr_rank_score",
        "hybrid_score",
    ]

    base_sorted = base_core.sort_values(key_cols).reset_index(drop=True)
    perm_sorted = perm_core.sort_values(key_cols).reset_index(drop=True)

    rank_identical = True
    for c in [
        "bcr_tcn_rank_score",
        "elasticnet_rank_score",
        "persistence_rank_score",
        "hgbr_rank_score",
    ]:
        rank_identical = rank_identical and bool(
            np.array_equal(
                base_sorted[c].to_numpy(dtype=float),
                perm_sorted[c].to_numpy(dtype=float),
            )
        )

    score_diff = np.abs(
        base_sorted["hybrid_score"].to_numpy(dtype=float)
        - perm_sorted["hybrid_score"].to_numpy(dtype=float)
    )
    score_identical = bool(np.array_equal(base_sorted["hybrid_score"].to_numpy(dtype=float), perm_sorted["hybrid_score"].to_numpy(dtype=float)))
    max_abs_diff = float(np.max(score_diff)) if len(score_diff) > 0 else 0.0

    return {
        "test_method": "permute_y_true_and_event_labels_in_memory_then_recompute_fixed_ranks_and_scores",
        "random_seed": 42,
        "rows_tested": int(len(base_sorted)),
        "thresholds_tested": [int(t) for t in sorted(thresholds)],
        "rank_values_identical": bool(rank_identical),
        "hybrid_scores_identical": bool(score_identical),
        "maximum_absolute_score_difference": float(max_abs_diff),
        "label_invariance_pass": bool(rank_identical and score_identical and abs(max_abs_diff) <= 0.0),
    }


def compute_score_summary(score_df: pd.DataFrame, canonical_wide_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    canonical_keys = canonical_wide_df[["outer_fold", "feature_date", "target_date"]].copy()

    for fold in sorted(score_df["outer_fold"].astype(int).unique().tolist()):
        canon_fold_keys = canonical_keys[canonical_keys["outer_fold"].astype(int) == int(fold)][
            ["feature_date", "target_date"]
        ].copy()

        for tau in sorted(score_df["threshold"].astype(int).unique().tolist()):
            sf = score_df[
                (score_df["outer_fold"].astype(int) == int(fold))
                & (score_df["threshold"].astype(int) == int(tau))
            ].copy()
            s = sf["hybrid_score"].to_numpy(dtype=float)

            score_counts = pd.Series(s).value_counts(dropna=False)
            tie_group_count = int((score_counts > 1).sum())
            maximum_tie_size = int(score_counts.max()) if len(score_counts) > 0 else 0

            sf_keys = sf[["feature_date", "target_date"]].copy()
            key_match = sf_keys.merge(canon_fold_keys, on=["feature_date", "target_date"], how="outer", indicator=True)
            canonical_key_pass = bool((key_match["_merge"] == "both").all())

            score_finite = bool(np.isfinite(s).all())
            score_range_pass = bool((s >= -1e-12).all() and (s <= 1.0 + 1e-12).all())

            rows.append(
                {
                    "outer_fold": int(fold),
                    "threshold": int(tau),
                    "N": int(len(sf)),
                    "score_min": float(np.min(s)) if len(s) > 0 else np.nan,
                    "score_max": float(np.max(s)) if len(s) > 0 else np.nan,
                    "score_mean": float(np.mean(s)) if len(s) > 0 else np.nan,
                    "score_std": float(np.std(s, ddof=0)) if len(s) > 0 else np.nan,
                    "score_unique_count": int(pd.Series(s).nunique(dropna=True)),
                    "tie_group_count": int(tie_group_count),
                    "maximum_tie_size": int(maximum_tie_size),
                    "score_finite": bool(score_finite),
                    "score_range_pass": bool(score_range_pass),
                    "canonical_key_pass": bool(canonical_key_pass),
                }
            )

    return pd.DataFrame(rows).sort_values(["outer_fold", "threshold"]).reset_index(drop=True)


def build_policy_definition_md() -> str:
    lines: List[str] = []
    lines.append("# Fixed Rank Ensemble Policy Definition (H5)")
    lines.append("")
    lines.append("Policy name: {}".format(POLICY_NAME))
    lines.append("Submitted alias: {}".format(SUBMITTED_ALIAS))
    lines.append("")
    lines.append("Components and fixed documented weights:")
    lines.append("- BCR-TCN v1.1 = 0.50")
    lines.append("- ElasticNet = 0.25")
    lines.append("- Persistence = 0.25")
    lines.append("- HGBR = 0.00")
    lines.append("")
    lines.append("Required policy properties:")
    lines.append("- weights are identical for all folds;")
    lines.append("- weights are identical for all thresholds;")
    lines.append("- the score itself is independent of alarm budget;")
    lines.append("- no corrected validation outcome was used to select weights;")
    lines.append("- no outer-test outcome was used to select weights;")
    lines.append("- no model was retrained;")
    lines.append("- HGBR remains in the documented component registry but contributes zero weight;")
    lines.append("- BCR-TCN contributes a threshold-specific probability;")
    lines.append("- regression components contribute point predictions;")
    lines.append("- score generation is retrospective because component ranks are normalized within the complete held-out outer-test fold;")
    lines.append("- this score is not a deployable sequential score;")
    lines.append("- a separate sequential policy will be created later.")
    lines.append("")
    lines.append("Preferred manuscript-facing term: fixed rank ensemble")
    lines.append("")
    lines.append("Equation for threshold tau:")
    lines.append("")
    lines.append("S_tau(t) =")
    lines.append("0.50 * RN(p_BCR-TCN,tau(t))")
    lines.append("+ 0.25 * RN(yhat_ElasticNet(t))")
    lines.append("+ 0.25 * RN(yhat_Persistence(t))")
    lines.append("+ 0.00 * RN(yhat_HGBR(t))")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_tie_rule_registry_md() -> str:
    lines: List[str] = []
    lines.append("# Future Alarm Tie Rule Registry")
    lines.append("")
    lines.append("The original submitted implementation did not define a deterministic secondary key for equal hybrid scores at a future top-k cutoff.")
    lines.append("")
    lines.append("This score-construction stage does not perform top-k selection.")
    lines.append("")
    lines.append("The later retrospective alarm-evaluation stage will use this revised deterministic ordering:")
    lines.append("1. descending risk score;")
    lines.append("2. ascending target_date;")
    lines.append("3. ascending feature_date;")
    lines.append("4. ascending canonical row identifier (outer_fold, feature_date, target_date).")
    lines.append("")
    lines.append("Explicit scope statement:")
    lines.append("- this rule is not used in the present score-construction stage;")
    lines.append("- this rule is a revision-era deterministic evaluation rule;")
    lines.append("- it is not claimed as an exact reproduction of the submitted implementation;")
    lines.append("- it does not use outcomes or event labels.")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_completion_report_md(
    run_id: str,
    git_commit: str,
    verification: Dict[str, Any],
    canonical_assembly_id: str,
    canonical_n: int,
    fully_nested_decision: str,
    fixed_provenance_ok: bool,
    rank_audit_df: pd.DataFrame,
    score_summary_df: pd.DataFrame,
    label_invariance: Dict[str, Any],
    source_preservation_pass: bool,
    final_decision: str,
) -> str:
    lines: List[str] = []
    lines.append("# Corrected Fixed H5 Rank Ensemble")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("- Construct corrected fixed H5 rank-ensemble scores from frozen canonical corrected component predictions only.")
    lines.append("")
    lines.append("## 2. Reason the Tuned HybridRank Pathway Was Not Retained")
    lines.append("- The tuned submitted pathway is not retained.")
    lines.append("- The fully nested tuned reconstruction stopped with Decision C because Fold 1 component validation had zero tau = 16 events.")
    lines.append("- The fixed documented vector was retained unchanged for this corrected fixed-policy stage.")
    lines.append("")
    lines.append("## 3. Input Verification")
    lines.append("- run_id: {}".format(run_id))
    lines.append("- git_commit: {}".format(git_commit))
    lines.append("- canonical_assembly_id: {}".format(canonical_assembly_id))
    lines.append("- canonical_N: {}".format(int(canonical_n)))
    lines.append("- fully_nested_feasibility_decision: {}".format(fully_nested_decision))
    lines.append("- input_verification_pass: {}".format(bool(verification.get("verification_pass", False))))
    lines.append("")
    lines.append("## 4. Canonical Corrected Component Scores")
    lines.append("- Score construction used only canonical_h5_predictions_wide.csv as corrected numerical input.")
    lines.append("- Submitted predictions were inspected for provenance only and were not used as corrected model inputs.")
    lines.append("")
    lines.append("## 5. Fixed Documented Weight Vector")
    lines.append("- policy_name: {}".format(POLICY_NAME))
    lines.append("- submitted_alias: {}".format(SUBMITTED_ALIAS))
    lines.append("- BCR-TCN v1.1 = 0.50")
    lines.append("- ElasticNet = 0.25")
    lines.append("- Persistence = 0.25")
    lines.append("- HGBR = 0.00")
    lines.append("- No corrected validation or outer-test outcome selected these weights.")
    lines.append("")
    lines.append("## 6. Threshold-Specific Component Inputs")
    lines.append("- tau 15 uses bcr_tcn_v11_p_tau15")
    lines.append("- tau 16 uses bcr_tcn_v11_p_tau16")
    lines.append("- tau 17 uses bcr_tcn_v11_p_tau17")
    lines.append("- ElasticNet, Persistence, and HGBR use their canonical point predictions for all thresholds.")
    lines.append("")
    lines.append("## 7. Fold-Wise Rank Normalization")
    lines.append("- Rank normalization is independent per outer fold, threshold, and component.")
    lines.append("- Ascending average ties and N-1 denominator were used.")
    lines.append("- Rank audit pass rate: {}/{} rows.".format(
        int(rank_audit_df["audit_pass"].astype(bool).sum()),
        int(len(rank_audit_df)),
    ))
    lines.append("")
    lines.append("## 8. Corrected Fixed Hybrid Scores")
    lines.append("- Output rows: {}".format(int(score_summary_df["N"].sum())))
    lines.append("- The fixed score is budget-independent.")
    lines.append("- The same fixed score may later be evaluated at r = 0.05 and r = 0.10.")
    lines.append("")
    lines.append("## 9. Label-Invariance Audit")
    lines.append("- rank_values_identical: {}".format(bool(label_invariance["rank_values_identical"])))
    lines.append("- hybrid_scores_identical: {}".format(bool(label_invariance["hybrid_scores_identical"])))
    lines.append("- maximum_absolute_score_difference: {}".format(label_invariance["maximum_absolute_score_difference"]))
    lines.append("- label_invariance_pass: {}".format(bool(label_invariance["label_invariance_pass"])))
    lines.append("")
    lines.append("## 10. Test-Outcome and Future-Fold Isolation")
    lines.append("- No validation or outer-test outcomes were used for weight selection.")
    lines.append("- No future-fold outcomes were used.")
    lines.append("- Source-preservation pass: {}".format(bool(source_preservation_pass)))
    lines.append("")
    lines.append("## 11. Difference from the Submitted Tuned Pathway")
    lines.append("- The tuned submitted pathway is not retained in this stage.")
    lines.append("- Evaluation at r = 0.10 is treated as corrected fixed-policy analysis, not the submitted tuned guarded policy.")
    lines.append("")
    lines.append("## 12. Retrospective-Only Interpretation")
    lines.append("- Within-test-fold rank normalization makes this score retrospective and non-deployable.")
    lines.append("- A separate sequential policy will be evaluated later.")
    lines.append("")
    lines.append("## 13. Determinism and Automated Tests")
    lines.append("- Automated tests are provided in test_fixed_hybridrank_h5_scores.py.")
    lines.append("- Determinism is validated by repeated generation in the test stage.")
    lines.append("")
    lines.append("## 14. Limitations")
    lines.append("1. The tuned submitted pathway is not retained after corrected chronology review.")
    lines.append("2. This stage is score-construction only; no alarm metrics were calculated in this stage.")
    lines.append("3. Fixed scores are retrospective and not a deployable sequential score.")
    lines.append("")
    lines.append("## 15. Readiness Decision")
    lines.append("- run_id: {}".format(run_id))
    lines.append("")
    lines.append("FINAL DECISION")
    lines.append("")
    lines.append("<!-- AUTO_DECISION_START -->")
    if final_decision == "A":
        lines.append("A. Corrected fixed H5 rank-ensemble scores passed; retrospective alarm-budget evaluation may begin.")
    elif final_decision == "B":
        lines.append("B. Fixed score construction completed, but one or more provenance or rank-reproduction issues require investigation.")
    elif final_decision == "C":
        lines.append("C. The fixed policy could not be reproduced safely from the available evidence.")
    else:
        lines.append("D. Canonical-input, source-preservation, label-invariance, or deterministic tests failed.")
    lines.append("<!-- AUTO_DECISION_END -->")
    lines.append("")
    return "\n".join(lines) + "\n"


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
    external_before = snapshot_file_checksums(EXTERNAL_PROTECTED_FILES)

    lock_sha_expected = expected_checksum_for_file(PROTOCOL_SHA_PATH, "canonical_dataset_lock.json")
    split_sha_expected = expected_checksum_for_file(PROTOCOL_SHA_PATH, "corrected_split_assignment.csv")

    canonical_long_sha_expected = expected_checksum_for_file(CANONICAL_CHECKSUMS_PATH, "canonical_h5_predictions_long.csv")
    canonical_wide_sha_expected = expected_checksum_for_file(CANONICAL_CHECKSUMS_PATH, "canonical_h5_predictions_wide.csv")

    lock_observed_sha = sha256_file(LOCK_PATH)
    split_observed_sha = sha256_file(SPLIT_PATH)
    canonical_long_observed_sha = sha256_file(CANONICAL_LONG_PATH)
    canonical_wide_observed_sha = sha256_file(CANONICAL_WIDE_PATH)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    dataset_path = Path(str(lock["absolute_path"]))
    dataset_sha_expected = str(lock["sha256"])
    if not dataset_path.exists():
        raise RuntimeError("Locked dataset path missing: {}".format(dataset_path))
    dataset_sha_observed = sha256_file(dataset_path)

    canonical_manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    canonical_report_decision = infer_decision_from_report(CANONICAL_REPORT_PATH)
    fully_nested_decision = infer_decision_from_report(FULLY_NESTED_REPORT_PATH)

    # Verify complete canonical and protocol manifests for source-preservation confidence.
    canonical_checks = verify_checksum_manifest(CANONICAL_CHECKSUMS_PATH, CANONICAL_DIR)
    protocol_checks = verify_checksum_manifest(PROTOCOL_SHA_PATH, PROTOCOL_DIR)

    long_df = pd.read_csv(CANONICAL_LONG_PATH)
    wide_df = pd.read_csv(CANONICAL_WIDE_PATH)

    for col in ["feature_date", "target_date"]:
        long_df[col] = normalize_date_col(long_df[col])
        wide_df[col] = normalize_date_col(wide_df[col])

    required_wide_cols = [
        "bcr_tcn_v11_p_tau15",
        "bcr_tcn_v11_p_tau16",
        "bcr_tcn_v11_p_tau17",
        "elasticnet_y_pred",
        "persistence_y_pred",
        "hgbr_y_pred",
    ]

    key_cols = ["horizon", "outer_fold", "feature_date", "target_date"]
    long_unique_keys = long_df[key_cols].drop_duplicates().copy()
    wide_unique_keys = wide_df[key_cols].drop_duplicates().copy()

    long_df["y_true"] = pd.to_numeric(long_df["y_true"], errors="coerce")
    if long_df["y_true"].isna().any():
        raise RuntimeError("canonical_h5_predictions_long.csv has NaN y_true values")

    ytrue_span = long_df.groupby(key_cols, dropna=False)["y_true"].agg(["min", "max"]).reset_index()
    ytrue_max_abs_diff = float(np.max(np.abs(ytrue_span["max"].to_numpy(dtype=float) - ytrue_span["min"].to_numpy(dtype=float))))
    long_y_true_consistent = bool(ytrue_max_abs_diff <= 1e-6)
    event_cols = ["event_tau15", "event_tau16", "event_tau17"]
    long_event_consistent = True
    for ec in event_cols:
        long_event_consistent = long_event_consistent and bool(
            long_df.groupby(key_cols, dropna=False)[ec].nunique(dropna=False).max() == 1
        )

    required_cols_present = bool(all(c in wide_df.columns for c in required_wide_cols))
    raw_finite_pass = True
    if required_cols_present:
        raw_finite_pass = bool(
            np.isfinite(wide_df[required_wide_cols].to_numpy(dtype=float)).all()
        )

    bcr_prob_bounds_pass = bool(
        (
            (wide_df["bcr_tcn_v11_p_tau15"].astype(float) >= 0.0)
            & (wide_df["bcr_tcn_v11_p_tau15"].astype(float) <= 1.0)
            & (wide_df["bcr_tcn_v11_p_tau16"].astype(float) >= 0.0)
            & (wide_df["bcr_tcn_v11_p_tau16"].astype(float) <= 1.0)
            & (wide_df["bcr_tcn_v11_p_tau17"].astype(float) >= 0.0)
            & (wide_df["bcr_tcn_v11_p_tau17"].astype(float) <= 1.0)
        ).all()
    )

    fixed_weights_provenance_df = pd.read_csv(ORIGINAL_FIXED_WEIGHTS_PATH)
    fixed_weights_observed = {
        "BCR-TCN v1.1": float(
            fixed_weights_provenance_df.loc[
                fixed_weights_provenance_df["component"].astype(str) == "tcn", "weight"
            ].iloc[0]
        ),
        "ElasticNet": float(
            fixed_weights_provenance_df.loc[
                fixed_weights_provenance_df["component"].astype(str) == "y_pred_enet", "weight"
            ].iloc[0]
        ),
        "Persistence": float(
            fixed_weights_provenance_df.loc[
                fixed_weights_provenance_df["component"].astype(str) == "y_pred_persist", "weight"
            ].iloc[0]
        ),
        "HGBR": float(
            fixed_weights_provenance_df.loc[
                fixed_weights_provenance_df["component"].astype(str) == "y_pred_hgbr", "weight"
            ].iloc[0]
        ),
    }
    fixed_vector_match = bool(
        abs(fixed_weights_observed["BCR-TCN v1.1"] - 0.50) <= 1e-12
        and abs(fixed_weights_observed["ElasticNet"] - 0.25) <= 1e-12
        and abs(fixed_weights_observed["Persistence"] - 0.25) <= 1e-12
        and abs(fixed_weights_observed["HGBR"] - 0.0) <= 1e-12
    )

    canonical_assembly_id = str(canonical_manifest.get("assembly_id", ""))
    if "canonical_assembly_id" not in wide_df.columns:
        raise RuntimeError("canonical_h5_predictions_wide.csv missing canonical_assembly_id")
    wide_assembly_ids = sorted(wide_df["canonical_assembly_id"].astype(str).unique().tolist())

    verification_checks = {
        "1_working_directory_pass": str(Path.cwd().resolve()) == str(ROOT.resolve()),
        "2_git_branch_pass": active_branch == EXPECTED_BRANCH,
        "3_working_tree_clean_stage_start_pass": len(outside_changes) == 0,
        "4_git_commit_present": True,
        "5_python_environment_present": True,
        "6_canonical_long_checksum_pass": canonical_long_observed_sha == canonical_long_sha_expected,
        "7_canonical_wide_checksum_pass": canonical_wide_observed_sha == canonical_wide_sha_expected,
        "8_canonical_assembly_decision_A_pass": canonical_report_decision == "A",
        "9_dataset_checksum_pass": dataset_sha_observed == dataset_sha_expected,
        "10_corrected_split_checksum_pass": split_observed_sha == split_sha_expected,
        "11_horizon_h5_only_pass": sorted(wide_df["horizon"].astype(int).unique().tolist()) == [H],
        "12_canonical_key_count_pass": int(len(wide_unique_keys)) == 747,
        "13_each_outer_fold_count_pass": bool((wide_df.groupby("outer_fold").size().astype(int) == 249).all()),
        "14_y_true_identical_across_component_models_pass": bool(long_y_true_consistent),
        "15_event_labels_identical_across_component_models_pass": bool(long_event_consistent),
        "16_required_canonical_component_columns_pass": bool(required_cols_present),
        "17_required_raw_component_scores_finite_pass": bool(raw_finite_pass),
        "18_bcr_probabilities_within_0_1_pass": bool(bcr_prob_bounds_pass),
        "19_fixed_vector_provenance_pass": bool(fixed_vector_match),
        "20_fully_nested_decision_C_pass": fully_nested_decision == "C",
    }

    git_commit = git_output(["git", "rev-parse", "HEAD"])
    python_executable = sys.executable
    python_version = platform.python_version()

    run_id = "fixed_hybridrank_h5_{}_{}_{}".format(
        git_commit[:12], dataset_sha_observed[:8], split_observed_sha[:8]
    )

    verification_pass = bool(all(bool(v) for v in verification_checks.values()))

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
        "canonical_long_file": str(CANONICAL_LONG_PATH),
        "canonical_long_sha256_expected": canonical_long_sha_expected,
        "canonical_long_sha256_observed": canonical_long_observed_sha,
        "canonical_wide_file": str(CANONICAL_WIDE_PATH),
        "canonical_wide_sha256_expected": canonical_wide_sha_expected,
        "canonical_wide_sha256_observed": canonical_wide_observed_sha,
        "canonical_assembly_manifest": str(CANONICAL_MANIFEST_PATH),
        "canonical_assembly_id": canonical_assembly_id,
        "canonical_assembly_decision": canonical_report_decision,
        "canonical_N": int(canonical_manifest.get("canonical_N", -1)),
        "canonical_wide_unique_assembly_ids": wide_assembly_ids,
        "protocol_checksums_file": str(PROTOCOL_SHA_PATH),
        "dataset_lock_file": str(LOCK_PATH),
        "dataset_path": str(dataset_path),
        "dataset_sha256_expected": dataset_sha_expected,
        "dataset_sha256_observed": dataset_sha_observed,
        "split_file": str(SPLIT_PATH),
        "split_sha256_expected": split_sha_expected,
        "split_sha256_observed": split_observed_sha,
        "horizon": int(H),
        "canonical_h5_key_count": int(len(wide_unique_keys)),
        "canonical_h5_rows_per_fold": {
            str(k): int(v)
            for k, v in wide_df.groupby("outer_fold").size().astype(int).to_dict().items()
        },
        "long_model_count": int(long_df["model"].nunique()),
        "long_models": sorted(long_df["model"].astype(str).unique().tolist()),
        "long_y_true_model_consistency_max_abs_diff": ytrue_max_abs_diff,
        "fixed_weight_provenance_file": str(ORIGINAL_FIXED_WEIGHTS_PATH),
        "fixed_weights_observed": fixed_weights_observed,
        "fully_nested_manifest": str(FULLY_NESTED_MANIFEST_PATH),
        "fully_nested_completion_report": str(FULLY_NESTED_REPORT_PATH),
        "fully_nested_fold1_support_diagnostic": str(FULLY_NESTED_FOLD1_DIAG_PATH),
        "fully_nested_checksum_registry": str(FULLY_NESTED_CHECKSUMS_PATH),
        "fully_nested_feasibility_decision": fully_nested_decision,
        "verification_checks": verification_checks,
        "verification_pass": bool(verification_pass),
        "protocol_checks": protocol_checks,
        "canonical_checks": canonical_checks,
        "timestamp": utc_now_iso(),
        "software_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    write_json(INPUT_VERIFICATION_PATH, input_verification)

    if not verification_pass:
        raise RuntimeError("Input verification failed; see {}".format(INPUT_VERIFICATION_PATH))

    policy_md = build_policy_definition_md()
    ensure_inside_authorized(POLICY_DEFINITION_PATH)
    POLICY_DEFINITION_PATH.write_text(policy_md, encoding="utf-8")

    # Build fixed weight registry.
    weights_rows: List[Dict[str, Any]] = []
    for fold in sorted(wide_df["outer_fold"].astype(int).unique().tolist()):
        for tau in THRESHOLDS:
            for comp, w in [
                ("BCR-TCN v1.1", 0.50),
                ("ElasticNet", 0.25),
                ("Persistence", 0.25),
                ("HGBR", 0.00),
            ]:
                weights_rows.append(
                    {
                        "policy_name": POLICY_NAME,
                        "submitted_alias": SUBMITTED_ALIAS,
                        "outer_fold": int(fold),
                        "threshold": int(tau),
                        "component": comp,
                        "weight": float(w),
                        "weight_source": FIXED_WEIGHT_SOURCE,
                        "weight_selection_performed": False,
                        "validation_outcomes_used": False,
                        "outer_test_outcomes_used": False,
                        "future_fold_outcomes_used": False,
                        "budget_specific": False,
                        "threshold_specific_weight": False,
                        "fold_specific_weight": False,
                        "dataset_sha256": dataset_sha_observed,
                        "split_sha256": split_observed_sha,
                        "canonical_assembly_id": canonical_assembly_id,
                        "run_id": run_id,
                    }
                )

    weights_df = pd.DataFrame(weights_rows).sort_values(
        ["outer_fold", "threshold", "component"]
    ).reset_index(drop=True)
    to_csv_with_dates(weights_df, WEIGHTS_PATH, [])

    # Construct fixed scores from canonical corrected component predictions only.
    core_scores = compute_core_scores_from_raw_components(wide_df, THRESHOLDS, FIXED_WEIGHTS)

    labels_and_meta = wide_df[
        [
            "horizon",
            "outer_fold",
            "feature_date",
            "target_date",
            "y_true",
            "event_tau15",
            "event_tau16",
            "event_tau17",
            "dataset_sha256",
            "split_sha256",
            "canonical_assembly_id",
        ]
    ].copy()

    score_df = core_scores.merge(
        labels_and_meta,
        on=["horizon", "outer_fold", "feature_date", "target_date"],
        how="left",
        validate="many_to_one",
    )

    score_df["policy_name"] = POLICY_NAME
    score_df["submitted_alias"] = SUBMITTED_ALIAS
    score_df["score_context"] = SCORE_CONTEXT
    score_df["bcr_tcn_weight"] = float(FIXED_WEIGHTS["BCR-TCN v1.1"])
    score_df["elasticnet_weight"] = float(FIXED_WEIGHTS["ElasticNet"])
    score_df["persistence_weight"] = float(FIXED_WEIGHTS["Persistence"])
    score_df["hgbr_weight"] = float(FIXED_WEIGHTS["HGBR"])
    score_df["component_weight_summary"] = "BCR-TCN v1.1=0.50|ElasticNet=0.25|Persistence=0.25|HGBR=0.00"
    score_df["rank_population"] = RANK_POPULATION
    score_df["rank_tie_method"] = RANK_TIE_METHOD
    score_df["rank_denominator"] = RANK_DENOMINATOR
    score_df["validation_outcomes_used"] = False
    score_df["outer_test_outcomes_used_for_weight_selection"] = False
    score_df["y_true_used_for_score_construction"] = False
    score_df["event_labels_used_for_score_construction"] = False
    score_df["run_id"] = run_id
    score_df["git_commit"] = git_commit

    score_columns = [
        "horizon",
        "outer_fold",
        "feature_date",
        "target_date",
        "y_true",
        "event_tau15",
        "event_tau16",
        "event_tau17",
        "threshold",
        "policy_name",
        "submitted_alias",
        "score_context",
        "hybrid_score",
        "bcr_tcn_raw_score",
        "bcr_tcn_rank_score",
        "elasticnet_raw_score",
        "elasticnet_rank_score",
        "persistence_raw_score",
        "persistence_rank_score",
        "hgbr_raw_score",
        "hgbr_rank_score",
        "bcr_tcn_weight",
        "elasticnet_weight",
        "persistence_weight",
        "hgbr_weight",
        "component_weight_summary",
        "rank_population",
        "rank_tie_method",
        "rank_denominator",
        "validation_outcomes_used",
        "outer_test_outcomes_used_for_weight_selection",
        "y_true_used_for_score_construction",
        "event_labels_used_for_score_construction",
        "dataset_sha256",
        "split_sha256",
        "canonical_assembly_id",
        "run_id",
        "git_commit",
    ]

    score_df = score_df[score_columns].sort_values(
        ["threshold", "outer_fold", "feature_date", "target_date"]
    ).reset_index(drop=True)

    if int(len(score_df)) != 2241:
        raise RuntimeError("Unexpected score row count: {} (expected 2241)".format(len(score_df)))

    to_csv_with_dates(score_df, SCORES_PATH, ["feature_date", "target_date"])

    rank_audit_df = compute_rank_audit(score_df)
    to_csv_with_dates(rank_audit_df, RANK_AUDIT_PATH, [])

    label_invariance = compute_label_invariance_audit(wide_df, THRESHOLDS, FIXED_WEIGHTS)
    write_json(LABEL_INVARIANCE_PATH, label_invariance)

    score_summary_df = compute_score_summary(score_df, wide_df)
    to_csv_with_dates(score_summary_df, SCORE_SUMMARY_PATH, [])

    tie_rule_md = build_tie_rule_registry_md()
    ensure_inside_authorized(TIE_RULE_PATH)
    TIE_RULE_PATH.write_text(tie_rule_md, encoding="utf-8")

    # Source registry (read-only lineage).
    source_rows: List[Dict[str, Any]] = []

    def add_source(
        source_type: str,
        path: Path,
        role: str,
        included_in_score: bool,
        notes: str,
    ) -> None:
        if not path.exists():
            raise RuntimeError("Source artifact missing: {}".format(path))
        source_rows.append(
            {
                "source_type": source_type,
                "source_path": str(path),
                "source_sha256": sha256_file(path),
                "source_role": role,
                "read_only": True,
                "numeric_values_modified": False,
                "included_in_score_construction": bool(included_in_score),
                "notes": notes,
            }
        )

    add_source("canonical", CANONICAL_WIDE_PATH, "corrected_component_score_input", True, "only corrected numerical source used for fixed score construction")
    add_source("canonical", CANONICAL_LONG_PATH, "cross_model_label_consistency_verification", False, "used to verify y_true/event label consistency across component models")
    add_source("canonical", CANONICAL_MANIFEST_PATH, "canonical_assembly_identity", False, "canonical assembly ID and canonical_N provenance")
    add_source("canonical", CANONICAL_SOURCE_REGISTRY_PATH, "canonical_source_lineage", False, "component source lineage and source decisions")
    add_source("protocol", LOCK_PATH, "dataset_lock_verification", False, "dataset checksum lock")
    add_source("submitted_source", ORIGINAL_SCRIPT_PATH, "submitted_hybridrank_implementation_reference", False, "read-only provenance inspection")
    add_source("submitted_run_provenance", ORIGINAL_RUN_CONFIG_PATH, "submitted_run_configuration", False, "read-only provenance inspection")
    add_source("submitted_run_provenance", ORIGINAL_FIXED_WEIGHTS_PATH, "primary_fixed_weight_provenance", False, "fixed documented vector provenance")
    add_source("submitted_run_provenance", ORIGINAL_RANK_SCORES_PATH, "submitted_rank_score_reference", False, "read-only provenance inspection")
    add_source("submitted_run_provenance", ORIGINAL_HYBRID_PREDS_PATH, "submitted_hybrid_prediction_reference", False, "read-only provenance inspection")
    add_source("submitted_run_provenance", ORIGINAL_MODEL_REGISTRY_PATH, "submitted_model_registry_reference", False, "submitted alias hybrid_paper_fixed provenance")
    add_source("fully_nested_feasibility", FULLY_NESTED_MANIFEST_PATH, "fully_nested_feasibility_manifest", False, "fully nested tuned pathway feasibility state")
    add_source("fully_nested_feasibility", FULLY_NESTED_REPORT_PATH, "fully_nested_feasibility_decision_report", False, "Decision C verification")
    add_source("fully_nested_feasibility", FULLY_NESTED_FOLD1_DIAG_PATH, "fold1_tau16_support_diagnostic", False, "documents zero tau16 events in Fold 1 component validation")
    add_source("fully_nested_feasibility", FULLY_NESTED_CHECKSUMS_PATH, "fully_nested_checksum_registry", False, "read-only checksum registry")

    source_registry_df = pd.DataFrame(source_rows)
    to_csv_with_dates(source_registry_df, SOURCE_REGISTRY_PATH, [])

    # Source-preservation check after all output writes.
    protected_after = snapshot_tree_checksums(PROTECTED_DIRS, ROOT)
    external_after = snapshot_file_checksums(EXTERNAL_PROTECTED_FILES)

    changed_protected = []
    for rel, before_sha in protected_before.items():
        if protected_after.get(rel) != before_sha:
            changed_protected.append(rel)

    changed_external = []
    for abs_path, before_sha in external_before.items():
        if external_after.get(abs_path) != before_sha:
            changed_external.append(abs_path)

    if changed_protected or changed_external:
        raise RuntimeError(
            "Protected source artifacts changed: protected={} external={}".format(
                changed_protected[:10], changed_external[:10]
            )
        )

    canonical_modified = bool(any(p.startswith("revision_2026/05_canonical_predictions/h5_cross_model/") for p in changed_protected))
    source_modified = bool(len(changed_protected) > 0 or len(changed_external) > 0)

    isolation_rows: List[Dict[str, Any]] = []
    for fold in sorted(score_df["outer_fold"].astype(int).unique().tolist()):
        canon_fold = wide_df[wide_df["outer_fold"].astype(int) == int(fold)].copy().sort_values(
            ["feature_date", "target_date"]
        )
        canon_keys_fold = canon_fold[["feature_date", "target_date"]].copy()

        for tau in THRESHOLDS:
            sf = score_df[
                (score_df["outer_fold"].astype(int) == int(fold))
                & (score_df["threshold"].astype(int) == int(tau))
            ].copy().sort_values(["feature_date", "target_date"])

            score_keys = sf[["feature_date", "target_date"]].copy()
            key_merge = score_keys.merge(canon_keys_fold, on=["feature_date", "target_date"], how="outer", indicator=True)
            component_keys_identical = bool((key_merge["_merge"] == "both").all()) and int(len(sf)) == int(len(canon_fold))

            bcr_col = "bcr_tcn_v11_p_tau{}".format(int(tau))
            raw_match = bool(
                np.allclose(sf["bcr_tcn_raw_score"].to_numpy(dtype=float), canon_fold[bcr_col].to_numpy(dtype=float), atol=0.0, rtol=0.0)
                and np.allclose(sf["elasticnet_raw_score"].to_numpy(dtype=float), canon_fold["elasticnet_y_pred"].to_numpy(dtype=float), atol=0.0, rtol=0.0)
                and np.allclose(sf["persistence_raw_score"].to_numpy(dtype=float), canon_fold["persistence_y_pred"].to_numpy(dtype=float), atol=0.0, rtol=0.0)
                and np.allclose(sf["hgbr_raw_score"].to_numpy(dtype=float), canon_fold["hgbr_y_pred"].to_numpy(dtype=float), atol=0.0, rtol=0.0)
            )

            isolation_rows.append(
                {
                    "outer_fold": int(fold),
                    "threshold": int(tau),
                    "canonical_rows": int(len(canon_fold)),
                    "component_keys_identical": bool(component_keys_identical),
                    "raw_scores_match_canonical": bool(raw_match),
                    "weights_fixed_before_score_construction": True,
                    "weight_selection_performed": False,
                    "validation_labels_used": False,
                    "outer_test_labels_used_for_weights": False,
                    "outer_test_labels_used_for_ranks": False,
                    "outer_test_labels_used_for_scores": False,
                    "future_fold_outcomes_used": False,
                    "rank_normalization_uses_complete_outer_test_fold": True,
                    "retrospective_only_label_present": bool((sf["score_context"].astype(str) == SCORE_CONTEXT).all()),
                    "source_files_modified": bool(source_modified),
                    "canonical_files_modified": bool(canonical_modified),
                    "isolation_pass": bool(
                        component_keys_identical
                        and raw_match
                        and (int(len(canon_fold)) == 249)
                        and not source_modified
                        and not canonical_modified
                    ),
                    "notes": "fixed rank ensemble; no label-dependent selection; no top-k evaluated",
                }
            )

    isolation_df = pd.DataFrame(isolation_rows).sort_values(["outer_fold", "threshold"]).reset_index(drop=True)
    to_csv_with_dates(isolation_df, ISOLATION_AUDIT_PATH, [])

    source_preservation_pass = bool(not source_modified and not canonical_modified)

    rank_audit_pass = bool(rank_audit_df["audit_pass"].astype(bool).all())
    summary_pass = bool(
        (score_summary_df["N"].astype(int) == 249).all()
        and score_summary_df["score_finite"].astype(bool).all()
        and score_summary_df["canonical_key_pass"].astype(bool).all()
    )
    isolation_pass = bool(isolation_df["isolation_pass"].astype(bool).all())
    label_invariance_pass = bool(label_invariance["label_invariance_pass"])

    final_decision = "A" if (rank_audit_pass and summary_pass and isolation_pass and label_invariance_pass and source_preservation_pass) else "B"

    # Initial completion report and manifest.
    completion_md = build_completion_report_md(
        run_id=run_id,
        git_commit=git_commit,
        verification=input_verification,
        canonical_assembly_id=canonical_assembly_id,
        canonical_n=int(canonical_manifest.get("canonical_N", -1)),
        fully_nested_decision=fully_nested_decision,
        fixed_provenance_ok=fixed_vector_match,
        rank_audit_df=rank_audit_df,
        score_summary_df=score_summary_df,
        label_invariance=label_invariance,
        source_preservation_pass=source_preservation_pass,
        final_decision=final_decision,
    )
    ensure_inside_authorized(COMPLETION_REPORT_PATH)
    COMPLETION_REPORT_PATH.write_text(completion_md, encoding="utf-8")

    source_checksums = {
        str(r["source_path"]): str(r["source_sha256"])
        for r in source_registry_df.to_dict("records")
    }

    manifest = {
        "run_id": run_id,
        "purpose": "Construct corrected fixed H5 rank-ensemble scores from frozen canonical corrected component predictions",
        "policy_name": POLICY_NAME,
        "submitted_alias": SUBMITTED_ALIAS,
        "horizon": int(H),
        "thresholds": [int(t) for t in THRESHOLDS],
        "fixed_weights": FIXED_WEIGHTS,
        "rank_normalization_rule": {
            "direction": "ascending",
            "tie_method": "average",
            "formula": "(average_rank-1)/(N-1)",
            "N_lte_1": "0",
        },
        "rank_population": RANK_POPULATION,
        "score_context": SCORE_CONTEXT,
        "canonical_assembly_id": canonical_assembly_id,
        "canonical_N": int(canonical_manifest.get("canonical_N", -1)),
        "dataset_sha256": dataset_sha_observed,
        "split_sha256": split_observed_sha,
        "protocol_tag": "corrected_protocol_v1",
        "fully_nested_feasibility_decision": fully_nested_decision,
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
        "source_files": [str(x) for x in source_registry_df["source_path"].astype(str).tolist()],
        "source_checksums": source_checksums,
        "output_files": [],
        "output_checksums": {},
        "label_invariance_result": label_invariance,
        "test_result": "not_run",
        "deterministic_result": "pending_test_script",
        "files_modified_outside_authorized_directory": outside_changes,
        "limitations": [
            "The tuned submitted pathway is not retained.",
            "Fully nested tuned reconstruction stopped with Decision C because Fold 1 component-validation had zero tau=16 events.",
            "This stage is fixed-policy score construction only and does not compute alarm metrics.",
            "Within-outer-test-fold rank normalization makes the score retrospective and non-deployable.",
            "A separate sequential policy will be evaluated later.",
        ],
        "source_preservation": {
            "protected_dirs_checked": [str(p) for p in PROTECTED_DIRS],
            "external_files_checked": [str(p) for p in EXTERNAL_PROTECTED_FILES],
            "changed_protected": changed_protected,
            "changed_external": changed_external,
            "pass": source_preservation_pass,
        },
        "input_verification_file": str(INPUT_VERIFICATION_PATH),
        "final_decision": final_decision,
    }

    # Populate output files/checksums excluding checksum file itself.
    output_checksums = compute_output_checksums(OUT_DIR, CHECKSUMS_PATH.name)
    manifest["output_files"] = sorted(output_checksums.keys())
    manifest["output_checksums"] = output_checksums

    write_json(MANIFEST_PATH, manifest)

    # Recompute output checksums once manifest is final, then write checksum registry.
    output_checksums = compute_output_checksums(OUT_DIR, CHECKSUMS_PATH.name)
    manifest["output_files"] = sorted(output_checksums.keys())
    manifest["output_checksums"] = output_checksums
    write_json(MANIFEST_PATH, manifest)

    write_checksums_file(OUT_DIR, CHECKSUMS_PATH)

    # Terminal summary (Part R).
    print("1. working directory: {}".format(Path.cwd().resolve()))
    print("2. active branch and Git commit: {} {}".format(active_branch, git_commit))
    print("3. Python executable and version: {} {}".format(python_executable, python_version))
    print("4. input-verification result: {}".format(bool(verification_pass)))
    print("5. canonical assembly ID and row count: {} {}".format(canonical_assembly_id, int(canonical_manifest.get("canonical_N", -1))))
    print("6. fully nested feasibility decision: {}".format(fully_nested_decision))
    print("7. fixed policy provenance: {}".format(ORIGINAL_FIXED_WEIGHTS_PATH))
    print("8. fixed weights: BCR-TCN v1.1=0.50 ElasticNet=0.25 Persistence=0.25 HGBR=0.00")
    print("9. thresholds processed: {}".format(list(THRESHOLDS)))
    print("10. score rows generated: {}".format(int(len(score_df))))
    print("11. rank-normalization audit: {}".format(rank_audit_pass))
    print("12. label-invariance result: {}".format(label_invariance_pass))
    print("13. source-preservation result: {}".format(source_preservation_pass))
    print("14. deterministic result: pending_test_script")
    print("15. number of passed assertions: pending_test_script")
    print("16. final decision: {}".format(final_decision))
    print("17. exactly one next action: Run the corrected retrospective top-k alarm-budget evaluation using the frozen canonical base-model scores and corrected fixed rank-ensemble scores.")


if __name__ == "__main__":
    main()
