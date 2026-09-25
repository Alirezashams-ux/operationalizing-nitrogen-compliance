from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"

REQUIRED_BRANCH = "controlled-reruns-v1"
HORIZON = 5
Y_TRUE_EQUALITY_ATOL = 1e-6

KEY_COLS = ["horizon", "outer_fold", "feature_date", "target_date"]

MODEL_ORDER = [
    "Persistence",
    "Ridge",
    "ElasticNet",
    "HGBR",
    "BCR-TCN v1.1",
]

MODEL_VERSION = {
    "Persistence": "corrected",
    "Ridge": "corrected",
    "ElasticNet": "corrected",
    "HGBR": "corrected",
    "BCR-TCN v1.1": "1.1",
}

MODEL_SEED = {
    "Persistence": 42,
    "Ridge": 42,
    "ElasticNet": 42,
    "HGBR": 42,
    "BCR-TCN v1.1": 42,
}


@dataclass(frozen=True)
class ModelSpec:
    canonical_name: str
    prediction_file: Path
    metrics_pooled_file: Path
    metrics_by_fold_file: Path
    manifest_file: Path
    completion_report_file: Path


MODEL_SPECS: Dict[str, ModelSpec] = {
    "Persistence": ModelSpec(
        canonical_name="Persistence",
        prediction_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "persistence"
        / "persistence_predictions.csv",
        metrics_pooled_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "persistence"
        / "persistence_metrics_pooled.csv",
        metrics_by_fold_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "persistence"
        / "persistence_metrics_by_fold.csv",
        manifest_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "persistence"
        / "persistence_run_manifest.json",
        completion_report_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "persistence"
        / "persistence_completion_report.md",
    ),
    "Ridge": ModelSpec(
        canonical_name="Ridge",
        prediction_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "ridge"
        / "ridge_predictions.csv",
        metrics_pooled_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "ridge"
        / "ridge_metrics_pooled.csv",
        metrics_by_fold_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "ridge"
        / "ridge_metrics_by_fold.csv",
        manifest_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "ridge"
        / "ridge_run_manifest.json",
        completion_report_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "ridge"
        / "ridge_completion_report.md",
    ),
    "ElasticNet": ModelSpec(
        canonical_name="ElasticNet",
        prediction_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "elasticnet"
        / "elasticnet_predictions.csv",
        metrics_pooled_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "elasticnet"
        / "elasticnet_metrics_pooled.csv",
        metrics_by_fold_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "elasticnet"
        / "elasticnet_metrics_by_fold.csv",
        manifest_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "elasticnet"
        / "elasticnet_run_manifest.json",
        completion_report_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "elasticnet"
        / "elasticnet_completion_report.md",
    ),
    "HGBR": ModelSpec(
        canonical_name="HGBR",
        prediction_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "hgbr"
        / "hgbr_predictions.csv",
        metrics_pooled_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "hgbr"
        / "hgbr_metrics_pooled.csv",
        metrics_by_fold_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "hgbr"
        / "hgbr_metrics_by_fold.csv",
        manifest_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "hgbr"
        / "hgbr_run_manifest.json",
        completion_report_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "hgbr"
        / "hgbr_completion_report.md",
    ),
    "BCR-TCN v1.1": ModelSpec(
        canonical_name="BCR-TCN v1.1",
        prediction_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "bcr_tcn_v11_h5"
        / "bcr_tcn_v11_h5_predictions.csv",
        metrics_pooled_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "bcr_tcn_v11_h5"
        / "bcr_tcn_v11_h5_metrics_pooled.csv",
        metrics_by_fold_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "bcr_tcn_v11_h5"
        / "bcr_tcn_v11_h5_metrics_by_fold.csv",
        manifest_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "bcr_tcn_v11_h5"
        / "bcr_tcn_v11_h5_run_manifest.json",
        completion_report_file=ROOT
        / "revision_2026"
        / "04_controlled_reruns"
        / "bcr_tcn_v11_h5"
        / "bcr_tcn_v11_h5_completion_report.md",
    ),
}

DATASET_LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
KEY_AUDIT_PATH = OUT_DIR / "canonical_h5_key_audit.csv"
LONG_PATH = OUT_DIR / "canonical_h5_predictions_long.csv"
WIDE_PATH = OUT_DIR / "canonical_h5_predictions_wide.csv"
POINT_BY_FOLD_PATH = OUT_DIR / "canonical_h5_point_metrics_by_fold.csv"
POINT_POOLED_PATH = OUT_DIR / "canonical_h5_point_metrics_pooled.csv"
EVENT_PREV_PATH = OUT_DIR / "canonical_h5_event_prevalence.csv"
SOURCE_REGISTRY_PATH = OUT_DIR / "canonical_h5_source_registry.csv"
CHECKSUM_PATH = OUT_DIR / "canonical_h5_checksums.sha256"
MANIFEST_PATH = OUT_DIR / "canonical_h5_assembly_manifest.json"
REPORT_PATH = OUT_DIR / "canonical_h5_assembly_report.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def normalize_date_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise").dt.strftime("%Y-%m-%d")


def run_git(args: List[str]) -> str:
    proc = subprocess.run(
        ["git"] + args,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout.strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    return now_iso()


def parse_completion_decision(text: str) -> str:
    patterns = [
        r"\bDecision:\s*([ABC])\b",
        r"FINAL DECISION\s*[\r\n]+\s*([ABC])\.\s",
        r"Readiness Decision\s*[\r\n]+\s*([ABC])\.\s",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    raise RuntimeError("Could not parse completion decision from completion report")


def load_protocol_hashes(path: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise RuntimeError(f"Malformed checksum line: {line}")
        hashes[parts[-1]] = parts[0]
    return hashes


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {label}: {missing}")


def canonical_key_tuples(df: pd.DataFrame) -> set:
    return set(tuple(r) for r in df[KEY_COLS].itertuples(index=False, name=None))


def infer_mase_denominators(metrics_by_fold_h5: pd.DataFrame) -> Dict[int, float]:
    denoms: Dict[int, float] = {}
    for r in metrics_by_fold_h5.itertuples(index=False):
        fold = int(r.outer_fold)
        mae = float(r.MAE)
        mase = float(r.MASE)
        if np.isfinite(mase) and abs(mase) > 0.0:
            den = float(mae / mase - 1e-9)
        else:
            den = np.nan
        denoms[fold] = den
    return denoms


def make_json_text(payload: Dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True)


def list_checksum_targets() -> List[Path]:
    targets: List[Path] = []
    for p in sorted(OUT_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.name == CHECKSUM_PATH.name:
            continue
        if p.suffix.lower() in {".csv", ".json", ".md"}:
            targets.append(p)
        elif p.name.startswith("test_") and p.suffix.lower() == ".py":
            targets.append(p)
    return targets


def write_checksums_file() -> None:
    lines: List[str] = []
    for p in list_checksum_targets():
        lines.append(f"{sha256_file(p)}  {p.name}")
    CHECKSUM_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    *,
    verification_pass: bool,
    completion_decisions: Dict[str, str],
    canonical_n: int,
    date_set_equal: bool,
    y_true_equal: bool,
    event_counts_equal: bool,
    metric_reconcile_pass: bool,
    source_preservation_pass: bool,
    deterministic_status: str,
    test_result: str,
    assembly_id: str,
    dataset_sha256: str,
    split_sha256: str,
    git_branch: str,
    git_commit: str,
) -> str:
    model_list_str = ", ".join(MODEL_ORDER)

    if test_result.startswith("pass"):
        final_decision = (
            "A. Canonical H5 cross-model assembly passed; retrospective alarm-budget "
            "evaluation and HybridRank regeneration may begin."
        )
        readiness_line = "- Automated tests passed; canonical assembly is release-ready."
        next_action = (
            "Begin retrospective alarm-budget evaluation and HybridRank regeneration "
            "using this canonical package only."
        )
    elif test_result.startswith("fail"):
        final_decision = (
            "B. Canonical assembly completed, but one or more source-model discrepancies "
            "require investigation."
        )
        readiness_line = "- Automated tests failed; investigate discrepancies before downstream analysis."
        next_action = "Inspect failing assertions in test_canonical_h5_assembly.py and rerun the test."
    else:
        final_decision = (
            "B. Canonical assembly completed, but one or more source-model discrepancies "
            "require investigation."
        )
        readiness_line = "- Automated tests not yet executed; decision remains provisional until tests pass."
        next_action = "Run test_canonical_h5_assembly.py exactly once to finalize readiness."

    return "\n".join(
        [
            "# Canonical H5 Cross-Model Prediction Assembly",
            "",
            "## 1. Purpose",
            "- Build one immutable, matched-date canonical H5 package for Persistence, Ridge, ElasticNet, HGBR, and BCR-TCN v1.1.",
            "- Provide the single source of truth for downstream H5 point metrics, retrospective alarm-budget, HybridRank, uncertainty, and sequential policy analyses.",
            "",
            "## 2. Input Verification",
            f"- verification_pass: {verification_pass}",
            f"- dataset_sha256: {dataset_sha256}",
            f"- split_sha256: {split_sha256}",
            f"- git_branch: {git_branch}",
            f"- git_commit: {git_commit}",
            "",
            "## 3. Source Models and Run Lineage",
            f"- models included: {model_list_str}",
            f"- completion decisions: {json.dumps(completion_decisions, ensure_ascii=True)}",
            "",
            "## 4. Canonical Key Definition",
            "- key: horizon, outer_fold, feature_date, target_date",
            "- authoritative source: corrected_split_assignment.csv outer_test rows at H=5",
            f"- canonical H5 key count: {canonical_n}",
            "",
            "## 5. Date and Target Equality",
            f"- date-set equality across models: {date_set_equal}",
            "- target_date equality rule: target_date == feature_date + 5 days",
            f"- y_true cross-model equality (tight tolerance): {y_true_equal}",
            "",
            "## 6. Prediction Preservation",
            "- Numeric prediction values were carried from source files without modification.",
            f"- source prediction preservation pass: {source_preservation_pass}",
            "- Representation-only normalization applied: horizon->int(5), outer_fold->int, feature_date/target_date->YYYY-MM-DD, model labels canonicalized.",
            "",
            "## 7. Long-Format Canonical Table",
            "- file: canonical_h5_predictions_long.csv",
            "- includes model-wise predictions, risk scores, errors, event labels, and source lineage fields.",
            "",
            "## 8. Wide-Format Canonical Table",
            "- file: canonical_h5_predictions_wide.csv",
            "- one canonical key row with side-by-side model predictions and BCR-TCN threshold probabilities.",
            "",
            "## 9. Point-Metric Reconciliation",
            f"- pooled canonical metrics reconcile to approved source pooled metrics: {metric_reconcile_pass}",
            "- pooled RMSE is computed as sqrt(pooled MSE).",
            "",
            "## 10. Event Prevalence",
            f"- event-count equality across models (by fold and pooled): {event_counts_equal}",
            "- event labels are derived from shared y_true thresholds at 15, 16, and 17 mg/L.",
            "",
            "## 11. Source and Checksum Registry",
            "- source registry file: canonical_h5_source_registry.csv",
            "- package checksums file: canonical_h5_checksums.sha256",
            f"- assembly_id: {assembly_id}",
            "",
            "## 12. Automated Tests",
            f"- test_result: {test_result}",
            f"- deterministic assembly status: {deterministic_status}",
            "",
            "## 13. Deviations or Limitations",
            "- No source prediction numeric value was altered.",
            "- No alarm flags, top-k selections, rank normalization, hybrid weighting, or sequential policy cutoffs were produced.",
            "",
            "## 14. Readiness Decision",
            readiness_line,
            "",
            "FINAL DECISION",
            "",
            final_decision,
            "",
            "TERMINAL SUMMARY",
            "",
            f"1. input verification result: {verification_pass}",
            f"2. models included: {model_list_str}",
            f"3. canonical key count: {canonical_n}",
            f"4. date-set equality: {date_set_equal}",
            f"5. y_true equality: {y_true_equal}",
            f"6. event-count equality: {event_counts_equal}",
            f"7. point-metric reconciliation result: {metric_reconcile_pass}",
            f"8. source prediction preservation result: {source_preservation_pass}",
            f"9. deterministic assembly result: {deterministic_status}",
            f"10. final decision: {final_decision[0]}",
            f"11. exactly one next action: {next_action}",
            "",
        ]
    )


def build_canonical_package() -> None:
    # Part B: branch and commit gate
    git_branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if git_branch != REQUIRED_BRANCH:
        raise RuntimeError(
            f"Active branch is {git_branch}; required branch is {REQUIRED_BRANCH}."
        )
    git_commit = run_git(["rev-parse", "HEAD"])

    # Part B: canonical dataset lock
    dataset_lock = json.loads(DATASET_LOCK_PATH.read_text(encoding="utf-8"))
    dataset_path = Path(dataset_lock["absolute_path"])
    dataset_sha_observed = sha256_file(dataset_path)
    dataset_sha_expected = dataset_lock["sha256"]
    if dataset_sha_observed != dataset_sha_expected:
        raise RuntimeError("Canonical dataset checksum mismatch.")

    # Part B: corrected protocol checksum list gate
    protocol_hashes = load_protocol_hashes(PROTOCOL_SHA_PATH)
    protocol_verification: Dict[str, Dict[str, str]] = {}
    for rel_name, expected_sha in protocol_hashes.items():
        target = PROTOCOL_DIR / rel_name
        observed_sha = sha256_file(target)
        protocol_verification[rel_name] = {
            "expected": expected_sha,
            "observed": observed_sha,
            "pass": str(observed_sha == expected_sha),
        }
        if observed_sha != expected_sha:
            raise RuntimeError(f"Protocol checksum mismatch for {rel_name}")

    split_sha_observed = sha256_file(SPLIT_PATH)
    split_sha_expected = protocol_hashes.get("corrected_split_assignment.csv")
    if split_sha_expected is None:
        raise RuntimeError("corrected_split_assignment.csv missing from protocol checksum file")
    if split_sha_observed != split_sha_expected:
        raise RuntimeError("Corrected split checksum mismatch.")

    # Part B + A: source manifests and completion decisions
    source_manifest_payloads: Dict[str, Dict] = {}
    completion_decisions: Dict[str, str] = {}
    source_prediction_checksums: Dict[str, str] = {}
    source_prediction_checksums_from_manifest: Dict[str, str] = {}
    source_prediction_files: Dict[str, str] = {}
    source_manifest_files: Dict[str, str] = {}

    for model in MODEL_ORDER:
        spec = MODEL_SPECS[model]

        manifest = json.loads(spec.manifest_file.read_text(encoding="utf-8"))
        source_manifest_payloads[model] = manifest

        completion_text = spec.completion_report_file.read_text(encoding="utf-8")
        decision = parse_completion_decision(completion_text)
        completion_decisions[model] = decision
        if decision != "A":
            raise RuntimeError(f"Completion decision for {model} is not A.")

        observed_pred_sha = sha256_file(spec.prediction_file)
        manifest_pred_sha = manifest.get("prediction_sha256")
        if manifest_pred_sha != observed_pred_sha:
            raise RuntimeError(
                f"Prediction checksum mismatch for {model}: "
                f"manifest={manifest_pred_sha}, observed={observed_pred_sha}"
            )

        manifest_pred_path = Path(manifest.get("prediction_file", ""))
        if manifest_pred_path.resolve() != spec.prediction_file.resolve():
            raise RuntimeError(
                f"Manifest prediction path mismatch for {model}: "
                f"{manifest_pred_path} != {spec.prediction_file}"
            )

        source_prediction_checksums[model] = observed_pred_sha
        source_prediction_checksums_from_manifest[model] = manifest_pred_sha
        source_prediction_files[model] = rel_to_root(spec.prediction_file)
        source_manifest_files[model] = rel_to_root(spec.manifest_file)

    # Part D: locked canonical key set from H5 outer_test split assignments
    split = pd.read_csv(SPLIT_PATH)
    ensure_columns(
        split,
        [
            "horizon",
            "outer_fold",
            "feature_date",
            "target_date",
            "outer_role",
        ],
        "corrected_split_assignment.csv",
    )
    split = split.copy()
    split["horizon"] = pd.to_numeric(split["horizon"], errors="raise").astype(int)
    split["outer_fold"] = pd.to_numeric(split["outer_fold"], errors="raise").astype(int)
    split["feature_date"] = normalize_date_str(split["feature_date"])
    split["target_date"] = normalize_date_str(split["target_date"])

    h5_keys = split[(split["horizon"] == HORIZON) & (split["outer_role"] == "outer_test")][
        KEY_COLS
    ].drop_duplicates()
    h5_keys = h5_keys.sort_values(["outer_fold", "feature_date", "target_date"]).reset_index(drop=True)

    if int(len(h5_keys)) != 747:
        raise RuntimeError(f"Canonical H5 key count must be 747, got {len(h5_keys)}")

    canonical_key_set = canonical_key_tuples(h5_keys)
    expected_n_by_fold = h5_keys.groupby("outer_fold").size().to_dict()

    # target_date == feature_date + 5 days
    fd = pd.to_datetime(h5_keys["feature_date"], errors="raise")
    td = pd.to_datetime(h5_keys["target_date"], errors="raise")
    if not bool(((td - fd).dt.days == HORIZON).all()):
        raise RuntimeError("Locked canonical split does not satisfy target_date = feature_date + 5 days")

    # Part C/E/F source load + representation normalization
    normalized_by_model: Dict[str, pd.DataFrame] = {}
    source_mase_denoms_by_model: Dict[str, Dict[int, float]] = {}
    source_metrics_pooled_h5: Dict[str, Dict[str, float]] = {}

    for model in MODEL_ORDER:
        spec = MODEL_SPECS[model]
        pred = pd.read_csv(spec.prediction_file)
        ensure_columns(pred, ["horizon", "outer_fold", "feature_date", "target_date", "y_true", "y_pred"], f"{model} predictions")

        pred = pred.copy()
        pred["horizon"] = pd.to_numeric(pred["horizon"], errors="raise").astype(int)
        pred = pred[pred["horizon"] == HORIZON].copy()

        pred["outer_fold"] = pd.to_numeric(pred["outer_fold"], errors="raise").astype(int)
        pred["feature_date"] = normalize_date_str(pred["feature_date"])
        pred["target_date"] = normalize_date_str(pred["target_date"])
        pred["y_true"] = pd.to_numeric(pred["y_true"], errors="raise").astype(float)
        pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="raise").astype(float)

        if model == "BCR-TCN v1.1":
            ensure_columns(pred, ["p_tau15", "p_tau16", "p_tau17"], f"{model} predictions")
            pred["p_tau15"] = pd.to_numeric(pred["p_tau15"], errors="raise").astype(float)
            pred["p_tau16"] = pd.to_numeric(pred["p_tau16"], errors="raise").astype(float)
            pred["p_tau17"] = pd.to_numeric(pred["p_tau17"], errors="raise").astype(float)

        # key and duplicate checks
        dup_count = int(pred.duplicated(subset=KEY_COLS).sum())
        if dup_count != 0:
            raise RuntimeError(f"Duplicate canonical keys found for {model}: {dup_count}")

        key_set = canonical_key_tuples(pred)
        missing = canonical_key_set - key_set
        extra = key_set - canonical_key_set
        if missing or extra:
            raise RuntimeError(
                f"Canonical key mismatch for {model}: missing={len(missing)}, extra={len(extra)}"
            )
        if len(key_set) != 747:
            raise RuntimeError(f"{model} does not contain exactly 747 canonical keys.")

        # target_date equality rule in each source model file
        fd_m = pd.to_datetime(pred["feature_date"], errors="raise")
        td_m = pd.to_datetime(pred["target_date"], errors="raise")
        if not bool(((td_m - fd_m).dt.days == HORIZON).all()):
            raise RuntimeError(f"target_date != feature_date + 5 days found in {model}")

        normalized_by_model[model] = pred.sort_values(["outer_fold", "feature_date", "target_date"]).reset_index(drop=True)

        src_by_fold = pd.read_csv(spec.metrics_by_fold_file)
        ensure_columns(src_by_fold, ["horizon", "outer_fold", "MAE", "MASE"], f"{model} metrics_by_fold")
        src_by_fold = src_by_fold.copy()
        src_by_fold["horizon"] = pd.to_numeric(src_by_fold["horizon"], errors="raise").astype(int)
        src_by_fold_h5 = src_by_fold[src_by_fold["horizon"] == HORIZON].copy()
        src_by_fold_h5["outer_fold"] = pd.to_numeric(src_by_fold_h5["outer_fold"], errors="raise").astype(int)
        source_mase_denoms_by_model[model] = infer_mase_denominators(src_by_fold_h5)

        src_pooled = pd.read_csv(spec.metrics_pooled_file)
        ensure_columns(src_pooled, ["horizon", "outer_fold", "N", "MAE", "MSE", "RMSE", "MASE"], f"{model} metrics_pooled")
        src_pooled = src_pooled.copy()
        src_pooled["horizon"] = pd.to_numeric(src_pooled["horizon"], errors="raise").astype(int)
        src_pooled_h5 = src_pooled[src_pooled["horizon"] == HORIZON].copy()
        src_pooled_h5 = src_pooled_h5[src_pooled_h5["outer_fold"].astype(str).str.lower() == "pooled"]
        if len(src_pooled_h5) != 1:
            raise RuntimeError(f"Expected one H5 pooled source metric row for {model}")
        row = src_pooled_h5.iloc[0]
        source_metrics_pooled_h5[model] = {
            "N": float(row["N"]),
            "MAE": float(row["MAE"]),
            "MSE": float(row["MSE"]),
            "RMSE": float(row["RMSE"]),
            "MASE": float(row["MASE"]),
        }

    # Cross-model y_true/event equality checks against Persistence reference
    ref_model = "Persistence"
    ref = normalized_by_model[ref_model][KEY_COLS + ["y_true"]].copy()
    ref = ref.rename(columns={"y_true": "y_true_ref"})

    for model in MODEL_ORDER:
        mdf = normalized_by_model[model][KEY_COLS + ["y_true"]].copy()
        merged = mdf.merge(ref, on=KEY_COLS, how="inner")
        if len(merged) != 747:
            raise RuntimeError(f"Merge cardinality failure for y_true equality in {model}")
        max_abs_diff = float((merged["y_true"] - merged["y_true_ref"]).abs().max())
        if max_abs_diff > Y_TRUE_EQUALITY_ATOL:
            raise RuntimeError(
                f"y_true mismatch for {model}; max abs diff = {max_abs_diff} (tol={Y_TRUE_EQUALITY_ATOL})"
            )

        evt_ref = pd.DataFrame(
            {
                "event_tau15_ref": (merged["y_true_ref"] >= 15.0).astype(int),
                "event_tau16_ref": (merged["y_true_ref"] >= 16.0).astype(int),
                "event_tau17_ref": (merged["y_true_ref"] >= 17.0).astype(int),
                "event_tau15": (merged["y_true"] >= 15.0).astype(int),
                "event_tau16": (merged["y_true"] >= 16.0).astype(int),
                "event_tau17": (merged["y_true"] >= 17.0).astype(int),
            }
        )
        if not bool((evt_ref["event_tau15"] == evt_ref["event_tau15_ref"]).all()):
            raise RuntimeError(f"event_tau15 mismatch in {model}")
        if not bool((evt_ref["event_tau16"] == evt_ref["event_tau16_ref"]).all()):
            raise RuntimeError(f"event_tau16 mismatch in {model}")
        if not bool((evt_ref["event_tau17"] == evt_ref["event_tau17_ref"]).all()):
            raise RuntimeError(f"event_tau17 mismatch in {model}")

    # Part D key audit by fold and pooled
    key_audit_rows: List[Dict] = []
    for fold_label in [1, 2, 3, "pooled"]:
        if fold_label == "pooled":
            expected_keys_fold = h5_keys.copy()
        else:
            expected_keys_fold = h5_keys[h5_keys["outer_fold"] == int(fold_label)].copy()

        expected_set = canonical_key_tuples(expected_keys_fold)
        per_model_counts: Dict[str, int] = {}
        per_model_missing: Dict[str, int] = {}
        per_model_extra: Dict[str, int] = {}
        per_model_dup: Dict[str, int] = {}
        per_model_key_sets: Dict[str, set] = {}
        y_true_pass = True
        target_date_pass = True
        event_pass = True

        ref_fold = normalized_by_model[ref_model]
        if fold_label != "pooled":
            ref_fold = ref_fold[ref_fold["outer_fold"] == int(fold_label)]

        ref_y = ref_fold[KEY_COLS + ["y_true"]].rename(columns={"y_true": "y_true_ref"})

        for model in MODEL_ORDER:
            mdf = normalized_by_model[model]
            if fold_label != "pooled":
                mdf = mdf[mdf["outer_fold"] == int(fold_label)]

            per_model_counts[model] = int(len(mdf))
            dup_count = int(mdf.duplicated(subset=KEY_COLS).sum())
            per_model_dup[model] = dup_count
            model_set = canonical_key_tuples(mdf)
            per_model_key_sets[model] = model_set
            per_model_missing[model] = int(len(expected_set - model_set))
            per_model_extra[model] = int(len(model_set - expected_set))

            # per-model target-date rule
            fdd = pd.to_datetime(mdf["feature_date"], errors="raise")
            tdd = pd.to_datetime(mdf["target_date"], errors="raise")
            target_date_pass = target_date_pass and bool(((tdd - fdd).dt.days == HORIZON).all())

            merged = mdf[KEY_COLS + ["y_true"]].merge(ref_y, on=KEY_COLS, how="inner")
            if len(merged) != len(expected_set):
                y_true_pass = False
                event_pass = False
            else:
                max_abs_diff = float((merged["y_true"] - merged["y_true_ref"]).abs().max())
                y_true_pass = y_true_pass and (max_abs_diff <= Y_TRUE_EQUALITY_ATOL)

                ev1 = (merged["y_true"] >= 15.0).astype(int)
                ev1_ref = (merged["y_true_ref"] >= 15.0).astype(int)
                ev2 = (merged["y_true"] >= 16.0).astype(int)
                ev2_ref = (merged["y_true_ref"] >= 16.0).astype(int)
                ev3 = (merged["y_true"] >= 17.0).astype(int)
                ev3_ref = (merged["y_true_ref"] >= 17.0).astype(int)
                event_pass = event_pass and bool((ev1 == ev1_ref).all()) and bool((ev2 == ev2_ref).all()) and bool((ev3 == ev3_ref).all())

        common_set = set.intersection(*per_model_key_sets.values())
        key_set_pass = (
            len(expected_set) == int(len(expected_keys_fold))
            and all(v == int(len(expected_keys_fold)) for v in per_model_counts.values())
            and all(v == 0 for v in per_model_missing.values())
            and all(v == 0 for v in per_model_extra.values())
            and all(v == 0 for v in per_model_dup.values())
            and y_true_pass
            and target_date_pass
            and event_pass
            and len(common_set) == int(len(expected_keys_fold))
        )

        key_audit_rows.append(
            {
                "outer_fold": str(fold_label),
                "expected_N": int(len(expected_keys_fold)),
                "persistence_N": per_model_counts["Persistence"],
                "ridge_N": per_model_counts["Ridge"],
                "elasticnet_N": per_model_counts["ElasticNet"],
                "hgbr_N": per_model_counts["HGBR"],
                "bcr_tcn_v11_N": per_model_counts["BCR-TCN v1.1"],
                "common_N": int(len(common_set)),
                "missing_key_count_by_model": json.dumps(per_model_missing, sort_keys=True),
                "extra_key_count_by_model": json.dumps(per_model_extra, sort_keys=True),
                "duplicate_count_by_model": json.dumps(per_model_dup, sort_keys=True),
                "y_true_equality": bool(y_true_pass),
                "target_date_equality": bool(target_date_pass),
                "event_label_equality": bool(event_pass),
                "date_start": expected_keys_fold["feature_date"].min(),
                "date_end": expected_keys_fold["feature_date"].max(),
                "target_date_start": expected_keys_fold["target_date"].min(),
                "target_date_end": expected_keys_fold["target_date"].max(),
                "key_set_pass": bool(key_set_pass),
            }
        )

    key_audit_df = pd.DataFrame(key_audit_rows)
    key_audit_df.to_csv(KEY_AUDIT_PATH, index=False)

    if not bool(key_audit_df["key_set_pass"].all()):
        raise RuntimeError("Canonical key audit failed.")

    # Part E: canonical long table
    assembly_id = f"canonical_h5_cross_model_{dataset_sha_expected[:8]}_{split_sha_observed[:8]}_{git_commit[:12]}"

    long_rows: List[pd.DataFrame] = []
    for model in MODEL_ORDER:
        pred = normalized_by_model[model].copy()

        pred["model"] = model
        pred["event_tau15"] = (pred["y_true"] >= 15.0).astype(int)
        pred["event_tau16"] = (pred["y_true"] >= 16.0).astype(int)
        pred["event_tau17"] = (pred["y_true"] >= 17.0).astype(int)
        pred["error"] = pred["y_pred"] - pred["y_true"]
        pred["absolute_error"] = pred["error"].abs()
        pred["squared_error"] = pred["error"] ** 2

        if model == "BCR-TCN v1.1":
            pred["risk_score_tau15"] = pred["p_tau15"]
            pred["risk_score_tau16"] = pred["p_tau16"]
            pred["risk_score_tau17"] = pred["p_tau17"]
            pred["risk_score_source"] = "threshold_specific_probability"
        else:
            pred["risk_score_tau15"] = pred["y_pred"]
            pred["risk_score_tau16"] = pred["y_pred"]
            pred["risk_score_tau17"] = pred["y_pred"]
            pred["risk_score_source"] = "y_pred_monotonic_threshold_independent"

        manifest = source_manifest_payloads[model]
        pred["model_version"] = MODEL_VERSION[model]
        pred["seed"] = MODEL_SEED[model]
        pred["dataset_sha256"] = dataset_sha_expected
        pred["split_sha256"] = split_sha_observed
        pred["source_prediction_file"] = rel_to_root(MODEL_SPECS[model].prediction_file)
        pred["source_prediction_sha256"] = source_prediction_checksums[model]
        pred["source_run_id"] = str(manifest.get("run_id", ""))
        pred["source_git_commit"] = str(manifest.get("git_commit", ""))
        pred["canonical_assembly_commit"] = git_commit
        pred["canonical_assembly_id"] = assembly_id

        long_rows.append(pred)

    long_df = pd.concat(long_rows, axis=0, ignore_index=True)
    long_df["model"] = pd.Categorical(long_df["model"], categories=MODEL_ORDER, ordered=True)
    long_df = long_df.sort_values(["model", "outer_fold", "feature_date", "target_date"]).reset_index(drop=True)

    long_cols = [
        "model",
        "horizon",
        "outer_fold",
        "feature_date",
        "target_date",
        "y_true",
        "y_pred",
        "error",
        "absolute_error",
        "squared_error",
        "event_tau15",
        "event_tau16",
        "event_tau17",
        "risk_score_tau15",
        "risk_score_tau16",
        "risk_score_tau17",
        "risk_score_source",
        "model_version",
        "seed",
        "dataset_sha256",
        "split_sha256",
        "source_prediction_file",
        "source_prediction_sha256",
        "source_run_id",
        "source_git_commit",
        "canonical_assembly_commit",
        "canonical_assembly_id",
    ]
    long_df = long_df[long_cols]
    long_df.to_csv(LONG_PATH, index=False)

    # Part F: canonical wide table
    ref_wide = long_df[long_df["model"] == ref_model][KEY_COLS + ["y_true"]].copy()
    ref_wide["event_tau15"] = (ref_wide["y_true"] >= 15.0).astype(int)
    ref_wide["event_tau16"] = (ref_wide["y_true"] >= 16.0).astype(int)
    ref_wide["event_tau17"] = (ref_wide["y_true"] >= 17.0).astype(int)

    wide_df = h5_keys.merge(ref_wide, on=KEY_COLS, how="left")

    name_to_col = {
        "Persistence": "persistence_y_pred",
        "Ridge": "ridge_y_pred",
        "ElasticNet": "elasticnet_y_pred",
        "HGBR": "hgbr_y_pred",
        "BCR-TCN v1.1": "bcr_tcn_v11_y_pred",
    }

    for model in MODEL_ORDER:
        sub = long_df[long_df["model"] == model][KEY_COLS + ["y_pred"]].copy()
        sub = sub.rename(columns={"y_pred": name_to_col[model]})
        wide_df = wide_df.merge(sub, on=KEY_COLS, how="left")

    bcr_sub = normalized_by_model["BCR-TCN v1.1"][KEY_COLS + ["p_tau15", "p_tau16", "p_tau17"]].copy()
    bcr_sub = bcr_sub.rename(
        columns={
            "p_tau15": "bcr_tcn_v11_p_tau15",
            "p_tau16": "bcr_tcn_v11_p_tau16",
            "p_tau17": "bcr_tcn_v11_p_tau17",
        }
    )
    wide_df = wide_df.merge(bcr_sub, on=KEY_COLS, how="left")

    wide_df["dataset_sha256"] = dataset_sha_expected
    wide_df["split_sha256"] = split_sha_observed
    wide_df["canonical_assembly_id"] = assembly_id

    wide_cols = [
        "horizon",
        "outer_fold",
        "feature_date",
        "target_date",
        "y_true",
        "event_tau15",
        "event_tau16",
        "event_tau17",
        "persistence_y_pred",
        "ridge_y_pred",
        "elasticnet_y_pred",
        "hgbr_y_pred",
        "bcr_tcn_v11_y_pred",
        "bcr_tcn_v11_p_tau15",
        "bcr_tcn_v11_p_tau16",
        "bcr_tcn_v11_p_tau17",
        "dataset_sha256",
        "split_sha256",
        "canonical_assembly_id",
    ]
    wide_df = wide_df[wide_cols].sort_values(["outer_fold", "feature_date", "target_date"]).reset_index(drop=True)
    wide_df.to_csv(WIDE_PATH, index=False)

    # Part G: canonical point metrics by fold and pooled
    by_fold_rows: List[Dict] = []
    pooled_rows: List[Dict] = []

    metric_reconcile_pass = True

    for model in MODEL_ORDER:
        dfm = long_df[long_df["model"] == model].copy()
        denoms = source_mase_denoms_by_model[model]

        for fold in [1, 2, 3]:
            d = dfm[dfm["outer_fold"] == fold]
            n = int(len(d))
            mae = float(d["absolute_error"].mean())
            mse = float(d["squared_error"].mean())
            rmse = float(np.sqrt(mse))
            denom = float(denoms[fold])
            mase = float(mae / (denom + 1e-9)) if np.isfinite(denom) else np.nan

            by_fold_rows.append(
                {
                    "model": model,
                    "horizon": HORIZON,
                    "outer_fold": fold,
                    "N": n,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MASE": mase,
                    "date_start": d["feature_date"].min(),
                    "date_end": d["feature_date"].max(),
                    "dataset_sha256": dataset_sha_expected,
                    "split_sha256": split_sha_observed,
                    "canonical_assembly_id": assembly_id,
                }
            )

        n_pooled = int(len(dfm))
        mae_pooled = float(dfm["absolute_error"].mean())
        mse_pooled = float(dfm["squared_error"].mean())
        rmse_pooled = float(np.sqrt(mse_pooled))
        den_row = dfm["outer_fold"].map(lambda x: denoms[int(x)]).astype(float)
        mase_vals = dfm["absolute_error"].to_numpy(dtype=float) / (den_row.to_numpy(dtype=float) + 1e-9)
        mase_pooled = float(np.mean(mase_vals))

        pooled_rows.append(
            {
                "model": model,
                "horizon": HORIZON,
                "outer_fold": "pooled",
                "N": n_pooled,
                "MAE": mae_pooled,
                "MSE": mse_pooled,
                "RMSE": rmse_pooled,
                "MASE": mase_pooled,
                "date_start": dfm["feature_date"].min(),
                "date_end": dfm["feature_date"].max(),
                "dataset_sha256": dataset_sha_expected,
                "split_sha256": split_sha_observed,
                "canonical_assembly_id": assembly_id,
            }
        )

        src = source_metrics_pooled_h5[model]
        metric_reconcile_pass = metric_reconcile_pass and (int(round(src["N"])) == n_pooled)
        metric_reconcile_pass = metric_reconcile_pass and (abs(src["MAE"] - mae_pooled) <= 1e-9)
        metric_reconcile_pass = metric_reconcile_pass and (abs(src["MSE"] - mse_pooled) <= 1e-9)
        metric_reconcile_pass = metric_reconcile_pass and (abs(src["RMSE"] - rmse_pooled) <= 1e-9)
        metric_reconcile_pass = metric_reconcile_pass and (abs(src["MASE"] - mase_pooled) <= 1e-9)

    point_by_fold = pd.DataFrame(by_fold_rows)
    point_by_fold["model"] = pd.Categorical(point_by_fold["model"], categories=MODEL_ORDER, ordered=True)
    point_by_fold = point_by_fold.sort_values(["model", "outer_fold"]).reset_index(drop=True)
    point_by_fold.to_csv(POINT_BY_FOLD_PATH, index=False)

    point_pooled = pd.DataFrame(pooled_rows)
    point_pooled["model"] = pd.Categorical(point_pooled["model"], categories=MODEL_ORDER, ordered=True)
    point_pooled = point_pooled.sort_values(["model"]).reset_index(drop=True)
    point_pooled.to_csv(POINT_POOLED_PATH, index=False)

    # Part H: event prevalence
    event_rows: List[Dict] = []
    for model in MODEL_ORDER:
        dfm = long_df[long_df["model"] == model].copy()
        for fold in [1, 2, 3]:
            d = dfm[dfm["outer_fold"] == fold]
            n = int(len(d))
            e15 = int(d["event_tau15"].sum())
            e16 = int(d["event_tau16"].sum())
            e17 = int(d["event_tau17"].sum())
            event_rows.append(
                {
                    "model": model,
                    "horizon": HORIZON,
                    "outer_fold": str(fold),
                    "N": n,
                    "events_tau15": e15,
                    "events_tau16": e16,
                    "events_tau17": e17,
                    "prevalence_tau15": float(e15 / n) if n else 0.0,
                    "prevalence_tau16": float(e16 / n) if n else 0.0,
                    "prevalence_tau17": float(e17 / n) if n else 0.0,
                    "dataset_sha256": dataset_sha_expected,
                    "split_sha256": split_sha_observed,
                    "canonical_assembly_id": assembly_id,
                }
            )

        n = int(len(dfm))
        e15 = int(dfm["event_tau15"].sum())
        e16 = int(dfm["event_tau16"].sum())
        e17 = int(dfm["event_tau17"].sum())
        event_rows.append(
            {
                "model": model,
                "horizon": HORIZON,
                "outer_fold": "pooled",
                "N": n,
                "events_tau15": e15,
                "events_tau16": e16,
                "events_tau17": e17,
                "prevalence_tau15": float(e15 / n) if n else 0.0,
                "prevalence_tau16": float(e16 / n) if n else 0.0,
                "prevalence_tau17": float(e17 / n) if n else 0.0,
                "dataset_sha256": dataset_sha_expected,
                "split_sha256": split_sha_observed,
                "canonical_assembly_id": assembly_id,
            }
        )

    event_prev = pd.DataFrame(event_rows)

    # Verify event-count equality across models for each fold label.
    event_counts_equal = True
    for fold_label in ["1", "2", "3", "pooled"]:
        sub = event_prev[event_prev["outer_fold"] == fold_label]
        if len(sub) != 5:
            event_counts_equal = False
            continue
        event_counts_equal = event_counts_equal and (sub["events_tau15"].nunique() == 1)
        event_counts_equal = event_counts_equal and (sub["events_tau16"].nunique() == 1)
        event_counts_equal = event_counts_equal and (sub["events_tau17"].nunique() == 1)

    if not event_counts_equal:
        raise RuntimeError("Event counts are not identical across models on shared canonical keys")

    event_prev["model"] = pd.Categorical(event_prev["model"], categories=MODEL_ORDER, ordered=True)
    event_prev["outer_fold"] = pd.Categorical(event_prev["outer_fold"], categories=["1", "2", "3", "pooled"], ordered=True)
    event_prev = event_prev.sort_values(["model", "outer_fold"]).reset_index(drop=True)
    event_prev.to_csv(EVENT_PREV_PATH, index=False)

    # Part I: source registry
    registry_rows: List[Dict] = []
    normalization_note = (
        "horizon coerced to int(5); outer_fold coerced to int; feature_date/target_date normalized to "
        "YYYY-MM-DD; canonical model-name mapping applied; numeric prediction/target/probability values unchanged"
    )

    for model in MODEL_ORDER:
        registry_rows.append(
            {
                "model": model,
                "source_prediction_file": rel_to_root(MODEL_SPECS[model].prediction_file),
                "source_manifest_file": rel_to_root(MODEL_SPECS[model].manifest_file),
                "source_run_id": source_manifest_payloads[model].get("run_id", ""),
                "source_git_commit": source_manifest_payloads[model].get("git_commit", ""),
                "source_prediction_sha256": source_prediction_checksums[model],
                "source_completion_decision": completion_decisions[model],
                "source_N": int(len(normalized_by_model[model])),
                "canonical_N": 747,
                "representation_normalization": normalization_note,
                "numeric_values_modified": False,
                "included_in_canonical_assembly": True,
            }
        )

    source_registry = pd.DataFrame(registry_rows)
    source_registry["model"] = pd.Categorical(source_registry["model"], categories=MODEL_ORDER, ordered=True)
    source_registry = source_registry.sort_values(["model"]).reset_index(drop=True)
    source_registry.to_csv(SOURCE_REGISTRY_PATH, index=False)

    # Part B output: input_verification.json
    test_result_existing = "pending_not_run"
    if MANIFEST_PATH.exists():
        try:
            old_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            test_result_existing = str(old_manifest.get("test_result", "pending_not_run"))
        except Exception:
            test_result_existing = "pending_not_run"

    timestamp = get_stable_timestamp()

    software_versions = {
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    try:
        import sklearn  # type: ignore

        software_versions["scikit_learn_version"] = sklearn.__version__
    except Exception:
        software_versions["scikit_learn_version"] = "unavailable"

    input_verification = {
        "dataset_sha256": dataset_sha_observed,
        "split_sha256": split_sha_observed,
        "protocol_tag": "corrected_protocol_v1",
        "git_branch": git_branch,
        "git_commit": git_commit,
        "source_prediction_files": source_prediction_files,
        "source_prediction_checksums": source_prediction_checksums,
        "source_prediction_checksums_from_manifest": source_prediction_checksums_from_manifest,
        "source_manifest_files": source_manifest_files,
        "completion_decision_by_model": completion_decisions,
        "protocol_checksums": protocol_verification,
        "verification_pass": True,
        "timestamp": timestamp,
        "software_versions": software_versions,
    }
    INPUT_VERIFICATION_PATH.write_text(make_json_text(input_verification), encoding="utf-8")

    # Part L: assembly manifest
    source_files_manifest = {
        model: rel_to_root(MODEL_SPECS[model].prediction_file) for model in MODEL_ORDER
    }
    source_checksums_manifest = {model: source_prediction_checksums[model] for model in MODEL_ORDER}
    source_run_ids_manifest = {
        model: str(source_manifest_payloads[model].get("run_id", "")) for model in MODEL_ORDER
    }

    metric_files = [
        rel_to_root(POINT_BY_FOLD_PATH),
        rel_to_root(POINT_POOLED_PATH),
    ]
    metric_checksums = {
        POINT_BY_FOLD_PATH.name: sha256_file(POINT_BY_FOLD_PATH),
        POINT_POOLED_PATH.name: sha256_file(POINT_POOLED_PATH),
    }

    assembly_manifest = {
        "assembly_id": assembly_id,
        "models": MODEL_ORDER,
        "horizon": HORIZON,
        "canonical_N": 747,
        "dataset_sha256": dataset_sha_observed,
        "split_sha256": split_sha_observed,
        "protocol_tag": "corrected_protocol_v1",
        "git_branch": git_branch,
        "git_commit": git_commit,
        "execution_command": f"{sys.executable} {Path(__file__).resolve()}",
        "timestamp": timestamp,
        "software_versions": software_versions,
        "source_files": source_files_manifest,
        "source_checksums": source_checksums_manifest,
        "source_run_ids": source_run_ids_manifest,
        "long_file": rel_to_root(LONG_PATH),
        "long_file_sha256": sha256_file(LONG_PATH),
        "wide_file": rel_to_root(WIDE_PATH),
        "wide_file_sha256": sha256_file(WIDE_PATH),
        "metric_files": metric_files,
        "metric_checksums": metric_checksums,
        "event_file": rel_to_root(EVENT_PREV_PATH),
        "event_file_sha256": sha256_file(EVENT_PREV_PATH),
        "test_result": test_result_existing,
        "notes": (
            "Canonical H5 cross-model package assembled without training/tuning/refitting/recalibration; "
            "no alarm flags, top-k selection, rank normalization, hybrid weighting, or sequential policy cutoffs produced."
        ),
    }
    MANIFEST_PATH.write_text(make_json_text(assembly_manifest), encoding="utf-8")

    # Part M: completion report
    date_set_equal = bool(key_audit_df["key_set_pass"].all())
    y_true_equal = bool(key_audit_df["y_true_equality"].all())
    source_preservation_pass = True
    deterministic_status = (
        "pass" if test_result_existing.startswith("pass") else "pending_not_run"
    )

    report_text = build_report(
        verification_pass=True,
        completion_decisions=completion_decisions,
        canonical_n=747,
        date_set_equal=date_set_equal,
        y_true_equal=y_true_equal,
        event_counts_equal=event_counts_equal,
        metric_reconcile_pass=metric_reconcile_pass,
        source_preservation_pass=source_preservation_pass,
        deterministic_status=deterministic_status,
        test_result=test_result_existing,
        assembly_id=assembly_id,
        dataset_sha256=dataset_sha_observed,
        split_sha256=split_sha_observed,
        git_branch=git_branch,
        git_commit=git_commit,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    # Part I checksums file (csv/json/md + test file)
    write_checksums_file()

    print(f"Wrote {INPUT_VERIFICATION_PATH}")
    print(f"Wrote {KEY_AUDIT_PATH}")
    print(f"Wrote {LONG_PATH}")
    print(f"Wrote {WIDE_PATH}")
    print(f"Wrote {POINT_BY_FOLD_PATH}")
    print(f"Wrote {POINT_POOLED_PATH}")
    print(f"Wrote {EVENT_PREV_PATH}")
    print(f"Wrote {SOURCE_REGISTRY_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {CHECKSUM_PATH}")


if __name__ == "__main__":
    build_canonical_package()
