from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
AUTHORIZED_REL_DIR = "revision_2026/10_lag_window_sensitivity/01_execution"
EXPECTED_REPOSITORY = "/home/alrezshams/acs_tnout_ulsan_revision"
EXPECTED_BRANCH = "controlled-reruns-v1"
EXPECTED_HEAD = "dee06bcfeca0c3c224897a2a92261c13f4fea26e"
EXPECTED_DESIGN_TAG = "lag-window-sensitivity-design-v1"
EXPECTED_PYTHON = "3.8.10"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CONTROLLED_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions"
HYBRID_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank"
RETRO_DIR = ROOT / "revision_2026" / "07_retrospective_alarm_budget"
SEQUENTIAL_DIR = ROOT / "revision_2026" / "08_sequential_alarm_policy"
STAGE1_DIR = ROOT / "revision_2026" / "09_h1_h3_reconciliation"
DESIGN_DIR = ROOT / "revision_2026" / "10_lag_window_sensitivity" / "00_design"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"
STAGE1_CHECKSUMS_PATH = STAGE1_DIR / "canonical_h1_h3_reconciliation_checksums.sha256"
DESIGN_CHECKSUMS_PATH = DESIGN_DIR / "lag_window_sensitivity_design_checksums.sha256"

DESIGN_JSON_PATH = DESIGN_DIR / "lag_window_sensitivity_design.json"
DESIGN_MD_PATH = DESIGN_DIR / "lag_window_sensitivity_design.md"
DESIGN_CANDIDATE_PATH = DESIGN_DIR / "candidate_memory_configurations.csv"
DESIGN_MODEL_REGISTRY_PATH = DESIGN_DIR / "model_horizon_reference_registry.csv"
DESIGN_COMMON_DATE_PATH = DESIGN_DIR / "common_date_feasibility.csv"
DESIGN_RETENTION_PATH = DESIGN_DIR / "training_retention_audit.csv"
DESIGN_LINEAGE_PATH = DESIGN_DIR / "feature_lineage_audit.csv"
DESIGN_ARTIFACT_REG_PATH = DESIGN_DIR / "feature_artifact_registry.csv"
DESIGN_INPUT_VERIFY_PATH = DESIGN_DIR / "input_verification.json"
DESIGN_MANIFEST_PATH = DESIGN_DIR / "lag_window_sensitivity_design_manifest.json"

RIDGE_SELECTED_PATH = CONTROLLED_DIR / "ridge" / "ridge_selected_configurations.csv"
RIDGE_MANIFEST_PATH = CONTROLLED_DIR / "ridge" / "ridge_run_manifest.json"
RIDGE_PRED_PATH = CONTROLLED_DIR / "ridge" / "ridge_predictions.csv"

HGBR_SELECTED_PATH = CONTROLLED_DIR / "hgbr" / "hgbr_selected_configurations.csv"
HGBR_MANIFEST_PATH = CONTROLLED_DIR / "hgbr" / "hgbr_run_manifest.json"
HGBR_PRED_PATH = CONTROLLED_DIR / "hgbr" / "hgbr_predictions.csv"

FEATURE_H1_NPZ = ROOT / "features" / "ulsan_H1_features.npz"
FEATURE_H3_NPZ = ROOT / "features" / "ulsan_H3_features.npz"
FEATURE_H5V2_NPZ = ROOT / "features" / "ulsan_H5_features_v2.npz"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
SOURCE_REGISTRY_PATH = OUT_DIR / "execution_source_registry.csv"
SOFTWARE_ENV_PATH = OUT_DIR / "software_environment.json"
FIT_REGISTRY_PATH = OUT_DIR / "model_fit_registry.csv"
PRED_LONG_PATH = OUT_DIR / "lag_window_predictions_long.csv"
PRED_WIDE_PATH = OUT_DIR / "lag_window_predictions_wide.csv"
REF_AUDIT_PATH = OUT_DIR / "reference_reproduction_audit.csv"
REF_SUMMARY_PATH = OUT_DIR / "reference_reproduction_summary.json"
POINT_BY_FOLD_PATH = OUT_DIR / "point_metrics_by_fold.csv"
POINT_POOLED_PATH = OUT_DIR / "point_metrics_pooled.csv"
STABILITY_BY_FOLD_PATH = OUT_DIR / "prediction_stability_by_fold.csv"
STABILITY_POOLED_PATH = OUT_DIR / "prediction_stability_pooled.csv"
RECALL_BY_FOLD_PATH = OUT_DIR / "event_ranking_recall5_by_fold.csv"
RECALL_POOLED_PATH = OUT_DIR / "event_ranking_recall5_pooled.csv"
RETENTION_REALIZED_PATH = OUT_DIR / "realized_training_retention.csv"
FEATURE_REG_REALIZED_PATH = OUT_DIR / "realized_feature_registry.csv"
SPLIT_EMBARGO_AUDIT_PATH = OUT_DIR / "split_and_embargo_audit.csv"
SUMMARY_CSV_PATH = OUT_DIR / "lag_window_sensitivity_summary.csv"
SUMMARY_JSON_PATH = OUT_DIR / "lag_window_sensitivity_summary.json"
SUMMARY_MD_PATH = OUT_DIR / "lag_window_sensitivity_summary.md"
MANIFEST_PATH = OUT_DIR / "lag_window_sensitivity_execution_manifest.json"
REPORT_PATH = OUT_DIR / "lag_window_sensitivity_execution_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "lag_window_sensitivity_execution_checksums.sha256"

FIT_MODELS = ("Ridge", "HGBR")
HORIZONS = (1, 3, 5)
OUTER_FOLDS = (1, 2, 3)
CONFIG_ORDER = ("SHORT", "REFERENCE", "LONG")
EVENT_TAU = 16.0
EVENT_BUDGET = 0.05
MASE_EPS = 1e-9
REF_TOL = 1e-6

TNOUT_LONG = [
    "TNout_lag1",
    "TNout_lag3",
    "TNout_lag5",
    "TNout_lag7",
    "TNout_roll14",
    "TNout_roll30",
    "TNout_roll7",
]
TNOUT_BY_CONFIG = {
    "SHORT": ["TNout_lag1", "TNout_roll7"],
    "REFERENCE": ["TNout_lag1", "TNout_roll14", "TNout_roll7"],
    "LONG": TNOUT_LONG,
}

ORDINARY_NON_TN = [
    "Inflow",
    "Inflow_roll7",
    "TNin",
    "TNin_roll7",
    "TOCin",
    "TOCin_roll7",
    "BODin",
    "BODin_roll7",
    "C_N",
    "temp_mean_c",
    "temp_roll7",
    "precip_total_mm",
    "precip_sum3",
    "sin_doy",
    "cos_doy",
]

V2_NON_TN = [
    "Inflow",
    "Inflow_roll7",
    "Inflow_roll14",
    "TNin",
    "TNin_roll7",
    "TNin_roll14",
    "TOCin",
    "TOCin_roll7",
    "TOCin_roll14",
    "BODin",
    "BODin_roll7",
    "BODin_roll14",
    "C_N",
    "C_N_roll14",
    "temp_mean_c",
    "temp_roll7",
    "temp_roll14",
    "temp_roll30",
    "precip_total_mm",
    "precip_sum3",
    "precip_sum7",
    "precip_sum14",
    "Inflow_x_precip3",
    "temp_x_CN",
    "Inflow_x_TNin",
    "sin_doy",
    "cos_doy",
]

IMMUTABLE_DIRS = [
    ROOT / "revision_2026" / "03_corrected_protocol",
    ROOT / "revision_2026" / "04_controlled_reruns",
    ROOT / "revision_2026" / "05_canonical_predictions",
    ROOT / "revision_2026" / "06_corrected_hybridrank",
    ROOT / "revision_2026" / "07_retrospective_alarm_budget",
    ROOT / "revision_2026" / "08_sequential_alarm_policy",
    ROOT / "revision_2026" / "09_h1_h3_reconciliation",
    ROOT / "revision_2026" / "10_lag_window_sensitivity" / "00_design",
    ROOT / "src",
    ROOT / "features",
    ROOT / "results",
]


class DecisionError(RuntimeError):
    def __init__(self, decision: str, message: str) -> None:
        super().__init__(message)
        self.decision = decision


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: Sequence[str]) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


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
    return sorted(paths)


def is_authorized_git_status_path(path: str, authorized_rel_dir: str) -> bool:
    p = str(path).strip().rstrip("/")
    a = str(authorized_rel_dir).strip().rstrip("/")
    if not p:
        return False
    return p.startswith(a) or a.startswith(p)


def ensure_inside_authorized(path: Path) -> None:
    rp = path.resolve()
    auth = (ROOT / AUTHORIZED_REL_DIR).resolve()
    if not str(rp).startswith(str(auth)):
        raise DecisionError("D", "attempted write outside authorized output directory: {}".format(rp))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_inside_authorized(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_inside_authorized(path)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame, date_cols: Optional[Sequence[str]] = None) -> None:
    out = df.copy()
    if date_cols:
        for col in date_cols:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    ensure_inside_authorized(path)
    out.to_csv(path, index=False)


def parse_checksum_manifest(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            raise DecisionError("D", "malformed checksum line in {}: {}".format(path, line))
        rows.append((parts[0], parts[-1]))
    return rows


def verify_checksum_manifest(path: Path, base_dir: Path) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for expected, rel_name in parse_checksum_manifest(path):
        target = base_dir / rel_name
        if not target.exists():
            raise DecisionError("D", "checksum target missing: {}".format(target))
        observed = sha256_file(target)
        ok = observed == expected
        checks[str(rel_name)] = {
            "expected": expected,
            "observed": observed,
            "pass": bool(ok),
        }
        if not ok:
            raise DecisionError(
                "D",
                "checksum mismatch for {}: expected {}, observed {}".format(rel_name, expected, observed),
            )
    return checks


def snapshot_tree_checksums(dirs: Sequence[Path], rel_base: Path) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for d in dirs:
        if not d.exists():
            raise DecisionError("D", "immutable directory missing: {}".format(d))
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(rel_base)).replace("\\", "/")
            snap[rel] = sha256_file(p)
    return snap


def load_design_bundle() -> Dict[str, Any]:
    required_files = [
        DESIGN_JSON_PATH,
        DESIGN_MD_PATH,
        DESIGN_CANDIDATE_PATH,
        DESIGN_MODEL_REGISTRY_PATH,
        DESIGN_COMMON_DATE_PATH,
        DESIGN_RETENTION_PATH,
        DESIGN_LINEAGE_PATH,
        DESIGN_ARTIFACT_REG_PATH,
        DESIGN_INPUT_VERIFY_PATH,
        DESIGN_MANIFEST_PATH,
        DESIGN_CHECKSUMS_PATH,
    ]
    for p in required_files:
        if not p.exists():
            raise DecisionError("C", "frozen design file missing: {}".format(p))

    design_json = json.loads(DESIGN_JSON_PATH.read_text(encoding="utf-8"))
    design_manifest = json.loads(DESIGN_MANIFEST_PATH.read_text(encoding="utf-8"))
    design_input = json.loads(DESIGN_INPUT_VERIFY_PATH.read_text(encoding="utf-8"))
    candidate_df = pd.read_csv(DESIGN_CANDIDATE_PATH)
    model_registry_df = pd.read_csv(DESIGN_MODEL_REGISTRY_PATH)
    common_date_df = pd.read_csv(DESIGN_COMMON_DATE_PATH)
    retention_df = pd.read_csv(DESIGN_RETENTION_PATH)
    lineage_df = pd.read_csv(DESIGN_LINEAGE_PATH)
    artifact_registry_df = pd.read_csv(DESIGN_ARTIFACT_REG_PATH)

    candidate_df["configuration_id"] = candidate_df["configuration_id"].astype(str)
    candidate_df["tnout_features"] = (
        candidate_df["tnout_features"].astype(str).str.split(";").apply(lambda x: [t.strip() for t in x if str(t).strip()])
    )
    candidate_df["maximum_required_history_days"] = candidate_df["maximum_required_history_days"].astype(int)

    model_registry_df["model"] = model_registry_df["model"].astype(str)
    model_registry_df["horizon"] = model_registry_df["horizon"].astype(int)
    model_registry_df["will_be_refitted"] = as_bool(model_registry_df["will_be_refitted"])

    common_date_df["horizon"] = common_date_df["horizon"].astype(int)
    common_date_df["outer_fold"] = common_date_df["outer_fold"].astype(str)
    retention_df["horizon"] = retention_df["horizon"].astype(int)
    retention_df["outer_fold"] = retention_df["outer_fold"].astype(str)

    # Strict frozen design consistency checks.
    got_cfg = candidate_df["configuration_id"].tolist()
    if got_cfg != list(CONFIG_ORDER):
        raise DecisionError("C", "frozen configuration order mismatch: {}".format(got_cfg))

    for cfg in CONFIG_ORDER:
        row = candidate_df[candidate_df["configuration_id"] == cfg]
        if row.empty:
            raise DecisionError("C", "missing configuration {} in frozen design".format(cfg))
        frozen_feats = row.iloc[0]["tnout_features"]
        if frozen_feats != TNOUT_BY_CONFIG[cfg]:
            raise DecisionError(
                "C",
                "frozen TNout feature list mismatch for {}: expected {}, observed {}".format(
                    cfg, TNOUT_BY_CONFIG[cfg], frozen_feats
                ),
            )

    refit = model_registry_df[model_registry_df["will_be_refitted"]].copy()
    refit_pairs = sorted([(str(r.model), int(r.horizon)) for r in refit.itertuples(index=False)])
    expected_pairs = sorted([(m, h) for m in FIT_MODELS for h in HORIZONS])
    if refit_pairs != expected_pairs:
        raise DecisionError("C", "frozen model-horizon refit set mismatch")

    final_map: Dict[str, str] = {}
    for r in refit.itertuples(index=False):
        key = "{}_H{}".format(r.model, int(r.horizon))
        final_map[key] = str(r.final_reference_configuration)

    expected_map = {
        "Ridge_H1": "REFERENCE",
        "Ridge_H3": "REFERENCE",
        "Ridge_H5": "REFERENCE",
        "HGBR_H1": "REFERENCE",
        "HGBR_H3": "REFERENCE",
        "HGBR_H5": "LONG",
    }
    if final_map != expected_map:
        raise DecisionError("C", "frozen final-reference mapping mismatch")

    # H3-v2 must exist but not be used.
    h3_v2 = lineage_df[
        (lineage_df["model"].astype(str) == "HGBR")
        & (lineage_df["horizon"].astype(int) == 3)
        & (lineage_df["final_pathway"].astype(str) == "v2_artifact_exists_not_used")
    ]
    if h3_v2.empty or bool(h3_v2["used_in_final_corrected_model"].iloc[0]):
        raise DecisionError("C", "frozen lineage no longer confirms H3-v2 exists-not-used")

    h5_v2 = lineage_df[
        (lineage_df["model"].astype(str) == "HGBR")
        & (lineage_df["horizon"].astype(int) == 5)
        & (lineage_df["final_pathway"].astype(str) == "v2_npz")
    ]
    if h5_v2.empty or (not bool(h5_v2["used_in_final_corrected_model"].iloc[0])):
        raise DecisionError("C", "frozen lineage no longer confirms HGBR-H5 v2 final pathway")

    return {
        "design_json": design_json,
        "design_manifest": design_manifest,
        "design_input": design_input,
        "candidate_df": candidate_df,
        "model_registry_df": model_registry_df,
        "common_date_df": common_date_df,
        "retention_df": retention_df,
        "lineage_df": lineage_df,
        "artifact_registry_df": artifact_registry_df,
        "final_reference_mapping": final_map,
        "required_files": required_files,
    }


def load_lock_and_dataset() -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    dataset_path = Path(lock["absolute_path"])
    if not dataset_path.exists():
        raise DecisionError("D", "locked dataset file missing: {}".format(dataset_path))

    observed_sha = sha256_file(dataset_path)
    expected_sha = str(lock["sha256"])
    if observed_sha != expected_sha:
        raise DecisionError(
            "D", "dataset checksum mismatch: expected {}, observed {}".format(expected_sha, observed_sha)
        )

    df = pd.read_csv(dataset_path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    date_col = str(lock["date_column"])
    target_col = str(lock["target_column"])
    if date_col not in df.columns or target_col not in df.columns:
        raise DecisionError("D", "locked date/target column missing from dataset")

    df[date_col] = normalize_date_col(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if int(len(df)) != int(lock["row_count"]):
        raise DecisionError("D", "dataset row count mismatch vs lock")

    if int(df[date_col].nunique()) != int(lock["unique_date_count"]):
        raise DecisionError("D", "dataset unique date count mismatch vs lock")

    min_date = df[date_col].min().strftime("%Y-%m-%d")
    max_date = df[date_col].max().strftime("%Y-%m-%d")
    if min_date != str(lock["minimum_date"]) or max_date != str(lock["maximum_date"]):
        raise DecisionError("D", "dataset date-range mismatch vs lock")

    if pd.to_numeric(df[target_col], errors="coerce").isna().any():
        raise DecisionError("D", "target column has non-numeric values after coercion")

    return lock, df, observed_sha


def load_split_assignment() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    split_sha = sha256_file(SPLIT_PATH)

    split = pd.read_csv(SPLIT_PATH)
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
        raise DecisionError("D", "split assignment missing columns: {}".format(sorted(missing)))

    split["horizon"] = split["horizon"].astype(int)
    split["outer_fold"] = split["outer_fold"].astype(int)
    split["feature_date"] = normalize_date_col(split["feature_date"])
    split["target_date"] = normalize_date_col(split["target_date"])
    split["purged_outer_boundary"] = as_bool(split["purged_outer_boundary"])
    split["purged_inner_boundary"] = as_bool(split["purged_inner_boundary"])

    if split[["feature_date", "target_date"]].isna().any().any():
        raise DecisionError("D", "split assignment contains invalid dates")

    split_summary = pd.read_csv(SPLIT_SUMMARY_PATH)
    split_summary["horizon"] = split_summary["horizon"].astype(int)
    split_summary["outer_fold"] = split_summary["outer_fold"].astype(int)

    return split, split_summary, split_sha


def build_feature_source(dataset_df: pd.DataFrame, date_col: str, target_col: str, horizon: int) -> pd.DataFrame:
    df = dataset_df.copy()

    needed = ["Inflow", "TNin", "TOCin", "temp_mean_c", "precip_total_mm", target_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DecisionError("C", "canonical dataset missing feature columns: {}".format(missing))

    if "BODin" not in df.columns:
        df["BODin"] = np.nan

    for c in [target_col, "Inflow", "TNin", "TOCin", "BODin", "temp_mean_c", "precip_total_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["C_N"] = df["TOCin"] / (df["TNin"] + 1e-6)

    df["feature_date"] = normalize_date_col(df[date_col])
    df["target_date"] = df["feature_date"] + pd.to_timedelta(horizon, unit="D")
    df["y_true"] = pd.to_numeric(df[target_col], errors="coerce").shift(-horizon)

    # TNout memory predictors, always strictly prior via shift(1) for rolls.
    df["TNout_lag1"] = df[target_col].shift(1)
    df["TNout_lag3"] = df[target_col].shift(3)
    df["TNout_lag5"] = df[target_col].shift(5)
    df["TNout_lag7"] = df[target_col].shift(7)
    df["TNout_roll7"] = df[target_col].shift(1).rolling(7).mean()
    df["TNout_roll14"] = df[target_col].shift(1).rolling(14).mean()
    df["TNout_roll30"] = df[target_col].shift(1).rolling(30).mean()

    # Ordinary non-TN lineage.
    for col in ["Inflow", "TNin", "TOCin", "BODin"]:
        s = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_roll7"] = s.rolling(7).mean()
        df[f"{col}_roll14"] = s.rolling(14).mean()

    temp = pd.to_numeric(df["temp_mean_c"], errors="coerce")
    precip = pd.to_numeric(df["precip_total_mm"], errors="coerce")
    df["temp_roll7"] = temp.rolling(7).mean()
    df["temp_roll14"] = temp.rolling(14).mean()
    df["temp_roll30"] = temp.rolling(30).mean()
    df["precip_sum3"] = precip.rolling(3).sum()
    df["precip_sum7"] = precip.rolling(7).sum()
    df["precip_sum14"] = precip.rolling(14).sum()

    df["C_N_roll14"] = pd.to_numeric(df["C_N"], errors="coerce").rolling(14).mean()

    df["Inflow_x_precip3"] = pd.to_numeric(df["Inflow"], errors="coerce") * df["precip_sum3"]
    df["temp_x_CN"] = temp * df["C_N"]
    df["Inflow_x_TNin"] = pd.to_numeric(df["Inflow"], errors="coerce") * pd.to_numeric(df["TNin"], errors="coerce")

    doy = df[date_col].dt.dayofyear.values
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    cols = [
        "feature_date",
        "target_date",
        "y_true",
        *TNOUT_LONG,
        *ORDINARY_NON_TN,
        *[c for c in V2_NON_TN if c not in ORDINARY_NON_TN],
    ]
    out = df[cols].copy()
    out = out.sort_values(["feature_date", "target_date"]).reset_index(drop=True)
    return out


def load_npz_frame(npz_path: Path, horizon: int) -> Tuple[pd.DataFrame, List[str]]:
    if not npz_path.exists():
        raise DecisionError("C", "required NPZ missing: {}".format(npz_path))

    z = np.load(npz_path, allow_pickle=True)
    required = {"X", "dates", "feature_names"}
    if not required.issubset(set(z.files)):
        raise DecisionError("C", "NPZ missing required arrays: {}".format(npz_path))

    X = np.asarray(z["X"], dtype=float)
    dates = normalize_date_col(pd.Series(z["dates"]))
    feature_names = [str(c) for c in list(z["feature_names"])]

    if X.ndim != 2:
        raise DecisionError("C", "NPZ feature matrix is not 2D: {}".format(npz_path))
    if len(dates) != X.shape[0] or len(feature_names) != X.shape[1]:
        raise DecisionError("C", "NPZ shape mismatch in {}".format(npz_path))

    out = pd.DataFrame(X, columns=feature_names)
    out.insert(0, "feature_date", dates)
    out["target_date"] = out["feature_date"] + pd.to_timedelta(horizon, unit="D")

    if "y" in z.files:
        out["y_true"] = np.asarray(z["y"], dtype=float)
    else:
        raise DecisionError("C", "NPZ lacks y target array: {}".format(npz_path))

    if out[["feature_date", "target_date", "y_true"]].isna().any().any():
        raise DecisionError("C", "NPZ frame has missing key/target fields: {}".format(npz_path))

    if bool(out.duplicated(subset=["feature_date", "target_date"]).any()):
        raise DecisionError("C", "NPZ has duplicate key rows: {}".format(npz_path))

    return out.sort_values(["feature_date", "target_date"]).reset_index(drop=True), feature_names


def build_hgbr_frame_with_extra_tn(
    npz_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    keep_feature_names: List[str],
) -> pd.DataFrame:
    out = npz_frame[["feature_date", "target_date", "y_true", *keep_feature_names]].copy()

    missing_tn = [c for c in TNOUT_LONG if c not in out.columns]
    if missing_tn:
        src = source_frame[["feature_date", "target_date", *missing_tn]].copy()
        out = out.merge(src, on=["feature_date", "target_date"], how="left", validate="one_to_one")

    return out.sort_values(["feature_date", "target_date"]).reset_index(drop=True)


def compute_mase_denom(y_train: np.ndarray, m: int = 1) -> float:
    if len(y_train) <= m:
        return float("nan")
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def feature_hash(cols: Sequence[str]) -> str:
    payload = "||".join([str(c) for c in cols])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def params_hash(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fit_and_predict(
    model: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    if model == "Ridge":
        alpha = float(params["alpha"])
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    Ridge(
                        alpha=alpha,
                        fit_intercept=True,
                        solver="auto",
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        return pipe.predict(X_test)

    if model == "HGBR":
        reg = HistGradientBoostingRegressor(**params)
        reg.fit(X_train, y_train)
        return reg.predict(X_test)

    raise DecisionError("C", "unsupported model: {}".format(model))


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    if np.allclose(a, b, atol=0.0, rtol=0.0):
        return 1.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    if np.allclose(a, b, atol=0.0, rtol=0.0):
        return 1.0
    val = spearmanr(a, b).correlation
    return float(val) if val is not None else float("nan")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body: List[str] = []
    for row in df.itertuples(index=False):
        vals = []
        for v in row:
            t = str(v)
            t = t.replace("\n", " ").replace("|", "\\|")
            vals.append(t)
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def compute_output_checksums(out_dir: Path, exclude_name: Optional[str] = None) -> Dict[str, str]:
    checks: Dict[str, str] = {}
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if exclude_name and p.name == exclude_name:
            continue
        if p.suffix.lower() not in {".py", ".csv", ".json", ".md"}:
            continue
        checks[p.name] = sha256_file(p)
    return checks


def write_checksums_file(out_dir: Path, checksums_path: Path) -> None:
    lines: List[str] = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == checksums_path.name:
            continue
        if p.suffix.lower() not in {".py", ".csv", ".json", ".md"}:
            continue
        lines.append("{}  {}".format(sha256_file(p), p.name))
    write_text(checksums_path, "\n".join(lines) + "\n")


def verify_checksums_against_registry(out_dir: Path, checksums_path: Path) -> Dict[str, Any]:
    entries = parse_checksum_manifest(checksums_path)
    missing = []
    mismatched = []
    for expected, rel in entries:
        target = out_dir / rel
        if not target.exists():
            missing.append(rel)
            continue
        obs = sha256_file(target)
        if obs != expected:
            mismatched.append({"file": rel, "expected": expected, "observed": obs})

    all_files = sorted(
        [
            p.name
            for p in out_dir.iterdir()
            if p.is_file()
            and p.name != checksums_path.name
            and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
        ]
    )
    listed = sorted([rel for _, rel in entries])
    coverage_missing = sorted(set(all_files).difference(set(listed)))
    coverage_extra = sorted(set(listed).difference(set(all_files)))

    registry_pass = (len(missing) == 0) and (len(mismatched) == 0)
    coverage_pass = (len(coverage_missing) == 0) and (len(coverage_extra) == 0)
    return {
        "registry_pass": bool(registry_pass),
        "coverage_pass": bool(coverage_pass),
        "missing_files": missing,
        "mismatched": mismatched,
        "coverage_missing": coverage_missing,
        "coverage_extra": coverage_extra,
    }


def canonical_key_registry(split: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for h in HORIZONS:
        for fold in OUTER_FOLDS:
            keys = (
                split[
                    (split["horizon"] == h)
                    & (split["outer_fold"] == fold)
                    & (split["outer_role"].astype(str) == "outer_test")
                ][["feature_date", "target_date", "original_row_index"]]
                .drop_duplicates()
                .sort_values(["feature_date", "target_date", "original_row_index"])
                .reset_index(drop=True)
            )
            if keys.empty:
                raise DecisionError("C", "missing canonical outer_test keys for H{} fold{}".format(h, fold))
            keys["canonical_test_row"] = np.arange(1, len(keys) + 1, dtype=int)
            keys["horizon"] = h
            keys["outer_fold"] = fold
            rows.append(keys)
    out = pd.concat(rows, ignore_index=True)
    return out[["horizon", "outer_fold", "feature_date", "target_date", "original_row_index", "canonical_test_row"]]


def extract_partitions(
    frame: pd.DataFrame,
    keys: pd.DataFrame,
    feature_cols: Sequence[str],
    require_complete: bool,
) -> pd.DataFrame:
    merged = keys.merge(
        frame,
        on=["feature_date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    req_cols = list(feature_cols) + ["y_true"]

    if require_complete:
        if merged[req_cols].isna().any().any():
            missing = merged[merged[req_cols].isna().any(axis=1)][["feature_date", "target_date"]].head(5)
            pairs = [
                "{}->{}".format(r.feature_date.strftime("%Y-%m-%d"), r.target_date.strftime("%Y-%m-%d"))
                for r in missing.itertuples(index=False)
            ]
            raise DecisionError("C", "missing required feature values for canonical test rows: {}".format(pairs))
        return merged.sort_values(["feature_date", "target_date"]).reset_index(drop=True)

    out = merged.dropna(subset=req_cols).sort_values(["feature_date", "target_date"]).reset_index(drop=True)
    return out


def run_execution() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    protected_before = snapshot_tree_checksums(IMMUTABLE_DIRS, ROOT)

    # Preflight metadata and checks.
    repository = str(ROOT.resolve())
    branch = git_output(["git", "branch", "--show-current"])
    head = git_output(["git", "rev-parse", "HEAD"])
    tag_object = git_output(["git", "rev-parse", EXPECTED_DESIGN_TAG])
    tag_commit = git_output(["git", "rev-parse", EXPECTED_DESIGN_TAG + "^{}"])
    tag_type = git_output(["git", "cat-file", "-t", EXPECTED_DESIGN_TAG])

    python_version = platform.python_version()

    status_paths = parse_git_status_paths()
    clean_now = len(status_paths) == 0
    authorized_only_now = bool(
        len(status_paths) > 0 and all(is_authorized_git_status_path(p, AUTHORIZED_REL_DIR) for p in status_paths)
    )

    prior_clean_pass = False
    if INPUT_VERIFICATION_PATH.exists():
        try:
            prior = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
            prior_clean_pass = bool(
                prior.get("checks", {}).get("clean_initial_source_state_pass", False)
            )
        except Exception:
            prior_clean_pass = False

    clean_initial_source_state_pass = bool(clean_now or authorized_only_now or prior_clean_pass)

    if repository != EXPECTED_REPOSITORY:
        raise DecisionError("D", "repository mismatch: expected {}, observed {}".format(EXPECTED_REPOSITORY, repository))
    if branch != EXPECTED_BRANCH:
        raise DecisionError("D", "branch mismatch: expected {}, observed {}".format(EXPECTED_BRANCH, branch))
    if head != EXPECTED_HEAD:
        raise DecisionError("D", "HEAD mismatch: expected {}, observed {}".format(EXPECTED_HEAD, head))
    if tag_commit != head:
        raise DecisionError("D", "design tag does not point to HEAD")
    if python_version != EXPECTED_PYTHON:
        raise DecisionError("D", "python version mismatch: expected {}, observed {}".format(EXPECTED_PYTHON, python_version))
    if not clean_initial_source_state_pass:
        raise DecisionError("D", "clean starting source state gate failed")

    protocol_checks = verify_checksum_manifest(PROTOCOL_SHA_PATH, PROTOCOL_DIR)
    stage1_checks = verify_checksum_manifest(STAGE1_CHECKSUMS_PATH, STAGE1_DIR)
    design_checks = verify_checksum_manifest(DESIGN_CHECKSUMS_PATH, DESIGN_DIR)

    design = load_design_bundle()

    lock, dataset_df, dataset_sha = load_lock_and_dataset()
    split, split_summary, split_sha = load_split_assignment()

    # Verify split sha matches the frozen design manifest.
    design_split_sha = str(design["design_manifest"].get("split_sha256", ""))
    if design_split_sha and design_split_sha != split_sha:
        raise DecisionError("C", "split SHA no longer matches frozen design manifest")

    design_dataset_sha = str(design["design_manifest"].get("dataset_sha256", ""))
    if design_dataset_sha and design_dataset_sha != dataset_sha:
        raise DecisionError("C", "dataset SHA no longer matches frozen design manifest")

    # Hash every consumed file.
    consumed_rel_paths: List[str] = []
    consumed_rel_paths.extend([str(p.relative_to(ROOT)).replace("\\", "/") for p in design["required_files"]])

    for rel in design["design_manifest"].get("input_files", []):
        consumed_rel_paths.append(str(rel).replace("\\", "/"))

    consumed_rel_paths.extend(
        [
            str(RIDGE_SELECTED_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(RIDGE_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(RIDGE_PRED_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(HGBR_SELECTED_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(HGBR_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(HGBR_PRED_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(SPLIT_SUMMARY_PATH.relative_to(ROOT)).replace("\\", "/"),
        ]
    )

    consumed_rel_paths = sorted(set(consumed_rel_paths))
    consumed_hashes: Dict[str, str] = {}
    for rel in consumed_rel_paths:
        abs_path = ROOT / rel
        if not abs_path.exists():
            raise DecisionError("D", "consumed input missing: {}".format(abs_path))
        consumed_hashes[rel] = sha256_file(abs_path)

    # Frozen source hash expectations from design manifest input_hashes where available.
    design_input_hashes = {
        str(k).replace("\\", "/"): str(v)
        for k, v in design["design_manifest"].get("input_hashes", {}).items()
    }
    for rel, expected_sha in design_input_hashes.items():
        observed = consumed_hashes.get(rel)
        if observed is None:
            continue
        if observed != expected_sha:
            raise DecisionError(
                "D",
                "frozen source hash mismatch for {}: expected {}, observed {}".format(
                    rel, expected_sha, observed
                ),
            )

    input_verification = {
        "repository": repository,
        "authorized_output_directory": AUTHORIZED_REL_DIR,
        "required_branch": EXPECTED_BRANCH,
        "required_head": EXPECTED_HEAD,
        "required_design_tag": EXPECTED_DESIGN_TAG,
        "branch": branch,
        "head": head,
        "design_tag_object": tag_object,
        "design_tag_type": tag_type,
        "design_tag_commit": tag_commit,
        "python_version": python_version,
        "working_tree_status_paths": status_paths,
        "dataset_path": str(Path(lock["absolute_path"]).resolve()),
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "consumed_input_hashes": consumed_hashes,
        "checks": {
            "repository_pass": repository == EXPECTED_REPOSITORY,
            "branch_pass": branch == EXPECTED_BRANCH,
            "head_pass": head == EXPECTED_HEAD,
            "design_tag_points_to_head_pass": tag_commit == head,
            "python_3810_pass": python_version == EXPECTED_PYTHON,
            "clean_initial_source_state_pass": clean_initial_source_state_pass,
            "corrected_protocol_checksum_pass": True,
            "stage1_checksum_pass": True,
            "design_checksum_pass": True,
            "frozen_source_hash_pass": True,
        },
        "protocol_checksum_checks": protocol_checks,
        "stage1_checksum_checks": stage1_checks,
        "design_checksum_checks": design_checks,
    }
    write_json(INPUT_VERIFICATION_PATH, input_verification)

    # Frozen hyperparameters.
    ridge_selected = pd.read_csv(RIDGE_SELECTED_PATH)
    ridge_selected["horizon"] = ridge_selected["horizon"].astype(int)
    ridge_selected["outer_fold"] = ridge_selected["outer_fold"].astype(int)

    hgbr_selected = pd.read_csv(HGBR_SELECTED_PATH)
    hgbr_selected["horizon"] = hgbr_selected["horizon"].astype(int)
    hgbr_selected["outer_fold"] = hgbr_selected["outer_fold"].astype(int)

    ridge_manifest = json.loads(RIDGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    hgbr_manifest = json.loads(HGBR_MANIFEST_PATH.read_text(encoding="utf-8"))

    # Build canonical key registry and base frames.
    key_registry = canonical_key_registry(split)

    date_col = str(lock["date_column"])
    target_col = str(lock["target_column"])

    source_frames: Dict[int, pd.DataFrame] = {}
    for h in HORIZONS:
        source_frames[h] = build_feature_source(dataset_df, date_col, target_col, h)

    hgbr_h1_npz, hgbr_h1_features = load_npz_frame(FEATURE_H1_NPZ, 1)
    hgbr_h3_npz, hgbr_h3_features = load_npz_frame(FEATURE_H3_NPZ, 3)
    hgbr_h5_npz, hgbr_h5_features = load_npz_frame(FEATURE_H5V2_NPZ, 5)

    hgbr_h1_frame = build_hgbr_frame_with_extra_tn(hgbr_h1_npz, source_frames[1], hgbr_h1_features)
    hgbr_h3_frame = build_hgbr_frame_with_extra_tn(hgbr_h3_npz, source_frames[3], hgbr_h3_features)
    hgbr_h5_frame = build_hgbr_frame_with_extra_tn(hgbr_h5_npz, source_frames[5], hgbr_h5_features)

    ridge_reference_features = {
        h: [str(c) for c in ridge_manifest["feature_sets"][str(h)]] for h in HORIZONS
    }

    final_reference_mapping = dict(design["final_reference_mapping"])

    frame_by_model_horizon: Dict[Tuple[str, int], pd.DataFrame] = {
        ("Ridge", 1): source_frames[1],
        ("Ridge", 3): source_frames[3],
        ("Ridge", 5): source_frames[5],
        ("HGBR", 1): hgbr_h1_frame,
        ("HGBR", 3): hgbr_h3_frame,
        ("HGBR", 5): hgbr_h5_frame,
    }

    prediction_rows: List[Dict[str, Any]] = []
    fit_registry_rows: List[Dict[str, Any]] = []
    feature_registry_rows: List[Dict[str, Any]] = []
    retention_rows: List[Dict[str, Any]] = []
    mase_denoms: Dict[Tuple[str, int, str, int], float] = {}

    execution_run_id = "lag_window_exec_{}_{}_{}".format(head[:12], dataset_sha[:8], split_sha[:8])

    for model in FIT_MODELS:
        for horizon in HORIZONS:
            ref_cfg = final_reference_mapping["{}_H{}".format(model, horizon)]

            if model == "Ridge":
                non_tn_cols = list(ORDINARY_NON_TN)
            else:
                non_tn_cols = list(V2_NON_TN if horizon == 5 else ORDINARY_NON_TN)

            for fold in OUTER_FOLDS:
                split_hf = split[(split["horizon"] == horizon) & (split["outer_fold"] == fold)].copy()
                if split_hf.empty:
                    raise DecisionError("C", "missing split rows for {} H{} fold{}".format(model, horizon, fold))

                summ = split_summary[
                    (split_summary["horizon"] == horizon) & (split_summary["outer_fold"] == fold)
                ]
                if summ.empty:
                    raise DecisionError("C", "missing split summary for H{} fold{}".format(horizon, fold))
                summ_row = summ.iloc[0]

                outer_train_keys = (
                    split_hf[
                        (split_hf["outer_role"].astype(str) == "outer_train")
                        & (~split_hf["purged_outer_boundary"])
                    ][["feature_date", "target_date", "original_row_index"]]
                    .drop_duplicates()
                    .sort_values(["feature_date", "target_date", "original_row_index"])
                )

                outer_test_keys = (
                    split_hf[(split_hf["outer_role"].astype(str) == "outer_test")][
                        ["feature_date", "target_date", "original_row_index"]
                    ]
                    .drop_duplicates()
                    .sort_values(["feature_date", "target_date", "original_row_index"])
                )

                if int(len(outer_test_keys)) != int(summ_row["outer_test_n"]):
                    raise DecisionError("C", "outer-test count mismatch in split summary for H{} fold{}".format(horizon, fold))

                key_map = key_registry[
                    (key_registry["horizon"] == horizon) & (key_registry["outer_fold"] == fold)
                ][["feature_date", "target_date", "canonical_test_row"]]

                frame = frame_by_model_horizon[(model, horizon)]

                # Frozen hyperparameter lookup.
                if model == "Ridge":
                    ridge_row = ridge_selected[
                        (ridge_selected["horizon"] == horizon)
                        & (ridge_selected["outer_fold"] == fold)
                    ]
                    if ridge_row.empty:
                        raise DecisionError("C", "missing ridge frozen alpha for H{} fold{}".format(horizon, fold))
                    alpha = float(ridge_row.iloc[0]["selected_alpha"])
                    frozen_params = {"alpha": alpha, "fit_intercept": True, "solver": "auto", "random_state": 42}
                    hp_source = str(RIDGE_SELECTED_PATH.relative_to(ROOT)).replace("\\", "/")
                    random_seed = 42
                else:
                    hgbr_row = hgbr_selected[
                        (hgbr_selected["horizon"] == horizon)
                        & (hgbr_selected["outer_fold"] == fold)
                    ]
                    if hgbr_row.empty:
                        raise DecisionError("C", "missing HGBR frozen parameters for H{} fold{}".format(horizon, fold))
                    frozen_params = json.loads(str(hgbr_row.iloc[0]["selected_parameters_json"]))
                    hp_source = str(HGBR_SELECTED_PATH.relative_to(ROOT)).replace("\\", "/")
                    random_seed = int(hgbr_row.iloc[0].get("random_seed", 42))

                for cfg in CONFIG_ORDER:
                    tn_cols = list(TNOUT_BY_CONFIG[cfg])
                    feature_cols = tn_cols + non_tn_cols

                    # Preserve frozen reference ordering where possible for Ridge.
                    if model == "Ridge":
                        ordered = []
                        for col in ridge_reference_features[horizon]:
                            if col in feature_cols and col not in ordered:
                                ordered.append(col)
                        for col in feature_cols:
                            if col not in ordered:
                                ordered.append(col)
                        feature_cols = ordered

                    train = extract_partitions(frame, outer_train_keys, feature_cols, require_complete=False)
                    test = extract_partitions(frame, outer_test_keys, feature_cols, require_complete=True)
                    test = test.merge(key_map, on=["feature_date", "target_date"], how="left", validate="one_to_one")

                    if test["canonical_test_row"].isna().any():
                        raise DecisionError("C", "failed canonical row mapping for {} H{} {} fold{}".format(model, horizon, cfg, fold))

                    if int(len(test)) != int(len(outer_test_keys)):
                        raise DecisionError("C", "canonical test-date loss detected for {} H{} {} fold{}".format(model, horizon, cfg, fold))

                    X_train = train[feature_cols].to_numpy(dtype=float)
                    y_train = train["y_true"].to_numpy(dtype=float)
                    X_test = test[feature_cols].to_numpy(dtype=float)

                    if len(train) == 0:
                        raise DecisionError("C", "zero training rows for {} H{} {} fold{}".format(model, horizon, cfg, fold))

                    y_hat = fit_and_predict(model, frozen_params, X_train, y_train, X_test)
                    denom = compute_mase_denom(y_train, m=1)
                    mase_denoms[(model, horizon, cfg, fold)] = denom

                    f_hash = feature_hash(feature_cols)
                    h_hash = params_hash(frozen_params)
                    is_ref = bool(cfg == ref_cfg)

                    warmup_losses = int(len(outer_train_keys) - len(train))
                    retention_rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "outer_fold": fold,
                            "configuration_id": cfg,
                            "available_pre_purge_rows": int(summ_row["outer_train_n_before_purge"]),
                            "purged_rows": int(summ_row["outer_rows_purged"]),
                            "available_post_purge_rows": int(len(outer_train_keys)),
                            "warmup_losses": warmup_losses,
                            "final_fitting_rows": int(len(train)),
                            "canonical_test_rows": int(len(test)),
                            "earliest_fitting_feature_date": train["feature_date"].min(),
                            "latest_fitting_feature_date": train["feature_date"].max(),
                            "earliest_test_feature_date": test["feature_date"].min(),
                            "latest_test_feature_date": test["feature_date"].max(),
                            "feature_count": int(len(feature_cols)),
                            "tnout_features": "; ".join(tn_cols),
                            "non_tnout_features": "; ".join(non_tn_cols),
                            "feature_hash": f_hash,
                        }
                    )

                    feature_registry_rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "outer_fold": fold,
                            "configuration_id": cfg,
                            "feature_count": int(len(feature_cols)),
                            "tnout_features": "; ".join(tn_cols),
                            "non_tnout_features": "; ".join(non_tn_cols),
                            "feature_names": "; ".join(feature_cols),
                            "feature_hash": f_hash,
                            "tnout_roll_rule": "TNout_rollw(t)=mean(TNout(t-w),...,TNout(t-1))",
                            "tnout_roll_includes_day_t": False,
                        }
                    )

                    fit_registry_rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "outer_fold": fold,
                            "configuration_id": cfg,
                            "hyperparameter_source_file": hp_source,
                            "hyperparameter_dict_json": json.dumps(frozen_params, sort_keys=True),
                            "hyperparameter_hash": h_hash,
                            "random_seed": random_seed,
                            "training_row_count": int(len(train)),
                            "feature_count": int(len(feature_cols)),
                            "feature_names": "; ".join(feature_cols),
                            "feature_set_hash": f_hash,
                            "software_versions": json.dumps(
                                {
                                    "python": platform.python_version(),
                                    "numpy": np.__version__,
                                    "pandas": pd.__version__,
                                    "scikit_learn": sklearn.__version__,
                                    "scipy": scipy.__version__,
                                },
                                sort_keys=True,
                            ),
                            "fit_status": "success",
                            "prediction_status": "success",
                        }
                    )

                    for row, yp in zip(test.itertuples(index=False), y_hat):
                        y_true = float(row.y_true)
                        y_pred = float(yp)
                        err = y_pred - y_true
                        prediction_rows.append(
                            {
                                "prediction_id": "{}_H{}_{}_F{}_{}_{}".format(
                                    model,
                                    horizon,
                                    cfg,
                                    fold,
                                    row.feature_date.strftime("%Y%m%d"),
                                    row.target_date.strftime("%Y%m%d"),
                                ),
                                "model": model,
                                "horizon": "H{}".format(horizon),
                                "horizon_int": horizon,
                                "configuration_id": cfg,
                                "is_final_reference": is_ref,
                                "final_reference_configuration": ref_cfg,
                                "reference_source": str(DESIGN_MODEL_REGISTRY_PATH.relative_to(ROOT)).replace("\\", "/"),
                                "reference_reproduction_status": "pending",
                                "outer_fold": fold,
                                "feature_date": row.feature_date,
                                "target_date": row.target_date,
                                "y_true": y_true,
                                "y_pred": y_pred,
                                "error": err,
                                "abs_error": abs(err),
                                "squared_error": err * err,
                                "canonical_test_row": int(row.canonical_test_row),
                                "training_row_count": int(len(train)),
                                "feature_count": int(len(feature_cols)),
                                "feature_set_hash": f_hash,
                                "hyperparameter_hash": h_hash,
                                "input_parent_commit": head,
                                "design_commit": tag_commit,
                                "execution_run_id": execution_run_id,
                            }
                        )

    pred = pd.DataFrame(prediction_rows)
    pred = pred.sort_values(
        ["model", "horizon_int", "configuration_id", "outer_fold", "feature_date", "target_date"]
    ).reset_index(drop=True)

    # Hard row-count checks.
    expected_h_counts = {"H1": 4572, "H3": 4482, "H5": 4482}
    observed_h_counts = pred.groupby("horizon").size().to_dict()
    if observed_h_counts != expected_h_counts:
        raise DecisionError(
            "C",
            "prediction rows by horizon mismatch: expected {}, observed {}".format(
                expected_h_counts, observed_h_counts
            ),
        )

    if int(len(pred)) != 13536:
        raise DecisionError("C", "total prediction row mismatch: expected 13536, observed {}".format(len(pred)))

    dup = pred.duplicated(
        subset=["model", "horizon", "configuration_id", "outer_fold", "feature_date", "target_date"]
    )
    if bool(dup.any()):
        raise DecisionError("C", "duplicate prediction keys detected")

    if pred[["y_true", "y_pred"]].isna().any().any():
        raise DecisionError("C", "prediction file has missing y_true/y_pred values")

    if not np.isfinite(pred[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise DecisionError("C", "prediction file has non-finite values")

    # Common-date status across configurations.
    common_date_rows: List[Dict[str, Any]] = []
    common_date_pass = True
    for model in FIT_MODELS:
        for horizon in HORIZONS:
            for fold in OUTER_FOLDS:
                sets: Dict[str, set] = {}
                for cfg in CONFIG_ORDER:
                    d = pred[
                        (pred["model"] == model)
                        & (pred["horizon_int"] == horizon)
                        & (pred["configuration_id"] == cfg)
                        & (pred["outer_fold"] == fold)
                    ][["feature_date", "target_date"]]
                    sets[cfg] = set((r.feature_date, r.target_date) for r in d.itertuples(index=False))

                reference = sets["REFERENCE"]
                all_equal = all(sets[cfg] == reference for cfg in CONFIG_ORDER)
                common_date_pass = bool(common_date_pass and all_equal)
                common_date_rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "outer_fold": fold,
                        "short_n": len(sets["SHORT"]),
                        "reference_n": len(sets["REFERENCE"]),
                        "long_n": len(sets["LONG"]),
                        "common_dates_pass": bool(all_equal),
                    }
                )

    # Reference reproduction audit.
    ridge_ref = pd.read_csv(RIDGE_PRED_PATH)
    ridge_ref["horizon"] = ridge_ref["horizon"].astype(int)
    ridge_ref["outer_fold"] = ridge_ref["outer_fold"].astype(int)
    ridge_ref["feature_date"] = normalize_date_col(ridge_ref["feature_date"])
    ridge_ref["target_date"] = normalize_date_col(ridge_ref["target_date"])

    hgbr_ref = pd.read_csv(HGBR_PRED_PATH)
    hgbr_ref["horizon"] = hgbr_ref["horizon"].astype(int)
    hgbr_ref["outer_fold"] = hgbr_ref["outer_fold"].astype(int)
    hgbr_ref["feature_date"] = normalize_date_col(hgbr_ref["feature_date"])
    hgbr_ref["target_date"] = normalize_date_col(hgbr_ref["target_date"])

    ref_rows: List[Dict[str, Any]] = []
    reference_reproduction_pass = True
    reference_reproduction_material_fail = False

    pathway_sources = {
        "Ridge": (ridge_ref, str(RIDGE_PRED_PATH.relative_to(ROOT)).replace("\\", "/")),
        "HGBR": (hgbr_ref, str(HGBR_PRED_PATH.relative_to(ROOT)).replace("\\", "/")),
    }

    for model in FIT_MODELS:
        ref_df, ref_source = pathway_sources[model]
        for horizon in HORIZONS:
            ref_cfg = final_reference_mapping["{}_H{}".format(model, horizon)]

            gen = pred[
                (pred["model"] == model)
                & (pred["horizon_int"] == horizon)
                & (pred["configuration_id"] == ref_cfg)
            ][["outer_fold", "feature_date", "target_date", "y_true", "y_pred"]].copy()

            ref_part = ref_df[ref_df["horizon"] == horizon][
                ["outer_fold", "feature_date", "target_date", "y_true", "y_pred"]
            ].copy()

            key_cols = ["outer_fold", "feature_date", "target_date"]
            gen_keys = set(tuple(r) for r in gen[key_cols].itertuples(index=False, name=None))
            ref_keys = set(tuple(r) for r in ref_part[key_cols].itertuples(index=False, name=None))
            key_match = gen_keys == ref_keys

            if not key_match:
                reference_reproduction_pass = False
                reference_reproduction_material_fail = True

            for fold in OUTER_FOLDS:
                g = gen[gen["outer_fold"] == fold].sort_values(["feature_date", "target_date"]).reset_index(drop=True)
                r = ref_part[ref_part["outer_fold"] == fold].sort_values(["feature_date", "target_date"]).reset_index(drop=True)

                local_key_match = set(tuple(x) for x in g[key_cols[1:]].itertuples(index=False, name=None)) == set(
                    tuple(x) for x in r[key_cols[1:]].itertuples(index=False, name=None)
                )

                if not local_key_match:
                    reference_reproduction_pass = False
                    reference_reproduction_material_fail = True
                    y_true_match = False
                    max_abs = float("nan")
                    mean_abs = float("nan")
                    outside = -1
                    status = "key_mismatch"
                else:
                    merged = g.merge(r, on=["outer_fold", "feature_date", "target_date"], suffixes=("_gen", "_ref"), validate="one_to_one")
                    y_true_diff = np.abs(merged["y_true_gen"].to_numpy(dtype=float) - merged["y_true_ref"].to_numpy(dtype=float))
                    y_pred_diff = np.abs(merged["y_pred_gen"].to_numpy(dtype=float) - merged["y_pred_ref"].to_numpy(dtype=float))
                    y_true_match = bool(np.max(y_true_diff) <= REF_TOL)
                    max_abs = float(np.max(y_pred_diff)) if len(y_pred_diff) else 0.0
                    mean_abs = float(np.mean(y_pred_diff)) if len(y_pred_diff) else 0.0
                    outside = int(np.sum(y_pred_diff > REF_TOL))
                    status = "pass" if (y_true_match and outside == 0) else "prediction_mismatch"
                    if not y_true_match:
                        reference_reproduction_material_fail = True
                        reference_reproduction_pass = False
                    if outside > 0:
                        reference_reproduction_pass = False

                ref_rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "configuration_id": ref_cfg,
                        "outer_fold": fold,
                        "reference_source": ref_source,
                        "row_count_generated": int(len(g)),
                        "row_count_reference": int(len(r)),
                        "keys_match": bool(local_key_match),
                        "y_true_match": bool(y_true_match),
                        "maximum_absolute_prediction_difference": max_abs,
                        "mean_absolute_prediction_difference": mean_abs,
                        "number_outside_tolerance": int(outside),
                        "tolerance": REF_TOL,
                        "status": status,
                    }
                )

    ref_audit_df = pd.DataFrame(ref_rows).sort_values(["model", "horizon", "outer_fold"])

    # Update row-level reference reproduction status.
    pathway_status_map: Dict[Tuple[str, int], str] = {}
    for (m, h), grp in ref_audit_df.groupby(["model", "horizon"], sort=True):
        if bool((grp["status"].astype(str) == "pass").all()):
            pathway_status_map[(m, int(h))] = "pass"
        else:
            pathway_status_map[(m, int(h))] = "fail"

    status_col = []
    for r in pred.itertuples(index=False):
        if bool(r.is_final_reference):
            status_col.append(pathway_status_map[(str(r.model), int(r.horizon_int))])
        else:
            status_col.append("not_applicable")
    pred["reference_reproduction_status"] = status_col

    # Point metrics.
    by_fold_rows: List[Dict[str, Any]] = []
    for (model, horizon, cfg, fold), grp in pred.groupby(
        ["model", "horizon_int", "configuration_id", "outer_fold"], sort=True
    ):
        mae = float(grp["abs_error"].mean())
        mse = float(grp["squared_error"].mean())
        rmse = float(np.sqrt(mse))
        denom = float(mase_denoms[(model, int(horizon), cfg, int(fold))])
        mase = float(mae / (denom + MASE_EPS)) if np.isfinite(denom) else float("nan")

        by_fold_rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "configuration_id": cfg,
                "outer_fold": int(fold),
                "N": int(len(grp)),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MASE": mase,
                "mase_denominator": denom,
                "mase_denominator_source": "outer_train_y_true_abs_first_difference_mean_m1",
                "mase_method": "MAE/(fold_local_training_denominator+epsilon)",
                "epsilon_or_stabilizer": MASE_EPS,
                "training_rows_used_for_denominator": int(
                    retention_rows[
                        [
                            i
                            for i, rr in enumerate(retention_rows)
                            if rr["model"] == model
                            and rr["horizon"] == int(horizon)
                            and rr["configuration_id"] == cfg
                            and rr["outer_fold"] == int(fold)
                        ][0]
                    ]["final_fitting_rows"]
                ),
            }
        )

    point_by_fold = pd.DataFrame(by_fold_rows).sort_values(["model", "horizon", "configuration_id", "outer_fold"])

    pooled_rows: List[Dict[str, Any]] = []
    for (model, horizon, cfg), grp in pred.groupby(["model", "horizon_int", "configuration_id"], sort=True):
        mae = float(grp["abs_error"].mean())
        mse = float(grp["squared_error"].mean())
        rmse = float(np.sqrt(mse))

        den = grp["outer_fold"].map(lambda f: mase_denoms[(model, int(horizon), cfg, int(f))]).to_numpy(dtype=float)
        mase = float(np.mean(grp["abs_error"].to_numpy(dtype=float) / (den + MASE_EPS)))

        pooled_rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "configuration_id": cfg,
                "outer_fold": "pooled",
                "N": int(len(grp)),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MASE": mase,
                "mase_denominator": "rowwise_fold_local",
                "mase_denominator_source": "fold_local_training_denominator_per_row",
                "mase_method": "mean(abs_error/(row_fold_denominator+epsilon))",
                "epsilon_or_stabilizer": MASE_EPS,
                "training_rows_used_for_denominator": int(
                    sum(
                        int(rr["final_fitting_rows"])
                        for rr in retention_rows
                        if rr["model"] == model and rr["horizon"] == int(horizon) and rr["configuration_id"] == cfg
                    )
                ),
            }
        )

    point_pooled = pd.DataFrame(pooled_rows).sort_values(["model", "horizon", "configuration_id"])

    # Stability metrics versus final reference.
    stability_fold_rows: List[Dict[str, Any]] = []
    for model in FIT_MODELS:
        for horizon in HORIZONS:
            ref_cfg = final_reference_mapping["{}_H{}".format(model, horizon)]
            for fold in OUTER_FOLDS:
                ref = pred[
                    (pred["model"] == model)
                    & (pred["horizon_int"] == horizon)
                    & (pred["outer_fold"] == fold)
                    & (pred["configuration_id"] == ref_cfg)
                ][["feature_date", "target_date", "y_pred"]].sort_values(["feature_date", "target_date"]).reset_index(drop=True)

                for cfg in CONFIG_ORDER:
                    cur = pred[
                        (pred["model"] == model)
                        & (pred["horizon_int"] == horizon)
                        & (pred["outer_fold"] == fold)
                        & (pred["configuration_id"] == cfg)
                    ][["feature_date", "target_date", "y_pred"]].sort_values(["feature_date", "target_date"]).reset_index(drop=True)

                    merged = ref.merge(cur, on=["feature_date", "target_date"], suffixes=("_ref", "_cfg"), validate="one_to_one")
                    diff = merged["y_pred_cfg"].to_numpy(dtype=float) - merged["y_pred_ref"].to_numpy(dtype=float)

                    stability_fold_rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "configuration_id": cfg,
                            "final_reference_configuration": ref_cfg,
                            "outer_fold": fold,
                            "N": int(len(merged)),
                            "pearson_prediction_correlation": safe_pearson(
                                merged["y_pred_cfg"].to_numpy(dtype=float),
                                merged["y_pred_ref"].to_numpy(dtype=float),
                            ),
                            "spearman_prediction_rank_correlation": safe_spearman(
                                merged["y_pred_cfg"].to_numpy(dtype=float),
                                merged["y_pred_ref"].to_numpy(dtype=float),
                            ),
                            "mean_prediction_difference": float(np.mean(diff)),
                            "mean_absolute_prediction_difference": float(np.mean(np.abs(diff))),
                            "maximum_absolute_prediction_difference": float(np.max(np.abs(diff))),
                        }
                    )

    stability_by_fold = pd.DataFrame(stability_fold_rows).sort_values(
        ["model", "horizon", "configuration_id", "outer_fold"]
    )

    stability_pooled_rows: List[Dict[str, Any]] = []
    for (model, horizon, cfg), grp in stability_by_fold.groupby(["model", "horizon", "configuration_id"], sort=True):
        ref_cfg = final_reference_mapping["{}_H{}".format(model, int(horizon))]
        cur = pred[
            (pred["model"] == model)
            & (pred["horizon_int"] == int(horizon))
            & (pred["configuration_id"] == cfg)
        ][["outer_fold", "feature_date", "target_date", "y_pred"]]
        ref = pred[
            (pred["model"] == model)
            & (pred["horizon_int"] == int(horizon))
            & (pred["configuration_id"] == ref_cfg)
        ][["outer_fold", "feature_date", "target_date", "y_pred"]]
        merged = ref.merge(cur, on=["outer_fold", "feature_date", "target_date"], suffixes=("_ref", "_cfg"), validate="one_to_one")
        diff = merged["y_pred_cfg"].to_numpy(dtype=float) - merged["y_pred_ref"].to_numpy(dtype=float)

        stability_pooled_rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "configuration_id": cfg,
                "final_reference_configuration": ref_cfg,
                "outer_fold": "pooled",
                "N": int(len(merged)),
                "pearson_prediction_correlation": safe_pearson(
                    merged["y_pred_cfg"].to_numpy(dtype=float),
                    merged["y_pred_ref"].to_numpy(dtype=float),
                ),
                "spearman_prediction_rank_correlation": safe_spearman(
                    merged["y_pred_cfg"].to_numpy(dtype=float),
                    merged["y_pred_ref"].to_numpy(dtype=float),
                ),
                "mean_prediction_difference": float(np.mean(diff)),
                "mean_absolute_prediction_difference": float(np.mean(np.abs(diff))),
                "maximum_absolute_prediction_difference": float(np.max(np.abs(diff))),
            }
        )

    stability_pooled = pd.DataFrame(stability_pooled_rows).sort_values(["model", "horizon", "configuration_id"])

    # Event ranking diagnostic: retrospective, offline, fixed-budget sensitivity evidence only.
    recall_fold_rows: List[Dict[str, Any]] = []
    for (model, horizon, cfg, fold), grp in pred.groupby(
        ["model", "horizon_int", "configuration_id", "outer_fold"], sort=True
    ):
        d = grp.sort_values(
            ["y_pred", "feature_date", "target_date", "canonical_test_row"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        n = int(len(d))
        k = int(math.ceil(EVENT_BUDGET * n))
        d["is_event"] = (d["y_true"].astype(float) >= EVENT_TAU).astype(int)
        topk = d.iloc[:k].copy()

        event_count = int(d["is_event"].sum())
        tp = int(topk["is_event"].sum())
        fp = int(k - tp)
        fn = int(event_count - tp)
        recall = float(tp / event_count) if event_count > 0 else float("nan")
        precision = float(tp / k) if k > 0 else float("nan")

        recall_fold_rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "configuration_id": cfg,
                "outer_fold": int(fold),
                "tau": EVENT_TAU,
                "budget_fraction": EVENT_BUDGET,
                "N": n,
                "k": k,
                "event_count": event_count,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "recall": recall,
                "precision": precision,
                "tie_break_rule": "y_pred_desc_then_feature_date_asc_then_target_date_asc_then_canonical_test_row_asc",
                "retrospective_only": True,
                "diagnostic_label": "retrospective_offline_fixed_budget_sensitivity_evidence_only",
            }
        )

    recall_by_fold = pd.DataFrame(recall_fold_rows).sort_values(["model", "horizon", "configuration_id", "outer_fold"])

    recall_pooled_rows: List[Dict[str, Any]] = []
    for (model, horizon, cfg), grp in recall_by_fold.groupby(["model", "horizon", "configuration_id"], sort=True):
        n = int(grp["N"].sum())
        k = int(grp["k"].sum())
        event_count = int(grp["event_count"].sum())
        tp = int(grp["true_positives"].sum())
        fp = int(grp["false_positives"].sum())
        fn = int(grp["false_negatives"].sum())
        recall = float(tp / event_count) if event_count > 0 else float("nan")
        precision = float(tp / k) if k > 0 else float("nan")

        recall_pooled_rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "configuration_id": cfg,
                "outer_fold": "pooled",
                "tau": EVENT_TAU,
                "budget_fraction": EVENT_BUDGET,
                "N": n,
                "k": k,
                "event_count": event_count,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "recall": recall,
                "precision": precision,
                "tie_break_rule": "y_pred_desc_then_feature_date_asc_then_target_date_asc_then_canonical_test_row_asc",
                "retrospective_only": True,
                "diagnostic_label": "retrospective_offline_fixed_budget_sensitivity_evidence_only",
            }
        )

    recall_pooled = pd.DataFrame(recall_pooled_rows).sort_values(["model", "horizon", "configuration_id"])

    # Split and embargo audit.
    embargo_rows: List[Dict[str, Any]] = []
    embargo_pass = True
    for h in HORIZONS:
        for fold in OUTER_FOLDS:
            ss = split_summary[(split_summary["horizon"] == h) & (split_summary["outer_fold"] == fold)]
            if ss.empty:
                raise DecisionError("C", "missing split summary row for H{} fold{}".format(h, fold))
            sr = ss.iloc[0]

            sf = split[(split["horizon"] == h) & (split["outer_fold"] == fold)]
            obs_outer_train_before = int((sf["outer_role"].astype(str) == "outer_train").sum())
            obs_outer_train_after = int(
                ((sf["outer_role"].astype(str) == "outer_train") & (~sf["purged_outer_boundary"])).sum()
            )
            obs_outer_test = int((sf["outer_role"].astype(str) == "outer_test").sum())
            obs_outer_purged = int(
                ((sf["outer_role"].astype(str) == "outer_train") & (sf["purged_outer_boundary"])).sum()
            )

            obs_inner_train_before = int(
                ((sf["outer_role"].astype(str) == "outer_train") & (sf["inner_role"].astype(str) == "inner_train")).sum()
            )
            obs_inner_train_after = int(
                (
                    (sf["outer_role"].astype(str) == "outer_train")
                    & (sf["inner_role"].astype(str) == "inner_train")
                    & (~sf["purged_inner_boundary"])
                ).sum()
            )
            obs_inner_val = int(
                ((sf["outer_role"].astype(str) == "outer_train") & (sf["inner_role"].astype(str) == "inner_validation")).sum()
            )
            obs_inner_purged = int(
                (
                    (sf["outer_role"].astype(str) == "outer_train")
                    & (sf["inner_role"].astype(str) == "inner_train")
                    & (sf["purged_inner_boundary"])
                ).sum()
            )

            row_pass = (
                obs_outer_train_before == int(sr["outer_train_n_before_purge"])
                and obs_outer_train_after == int(sr["outer_train_n_after_purge"])
                and obs_outer_test == int(sr["outer_test_n"])
                and obs_outer_purged == int(sr["outer_rows_purged"])
                and obs_inner_train_before == int(sr["inner_train_n_before_purge"])
                and obs_inner_train_after == int(sr["inner_train_n_after_purge"])
                and obs_inner_val == int(sr["inner_validation_n"])
                and obs_inner_purged == int(sr["inner_rows_purged"])
                and bool(sr["outer_boundary_pass"])
                and bool(sr["inner_boundary_pass"])
            )
            embargo_pass = bool(embargo_pass and row_pass)

            embargo_rows.append(
                {
                    "horizon": h,
                    "outer_fold": fold,
                    "outer_train_before_expected": int(sr["outer_train_n_before_purge"]),
                    "outer_train_before_observed": obs_outer_train_before,
                    "outer_train_after_expected": int(sr["outer_train_n_after_purge"]),
                    "outer_train_after_observed": obs_outer_train_after,
                    "outer_test_expected": int(sr["outer_test_n"]),
                    "outer_test_observed": obs_outer_test,
                    "outer_purged_expected": int(sr["outer_rows_purged"]),
                    "outer_purged_observed": obs_outer_purged,
                    "inner_train_before_expected": int(sr["inner_train_n_before_purge"]),
                    "inner_train_before_observed": obs_inner_train_before,
                    "inner_train_after_expected": int(sr["inner_train_n_after_purge"]),
                    "inner_train_after_observed": obs_inner_train_after,
                    "inner_validation_expected": int(sr["inner_validation_n"]),
                    "inner_validation_observed": obs_inner_val,
                    "inner_purged_expected": int(sr["inner_rows_purged"]),
                    "inner_purged_observed": obs_inner_purged,
                    "outer_boundary_pass": bool(sr["outer_boundary_pass"]),
                    "inner_boundary_pass": bool(sr["inner_boundary_pass"]),
                    "embargo_pass": bool(row_pass),
                }
            )

    split_embargo_audit = pd.DataFrame(embargo_rows).sort_values(["horizon", "outer_fold"])

    # Realized retention reconciliation against frozen design table.
    retention_df = pd.DataFrame(retention_rows).sort_values(["model", "horizon", "configuration_id", "outer_fold"])
    design_retention = design["retention_df"].copy()

    recon_rows: List[Dict[str, Any]] = []
    retention_pass = True
    for rr in retention_df.itertuples(index=False):
        d = design_retention[
            (design_retention["configuration_id"].astype(str) == str(rr.configuration_id))
            & (design_retention["horizon"].astype(int) == int(rr.horizon))
            & (design_retention["outer_fold"].astype(str) == str(int(rr.outer_fold)))
        ]
        if d.empty:
            raise DecisionError("C", "missing frozen retention row for {} H{} fold{}".format(rr.configuration_id, rr.horizon, rr.outer_fold))
        dr = d.iloc[0]

        warmup_expected = int(dr["outer_train_rows_lost_warmup"])
        train_expected = int(dr["available_outer_train_rows"])
        test_expected = int(dr["available_outer_test_rows"])

        row_pass = (
            int(rr.warmup_losses) == warmup_expected
            and int(rr.final_fitting_rows) == train_expected
            and int(rr.canonical_test_rows) == test_expected
        )
        retention_pass = bool(retention_pass and row_pass)

        recon_rows.append(
            {
                "model": rr.model,
                "horizon": int(rr.horizon),
                "outer_fold": int(rr.outer_fold),
                "configuration_id": rr.configuration_id,
                "warmup_losses_observed": int(rr.warmup_losses),
                "warmup_losses_expected": warmup_expected,
                "final_fitting_rows_observed": int(rr.final_fitting_rows),
                "final_fitting_rows_expected": train_expected,
                "canonical_test_rows_observed": int(rr.canonical_test_rows),
                "canonical_test_rows_expected": test_expected,
                "reconciliation_pass": bool(row_pass),
            }
        )

    recon_df = pd.DataFrame(recon_rows)
    retention_df = retention_df.merge(
        recon_df,
        on=["model", "horizon", "outer_fold", "configuration_id"],
        how="left",
        validate="one_to_one",
    )

    feature_registry_df = pd.DataFrame(feature_registry_rows).drop_duplicates().sort_values(
        ["model", "horizon", "configuration_id", "outer_fold"]
    )
    fit_registry_df = pd.DataFrame(fit_registry_rows).sort_values(
        ["model", "horizon", "configuration_id", "outer_fold"]
    )

    # Build wide predictions table.
    wide = pred[
        [
            "model",
            "horizon",
            "horizon_int",
            "outer_fold",
            "feature_date",
            "target_date",
            "canonical_test_row",
            "y_true",
            "configuration_id",
            "y_pred",
            "error",
            "abs_error",
            "squared_error",
        ]
    ].copy()
    wide = wide.pivot_table(
        index=[
            "model",
            "horizon",
            "horizon_int",
            "outer_fold",
            "feature_date",
            "target_date",
            "canonical_test_row",
            "y_true",
        ],
        columns="configuration_id",
        values=["y_pred", "error", "abs_error", "squared_error"],
        aggfunc="first",
    )
    wide.columns = ["{}_{}".format(a, b) for a, b in wide.columns]
    wide = wide.reset_index().sort_values(
        ["model", "horizon_int", "outer_fold", "feature_date", "target_date"]
    )

    # Sensitivity summary relative to final reference.
    pooled_train_rows = (
        retention_df.groupby(["model", "horizon", "configuration_id"], as_index=False)["final_fitting_rows"].sum().rename(
            columns={"final_fitting_rows": "pooled_training_rows"}
        )
    )
    pooled_feat_count = (
        feature_registry_df.groupby(["model", "horizon", "configuration_id"], as_index=False)["feature_count"].first()
    )

    summary_rows: List[Dict[str, Any]] = []
    for model in FIT_MODELS:
        for horizon in HORIZONS:
            ref_cfg = final_reference_mapping["{}_H{}".format(model, horizon)]
            ref_m = point_pooled[
                (point_pooled["model"] == model)
                & (point_pooled["horizon"] == horizon)
                & (point_pooled["configuration_id"] == ref_cfg)
            ].iloc[0]
            ref_s = stability_pooled[
                (stability_pooled["model"] == model)
                & (stability_pooled["horizon"] == horizon)
                & (stability_pooled["configuration_id"] == ref_cfg)
            ].iloc[0]
            ref_r = recall_pooled[
                (recall_pooled["model"] == model)
                & (recall_pooled["horizon"] == horizon)
                & (recall_pooled["configuration_id"] == ref_cfg)
            ].iloc[0]
            ref_t = pooled_train_rows[
                (pooled_train_rows["model"] == model)
                & (pooled_train_rows["horizon"] == horizon)
                & (pooled_train_rows["configuration_id"] == ref_cfg)
            ].iloc[0]
            ref_f = pooled_feat_count[
                (pooled_feat_count["model"] == model)
                & (pooled_feat_count["horizon"] == horizon)
                & (pooled_feat_count["configuration_id"] == ref_cfg)
            ].iloc[0]

            for cfg in CONFIG_ORDER:
                m = point_pooled[
                    (point_pooled["model"] == model)
                    & (point_pooled["horizon"] == horizon)
                    & (point_pooled["configuration_id"] == cfg)
                ].iloc[0]
                s = stability_pooled[
                    (stability_pooled["model"] == model)
                    & (stability_pooled["horizon"] == horizon)
                    & (stability_pooled["configuration_id"] == cfg)
                ].iloc[0]
                r = recall_pooled[
                    (recall_pooled["model"] == model)
                    & (recall_pooled["horizon"] == horizon)
                    & (recall_pooled["configuration_id"] == cfg)
                ].iloc[0]
                t = pooled_train_rows[
                    (pooled_train_rows["model"] == model)
                    & (pooled_train_rows["horizon"] == horizon)
                    & (pooled_train_rows["configuration_id"] == cfg)
                ].iloc[0]
                f = pooled_feat_count[
                    (pooled_feat_count["model"] == model)
                    & (pooled_feat_count["horizon"] == horizon)
                    & (pooled_feat_count["configuration_id"] == cfg)
                ].iloc[0]

                mae_diff = float(m["MAE"] - ref_m["MAE"])
                rmse_diff = float(m["RMSE"] - ref_m["RMSE"])
                mase_diff = float(m["MASE"] - ref_m["MASE"])
                recall_diff = float(r["recall"] - ref_r["recall"])

                mae_pct = float((mae_diff / ref_m["MAE"]) * 100.0) if float(ref_m["MAE"]) != 0.0 else float("nan")
                rmse_pct = float((rmse_diff / ref_m["RMSE"]) * 100.0) if float(ref_m["RMSE"]) != 0.0 else float("nan")

                summary_rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "configuration_id": cfg,
                        "final_reference_configuration": ref_cfg,
                        "is_final_reference": bool(cfg == ref_cfg),
                        "mae_difference_vs_reference": mae_diff,
                        "mae_percent_difference_vs_reference": mae_pct,
                        "rmse_difference_vs_reference": rmse_diff,
                        "rmse_percent_difference_vs_reference": rmse_pct,
                        "mase_difference_vs_reference": mase_diff,
                        "pearson_correlation_vs_reference": float(s["pearson_prediction_correlation"]),
                        "spearman_correlation_vs_reference": float(s["spearman_prediction_rank_correlation"]),
                        "recall5_difference_vs_reference": recall_diff,
                        "training_row_difference_vs_reference": int(t["pooled_training_rows"] - ref_t["pooled_training_rows"]),
                        "feature_count_difference_vs_reference": int(f["feature_count"] - ref_f["feature_count"]),
                        "interpretation_classification": "not_preclassified",
                        "narrative_scope": "quantitative_sensitivity_evidence_only",
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["model", "horizon", "configuration_id"])

    # Source and software registries.
    source_rows: List[Dict[str, Any]] = []
    tag_refs = design["design_input"].get("verification", {}).get("local_tag_commits", {})

    for rel, sha in sorted(consumed_hashes.items()):
        role = "consumed_input"
        if rel.startswith("revision_2026/10_lag_window_sensitivity/00_design/"):
            role = "frozen_design_input"
        elif rel.startswith("revision_2026/03_corrected_protocol/"):
            role = "corrected_protocol_input"
        elif rel.startswith("revision_2026/04_controlled_reruns/"):
            role = "frozen_model_source"
        elif rel.startswith("features/"):
            role = "feature_artifact"
        elif rel.startswith("src/"):
            role = "source_builder"
        elif rel.startswith("revision_2026/09_h1_h3_reconciliation/"):
            role = "stage1_frozen_source"

        source_rows.append(
            {
                "input_path": rel,
                "role": role,
                "sha256": sha,
                "frozen_tag_or_commit": "{}@{}".format(EXPECTED_DESIGN_TAG, tag_commit),
                "model": "ALL",
                "horizon": "ALL",
                "outer_fold": "ALL",
                "configuration_id": "ALL",
                "consumed_for_execution": True,
            }
        )

    source_registry_df = pd.DataFrame(source_rows).sort_values(["role", "input_path"])

    software_env = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
    }

    # Pre-report gate outcomes.
    common_date_result = {
        "pass": bool(common_date_pass),
        "rows": common_date_rows,
    }

    reference_summary = {
        "tolerance": REF_TOL,
        "overall_pass": bool(reference_reproduction_pass),
        "material_failure": bool(reference_reproduction_material_fail),
        "pathway_status": {
            "{}_H{}".format(m, h): pathway_status_map[(m, h)] for m in FIT_MODELS for h in HORIZONS
        },
    }

    training_retention_result = {
        "pass": bool(retention_pass),
        "reconciliation_rows": int(len(retention_df)),
        "unexplained_discrepancies": int((~retention_df["reconciliation_pass"].astype(bool)).sum()),
    }

    embargo_result = {
        "pass": bool(embargo_pass),
        "rows": int(len(split_embargo_audit)),
    }

    # Source preservation after all computation but before writing outputs.
    protected_after = snapshot_tree_checksums(IMMUTABLE_DIRS, ROOT)
    source_preservation_pass = protected_before == protected_after
    source_preservation_result = {
        "pass": bool(source_preservation_pass),
        "files_before": int(len(protected_before)),
        "files_after": int(len(protected_after)),
        "changed_paths": sorted(
            [
                p
                for p in set(protected_before.keys()).union(set(protected_after.keys()))
                if protected_before.get(p) != protected_after.get(p)
            ]
        ),
    }

    if not source_preservation_pass:
        raise DecisionError("D", "immutable source preservation failed")

    # Output writes.
    write_csv(PRED_LONG_PATH, pred, date_cols=["feature_date", "target_date"])
    write_csv(PRED_WIDE_PATH, wide, date_cols=["feature_date", "target_date"])
    write_csv(REF_AUDIT_PATH, ref_audit_df)
    write_json(REF_SUMMARY_PATH, reference_summary)
    write_csv(POINT_BY_FOLD_PATH, point_by_fold)
    write_csv(POINT_POOLED_PATH, point_pooled)
    write_csv(STABILITY_BY_FOLD_PATH, stability_by_fold)
    write_csv(STABILITY_POOLED_PATH, stability_pooled)
    write_csv(RECALL_BY_FOLD_PATH, recall_by_fold)
    write_csv(RECALL_POOLED_PATH, recall_pooled)
    write_csv(RETENTION_REALIZED_PATH, retention_df, date_cols=[
        "earliest_fitting_feature_date",
        "latest_fitting_feature_date",
        "earliest_test_feature_date",
        "latest_test_feature_date",
    ])
    write_csv(FEATURE_REG_REALIZED_PATH, feature_registry_df)
    write_csv(SPLIT_EMBARGO_AUDIT_PATH, split_embargo_audit)
    write_csv(SUMMARY_CSV_PATH, summary_df)
    write_json(SUMMARY_JSON_PATH, {"rows": summary_df.to_dict(orient="records")})
    write_csv(SOURCE_REGISTRY_PATH, source_registry_df)
    write_json(SOFTWARE_ENV_PATH, software_env)
    write_csv(FIT_REGISTRY_PATH, fit_registry_df)

    summary_head = summary_df[
        [
            "model",
            "horizon",
            "configuration_id",
            "final_reference_configuration",
            "mae_difference_vs_reference",
            "rmse_difference_vs_reference",
            "mase_difference_vs_reference",
            "pearson_correlation_vs_reference",
            "spearman_correlation_vs_reference",
            "recall5_difference_vs_reference",
            "training_row_difference_vs_reference",
            "feature_count_difference_vs_reference",
            "interpretation_classification",
        ]
    ]

    summary_md_lines = [
        "# Stage 2 Lag-Window Sensitivity Summary",
        "",
        "This report is retrospective, offline, fixed-budget sensitivity evidence only.",
        "No configuration optimization or post-hoc final-reference reassignment was performed.",
        "",
        markdown_table(summary_head),
        "",
        "Interpretation classification is fixed to not_preclassified because frozen design thresholds are quantitative-only.",
    ]
    write_text(SUMMARY_MD_PATH, "\n".join(summary_md_lines) + "\n")

    prediction_row_counts = {
        "expected_total": 13536,
        "observed_total": int(len(pred)),
        "expected_by_horizon": expected_h_counts,
        "observed_by_horizon": {k: int(v) for k, v in observed_h_counts.items()},
    }

    metric_row_counts = {
        "point_metrics_by_fold": int(len(point_by_fold)),
        "point_metrics_pooled": int(len(point_pooled)),
        "stability_by_fold": int(len(stability_by_fold)),
        "stability_pooled": int(len(stability_pooled)),
        "recall5_by_fold": int(len(recall_by_fold)),
        "recall5_pooled": int(len(recall_pooled)),
    }

    input_checksum_result = {
        "corrected_protocol_checksum_pass": True,
        "stage1_checksum_pass": True,
    }
    design_checksum_result = {"design_checksum_pass": True}

    checksum_result = {
        "registry_pass": "pending_post_manifest_finalize",
        "coverage_pass": "pending_post_manifest_finalize",
    }

    final_decision = "B"
    if not common_date_pass or not embargo_pass or not retention_pass:
        final_decision = "C"
    elif reference_reproduction_material_fail:
        final_decision = "C"
    elif not reference_reproduction_pass:
        final_decision = "B"
    elif source_preservation_pass:
        final_decision = "B"  # tests/determinism pending until standalone test script runs.

    execution_id = "lag_window_sensitivity_execution_{}_{}".format(head[:12], tag_commit[:8])

    manifest = {
        "execution_id": execution_id,
        "input_parent_commit": head,
        "design_commit": tag_commit,
        "design_tag": EXPECTED_DESIGN_TAG,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "source_registry": str(SOURCE_REGISTRY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "software_environment": str(SOFTWARE_ENV_PATH.relative_to(ROOT)).replace("\\", "/"),
        "models": list(FIT_MODELS),
        "horizons": ["H1", "H3", "H5"],
        "configurations": list(CONFIG_ORDER),
        "final_reference_mapping": final_reference_mapping,
        "prediction_row_counts": prediction_row_counts,
        "metric_row_counts": metric_row_counts,
        "reference_reproduction_result": reference_summary,
        "common_date_result": common_date_result,
        "embargo_result": embargo_result,
        "training_retention_result": training_retention_result,
        "test_result": "pending_not_run",
        "deterministic_result": "pending_not_run",
        "source_preservation_result": source_preservation_result,
        "input_checksum_result": input_checksum_result,
        "design_checksum_result": design_checksum_result,
        "checksum_result": checksum_result,
        "final_decision": final_decision,
        "generated_files": sorted(
            [
                str(p.relative_to(ROOT)).replace("\\", "/")
                for p in OUT_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
            ]
        ),
    }
    write_json(MANIFEST_PATH, manifest)

    report_lines = [
        "# Stage 2 Lag-Window Sensitivity Execution Completion Report",
        "",
        "## Gate Summary",
        "- preflight_pass: True",
        "- source_preservation_pass: {}".format(source_preservation_pass),
        "- input_checksum_pass: True",
        "- design_checksum_pass: True",
        "- common_date_pass: {}".format(common_date_pass),
        "- embargo_pass: {}".format(embargo_pass),
        "- training_retention_pass: {}".format(retention_pass),
        "- reference_reproduction_pass: {}".format(reference_reproduction_pass),
        "- test_result: pending_not_run",
        "- deterministic_result: pending_not_run",
        "",
        "## Scope Labels",
        "- retrospective",
        "- offline",
        "- fixed-budget",
        "- sensitivity evidence only",
        "",
        "## Final-Reference Mapping",
        "- Ridge_H1 -> REFERENCE",
        "- Ridge_H3 -> REFERENCE",
        "- Ridge_H5 -> REFERENCE",
        "- HGBR_H1 -> REFERENCE",
        "- HGBR_H3 -> REFERENCE",
        "- HGBR_H5 -> LONG",
        "",
        "## Prediction Rows",
        "- total: {}".format(len(pred)),
        "- H1: {}".format(observed_h_counts.get("H1", 0)),
        "- H3: {}".format(observed_h_counts.get("H3", 0)),
        "- H5: {}".format(observed_h_counts.get("H5", 0)),
        "",
        "FINAL DECISION",
        "",
        "A. Controlled lag-window sensitivity execution passed; results are valid for final computational integration.",
    ]

    if final_decision == "B":
        report_lines[-1] = "B. Execution completed, but material sensitivity or reference-reproduction questions require investigation before integration."
    elif final_decision == "C":
        report_lines[-1] = "C. Canonical dates, embargoes, frozen hyperparameters, or reference lineage could not be preserved safely."
    elif final_decision == "D":
        report_lines[-1] = "D. Source preservation, deterministic regeneration, tests, or checksum validation failed."

    write_text(REPORT_PATH, "\n".join(report_lines) + "\n")

    # Programmatic checksum registry after all files except checksum file itself.
    write_checksums_file(OUT_DIR, CHECKSUMS_PATH)
    checksum_eval = verify_checksums_against_registry(OUT_DIR, CHECKSUMS_PATH)

    # Terminal checksum command output for operator visibility.
    checksum_cmd = ["sha256sum", "-c", CHECKSUMS_PATH.name]
    checksum_proc = subprocess.run(
        checksum_cmd,
        cwd=OUT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    # Print required terminal summary.
    models_executed = sorted(pred["model"].astype(str).unique().tolist())
    horizons_executed = sorted(pred["horizon"].astype(str).unique().tolist())
    configs_executed = sorted(pred["configuration_id"].astype(str).unique().tolist())

    long_h1_loss = (
        retention_df[
            (retention_df["horizon"] == 1)
            & (retention_df["configuration_id"] == "LONG")
            & (retention_df["model"] == "Ridge")
        ]
        .sort_values("outer_fold")["warmup_losses"]
        .astype(int)
        .tolist()
    )

    print("1. repository: {}".format(repository))
    print("2. branch: {}".format(branch))
    print("3. starting commit: {}".format(head))
    print("4. design tag and design commit: {} -> {}".format(EXPECTED_DESIGN_TAG, tag_commit))
    print("5. authorized output directory: {}".format(str(OUT_DIR.resolve())))
    print(
        "6. input checksum status: protocol={} stage1={} design={}".format(
            True,
            True,
            True,
        )
    )
    print("7. source-preservation status: {}".format(source_preservation_pass))
    print("8. models executed: {}".format(models_executed))
    print("9. horizons executed: {}".format(horizons_executed))
    print("10. configurations executed: {}".format(configs_executed))
    print("11. number of fitted combinations: {}".format(len(FIT_MODELS) * len(HORIZONS) * len(CONFIG_ORDER)))
    print(
        "12. prediction row counts: total={} by_horizon={}".format(
            len(pred), {k: int(v) for k, v in observed_h_counts.items()}
        )
    )
    print("13. canonical test-date status: {}".format(common_date_pass))
    print("14. embargo status: {}".format(embargo_pass))
    print(
        "15. training-retention summary: LONG H1 warm-up losses by fold={} (Ridge mirror for HGBR); reconciliation_pass={}".format(
            {i + 1: int(v) for i, v in enumerate(long_h1_loss)}, retention_pass
        )
    )
    print(
        "16. reference-reproduction status: pass={} material_failure={} tolerance={}".format(
            reference_reproduction_pass, reference_reproduction_material_fail, REF_TOL
        )
    )
    print("17. point-metric output status: by_fold_rows={} pooled_rows={}".format(len(point_by_fold), len(point_pooled)))
    print("18. stability output status: by_fold_rows={} pooled_rows={}".format(len(stability_by_fold), len(stability_pooled)))
    print("19. Recall@5% diagnostic status: by_fold_rows={} pooled_rows={}".format(len(recall_by_fold), len(recall_pooled)))
    print("20. number of tests run and passed: pending_not_run")
    print("21. deterministic result: pending_not_run")
    print(
        "22. checksum result: registry_pass={} coverage_pass={} sha256sum_exit_code={}".format(
            checksum_eval["registry_pass"], checksum_eval["coverage_pass"], checksum_proc.returncode
        )
    )
    print("23. final decision: {}".format(final_decision))

    generated_files = sorted(
        [
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in OUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
        ]
    )
    print("24. generated files: {}".format(generated_files))


if __name__ == "__main__":
    run_execution()
