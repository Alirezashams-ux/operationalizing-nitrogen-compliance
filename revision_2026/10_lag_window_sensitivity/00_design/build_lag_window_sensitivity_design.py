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


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
AUTHORIZED_REL_DIR = "revision_2026/10_lag_window_sensitivity/00_design"

EXPECTED_BRANCH = "controlled-reruns-v1"
EXPECTED_HEAD = "15cc6f8414df3e60692700963afc296e7653236e"
EXPECTED_STAGE1_TAG = "canonical-h1-h3-reconciliation-v1"
EXPECTED_PYTHON_VERSION = "3.8.10"
EXPECTED_DATASET_SHA256 = "6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb"
EXPECTED_SPLIT_SHA256 = "f83baf5732ffbf88b9a448a3d9ab9fa270f7495fe44d1559c1ac49795a30881a"

HORIZONS = (1, 3, 5)
FOLDS = (1, 2, 3)
MODELS = ("Persistence", "Ridge", "ElasticNet", "HGBR", "BCR-TCN")

CONFIGS = [
    {
        "configuration_id": "SHORT",
        "configuration_label": "SHORT",
        "tnout_features": ["TNout_lag1", "TNout_roll7"],
        "maximum_required_history_days": 7,
        "scientific_role": "reduced-memory sensitivity",
        "predeclared": True,
        "uses_future_information": False,
        "notes": "Shortest TN-memory candidate; strictly prior TNout only.",
    },
    {
        "configuration_id": "REFERENCE",
        "configuration_label": "REFERENCE",
        "tnout_features": ["TNout_lag1", "TNout_roll7", "TNout_roll14"],
        "maximum_required_history_days": 14,
        "scientific_role": "ordinary corrected feature reference",
        "predeclared": True,
        "uses_future_information": False,
        "notes": "Reference TN-memory used by final corrected Ridge and HGBR H1/H3.",
    },
    {
        "configuration_id": "LONG",
        "configuration_label": "LONG",
        "tnout_features": [
            "TNout_lag1",
            "TNout_lag3",
            "TNout_lag5",
            "TNout_lag7",
            "TNout_roll7",
            "TNout_roll14",
            "TNout_roll30",
        ],
        "maximum_required_history_days": 30,
        "scientific_role": "extended-memory sensitivity and final HGBR-H5/BCR-TCN-H5 TN-memory lineage",
        "predeclared": True,
        "uses_future_information": False,
        "notes": "Final corrected HGBR-H5/BCR-TCN-H5 TN-memory lineage.",
    },
]

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
RERUNS_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
STAGE1_DIR = ROOT / "revision_2026" / "09_h1_h3_reconciliation"

DATASET_LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
STAGE1_CHECKSUM_PATH = STAGE1_DIR / "canonical_h1_h3_reconciliation_checksums.sha256"
STAGE1_MANIFEST_PATH = STAGE1_DIR / "canonical_h1_h3_reconciliation_manifest.json"

RIDGE_DIR = RERUNS_DIR / "ridge"
ELASTICNET_DIR = RERUNS_DIR / "elasticnet"
HGBR_DIR = RERUNS_DIR / "hgbr"
PERSISTENCE_DIR = RERUNS_DIR / "persistence"
BCR_DIR = RERUNS_DIR / "bcr_tcn_v11_h5"

RIDGE_MANIFEST_PATH = RIDGE_DIR / "ridge_run_manifest.json"
ELASTICNET_MANIFEST_PATH = ELASTICNET_DIR / "elasticnet_run_manifest.json"
HGBR_MANIFEST_PATH = HGBR_DIR / "hgbr_run_manifest.json"
PERSISTENCE_MANIFEST_PATH = PERSISTENCE_DIR / "persistence_run_manifest.json"
BCR_MANIFEST_PATH = BCR_DIR / "bcr_tcn_v11_h5_run_manifest.json"

RIDGE_PLAN_PATH = RIDGE_DIR / "ridge_selection_plan.json"
ELASTICNET_PLAN_PATH = ELASTICNET_DIR / "elasticnet_selection_plan.json"
HGBR_PLAN_PATH = HGBR_DIR / "hgbr_selection_plan.json"
BCR_FIXED_CONFIG_PATH = BCR_DIR / "bcr_tcn_v11_fixed_configuration.json"

RIDGE_RUNNER_PATH = RIDGE_DIR / "run_corrected_ridge.py"
ELASTICNET_RUNNER_PATH = ELASTICNET_DIR / "run_corrected_elasticnet.py"
HGBR_RUNNER_PATH = HGBR_DIR / "run_corrected_hgbr.py"
BCR_RUNNER_PATH = BCR_DIR / "run_corrected_bcr_tcn_v11_h5.py"

SRC_BUILDERS = [
    ROOT / "src" / "build_ulsan_npz_H1.py",
    ROOT / "src" / "build_ulsan_npz.py",
    ROOT / "src" / "build_ulsan_npz_v2.py",
]

NPZ_PATHS = {
    "H1_ordinary": ROOT / "features" / "ulsan_H1_features.npz",
    "H3_ordinary": ROOT / "features" / "ulsan_H3_features.npz",
    "H3_v2": ROOT / "features" / "ulsan_H3_features_v2.npz",
    "H5_ordinary": ROOT / "features" / "ulsan_H5_features.npz",
    "H5_v2": ROOT / "features" / "ulsan_H5_features_v2.npz",
}

EXPECTED_OUTER_TEST_COUNTS = {
    1: {1: 254, 2: 254, 3: 254, "pooled": 762},
    3: {1: 249, 2: 249, 3: 249, "pooled": 747},
    5: {1: 249, 2: 249, 3: 249, "pooled": 747},
}

SOURCE_TAG_PATHS = {
    "corrected-protocol-v1": ROOT / "revision_2026" / "03_corrected_protocol",
    "persistence-corrected-v1": ROOT / "revision_2026" / "04_controlled_reruns" / "persistence",
    "ridge-corrected-v1": ROOT / "revision_2026" / "04_controlled_reruns" / "ridge",
    "elasticnet-corrected-v1": ROOT / "revision_2026" / "04_controlled_reruns" / "elasticnet",
    "hgbr-corrected-v1": ROOT / "revision_2026" / "04_controlled_reruns" / "hgbr",
    "canonical-h1-h3-reconciliation-v1": ROOT / "revision_2026" / "09_h1_h3_reconciliation",
}

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
FEATURE_LINEAGE_CSV_PATH = OUT_DIR / "feature_lineage_audit.csv"
FEATURE_LINEAGE_MD_PATH = OUT_DIR / "feature_lineage_audit.md"
FEATURE_ARTIFACT_REGISTRY_PATH = OUT_DIR / "feature_artifact_registry.csv"
CANDIDATE_CONFIG_PATH = OUT_DIR / "candidate_memory_configurations.csv"
MODEL_HORIZON_REGISTRY_PATH = OUT_DIR / "model_horizon_reference_registry.csv"
COMMON_DATE_FEASIBILITY_PATH = OUT_DIR / "common_date_feasibility.csv"
TRAINING_RETENTION_PATH = OUT_DIR / "training_retention_audit.csv"
DESIGN_MD_PATH = OUT_DIR / "lag_window_sensitivity_design.md"
DESIGN_JSON_PATH = OUT_DIR / "lag_window_sensitivity_design.json"
MANIFEST_PATH = OUT_DIR / "lag_window_sensitivity_design_manifest.json"
REPORT_PATH = OUT_DIR / "lag_window_sensitivity_design_completion_report.md"
CHECKSUM_PATH = OUT_DIR / "lag_window_sensitivity_design_checksums.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise")


def parse_checksum_manifest(path: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            raise RuntimeError("Malformed checksum line in {}: {}".format(path, line))
        rows[parts[-1]] = parts[0]
    return rows


def run_shell(command: str, cwd: Path) -> Dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, shell=True, text=True, capture_output=True)
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def git_output(args: Sequence[str]) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def worktree_paths() -> List[str]:
    txt = git_output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    rows = [r for r in txt.splitlines() if r.strip()]
    out: List[str] = []
    for r in rows:
        if len(r) < 4:
            continue
        p = r[3:].strip()
        if " -> " in p:
            p = p.split(" -> ")[-1]
        out.append(p)
    return out


def ensure_inside_authorized(path: Path) -> None:
    rp = path.resolve()
    auth = (ROOT / AUTHORIZED_REL_DIR).resolve()
    if not str(rp).startswith(str(auth)):
        raise RuntimeError("Attempted write outside authorized directory: {}".format(rp))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_inside_authorized(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame, date_cols: Sequence[str]) -> None:
    out = df.copy()
    for c in date_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    ensure_inside_authorized(path)
    out.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    ensure_inside_authorized(path)
    path.write_text(text, encoding="utf-8")


def rel_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def rel_to_stage(path: Path) -> str:
    return str(path.resolve().relative_to(OUT_DIR.resolve())).replace("\\", "/")


def sorted_join(xs: Iterable[str]) -> str:
    return "; ".join(sorted(list(xs)))


def get_existing_test_result() -> str:
    if not MANIFEST_PATH.exists():
        return "pending_not_run"
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return str(payload.get("test_result", "pending_not_run"))
    except Exception:
        return "pending_not_run"


def compute_decision(
    preflight_pass: bool,
    lineage_established_pass: bool,
    common_date_pass: bool,
    source_preservation_pass: bool,
    checksum_pass: bool,
    deterministic_result: str,
    test_result: str,
) -> str:
    if (not preflight_pass) or (not source_preservation_pass):
        return "D"
    if not lineage_established_pass:
        return "C"
    if str(test_result).startswith("fail"):
        return "D"
    if str(deterministic_result).startswith("fail"):
        return "D"
    if not common_date_pass:
        return "B"
    if str(test_result).startswith("pass") and checksum_pass:
        return "A"
    return "B"


def get_consumed_files() -> List[Path]:
    files = [
        DATASET_LOCK_PATH,
        PROTOCOL_SHA_PATH,
        SPLIT_PATH,
        SPLIT_SUMMARY_PATH,
        STAGE1_CHECKSUM_PATH,
        STAGE1_MANIFEST_PATH,
        RIDGE_MANIFEST_PATH,
        ELASTICNET_MANIFEST_PATH,
        HGBR_MANIFEST_PATH,
        PERSISTENCE_MANIFEST_PATH,
        BCR_MANIFEST_PATH,
        RIDGE_PLAN_PATH,
        ELASTICNET_PLAN_PATH,
        HGBR_PLAN_PATH,
        BCR_FIXED_CONFIG_PATH,
        RIDGE_RUNNER_PATH,
        ELASTICNET_RUNNER_PATH,
        HGBR_RUNNER_PATH,
        BCR_RUNNER_PATH,
    ]
    files.extend(SRC_BUILDERS)
    files.extend(list(NPZ_PATHS.values()))
    return sorted(list(dict.fromkeys(files)))


def inspect_npz(path: Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    key_list = sorted(list(z.files))
    shapes: Dict[str, Any] = {}
    dtypes: Dict[str, str] = {}
    for k in key_list:
        arr = np.asarray(z[k])
        shapes[k] = list(arr.shape)
        dtypes[k] = str(arr.dtype)

    feature_names = [str(x) for x in list(z["feature_names"]) if x is not None]
    dates = pd.to_datetime(pd.Series(z["dates"]), errors="raise")

    return {
        "relative_path": rel_to_root(path),
        "sha256": sha256_file(path),
        "keys": key_list,
        "shapes": shapes,
        "dtypes": dtypes,
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "date_start": str(dates.min().strftime("%Y-%m-%d")),
        "date_end": str(dates.max().strftime("%Y-%m-%d")),
        "row_count": int(len(dates)),
        "unique_date_count": int(dates.nunique()),
    }


def build_ordinary_feature_frame(dataset_df: pd.DataFrame, horizon: int, feature_cols: List[str]) -> pd.DataFrame:
    df = dataset_df.copy()

    for c in ["TNout", "Inflow", "TNin", "TOCin", "temp_mean_c", "precip_total_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "BODin" not in df.columns:
        df["BODin"] = np.nan
    df["BODin"] = pd.to_numeric(df["BODin"], errors="coerce")

    df["C_N"] = df["TOCin"] / (df["TNin"] + 1e-6)

    df["feature_date"] = pd.to_datetime(df["Date"], errors="raise")
    df["target_date"] = df["feature_date"] + pd.to_timedelta(int(horizon), unit="D")
    df["y_true"] = df["TNout"].shift(-int(horizon))

    df["TNout_lag1"] = df["TNout"].shift(1)
    df["TNout_roll7"] = df["TNout"].shift(1).rolling(7).mean()
    df["TNout_roll14"] = df["TNout"].shift(1).rolling(14).mean()

    for col in ["Inflow", "TNin", "TOCin", "BODin"]:
        df[col + "_roll7"] = pd.to_numeric(df[col], errors="coerce").rolling(7).mean()

    df["temp_roll7"] = pd.to_numeric(df["temp_mean_c"], errors="coerce").rolling(7).mean()
    df["precip_sum3"] = pd.to_numeric(df["precip_total_mm"], errors="coerce").rolling(3).sum()

    doy = df["feature_date"].dt.dayofyear.values
    df["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError("Missing ordinary feature columns for H{}: {}".format(horizon, missing))

    out = df[["feature_date", "target_date", "y_true"] + feature_cols].copy()
    out = out.dropna(subset=feature_cols + ["y_true"]).sort_values("feature_date").reset_index(drop=True)
    return out


def build_v2_feature_frame(dataset_df: pd.DataFrame, horizon: int, feature_cols: List[str]) -> pd.DataFrame:
    df = dataset_df.copy()

    for c in ["TNout", "Inflow", "TNin", "TOCin", "temp_mean_c", "precip_total_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "BODin" not in df.columns:
        df["BODin"] = np.nan
    df["BODin"] = pd.to_numeric(df["BODin"], errors="coerce")

    df["C_N"] = df["TOCin"] / (df["TNin"] + 1e-6)

    df["feature_date"] = pd.to_datetime(df["Date"], errors="raise")
    df["target_date"] = df["feature_date"] + pd.to_timedelta(int(horizon), unit="D")
    df["y_true"] = df["TNout"].shift(-int(horizon))

    df["TNout_lag1"] = df["TNout"].shift(1)
    df["TNout_lag3"] = df["TNout"].shift(3)
    df["TNout_lag5"] = df["TNout"].shift(5)
    df["TNout_lag7"] = df["TNout"].shift(7)

    df["TNout_roll7"] = df["TNout"].shift(1).rolling(7).mean()
    df["TNout_roll14"] = df["TNout"].shift(1).rolling(14).mean()
    df["TNout_roll30"] = df["TNout"].shift(1).rolling(30).mean()

    for col in ["Inflow", "TNin", "TOCin", "BODin"]:
        s = pd.to_numeric(df[col], errors="coerce")
        df[col + "_roll7"] = s.rolling(7).mean()
        df[col + "_roll14"] = s.rolling(14).mean()

    temp = pd.to_numeric(df["temp_mean_c"], errors="coerce")
    prcp = pd.to_numeric(df["precip_total_mm"], errors="coerce")
    df["temp_roll7"] = temp.rolling(7).mean()
    df["temp_roll14"] = temp.rolling(14).mean()
    df["temp_roll30"] = temp.rolling(30).mean()
    df["precip_sum3"] = prcp.rolling(3).sum()
    df["precip_sum7"] = prcp.rolling(7).sum()
    df["precip_sum14"] = prcp.rolling(14).sum()

    df["C_N_roll14"] = pd.to_numeric(df["C_N"], errors="coerce").rolling(14).mean()

    df["Inflow_x_precip3"] = pd.to_numeric(df["Inflow"], errors="coerce") * df["precip_sum3"]
    df["temp_x_CN"] = temp * df["C_N"]
    df["Inflow_x_TNin"] = pd.to_numeric(df["Inflow"], errors="coerce") * pd.to_numeric(df["TNin"], errors="coerce")

    doy = df["feature_date"].dt.dayofyear.values
    df["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError("Missing v2 feature columns for H{}: {}".format(horizon, missing))

    out = df[["feature_date", "target_date", "y_true"] + feature_cols].copy()
    out = out.dropna(subset=feature_cols + ["y_true"]).sort_values("feature_date").reset_index(drop=True)
    return out


def parse_feature_groups(feature_cols: List[str]) -> Dict[str, Any]:
    tnout_lags = sorted([c for c in feature_cols if c.startswith("TNout_lag")])
    tnout_rolls = sorted([c for c in feature_cols if c.startswith("TNout_roll")])
    non_tnout_rolls = sorted(
        [
            c
            for c in feature_cols
            if (c not in tnout_lags and c not in tnout_rolls)
            and (("_roll" in c) or c.startswith("precip_sum"))
        ]
    )

    return {
        "tnout_lags": tnout_lags,
        "tnout_rolls": tnout_rolls,
        "non_tnout_rolls": non_tnout_rolls,
    }


def build_checksums_registry() -> None:
    lines: List[str] = []
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == CHECKSUM_PATH.resolve():
            continue
        if p.suffix.lower() not in {".py", ".csv", ".json", ".md"}:
            continue
        lines.append("{}  {}".format(sha256_file(p), rel_to_stage(p)))
    write_text(CHECKSUM_PATH, "\n".join(lines) + "\n")


def verify_checksums_registry() -> Dict[str, Any]:
    res = run_shell("sha256sum -c {}".format(CHECKSUM_PATH.name), cwd=OUT_DIR)

    expected = sorted(
        [
            rel_to_stage(p)
            for p in OUT_DIR.rglob("*")
            if p.is_file()
            and p.resolve() != CHECKSUM_PATH.resolve()
            and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
        ]
    )
    listed = sorted(parse_checksum_manifest(CHECKSUM_PATH).keys()) if CHECKSUM_PATH.exists() else []

    return {
        "exit_code": int(res["exit_code"]),
        "stdout": res["stdout"],
        "stderr": res["stderr"],
        "registry_pass": bool(res["exit_code"] == 0),
        "coverage_pass": bool(set(expected) == set(listed)),
        "coverage_missing": sorted(list(set(expected) - set(listed))),
        "coverage_extra": sorted(list(set(listed) - set(expected))),
    }


def build_feature_lineage_markdown(lineage_df: pd.DataFrame, npz_registry_df: pd.DataFrame) -> str:
    def df_to_markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "(no rows)"
        cols = [str(c) for c in df.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows: List[str] = []
        for row in df.itertuples(index=False):
            vals = []
            for v in row:
                txt = str(v)
                txt = txt.replace("\n", " ").replace("|", "\\|")
                vals.append(txt)
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + rows)

    lines: List[str] = []
    lines.append("# Feature Lineage Audit")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Models covered: Persistence, Ridge, ElasticNet, HGBR, BCR-TCN (H5 context row).")
    lines.append("- Horizons covered: H1, H3, H5.")
    lines.append("- This is design-only lineage and feasibility auditing; no training/tuning/prediction was executed.")
    lines.append("")
    lines.append("## Core Construction Rules")
    lines.append("- TNout_lagk(t) = TNout(t-k).")
    lines.append("- TNout_rollw(t) = mean(TNout(t-w), ..., TNout(t-1)); day t is excluded.")
    lines.append("- Non-TN rolling predictors in ordinary/v2 builders use rolling windows anchored at day t (include day t).")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("- H1/H3 ordinary corrected lineage uses TNout_lag1 + TNout_roll7 + TNout_roll14 with 18 features.")
    lines.append("- H5 v2 lineage uses expanded TN memory (lags 1/3/5/7 and rolls 7/14/30) with 34 features.")
    lines.append("- ulsan_H3_features_v2.npz exists but final corrected H3 models do not consume it.")
    lines.append("- Persistence has no adjustable lag-window feature set.")
    lines.append("")
    lines.append("## NPZ Inspection")
    for r in npz_registry_df.itertuples(index=False):
        lines.append(
            "- {}: keys={} X_shape={} y_shape={} feature_count={} date_range={}..{} sha256={}".format(
                r.relative_path,
                r.keys,
                r.X_shape,
                r.y_shape,
                int(r.feature_count),
                r.date_start,
                r.date_end,
                r.sha256,
            )
        )

    lines.append("")
    lines.append("## Lineage Table")
    lines.append(df_to_markdown_table(lineage_df))
    lines.append("")
    return "\n".join(lines)


def build_design_markdown(design_payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Lag-Window Sensitivity Design (Stage 2, Design Gate)")
    lines.append("")
    lines.append("## Objective")
    lines.append("- Establish a predeclared, leakage-safe TNout-memory sensitivity design without refitting in this stage.")
    lines.append("")
    lines.append("## Reviewer Issue Addressed")
    lines.append("- Assess whether forecast robustness depends materially on TNout memory depth while preserving corrected protocol constraints.")
    lines.append("")
    lines.append("## Authoritative Feature Lineage")
    lines.append("- Ordinary corrected (H1/H3 and Ridge/ElasticNet H5): TNout_lag1 + TNout_roll7 + TNout_roll14.")
    lines.append("- V2 corrected H5 lineage (HGBR/BCR-TCN): TNout_lag1/3/5/7 + TNout_roll7/14/30.")
    lines.append("- H3 v2 artifact exists but is not used by final corrected H3.")
    lines.append("")
    lines.append("## Configuration Definitions")
    for cfg in design_payload["candidate_configurations"]:
        lines.append(
            "- {}: TNout features={} max_history_days={} earliest_possible_date={} role={}".format(
                cfg["configuration_id"],
                cfg["tnout_features"],
                cfg["maximum_required_history_days"],
                cfg["earliest_possible_date_from_raw_start"],
                cfg["scientific_role"],
            )
        )
    lines.append("")
    lines.append("## Model and Horizon Selection")
    lines.append("- Fitted sensitivity models (future execution only): Ridge and HGBR across H1/H3/H5.")
    lines.append("- Context-only baseline: Persistence (no fitting).")
    lines.append("- ElasticNet: frozen contextual evidence only; no refit in minimal design.")
    lines.append("- BCR-TCN v1.1: lineage documented for H5 LONG context only; not refit in minimal design.")
    lines.append("")
    lines.append("## Final Reference Mapping")
    lines.append("- Ridge H1/H3/H5 -> REFERENCE")
    lines.append("- HGBR H1/H3 -> REFERENCE")
    lines.append("- HGBR H5 -> LONG")
    lines.append("")
    lines.append("## Non-TN Feature Freeze Rule")
    lines.append("- Preserve final corrected non-TN predictors for each model/horizon.")
    lines.append("- Vary only TNout memory columns between SHORT/REFERENCE/LONG.")
    lines.append("- Do not add/remove weather, influent, or seasonal predictors during sensitivity runs.")
    lines.append("")
    lines.append("## Hyperparameter Freeze Rule")
    lines.append("- Preserve corrected fold-specific hyperparameters from frozen rerun manifests.")
    lines.append("- No retuning, no inner/outer-test hyperparameter search in sensitivity execution.")
    lines.append("")
    lines.append("## Common-Date Requirement")
    lines.append("- Canonical outer-test dates must be identical across configurations.")
    lines.append("- Any configuration requiring canonical test-date removal is not authorized.")
    lines.append("")
    lines.append("## Warm-Up and Sample-Retention Expectations")
    lines.append("- SHORT warm-up=7 days; REFERENCE warm-up=14 days; LONG warm-up=30 days from raw start 2021-01-04.")
    lines.append("- Computed earliest dates: SHORT=2021-01-11, REFERENCE=2021-01-18, LONG=2021-02-03.")
    lines.append("- LONG may reduce early H1 training availability while preserving canonical H1 outer-test dates.")
    lines.append("")
    lines.append("## Planned Outcomes")
    lines.append("- Future execution will report MAE/MSE/RMSE/MASE and reference-correlations (Pearson/Spearman).")
    lines.append("- Representative event-ranking diagnostic: offline Recall@5% at tau=16 mg/L, fold-wise top-k.")
    lines.append("")
    lines.append("## Interpretation Rules")
    lines.append("- This is sensitivity analysis, not model optimization.")
    lines.append("- Test performance will not be used to choose lag windows.")
    lines.append("- Compare robustness patterns; do not declare an optimal configuration.")
    lines.append("")
    lines.append("## Exclusions")
    lines.append("- No model fitting, no prediction generation, no sensitivity metric calculation in this stage.")
    lines.append("- No manuscript editing in Stage 2 execution gate.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("1. Offline sensitivity cannot substitute for prospective operational validation.")
    lines.append("2. BCR-TCN behavior is not directly validated by minimal Ridge/HGBR refits.")
    lines.append("3. Mixed sensitivity outcomes remain possible despite strict protocol controls.")
    lines.append("")
    lines.append("## Execution Stop Conditions")
    lines.append("- Abort if embargo/common-date/hyperparameter-freeze constraints are violated.")
    lines.append("- Abort on any source-preservation or checksum integrity failure.")
    lines.append("")
    lines.append("## Future Decision Options")
    lines.append("- Robust: proceed with confidence statement while preserving non-optimization framing.")
    lines.append("- Mixed/Sensitive: report conditional robustness with explicit caution and no post-hoc changes.")
    lines.append("")
    return "\n".join(lines)


def build_completion_report(
    preflight_pass: bool,
    lineage_pass: bool,
    common_date_pass: bool,
    source_preservation_pass: bool,
    checksum_info: Dict[str, Any],
    test_result: str,
    deterministic_result: str,
    final_decision: str,
    branch: str,
    head: str,
    tag_commit: str,
    npz_status: Dict[str, Any],
    feature_lineages: Dict[str, str],
    config_defs: Dict[str, List[str]],
    mapping: Dict[str, str],
    retention_notes: str,
    generated_files: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# Stage 2 Design-Gate Completion Report")
    lines.append("")
    lines.append("## Gate Summary")
    lines.append("- preflight_pass: {}".format(bool(preflight_pass)))
    lines.append("- feature_lineage_established_pass: {}".format(bool(lineage_pass)))
    lines.append("- common_date_feasibility_pass: {}".format(bool(common_date_pass)))
    lines.append("- source_preservation_pass: {}".format(bool(source_preservation_pass)))
    lines.append("- checksum_registry_pass: {}".format(bool(checksum_info.get("registry_pass", False))))
    lines.append("- checksum_coverage_pass: {}".format(bool(checksum_info.get("coverage_pass", False))))
    lines.append("- test_result: {}".format(test_result))
    lines.append("- deterministic_result: {}".format(deterministic_result))
    lines.append("")
    lines.append("## Context")
    lines.append("- repository: {}".format(ROOT))
    lines.append("- branch: {}".format(branch))
    lines.append("- starting commit: {}".format(head))
    lines.append("- Stage 1 tag {} resolves to {}".format(EXPECTED_STAGE1_TAG, tag_commit))
    lines.append("")
    lines.append("## Feature Lineage by Model/Horizon")
    for k in sorted(feature_lineages.keys()):
        lines.append("- {}: {}".format(k, feature_lineages[k]))
    lines.append("")
    lines.append("## Predeclared TN-Memory Definitions")
    for cid in ["SHORT", "REFERENCE", "LONG"]:
        lines.append("- {}: {}".format(cid, config_defs[cid]))
    lines.append("")
    lines.append("## Final-Reference Mapping")
    for k in sorted(mapping.keys()):
        lines.append("- {} -> {}".format(k, mapping[k]))
    lines.append("")
    lines.append("## NPZ Inspection Status")
    lines.append("- all_required_npz_present: {}".format(bool(npz_status["all_present"])))
    lines.append("- schema_ok: {}".format(bool(npz_status["schema_ok"])))
    lines.append("- hashes_recorded: {}".format(bool(npz_status["hashes_recorded"])))
    lines.append("")
    lines.append("## Training Retention Findings")
    lines.append("- {}".format(retention_notes))
    lines.append("")
    lines.append("FINAL DECISION")
    lines.append("")
    if final_decision == "A":
        lines.append("A. Feature lineage and predeclared lag-window sensitivity design passed; controlled sensitivity execution may be prepared.")
    elif final_decision == "B":
        lines.append("B. Design completed, but feature-lineage or common-date questions require investigation.")
    elif final_decision == "C":
        lines.append("C. Final model feature lineage could not be established safely.")
    else:
        lines.append("D. Source preservation, determinism, or design validation failed.")

    lines.append("")
    lines.append("Generated files:")
    for p in generated_files:
        lines.append("- {}".format(p))

    return "\n".join(lines) + "\n"


def build_stage() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("Run this script from repository root: {}".format(ROOT))

    # --- Preflight gates ---
    status_paths = worktree_paths()
    clean_now = len(status_paths) == 0
    authorized_only_now = bool(
        len(status_paths) > 0
        and all(
            p == AUTHORIZED_REL_DIR
            or p.startswith(AUTHORIZED_REL_DIR + "/")
            for p in status_paths
        )
    )

    clean_or_authorized_scope_pass = bool(clean_now or authorized_only_now)
    clean_initial_pass = bool(clean_or_authorized_scope_pass)

    branch = git_output(["git", "branch", "--show-current"])
    head = git_output(["git", "rev-parse", "HEAD"])
    tag_object = git_output(["git", "rev-parse", EXPECTED_STAGE1_TAG])
    tag_commit = git_output(["git", "rev-parse", EXPECTED_STAGE1_TAG + "^{}"])
    tag_type = git_output(["git", "cat-file", "-t", EXPECTED_STAGE1_TAG])

    branch_pass = bool(branch == EXPECTED_BRANCH)
    head_pass = bool(head == EXPECTED_HEAD)
    tag_pass = bool(tag_commit == head)

    py_version = platform.python_version()
    python_pass = bool(py_version == EXPECTED_PYTHON_VERSION)

    protocol_cmd = run_shell("sha256sum -c corrected_protocol_v1.sha256", cwd=PROTOCOL_DIR)
    protocol_expected = parse_checksum_manifest(PROTOCOL_SHA_PATH)
    protocol_checks: Dict[str, Dict[str, Any]] = {}
    for rel_path, expected_sha in protocol_expected.items():
        fp = PROTOCOL_DIR / rel_path
        exists = fp.exists()
        observed = sha256_file(fp) if exists else None
        ok = bool(exists and observed == expected_sha)
        protocol_checks[rel_path] = {
            "exists": bool(exists),
            "expected": expected_sha,
            "observed": observed,
            "pass": bool(ok),
        }

    protocol_pass = bool(protocol_cmd["exit_code"] == 0 and all(v["pass"] for v in protocol_checks.values()))

    stage1_cmd = run_shell("sha256sum -c canonical_h1_h3_reconciliation_checksums.sha256", cwd=STAGE1_DIR)
    stage1_pass = bool(stage1_cmd["exit_code"] == 0)

    local_tag_commits: Dict[str, str] = {}
    source_preservation_rows: Dict[str, Dict[str, Any]] = {}
    source_preservation_pass = True

    for tag, path in SOURCE_TAG_PATHS.items():
        tag_commit_local = git_output(["git", "rev-parse", tag + "^{}"])
        local_tag_commits[tag] = tag_commit_local
        changed = git_output(["git", "diff", "--name-only", tag + "^{}", "--", rel_to_root(path)])
        changed_paths = [x for x in changed.splitlines() if x.strip()]
        ok = len(changed_paths) == 0
        source_preservation_pass = source_preservation_pass and ok
        source_preservation_rows[tag] = {
            "source_dir": rel_to_root(path),
            "tag_commit": tag_commit_local,
            "changed_paths": changed_paths,
            "pass": bool(ok),
        }

    lock_payload = json.loads(DATASET_LOCK_PATH.read_text(encoding="utf-8"))
    dataset_path = Path(lock_payload["absolute_path"])
    dataset_sha_observed = sha256_file(dataset_path)
    dataset_sha_expected = str(lock_payload["sha256"])
    dataset_pass = bool(dataset_sha_observed == EXPECTED_DATASET_SHA256 and dataset_sha_observed == dataset_sha_expected)

    split_sha_observed = sha256_file(SPLIT_PATH)
    split_sha_expected = str(protocol_expected["corrected_split_assignment.csv"])
    split_pass = bool(split_sha_observed == EXPECTED_SPLIT_SHA256 and split_sha_observed == split_sha_expected)

    consumed_files = get_consumed_files()
    consumed_hashes: Dict[str, str] = {}
    for p in consumed_files:
        if not p.exists():
            raise RuntimeError("Consumed source file is missing: {}".format(p))
        consumed_hashes[rel_to_root(p)] = sha256_file(p)

    preflight_pass = bool(
        branch_pass
        and head_pass
        and tag_pass
        and python_pass
        and clean_initial_pass
        and clean_or_authorized_scope_pass
        and protocol_pass
        and stage1_pass
        and source_preservation_pass
        and dataset_pass
        and split_pass
    )

    input_verification = {
        "repository": str(ROOT),
        "authorized_output_directory": AUTHORIZED_REL_DIR,
        "required_branch": EXPECTED_BRANCH,
        "required_head": EXPECTED_HEAD,
        "required_stage1_tag": EXPECTED_STAGE1_TAG,
        "required_python_version": EXPECTED_PYTHON_VERSION,
        "verification": {
            "working_tree_status_paths": status_paths,
            "branch": branch,
            "head": head,
            "tag_object": tag_object,
            "tag_type": tag_type,
            "tag_commit": tag_commit,
            "python_version": py_version,
            "protocol_checksum_command": protocol_cmd,
            "protocol_checksum_checks": protocol_checks,
            "stage1_checksum_command": stage1_cmd,
            "local_tag_commits": local_tag_commits,
            "source_preservation": source_preservation_rows,
            "dataset_path": str(dataset_path),
            "dataset_sha_observed": dataset_sha_observed,
            "dataset_sha_expected_from_lock": dataset_sha_expected,
            "split_sha_observed": split_sha_observed,
            "split_sha_expected_from_manifest": split_sha_expected,
            "consumed_input_hashes": consumed_hashes,
            "checks": {
                "branch_pass": bool(branch_pass),
                "head_pass": bool(head_pass),
                "stage1_tag_points_to_head_pass": bool(tag_pass),
                "python_3810_pass": bool(python_pass),
                "clean_initial_working_tree_pass": bool(clean_initial_pass),
                "clean_or_authorized_scope_pass": bool(clean_or_authorized_scope_pass),
                "corrected_protocol_checksum_pass": bool(protocol_pass),
                "stage1_checksum_pass": bool(stage1_pass),
                "source_preservation_pass": bool(source_preservation_pass),
                "dataset_checksum_pass": bool(dataset_pass),
                "split_checksum_pass": bool(split_pass),
                "preflight_pass": bool(preflight_pass),
            },
        },
    }

    write_json(INPUT_VERIFICATION_PATH, input_verification)

    if not preflight_pass:
        test_result = get_existing_test_result()
        deterministic_result = "pass" if str(test_result).startswith("pass") else "pending_not_run"

        manifest = {
            "assembly_id": "preflight_failed",
            "branch": branch,
            "head": head,
            "stage1_tag": EXPECTED_STAGE1_TAG,
            "stage1_tag_commit": tag_commit,
            "python_version": py_version,
            "verification_pass": False,
            "source_preservation_pass": bool(source_preservation_pass),
            "test_result": test_result,
            "deterministic_result": deterministic_result,
            "final_decision": "D",
            "notes": "Preflight gate failed; design generation intentionally stopped.",
        }
        write_json(MANIFEST_PATH, manifest)

        report = "# Stage 2 Design-Gate Completion Report\n\nFINAL DECISION\n\nD. Source preservation, determinism, or design validation failed.\n"
        write_text(REPORT_PATH, report)

        build_checksums_registry()
        print("1. repository: {}".format(ROOT))
        print("2. branch: {}".format(branch))
        print("3. starting commit: {}".format(head))
        print("4. Stage 1 tag identity: {} -> {}".format(EXPECTED_STAGE1_TAG, tag_commit))
        print("5. authorized output directory: {}".format(OUT_DIR))
        print("6. input checksum status: protocol={} stage1={}".format(protocol_pass, stage1_pass))
        print("7. final decision: D")
        return

    # --- Load authoritative sources ---
    split = pd.read_csv(SPLIT_PATH)
    split["horizon"] = pd.to_numeric(split["horizon"], errors="raise").astype(int)
    split["outer_fold"] = pd.to_numeric(split["outer_fold"], errors="raise").astype(int)
    split["feature_date"] = normalize_date(split["feature_date"])
    split["target_date"] = normalize_date(split["target_date"])
    split["purged_outer_boundary"] = split["purged_outer_boundary"].astype(str).str.lower().map({"true": True, "false": False})
    split["purged_inner_boundary"] = split["purged_inner_boundary"].astype(str).str.lower().map({"true": True, "false": False})

    split_summary = pd.read_csv(SPLIT_SUMMARY_PATH)
    split_summary["horizon"] = pd.to_numeric(split_summary["horizon"], errors="raise").astype(int)
    split_summary["outer_fold"] = pd.to_numeric(split_summary["outer_fold"], errors="raise").astype(int)

    dataset_df = pd.read_csv(dataset_path)
    dataset_df = dataset_df.drop(columns=[c for c in dataset_df.columns if str(c).startswith("Unnamed")], errors="ignore")
    dataset_df["Date"] = pd.to_datetime(dataset_df["Date"], errors="raise")
    dataset_df = dataset_df.sort_values("Date").reset_index(drop=True)

    raw_start = pd.to_datetime(lock_payload["minimum_date"], errors="raise")

    ridge_manifest = json.loads(RIDGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    elasticnet_manifest = json.loads(ELASTICNET_MANIFEST_PATH.read_text(encoding="utf-8"))
    hgbr_manifest = json.loads(HGBR_MANIFEST_PATH.read_text(encoding="utf-8"))
    persistence_manifest = json.loads(PERSISTENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    bcr_manifest = json.loads(BCR_MANIFEST_PATH.read_text(encoding="utf-8"))

    # --- NPZ inspection ---
    npz_info: Dict[str, Dict[str, Any]] = {}
    npz_registry_rows: List[Dict[str, Any]] = []
    for key, path in NPZ_PATHS.items():
        info = inspect_npz(path)
        npz_info[key] = info
        npz_registry_rows.append(
            {
                "artifact_id": key,
                "relative_path": info["relative_path"],
                "sha256": info["sha256"],
                "keys": json.dumps(info["keys"], sort_keys=True),
                "X_shape": str(info["shapes"].get("X")),
                "y_shape": str(info["shapes"].get("y")),
                "dates_shape": str(info["shapes"].get("dates")),
                "feature_names_shape": str(info["shapes"].get("feature_names")),
                "feature_count": int(info["feature_count"]),
                "date_start": info["date_start"],
                "date_end": info["date_end"],
                "row_count": int(info["row_count"]),
            }
        )

    npz_registry_df = pd.DataFrame(npz_registry_rows).sort_values("artifact_id").reset_index(drop=True)

    # --- Feature frames and lineage extraction ---
    ordinary_frames: Dict[int, pd.DataFrame] = {}
    ordinary_features_by_h: Dict[int, List[str]] = {
        int(h): [str(c) for c in ridge_manifest["feature_sets"][str(h)]]
        for h in HORIZONS
    }
    for h in HORIZONS:
        ordinary_frames[int(h)] = build_ordinary_feature_frame(dataset_df, int(h), ordinary_features_by_h[int(h)])

    h5_v2_features = [str(c) for c in hgbr_manifest["feature_sets"]["5"]["feature_names"]]
    v2_frame_h5 = build_v2_feature_frame(dataset_df, 5, h5_v2_features)

    lineage_rows: List[Dict[str, Any]] = []

    def add_lineage_row(
        model: str,
        horizon: int,
        final_pathway: str,
        feature_set_id: str,
        feature_cols: List[str],
        builder_source: str,
        artifact_source: str,
        artifact_sha256: str,
        model_ready_start: str,
        model_ready_end: str,
        warmup_days: int,
        used: bool,
        notes: str,
    ) -> None:
        grp = parse_feature_groups(feature_cols)
        lineage_rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "final_pathway": final_pathway,
                "feature_set_id": feature_set_id,
                "feature_count": int(len(feature_cols)),
                "tnout_lags": sorted_join(grp["tnout_lags"]),
                "tnout_rolls": sorted_join(grp["tnout_rolls"]),
                "tnout_roll_construction": "TNout_rollw(t)=mean(TNout(t-w),...,TNout(t-1))",
                "tnout_roll_includes_day_t": False,
                "non_tnout_rolls": sorted_join(grp["non_tnout_rolls"]),
                "non_tnout_rolls_include_day_t": True,
                "builder_source": builder_source,
                "artifact_source": artifact_source,
                "artifact_sha256": artifact_sha256,
                "model_ready_start": model_ready_start,
                "model_ready_end": model_ready_end,
                "warmup_days": int(warmup_days),
                "used_in_final_corrected_model": bool(used),
                "lineage_notes": notes,
            }
        )

    # Persistence (context only, no tunable lag-window feature set)
    for h in HORIZONS:
        sh = split[(split["horizon"] == int(h)) & (~split["purged_outer_boundary"].astype(bool))]
        add_lineage_row(
            model="Persistence",
            horizon=int(h),
            final_pathway="direct_naive_context_only",
            feature_set_id="none_naive_lag_h",
            feature_cols=[],
            builder_source=rel_to_root(PERSISTENCE_DIR / "run_corrected_persistence.py"),
            artifact_source=rel_to_root(PERSISTENCE_DIR / "persistence_predictions.csv"),
            artifact_sha256=sha256_file(PERSISTENCE_DIR / "persistence_predictions.csv"),
            model_ready_start=str(sh["feature_date"].min().strftime("%Y-%m-%d")),
            model_ready_end=str(sh["feature_date"].max().strftime("%Y-%m-%d")),
            warmup_days=0,
            used=True,
            notes="Persistence has no adjustable TNout-memory feature set in Stage 2.",
        )

    # Ridge and ElasticNet (ordinary submitted non-v2 features)
    for model, manifest, runner in [
        ("Ridge", ridge_manifest, RIDGE_RUNNER_PATH),
        ("ElasticNet", elasticnet_manifest, ELASTICNET_RUNNER_PATH),
    ]:
        for h in HORIZONS:
            feat = [str(c) for c in manifest["feature_sets"][str(h)]]
            frame = ordinary_frames[int(h)]
            add_lineage_row(
                model=model,
                horizon=int(h),
                final_pathway="ordinary_submitted_non_v2",
                feature_set_id="submitted_main_linear_features_H{}_{}col".format(int(h), len(feat)),
                feature_cols=feat,
                builder_source="{}; {}".format(rel_to_root(runner), rel_to_root(ROOT / "src" / "build_ulsan_npz.py")),
                artifact_source="derived_in_run_from_canonical_dataset",
                artifact_sha256="",
                model_ready_start=str(frame["feature_date"].min().strftime("%Y-%m-%d")),
                model_ready_end=str(frame["feature_date"].max().strftime("%Y-%m-%d")),
                warmup_days=14,
                used=True,
                notes="Final corrected ordinary lineage with TNout_lag1/TNout_roll7/TNout_roll14.",
            )

    # HGBR H1/H3 ordinary NPZ
    for h, npz_key in [(1, "H1_ordinary"), (3, "H3_ordinary")]:
        feat = [str(c) for c in hgbr_manifest["feature_sets"][str(h)]["feature_names"]]
        info = npz_info[npz_key]
        add_lineage_row(
            model="HGBR",
            horizon=int(h),
            final_pathway="ordinary_npz",
            feature_set_id=str(hgbr_manifest["feature_sets"][str(h)]["feature_set_id"]),
            feature_cols=feat,
            builder_source="{}; {}".format(rel_to_root(HGBR_RUNNER_PATH), rel_to_root(ROOT / "src" / "build_ulsan_npz.py")),
            artifact_source=info["relative_path"],
            artifact_sha256=info["sha256"],
            model_ready_start=info["date_start"],
            model_ready_end=info["date_end"],
            warmup_days=14,
            used=True,
            notes="Final corrected HGBR H{} uses ordinary non-v2 NPZ lineage.".format(h),
        )

    # HGBR H5 v2 NPZ
    info_h5_v2 = npz_info["H5_v2"]
    add_lineage_row(
        model="HGBR",
        horizon=5,
        final_pathway="v2_npz",
        feature_set_id=str(hgbr_manifest["feature_sets"]["5"]["feature_set_id"]),
        feature_cols=[str(c) for c in hgbr_manifest["feature_sets"]["5"]["feature_names"]],
        builder_source="{}; {}".format(rel_to_root(HGBR_RUNNER_PATH), rel_to_root(ROOT / "src" / "build_ulsan_npz_v2.py")),
        artifact_source=info_h5_v2["relative_path"],
        artifact_sha256=info_h5_v2["sha256"],
        model_ready_start=info_h5_v2["date_start"],
        model_ready_end=info_h5_v2["date_end"],
        warmup_days=30,
        used=True,
        notes="Final corrected HGBR H5 uses v2 lineage; LONG is the corrected TN-memory reference.",
    )

    # BCR-TCN H5 context lineage (not selected for refit)
    add_lineage_row(
        model="BCR-TCN",
        horizon=5,
        final_pathway="v2_npz_context_only",
        feature_set_id=str(bcr_manifest["feature_set"]["feature_set_id"]),
        feature_cols=[str(c) for c in bcr_manifest["feature_set"]["feature_names"]],
        builder_source="{}; {}".format(rel_to_root(BCR_RUNNER_PATH), rel_to_root(ROOT / "src" / "build_ulsan_npz_v2.py")),
        artifact_source=rel_to_root(Path(bcr_manifest["feature_set"]["feature_npz_path"])),
        artifact_sha256=str(bcr_manifest["feature_set"]["feature_npz_sha256"]),
        model_ready_start=str(bcr_manifest["feature_set"]["feature_date_min"]),
        model_ready_end=str(bcr_manifest["feature_set"]["feature_date_max"]),
        warmup_days=30,
        used=True,
        notes="BCR-TCN lineage documented for context only; minimal Stage 2 design does not refit BCR-TCN.",
    )

    # Explicit H3-v2 artifact existence-not-used row.
    info_h3_v2 = npz_info["H3_v2"]
    add_lineage_row(
        model="HGBR",
        horizon=3,
        final_pathway="v2_artifact_exists_not_used",
        feature_set_id="ulsan_H3_features_v2_npz_34col",
        feature_cols=info_h3_v2["feature_names"],
        builder_source=rel_to_root(ROOT / "src" / "build_ulsan_npz_v2.py"),
        artifact_source=info_h3_v2["relative_path"],
        artifact_sha256=info_h3_v2["sha256"],
        model_ready_start=info_h3_v2["date_start"],
        model_ready_end=info_h3_v2["date_end"],
        warmup_days=30,
        used=False,
        notes="Artifact exists but final corrected H3 runners/manifests reference ordinary H3 feature lineage.",
    )

    lineage_df = pd.DataFrame(lineage_rows)
    lineage_df = lineage_df.sort_values(["model", "horizon", "final_pathway"]).reset_index(drop=True)

    # --- Feature artifact registry ---
    artifact_rows: List[Dict[str, Any]] = []

    def add_artifact_row(role: str, model: str, horizon: str, path: Path, category: str, notes: str) -> None:
        artifact_rows.append(
            {
                "artifact_role": role,
                "model": model,
                "horizon": horizon,
                "relative_path": rel_to_root(path),
                "sha256": sha256_file(path),
                "exists": True,
                "consumed_for_stage2": True,
                "source_category": category,
                "notes": notes,
            }
        )

    for p in SRC_BUILDERS:
        add_artifact_row("feature_builder", "ALL", "ALL", p, "source_builder", "Source NPZ builder inspected for feature construction rules.")

    for p in [RIDGE_RUNNER_PATH, ELASTICNET_RUNNER_PATH, HGBR_RUNNER_PATH, BCR_RUNNER_PATH]:
        add_artifact_row("model_runner", "ALL", "ALL", p, "controlled_rerun_runner", "Final corrected runner inspected for lineage usage.")

    for p in [
        RIDGE_MANIFEST_PATH,
        ELASTICNET_MANIFEST_PATH,
        HGBR_MANIFEST_PATH,
        PERSISTENCE_MANIFEST_PATH,
        BCR_MANIFEST_PATH,
        STAGE1_MANIFEST_PATH,
    ]:
        add_artifact_row("manifest", "ALL", "ALL", p, "manifest", "Frozen manifest consumed for lineage/provenance.")

    for p in [RIDGE_PLAN_PATH, ELASTICNET_PLAN_PATH, HGBR_PLAN_PATH, BCR_FIXED_CONFIG_PATH]:
        add_artifact_row("selection_plan", "ALL", "ALL", p, "plan", "Frozen selection/fixed-configuration artifact consumed.")

    for key, path in NPZ_PATHS.items():
        add_artifact_row("feature_npz", "ALL", "ALL", path, "npz", "NPZ schema/date/hash inspected ({})".format(key))

    for p in [DATASET_LOCK_PATH, PROTOCOL_SHA_PATH, SPLIT_PATH, SPLIT_SUMMARY_PATH, STAGE1_CHECKSUM_PATH]:
        add_artifact_row("protocol_input", "ALL", "ALL", p, "protocol", "Preflight checksum or split input consumed.")

    artifact_df = pd.DataFrame(artifact_rows).sort_values(["artifact_role", "relative_path"]).reset_index(drop=True)

    # --- Candidate memory configurations ---
    config_rows: List[Dict[str, Any]] = []
    for cfg in CONFIGS:
        earliest = raw_start + pd.to_timedelta(int(cfg["maximum_required_history_days"]), unit="D")
        config_rows.append(
            {
                "configuration_id": cfg["configuration_id"],
                "configuration_label": cfg["configuration_label"],
                "tnout_features": sorted_join(cfg["tnout_features"]),
                "maximum_required_history_days": int(cfg["maximum_required_history_days"]),
                "earliest_possible_date_from_raw_start": str(earliest.strftime("%Y-%m-%d")),
                "scientific_role": cfg["scientific_role"],
                "predeclared": bool(cfg["predeclared"]),
                "uses_future_information": bool(cfg["uses_future_information"]),
                "notes": cfg["notes"],
            }
        )
    config_df = pd.DataFrame(config_rows)

    # --- Common-date feasibility and retention ---
    canonical_outer_test = split[(split["outer_role"].astype(str) == "outer_test") & (~split["purged_outer_boundary"].astype(bool))].copy()
    canonical_outer_train = split[(split["outer_role"].astype(str) == "outer_train") & (~split["purged_outer_boundary"].astype(bool))].copy()

    common_rows: List[Dict[str, Any]] = []
    retention_rows: List[Dict[str, Any]] = []

    for cfg in CONFIGS:
        cfg_id = str(cfg["configuration_id"])
        cfg_label = str(cfg["configuration_label"])
        history_days = int(cfg["maximum_required_history_days"])
        threshold = raw_start + pd.to_timedelta(history_days, unit="D")

        for h in HORIZONS:
            for fold in FOLDS:
                test_rows = canonical_outer_test[
                    (canonical_outer_test["horizon"] == int(h))
                    & (canonical_outer_test["outer_fold"] == int(fold))
                ].copy()
                test_avail = test_rows[test_rows["feature_date"] >= threshold].copy()

                common_rows.append(
                    {
                        "configuration_id": cfg_id,
                        "configuration_label": cfg_label,
                        "maximum_required_history_days": history_days,
                        "earliest_possible_feature_date": str(threshold.strftime("%Y-%m-%d")),
                        "horizon": int(h),
                        "outer_fold": int(fold),
                        "canonical_outer_test_rows": int(len(test_rows)),
                        "available_outer_test_rows": int(len(test_avail)),
                        "rows_lost_due_warmup": int(len(test_rows) - len(test_avail)),
                        "canonical_feature_date_start": str(test_rows["feature_date"].min().strftime("%Y-%m-%d")),
                        "canonical_feature_date_end": str(test_rows["feature_date"].max().strftime("%Y-%m-%d")),
                        "available_feature_date_start": str(test_avail["feature_date"].min().strftime("%Y-%m-%d")) if len(test_avail) > 0 else "",
                        "available_feature_date_end": str(test_avail["feature_date"].max().strftime("%Y-%m-%d")) if len(test_avail) > 0 else "",
                        "canonical_target_date_start": str(test_rows["target_date"].min().strftime("%Y-%m-%d")),
                        "canonical_target_date_end": str(test_rows["target_date"].max().strftime("%Y-%m-%d")),
                        "available_target_date_start": str(test_avail["target_date"].min().strftime("%Y-%m-%d")) if len(test_avail) > 0 else "",
                        "available_target_date_end": str(test_avail["target_date"].max().strftime("%Y-%m-%d")) if len(test_avail) > 0 else "",
                        "preserves_all_canonical_test_dates": bool(len(test_rows) == len(test_avail)),
                    }
                )

                outer_train_rows = canonical_outer_train[
                    (canonical_outer_train["horizon"] == int(h))
                    & (canonical_outer_train["outer_fold"] == int(fold))
                ].copy()
                outer_train_avail = outer_train_rows[outer_train_rows["feature_date"] >= threshold].copy()

                inner_train_rows = split[
                    (split["horizon"] == int(h))
                    & (split["outer_fold"] == int(fold))
                    & (split["outer_role"].astype(str) == "outer_train")
                    & (split["inner_role"].astype(str) == "inner_train")
                    & (~split["purged_outer_boundary"].astype(bool))
                    & (~split["purged_inner_boundary"].astype(bool))
                ].copy()
                inner_train_avail = inner_train_rows[inner_train_rows["feature_date"] >= threshold].copy()

                inner_val_rows = split[
                    (split["horizon"] == int(h))
                    & (split["outer_fold"] == int(fold))
                    & (split["outer_role"].astype(str) == "outer_train")
                    & (split["inner_role"].astype(str) == "inner_validation")
                    & (~split["purged_outer_boundary"].astype(bool))
                ].copy()
                inner_val_avail = inner_val_rows[inner_val_rows["feature_date"] >= threshold].copy()

                retention_rows.append(
                    {
                        "configuration_id": cfg_id,
                        "configuration_label": cfg_label,
                        "horizon": int(h),
                        "outer_fold": int(fold),
                        "canonical_outer_train_rows": int(len(outer_train_rows)),
                        "available_outer_train_rows": int(len(outer_train_avail)),
                        "outer_train_rows_lost_warmup": int(len(outer_train_rows) - len(outer_train_avail)),
                        "canonical_inner_train_rows": int(len(inner_train_rows)),
                        "available_inner_train_rows": int(len(inner_train_avail)),
                        "inner_train_rows_lost_warmup": int(len(inner_train_rows) - len(inner_train_avail)),
                        "canonical_inner_validation_rows": int(len(inner_val_rows)),
                        "available_inner_validation_rows": int(len(inner_val_avail)),
                        "inner_validation_rows_lost_warmup": int(len(inner_val_rows) - len(inner_val_avail)),
                        "canonical_outer_test_rows": int(len(test_rows)),
                        "available_outer_test_rows": int(len(test_avail)),
                        "outer_test_rows_lost_warmup": int(len(test_rows) - len(test_avail)),
                        "earliest_available_feature_date": str(threshold.strftime("%Y-%m-%d")),
                        "latest_available_feature_date": str(test_avail["feature_date"].max().strftime("%Y-%m-%d")) if len(test_avail) > 0 else "",
                    }
                )

            # pooled row for common-date
            tpool = canonical_outer_test[canonical_outer_test["horizon"] == int(h)].copy()
            tpool_avail = tpool[tpool["feature_date"] >= threshold].copy()
            common_rows.append(
                {
                    "configuration_id": cfg_id,
                    "configuration_label": cfg_label,
                    "maximum_required_history_days": history_days,
                    "earliest_possible_feature_date": str(threshold.strftime("%Y-%m-%d")),
                    "horizon": int(h),
                    "outer_fold": "pooled",
                    "canonical_outer_test_rows": int(len(tpool)),
                    "available_outer_test_rows": int(len(tpool_avail)),
                    "rows_lost_due_warmup": int(len(tpool) - len(tpool_avail)),
                    "canonical_feature_date_start": str(tpool["feature_date"].min().strftime("%Y-%m-%d")),
                    "canonical_feature_date_end": str(tpool["feature_date"].max().strftime("%Y-%m-%d")),
                    "available_feature_date_start": str(tpool_avail["feature_date"].min().strftime("%Y-%m-%d")) if len(tpool_avail) > 0 else "",
                    "available_feature_date_end": str(tpool_avail["feature_date"].max().strftime("%Y-%m-%d")) if len(tpool_avail) > 0 else "",
                    "canonical_target_date_start": str(tpool["target_date"].min().strftime("%Y-%m-%d")),
                    "canonical_target_date_end": str(tpool["target_date"].max().strftime("%Y-%m-%d")),
                    "available_target_date_start": str(tpool_avail["target_date"].min().strftime("%Y-%m-%d")) if len(tpool_avail) > 0 else "",
                    "available_target_date_end": str(tpool_avail["target_date"].max().strftime("%Y-%m-%d")) if len(tpool_avail) > 0 else "",
                    "preserves_all_canonical_test_dates": bool(len(tpool) == len(tpool_avail)),
                }
            )

            # pooled row for retention
            otr_pool = canonical_outer_train[canonical_outer_train["horizon"] == int(h)].copy()
            otr_pool_avail = otr_pool[otr_pool["feature_date"] >= threshold].copy()

            itr_pool = split[
                (split["horizon"] == int(h))
                & (split["outer_role"].astype(str) == "outer_train")
                & (split["inner_role"].astype(str) == "inner_train")
                & (~split["purged_outer_boundary"].astype(bool))
                & (~split["purged_inner_boundary"].astype(bool))
            ].copy()
            itr_pool_avail = itr_pool[itr_pool["feature_date"] >= threshold].copy()

            iv_pool = split[
                (split["horizon"] == int(h))
                & (split["outer_role"].astype(str) == "outer_train")
                & (split["inner_role"].astype(str) == "inner_validation")
                & (~split["purged_outer_boundary"].astype(bool))
            ].copy()
            iv_pool_avail = iv_pool[iv_pool["feature_date"] >= threshold].copy()

            retention_rows.append(
                {
                    "configuration_id": cfg_id,
                    "configuration_label": cfg_label,
                    "horizon": int(h),
                    "outer_fold": "pooled",
                    "canonical_outer_train_rows": int(len(otr_pool)),
                    "available_outer_train_rows": int(len(otr_pool_avail)),
                    "outer_train_rows_lost_warmup": int(len(otr_pool) - len(otr_pool_avail)),
                    "canonical_inner_train_rows": int(len(itr_pool)),
                    "available_inner_train_rows": int(len(itr_pool_avail)),
                    "inner_train_rows_lost_warmup": int(len(itr_pool) - len(itr_pool_avail)),
                    "canonical_inner_validation_rows": int(len(iv_pool)),
                    "available_inner_validation_rows": int(len(iv_pool_avail)),
                    "inner_validation_rows_lost_warmup": int(len(iv_pool) - len(iv_pool_avail)),
                    "canonical_outer_test_rows": int(len(tpool)),
                    "available_outer_test_rows": int(len(tpool_avail)),
                    "outer_test_rows_lost_warmup": int(len(tpool) - len(tpool_avail)),
                    "earliest_available_feature_date": str(threshold.strftime("%Y-%m-%d")),
                    "latest_available_feature_date": str(tpool_avail["feature_date"].max().strftime("%Y-%m-%d")) if len(tpool_avail) > 0 else "",
                }
            )

    common_df = pd.DataFrame(common_rows)
    common_df = common_df.sort_values(["configuration_id", "horizon", "outer_fold"], key=lambda s: s.astype(str)).reset_index(drop=True)

    retention_df = pd.DataFrame(retention_rows)
    retention_df = retention_df.sort_values(["configuration_id", "horizon", "outer_fold"], key=lambda s: s.astype(str)).reset_index(drop=True)

    # Canonical count checks.
    canonical_count_pass = True
    for h in HORIZONS:
        for fold in FOLDS:
            n = int(
                len(
                    canonical_outer_test[
                        (canonical_outer_test["horizon"] == int(h))
                        & (canonical_outer_test["outer_fold"] == int(fold))
                    ]
                )
            )
            if n != int(EXPECTED_OUTER_TEST_COUNTS[int(h)][int(fold)]):
                canonical_count_pass = False
        pooled_n = int(len(canonical_outer_test[canonical_outer_test["horizon"] == int(h)]))
        if pooled_n != int(EXPECTED_OUTER_TEST_COUNTS[int(h)]["pooled"]):
            canonical_count_pass = False

    no_canonical_test_date_loss_pass = bool(common_df["preserves_all_canonical_test_dates"].astype(bool).all())
    common_date_pass = bool(canonical_count_pass and no_canonical_test_date_loss_pass)

    # Warm-up expectations.
    cfg_expectations = {
        "SHORT": "2021-01-11",
        "REFERENCE": "2021-01-18",
        "LONG": "2021-02-03",
    }
    warmup_pass = True
    for cfg in CONFIGS:
        cfg_id = cfg["configuration_id"]
        threshold = (raw_start + pd.to_timedelta(int(cfg["maximum_required_history_days"]), unit="D")).strftime("%Y-%m-%d")
        if threshold != cfg_expectations[cfg_id]:
            warmup_pass = False

    # Retention note (H1 LONG expected loss per fold).
    long_h1 = retention_df[
        (retention_df["configuration_id"] == "LONG")
        & (retention_df["horizon"] == 1)
        & (retention_df["outer_fold"].astype(str).isin(["1", "2", "3"]))
    ]
    long_h1_losses = {
        int(r.outer_fold): int(r.outer_train_rows_lost_warmup)
        for r in long_h1.itertuples(index=False)
    }
    long_h1_pooled_loss = int(
        retention_df[
            (retention_df["configuration_id"] == "LONG")
            & (retention_df["horizon"] == 1)
            & (retention_df["outer_fold"].astype(str) == "pooled")
        ]["outer_train_rows_lost_warmup"].iloc[0]
    )
    retention_note = "LONG H1 outer-train warm-up losses by fold={} (pooled={}); all canonical outer-test rows are preserved across SHORT/REFERENCE/LONG for H1/H3/H5.".format(
        long_h1_losses,
        long_h1_pooled_loss,
    )

    # --- Model-horizon reference registry ---
    reg_rows: List[Dict[str, Any]] = []

    def add_registry(
        model: str,
        horizon: int,
        sensitivity_role: str,
        will_be_refitted: bool,
        final_reference_configuration: str,
        non_tnout_feature_source: str,
        hyperparameter_source: str,
        retuning_allowed: bool,
        reason: str,
        limitations: str,
    ) -> None:
        reg_rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "sensitivity_role": sensitivity_role,
                "will_be_refitted": bool(will_be_refitted),
                "final_reference_configuration": final_reference_configuration,
                "non_tnout_feature_source": non_tnout_feature_source,
                "hyperparameter_source": hyperparameter_source,
                "retuning_allowed": bool(retuning_allowed),
                "reason_for_inclusion_or_exclusion": reason,
                "limitations": limitations,
            }
        )

    for h in HORIZONS:
        add_registry(
            model="Ridge",
            horizon=int(h),
            sensitivity_role="minimal_refit",
            will_be_refitted=True,
            final_reference_configuration="REFERENCE",
            non_tnout_feature_source="final corrected ordinary non-v2 feature set (18 columns)",
            hyperparameter_source=rel_to_root(RIDGE_MANIFEST_PATH) + " selected_alphas_by_fold",
            retuning_allowed=False,
            reason="Included as regularized linear sensitivity model with frozen corrected fold-specific alpha.",
            limitations="No retuning; conclusions are robustness-oriented, not optimization.",
        )

    add_registry(
        model="HGBR",
        horizon=1,
        sensitivity_role="minimal_refit",
        will_be_refitted=True,
        final_reference_configuration="REFERENCE",
        non_tnout_feature_source="final corrected ordinary non-v2 feature set (18 columns)",
        hyperparameter_source=rel_to_root(HGBR_MANIFEST_PATH) + " selected_configurations_by_fold[H1]",
        retuning_allowed=False,
        reason="Included as nonlinear tree model; H1 reference remains ordinary corrected TN memory.",
        limitations="No retuning; sensitivity does not optimize parameters.",
    )
    add_registry(
        model="HGBR",
        horizon=3,
        sensitivity_role="minimal_refit",
        will_be_refitted=True,
        final_reference_configuration="REFERENCE",
        non_tnout_feature_source="final corrected ordinary non-v2 feature set (18 columns)",
        hyperparameter_source=rel_to_root(HGBR_MANIFEST_PATH) + " selected_configurations_by_fold[H3]",
        retuning_allowed=False,
        reason="Included as nonlinear tree model; H3 reference remains ordinary corrected TN memory.",
        limitations="No retuning; sensitivity does not optimize parameters.",
    )
    add_registry(
        model="HGBR",
        horizon=5,
        sensitivity_role="minimal_refit",
        will_be_refitted=True,
        final_reference_configuration="LONG",
        non_tnout_feature_source="final corrected H5 v2 non-TN feature set (34-column v2 lineage)",
        hyperparameter_source=rel_to_root(HGBR_MANIFEST_PATH) + " selected_configurations_by_fold[H5]",
        retuning_allowed=False,
        reason="Included as nonlinear tree model; LONG is final corrected HGBR-H5 TN-memory reference.",
        limitations="No retuning; sensitivity does not optimize parameters.",
    )

    for h in HORIZONS:
        add_registry(
            model="Persistence",
            horizon=int(h),
            sensitivity_role="context_only_baseline",
            will_be_refitted=False,
            final_reference_configuration="none_not_applicable",
            non_tnout_feature_source="none_naive_baseline",
            hyperparameter_source=rel_to_root(PERSISTENCE_MANIFEST_PATH),
            retuning_allowed=False,
            reason="Context-only naive baseline; no tunable lag-window feature set.",
            limitations="Does not provide tunable memory configuration comparisons.",
        )

    for h in HORIZONS:
        add_registry(
            model="ElasticNet",
            horizon=int(h),
            sensitivity_role="frozen_context_only",
            will_be_refitted=False,
            final_reference_configuration="REFERENCE",
            non_tnout_feature_source="final corrected ordinary non-v2 feature set (18 columns)",
            hyperparameter_source=rel_to_root(ELASTICNET_MANIFEST_PATH) + " selected_configs_by_fold",
            retuning_allowed=False,
            reason="Excluded from minimal refit because Ridge already represents regularized linear behavior.",
            limitations="Context only; no direct sensitivity rerun in minimal design.",
        )

    add_registry(
        model="BCR-TCN",
        horizon=5,
        sensitivity_role="lineage_context_only",
        will_be_refitted=False,
        final_reference_configuration="LONG",
        non_tnout_feature_source="final corrected H5 v2 non-TN feature set (34-column v2 lineage)",
        hyperparameter_source=rel_to_root(BCR_MANIFEST_PATH) + " fixed_configuration",
        retuning_allowed=False,
        reason="Not selected for minimal Stage 2 refitting; retained only for lineage context.",
        limitations="Sensitivity conclusions from Ridge/HGBR do not directly validate BCR-TCN behavior.",
    )

    registry_df = pd.DataFrame(reg_rows).sort_values(["model", "horizon"]).reset_index(drop=True)

    # --- design json payload ---
    cfg_payload_rows: List[Dict[str, Any]] = []
    for r in config_df.itertuples(index=False):
        cfg_payload_rows.append(
            {
                "configuration_id": r.configuration_id,
                "configuration_label": r.configuration_label,
                "tnout_features": [x.strip() for x in str(r.tnout_features).split(";") if x.strip()],
                "maximum_required_history_days": int(r.maximum_required_history_days),
                "earliest_possible_date_from_raw_start": r.earliest_possible_date_from_raw_start,
                "scientific_role": r.scientific_role,
                "predeclared": bool(r.predeclared),
                "uses_future_information": bool(r.uses_future_information),
                "notes": r.notes,
            }
        )

    final_reference_mapping = {
        "Ridge_H1": "REFERENCE",
        "Ridge_H3": "REFERENCE",
        "Ridge_H5": "REFERENCE",
        "HGBR_H1": "REFERENCE",
        "HGBR_H3": "REFERENCE",
        "HGBR_H5": "LONG",
    }

    design_json = {
        "objective": "Design-only predeclaration of leakage-safe lag-window sensitivity configurations.",
        "reviewer_issue_addressed": "TNout-memory robustness without post-hoc optimization.",
        "authoritative_feature_lineage": {
            "ordinary": "TNout_lag1 + TNout_roll7 + TNout_roll14 (18-column corrected ordinary feature set)",
            "v2": "TNout_lag1/3/5/7 + TNout_roll7/14/30 (34-column corrected v2 feature set)",
            "h3_v2_exists_not_used": True,
        },
        "candidate_configurations": cfg_payload_rows,
        "model_horizon_selection": {
            "refit_models": ["Ridge", "HGBR"],
            "refit_horizons": [1, 3, 5],
            "context_only_models": ["Persistence", "ElasticNet", "BCR-TCN"],
            "final_reference_mapping": final_reference_mapping,
        },
        "non_tn_feature_freeze_rule": "Preserve final corrected non-TN predictors; vary only TNout-memory columns.",
        "hyperparameter_freeze_rule": "Use frozen corrected fold-specific hyperparameters; retuning is prohibited.",
        "common_date_requirement": "Canonical outer-test dates must be preserved across SHORT/REFERENCE/LONG.",
        "warmup_expectations": {
            "raw_start": str(raw_start.strftime("%Y-%m-%d")),
            "SHORT": "2021-01-11",
            "REFERENCE": "2021-01-18",
            "LONG": "2021-02-03",
        },
        "planned_outcomes": [
            "MAE", "MSE", "RMSE", "MASE",
            "Pearson correlation vs final reference predictions",
            "Spearman rank correlation vs final reference predictions",
            "offline Recall@5% tau=16 diagnostic",
        ],
        "event_ranking_diagnostic": {
            "type": "retrospective_offline_recall_at_5_percent",
            "tau_mgL": 16,
            "top_k_rule": "k=ceil(0.05*fold_N), fold-wise",
            "not_for": ["online operation", "prospective validation", "deployable alarm rule", "lag-window optimization"],
        },
        "interpretation_rules": {
            "sensitivity_not_optimization": True,
            "do_not_choose_best_configuration": True,
            "robust_mixed_sensitive_outcomes_possible": True,
            "retain_mixed_results_without_posthoc_changes": True,
        },
        "exclusions": {
            "no_model_fitting_in_design_stage": True,
            "no_prediction_generation_in_design_stage": True,
            "no_metric_execution_in_design_stage": True,
            "no_manuscript_editing_in_stage2": True,
        },
        "execution_stop_conditions": [
            "embargo violation",
            "canonical test-date mismatch across configurations",
            "future-information leakage",
            "hyperparameter-retuning attempt",
            "source-preservation or checksum failure",
        ],
        "future_decision_options": ["robust", "mixed", "sensitive"],
    }

    # --- write core artifacts ---
    write_csv(FEATURE_LINEAGE_CSV_PATH, lineage_df, date_cols=["model_ready_start", "model_ready_end"])
    write_csv(FEATURE_ARTIFACT_REGISTRY_PATH, artifact_df, date_cols=[])
    write_csv(CANDIDATE_CONFIG_PATH, config_df, date_cols=["earliest_possible_date_from_raw_start"])
    write_csv(MODEL_HORIZON_REGISTRY_PATH, registry_df, date_cols=[])
    write_csv(COMMON_DATE_FEASIBILITY_PATH, common_df, date_cols=[
        "earliest_possible_feature_date",
        "canonical_feature_date_start",
        "canonical_feature_date_end",
        "available_feature_date_start",
        "available_feature_date_end",
        "canonical_target_date_start",
        "canonical_target_date_end",
        "available_target_date_start",
        "available_target_date_end",
    ])
    write_csv(TRAINING_RETENTION_PATH, retention_df, date_cols=["earliest_available_feature_date", "latest_available_feature_date"])

    lineage_md = build_feature_lineage_markdown(lineage_df, npz_registry_df)
    write_text(FEATURE_LINEAGE_MD_PATH, lineage_md)

    write_json(DESIGN_JSON_PATH, design_json)
    design_md = build_design_markdown(design_json)
    write_text(DESIGN_MD_PATH, design_md)

    lineage_established_pass = True

    # Required lineage checks for gate meaning.
    # H3-v2 must exist but not be final H3 lineage.
    h3_v2_exists = NPZ_PATHS["H3_v2"].exists()
    hgbr_h3_path = str(hgbr_manifest["feature_sets"]["3"]["feature_npz_path"])
    h3_v2_not_used = bool("H3_features_v2" not in hgbr_h3_path)

    # H5-v2 must be final for HGBR and BCR.
    hgbr_h5_v2 = bool("H5_features_v2" in str(hgbr_manifest["feature_sets"]["5"]["feature_npz_path"]))
    bcr_h5_v2 = bool("H5_features_v2" in str(bcr_manifest["feature_set"]["feature_npz_path"]))

    # H1/H3 ordinary must be final for Ridge/HGBR.
    ridge_h13_reference = bool(
        all(len(ridge_manifest["feature_sets"][str(h)]) == 18 for h in [1, 3])
    )
    hgbr_h13_ordinary = bool(
        ("H1_features.npz" in str(hgbr_manifest["feature_sets"]["1"]["feature_npz_path"]))
        and ("H3_features.npz" in str(hgbr_manifest["feature_sets"]["3"]["feature_npz_path"]))
    )

    lineage_established_pass = bool(
        h3_v2_exists
        and h3_v2_not_used
        and hgbr_h5_v2
        and bcr_h5_v2
        and ridge_h13_reference
        and hgbr_h13_ordinary
    )

    npz_status = {
        "all_present": bool(all(p.exists() for p in NPZ_PATHS.values())),
        "schema_ok": bool(all(set(npz_info[k]["keys"]) == {"X", "y", "dates", "feature_names"} for k in npz_info.keys())),
        "hashes_recorded": bool(len(npz_info) == len(NPZ_PATHS)),
    }

    test_result = get_existing_test_result()
    deterministic_result = "pass" if str(test_result).startswith("pass") else "pending_not_run"

    checksum_info_placeholder = {
        "registry_pass": False,
        "coverage_pass": False,
        "coverage_missing": [],
        "coverage_extra": [],
    }

    checksum_pass_for_decision = False

    final_decision = compute_decision(
        preflight_pass=preflight_pass,
        lineage_established_pass=lineage_established_pass,
        common_date_pass=common_date_pass,
        source_preservation_pass=source_preservation_pass,
        checksum_pass=checksum_pass_for_decision,
        deterministic_result=deterministic_result,
        test_result=test_result,
    )

    # Build manifest first.
    assembly_id_payload = {
        "consumed_hashes": {k: consumed_hashes[k] for k in sorted(consumed_hashes.keys())},
        "configs": [
            {
                "id": c["configuration_id"],
                "tn": c["tnout_features"],
                "history": c["maximum_required_history_days"],
            }
            for c in CONFIGS
        ],
    }
    assembly_id_hash = hashlib.sha256(
        json.dumps(assembly_id_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    assembly_id = "lag_window_sensitivity_design_{}".format(assembly_id_hash)

    manifest = {
        "assembly_id": assembly_id,
        "repository": str(ROOT),
        "authorized_output_directory": AUTHORIZED_REL_DIR,
        "required_branch": EXPECTED_BRANCH,
        "required_head": EXPECTED_HEAD,
        "required_stage1_tag": EXPECTED_STAGE1_TAG,
        "branch": branch,
        "head": head,
        "stage1_tag_commit": tag_commit,
        "python_version": py_version,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha_observed,
        "split_sha256": split_sha_observed,
        "input_files": sorted(list(consumed_hashes.keys())),
        "input_hashes": consumed_hashes,
        "candidate_configurations": cfg_payload_rows,
        "model_horizon_reference_mapping": final_reference_mapping,
        "lineage_checks": {
            "h3_v2_exists": bool(h3_v2_exists),
            "h3_v2_not_used_in_final_h3": bool(h3_v2_not_used),
            "hgbr_h5_v2_final": bool(hgbr_h5_v2),
            "bcr_h5_v2_lineage": bool(bcr_h5_v2),
            "ridge_h1_h3_ordinary": bool(ridge_h13_reference),
            "hgbr_h1_h3_ordinary": bool(hgbr_h13_ordinary),
            "lineage_established_pass": bool(lineage_established_pass),
        },
        "feasibility_checks": {
            "canonical_count_pass": bool(canonical_count_pass),
            "no_canonical_test_date_loss_pass": bool(no_canonical_test_date_loss_pass),
            "common_date_pass": bool(common_date_pass),
            "warmup_expectation_pass": bool(warmup_pass),
            "long_h1_outer_train_losses_by_fold": long_h1_losses,
            "long_h1_outer_train_loss_pooled": long_h1_pooled_loss,
        },
        "npz_inspection_status": npz_status,
        "verification_pass": bool(preflight_pass),
        "source_preservation_pass": bool(source_preservation_pass),
        "test_result": str(test_result),
        "deterministic_result": str(deterministic_result),
        "final_decision": str(final_decision),
    }
    write_json(MANIFEST_PATH, manifest)

    feature_lineages_summary = {}
    for r in lineage_df.itertuples(index=False):
        key = "{}_H{}".format(r.model, int(r.horizon))
        val = "{} | feature_set_id={} | feature_count={} | used={}".format(
            r.final_pathway,
            r.feature_set_id,
            int(r.feature_count),
            bool(r.used_in_final_corrected_model),
        )
        feature_lineages_summary[key] = val

    config_defs = {
        row["configuration_id"]: row["tnout_features"] for row in config_rows
    }

    generated_files = sorted(
        [
            rel_to_root(p)
            for p in OUT_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
            and p.resolve() != CHECKSUM_PATH.resolve()
        ]
    )

    report_text = build_completion_report(
        preflight_pass=preflight_pass,
        lineage_pass=lineage_established_pass,
        common_date_pass=common_date_pass,
        source_preservation_pass=source_preservation_pass,
        checksum_info=checksum_info_placeholder,
        test_result=test_result,
        deterministic_result=deterministic_result,
        final_decision=final_decision,
        branch=branch,
        head=head,
        tag_commit=tag_commit,
        npz_status=npz_status,
        feature_lineages=feature_lineages_summary,
        config_defs=config_defs,
        mapping=final_reference_mapping,
        retention_notes=retention_note,
        generated_files=generated_files,
    )
    write_text(REPORT_PATH, report_text)

    build_checksums_registry()
    checksum_info = verify_checksums_registry()

    checksum_pass_for_decision = bool(checksum_info["registry_pass"] and checksum_info["coverage_pass"])
    final_decision = compute_decision(
        preflight_pass=preflight_pass,
        lineage_established_pass=lineage_established_pass,
        common_date_pass=common_date_pass,
        source_preservation_pass=source_preservation_pass,
        checksum_pass=checksum_pass_for_decision,
        deterministic_result=deterministic_result,
        test_result=test_result,
    )

    manifest["checksum_result"] = checksum_info
    manifest["final_decision"] = final_decision
    write_json(MANIFEST_PATH, manifest)

    generated_files = sorted(
        [
            rel_to_root(p)
            for p in OUT_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
            and p.resolve() != CHECKSUM_PATH.resolve()
        ]
    )

    report_text = build_completion_report(
        preflight_pass=preflight_pass,
        lineage_pass=lineage_established_pass,
        common_date_pass=common_date_pass,
        source_preservation_pass=source_preservation_pass,
        checksum_info=checksum_info,
        test_result=test_result,
        deterministic_result=deterministic_result,
        final_decision=final_decision,
        branch=branch,
        head=head,
        tag_commit=tag_commit,
        npz_status=npz_status,
        feature_lineages=feature_lineages_summary,
        config_defs=config_defs,
        mapping=final_reference_mapping,
        retention_notes=retention_note,
        generated_files=generated_files,
    )
    write_text(REPORT_PATH, report_text)

    build_checksums_registry()
    checksum_info = verify_checksums_registry()

    # Terminal summary required by instruction.
    print("1. repository: {}".format(ROOT))
    print("2. branch: {}".format(branch))
    print("3. starting commit: {}".format(head))
    print("4. Stage 1 tag identity: {} -> {}".format(EXPECTED_STAGE1_TAG, tag_commit))
    print("5. authorized output directory: {}".format(OUT_DIR))
    print(
        "6. input checksum status: protocol={} stage1={} dataset={} split={}".format(
            protocol_pass,
            stage1_pass,
            dataset_pass,
            split_pass,
        )
    )

    print("7. feature lineages by model and horizon: {}".format(feature_lineages_summary))
    print("8. SHORT/REFERENCE/LONG definitions: {}".format(config_defs))
    print("9. model and horizon selection: refit_models=['Ridge','HGBR'], horizons=[1,3,5], context_only=['Persistence','ElasticNet','BCR-TCN']")
    print("10. final-reference mapping: {}".format(final_reference_mapping))
    print("11. NPZ inspection status: {}".format(npz_status))
    print("12. common-date feasibility: {}".format(common_date_pass))
    print("13. training-retention findings: {}".format(retention_note))
    print("14. number of tests passed: {}".format(test_result))
    print("15. deterministic result: {}".format(deterministic_result))
    print("16. source-preservation result: {}".format(source_preservation_pass))
    print(
        "17. checksum result: registry_pass={} coverage_pass={}".format(
            checksum_info["registry_pass"],
            checksum_info["coverage_pass"],
        )
    )
    print("18. final decision: {}".format(final_decision))
    print("19. generated files: {}".format(generated_files))


if __name__ == "__main__":
    build_stage()
