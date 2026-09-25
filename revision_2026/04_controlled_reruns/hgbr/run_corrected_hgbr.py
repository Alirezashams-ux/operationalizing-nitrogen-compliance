from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
DEFINITION_PATH = OUT_DIR / "hgbr_definition_and_provenance.md"
SELECTION_PLAN_PATH = OUT_DIR / "hgbr_selection_plan.json"
PREDICTIONS_PATH = OUT_DIR / "hgbr_predictions.csv"
INNER_SELECTION_PATH = OUT_DIR / "hgbr_inner_selection_results.csv"
SELECTED_CONFIG_PATH = OUT_DIR / "hgbr_selected_configurations.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "hgbr_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "hgbr_metrics_pooled.csv"
EVENT_PREVALENCE_PATH = OUT_DIR / "hgbr_event_prevalence.csv"
MANIFEST_PATH = OUT_DIR / "hgbr_run_manifest.json"

PROTOCOL_TAG = "corrected_protocol_v1"
MODEL_NAME = "HGBR"
KNOWN_HORIZONS = (1, 3, 5)
OUTER_FOLDS = (1, 2, 3)
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
    out = subprocess.check_output(args, cwd=ROOT, text=True)
    return out.strip()


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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def params_to_json(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def verify_protocol_checksums(protocol_sha_path: Path, protocol_dir: Path) -> List[dict]:
    checks: List[dict] = []

    lines = [
        ln.strip()
        for ln in protocol_sha_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            raise RuntimeError(f"Malformed checksum line: {line}")
        expected_hash = parts[0]
        rel_name = parts[-1]
        path = protocol_dir / rel_name
        if not path.exists():
            raise RuntimeError(f"Protocol checksum target file is missing: {path}")
        observed_hash = sha256_file(path)
        ok = observed_hash == expected_hash
        checks.append(
            {
                "file": str(path),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "pass": ok,
            }
        )
        if not ok:
            raise RuntimeError(
                f"Protocol checksum mismatch for {path.name}: expected {expected_hash}, observed {observed_hash}"
            )
    return checks


def load_lock(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_canonical_dataset(lock: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    dataset_path = Path(lock["absolute_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Locked dataset file does not exist: {dataset_path}")

    observed_sha = sha256_file(dataset_path)
    expected_sha = lock["sha256"]
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Dataset checksum mismatch: expected {expected_sha}, observed {observed_sha}"
        )

    df = pd.read_csv(dataset_path)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    date_col = lock["date_column"]
    target_col = lock["target_column"]

    if date_col not in df.columns:
        raise RuntimeError(f"Locked date column is missing: {date_col}")
    if target_col not in df.columns:
        raise RuntimeError(f"Locked target column is missing: {target_col}")

    df[date_col] = normalize_date_col(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if len(df) != int(lock["row_count"]):
        raise RuntimeError(
            f"Dataset row count mismatch: expected {lock['row_count']}, observed {len(df)}"
        )
    if int(df[date_col].nunique()) != int(lock["unique_date_count"]):
        raise RuntimeError(
            "Dataset unique-date count mismatch: "
            f"expected {lock['unique_date_count']}, observed {int(df[date_col].nunique())}"
        )
    if int(df.duplicated(subset=[date_col]).sum()) != int(lock["duplicate_date_count"]):
        raise RuntimeError("Dataset duplicate-date count mismatch")

    min_date = df[date_col].min().strftime("%Y-%m-%d")
    max_date = df[date_col].max().strftime("%Y-%m-%d")
    if min_date != lock["minimum_date"] or max_date != lock["maximum_date"]:
        raise RuntimeError(
            "Dataset date-range mismatch: "
            f"expected [{lock['minimum_date']}, {lock['maximum_date']}], observed [{min_date}, {max_date}]"
        )

    all_days = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    missing_days = int(len(all_days) - int(df[date_col].nunique()))
    if missing_days != int(lock["missing_calendar_date_count"]):
        raise RuntimeError(
            "Dataset missing-calendar-date mismatch: "
            f"expected {lock['missing_calendar_date_count']}, observed {missing_days}"
        )

    if df[target_col].isna().any():
        raise RuntimeError("Target column contains NaN values in the locked dataset")

    return df, observed_sha


def load_split_assignments(path: Path) -> pd.DataFrame:
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
        raise RuntimeError(f"Split assignment is missing required columns: {sorted(missing)}")

    split["feature_date"] = normalize_date_col(split["feature_date"])
    split["target_date"] = normalize_date_col(split["target_date"])
    if split["feature_date"].isna().any() or split["target_date"].isna().any():
        raise RuntimeError("Split assignment has unparsable feature_date or target_date")

    split["purged_outer_boundary"] = as_bool(split["purged_outer_boundary"])
    split["purged_inner_boundary"] = as_bool(split["purged_inner_boundary"])
    return split


def load_selection_plan(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    required_top = {
        "provenance_status_by_horizon",
        "candidate_space_by_horizon",
        "feature_set_by_horizon",
        "selection_metric",
        "inner_split_source",
        "imputation_method",
        "scaling_method",
        "loss",
        "early_stopping",
        "validation_strategy",
        "random_seed",
        "tie_break_rule",
        "outer_test_isolation",
        "refit_strategy",
        "search_method",
        "search_budget",
    }
    missing = sorted(required_top.difference(payload.keys()))
    if missing:
        raise RuntimeError(f"Selection plan is missing required fields: {missing}")

    statuses = {int(k): str(v) for k, v in payload["provenance_status_by_horizon"].items()}
    authorized_horizons = payload.get("authorized_horizons", sorted(statuses.keys()))
    authorized_horizons = [int(h) for h in authorized_horizons]

    for h in authorized_horizons:
        if h not in KNOWN_HORIZONS:
            raise RuntimeError(f"Unsupported horizon in selection plan: {h}")
        status = statuses.get(h)
        if status is None:
            raise RuntimeError(f"Missing provenance status for H{h}")
        if status in {"C", "D"}:
            raise RuntimeError(
                f"H{h} is classified {status}; corrected fitting is forbidden for status C/D horizons"
            )
        if status not in {"A", "B"}:
            raise RuntimeError(f"Invalid provenance status for H{h}: {status}")

    candidate_space_by_h = {
        int(k): v for k, v in payload["candidate_space_by_horizon"].items()
    }
    feature_set_by_h = {int(k): v for k, v in payload["feature_set_by_horizon"].items()}

    budget_obj = payload["search_budget"]
    if isinstance(budget_obj, dict):
        budget_by_h = {int(k): int(v) for k, v in budget_obj.items()}
    else:
        budget_by_h = {h: int(budget_obj) for h in authorized_horizons}

    random_seed = payload["random_seed"]
    if "candidate_sampling_seed_by_horizon" not in random_seed:
        raise RuntimeError("Selection plan random_seed must include candidate_sampling_seed_by_horizon")
    candidate_seed_by_h = {
        int(k): int(v)
        for k, v in random_seed["candidate_sampling_seed_by_horizon"].items()
    }

    for h in authorized_horizons:
        if h not in candidate_space_by_h:
            raise RuntimeError(f"Missing candidate space for H{h}")
        if h not in feature_set_by_h:
            raise RuntimeError(f"Missing feature set for H{h}")
        if h not in budget_by_h:
            raise RuntimeError(f"Missing search budget for H{h}")
        if h not in candidate_seed_by_h:
            raise RuntimeError(f"Missing candidate sampling seed for H{h}")

    payload["_statuses"] = statuses
    payload["_authorized_horizons"] = tuple(sorted(authorized_horizons))
    payload["_candidate_space_by_h"] = candidate_space_by_h
    payload["_feature_set_by_h"] = feature_set_by_h
    payload["_budget_by_h"] = budget_by_h
    payload["_candidate_seed_by_h"] = candidate_seed_by_h

    return payload


def load_npz_feature_frame(
    npz_path: Path,
    horizon: int,
    date_to_target: Dict[pd.Timestamp, float],
    expected_feature_count: int,
) -> Tuple[pd.DataFrame, List[str]]:
    if not npz_path.exists():
        raise RuntimeError(f"Feature NPZ is missing for H{horizon}: {npz_path}")

    z = np.load(npz_path, allow_pickle=True)
    required_keys = {"X", "dates", "feature_names"}
    if not required_keys.issubset(set(z.files)):
        raise RuntimeError(
            f"Feature NPZ for H{horizon} missing required arrays: {sorted(required_keys)}"
        )

    X = np.asarray(z["X"], dtype=float)
    dates = normalize_date_col(pd.Series(z["dates"]))
    feature_names = [str(c) for c in list(z["feature_names"])]

    if X.ndim != 2:
        raise RuntimeError(f"Feature matrix for H{horizon} must be 2D")
    if len(dates) != X.shape[0]:
        raise RuntimeError(
            f"Date length mismatch for H{horizon}: {len(dates)} vs {X.shape[0]}"
        )
    if len(feature_names) != X.shape[1]:
        raise RuntimeError(
            f"Feature-name count mismatch for H{horizon}: {len(feature_names)} vs {X.shape[1]}"
        )
    if expected_feature_count and len(feature_names) != int(expected_feature_count):
        raise RuntimeError(
            f"Expected {expected_feature_count} features for H{horizon}, observed {len(feature_names)}"
        )
    if dates.isna().any():
        raise RuntimeError(f"Feature NPZ has unparsable dates for H{horizon}")

    frame = pd.DataFrame(X, columns=feature_names)
    frame.insert(0, "feature_date", dates)
    frame["target_date"] = frame["feature_date"] + pd.to_timedelta(horizon, unit="D")
    frame["y_true"] = frame["target_date"].map(date_to_target)

    if frame[feature_names].isna().any().any():
        raise RuntimeError(f"Feature NPZ contains NaN feature values for H{horizon}")

    frame = frame.dropna(subset=["y_true"]).copy()
    frame = frame.sort_values("feature_date").reset_index(drop=True)

    if bool(frame.duplicated(subset=["feature_date", "target_date"]).any()):
        raise RuntimeError(
            f"Feature frame has duplicate feature_date-target_date rows for H{horizon}"
        )

    return frame, feature_names


def extract_partition(
    feature_frame: pd.DataFrame,
    split_rows: pd.DataFrame,
    label: str,
    horizon: int,
    outer_fold: int,
) -> pd.DataFrame:
    keys = split_rows[["feature_date", "target_date"]].drop_duplicates().copy()
    merged = keys.merge(
        feature_frame,
        on=["feature_date", "target_date"],
        how="left",
        validate="one_to_one",
    )

    if merged["y_true"].isna().any():
        missing = merged[merged["y_true"].isna()][["feature_date", "target_date"]].head(5)
        pairs = [
            f"{r.feature_date.strftime('%Y-%m-%d')}->{r.target_date.strftime('%Y-%m-%d')}"
            for r in missing.itertuples(index=False)
        ]
        raise RuntimeError(
            f"Missing feature/target rows for {label} H{horizon} fold{outer_fold}: {pairs}"
        )

    merged = merged.sort_values("feature_date").reset_index(drop=True)
    return merged


def compute_mase_denom(y_train: np.ndarray, m: int = 1) -> float:
    if len(y_train) <= m:
        return np.nan
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def sample_param_value(rng: np.random.Generator, spec: Dict[str, Any]) -> Any:
    dist = str(spec["distribution"])
    low = spec["low"]
    high = spec["high"]

    if dist == "log_uniform":
        return float(np.exp(rng.uniform(np.log(float(low)), np.log(float(high)))))
    if dist == "int_uniform":
        return int(rng.integers(int(low), int(high) + 1))
    raise RuntimeError(f"Unsupported parameter distribution: {dist}")


def build_candidate_configurations(
    horizon: int,
    candidate_space: Dict[str, Any],
    budget: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if budget <= 0:
        raise RuntimeError(f"Search budget must be positive for H{horizon}")

    param_space = candidate_space["parameter_space"]
    fixed_params = dict(candidate_space.get("fixed_parameters", {}))

    param_order = [
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "max_bins",
    ]
    missing_params = [p for p in param_order if p not in param_space]
    if missing_params:
        raise RuntimeError(
            f"Candidate space for H{horizon} missing required parameters: {missing_params}"
        )

    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    max_attempts = budget * 200

    while len(candidates) < budget:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {budget} unique candidates for H{horizon}"
            )

        sampled = {p: sample_param_value(rng, param_space[p]) for p in param_order}
        full_params = dict(sampled)
        full_params.update(fixed_params)
        params_json = params_to_json(full_params)

        if params_json in seen:
            continue
        seen.add(params_json)

        cfg_id = f"h{horizon}_cfg_{len(candidates) + 1:03d}"
        candidates.append(
            {
                "candidate_configuration_id": cfg_id,
                "candidate_parameters": full_params,
                "candidate_parameters_json": params_json,
            }
        )

    return candidates


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DEFINITION_PATH.exists() or not SELECTION_PLAN_PATH.exists():
        raise RuntimeError(
            "Required predeclared governance files are missing: "
            "hgbr_definition_and_provenance.md and/or hgbr_selection_plan.json"
        )

    selection_plan = load_selection_plan(SELECTION_PLAN_PATH)

    software_versions = {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    verification_payload: Dict[str, Any] = {
        "dataset_path": "",
        "dataset_sha256_expected": "",
        "dataset_sha256_observed": "",
        "split_path": str(SPLIT_PATH.resolve()),
        "split_sha256": "",
        "protocol_tag": PROTOCOL_TAG,
        "git_branch": "",
        "git_commit": "",
        "verification_pass": False,
        "timestamp": utc_now_iso(),
        **software_versions,
    }

    try:
        lock = load_lock(LOCK_PATH)
        verification_payload["dataset_path"] = str(Path(lock["absolute_path"]).resolve())
        verification_payload["dataset_sha256_expected"] = lock["sha256"]

        active_branch = git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        verification_payload["git_branch"] = active_branch
        if active_branch != "controlled-reruns-v1":
            raise RuntimeError(
                f"Active branch is {active_branch}, expected controlled-reruns-v1"
            )

        git_commit = git_output(["git", "rev-parse", "HEAD"])
        verification_payload["git_commit"] = git_commit

        protocol_checks = verify_protocol_checksums(PROTOCOL_SHA_PATH, PROTOCOL_DIR)

        dataset_df, dataset_sha = load_canonical_dataset(lock)
        verification_payload["dataset_sha256_observed"] = dataset_sha

        split_sha = sha256_file(SPLIT_PATH)
        verification_payload["split_sha256"] = split_sha

        verification_payload["verification_pass"] = True
        verification_payload["protocol_checksum_checks"] = protocol_checks
        write_json(INPUT_VERIFICATION_PATH, verification_payload)

    except Exception as exc:
        verification_payload["verification_pass"] = False
        verification_payload["error"] = str(exc)
        write_json(INPUT_VERIFICATION_PATH, verification_payload)
        raise

    lock = load_lock(LOCK_PATH)
    split = load_split_assignments(SPLIT_PATH)
    dataset_df, _ = load_canonical_dataset(lock)

    statuses = selection_plan["_statuses"]
    authorized_horizons = selection_plan["_authorized_horizons"]
    candidate_space_by_h = selection_plan["_candidate_space_by_h"]
    feature_set_by_h = selection_plan["_feature_set_by_h"]
    budget_by_h = selection_plan["_budget_by_h"]
    candidate_seed_by_h = selection_plan["_candidate_seed_by_h"]

    split_sha = verification_payload["split_sha256"]
    git_commit = verification_payload["git_commit"]
    dataset_sha = verification_payload["dataset_sha256_observed"]
    git_branch = verification_payload["git_branch"]

    run_id = f"hgbr_corrected_{git_commit[:12]}_{dataset_sha[:8]}_{split_sha[:8]}"

    date_col = lock["date_column"]
    target_col = lock["target_column"]

    date_to_target = {
        d: float(y)
        for d, y in zip(
            dataset_df[date_col].tolist(),
            pd.to_numeric(dataset_df[target_col], errors="coerce").tolist(),
        )
    }
    if any(pd.isna(v) for v in date_to_target.values()):
        raise RuntimeError("Canonical dataset target contains NaN after numeric coercion")

    prediction_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    mase_denoms: Dict[Tuple[int, int], float] = {}

    selected_by_fold: Dict[int, Dict[int, Dict[str, Any]]] = {
        h: {} for h in authorized_horizons
    }
    feature_manifest: Dict[str, Dict[str, Any]] = {}
    candidate_manifest: Dict[str, List[Dict[str, Any]]] = {}

    for h in authorized_horizons:
        feature_info = feature_set_by_h[h]
        feature_set_id = str(feature_info["feature_set_id"])
        feature_npz_path = ROOT / str(feature_info["feature_npz_path"])
        expected_feature_count = int(feature_info.get("expected_feature_count", 0))

        feature_frame, feature_cols = load_npz_feature_frame(
            npz_path=feature_npz_path,
            horizon=h,
            date_to_target=date_to_target,
            expected_feature_count=expected_feature_count,
        )

        feature_manifest[str(h)] = {
            "feature_set_id": feature_set_id,
            "feature_npz_path": str(feature_npz_path.resolve()),
            "feature_count": len(feature_cols),
            "feature_names": feature_cols,
        }

        candidate_space = candidate_space_by_h[h]
        budget = int(budget_by_h[h])
        seed = int(candidate_seed_by_h[h])
        candidates = build_candidate_configurations(
            horizon=h,
            candidate_space=candidate_space,
            budget=budget,
            seed=seed,
        )
        candidate_manifest[str(h)] = [
            {
                "candidate_configuration_id": c["candidate_configuration_id"],
                "candidate_parameters_json": c["candidate_parameters_json"],
            }
            for c in candidates
        ]

        split_h = split[split["horizon"] == h].copy()
        if split_h.empty:
            raise RuntimeError(f"No split rows found for horizon {h}")

        for fold in OUTER_FOLDS:
            split_hf = split_h[split_h["outer_fold"] == fold].copy()
            if split_hf.empty:
                raise RuntimeError(f"No split rows found for horizon {h}, fold {fold}")

            outer_test_rows = split_hf[
                split_hf["outer_role"] == "outer_test"
            ][["feature_date", "target_date"]].drop_duplicates()

            outer_train_rows = split_hf[
                (split_hf["outer_role"] == "outer_train")
                & (~split_hf["purged_outer_boundary"])
            ][["feature_date", "target_date"]].drop_duplicates()

            if outer_test_rows.empty:
                raise RuntimeError(f"No outer_test rows for H{h} fold{fold}")
            if outer_train_rows.empty:
                raise RuntimeError(f"No purged outer_train rows for H{h} fold{fold}")

            inner_fold_ids = sorted(
                split_hf[
                    (split_hf["outer_role"] == "outer_train")
                    & (split_hf["inner_role"] == "inner_validation")
                ]["inner_fold"].dropna().astype(int).unique().tolist()
            )
            if not inner_fold_ids:
                raise RuntimeError(f"No inner validation fold ids found for H{h} fold{fold}")

            candidate_metrics: List[Dict[str, Any]] = []

            for candidate in candidates:
                cfg_id = str(candidate["candidate_configuration_id"])
                params = dict(candidate["candidate_parameters"])
                params_json = str(candidate["candidate_parameters_json"])

                pooled_y_true: List[np.ndarray] = []
                pooled_y_pred: List[np.ndarray] = []
                all_inner_train_dates: List[pd.Timestamp] = []
                all_inner_val_dates: List[pd.Timestamp] = []
                inner_train_n = 0
                inner_val_n = 0

                for inner_fold in inner_fold_ids:
                    inner_train_rows = split_hf[
                        (split_hf["outer_role"] == "outer_train")
                        & (split_hf["inner_fold"].astype(float) == float(inner_fold))
                        & (split_hf["inner_role"] == "inner_train")
                        & (~split_hf["purged_inner_boundary"])
                    ][["feature_date", "target_date"]].drop_duplicates()

                    inner_val_rows = split_hf[
                        (split_hf["outer_role"] == "outer_train")
                        & (split_hf["inner_fold"].astype(float) == float(inner_fold))
                        & (split_hf["inner_role"] == "inner_validation")
                    ][["feature_date", "target_date"]].drop_duplicates()

                    if inner_train_rows.empty:
                        raise RuntimeError(
                            f"No purged inner_train rows for H{h} fold{fold} inner_fold{inner_fold}"
                        )
                    if inner_val_rows.empty:
                        raise RuntimeError(
                            f"No inner_validation rows for H{h} fold{fold} inner_fold{inner_fold}"
                        )

                    inner_train = extract_partition(
                        feature_frame, inner_train_rows, "inner_train", h, fold
                    )
                    inner_val = extract_partition(
                        feature_frame, inner_val_rows, "inner_validation", h, fold
                    )

                    X_inner_train = inner_train[feature_cols].to_numpy(dtype=float)
                    y_inner_train = inner_train["y_true"].to_numpy(dtype=float)
                    X_inner_val = inner_val[feature_cols].to_numpy(dtype=float)
                    y_inner_val = inner_val["y_true"].to_numpy(dtype=float)

                    model = HistGradientBoostingRegressor(**params)
                    model.fit(X_inner_train, y_inner_train)
                    y_hat_val = model.predict(X_inner_val)

                    pooled_y_true.append(y_inner_val)
                    pooled_y_pred.append(y_hat_val)
                    all_inner_train_dates.extend(inner_train["feature_date"].tolist())
                    all_inner_val_dates.extend(inner_val["feature_date"].tolist())
                    inner_train_n += int(len(inner_train))
                    inner_val_n += int(len(inner_val))

                y_true_cat = np.concatenate(pooled_y_true)
                y_pred_cat = np.concatenate(pooled_y_pred)
                err = y_pred_cat - y_true_cat
                mae = float(np.mean(np.abs(err)))
                mse = float(np.mean(np.square(err)))
                rmse = float(np.sqrt(mse))

                candidate_metrics.append(
                    {
                        "candidate_configuration_id": cfg_id,
                        "candidate_parameters_json": params_json,
                        "candidate_parameters": params,
                        "inner_train_N": inner_train_n,
                        "inner_validation_N": inner_val_n,
                        "inner_validation_MAE": mae,
                        "inner_validation_MSE": mse,
                        "inner_validation_RMSE": rmse,
                        "inner_train_date_start": min(all_inner_train_dates).strftime("%Y-%m-%d"),
                        "inner_train_date_end": max(all_inner_train_dates).strftime("%Y-%m-%d"),
                        "inner_validation_date_start": min(all_inner_val_dates).strftime("%Y-%m-%d"),
                        "inner_validation_date_end": max(all_inner_val_dates).strftime("%Y-%m-%d"),
                    }
                )

            best_mae = min(c["inner_validation_MAE"] for c in candidate_metrics)
            tied = [c for c in candidate_metrics if abs(c["inner_validation_MAE"] - best_mae) <= TIE_TOL]
            selected = sorted(tied, key=lambda c: c["candidate_configuration_id"])[0]
            selected_id = str(selected["candidate_configuration_id"])
            selected_params = dict(selected["candidate_parameters"])
            selected_params_json = str(selected["candidate_parameters_json"])

            selected_by_fold[h][fold] = {
                "selected_configuration_id": selected_id,
                "selected_parameters_json": selected_params_json,
                "selected_parameters": selected_params,
            }

            for c in candidate_metrics:
                is_selected = c["candidate_configuration_id"] == selected_id
                reason = (
                    "selected_min_inner_validation_MAE_with_tie_break_smallest_candidate_configuration_id"
                    if is_selected
                    else "not_selected_higher_inner_validation_MAE_or_tie_break_not_won"
                )
                selection_rows.append(
                    {
                        "horizon": h,
                        "outer_fold": fold,
                        "candidate_configuration_id": c["candidate_configuration_id"],
                        "candidate_parameters_json": c["candidate_parameters_json"],
                        "inner_train_N": c["inner_train_N"],
                        "inner_validation_N": c["inner_validation_N"],
                        "inner_validation_MAE": c["inner_validation_MAE"],
                        "inner_validation_MSE": c["inner_validation_MSE"],
                        "inner_validation_RMSE": c["inner_validation_RMSE"],
                        "selected": bool(is_selected),
                        "selection_reason": reason,
                        "inner_train_date_start": c["inner_train_date_start"],
                        "inner_train_date_end": c["inner_train_date_end"],
                        "inner_validation_date_start": c["inner_validation_date_start"],
                        "inner_validation_date_end": c["inner_validation_date_end"],
                        "outer_test_accessed_during_selection": False,
                    }
                )

            selected_rows.append(
                {
                    "horizon": h,
                    "outer_fold": fold,
                    "selected_configuration_id": selected_id,
                    "selected_parameters_json": selected_params_json,
                    "feature_set_id": feature_set_id,
                    "loss": selection_plan["loss"],
                    "early_stopping": bool(selection_plan["early_stopping"]),
                    "random_seed": int(selection_plan["random_seed"]["model_fit_random_state"]),
                    "selection_metric": str(selection_plan["selection_metric"]),
                    "selection_source": "corrected_split_assignment_inner_validation",
                }
            )

            outer_train = extract_partition(feature_frame, outer_train_rows, "outer_train", h, fold)
            outer_test = extract_partition(feature_frame, outer_test_rows, "outer_test", h, fold)

            X_outer_train = outer_train[feature_cols].to_numpy(dtype=float)
            y_outer_train = outer_train["y_true"].to_numpy(dtype=float)
            X_outer_test = outer_test[feature_cols].to_numpy(dtype=float)

            final_model = HistGradientBoostingRegressor(**selected_params)
            final_model.fit(X_outer_train, y_outer_train)
            y_hat_test = final_model.predict(X_outer_test)

            mase_denoms[(h, fold)] = compute_mase_denom(y_outer_train, m=1)

            for row, yp in zip(outer_test.itertuples(index=False), y_hat_test):
                y_true = float(row.y_true)
                y_pred = float(yp)
                err = y_pred - y_true
                prediction_rows.append(
                    {
                        "model": MODEL_NAME,
                        "horizon": h,
                        "outer_fold": fold,
                        "feature_date": row.feature_date.strftime("%Y-%m-%d"),
                        "target_date": row.target_date.strftime("%Y-%m-%d"),
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "error": err,
                        "absolute_error": abs(err),
                        "squared_error": err * err,
                        "feature_set_id": feature_set_id,
                        "selected_configuration_id": selected_id,
                        "selected_parameters_json": selected_params_json,
                        "random_seed": int(selection_plan["random_seed"]["model_fit_random_state"]),
                        "dataset_sha256": dataset_sha,
                        "split_sha256": split_sha,
                        "git_commit": git_commit,
                        "run_id": run_id,
                    }
                )

    pred = pd.DataFrame(prediction_rows)
    pred = pred.sort_values(["horizon", "outer_fold", "feature_date"]).reset_index(drop=True)

    if bool(pred.duplicated(subset=["horizon", "outer_fold", "feature_date"]).any()):
        raise RuntimeError("Duplicate HGBR prediction keys detected")

    expected = split[
        (split["outer_role"] == "outer_test")
        & (split["horizon"].isin(authorized_horizons))
    ][["horizon", "outer_fold", "feature_date", "target_date"]].drop_duplicates()
    expected["feature_date"] = expected["feature_date"].dt.strftime("%Y-%m-%d")
    expected["target_date"] = expected["target_date"].dt.strftime("%Y-%m-%d")

    got = pred[["horizon", "outer_fold", "feature_date", "target_date"]].copy()

    expected_keys = set(
        tuple(r)
        for r in expected.sort_values(["horizon", "outer_fold", "feature_date"]).itertuples(
            index=False, name=None
        )
    )
    got_keys = set(
        tuple(r)
        for r in got.sort_values(["horizon", "outer_fold", "feature_date"]).itertuples(
            index=False, name=None
        )
    )

    if expected_keys != got_keys:
        missing = sorted(expected_keys - got_keys)[:5]
        extra = sorted(got_keys - expected_keys)[:5]
        raise RuntimeError(f"HGBR test-date set mismatch; missing={missing}; extra={extra}")

    pred.to_csv(PREDICTIONS_PATH, index=False)

    selection_df = pd.DataFrame(selection_rows).sort_values(
        ["horizon", "outer_fold", "candidate_configuration_id"]
    )
    selection_df.to_csv(INNER_SELECTION_PATH, index=False)

    selected_df = pd.DataFrame(selected_rows).sort_values(["horizon", "outer_fold"])
    selected_df.to_csv(SELECTED_CONFIG_PATH, index=False)

    metrics_by_fold_rows: List[Dict[str, Any]] = []
    for h in authorized_horizons:
        for fold in OUTER_FOLDS:
            d = pred[(pred["horizon"] == h) & (pred["outer_fold"] == fold)].copy()
            if d.empty:
                raise RuntimeError(f"Missing HGBR predictions for H{h} fold{fold}")

            mae = float(d["absolute_error"].mean())
            mse = float(d["squared_error"].mean())
            rmse = float(np.sqrt(mse))
            denom = mase_denoms[(h, fold)]
            mase = float(mae / (denom + 1e-9)) if np.isfinite(denom) else np.nan

            metrics_by_fold_rows.append(
                {
                    "model": MODEL_NAME,
                    "horizon": h,
                    "outer_fold": fold,
                    "N": int(len(d)),
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MASE": mase,
                    "date_start": d["feature_date"].min(),
                    "date_end": d["feature_date"].max(),
                    "dataset_sha256": dataset_sha,
                    "split_sha256": split_sha,
                    "run_id": run_id,
                }
            )

    metrics_by_fold = pd.DataFrame(metrics_by_fold_rows).sort_values(["horizon", "outer_fold"])
    metrics_by_fold.to_csv(METRICS_BY_FOLD_PATH, index=False)

    pooled_rows: List[Dict[str, Any]] = []
    for h in authorized_horizons:
        d = pred[pred["horizon"] == h].copy()
        if d.empty:
            raise RuntimeError(f"Missing pooled HGBR predictions for H{h}")

        mae = float(d["absolute_error"].mean())
        mse = float(d["squared_error"].mean())
        rmse = float(np.sqrt(mse))

        denoms_for_rows = d["outer_fold"].map(lambda f: mase_denoms[(h, int(f))]).astype(float)
        if np.isfinite(denoms_for_rows.to_numpy()).all():
            mase_values = d["absolute_error"].to_numpy(dtype=float) / (
                denoms_for_rows.to_numpy(dtype=float) + 1e-9
            )
            mase = float(np.mean(mase_values))
        else:
            mase = np.nan

        pooled_rows.append(
            {
                "model": MODEL_NAME,
                "horizon": h,
                "outer_fold": "pooled",
                "N": int(len(d)),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MASE": mase,
                "date_start": d["feature_date"].min(),
                "date_end": d["feature_date"].max(),
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "run_id": run_id,
            }
        )

    metrics_pooled = pd.DataFrame(pooled_rows).sort_values(["horizon"])
    metrics_pooled.to_csv(METRICS_POOLED_PATH, index=False)

    prevalence_rows: List[Dict[str, Any]] = []
    for h in authorized_horizons:
        for fold in OUTER_FOLDS:
            d = pred[(pred["horizon"] == h) & (pred["outer_fold"] == fold)].copy()
            n = int(len(d))
            ev15 = int((d["y_true"] >= 15.0).sum())
            ev16 = int((d["y_true"] >= 16.0).sum())
            ev17 = int((d["y_true"] >= 17.0).sum())

            prevalence_rows.append(
                {
                    "model": MODEL_NAME,
                    "horizon": h,
                    "outer_fold": fold,
                    "N": n,
                    "events_tau15": ev15,
                    "events_tau16": ev16,
                    "events_tau17": ev17,
                    "prevalence_tau15": float(ev15 / n) if n else np.nan,
                    "prevalence_tau16": float(ev16 / n) if n else np.nan,
                    "prevalence_tau17": float(ev17 / n) if n else np.nan,
                    "dataset_sha256": dataset_sha,
                    "split_sha256": split_sha,
                    "run_id": run_id,
                }
            )

    prevalence_df = pd.DataFrame(prevalence_rows).sort_values(["horizon", "outer_fold"])
    prevalence_df.to_csv(EVENT_PREVALENCE_PATH, index=False)

    manifest = {
        "run_id": run_id,
        "model": MODEL_NAME,
        "horizons": [int(h) for h in authorized_horizons],
        "provenance_status_by_horizon": {str(h): statuses[h] for h in authorized_horizons},
        "dataset_path": verification_payload["dataset_path"],
        "dataset_sha256": dataset_sha,
        "split_path": str(SPLIT_PATH.resolve()),
        "split_sha256": split_sha,
        "protocol_tag": PROTOCOL_TAG,
        "git_branch": git_branch,
        "git_commit": git_commit,
        "execution_command": f"{sys.executable} {Path(__file__).resolve()}",
        "timestamp": utc_now_iso(),
        "software_versions": software_versions,
        "feature_sets": feature_manifest,
        "candidate_spaces": {
            str(h): candidate_space_by_h[h] for h in authorized_horizons
        },
        "search_method": selection_plan["search_method"],
        "search_budget": {str(h): int(budget_by_h[h]) for h in authorized_horizons},
        "random_seeds": selection_plan["random_seed"],
        "selected_configurations_by_fold": {
            str(h): {
                str(f): {
                    "selected_configuration_id": selected_by_fold[h][f]["selected_configuration_id"],
                    "selected_parameters_json": selected_by_fold[h][f]["selected_parameters_json"],
                }
                for f in OUTER_FOLDS
            }
            for h in authorized_horizons
        },
        "candidate_configurations": candidate_manifest,
        "prediction_file": str(PREDICTIONS_PATH.resolve()),
        "prediction_sha256": sha256_file(PREDICTIONS_PATH),
        "metrics_files": [
            str(INNER_SELECTION_PATH.resolve()),
            str(SELECTED_CONFIG_PATH.resolve()),
            str(METRICS_BY_FOLD_PATH.resolve()),
            str(METRICS_POOLED_PATH.resolve()),
            str(EVENT_PREVALENCE_PATH.resolve()),
        ],
        "metrics_sha256": {
            INNER_SELECTION_PATH.name: sha256_file(INNER_SELECTION_PATH),
            SELECTED_CONFIG_PATH.name: sha256_file(SELECTED_CONFIG_PATH),
            METRICS_BY_FOLD_PATH.name: sha256_file(METRICS_BY_FOLD_PATH),
            METRICS_POOLED_PATH.name: sha256_file(METRICS_POOLED_PATH),
            EVENT_PREVALENCE_PATH.name: sha256_file(EVENT_PREVALENCE_PATH),
        },
        "test_result": "pending_not_run",
        "legacy_sources": [
            "src/train_hgbr_optuna_multi.py",
            "src/train_hgbr_optuna_v2.py",
            "src/save_hgbr_predictions.py",
            "src/save_hgbr_predictions_v2.py",
            "reports/feature_engineering_evolution_complete_report_20260401.md",
            "deliverables/si_upload_packages_20260329/generated/final_chosen_hyperparameters_by_horizon_model.csv",
            "revision_2026/02_corrected_pipeline/model_rerun_decision.csv",
            "revision_2026/02_corrected_pipeline/temporal_leakage_audit_decision.md",
            "revision_2026/03_corrected_protocol/corrected_evaluation_protocol.md",
            "revision_2026/03_corrected_protocol/rerun_execution_plan.csv"
        ],
        "notes": "HGBR rebuilt under corrected protocol using predeclared recovered candidate spaces with fold-local purged inner validation and strict outer-test isolation.",
    }
    write_json(MANIFEST_PATH, manifest)

    print(f"Wrote {INPUT_VERIFICATION_PATH}")
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {INNER_SELECTION_PATH}")
    print(f"Wrote {SELECTED_CONFIG_PATH}")
    print(f"Wrote {METRICS_BY_FOLD_PATH}")
    print(f"Wrote {METRICS_POOLED_PATH}")
    print(f"Wrote {EVENT_PREVALENCE_PATH}")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
