from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
DEFINITION_PATH = OUT_DIR / "ridge_definition_and_provenance.md"
SELECTION_PLAN_PATH = OUT_DIR / "ridge_selection_plan.json"
PREDICTIONS_PATH = OUT_DIR / "ridge_predictions.csv"
INNER_SELECTION_PATH = OUT_DIR / "ridge_inner_selection_results.csv"
SELECTED_CONFIG_PATH = OUT_DIR / "ridge_selected_configurations.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "ridge_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "ridge_metrics_pooled.csv"
EVENT_PREVALENCE_PATH = OUT_DIR / "ridge_event_prevalence.csv"
MANIFEST_PATH = OUT_DIR / "ridge_run_manifest.json"

LEGACY_FEATURE_JSONS = {
    1: ROOT / "results" / "metrics" / "main_linear_metrics_H1.json",
    3: ROOT / "results" / "metrics" / "main_linear_metrics_H3.json",
    5: ROOT / "results" / "metrics" / "main_linear_metrics_H5.json",
}

PROTOCOL_TAG = "corrected_protocol_v1"
MODEL_NAME = "Ridge"
HORIZONS = (1, 3, 5)
OUTER_FOLDS = (1, 2, 3)
CONFIGURATION_STATUS = "B"
CANDIDATE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SELECTION_METRIC = "inner_validation_MAE"
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


def load_lock(lock_path: Path) -> dict:
    with lock_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def verify_protocol_checksums(protocol_sha_path: Path, protocol_dir: Path) -> List[dict]:
    checks: List[dict] = []

    with protocol_sha_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

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


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def load_canonical_dataset(lock: dict) -> Tuple[pd.DataFrame, str]:
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

    # Lock metadata consistency
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


def load_feature_sets_by_horizon() -> Dict[int, List[str]]:
    feature_sets: Dict[int, List[str]] = {}
    for h, path in LEGACY_FEATURE_JSONS.items():
        if not path.exists():
            raise RuntimeError(f"Missing legacy feature provenance file for H{h}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("features")
        if not isinstance(features, list) or not features:
            raise RuntimeError(f"Invalid feature list in {path}")
        feature_sets[h] = [str(c) for c in features]

    return feature_sets


def build_submitted_feature_frame(
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
        raise RuntimeError(
            f"Canonical dataset is missing required submitted-feature columns for H{horizon}: {missing}"
        )

    if "BODin" not in df.columns:
        df["BODin"] = np.nan

    # Numeric coercion follows legacy submitted feature builder behavior.
    for c in [target_col, "Inflow", "TNin", "TOCin", "BODin", "temp_mean_c", "precip_total_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["C_N"] = df["TOCin"] / (df["TNin"] + 1e-6)

    # Horizon target and dates.
    df["feature_date"] = df[date_col]
    df["target_date"] = df[date_col] + pd.to_timedelta(horizon, unit="D")
    df["y_true"] = df[target_col].shift(-horizon)

    # Memory and rolling predictors (submitted non-v2 feature schema).
    df["TNout_lag1"] = df[target_col].shift(1)
    df["TNout_roll7"] = df[target_col].shift(1).rolling(7).mean()
    df["TNout_roll14"] = df[target_col].shift(1).rolling(14).mean()

    for col in ["Inflow", "TNin", "TOCin", "BODin"]:
        df[f"{col}_roll7"] = pd.to_numeric(df[col], errors="coerce").rolling(7).mean()

    df["temp_roll7"] = pd.to_numeric(df["temp_mean_c"], errors="coerce").rolling(7).mean()
    df["precip_sum3"] = pd.to_numeric(df["precip_total_mm"], errors="coerce").rolling(3).sum()

    doy = df[date_col].dt.dayofyear.values
    df["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise RuntimeError(
            f"Recovered submitted feature list has unavailable columns for H{horizon}: {missing_features}"
        )

    out_cols = ["feature_date", "target_date", "y_true"] + feature_cols
    out = df[out_cols].copy()
    out = out.dropna(subset=feature_cols + ["y_true"]).reset_index(drop=True)
    out = out.sort_values("feature_date").reset_index(drop=True)

    if bool(out.duplicated(subset=["feature_date", "target_date"]).any()):
        raise RuntimeError(
            f"Feature frame has duplicate feature_date-target_date rows for H{horizon}"
        )

    return out


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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    software_versions = {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    verification_payload = {
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

    # Verified inputs are now guaranteed.
    lock = load_lock(LOCK_PATH)
    split = load_split_assignments(SPLIT_PATH)
    dataset_df, _ = load_canonical_dataset(lock)
    feature_sets = load_feature_sets_by_horizon()

    split_sha = verification_payload["split_sha256"]
    git_commit = verification_payload["git_commit"]
    dataset_sha = verification_payload["dataset_sha256_observed"]
    git_branch = verification_payload["git_branch"]

    run_id = f"ridge_corrected_{git_commit[:12]}_{dataset_sha[:8]}_{split_sha[:8]}"

    date_col = lock["date_column"]
    target_col = lock["target_column"]

    prediction_rows: List[dict] = []
    selection_rows: List[dict] = []
    selected_rows: List[dict] = []
    mase_denoms: Dict[Tuple[int, int], float] = {}

    selected_alpha_by_fold: Dict[int, Dict[int, float]] = {h: {} for h in HORIZONS}
    feature_set_id_by_h: Dict[int, str] = {}

    for h in HORIZONS:
        feature_cols = feature_sets[h]
        feature_set_id = f"submitted_main_linear_features_H{h}_{len(feature_cols)}col"
        feature_set_id_by_h[h] = feature_set_id

        feature_frame = build_submitted_feature_frame(
            dataset_df=dataset_df,
            date_col=date_col,
            target_col=target_col,
            horizon=h,
            feature_cols=feature_cols,
        )

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

            # Inner selection partitions are sourced directly from corrected split labels.
            inner_fold_ids = sorted(
                split_hf[
                    (split_hf["outer_role"] == "outer_train")
                    & (split_hf["inner_role"] == "inner_validation")
                ]["inner_fold"].dropna().astype(int).unique().tolist()
            )
            if not inner_fold_ids:
                raise RuntimeError(f"No inner validation fold ids found for H{h} fold{fold}")

            candidate_metrics: List[dict] = []

            for alpha in CANDIDATE_ALPHAS:
                pooled_y_true: List[np.ndarray] = []
                pooled_y_pred: List[np.ndarray] = []
                all_inner_train_dates: List[pd.Timestamp] = []
                all_inner_val_dates: List[pd.Timestamp] = []

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

                    pipe = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "ridge",
                                Ridge(
                                    alpha=float(alpha),
                                    fit_intercept=True,
                                    solver="auto",
                                    random_state=42,
                                ),
                            ),
                        ]
                    )
                    pipe.fit(X_inner_train, y_inner_train)
                    y_hat_val = pipe.predict(X_inner_val)

                    pooled_y_true.append(y_inner_val)
                    pooled_y_pred.append(y_hat_val)

                    all_inner_train_dates.extend(inner_train["feature_date"].tolist())
                    all_inner_val_dates.extend(inner_val["feature_date"].tolist())

                y_true_cat = np.concatenate(pooled_y_true)
                y_pred_cat = np.concatenate(pooled_y_pred)
                err = y_pred_cat - y_true_cat
                mae = float(np.mean(np.abs(err)))
                mse = float(np.mean(np.square(err)))
                rmse = float(np.sqrt(mse))

                candidate_metrics.append(
                    {
                        "alpha": float(alpha),
                        "inner_validation_N": int(len(y_true_cat)),
                        "inner_validation_MAE": mae,
                        "inner_validation_MSE": mse,
                        "inner_validation_RMSE": rmse,
                        "inner_train_date_start": min(all_inner_train_dates).strftime("%Y-%m-%d"),
                        "inner_train_date_end": max(all_inner_train_dates).strftime("%Y-%m-%d"),
                        "inner_validation_date_start": min(all_inner_val_dates).strftime("%Y-%m-%d"),
                        "inner_validation_date_end": max(all_inner_val_dates).strftime("%Y-%m-%d"),
                    }
                )

            min_mae = min(c["inner_validation_MAE"] for c in candidate_metrics)
            tied = [
                c for c in candidate_metrics if abs(c["inner_validation_MAE"] - min_mae) <= TIE_TOL
            ]
            selected_alpha = float(min(c["alpha"] for c in tied))
            selected_alpha_by_fold[h][fold] = selected_alpha

            for c in candidate_metrics:
                is_selected = abs(c["alpha"] - selected_alpha) <= TIE_TOL
                selection_reason = (
                    "selected_min_inner_validation_MAE_with_tie_break_smallest_alpha"
                    if is_selected
                    else "not_selected_higher_inner_validation_MAE"
                )
                selection_rows.append(
                    {
                        "horizon": h,
                        "outer_fold": fold,
                        "candidate_alpha": c["alpha"],
                        "inner_validation_N": c["inner_validation_N"],
                        "inner_validation_MAE": c["inner_validation_MAE"],
                        "inner_validation_MSE": c["inner_validation_MSE"],
                        "inner_validation_RMSE": c["inner_validation_RMSE"],
                        "selected": bool(is_selected),
                        "selection_reason": selection_reason,
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
                    "selected_alpha": selected_alpha,
                    "feature_set_id": feature_set_id,
                    "scaler": "StandardScaler",
                    "imputer": "none",
                    "solver": "auto",
                    "fit_intercept": True,
                    "selection_metric": SELECTION_METRIC,
                    "selection_source": "corrected_split_assignment_inner_validation",
                }
            )

            # Final outer-fold fit uses only purged outer-train rows.
            outer_train = extract_partition(feature_frame, outer_train_rows, "outer_train", h, fold)
            outer_test = extract_partition(feature_frame, outer_test_rows, "outer_test", h, fold)

            X_outer_train = outer_train[feature_cols].to_numpy(dtype=float)
            y_outer_train = outer_train["y_true"].to_numpy(dtype=float)
            X_outer_test = outer_test[feature_cols].to_numpy(dtype=float)

            final_pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "ridge",
                        Ridge(
                            alpha=selected_alpha,
                            fit_intercept=True,
                            solver="auto",
                            random_state=42,
                        ),
                    ),
                ]
            )
            final_pipe.fit(X_outer_train, y_outer_train)
            y_hat_test = final_pipe.predict(X_outer_test)

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
                        "selected_alpha": selected_alpha,
                        "feature_set_id": feature_set_id,
                        "dataset_sha256": dataset_sha,
                        "split_sha256": split_sha,
                        "git_commit": git_commit,
                        "run_id": run_id,
                    }
                )

    pred = pd.DataFrame(prediction_rows)
    pred = pred.sort_values(["horizon", "outer_fold", "feature_date"]).reset_index(drop=True)

    if bool(pred.duplicated(subset=["horizon", "outer_fold", "feature_date"]).any()):
        raise RuntimeError("Duplicate ridge prediction keys detected")

    expected = split[split["outer_role"] == "outer_test"][
        ["horizon", "outer_fold", "feature_date", "target_date"]
    ].drop_duplicates()
    expected["feature_date"] = expected["feature_date"].dt.strftime("%Y-%m-%d")
    expected["target_date"] = expected["target_date"].dt.strftime("%Y-%m-%d")

    got = pred[["horizon", "outer_fold", "feature_date", "target_date"]].copy()

    expected_keys = set(
        tuple(r)
        for r in expected.sort_values(["horizon", "outer_fold", "feature_date"]).itertuples(index=False, name=None)
    )
    got_keys = set(
        tuple(r)
        for r in got.sort_values(["horizon", "outer_fold", "feature_date"]).itertuples(index=False, name=None)
    )

    if expected_keys != got_keys:
        missing = sorted(expected_keys - got_keys)[:5]
        extra = sorted(got_keys - expected_keys)[:5]
        raise RuntimeError(
            f"Ridge test-date set mismatch; missing={missing}; extra={extra}"
        )

    pred.to_csv(PREDICTIONS_PATH, index=False)

    selection_df = pd.DataFrame(selection_rows).sort_values(
        ["horizon", "outer_fold", "candidate_alpha"]
    )
    selection_df.to_csv(INNER_SELECTION_PATH, index=False)

    selected_df = pd.DataFrame(selected_rows).sort_values(["horizon", "outer_fold"])
    selected_df.to_csv(SELECTED_CONFIG_PATH, index=False)

    metrics_by_fold_rows: List[dict] = []
    for h in HORIZONS:
        for fold in OUTER_FOLDS:
            d = pred[(pred["horizon"] == h) & (pred["outer_fold"] == fold)].copy()
            if d.empty:
                raise RuntimeError(f"Missing ridge predictions for H{h} fold{fold}")

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

    pooled_rows: List[dict] = []
    for h in HORIZONS:
        d = pred[pred["horizon"] == h].copy()
        if d.empty:
            raise RuntimeError(f"Missing pooled ridge predictions for H{h}")

        mae = float(d["absolute_error"].mean())
        mse = float(d["squared_error"].mean())
        rmse = float(np.sqrt(mse))

        denoms_for_rows = d["outer_fold"].map(lambda f: mase_denoms[(h, int(f))]).astype(float)
        if np.isfinite(denoms_for_rows.to_numpy()).all():
            mase_values = d["absolute_error"].to_numpy(dtype=float) / (denoms_for_rows.to_numpy(dtype=float) + 1e-9)
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

    prevalence_rows: List[dict] = []
    for h in HORIZONS:
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
        "horizons": [1, 3, 5],
        "configuration_provenance_status": CONFIGURATION_STATUS,
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
        "feature_sets": {str(h): feature_sets[h] for h in HORIZONS},
        "candidate_alphas": CANDIDATE_ALPHAS,
        "selected_alphas_by_fold": {
            str(h): {str(f): float(selected_alpha_by_fold[h][f]) for f in OUTER_FOLDS}
            for h in HORIZONS
        },
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
            "src/train_main_linear.py",
            "src/build_ulsan_npz.py",
            "results/metrics/main_linear_metrics_H1.json",
            "results/metrics/main_linear_metrics_H3.json",
            "results/metrics/main_linear_metrics_H5.json",
            "results/metrics/main_linear_metrics.json",
            "results/tables/main_linear_leaderboard.csv",
            "reports/Optimization_Process_Exact_Trustable_Results.md",
            "revision_2026/02_corrected_pipeline/model_rerun_decision.csv",
            "revision_2026/03_corrected_protocol/rerun_execution_plan.csv"
        ],
        "notes": "Ridge rebuilt under corrected purged nested protocol with fold-local inner alpha selection and strict outer-test isolation.",
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
