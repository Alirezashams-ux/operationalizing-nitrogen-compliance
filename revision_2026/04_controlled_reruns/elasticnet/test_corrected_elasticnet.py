from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
PERSISTENCE_DIR = ROOT / "revision_2026" / "04_controlled_reruns" / "persistence"

RUN_SCRIPT = OUT_DIR / "run_corrected_elasticnet.py"
LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"
SELECTION_PLAN_PATH = OUT_DIR / "elasticnet_selection_plan.json"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
PREDICTIONS_PATH = OUT_DIR / "elasticnet_predictions.csv"
INNER_SELECTION_PATH = OUT_DIR / "elasticnet_inner_selection_results.csv"
SELECTED_CONFIG_PATH = OUT_DIR / "elasticnet_selected_configurations.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "elasticnet_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "elasticnet_metrics_pooled.csv"
EVENT_PREVALENCE_PATH = OUT_DIR / "elasticnet_event_prevalence.csv"
MANIFEST_PATH = OUT_DIR / "elasticnet_run_manifest.json"

PERSISTENCE_PRED_PATH = PERSISTENCE_DIR / "persistence_predictions.csv"
PERSISTENCE_EVENT_PATH = PERSISTENCE_DIR / "persistence_event_prevalence.csv"

HORIZONS = (1, 3, 5)
FOLDS = (1, 2, 3)
OUTPUT_FILES = [
    PREDICTIONS_PATH,
    INNER_SELECTION_PATH,
    SELECTED_CONFIG_PATH,
    METRICS_BY_FOLD_PATH,
    METRICS_POOLED_PATH,
    EVENT_PREVALENCE_PATH,
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_runner() -> None:
    subprocess.run([sys.executable, str(RUN_SCRIPT)], cwd=ROOT, check=True)


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise").dt.normalize()


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


def load_expected_protocol_hashes() -> Dict[str, str]:
    hashes = {}
    for line in PROTOCOL_SHA_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        expected = parts[0]
        rel = parts[-1]
        hashes[rel] = expected
    return hashes


def update_manifest_test_result(success: bool, tests_run: int, failures: int, errors: int) -> None:
    if not MANIFEST_PATH.exists():
        return
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["test_result"] = "pass_15_of_15" if success else f"fail_{failures + errors}_of_15"
    payload["test_command"] = f"{sys.executable} {Path(__file__).resolve()}"
    payload["test_summary"] = f"Ran {tests_run} tests; failures={failures}; errors={errors}."
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestCorrectedElasticNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_runner()

        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        cls.selection_plan = json.loads(SELECTION_PLAN_PATH.read_text(encoding="utf-8"))
        cls.input_ver = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        cls.pred = pd.read_csv(PREDICTIONS_PATH)
        cls.inner = pd.read_csv(INNER_SELECTION_PATH)
        cls.selected_cfg = pd.read_csv(SELECTED_CONFIG_PATH)
        cls.metrics_fold = pd.read_csv(METRICS_BY_FOLD_PATH)
        cls.metrics_pool = pd.read_csv(METRICS_POOLED_PATH)
        cls.prevalence = pd.read_csv(EVENT_PREVALENCE_PATH)
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        cls.persistence_pred = pd.read_csv(PERSISTENCE_PRED_PATH)
        cls.persistence_prev = pd.read_csv(PERSISTENCE_EVENT_PATH)

        cls.split = pd.read_csv(SPLIT_PATH)
        cls.split["feature_date"] = normalize_date_col(cls.split["feature_date"])
        cls.split["target_date"] = normalize_date_col(cls.split["target_date"])
        cls.split["purged_outer_boundary"] = as_bool(cls.split["purged_outer_boundary"])
        cls.split["purged_inner_boundary"] = as_bool(cls.split["purged_inner_boundary"])

        cls.pred["feature_date"] = normalize_date_col(cls.pred["feature_date"])
        cls.pred["target_date"] = normalize_date_col(cls.pred["target_date"])
        cls.persistence_pred["feature_date"] = normalize_date_col(cls.persistence_pred["feature_date"])
        cls.persistence_pred["target_date"] = normalize_date_col(cls.persistence_pred["target_date"])

        df = pd.read_csv(Path(cls.lock["absolute_path"]))
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)
        df[cls.lock["date_column"]] = normalize_date_col(df[cls.lock["date_column"]])
        df = df.dropna(subset=[cls.lock["date_column"]]).sort_values(cls.lock["date_column"]).reset_index(drop=True)
        cls.dataset = df

        cls.date_to_tnout = {
            d: float(y)
            for d, y in zip(df[cls.lock["date_column"]], df[cls.lock["target_column"]])
        }

        for col in [
            "inner_train_date_start",
            "inner_train_date_end",
            "inner_validation_date_start",
            "inner_validation_date_end",
        ]:
            cls.inner[col] = normalize_date_col(cls.inner[col])

        cls.inner["selected"] = as_bool(cls.inner["selected"])
        cls.inner["outer_test_accessed_during_selection"] = as_bool(
            cls.inner["outer_test_accessed_during_selection"]
        )

    def _mase_denom(self, h: int, fold: int) -> float:
        tr = self.split[
            (self.split["horizon"] == h)
            & (self.split["outer_fold"] == fold)
            & (self.split["outer_role"] == "outer_train")
            & (~self.split["purged_outer_boundary"])
        ].sort_values("feature_date")
        y_train = np.array([self.date_to_tnout[d] for d in tr["target_date"].tolist()], dtype=float)
        if len(y_train) <= 1:
            return np.nan
        return float(np.mean(np.abs(y_train[1:] - y_train[:-1])))

    def test_01_dataset_checksum_matches_lock(self) -> None:
        observed = sha256_file(Path(self.lock["absolute_path"]))
        self.assertEqual(observed, self.lock["sha256"])
        self.assertTrue(bool(self.input_ver["verification_pass"]))
        self.assertEqual(self.input_ver["dataset_sha256_observed"], self.lock["sha256"])

    def test_02_split_checksum_matches_approved_protocol(self) -> None:
        expected_hashes = load_expected_protocol_hashes()
        self.assertIn("corrected_split_assignment.csv", expected_hashes)
        observed = sha256_file(SPLIT_PATH)
        self.assertEqual(observed, expected_hashes["corrected_split_assignment.csv"])
        self.assertEqual(self.input_ver["split_sha256"], observed)

    def test_03_test_dates_exactly_match_locked_canonical_dates(self) -> None:
        exp = self.split[self.split["outer_role"] == "outer_test"].copy()
        got = self.pred[["horizon", "outer_fold", "feature_date"]].copy()

        for h in HORIZONS:
            for fold in FOLDS:
                exp_set = set(
                    exp[(exp["horizon"] == h) & (exp["outer_fold"] == fold)]["feature_date"].tolist()
                )
                got_set = set(
                    got[(got["horizon"] == h) & (got["outer_fold"] == fold)]["feature_date"].tolist()
                )
                self.assertSetEqual(got_set, exp_set, f"Test-date mismatch at H{h} fold{fold}")

    def test_04_prediction_counts_match_split_assignments(self) -> None:
        exp_counts = (
            self.split[self.split["outer_role"] == "outer_test"]
            .groupby(["horizon", "outer_fold"])
            .size()
            .to_dict()
        )
        got_counts = self.pred.groupby(["horizon", "outer_fold"]).size().to_dict()
        self.assertEqual(got_counts, exp_counts)

    def test_05_no_duplicate_prediction_keys(self) -> None:
        dup = self.pred.duplicated(subset=["horizon", "outer_fold", "feature_date"])
        self.assertFalse(bool(dup.any()))

    def test_06_target_dates_match_assignments(self) -> None:
        expected = self.split[self.split["outer_role"] == "outer_test"][
            ["horizon", "outer_fold", "feature_date", "target_date"]
        ].copy()
        merged = self.pred.merge(
            expected,
            on=["horizon", "outer_fold", "feature_date"],
            how="left",
            suffixes=("_pred", "_split"),
        )
        self.assertFalse(merged["target_date_split"].isna().any())
        self.assertTrue(bool((merged["target_date_pred"] == merged["target_date_split"]).all()))

    def test_07_imputation_and_scaling_fitted_on_training_partitions_only(self) -> None:
        self.assertTrue(bool((self.selected_cfg["scaler"] == "StandardScaler").all()))
        self.assertTrue(bool((self.selected_cfg["imputer"] == "none").all()))

        for h in HORIZONS:
            for fold in FOLDS:
                inner_rows = self.inner[(self.inner["horizon"] == h) & (self.inner["outer_fold"] == fold)]
                self.assertGreater(len(inner_rows), 0)

                tr_end = inner_rows["inner_train_date_end"].max()
                val_start = inner_rows["inner_validation_date_start"].min()
                val_end = inner_rows["inner_validation_date_end"].max()

                outer_test_start = self.split[
                    (self.split["horizon"] == h)
                    & (self.split["outer_fold"] == fold)
                    & (self.split["outer_role"] == "outer_test")
                ]["feature_date"].min()

                self.assertLess(tr_end, val_start)
                self.assertLess(val_end, outer_test_start)

    def test_08_outer_test_outcomes_not_accessed_during_selection(self) -> None:
        self.assertTrue(bool((self.inner["outer_test_accessed_during_selection"] == False).all()))

        for h in HORIZONS:
            for fold in FOLDS:
                g = self.inner[(self.inner["horizon"] == h) & (self.inner["outer_fold"] == fold)].copy()
                self.assertEqual(len(g), len(self.selection_plan["candidate_configs"]))

                best_mae = float(g["inner_validation_MAE"].min())
                tied = g[np.abs(g["inner_validation_MAE"] - best_mae) <= 1e-12].copy()
                tied = tied.sort_values(["candidate_alpha", "candidate_l1_ratio"])
                expected = tied.iloc[0]

                selected = g[g["selected"] == True]
                self.assertEqual(len(selected), 1)
                row = selected.iloc[0]

                self.assertAlmostEqual(float(row["candidate_alpha"]), float(expected["candidate_alpha"]), places=12)
                self.assertAlmostEqual(float(row["candidate_l1_ratio"]), float(expected["candidate_l1_ratio"]), places=12)

    def test_09_selected_config_belongs_to_candidate_grid(self) -> None:
        grid = {
            (float(c["alpha"]), float(c["l1_ratio"]))
            for c in self.selection_plan["candidate_configs"]
        }
        sel = {
            (float(a), float(l1))
            for a, l1 in zip(
                self.selected_cfg["selected_alpha"].tolist(),
                self.selected_cfg["selected_l1_ratio"].tolist(),
            )
        }
        self.assertTrue(sel.issubset(grid))

    def test_10_recomputed_metrics_match_saved(self) -> None:
        for h in HORIZONS:
            for fold in FOLDS:
                d = self.pred[(self.pred["horizon"] == h) & (self.pred["outer_fold"] == fold)]
                self.assertFalse(d.empty)

                mae = float(d["absolute_error"].mean())
                mse = float(d["squared_error"].mean())
                rmse = float(np.sqrt(mse))
                denom = self._mase_denom(h, fold)
                mase = float(mae / (denom + 1e-9)) if np.isfinite(denom) else np.nan

                m = self.metrics_fold[
                    (self.metrics_fold["horizon"] == h) & (self.metrics_fold["outer_fold"] == fold)
                ]
                self.assertEqual(len(m), 1)
                row = m.iloc[0]

                self.assertAlmostEqual(mae, float(row["MAE"]), places=12)
                self.assertAlmostEqual(mse, float(row["MSE"]), places=12)
                self.assertAlmostEqual(rmse, float(row["RMSE"]), places=12)
                if np.isfinite(mase):
                    self.assertAlmostEqual(mase, float(row["MASE"]), places=12)

    def test_11_pooled_rmse_equals_sqrt_pooled_mse(self) -> None:
        for h in HORIZONS:
            row = self.metrics_pool[self.metrics_pool["horizon"] == h].iloc[0]
            self.assertAlmostEqual(float(row["RMSE"]), np.sqrt(float(row["MSE"])), places=12)

    def test_12_event_counts_match_y_true_thresholds(self) -> None:
        for h in HORIZONS:
            for fold in FOLDS:
                d = self.pred[(self.pred["horizon"] == h) & (self.pred["outer_fold"] == fold)]
                r = self.prevalence[
                    (self.prevalence["horizon"] == h) & (self.prevalence["outer_fold"] == fold)
                ]
                self.assertEqual(len(r), 1)
                rr = r.iloc[0]

                ev15 = int((d["y_true"] >= 15.0).sum())
                ev16 = int((d["y_true"] >= 16.0).sum())
                ev17 = int((d["y_true"] >= 17.0).sum())

                self.assertEqual(ev15, int(rr["events_tau15"]))
                self.assertEqual(ev16, int(rr["events_tau16"]))
                self.assertEqual(ev17, int(rr["events_tau17"]))

    def test_13_elasticnet_and_persistence_event_counts_match_on_identical_dates(self) -> None:
        enet_dates = self.pred[["horizon", "outer_fold", "feature_date"]].drop_duplicates()
        pers_dates = self.persistence_pred[["horizon", "outer_fold", "feature_date"]].drop_duplicates()
        self.assertEqual(
            set(tuple(x) for x in enet_dates.itertuples(index=False, name=None)),
            set(tuple(x) for x in pers_dates.itertuples(index=False, name=None)),
        )

        ep = self.prevalence[["horizon", "outer_fold", "events_tau15", "events_tau16", "events_tau17"]].copy()
        pp = self.persistence_prev[["horizon", "outer_fold", "events_tau15", "events_tau16", "events_tau17"]].copy()
        merged = ep.merge(pp, on=["horizon", "outer_fold"], suffixes=("_enet", "_pers"))
        self.assertEqual(len(merged), 9)

        for tau in [15, 16, 17]:
            self.assertTrue(bool((merged[f"events_tau{tau}_enet"] == merged[f"events_tau{tau}_pers"]).all()))

    def test_14_repeated_execution_is_deterministic(self) -> None:
        before = {p.name: sha256_file(p) for p in OUTPUT_FILES}
        run_runner()
        after = {p.name: sha256_file(p) for p in OUTPUT_FILES}
        self.assertEqual(before, after)

    def test_15_prediction_checksum_identical_across_repeated_execution(self) -> None:
        before = sha256_file(PREDICTIONS_PATH)
        run_runner()
        after = sha256_file(PREDICTIONS_PATH)
        self.assertEqual(before, after)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCorrectedElasticNet)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    update_manifest_test_result(
        success=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
    )
    sys.exit(0 if result.wasSuccessful() else 1)
