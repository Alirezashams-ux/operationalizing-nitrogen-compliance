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

RUN_SCRIPT = OUT_DIR / "run_corrected_persistence.py"
LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
PREDICTIONS_PATH = OUT_DIR / "persistence_predictions.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "persistence_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "persistence_metrics_pooled.csv"
EVENT_PREVALENCE_PATH = OUT_DIR / "persistence_event_prevalence.csv"
MANIFEST_PATH = OUT_DIR / "persistence_run_manifest.json"

HORIZONS = (1, 3, 5)
FOLDS = (1, 2, 3)


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


class TestCorrectedPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_runner()

        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        cls.input_ver = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        cls.pred = pd.read_csv(PREDICTIONS_PATH)
        cls.metrics_fold = pd.read_csv(METRICS_BY_FOLD_PATH)
        cls.metrics_pool = pd.read_csv(METRICS_POOLED_PATH)
        cls.prevalence = pd.read_csv(EVENT_PREVALENCE_PATH)
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        cls.split = pd.read_csv(SPLIT_PATH)
        cls.split["feature_date"] = normalize_date_col(cls.split["feature_date"])
        cls.split["target_date"] = normalize_date_col(cls.split["target_date"])

        cls.pred["feature_date"] = normalize_date_col(cls.pred["feature_date"])
        cls.pred["target_date"] = normalize_date_col(cls.pred["target_date"])

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

    def test_03_test_dates_equal_canonical_locked_test_dates(self) -> None:
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

    def test_04_counts_match_corrected_split_assignments(self) -> None:
        exp_counts = (
            self.split[self.split["outer_role"] == "outer_test"]
            .groupby(["horizon", "outer_fold"])
            .size()
            .to_dict()
        )
        got_counts = self.pred.groupby(["horizon", "outer_fold"]).size().to_dict()
        self.assertEqual(got_counts, exp_counts)

    def test_05_no_duplicate_horizon_fold_feature_date(self) -> None:
        dup = self.pred.duplicated(subset=["horizon", "outer_fold", "feature_date"])
        self.assertFalse(bool(dup.any()))

    def test_06_target_dates_equal_assigned_target_dates(self) -> None:
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

    def test_07_predictions_use_feature_date_tnout_only(self) -> None:
        expected = self.pred["feature_date"].map(lambda d: self.date_to_tnout[d]).astype(float)
        self.assertTrue(np.allclose(self.pred["y_pred"].to_numpy(dtype=float), expected.to_numpy(dtype=float)))

    def test_08_no_y_true_used_to_form_y_pred(self) -> None:
        # Constructed prediction must be feature-date TNout and should not collapse
        # into direct copy of y_true across all rows.
        self.assertTrue(bool((self.pred["y_pred"] == self.pred["feature_date"].map(lambda d: self.date_to_tnout[d])).all()))
        self.assertFalse(bool((self.pred["y_pred"] == self.pred["y_true"]).all()))

    def _mase_denom(self, h: int, fold: int) -> float:
        tr = self.split[
            (self.split["horizon"] == h)
            & (self.split["outer_fold"] == fold)
            & (self.split["outer_role"] == "outer_train")
            & (~self.split["purged_outer_boundary"].astype(bool))
        ].sort_values("feature_date")
        y_train = np.array([self.date_to_tnout[d] for d in tr["target_date"].tolist()], dtype=float)
        if len(y_train) <= 1:
            return np.nan
        return float(np.mean(np.abs(y_train[1:] - y_train[:-1])))

    def test_09_recomputed_point_metrics_match_saved(self) -> None:
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

    def test_10_pooled_metrics_from_pooled_predictions_directly(self) -> None:
        for h in HORIZONS:
            d = self.pred[self.pred["horizon"] == h].copy()
            mae = float(d["absolute_error"].mean())
            mse = float(d["squared_error"].mean())
            rmse = float(np.sqrt(mse))

            denoms = d["outer_fold"].map(lambda f: self._mase_denom(h, int(f))).astype(float)
            mase = float(np.mean(d["absolute_error"].to_numpy(dtype=float) / (denoms.to_numpy(dtype=float) + 1e-9)))

            r = self.metrics_pool[self.metrics_pool["horizon"] == h]
            self.assertEqual(len(r), 1)
            row = r.iloc[0]
            self.assertAlmostEqual(mae, float(row["MAE"]), places=12)
            self.assertAlmostEqual(mse, float(row["MSE"]), places=12)
            self.assertAlmostEqual(rmse, float(row["RMSE"]), places=12)
            self.assertAlmostEqual(mase, float(row["MASE"]), places=12)

    def test_11_event_counts_match_threshold_logic(self) -> None:
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
                self.assertAlmostEqual(ev15 / len(d), float(rr["prevalence_tau15"]), places=12)
                self.assertAlmostEqual(ev16 / len(d), float(rr["prevalence_tau16"]), places=12)
                self.assertAlmostEqual(ev17 / len(d), float(rr["prevalence_tau17"]), places=12)

    def test_12_repeated_execution_deterministic_checksum_identical(self) -> None:
        tracked_files = [
            PREDICTIONS_PATH,
            METRICS_BY_FOLD_PATH,
            METRICS_POOLED_PATH,
            EVENT_PREVALENCE_PATH,
        ]
        before = {p.name: sha256_file(p) for p in tracked_files}

        run_runner()

        after = {p.name: sha256_file(p) for p in tracked_files}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
