from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"

PRED_PATH = OUT_DIR / "persistence_predictions.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "persistence_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "persistence_metrics_pooled.csv"
RECON_PATH = OUT_DIR / "persistence_metric_reconciliation.csv"
MANIFEST_PATH = OUT_DIR / "persistence_run_manifest.json"
LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"

EXPECTED_PREDICTION_SHA256 = "383014421fa4419e7ba626a9a0e0df61f1b302431bb546c9049ce9f8a4b51754"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TestPersistenceMetricReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pred = pd.read_csv(PRED_PATH)
        cls.metrics_by_fold = pd.read_csv(METRICS_BY_FOLD_PATH)
        cls.metrics_pooled = pd.read_csv(METRICS_POOLED_PATH)
        cls.recon = pd.read_csv(RECON_PATH)
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        cls.pred["feature_date"] = pd.to_datetime(cls.pred["feature_date"]).dt.normalize()
        cls.pred["target_date"] = pd.to_datetime(cls.pred["target_date"]).dt.normalize()

        cls.split = pd.read_csv(SPLIT_PATH)
        cls.split["feature_date"] = pd.to_datetime(cls.split["feature_date"]).dt.normalize()
        cls.split["target_date"] = pd.to_datetime(cls.split["target_date"]).dt.normalize()

        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        dset = pd.read_csv(lock["absolute_path"])
        unnamed = [c for c in dset.columns if str(c).startswith("Unnamed")]
        if unnamed:
            dset = dset.drop(columns=unnamed)

        dset[lock["date_column"]] = pd.to_datetime(dset[lock["date_column"]]).dt.normalize()
        cls.date_to_tnout = {
            d: float(y)
            for d, y in zip(dset[lock["date_column"]], dset[lock["target_column"]])
        }

    def test_01_pooled_mse_equals_mean_squared_error(self) -> None:
        for h in [1, 3, 5]:
            d = self.pred[self.pred["horizon"] == h]
            mse = float(d["squared_error"].mean())
            saved = float(self.metrics_pooled[self.metrics_pooled["horizon"] == h]["MSE"].iloc[0])
            self.assertAlmostEqual(mse, saved, places=12)

    def test_02_pooled_rmse_equals_sqrt_pooled_mse(self) -> None:
        for h in [1, 3, 5]:
            row = self.metrics_pooled[self.metrics_pooled["horizon"] == h].iloc[0]
            pooled_mse = float(row["MSE"])
            pooled_rmse = float(row["RMSE"])
            self.assertAlmostEqual(pooled_rmse, math.sqrt(pooled_mse), places=12)

    def test_03_pooled_rmse_not_replaced_by_mean_fold_rmse(self) -> None:
        for h in [1, 3, 5]:
            d = self.pred[self.pred["horizon"] == h]
            fold_rmse = (
                d.groupby("outer_fold")["squared_error"]
                .mean()
                .map(float)
                .map(math.sqrt)
                .mean()
            )
            pooled_rmse = float(self.metrics_pooled[self.metrics_pooled["horizon"] == h]["RMSE"].iloc[0])
            self.assertGreater(abs(pooled_rmse - float(fold_rmse)), 1e-6)

    def test_04_reconciliation_aggregation_columns_match_formulas(self) -> None:
        for h in [1, 3, 5]:
            d = self.pred[self.pred["horizon"] == h]
            fold = d.groupby("outer_fold").agg(
                n=("outer_fold", "size"),
                mae=("absolute_error", "mean"),
                mse=("squared_error", "mean"),
            )
            fold["rmse"] = np.sqrt(fold["mse"])

            expected = {
                "MAE": {
                    "pooled": float(d["absolute_error"].mean()),
                    "uw": float(fold["mae"].mean()),
                    "w": float(np.average(fold["mae"], weights=fold["n"])),
                },
                "MSE": {
                    "pooled": float(d["squared_error"].mean()),
                    "uw": float(fold["mse"].mean()),
                    "w": float(np.average(fold["mse"], weights=fold["n"])),
                },
                "RMSE": {
                    "pooled": float(math.sqrt(float(d["squared_error"].mean()))),
                    "uw": float(fold["rmse"].mean()),
                    "w": float(np.average(fold["rmse"], weights=fold["n"])),
                },
            }

            # MASE values come from saved metrics files.
            b = self.metrics_by_fold[self.metrics_by_fold["horizon"] == h]
            p = self.metrics_pooled[self.metrics_pooled["horizon"] == h].iloc[0]
            expected["MASE"] = {
                "pooled": float(p["MASE"]),
                "uw": float(b["MASE"].mean()),
                "w": float(np.average(b["MASE"], weights=b["N"])),
            }

            sqrt_uw_mse = float(math.sqrt(expected["MSE"]["uw"]))
            sqrt_w_mse = float(math.sqrt(expected["MSE"]["w"]))

            r = self.recon[self.recon["horizon"] == h]
            self.assertEqual(set(r["metric"].tolist()), {"MAE", "MSE", "RMSE", "MASE"})

            for metric in ["MAE", "MSE", "RMSE", "MASE"]:
                row = r[r["metric"] == metric].iloc[0]
                self.assertAlmostEqual(float(row["corrected_pooled_value"]), expected[metric]["pooled"], places=12)
                self.assertAlmostEqual(float(row["unweighted_fold_mean"]), expected[metric]["uw"], places=12)
                self.assertAlmostEqual(float(row["weighted_fold_mean"]), expected[metric]["w"], places=12)
                self.assertAlmostEqual(float(row["sqrt_unweighted_fold_mse"]), sqrt_uw_mse, places=12)
                self.assertAlmostEqual(float(row["sqrt_weighted_fold_mse"]), sqrt_w_mse, places=12)

    def test_05_mase_uses_training_only_denominators(self) -> None:
        for h in [1, 3, 5]:
            for fold in [1, 2, 3]:
                tr = self.split[
                    (self.split["horizon"] == h)
                    & (self.split["outer_fold"] == fold)
                    & (self.split["outer_role"] == "outer_train")
                    & (~self.split["purged_outer_boundary"].astype(bool))
                ].sort_values("feature_date")

                y_train = np.array([self.date_to_tnout[d] for d in tr["target_date"]], dtype=float)
                denom = float(np.mean(np.abs(y_train[1:] - y_train[:-1])))

                saved = self.metrics_by_fold[
                    (self.metrics_by_fold["horizon"] == h)
                    & (self.metrics_by_fold["outer_fold"] == fold)
                ]["MASE_denom_train_only"].iloc[0]
                self.assertAlmostEqual(float(saved), denom, places=12)

    def test_06_recommended_values_match_pooled_canonical_predictions(self) -> None:
        for h in [1, 3, 5]:
            d = self.pred[self.pred["horizon"] == h]
            pooled = {
                "MAE": float(d["absolute_error"].mean()),
                "MSE": float(d["squared_error"].mean()),
                "RMSE": float(math.sqrt(float(d["squared_error"].mean()))),
                "MASE": float(self.metrics_pooled[self.metrics_pooled["horizon"] == h]["MASE"].iloc[0]),
            }
            r = self.recon[self.recon["horizon"] == h]
            for metric, expected in pooled.items():
                got = float(r[r["metric"] == metric]["recommended_revision_value"].iloc[0])
                self.assertAlmostEqual(got, expected, places=12)

    def test_07_prediction_rows_not_modified(self) -> None:
        observed = sha256_file(PRED_PATH)
        self.assertEqual(observed, EXPECTED_PREDICTION_SHA256)
        self.assertEqual(observed, self.manifest["prediction_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
