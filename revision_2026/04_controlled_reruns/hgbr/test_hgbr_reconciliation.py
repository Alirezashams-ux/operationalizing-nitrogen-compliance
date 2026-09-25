from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = OUT_DIR / "hgbr_predictions.csv"
METRICS_POOLED_PATH = OUT_DIR / "hgbr_metrics_pooled.csv"
MANIFEST_PATH = OUT_DIR / "hgbr_run_manifest.json"
ISSUE_REGISTRY_PATH = OUT_DIR / "hgbr_decision_b_issue_registry.csv"
RECON_PATH = OUT_DIR / "hgbr_metric_reconciliation.csv"
SELECTED_CONFIG_PATH = OUT_DIR / "hgbr_selected_configurations.csv"
SPLIT_PATH = ROOT / "revision_2026" / "03_corrected_protocol" / "corrected_split_assignment.csv"

SUBMITTED_PREDICTION_PATHS = {
    "H3_ordinary": ROOT / "results" / "predictions" / "hgbr_optuna_H3_preds.csv",
    "H5_ordinary": ROOT / "results" / "predictions" / "hgbr_optuna_H5_preds.csv",
    "H5_v2": ROOT / "results" / "predictions" / "hgbr_optuna_H5_v2_preds.csv",
}

EXPECTED_PREDICTION_SHA256 = "f5b4689128aa9cab965bbf8eac3879d05050b32a0c62572c7b3365f7bdc3ba36"
EXPECTED_SELECTED_CONFIG_SHA256 = "32702856123e29efd7c214f83a1d188643eabc81a177096b0f92287e18ef74aa"
EXPECTED_COUNTS = {1: 762, 3: 747, 5: 747}
EXPECTED_SELECTED_IDS = {
    (1, 1): "h1_cfg_014",
    (1, 2): "h1_cfg_010",
    (1, 3): "h1_cfg_007",
    (3, 1): "h3_cfg_010",
    (3, 2): "h3_cfg_007",
    (3, 3): "h3_cfg_004",
    (5, 1): "h5_cfg_006",
    (5, 2): "h5_cfg_003",
    (5, 3): "h5_cfg_012",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_aggregation_variants(df: pd.DataFrame, fold_col: str) -> Dict[str, float]:
    errors = df["y_pred"].to_numpy(dtype=float) - df["y_true"].to_numpy(dtype=float)
    pooled_mae = float(np.mean(np.abs(errors)))
    pooled_mse = float(np.mean(np.square(errors)))
    pooled_rmse = float(np.sqrt(pooled_mse))

    grp = df.groupby(fold_col)
    fold_n = grp.size().astype(float)
    fold_mae = grp.apply(lambda g: float(np.mean(np.abs(g["y_pred"] - g["y_true"]))))
    fold_mse = grp.apply(lambda g: float(np.mean(np.square(g["y_pred"] - g["y_true"]))))
    fold_rmse = np.sqrt(fold_mse)

    return {
        "pooled_MAE": pooled_mae,
        "pooled_MSE": pooled_mse,
        "pooled_RMSE": pooled_rmse,
        "unweighted_mean_fold_MAE": float(fold_mae.mean()),
        "weighted_mean_fold_MAE": float(np.average(fold_mae.values, weights=fold_n.values)),
        "unweighted_mean_fold_RMSE": float(fold_rmse.mean()),
        "weighted_mean_fold_RMSE": float(np.average(fold_rmse.values, weights=fold_n.values)),
        "sqrt_unweighted_mean_fold_MSE": float(np.sqrt(fold_mse.mean())),
        "sqrt_weighted_mean_fold_MSE": float(np.sqrt(np.average(fold_mse.values, weights=fold_n.values))),
        "unweighted_mean_fold_MSE": float(fold_mse.mean()),
        "weighted_mean_fold_MSE": float(np.average(fold_mse.values, weights=fold_n.values)),
    }


def parse_effect_metric(effect_json: str, metric: str) -> float:
    payload = json.loads(effect_json)
    value = payload.get(metric)
    if value is None:
        return 0.0
    return float(value)


class TestHGBRReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pred = pd.read_csv(PREDICTIONS_PATH)
        cls.metrics = pd.read_csv(METRICS_POOLED_PATH)
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.issue_registry = pd.read_csv(ISSUE_REGISTRY_PATH)
        cls.recon = pd.read_csv(RECON_PATH)
        cls.selected = pd.read_csv(SELECTED_CONFIG_PATH)
        cls.split = pd.read_csv(SPLIT_PATH)

    def test_01_corrected_prediction_file_is_unchanged(self) -> None:
        observed = sha256_file(PREDICTIONS_PATH)
        self.assertEqual(observed, EXPECTED_PREDICTION_SHA256)

    def test_02_corrected_prediction_checksum_matches_manifest(self) -> None:
        observed = sha256_file(PREDICTIONS_PATH)
        self.assertEqual(observed, self.manifest["prediction_sha256"])

    def test_03_corrected_pooled_metrics_reproduce_saved_values(self) -> None:
        for h in [1, 3, 5]:
            d = self.pred[self.pred["horizon"] == h].copy()
            e = d["y_pred"].to_numpy(dtype=float) - d["y_true"].to_numpy(dtype=float)
            mae = float(np.mean(np.abs(e)))
            mse = float(np.mean(np.square(e)))
            rmse = float(np.sqrt(mse))

            row = self.metrics[self.metrics["horizon"] == h].iloc[0]
            self.assertAlmostEqual(mae, float(row["MAE"]), places=12)
            self.assertAlmostEqual(mse, float(row["MSE"]), places=12)
            self.assertAlmostEqual(rmse, float(row["RMSE"]), places=12)

    def test_04_pooled_rmse_equals_sqrt_pooled_mse(self) -> None:
        for h in [1, 3, 5]:
            row = self.metrics[self.metrics["horizon"] == h].iloc[0]
            self.assertAlmostEqual(float(row["RMSE"]), math.sqrt(float(row["MSE"])), places=12)

    def test_05_all_aggregation_variants_reproduce_formulas(self) -> None:
        split = self.split.copy()
        split["feature_date"] = pd.to_datetime(split["feature_date"]).dt.normalize()
        canonical_dates = {
            h: set(split[(split["horizon"] == h) & (split["outer_role"] == "outer_test")]["feature_date"])
            for h in [3, 5]
        }

        for name, path in SUBMITTED_PREDICTION_PATHS.items():
            d = pd.read_csv(path)
            variants = compute_aggregation_variants(d, "fold")

            # Formula checks for native submitted rows.
            self.assertAlmostEqual(
                variants["pooled_MSE"],
                variants["weighted_mean_fold_MSE"],
                places=12,
            )
            self.assertAlmostEqual(
                variants["pooled_RMSE"],
                math.sqrt(variants["pooled_MSE"]),
                places=12,
            )
            self.assertAlmostEqual(
                variants["sqrt_unweighted_mean_fold_MSE"],
                math.sqrt(variants["unweighted_mean_fold_MSE"]),
                places=12,
            )
            self.assertAlmostEqual(
                variants["sqrt_weighted_mean_fold_MSE"],
                math.sqrt(variants["weighted_mean_fold_MSE"]),
                places=12,
            )

            # Repeat formula checks on canonical-date restriction where applicable.
            h = 3 if "H3" in name else 5
            d["feature_date"] = pd.to_datetime(d["date"]).dt.normalize()
            d_c = d[d["feature_date"].isin(canonical_dates[h])].copy()
            v_c = compute_aggregation_variants(d_c, "fold")
            self.assertAlmostEqual(v_c["pooled_MSE"], v_c["weighted_mean_fold_MSE"], places=12)
            self.assertAlmostEqual(v_c["pooled_RMSE"], math.sqrt(v_c["pooled_MSE"]), places=12)

    def test_06_canonical_prediction_counts_match_expected(self) -> None:
        counts = self.pred.groupby("horizon").size().to_dict()
        self.assertEqual({int(k): int(v) for k, v in counts.items()}, EXPECTED_COUNTS)

    def test_07_no_outer_test_metric_used_to_alter_selected_configurations(self) -> None:
        observed_hash = sha256_file(SELECTED_CONFIG_PATH)
        self.assertEqual(observed_hash, EXPECTED_SELECTED_CONFIG_SHA256)
        self.assertEqual(observed_hash, self.manifest["metrics_sha256"]["hgbr_selected_configurations.csv"])

        selected_map = {
            (int(r.horizon), int(r.outer_fold)): str(r.selected_configuration_id)
            for r in self.selected.itertuples(index=False)
        }
        self.assertEqual(selected_map, EXPECTED_SELECTED_IDS)

    def test_08_recommended_revised_metrics_equal_canonical_pooled_metrics(self) -> None:
        metric_map = {
            int(r.horizon): {
                "N": int(r.N),
                "MAE": float(r.MAE),
                "MSE": float(r.MSE),
                "RMSE": float(r.RMSE),
            }
            for r in self.metrics.itertuples(index=False)
        }

        for row in self.recon.itertuples(index=False):
            rec = json.loads(row.recommended_revision_value)
            h = int(str(row.horizon).replace("H", ""))
            self.assertEqual(int(rec["N"]), metric_map[h]["N"])
            self.assertAlmostEqual(float(rec["MAE"]), metric_map[h]["MAE"], places=12)
            self.assertAlmostEqual(float(rec["MSE"]), metric_map[h]["MSE"], places=12)
            self.assertAlmostEqual(float(rec["RMSE"]), metric_map[h]["RMSE"], places=12)

    def test_09_issue_registry_has_no_unresolved_validity_critical_issue(self) -> None:
        resolved = self.issue_registry["resolved"].astype(str).str.strip().str.upper()
        self.assertTrue(bool((resolved == "TRUE").all()))

        flags = self.issue_registry["effect_on_validity"].astype(str).str.lower().tolist()
        bad = [f for f in flags if ("validity_critical" in f and "not" not in f and "no_" not in f)]
        self.assertEqual(bad, [])

    def test_10_numeric_decompositions_are_internally_consistent(self) -> None:
        for row in self.recon.itertuples(index=False):
            submitted_mae = pd.to_numeric(row.submitted_MAE, errors="coerce")
            submitted_rmse = pd.to_numeric(row.submitted_RMSE, errors="coerce")
            corrected_mae = pd.to_numeric(row.corrected_MAE, errors="coerce")
            corrected_rmse = pd.to_numeric(row.corrected_RMSE, errors="coerce")

            if np.isnan(submitted_mae) or np.isnan(submitted_rmse):
                continue

            mae_effect_sum = 0.0
            rmse_effect_sum = 0.0
            for col in [
                "aggregation_effect",
                "date_set_effect",
                "purge_effect",
                "nested_selection_effect",
                "lineage_effect",
                "unresolved_effect",
            ]:
                mae_effect_sum += parse_effect_metric(getattr(row, col), "MAE")
                rmse_effect_sum += parse_effect_metric(getattr(row, col), "RMSE")

            self.assertAlmostEqual(float(corrected_mae - submitted_mae), mae_effect_sum, places=12)
            self.assertAlmostEqual(float(corrected_rmse - submitted_rmse), rmse_effect_sum, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
