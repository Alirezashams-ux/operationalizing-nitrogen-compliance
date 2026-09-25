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
BASELINE_DIR = ROOT / "revision_2026" / "04_controlled_reruns"

RUN_SCRIPT = OUT_DIR / "run_corrected_hgbr.py"
LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"
SELECTION_PLAN_PATH = OUT_DIR / "hgbr_selection_plan.json"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
PREDICTIONS_PATH = OUT_DIR / "hgbr_predictions.csv"
INNER_SELECTION_PATH = OUT_DIR / "hgbr_inner_selection_results.csv"
SELECTED_CONFIG_PATH = OUT_DIR / "hgbr_selected_configurations.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "hgbr_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "hgbr_metrics_pooled.csv"
EVENT_PREVALENCE_PATH = OUT_DIR / "hgbr_event_prevalence.csv"
MANIFEST_PATH = OUT_DIR / "hgbr_run_manifest.json"

PERSISTENCE_PRED_PATH = BASELINE_DIR / "persistence" / "persistence_predictions.csv"
RIDGE_PRED_PATH = BASELINE_DIR / "ridge" / "ridge_predictions.csv"
ELASTICNET_PRED_PATH = BASELINE_DIR / "elasticnet" / "elasticnet_predictions.csv"

PERSISTENCE_EVENT_PATH = BASELINE_DIR / "persistence" / "persistence_event_prevalence.csv"
RIDGE_EVENT_PATH = BASELINE_DIR / "ridge" / "ridge_event_prevalence.csv"
ELASTICNET_EVENT_PATH = BASELINE_DIR / "elasticnet" / "elasticnet_event_prevalence.csv"

EXPECTED_COUNTS = {1: 762, 3: 747, 5: 747}
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
        hashes[parts[-1]] = parts[0]
    return hashes


def update_manifest_test_result(success: bool, tests_run: int, failures: int, errors: int) -> None:
    if not MANIFEST_PATH.exists():
        return
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["test_result"] = "pass_20_of_20" if success else f"fail_{failures + errors}_of_20"
    payload["test_command"] = f"{sys.executable} {Path(__file__).resolve()}"
    payload["test_summary"] = f"Ran {tests_run} tests; failures={failures}; errors={errors}."
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestCorrectedHGBR(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_runner()

        cls.selection_plan = json.loads(SELECTION_PLAN_PATH.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        cls.input_ver = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        cls.pred = pd.read_csv(PREDICTIONS_PATH)
        cls.inner = pd.read_csv(INNER_SELECTION_PATH)
        cls.selected_cfg = pd.read_csv(SELECTED_CONFIG_PATH)
        cls.metrics_fold = pd.read_csv(METRICS_BY_FOLD_PATH)
        cls.metrics_pool = pd.read_csv(METRICS_POOLED_PATH)
        cls.prevalence = pd.read_csv(EVENT_PREVALENCE_PATH)
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        cls.split = pd.read_csv(SPLIT_PATH)
        cls.split["feature_date"] = normalize_date_col(cls.split["feature_date"])
        cls.split["target_date"] = normalize_date_col(cls.split["target_date"])
        cls.split["purged_outer_boundary"] = as_bool(cls.split["purged_outer_boundary"])
        cls.split["purged_inner_boundary"] = as_bool(cls.split["purged_inner_boundary"])

        cls.pred["feature_date"] = normalize_date_col(cls.pred["feature_date"])
        cls.pred["target_date"] = normalize_date_col(cls.pred["target_date"])

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

        cls.selected_cfg["early_stopping"] = as_bool(cls.selected_cfg["early_stopping"])

        authorized = cls.selection_plan.get("authorized_horizons", [1, 3, 5])
        cls.horizons = tuple(sorted(int(h) for h in authorized))

        df = pd.read_csv(Path(cls.lock["absolute_path"]))
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)
        df[cls.lock["date_column"]] = normalize_date_col(df[cls.lock["date_column"]])
        df = df.dropna(subset=[cls.lock["date_column"]]).sort_values(cls.lock["date_column"]).reset_index(drop=True)
        cls.dataset = df

        cls.date_to_tnout = {
            d: float(y)
            for d, y in zip(df[cls.lock["date_column"]], pd.to_numeric(df[cls.lock["target_column"]], errors="raise"))
        }

        cls.persistence_pred = pd.read_csv(PERSISTENCE_PRED_PATH)
        cls.ridge_pred = pd.read_csv(RIDGE_PRED_PATH)
        cls.elasticnet_pred = pd.read_csv(ELASTICNET_PRED_PATH)
        for df_pred in [cls.persistence_pred, cls.ridge_pred, cls.elasticnet_pred]:
            df_pred["feature_date"] = normalize_date_col(df_pred["feature_date"])
            df_pred["target_date"] = normalize_date_col(df_pred["target_date"])

        cls.persistence_prev = pd.read_csv(PERSISTENCE_EVENT_PATH)
        cls.ridge_prev = pd.read_csv(RIDGE_EVENT_PATH)
        cls.elasticnet_prev = pd.read_csv(ELASTICNET_EVENT_PATH)

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

        for h in self.horizons:
            for fold in FOLDS:
                exp_set = set(
                    exp[(exp["horizon"] == h) & (exp["outer_fold"] == fold)]["feature_date"].tolist()
                )
                got_set = set(
                    got[(got["horizon"] == h) & (got["outer_fold"] == fold)]["feature_date"].tolist()
                )
                self.assertSetEqual(got_set, exp_set, f"Test-date mismatch at H{h} fold{fold}")

    def test_04_prediction_counts_equal_expected_horizon_totals(self) -> None:
        got_total = self.pred.groupby("horizon").size().to_dict()
        for h in self.horizons:
            self.assertEqual(int(got_total.get(h, -1)), EXPECTED_COUNTS[h])

        exp_counts_by_fold = (
            self.split[self.split["outer_role"] == "outer_test"]
            .groupby(["horizon", "outer_fold"])
            .size()
            .to_dict()
        )
        got_counts_by_fold = self.pred.groupby(["horizon", "outer_fold"]).size().to_dict()
        self.assertEqual(got_counts_by_fold, exp_counts_by_fold)

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

    def test_07_preprocessing_fitted_only_within_authorized_training_partitions(self) -> None:
        self.assertTrue(bool((self.selected_cfg["early_stopping"] == True).all()))

        for h in self.horizons:
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

    def test_08_candidate_evaluation_uses_only_inner_validation(self) -> None:
        for h in self.horizons:
            for fold in FOLDS:
                g = self.inner[(self.inner["horizon"] == h) & (self.inner["outer_fold"] == fold)]
                self.assertGreater(len(g), 0)

                expected_val_n = int(
                    self.split[
                        (self.split["horizon"] == h)
                        & (self.split["outer_fold"] == fold)
                        & (self.split["outer_role"] == "outer_train")
                        & (self.split["inner_role"] == "inner_validation")
                    ].shape[0]
                )
                expected_train_n = int(
                    self.split[
                        (self.split["horizon"] == h)
                        & (self.split["outer_fold"] == fold)
                        & (self.split["outer_role"] == "outer_train")
                        & (self.split["inner_role"] == "inner_train")
                        & (~self.split["purged_inner_boundary"])
                    ].shape[0]
                )

                self.assertTrue(bool((g["inner_validation_N"] == expected_val_n).all()))
                self.assertTrue(bool((g["inner_train_N"] == expected_train_n).all()))

    def test_09_outer_test_outcomes_not_accessed_during_selection(self) -> None:
        self.assertTrue(bool((self.inner["outer_test_accessed_during_selection"] == False).all()))

    def test_10_selected_configs_belong_to_predeclared_candidate_space(self) -> None:
        budget_by_h = {int(k): int(v) for k, v in self.selection_plan["search_budget"].items()}

        for h in self.horizons:
            expected_budget = budget_by_h[h]
            all_ids = set(self.inner[self.inner["horizon"] == h]["candidate_configuration_id"].tolist())
            self.assertEqual(len(all_ids), expected_budget)

            sel_ids = set(self.selected_cfg[self.selected_cfg["horizon"] == h]["selected_configuration_id"].tolist())
            self.assertTrue(sel_ids.issubset(all_ids))

    def test_11_selected_configs_reproduce_deterministic_selection_rule(self) -> None:
        for h in self.horizons:
            for fold in FOLDS:
                g = self.inner[(self.inner["horizon"] == h) & (self.inner["outer_fold"] == fold)].copy()
                self.assertGreater(len(g), 0)

                best_mae = float(g["inner_validation_MAE"].min())
                tied = g[np.abs(g["inner_validation_MAE"] - best_mae) <= 1e-12].copy()
                expected_id = sorted(tied["candidate_configuration_id"].tolist())[0]

                selected = g[g["selected"] == True]
                self.assertEqual(len(selected), 1)
                selected_id = str(selected.iloc[0]["candidate_configuration_id"])
                self.assertEqual(selected_id, expected_id)

    def test_12_seeds_are_recorded_and_fixed(self) -> None:
        self.assertTrue(bool((self.pred["random_seed"] == 42).all()))
        self.assertTrue(bool((self.selected_cfg["random_seed"] == 42).all()))

        for s in self.selected_cfg["selected_parameters_json"].tolist():
            params = json.loads(s)
            self.assertIn("random_state", params)
            self.assertEqual(int(params["random_state"]), 42)

    def test_13_pooled_metrics_match_recomputation(self) -> None:
        for h in self.horizons:
            d = self.pred[self.pred["horizon"] == h].copy()
            self.assertFalse(d.empty)

            mae = float(d["absolute_error"].mean())
            mse = float(d["squared_error"].mean())
            rmse = float(np.sqrt(mse))

            denoms = d["outer_fold"].map(lambda f: self._mase_denom(h, int(f))).astype(float)
            mase = float(np.mean(d["absolute_error"].to_numpy(dtype=float) / (denoms.to_numpy(dtype=float) + 1e-9)))

            row = self.metrics_pool[self.metrics_pool["horizon"] == h]
            self.assertEqual(len(row), 1)
            r = row.iloc[0]

            self.assertAlmostEqual(mae, float(r["MAE"]), places=12)
            self.assertAlmostEqual(mse, float(r["MSE"]), places=12)
            self.assertAlmostEqual(rmse, float(r["RMSE"]), places=12)
            self.assertAlmostEqual(mase, float(r["MASE"]), places=12)

    def test_14_pooled_rmse_equals_sqrt_pooled_mse(self) -> None:
        for h in self.horizons:
            row = self.metrics_pool[self.metrics_pool["horizon"] == h].iloc[0]
            self.assertAlmostEqual(float(row["RMSE"]), np.sqrt(float(row["MSE"])), places=12)

    def test_15_mase_uses_training_only_denominators(self) -> None:
        for h in self.horizons:
            for fold in FOLDS:
                d = self.pred[(self.pred["horizon"] == h) & (self.pred["outer_fold"] == fold)]
                self.assertFalse(d.empty)

                mae = float(d["absolute_error"].mean())
                denom = self._mase_denom(h, fold)
                expected_mase = float(mae / (denom + 1e-9))

                m = self.metrics_fold[
                    (self.metrics_fold["horizon"] == h) & (self.metrics_fold["outer_fold"] == fold)
                ]
                self.assertEqual(len(m), 1)
                self.assertAlmostEqual(expected_mase, float(m.iloc[0]["MASE"]), places=12)

    def test_16_event_counts_match_y_true_thresholds(self) -> None:
        for h in self.horizons:
            for fold in FOLDS:
                d = self.pred[(self.pred["horizon"] == h) & (self.pred["outer_fold"] == fold)]
                r = self.prevalence[
                    (self.prevalence["horizon"] == h) & (self.prevalence["outer_fold"] == fold)
                ]
                self.assertEqual(len(r), 1)
                rr = r.iloc[0]

                self.assertEqual(int((d["y_true"] >= 15.0).sum()), int(rr["events_tau15"]))
                self.assertEqual(int((d["y_true"] >= 16.0).sum()), int(rr["events_tau16"]))
                self.assertEqual(int((d["y_true"] >= 17.0).sum()), int(rr["events_tau17"]))

    def test_17_event_counts_equal_persistence_ridge_elasticnet_on_identical_dates(self) -> None:
        hg_dates = self.pred[["horizon", "outer_fold", "feature_date"]].drop_duplicates()

        for name, other_pred in [
            ("Persistence", self.persistence_pred),
            ("Ridge", self.ridge_pred),
            ("ElasticNet", self.elasticnet_pred),
        ]:
            other_dates = other_pred[["horizon", "outer_fold", "feature_date"]].drop_duplicates()
            self.assertEqual(
                set(tuple(x) for x in hg_dates.itertuples(index=False, name=None)),
                set(tuple(x) for x in other_dates.itertuples(index=False, name=None)),
                f"Date-set mismatch against {name}",
            )

        hp = self.prevalence[["horizon", "outer_fold", "events_tau15", "events_tau16", "events_tau17"]].copy()
        for name, other_prev in [
            ("Persistence", self.persistence_prev),
            ("Ridge", self.ridge_prev),
            ("ElasticNet", self.elasticnet_prev),
        ]:
            op = other_prev[["horizon", "outer_fold", "events_tau15", "events_tau16", "events_tau17"]].copy()
            merged = hp.merge(op, on=["horizon", "outer_fold"], suffixes=("_hg", "_other"))
            self.assertEqual(len(merged), 9)
            for tau in [15, 16, 17]:
                self.assertTrue(
                    bool((merged[f"events_tau{tau}_hg"] == merged[f"events_tau{tau}_other"]).all()),
                    f"Event-count mismatch for tau={tau} against {name}",
                )

    def test_18_repeated_execution_is_deterministic(self) -> None:
        before = {p.name: sha256_file(p) for p in OUTPUT_FILES}
        run_runner()
        after = {p.name: sha256_file(p) for p in OUTPUT_FILES}
        self.assertEqual(before, after)

    def test_19_prediction_checksum_stable_across_repeated_execution(self) -> None:
        before = sha256_file(PREDICTIONS_PATH)
        run_runner()
        after = sha256_file(PREDICTIONS_PATH)
        self.assertEqual(before, after)

    def test_20_no_legacy_outer_test_or_submitted_best_consumed_by_runner(self) -> None:
        source = RUN_SCRIPT.read_text(encoding="utf-8")
        forbidden_literals = [
            "results/predictions/hgbr_optuna_H3_preds.csv",
            "results/predictions/hgbr_optuna_H5_preds.csv",
            "results/predictions/hgbr_optuna_H5_v2_preds.csv",
            "results/hgbr/hgbr_optuna_H3.json",
            "results/hgbr/hgbr_optuna_H5.json",
            "results/hgbr/hgbr_optuna_H5_v2.json",
            "alarm_budget_compare_H3.csv",
            "alarm_budget_compare_H5.csv",
            "alarm_budget_compare_v2_H5.csv",
        ]
        for lit in forbidden_literals:
            self.assertNotIn(lit, source)

        self.assertTrue(
            bool((self.selected_cfg["selection_source"] == "corrected_split_assignment_inner_validation").all())
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCorrectedHGBR)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    update_manifest_test_result(
        success=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
    )
    sys.exit(0 if result.wasSuccessful() else 1)
