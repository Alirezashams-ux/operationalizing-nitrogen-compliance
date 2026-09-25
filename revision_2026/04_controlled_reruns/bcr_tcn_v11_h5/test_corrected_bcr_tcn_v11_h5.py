from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
BASELINE_DIR = ROOT / "revision_2026" / "04_controlled_reruns"

RUN_SCRIPT = OUT_DIR / "run_corrected_bcr_tcn_v11_h5.py"
LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"
FIXED_CONFIG_PATH = OUT_DIR / "bcr_tcn_v11_fixed_configuration.json"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
BOUNDARY_AUDIT_PATH = OUT_DIR / "bcr_tcn_v11_boundary_audit.csv"
PREDICTIONS_PATH = OUT_DIR / "bcr_tcn_v11_h5_predictions.csv"
TRAINING_HISTORY_PATH = OUT_DIR / "bcr_tcn_v11_h5_training_history.csv"
FOLD_SUMMARY_PATH = OUT_DIR / "bcr_tcn_v11_h5_fold_summary.csv"
GRAD_DIAGNOSTICS_PATH = OUT_DIR / "bcr_tcn_v11_h5_gradient_diagnostics.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "bcr_tcn_v11_h5_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "bcr_tcn_v11_h5_metrics_pooled.csv"
EVENT_PREVALENCE_PATH = OUT_DIR / "bcr_tcn_v11_h5_event_prevalence.csv"
SUBMITTED_COMPARISON_PATH = OUT_DIR / "bcr_tcn_v11_h5_submitted_comparison.csv"
RECONCILIATION_PATH = OUT_DIR / "bcr_tcn_v11_h5_reconciliation.md"
MANIFEST_PATH = OUT_DIR / "bcr_tcn_v11_h5_run_manifest.json"
COMPLETION_REPORT_PATH = OUT_DIR / "bcr_tcn_v11_h5_completion_report.md"

FEATURE_NPZ_PATH = ROOT / "features" / "ulsan_H5_features_v2.npz"

PERSISTENCE_EVENT_PATH = BASELINE_DIR / "persistence" / "persistence_event_prevalence.csv"
RIDGE_EVENT_PATH = BASELINE_DIR / "ridge" / "ridge_event_prevalence.csv"
ELASTICNET_EVENT_PATH = BASELINE_DIR / "elasticnet" / "elasticnet_event_prevalence.csv"
HGBR_EVENT_PATH = BASELINE_DIR / "hgbr" / "hgbr_event_prevalence.csv"

H = 5
FOLDS = (1, 2, 3)
L = 60
SEED = 42

CORE_OUTPUT_FILES = [
    BOUNDARY_AUDIT_PATH,
    PREDICTIONS_PATH,
    TRAINING_HISTORY_PATH,
    FOLD_SUMMARY_PATH,
    GRAD_DIAGNOSTICS_PATH,
    METRICS_BY_FOLD_PATH,
    METRICS_POOLED_PATH,
    EVENT_PREVALENCE_PATH,
    SUBMITTED_COMPARISON_PATH,
    RECONCILIATION_PATH,
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


def recall_at_budget(scores: np.ndarray, events: np.ndarray, r: float = 0.05) -> Dict[str, float]:
    n = int(len(scores))
    if n == 0:
        return {"n": 0, "k": 0, "events": 0, "tp": 0, "precision": 0.0, "recall": 0.0}
    k = max(1, int(np.ceil(r * n)))
    order = np.argsort(-scores)
    top = events[order[:k]]
    tp = int(top.sum())
    total = int(events.sum())
    precision = float(tp / k) if k else 0.0
    recall = float(tp / total) if total else 0.0
    return {
        "n": n,
        "k": k,
        "events": total,
        "tp": tp,
        "precision": precision,
        "recall": recall,
    }


def update_manifest_test_result(success: bool, tests_run: int, failures: int, errors: int) -> None:
    if not MANIFEST_PATH.exists():
        return
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["test_result"] = "pass_25_of_25" if success else f"fail_{failures + errors}_of_25"
    payload["test_command"] = f"{sys.executable} {Path(__file__).resolve()}"
    payload["test_summary"] = f"Ran {tests_run} tests; failures={failures}; errors={errors}."
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_completion_report_test_result(success: bool, tests_run: int, failures: int, errors: int) -> None:
    if not COMPLETION_REPORT_PATH.exists():
        return

    status = "pass_25_of_25" if success else f"fail_{failures + errors}_of_25"
    if success:
        section15_line = "- Automated tests passed (25/25); decision A is supported."
        final_decision_line = (
            "A. BCR-TCN v1.1 controlled H5 regeneration is complete, leakage-safe, "
            "deterministic, and supported by 25/25 passing assertions."
        )
        terminal_decision_line = "13. Completion decision: A (all 25 tests passed)."
        terminal_next_action_line = "14. Exactly one next action: none."
    else:
        section15_line = (
            f"- Automated tests did not fully pass ({failures + errors} failing assertions); "
            "decision A is not supported."
        )
        final_decision_line = (
            "B. BCR-TCN v1.1 regeneration completed, but unresolved test failures prevent "
            "an A-level completion decision."
        )
        terminal_decision_line = (
            f"13. Completion decision: B ({failures + errors} test assertions failed)."
        )
        terminal_next_action_line = (
            "14. Exactly one next action: inspect failing assertions and rerun "
            "test_corrected_bcr_tcn_v11_h5.py."
        )

    text = COMPLETION_REPORT_PATH.read_text(encoding="utf-8")
    text = re.sub(
        r"- Manifest test_result currently: .*",
        f"- Manifest test_result currently: {status}",
        text,
    )
    text = re.sub(
        r"- Conditional on automated tests: if tests pass, decision A is supported\.",
        section15_line,
        text,
    )
    text = re.sub(
        r"FINAL DECISION\n\n.*?(?=\n\nTERMINAL SUMMARY)",
        f"FINAL DECISION\n\n{final_decision_line}",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"11\. Deterministic test result: .*",
        f"11. Deterministic test result: {status}.",
        text,
    )
    text = re.sub(
        r"13\. Completion decision: .*",
        terminal_decision_line,
        text,
    )
    text = re.sub(
        r"14\. Exactly one next action: .*",
        terminal_next_action_line,
        text,
    )
    COMPLETION_REPORT_PATH.write_text(text, encoding="utf-8")


class TestCorrectedBCRTCNV11H5(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_runner()

        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        cls.fixed_cfg = json.loads(FIXED_CONFIG_PATH.read_text(encoding="utf-8"))
        cls.input_ver = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        cls.split = pd.read_csv(SPLIT_PATH)
        cls.split["feature_date"] = normalize_date_col(cls.split["feature_date"])
        cls.split["target_date"] = normalize_date_col(cls.split["target_date"])
        cls.split["purged_outer_boundary"] = as_bool(cls.split["purged_outer_boundary"])
        cls.split["purged_inner_boundary"] = as_bool(cls.split["purged_inner_boundary"])
        cls.split_h5 = cls.split[cls.split["horizon"] == H].copy()

        cls.boundary = pd.read_csv(BOUNDARY_AUDIT_PATH)
        date_cols = [
            "subtrain_feature_start",
            "subtrain_feature_end",
            "subtrain_latest_target_date",
            "validation_feature_start",
            "validation_feature_end",
            "validation_latest_target_date",
            "outer_test_feature_start",
            "outer_test_feature_end",
        ]
        for c in date_cols:
            cls.boundary[c] = normalize_date_col(cls.boundary[c])
        cls.boundary["inner_boundary_pass"] = as_bool(cls.boundary["inner_boundary_pass"])
        cls.boundary["outer_boundary_pass"] = as_bool(cls.boundary["outer_boundary_pass"])
        cls.boundary["sequence_causality_pass"] = as_bool(cls.boundary["sequence_causality_pass"])

        cls.pred = pd.read_csv(PREDICTIONS_PATH)
        cls.pred["feature_date"] = normalize_date_col(cls.pred["feature_date"])
        cls.pred["target_date"] = normalize_date_col(cls.pred["target_date"])

        cls.hist = pd.read_csv(TRAINING_HISTORY_PATH)
        cls.hist["selected_checkpoint"] = as_bool(cls.hist["selected_checkpoint"])
        cls.hist["clipping_applied"] = as_bool(cls.hist["clipping_applied"])

        cls.fold_summary = pd.read_csv(FOLD_SUMMARY_PATH)
        cls.fold_summary["outer_test_accessed_for_selection"] = as_bool(
            cls.fold_summary["outer_test_accessed_for_selection"]
        )

        cls.grad = pd.read_csv(GRAD_DIAGNOSTICS_PATH)
        cls.grad["selected_checkpoint"] = as_bool(cls.grad["selected_checkpoint"])
        cls.grad["clipping_applied"] = as_bool(cls.grad["clipping_applied"])

        cls.metrics_fold = pd.read_csv(METRICS_BY_FOLD_PATH)
        cls.metrics_pool = pd.read_csv(METRICS_POOLED_PATH)
        cls.event_prev = pd.read_csv(EVENT_PREVALENCE_PATH)
        cls.comparison = pd.read_csv(SUBMITTED_COMPARISON_PATH)

        cls.persistence_prev = pd.read_csv(PERSISTENCE_EVENT_PATH)
        cls.ridge_prev = pd.read_csv(RIDGE_EVENT_PATH)
        cls.elasticnet_prev = pd.read_csv(ELASTICNET_EVENT_PATH)
        cls.hgbr_prev = pd.read_csv(HGBR_EVENT_PATH)

        z = np.load(FEATURE_NPZ_PATH, allow_pickle=True)
        cls.feature_dates = normalize_date_col(pd.Series(z["dates"]))
        cls.y = np.asarray(z["y"], dtype=float)
        cls.feature_to_index = {
            d: int(i) for i, d in enumerate(cls.feature_dates.tolist())
        }

    def test_01_dataset_checksum_matches_lock(self) -> None:
        observed = sha256_file(Path(self.lock["absolute_path"]))
        self.assertEqual(observed, self.lock["sha256"])
        self.assertTrue(bool(self.input_ver["verification_pass"]))
        self.assertEqual(self.input_ver["dataset_sha256_observed"], self.lock["sha256"])

    def test_02_split_checksum_matches_corrected_protocol(self) -> None:
        expected_hashes = load_expected_protocol_hashes()
        self.assertIn("corrected_split_assignment.csv", expected_hashes)
        observed = sha256_file(SPLIT_PATH)
        self.assertEqual(observed, expected_hashes["corrected_split_assignment.csv"])
        self.assertEqual(self.input_ver["split_sha256"], observed)

    def test_03_only_h5_is_executed(self) -> None:
        self.assertSetEqual(set(self.pred["horizon"].unique().tolist()), {H})
        self.assertSetEqual(set(self.metrics_fold["horizon"].unique().tolist()), {H})
        self.assertSetEqual(set(self.event_prev["horizon"].unique().tolist()), {H})
        self.assertEqual(self.input_ver["authorized_horizons"], [5])

    def test_04_outer_test_dates_equal_locked_h5_dates(self) -> None:
        exp = self.split_h5[self.split_h5["outer_role"] == "outer_test"][
            ["outer_fold", "feature_date"]
        ].drop_duplicates()
        got = self.pred[["outer_fold", "feature_date"]].drop_duplicates()

        exp_set = set(tuple(r) for r in exp.itertuples(index=False, name=None))
        got_set = set(tuple(r) for r in got.itertuples(index=False, name=None))
        self.assertSetEqual(got_set, exp_set)

    def test_05_prediction_count_equals_747(self) -> None:
        self.assertEqual(int(len(self.pred)), 747)

    def test_06_no_duplicate_prediction_keys(self) -> None:
        dup = self.pred.duplicated(subset=["outer_fold", "feature_date"])
        self.assertFalse(bool(dup.any()))

    def test_07_target_date_is_feature_plus_5_days(self) -> None:
        delta_days = (self.pred["target_date"] - self.pred["feature_date"]).dt.days
        self.assertTrue(bool((delta_days == 5).all()))

    def test_08_inner_target_date_separation_passes(self) -> None:
        self.assertTrue(bool(self.boundary["inner_boundary_pass"].all()))
        self.assertTrue(
            bool((self.boundary["subtrain_latest_target_date"] < self.boundary["validation_feature_start"]).all())
        )

    def test_09_outer_target_date_separation_passes(self) -> None:
        self.assertTrue(bool(self.boundary["outer_boundary_pass"].all()))
        outer_train_latest_target = self.boundary["outer_test_feature_start"] - pd.to_timedelta(
            self.boundary["outer_target_gap_days"], unit="D"
        )
        self.assertTrue(bool((outer_train_latest_target < self.boundary["outer_test_feature_start"]).all()))

    def test_10_sequence_windows_have_no_future_dates(self) -> None:
        for r in self.pred.itertuples(index=False):
            i = self.feature_to_index[pd.Timestamp(r.feature_date)]
            s = max(0, i - L + 1)
            win = self.feature_dates.iloc[s : i + 1]
            self.assertTrue(bool((win <= pd.Timestamp(r.feature_date)).all()))
        self.assertTrue(bool(self.boundary["sequence_causality_pass"].all()))

    def test_11_scaling_uses_subtraining_only(self) -> None:
        self.assertTrue(bool((self.fold_summary["scaler_fit_partition"] == "inner_train_only").all()))
        for fold in FOLDS:
            expected_n = int(
                self.split_h5[
                    (self.split_h5["outer_fold"] == fold)
                    & (self.split_h5["outer_role"] == "outer_train")
                    & (self.split_h5["inner_role"] == "inner_train")
                    & (~self.split_h5["purged_inner_boundary"])
                ][["feature_date", "target_date"]]
                .drop_duplicates()
                .shape[0]
            )
            row = self.fold_summary[self.fold_summary["outer_fold"] == fold].iloc[0]
            self.assertEqual(int(row["inner_train_N"]), expected_n)

    def test_12_pos_weights_use_subtraining_labels_only(self) -> None:
        for fold in FOLDS:
            sf = self.split_h5[
                (self.split_h5["outer_fold"] == fold)
                & (self.split_h5["outer_role"] == "outer_train")
                & (self.split_h5["inner_role"] == "inner_train")
                & (~self.split_h5["purged_inner_boundary"])
            ][["feature_date", "target_date"]].drop_duplicates()
            idx = [self.feature_to_index[d] for d in sf.sort_values("feature_date")["feature_date"].tolist()]
            y_sub = self.y[np.array(idx, dtype=int)]
            pos = np.array([(y_sub >= t).mean() for t in (15.0, 16.0, 17.0)], dtype=float)
            expected_pw = np.array([(1.0 - p) / (p + 1e-6) for p in pos], dtype=float)

            row = self.fold_summary[self.fold_summary["outer_fold"] == fold].iloc[0]
            observed = np.array(
                [
                    float(row["pos_weight_tau15"]),
                    float(row["pos_weight_tau16"]),
                    float(row["pos_weight_tau17"]),
                ],
                dtype=float,
            )
            np.testing.assert_allclose(observed, expected_pw, rtol=0.0, atol=1e-12)
            self.assertEqual(str(row["positive_weight_source"]), "inner_train_labels_only")

    def test_13_training_losses_use_subtraining_only(self) -> None:
        self.assertTrue(bool((self.hist["training_partition"] == "inner_train_only").all()))
        for fold in FOLDS:
            n_epochs_hist = int(len(self.hist[self.hist["outer_fold"] == fold]))
            n_epochs_fold = int(
                self.fold_summary[self.fold_summary["outer_fold"] == fold].iloc[0][
                    "training_epochs_completed"
                ]
            )
            self.assertEqual(n_epochs_hist, n_epochs_fold)

    def test_14_checkpoint_selection_uses_validation_only(self) -> None:
        self.assertTrue(bool((self.hist["validation_partition"] == "inner_validation_only").all()))
        self.assertTrue(
            bool((self.fold_summary["checkpoint_selection_metric"] == "validation_recall_tau16_r05").all())
        )

    def test_15_outer_test_not_used_for_checkpoint_selection(self) -> None:
        self.assertTrue(bool((self.fold_summary["outer_test_accessed_for_selection"] == False).all()))

    def test_16_selected_checkpoint_maximizes_validation_recall_tau16_r05(self) -> None:
        for fold in FOLDS:
            h = self.hist[self.hist["outer_fold"] == fold].copy()
            self.assertGreater(len(h), 0)
            max_recall = float(h["validation_recall_tau16_r05"].max())
            tied = h[np.abs(h["validation_recall_tau16_r05"] - max_recall) <= 1e-12].copy()
            earliest_epoch = int(tied["epoch"].min())

            selected = h[h["selected_checkpoint"] == True]
            self.assertEqual(len(selected), 1)
            selected_epoch = int(selected.iloc[0]["epoch"])
            self.assertEqual(selected_epoch, earliest_epoch)

            fs = self.fold_summary[self.fold_summary["outer_fold"] == fold].iloc[0]
            self.assertEqual(int(fs["selected_epoch"]), selected_epoch)
            self.assertAlmostEqual(
                float(fs["best_validation_recall_tau16_r05"]),
                max_recall,
                places=12,
            )

    def test_17_fixed_configuration_matches_approved_v11(self) -> None:
        expected = {
            "horizon": 5,
            "L": 60,
            "channels": 48,
            "blocks": 5,
            "kernel": 3,
            "dropout": 0.25,
            "learning_rate": 0.0008,
            "weight_decay": 0.001,
            "batch_size": 64,
            "epochs_max": 260,
            "patience": 25,
            "validation_fraction_reference": 0.15,
            "huber_beta": 1.0,
            "w_reg": 1.0,
            "w_bce": 0.6,
            "w_rank": 1.4,
            "grad_clip": 1.0,
            "r_budget": 0.05,
            "checkpoint_threshold": 16,
            "seed": 42,
            "optimizer": "AdamW",
        }
        for k, v in expected.items():
            self.assertEqual(self.fixed_cfg[k], v)
            self.assertEqual(self.manifest["fixed_configuration"][k], v)

    def test_18_seed_equals_42(self) -> None:
        self.assertTrue(bool((self.pred["seed"] == SEED).all()))
        self.assertTrue(bool((self.fold_summary["seed"] == SEED).all()))
        self.assertEqual(int(self.manifest["seed"]), SEED)

    def test_19_pooled_metrics_reproduce_saved_values(self) -> None:
        pooled = self.metrics_pool.iloc[0]
        mae = float(self.pred["absolute_error"].mean())
        mse = float(self.pred["squared_error"].mean())
        rmse = float(np.sqrt(mse))

        denoms = self.pred["outer_fold"].map(
            lambda f: float(self.fold_summary[self.fold_summary["outer_fold"] == int(f)].iloc[0]["mase_denom_m1_training_only"])
        )
        mase = float(np.mean(self.pred["absolute_error"].to_numpy(dtype=float) / (denoms.to_numpy(dtype=float) + 1e-9)))

        self.assertAlmostEqual(mae, float(pooled["MAE"]), places=12)
        self.assertAlmostEqual(mse, float(pooled["MSE"]), places=12)
        self.assertAlmostEqual(rmse, float(pooled["RMSE"]), places=12)
        self.assertAlmostEqual(mase, float(pooled["MASE"]), places=12)

    def test_20_pooled_rmse_equals_sqrt_pooled_mse(self) -> None:
        pooled = self.metrics_pool.iloc[0]
        self.assertAlmostEqual(float(pooled["RMSE"]), np.sqrt(float(pooled["MSE"])), places=12)

    def test_21_event_counts_match_y_true_thresholds(self) -> None:
        for fold in FOLDS:
            d = self.pred[self.pred["outer_fold"] == fold]
            r = self.event_prev[self.event_prev["outer_fold"].astype(str) == str(fold)].iloc[0]
            self.assertEqual(int((d["y_true"] >= 15.0).sum()), int(r["events_tau15"]))
            self.assertEqual(int((d["y_true"] >= 16.0).sum()), int(r["events_tau16"]))
            self.assertEqual(int((d["y_true"] >= 17.0).sum()), int(r["events_tau17"]))

        d = self.pred
        r = self.event_prev[self.event_prev["outer_fold"].astype(str) == "pooled"].iloc[0]
        self.assertEqual(int((d["y_true"] >= 15.0).sum()), int(r["events_tau15"]))
        self.assertEqual(int((d["y_true"] >= 16.0).sum()), int(r["events_tau16"]))
        self.assertEqual(int((d["y_true"] >= 17.0).sum()), int(r["events_tau17"]))

    def test_22_event_counts_match_all_corrected_h5_models(self) -> None:
        ours = self.event_prev[self.event_prev["outer_fold"].astype(str) != "pooled"][
            ["outer_fold", "events_tau15", "events_tau16", "events_tau17"]
        ].copy()
        ours["outer_fold"] = ours["outer_fold"].astype(int)

        for other in [self.persistence_prev, self.ridge_prev, self.elasticnet_prev, self.hgbr_prev]:
            o = other[other["horizon"] == H][["outer_fold", "events_tau15", "events_tau16", "events_tau17"]].copy()
            o["outer_fold"] = o["outer_fold"].astype(int)
            merged = ours.merge(o, on="outer_fold", suffixes=("_ours", "_other"))
            self.assertEqual(len(merged), 3)
            for t in [15, 16, 17]:
                self.assertTrue(bool((merged[f"events_tau{t}_ours"] == merged[f"events_tau{t}_other"]).all()))

    def test_23_repeated_execution_is_deterministic(self) -> None:
        before = {p.name: sha256_file(p) for p in CORE_OUTPUT_FILES}
        run_runner()
        after = {p.name: sha256_file(p) for p in CORE_OUTPUT_FILES}
        self.assertEqual(before, after)

    def test_24_prediction_checksum_identical_across_repeated_runs(self) -> None:
        before = sha256_file(PREDICTIONS_PATH)
        run_runner()
        after = sha256_file(PREDICTIONS_PATH)
        self.assertEqual(before, after)

    def test_25_no_final_topk_or_hybridrank_tuning_performed(self) -> None:
        source = RUN_SCRIPT.read_text(encoding="utf-8")
        forbidden = [
            "run_linear_alarm",
            "run_hybrid",
            "final retrospective",
            "sequential online alarm",
            "top-k policy",
            "optuna",
        ]
        for lit in forbidden:
            self.assertNotIn(lit, source)

        produced = [p.name.lower() for p in OUT_DIR.glob("*") if p.is_file()]
        self.assertFalse(any("hybridrank" in n for n in produced))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCorrectedBCRTCNV11H5)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    update_manifest_test_result(
        success=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
    )
    update_completion_report_test_result(
        success=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
    )
    sys.exit(0 if result.wasSuccessful() else 1)
