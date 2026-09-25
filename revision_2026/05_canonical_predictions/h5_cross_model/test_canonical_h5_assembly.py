from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"

BUILD_SCRIPT = OUT_DIR / "build_canonical_h5_predictions.py"

DATASET_LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
KEY_AUDIT_PATH = OUT_DIR / "canonical_h5_key_audit.csv"
LONG_PATH = OUT_DIR / "canonical_h5_predictions_long.csv"
WIDE_PATH = OUT_DIR / "canonical_h5_predictions_wide.csv"
POINT_BY_FOLD_PATH = OUT_DIR / "canonical_h5_point_metrics_by_fold.csv"
POINT_POOLED_PATH = OUT_DIR / "canonical_h5_point_metrics_pooled.csv"
EVENT_PREV_PATH = OUT_DIR / "canonical_h5_event_prevalence.csv"
SOURCE_REGISTRY_PATH = OUT_DIR / "canonical_h5_source_registry.csv"
CHECKSUM_PATH = OUT_DIR / "canonical_h5_checksums.sha256"
MANIFEST_PATH = OUT_DIR / "canonical_h5_assembly_manifest.json"
REPORT_PATH = OUT_DIR / "canonical_h5_assembly_report.md"

MODEL_ORDER = ["Persistence", "Ridge", "ElasticNet", "HGBR", "BCR-TCN v1.1"]
KEY_COLS = ["horizon", "outer_fold", "feature_date", "target_date"]

SOURCE_PREDICTIONS = {
    "Persistence": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "persistence"
    / "persistence_predictions.csv",
    "Ridge": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "ridge"
    / "ridge_predictions.csv",
    "ElasticNet": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "elasticnet"
    / "elasticnet_predictions.csv",
    "HGBR": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "hgbr"
    / "hgbr_predictions.csv",
    "BCR-TCN v1.1": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "bcr_tcn_v11_h5"
    / "bcr_tcn_v11_h5_predictions.csv",
}

SOURCE_MANIFESTS = {
    "Persistence": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "persistence"
    / "persistence_run_manifest.json",
    "Ridge": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "ridge"
    / "ridge_run_manifest.json",
    "ElasticNet": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "elasticnet"
    / "elasticnet_run_manifest.json",
    "HGBR": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "hgbr"
    / "hgbr_run_manifest.json",
    "BCR-TCN v1.1": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "bcr_tcn_v11_h5"
    / "bcr_tcn_v11_h5_run_manifest.json",
}

SOURCE_METRICS_POOLED = {
    "Persistence": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "persistence"
    / "persistence_metrics_pooled.csv",
    "Ridge": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "ridge"
    / "ridge_metrics_pooled.csv",
    "ElasticNet": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "elasticnet"
    / "elasticnet_metrics_pooled.csv",
    "HGBR": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "hgbr"
    / "hgbr_metrics_pooled.csv",
    "BCR-TCN v1.1": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "bcr_tcn_v11_h5"
    / "bcr_tcn_v11_h5_metrics_pooled.csv",
}

SOURCE_METRICS_BY_FOLD = {
    "Persistence": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "persistence"
    / "persistence_metrics_by_fold.csv",
    "Ridge": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "ridge"
    / "ridge_metrics_by_fold.csv",
    "ElasticNet": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "elasticnet"
    / "elasticnet_metrics_by_fold.csv",
    "HGBR": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "hgbr"
    / "hgbr_metrics_by_fold.csv",
    "BCR-TCN v1.1": ROOT
    / "revision_2026"
    / "04_controlled_reruns"
    / "bcr_tcn_v11_h5"
    / "bcr_tcn_v11_h5_metrics_by_fold.csv",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_build_script() -> None:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=ROOT, check=True)


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise").dt.strftime("%Y-%m-%d")


def key_tuples(df: pd.DataFrame) -> set:
    return set(tuple(r) for r in df[KEY_COLS].itertuples(index=False, name=None))


def load_protocol_hashes() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in PROTOCOL_SHA_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        out[parts[-1]] = parts[0]
    return out


def update_manifest_test_result(success: bool, tests_run: int, failures: int, errors: int) -> None:
    if not MANIFEST_PATH.exists():
        return
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["test_result"] = "pass_21_of_21" if success else f"fail_{failures + errors}_of_21"
    payload["test_command"] = f"{sys.executable} {Path(__file__).resolve()}"
    payload["test_summary"] = f"Ran {tests_run} tests; failures={failures}; errors={errors}."
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestCanonicalH5Assembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_build_script()

        cls.lock = json.loads(DATASET_LOCK_PATH.read_text(encoding="utf-8"))
        cls.input_ver = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        cls.split = pd.read_csv(SPLIT_PATH)
        cls.split["horizon"] = pd.to_numeric(cls.split["horizon"], errors="raise").astype(int)
        cls.split["outer_fold"] = pd.to_numeric(cls.split["outer_fold"], errors="raise").astype(int)
        cls.split["feature_date"] = normalize_date(cls.split["feature_date"])
        cls.split["target_date"] = normalize_date(cls.split["target_date"])
        cls.h5_outer = cls.split[(cls.split["horizon"] == 5) & (cls.split["outer_role"] == "outer_test")][
            KEY_COLS
        ].drop_duplicates()

        cls.long = pd.read_csv(LONG_PATH)
        cls.long["horizon"] = pd.to_numeric(cls.long["horizon"], errors="raise").astype(int)
        cls.long["outer_fold"] = pd.to_numeric(cls.long["outer_fold"], errors="raise").astype(int)
        cls.long["feature_date"] = normalize_date(cls.long["feature_date"])
        cls.long["target_date"] = normalize_date(cls.long["target_date"])

        cls.wide = pd.read_csv(WIDE_PATH)
        cls.wide["horizon"] = pd.to_numeric(cls.wide["horizon"], errors="raise").astype(int)
        cls.wide["outer_fold"] = pd.to_numeric(cls.wide["outer_fold"], errors="raise").astype(int)
        cls.wide["feature_date"] = normalize_date(cls.wide["feature_date"])
        cls.wide["target_date"] = normalize_date(cls.wide["target_date"])

        cls.metrics_fold = pd.read_csv(POINT_BY_FOLD_PATH)
        cls.metrics_pool = pd.read_csv(POINT_POOLED_PATH)
        cls.event_prev = pd.read_csv(EVENT_PREV_PATH)
        cls.source_registry = pd.read_csv(SOURCE_REGISTRY_PATH)
        cls.key_audit = pd.read_csv(KEY_AUDIT_PATH)

    def test_01_dataset_checksum_matches_lock(self) -> None:
        observed = sha256_file(Path(self.lock["absolute_path"]))
        self.assertEqual(observed, self.lock["sha256"])
        self.assertEqual(self.input_ver["dataset_sha256"], self.lock["sha256"])

    def test_02_split_checksum_matches_corrected_protocol(self) -> None:
        protocol_hashes = load_protocol_hashes()
        expected = protocol_hashes["corrected_split_assignment.csv"]
        observed = sha256_file(SPLIT_PATH)
        self.assertEqual(observed, expected)
        self.assertEqual(self.input_ver["split_sha256"], observed)

    def test_03_source_prediction_checksums_match_manifests(self) -> None:
        for model in MODEL_ORDER:
            manifest = json.loads(SOURCE_MANIFESTS[model].read_text(encoding="utf-8"))
            observed = sha256_file(SOURCE_PREDICTIONS[model])
            self.assertEqual(observed, manifest["prediction_sha256"])

    def test_04_all_source_completion_decisions_equal_A(self) -> None:
        decisions = self.input_ver["completion_decision_by_model"]
        self.assertEqual(set(decisions.keys()), set(MODEL_ORDER))
        self.assertTrue(all(v == "A" for v in decisions.values()))

    def test_05_only_h5_is_included(self) -> None:
        self.assertSetEqual(set(self.long["horizon"].unique().tolist()), {5})
        self.assertSetEqual(set(self.wide["horizon"].unique().tolist()), {5})
        self.assertSetEqual(set(pd.to_numeric(self.metrics_fold["horizon"], errors="raise").astype(int).unique().tolist()), {5})
        self.assertSetEqual(set(pd.to_numeric(self.metrics_pool["horizon"], errors="raise").astype(int).unique().tolist()), {5})

    def test_06_canonical_key_count_equals_747(self) -> None:
        self.assertEqual(int(len(self.h5_outer)), 747)
        self.assertEqual(int(len(self.wide)), 747)

    def test_07_every_model_has_exactly_747_rows(self) -> None:
        counts = self.long.groupby("model").size().to_dict()
        self.assertEqual(set(counts.keys()), set(MODEL_ORDER))
        self.assertTrue(all(int(v) == 747 for v in counts.values()))

    def test_08_all_model_key_sets_are_identical(self) -> None:
        expected = key_tuples(self.h5_outer)
        for model in MODEL_ORDER:
            d = self.long[self.long["model"] == model]
            self.assertSetEqual(key_tuples(d), expected)

    def test_09_no_duplicate_model_key_rows_exist(self) -> None:
        dup = self.long.duplicated(subset=["model"] + KEY_COLS)
        self.assertFalse(bool(dup.any()))

    def test_10_y_true_is_identical_across_models(self) -> None:
        ref = self.long[self.long["model"] == "Persistence"][KEY_COLS + ["y_true"]].rename(columns={"y_true": "y_true_ref"})
        for model in MODEL_ORDER:
            d = self.long[self.long["model"] == model][KEY_COLS + ["y_true"]]
            m = d.merge(ref, on=KEY_COLS, how="inner")
            self.assertEqual(len(m), 747)
            self.assertLessEqual(float((m["y_true"] - m["y_true_ref"]).abs().max()), 1e-6)

    def test_11_target_date_equals_feature_plus_5_days(self) -> None:
        delta = (
            pd.to_datetime(self.long["target_date"], errors="raise")
            - pd.to_datetime(self.long["feature_date"], errors="raise")
        ).dt.days
        self.assertTrue(bool((delta == 5).all()))

    def test_12_fold_membership_matches_locked_split(self) -> None:
        expected = key_tuples(self.h5_outer)
        observed = key_tuples(self.wide)
        self.assertSetEqual(observed, expected)

    def test_13_event_labels_reproduce_y_true_thresholds(self) -> None:
        for tau, col in [(15.0, "event_tau15"), (16.0, "event_tau16"), (17.0, "event_tau17")]:
            expected = (self.long["y_true"] >= tau).astype(int)
            self.assertTrue(bool((expected == self.long[col]).all()))

    def test_14_event_counts_are_identical_across_models(self) -> None:
        ev = self.event_prev.copy()
        for fold in ["1", "2", "3", "pooled"]:
            sub = ev[ev["outer_fold"].astype(str) == fold]
            self.assertEqual(len(sub), 5)
            self.assertEqual(sub["events_tau15"].nunique(), 1)
            self.assertEqual(sub["events_tau16"].nunique(), 1)
            self.assertEqual(sub["events_tau17"].nunique(), 1)

    def test_15_bcr_threshold_scores_reproduce_source_probabilities(self) -> None:
        src = pd.read_csv(SOURCE_PREDICTIONS["BCR-TCN v1.1"])
        src = src[pd.to_numeric(src["horizon"], errors="raise").astype(int) == 5].copy()
        src["outer_fold"] = pd.to_numeric(src["outer_fold"], errors="raise").astype(int)
        src["feature_date"] = normalize_date(src["feature_date"])
        src["target_date"] = normalize_date(src["target_date"])

        d = self.long[self.long["model"] == "BCR-TCN v1.1"][KEY_COLS + ["risk_score_tau15", "risk_score_tau16", "risk_score_tau17"]]
        m = d.merge(src[KEY_COLS + ["p_tau15", "p_tau16", "p_tau17"]], on=KEY_COLS, how="inner")
        self.assertEqual(len(m), 747)
        np.testing.assert_allclose(m["risk_score_tau15"].to_numpy(dtype=float), m["p_tau15"].to_numpy(dtype=float), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(m["risk_score_tau16"].to_numpy(dtype=float), m["p_tau16"].to_numpy(dtype=float), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(m["risk_score_tau17"].to_numpy(dtype=float), m["p_tau17"].to_numpy(dtype=float), rtol=0.0, atol=0.0)

    def test_16_regression_model_risk_scores_reproduce_y_pred(self) -> None:
        for model in ["Persistence", "Ridge", "ElasticNet", "HGBR"]:
            d = self.long[self.long["model"] == model]
            np.testing.assert_allclose(d["risk_score_tau15"].to_numpy(dtype=float), d["y_pred"].to_numpy(dtype=float), rtol=0.0, atol=0.0)
            np.testing.assert_allclose(d["risk_score_tau16"].to_numpy(dtype=float), d["y_pred"].to_numpy(dtype=float), rtol=0.0, atol=0.0)
            np.testing.assert_allclose(d["risk_score_tau17"].to_numpy(dtype=float), d["y_pred"].to_numpy(dtype=float), rtol=0.0, atol=0.0)

    def test_17_no_source_numeric_prediction_value_was_modified(self) -> None:
        for model in MODEL_ORDER:
            src = pd.read_csv(SOURCE_PREDICTIONS[model])
            src = src[pd.to_numeric(src["horizon"], errors="raise").astype(int) == 5].copy()
            src["outer_fold"] = pd.to_numeric(src["outer_fold"], errors="raise").astype(int)
            src["feature_date"] = normalize_date(src["feature_date"])
            src["target_date"] = normalize_date(src["target_date"])
            src["y_pred"] = pd.to_numeric(src["y_pred"], errors="raise").astype(float)

            dst = self.long[self.long["model"] == model][KEY_COLS + ["y_pred"]]
            merged = dst.merge(src[KEY_COLS + ["y_pred"]], on=KEY_COLS, suffixes=("_dst", "_src"), how="inner")
            self.assertEqual(len(merged), 747)
            np.testing.assert_allclose(
                merged["y_pred_dst"].to_numpy(dtype=float),
                merged["y_pred_src"].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )

        # Registry explicitly states no numeric modification for every model.
        self.assertTrue(bool((self.source_registry["numeric_values_modified"].astype(str).str.lower() == "false").all()))

    def test_18_canonical_point_metrics_reproduce_source_metrics(self) -> None:
        pool = self.metrics_pool.copy()
        fold = self.metrics_fold.copy()

        for model in MODEL_ORDER:
            src_pool = pd.read_csv(SOURCE_METRICS_POOLED[model])
            src_pool["horizon"] = pd.to_numeric(src_pool["horizon"], errors="raise").astype(int)
            src_pool = src_pool[(src_pool["horizon"] == 5) & (src_pool["outer_fold"].astype(str).str.lower() == "pooled")]
            self.assertEqual(len(src_pool), 1)
            src_row = src_pool.iloc[0]

            dst_row = pool[pool["model"] == model].iloc[0]
            self.assertEqual(int(dst_row["N"]), int(src_row["N"]))
            self.assertAlmostEqual(float(dst_row["MAE"]), float(src_row["MAE"]), places=9)
            self.assertAlmostEqual(float(dst_row["MSE"]), float(src_row["MSE"]), places=9)
            self.assertAlmostEqual(float(dst_row["RMSE"]), float(src_row["RMSE"]), places=9)
            self.assertAlmostEqual(float(dst_row["MASE"]), float(src_row["MASE"]), places=9)

            src_fold = pd.read_csv(SOURCE_METRICS_BY_FOLD[model])
            src_fold["horizon"] = pd.to_numeric(src_fold["horizon"], errors="raise").astype(int)
            src_fold["outer_fold"] = pd.to_numeric(src_fold["outer_fold"], errors="raise").astype(int)
            src_fold = src_fold[src_fold["horizon"] == 5].copy()

            dst_fold = fold[fold["model"] == model].copy()
            dst_fold["outer_fold"] = pd.to_numeric(dst_fold["outer_fold"], errors="raise").astype(int)
            merged = dst_fold.merge(src_fold[["outer_fold", "N", "MAE", "MSE", "RMSE", "MASE"]], on="outer_fold", suffixes=("_dst", "_src"))
            self.assertEqual(len(merged), 3)
            for col in ["N", "MAE", "MSE", "RMSE", "MASE"]:
                if col == "N":
                    self.assertTrue(bool((merged[f"{col}_dst"].astype(int) == merged[f"{col}_src"].astype(int)).all()))
                else:
                    np.testing.assert_allclose(
                        merged[f"{col}_dst"].to_numpy(dtype=float),
                        merged[f"{col}_src"].to_numpy(dtype=float),
                        rtol=0.0,
                        atol=1e-9,
                    )

    def test_19_pooled_rmse_equals_sqrt_pooled_mse(self) -> None:
        for r in self.metrics_pool.itertuples(index=False):
            self.assertAlmostEqual(float(r.RMSE), float(np.sqrt(float(r.MSE))), places=12)

    def test_20_repeated_assembly_is_deterministic_checksum_identical(self) -> None:
        targets = [
            INPUT_VERIFICATION_PATH,
            KEY_AUDIT_PATH,
            LONG_PATH,
            WIDE_PATH,
            POINT_BY_FOLD_PATH,
            POINT_POOLED_PATH,
            EVENT_PREV_PATH,
            SOURCE_REGISTRY_PATH,
            CHECKSUM_PATH,
            MANIFEST_PATH,
            REPORT_PATH,
        ]
        before = {p.name: sha256_file(p) for p in targets}
        run_build_script()
        after = {p.name: sha256_file(p) for p in targets}
        self.assertEqual(before, after)

    def test_21_no_alarm_or_hybrid_policy_outputs_are_produced(self) -> None:
        forbidden_terms = [
            "alarm",
            "topk",
            "top_k",
            "hybrid",
            "rank_norm",
            "sequential_cutoff",
            "sequential_policy",
            "hybrid_weight",
        ]

        # Output filenames
        for p in OUT_DIR.iterdir():
            if not p.is_file():
                continue
            if p.name in {BUILD_SCRIPT.name, Path(__file__).name}:
                continue
            lname = p.name.lower()
            # Allowed phrase in report title/content context only, not in generated data artifact names.
            self.assertFalse(
                any(term in lname for term in forbidden_terms),
                msg=f"Forbidden policy artifact filename detected: {p.name}",
            )

        # Data columns should not include alarm/policy fields.
        for df in [self.long, self.wide, self.metrics_fold, self.metrics_pool, self.event_prev]:
            cols = [c.lower() for c in df.columns]
            self.assertFalse(any("alarm" in c for c in cols))
            self.assertFalse(any("topk" in c or "top_k" in c for c in cols))
            self.assertFalse(any("hybrid" in c for c in cols))
            self.assertFalse(any("rank_norm" in c for c in cols))
            self.assertFalse(any("sequential" in c for c in cols))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCanonicalH5Assembly)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    update_manifest_test_result(
        success=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
    )

    # Re-run assembly once to propagate test_result into report/checksums deterministically.
    run_build_script()

    sys.exit(0 if result.wasSuccessful() else 1)
