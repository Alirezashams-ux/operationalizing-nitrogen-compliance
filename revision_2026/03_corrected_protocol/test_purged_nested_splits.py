from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = Path(__file__).resolve().parent
SCRIPT = WORKDIR / "build_purged_nested_splits.py"
SPLIT_OUT = WORKDIR / "corrected_split_assignment.csv"
SUMMARY_OUT = WORKDIR / "corrected_split_summary.csv"

HORIZONS = (1, 3, 5)
FOLDS = (1, 2, 3)


def run_builder() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise").dt.normalize()


class TestPurgedNestedSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_builder()
        cls.split_df = pd.read_csv(SPLIT_OUT)
        cls.summary_df = pd.read_csv(SUMMARY_OUT)

        cls.split_df["feature_date"] = norm_dates(cls.split_df["feature_date"])
        cls.split_df["target_date"] = norm_dates(cls.split_df["target_date"])

    def _subset(self, h: int, fold: int) -> pd.DataFrame:
        return self.split_df[(self.split_df["horizon"] == h) & (self.split_df["outer_fold"] == fold)].copy()

    def test_01_no_duplicate_feature_dates_within_horizon_fold_role(self) -> None:
        dup_outer = self.split_df.duplicated(
            subset=["horizon", "outer_fold", "outer_role", "feature_date"]
        )
        self.assertFalse(bool(dup_outer.any()), "Duplicate feature_date within outer role")

        inner_applicable = self.split_df[self.split_df["inner_role"] != "not_applicable"]
        dup_inner = inner_applicable.duplicated(
            subset=["horizon", "outer_fold", "inner_fold", "inner_role", "feature_date"]
        )
        self.assertFalse(bool(dup_inner.any()), "Duplicate feature_date within inner role")

    def test_02_train_validation_test_feature_dates_do_not_overlap(self) -> None:
        for h in HORIZONS:
            for fold in FOLDS:
                sub = self._subset(h, fold)
                outer_train_kept = sub[(sub["outer_role"] == "outer_train") & (~sub["purged_outer_boundary"])]
                inner_train_kept = outer_train_kept[
                    (outer_train_kept["inner_role"] == "inner_train")
                    & (~outer_train_kept["purged_inner_boundary"])
                ]
                inner_val = outer_train_kept[outer_train_kept["inner_role"] == "inner_validation"]
                outer_test = sub[sub["outer_role"] == "outer_test"]

                a = set(inner_train_kept["feature_date"])
                b = set(inner_val["feature_date"])
                c = set(outer_test["feature_date"])

                self.assertTrue(a.isdisjoint(b), f"Inner train/validation overlap at H{h} fold{fold}")
                self.assertTrue(a.isdisjoint(c), f"Inner train/test overlap at H{h} fold{fold}")
                self.assertTrue(b.isdisjoint(c), f"Validation/test overlap at H{h} fold{fold}")

    def test_03_outer_boundary_condition(self) -> None:
        for h in HORIZONS:
            for fold in FOLDS:
                sub = self._subset(h, fold)
                outer_train_kept = sub[(sub["outer_role"] == "outer_train") & (~sub["purged_outer_boundary"])]
                outer_test = sub[sub["outer_role"] == "outer_test"]

                self.assertFalse(outer_train_kept.empty, f"Empty outer train after purge H{h} F{fold}")
                self.assertFalse(outer_test.empty, f"Empty outer test H{h} F{fold}")

                self.assertLess(
                    outer_train_kept["target_date"].max(),
                    outer_test["feature_date"].min(),
                    f"Outer leakage boundary failed for H{h} fold{fold}",
                )

    def test_04_inner_boundary_condition(self) -> None:
        for h in HORIZONS:
            for fold in FOLDS:
                sub = self._subset(h, fold)
                outer_train_kept = sub[(sub["outer_role"] == "outer_train") & (~sub["purged_outer_boundary"])]

                inner_train_kept = outer_train_kept[
                    (outer_train_kept["inner_role"] == "inner_train")
                    & (~outer_train_kept["purged_inner_boundary"])
                ]
                inner_val = outer_train_kept[outer_train_kept["inner_role"] == "inner_validation"]

                self.assertFalse(inner_train_kept.empty, f"Empty inner train after purge H{h} F{fold}")
                self.assertFalse(inner_val.empty, f"Empty inner validation H{h} F{fold}")

                self.assertLess(
                    inner_train_kept["target_date"].max(),
                    inner_val["feature_date"].min(),
                    f"Inner leakage boundary failed for H{h} fold{fold}",
                )

    def test_05_test_dates_match_canonical_test_windows(self) -> None:
        for h in HORIZONS:
            assignment = pd.read_csv(ROOT / "results" / "tables" / f"blocked_cv_assignment_matrix_H{h}.csv")
            assignment["date"] = norm_dates(assignment["date"])

            for fold in FOLDS:
                expected = set(assignment.loc[assignment[f"fold{fold}_set"] == "test", "date"])
                observed = set(
                    self.split_df[
                        (self.split_df["horizon"] == h)
                        & (self.split_df["outer_fold"] == fold)
                        & (self.split_df["outer_role"] == "outer_test")
                    ]["feature_date"]
                )
                self.assertSetEqual(
                    observed,
                    expected,
                    f"Canonical test-window mismatch for H{h} fold{fold}",
                )

    def test_06_test_dates_unchanged_across_model_families_by_construction(self) -> None:
        # The split assignment has no model-family dimension, so one canonical test-date set
        # is enforced per horizon/fold for every model run that consumes this file.
        self.assertNotIn("model", self.split_df.columns)

        signatures = {}
        for h in HORIZONS:
            for fold in FOLDS:
                dates = sorted(
                    self.split_df[
                        (self.split_df["horizon"] == h)
                        & (self.split_df["outer_fold"] == fold)
                        & (self.split_df["outer_role"] == "outer_test")
                    ]["feature_date"].dt.strftime("%Y-%m-%d").tolist()
                )
                digest = hashlib.sha256("|".join(dates).encode("utf-8")).hexdigest()
                signatures[(h, fold)] = digest

        self.assertEqual(len(signatures), 9)
        self.assertEqual(len(set(signatures.values())), 9)

    def test_07_target_dates_are_horizon_shifted(self) -> None:
        delta_days = (self.split_df["target_date"] - self.split_df["feature_date"]).dt.days
        self.assertTrue(
            bool((delta_days == self.split_df["horizon"]).all()),
            "target_date is not always feature_date + horizon",
        )

    def test_08_generator_contains_no_preprocessing_or_model_fitting(self) -> None:
        script_text = SCRIPT.read_text(encoding="utf-8").lower()
        forbidden_fragments = [
            ".fit(",
            "sklearn",
            "torch",
            "optuna",
            "xgboost",
            "lightgbm",
            "catboost",
        ]
        for frag in forbidden_fragments:
            self.assertNotIn(frag, script_text, f"Forbidden training/preprocessing fragment found: {frag}")

    def test_09_split_construction_is_deterministic(self) -> None:
        before_split = pd.read_csv(SPLIT_OUT)
        before_summary = pd.read_csv(SUMMARY_OUT)

        run_builder()

        after_split = pd.read_csv(SPLIT_OUT)
        after_summary = pd.read_csv(SUMMARY_OUT)

        pd.testing.assert_frame_equal(before_split, after_split, check_dtype=False)
        pd.testing.assert_frame_equal(before_summary, after_summary, check_dtype=False)

    def test_10_repeated_execution_checksum_identical(self) -> None:
        split_sha_1 = file_sha256(SPLIT_OUT)
        summary_sha_1 = file_sha256(SUMMARY_OUT)

        run_builder()

        split_sha_2 = file_sha256(SPLIT_OUT)
        summary_sha_2 = file_sha256(SUMMARY_OUT)

        self.assertEqual(split_sha_1, split_sha_2, "Split assignment checksum changed across runs")
        self.assertEqual(summary_sha_1, summary_sha_2, "Split summary checksum changed across runs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
