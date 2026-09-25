from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = OUT_DIR / "build_quota_enforced_h5_alarm_policy.py"

EXPECTED_BRANCH = "controlled-reruns-v1"
MODELS = [
    "Persistence",
    "Ridge",
    "ElasticNet",
    "HGBR",
    "BCR-TCN v1.1",
    "HybridRank_fixed_documented",
]
THRESHOLDS = [15, 16, 17]
BUDGETS = [0.05, 0.10]

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

SOURCE_DIR = ROOT / "revision_2026" / "08_sequential_alarm_policy" / "h5"
SOURCE_DECISIONS_PATH = SOURCE_DIR / "sequential_h5_alarm_decisions.csv"
SOURCE_MANIFEST_PATH = SOURCE_DIR / "sequential_h5_alarm_policy_manifest.json"
SOURCE_CHECKSUMS_PATH = SOURCE_DIR / "sequential_h5_alarm_policy_checksums.sha256"

INPUT_VERIFICATION_PATH = OUT_DIR / "quota_enforced_h5_input_verification.json"
DECISIONS_PATH = OUT_DIR / "quota_enforced_h5_alarm_decisions.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "quota_enforced_h5_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "quota_enforced_h5_metrics_pooled.csv"
CAPACITY_AUDIT_PATH = OUT_DIR / "quota_enforced_h5_capacity_audit.csv"
SUPPRESSION_AUDIT_PATH = OUT_DIR / "quota_enforced_h5_suppression_audit.csv"
COMPARISON_PATH = OUT_DIR / "quota_enforced_vs_unconstrained_sequential.csv"
INFO_AUDIT_PATH = OUT_DIR / "quota_enforced_h5_information_isolation_audit.json"
MANIFEST_PATH = OUT_DIR / "quota_enforced_h5_manifest.json"
REPORT_PATH = OUT_DIR / "quota_enforced_h5_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "quota_enforced_h5_checksums.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksum_manifest(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        out[parts[-1]] = parts[0]
    return out


def expected_checksum_for_filename(path: Path, filename: str) -> str:
    lookup = parse_checksum_manifest(path)
    if filename in lookup:
        return lookup[filename]
    for k, v in lookup.items():
        if k.endswith("/" + filename):
            return v
    raise KeyError("Missing checksum for {} in {}".format(filename, path))


def run_build_script() -> None:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=ROOT, check=True)


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def update_manifest_test_result(success: bool, tests_run: int, failures: int, errors: int) -> None:
    if not MANIFEST_PATH.exists():
        return

    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["test_result"] = "pass_25_of_25" if success else "fail_{}_of_25".format(failures + errors)
    payload["test_command"] = "{} {}".format(sys.executable, Path(__file__).resolve())
    payload["test_summary"] = "Ran {} tests; failures={}; errors={}.".format(
        tests_run, failures, errors
    )
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def deterministic_snapshot(paths: List[Path]) -> Dict[str, str]:
    return {p.name: sha256_file(p) for p in paths}


class TestQuotaEnforcedH5AlarmPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_build_script()

        cls.input_ver = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.report = REPORT_PATH.read_text(encoding="utf-8")

        cls.decisions = pd.read_csv(DECISIONS_PATH)
        cls.decisions["horizon"] = pd.to_numeric(cls.decisions["horizon"], errors="raise").astype(int)
        cls.decisions["outer_fold"] = pd.to_numeric(cls.decisions["outer_fold"], errors="raise").astype(int)
        cls.decisions["canonical_row_id"] = pd.to_numeric(cls.decisions["canonical_row_id"], errors="raise").astype(int)
        cls.decisions["threshold"] = pd.to_numeric(cls.decisions["threshold"], errors="raise").astype(int)
        cls.decisions["nominal_budget_r"] = pd.to_numeric(cls.decisions["nominal_budget_r"], errors="raise").astype(float)
        cls.decisions["feature_date"] = normalize_date_col(cls.decisions["feature_date"])
        cls.decisions["target_date"] = normalize_date_col(cls.decisions["target_date"])

        bool_cols = [
            "evaluation_eligible",
            "startup_calibration",
            "future_scores_used",
            "labels_used_for_score",
            "labels_used_for_cutoff",
        ]
        for c in bool_cols:
            cls.decisions[c] = cls.decisions[c].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )

        int_cols = [
            "candidate_alarm",
            "capacity_to_date",
            "alarms_before_current_date",
            "quota_available",
            "alarm_flag",
            "suppressed_by_quota",
            "unused_capacity_after_decision",
            "cumulative_alarms_to_date",
            "unconstrained_alarm_flag",
            "past_score_count",
        ]
        for c in int_cols:
            cls.decisions[c] = pd.to_numeric(cls.decisions[c], errors="raise").astype(int)

        cls.metrics_by_fold = pd.read_csv(METRICS_BY_FOLD_PATH)
        cls.metrics_pooled = pd.read_csv(METRICS_POOLED_PATH)
        cls.capacity_audit = pd.read_csv(CAPACITY_AUDIT_PATH)
        cls.suppression = pd.read_csv(SUPPRESSION_AUDIT_PATH)
        cls.comparison = pd.read_csv(COMPARISON_PATH)
        cls.info_audit = json.loads(INFO_AUDIT_PATH.read_text(encoding="utf-8"))

        cls.source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))

        cls.split = pd.read_csv(SPLIT_PATH)
        cls.split["horizon"] = pd.to_numeric(cls.split["horizon"], errors="raise").astype(int)
        cls.split["outer_fold"] = pd.to_numeric(cls.split["outer_fold"], errors="raise").astype(int)
        cls.split["feature_date"] = normalize_date_col(cls.split["feature_date"])
        cls.split["target_date"] = normalize_date_col(cls.split["target_date"])
        cls.h5_outer = cls.split[(cls.split["horizon"] == 5) & (cls.split["outer_role"] == "outer_test")][
            ["horizon", "outer_fold", "feature_date", "target_date"]
        ].drop_duplicates()

    def test_01_dataset_checksum_matches_lock(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        observed = sha256_file(Path(lock["absolute_path"]))
        self.assertEqual(observed, lock["sha256"])
        self.assertEqual(self.input_ver["dataset_sha256"], lock["sha256"])

    def test_02_split_checksum_matches_corrected_protocol(self) -> None:
        expected = expected_checksum_for_filename(PROTOCOL_SHA_PATH, "corrected_split_assignment.csv")
        observed = sha256_file(SPLIT_PATH)
        self.assertEqual(observed, expected)
        self.assertEqual(self.input_ver["split_sha256"], observed)

    def test_03_source_prediction_checksum_matches_frozen_manifest(self) -> None:
        expected_from_checksums = expected_checksum_for_filename(
            SOURCE_CHECKSUMS_PATH, "sequential_h5_alarm_decisions.csv"
        )
        observed = sha256_file(SOURCE_DECISIONS_PATH)
        self.assertEqual(observed, expected_from_checksums)

        source_output_checksums = self.source_manifest["output_checksums"]
        self.assertEqual(source_output_checksums["sequential_h5_alarm_decisions.csv"], observed)

    def test_04_only_authorized_models_thresholds_budgets_h5(self) -> None:
        self.assertSetEqual(set(self.decisions["model"].unique().tolist()), set(MODELS))
        self.assertSetEqual(set(self.decisions["threshold"].astype(int).unique().tolist()), set(THRESHOLDS))
        self.assertSetEqual(set(np.round(self.decisions["nominal_budget_r"].astype(float).unique(), 10).tolist()), set(BUDGETS))
        self.assertSetEqual(set(self.decisions["horizon"].astype(int).unique().tolist()), {5})

    def test_05_canonical_key_coverage_per_group(self) -> None:
        expected_keys = set(
            tuple(r)
            for r in self.h5_outer[["horizon", "outer_fold", "feature_date", "target_date"]].itertuples(
                index=False, name=None
            )
        )

        for _, g in self.decisions.groupby(["model", "threshold", "nominal_budget_r"], sort=True):
            keys = set(
                tuple(r)
                for r in g[["horizon", "outer_fold", "feature_date", "target_date"]].itertuples(
                    index=False, name=None
                )
            )
            self.assertEqual(len(g), 747)
            self.assertSetEqual(keys, expected_keys)
            self.assertEqual(
                int(g.duplicated(subset=["horizon", "outer_fold", "feature_date", "target_date"]).sum()),
                0,
            )

    def test_06_startup_window_no_alarm_and_eligible_count_219(self) -> None:
        for _, g in self.decisions.groupby(["model", "outer_fold", "threshold", "nominal_budget_r"], sort=True):
            gs = g.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
            startup = gs["startup_calibration"].astype(bool)
            eligible = gs["evaluation_eligible"].astype(bool)
            self.assertEqual(int(startup.sum()), 30)
            self.assertEqual(int(eligible.sum()), 219)
            self.assertEqual(int(gs[startup]["alarm_flag"].astype(int).sum()), 0)

    def test_07_candidate_alarm_reproduces_frozen_unconstrained_baseline(self) -> None:
        self.assertTrue(
            bool(
                (
                    self.decisions["candidate_alarm"].astype(int)
                    == self.decisions["unconstrained_alarm_flag"].astype(int)
                ).all()
            )
        )

    def test_08_quota_rule_equation_holds_rowwise(self) -> None:
        d = self.decisions.copy()
        lhs = d["alarm_flag"].astype(int)
        rhs = (
            (d["candidate_alarm"].astype(int) == 1)
            & (d["alarms_before_current_date"].astype(int) < d["capacity_to_date"].astype(int))
        ).astype(int)
        self.assertTrue(bool((lhs == rhs).all()))

    def test_09_cumulative_quota_never_exceeds_prefix_capacity(self) -> None:
        for _, g in self.decisions.groupby(["model", "outer_fold", "threshold", "nominal_budget_r"], sort=True):
            e = g[g["evaluation_eligible"].astype(bool)].sort_values(
                ["target_date", "feature_date", "canonical_row_id"]
            )
            cum = e["cumulative_alarms_to_date"].astype(int).to_numpy()
            cap = e["capacity_to_date"].astype(int).to_numpy()
            self.assertTrue(bool((cum <= cap).all()))

    def test_10_final_alarm_caps_fold_and_pooled(self) -> None:
        f = self.suppression[self.suppression["outer_fold"].astype(str).isin(["1", "2", "3"])].copy()
        f005 = f[np.isclose(f["nominal_budget_r"].astype(float), 0.05)]
        f010 = f[np.isclose(f["nominal_budget_r"].astype(float), 0.10)]
        self.assertTrue(bool((f005["issued_alarms"].astype(int) <= 11).all()))
        self.assertTrue(bool((f010["issued_alarms"].astype(int) <= 22).all()))

        p = self.suppression[self.suppression["outer_fold"].astype(str) == "pooled"].copy()
        p005 = p[np.isclose(p["nominal_budget_r"].astype(float), 0.05)]
        p010 = p[np.isclose(p["nominal_budget_r"].astype(float), 0.10)]
        self.assertTrue(bool((p005["issued_alarms"].astype(int) <= 33).all()))
        self.assertTrue(bool((p010["issued_alarms"].astype(int) <= 66).all()))

    def test_11_only_quota_gate_can_suppress_candidates(self) -> None:
        d = self.decisions.copy()
        delta = d["candidate_alarm"].astype(int) - d["alarm_flag"].astype(int)
        self.assertTrue(bool(((delta == 0) | (delta == 1)).all()))
        self.assertTrue(bool((delta == d["suppressed_by_quota"].astype(int)).all()))

    def test_12_maximum_prefix_excess_zero(self) -> None:
        self.assertTrue(
            bool((self.suppression["maximum_prefix_excess_over_capacity"].astype(int) == 0).all())
        )
        self.assertTrue(
            bool((self.capacity_audit["maximum_prefix_excess_over_capacity"].astype(int) == 0).all())
        )

    def test_13_no_future_or_label_usage(self) -> None:
        self.assertTrue(bool((self.decisions["future_scores_used"].astype(bool) == False).all()))
        self.assertTrue(bool((self.decisions["labels_used_for_score"].astype(bool) == False).all()))
        self.assertTrue(bool((self.decisions["labels_used_for_cutoff"].astype(bool) == False).all()))

    def test_14_strict_equality_to_cutoff_is_no_alarm(self) -> None:
        eligible = self.decisions[self.decisions["evaluation_eligible"].astype(bool)].copy()
        eq = np.isclose(
            eligible["policy_score"].astype(float).to_numpy(),
            eligible["past_only_cutoff"].astype(float).to_numpy(),
            rtol=0.0,
            atol=0.0,
        )
        if eq.any():
            self.assertTrue(bool((eligible.loc[eq, "candidate_alarm"].astype(int) == 0).all()))
            self.assertTrue(bool((eligible.loc[eq, "alarm_flag"].astype(int) == 0).all()))

    def test_15_decision_output_has_required_columns(self) -> None:
        required = {
            "candidate_alarm",
            "capacity_to_date",
            "alarms_before_current_date",
            "quota_available",
            "alarm_flag",
            "suppressed_by_quota",
            "unused_capacity_after_decision",
            "past_only_cutoff",
            "policy_score",
            "evaluation_eligible",
            "startup_calibration",
        }
        self.assertTrue(required.issubset(set(self.decisions.columns)))

    def test_16_metrics_consistency_fold_level(self) -> None:
        m = self.metrics_by_fold.copy()
        self.assertTrue(bool((m["eligible_N"].astype(int) == 219).all()))
        self.assertTrue(
            bool(
                (
                    m["TP"].astype(int)
                    + m["FP"].astype(int)
                    + m["FN"].astype(int)
                    + m["TN"].astype(int)
                    == m["eligible_N"].astype(int)
                ).all()
            )
        )

    def test_17_metrics_consistency_pooled(self) -> None:
        agg = (
            self.metrics_by_fold.groupby(["model", "horizon", "threshold", "nominal_budget_r"], as_index=False)
            .agg(
                pooled_eligible_N=("eligible_N", "sum"),
                pooled_eligible_events=("eligible_events", "sum"),
                pooled_alarms=("alarms_issued", "sum"),
                pooled_TP=("TP", "sum"),
                pooled_FP=("FP", "sum"),
                pooled_FN=("FN", "sum"),
                pooled_TN=("TN", "sum"),
            )
        )
        merged = self.metrics_pooled.merge(
            agg,
            on=["model", "horizon", "threshold", "nominal_budget_r"],
            how="inner",
            suffixes=("", "_re"),
        )
        self.assertEqual(len(merged), 36)
        for c in [
            "pooled_eligible_N",
            "pooled_eligible_events",
            "pooled_alarms",
            "pooled_TP",
            "pooled_FP",
            "pooled_FN",
            "pooled_TN",
        ]:
            self.assertTrue(bool((merged[c].astype(int) == merged[c + "_re"].astype(int)).all()))

    def test_18_suppression_accounting(self) -> None:
        s = self.suppression.copy()
        self.assertTrue(
            bool(
                (
                    s["candidate_alarms"].astype(int)
                    - s["issued_alarms"].astype(int)
                    == s["alarms_suppressed_by_quota"].astype(int)
                ).all()
            )
        )
        self.assertTrue(bool((s["unused_final_capacity"].astype(int) >= 0).all()))

    def test_19_comparison_file_properties(self) -> None:
        c = self.comparison.copy()
        self.assertEqual(len(c), 144)
        self.assertTrue(bool((c["quota_enforced_exceeds_capacity"].astype(str).str.lower() == "false").all()))
        # At least one unconstrained case should exceed nominal capacity, otherwise comparison would be degenerate.
        self.assertTrue(bool((c["unconstrained_exceeds_capacity"].astype(str).str.lower() == "true").any()))

    def test_20_information_isolation_passes(self) -> None:
        self.assertTrue(bool(self.info_audit["label_permutation_pass"]))
        self.assertTrue(bool(self.info_audit["future_score_perturbation_pass"]))
        self.assertTrue(bool(self.info_audit["current_score_locality_pass"]))
        self.assertTrue(bool(self.info_audit["no_outcome_dependency_pass"]))

    def test_21_source_package_remains_unchanged(self) -> None:
        observed = sha256_file(SOURCE_DECISIONS_PATH)
        expected = expected_checksum_for_filename(SOURCE_CHECKSUMS_PATH, "sequential_h5_alarm_decisions.csv")
        self.assertEqual(observed, expected)

    def test_22_required_comparison_statements_in_report(self) -> None:
        txt = self.report
        self.assertIn("The unconstrained policy is causally valid but may exceed capacity.", txt)
        self.assertIn("The quota-enforced policy guarantees the cumulative alarm cap.", txt)
        self.assertIn("Both remain historical simulations rather than prospective field validation.", txt)

    def test_23_capacity_audit_passes_all_groups(self) -> None:
        self.assertTrue(bool((self.capacity_audit["capacity_audit_pass"].astype(str).str.lower() == "true").all()))

    def test_24_deterministic_regeneration_stable(self) -> None:
        stable = [
            INPUT_VERIFICATION_PATH,
            DECISIONS_PATH,
            METRICS_BY_FOLD_PATH,
            METRICS_POOLED_PATH,
            CAPACITY_AUDIT_PATH,
            SUPPRESSION_AUDIT_PATH,
            COMPARISON_PATH,
            INFO_AUDIT_PATH,
            REPORT_PATH,
            CHECKSUMS_PATH,
            BUILD_SCRIPT,
            Path(__file__).resolve(),
        ]
        before = deterministic_snapshot(stable)
        run_build_script()
        after = deterministic_snapshot(stable)
        self.assertEqual(before, after)

    def test_25_branch_unchanged(self) -> None:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        self.assertEqual(branch, EXPECTED_BRANCH)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestQuotaEnforcedH5AlarmPolicy)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    update_manifest_test_result(
        success=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
    )

    # Re-run build once so report/manifest/checksums incorporate test_result deterministically.
    run_build_script()

    sys.exit(0 if result.wasSuccessful() else 1)
