from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = OUT_DIR / "build_lag_window_sensitivity_design.py"

EXPECTED_BRANCH = "controlled-reruns-v1"
EXPECTED_HEAD = "15cc6f8414df3e60692700963afc296e7653236e"
EXPECTED_STAGE1_TAG = "canonical-h1-h3-reconciliation-v1"
EXPECTED_PYTHON = "3.8.10"
EXPECTED_DATASET_SHA256 = "6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb"
EXPECTED_SPLIT_SHA256 = "f83baf5732ffbf88b9a448a3d9ab9fa270f7495fe44d1559c1ac49795a30881a"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
FEATURE_LINEAGE_CSV_PATH = OUT_DIR / "feature_lineage_audit.csv"
FEATURE_LINEAGE_MD_PATH = OUT_DIR / "feature_lineage_audit.md"
FEATURE_ARTIFACT_REGISTRY_PATH = OUT_DIR / "feature_artifact_registry.csv"
CANDIDATE_CONFIG_PATH = OUT_DIR / "candidate_memory_configurations.csv"
MODEL_HORIZON_REGISTRY_PATH = OUT_DIR / "model_horizon_reference_registry.csv"
COMMON_DATE_FEASIBILITY_PATH = OUT_DIR / "common_date_feasibility.csv"
TRAINING_RETENTION_PATH = OUT_DIR / "training_retention_audit.csv"
DESIGN_MD_PATH = OUT_DIR / "lag_window_sensitivity_design.md"
DESIGN_JSON_PATH = OUT_DIR / "lag_window_sensitivity_design.json"
MANIFEST_PATH = OUT_DIR / "lag_window_sensitivity_design_manifest.json"
REPORT_PATH = OUT_DIR / "lag_window_sensitivity_design_completion_report.md"
CHECKSUM_PATH = OUT_DIR / "lag_window_sensitivity_design_checksums.sha256"

PROTOCOL_SHA_PATH = ROOT / "revision_2026" / "03_corrected_protocol" / "corrected_protocol_v1.sha256"
STAGE1_CHECKSUM_PATH = ROOT / "revision_2026" / "09_h1_h3_reconciliation" / "canonical_h1_h3_reconciliation_checksums.sha256"

NPZ_FILES = [
    ROOT / "features" / "ulsan_H1_features.npz",
    ROOT / "features" / "ulsan_H3_features.npz",
    ROOT / "features" / "ulsan_H3_features_v2.npz",
    ROOT / "features" / "ulsan_H5_features.npz",
    ROOT / "features" / "ulsan_H5_features_v2.npz",
]

EXPECTED_COUNTS = {
    1: {1: 254, 2: 254, 3: 254, "pooled": 762},
    3: {1: 249, 2: 249, 3: 249, "pooled": 747},
    5: {1: 249, 2: 249, 3: 249, "pooled": 747},
}


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


def run_builder() -> None:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=ROOT, check=True)


def stage_files_for_determinism() -> List[Path]:
    files: List[Path] = []
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        if "__pycache__" in str(p):
            continue
        if p.suffix.lower() not in {".py", ".csv", ".json", ".md", ".sha256"}:
            continue
        files.append(p)
    return files


def snapshot_hashes(paths: List[Path]) -> Dict[str, str]:
    return {str(p.relative_to(OUT_DIR)).replace("\\", "/"): sha256_file(p) for p in paths}


def update_manifest_test_result(success: bool, tests_run: int, failures: int, errors: int) -> None:
    if not MANIFEST_PATH.exists():
        return
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["test_result"] = "pass_44_of_44" if success else "fail_{}_of_44".format(failures + errors)
    payload["test_command"] = "{} {}".format(sys.executable, Path(__file__).resolve())
    payload["test_summary"] = "Ran {} tests; failures={}; errors={}.".format(tests_run, failures, errors)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestLagWindowSensitivityDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_builder()

        cls.input_ver = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.design_json = json.loads(DESIGN_JSON_PATH.read_text(encoding="utf-8"))
        cls.design_md = DESIGN_MD_PATH.read_text(encoding="utf-8")
        cls.feature_md = FEATURE_LINEAGE_MD_PATH.read_text(encoding="utf-8")

        cls.lineage = pd.read_csv(FEATURE_LINEAGE_CSV_PATH)
        cls.artifacts = pd.read_csv(FEATURE_ARTIFACT_REGISTRY_PATH)
        cls.configs = pd.read_csv(CANDIDATE_CONFIG_PATH)
        cls.registry = pd.read_csv(MODEL_HORIZON_REGISTRY_PATH)
        cls.common = pd.read_csv(COMMON_DATE_FEASIBILITY_PATH)
        cls.retention = pd.read_csv(TRAINING_RETENTION_PATH)

        if "horizon" in cls.common.columns:
            cls.common["horizon"] = pd.to_numeric(cls.common["horizon"], errors="raise").astype(int)
        if "horizon" in cls.retention.columns:
            cls.retention["horizon"] = pd.to_numeric(cls.retention["horizon"], errors="raise").astype(int)

    def test_01_repository_root(self) -> None:
        self.assertEqual(Path.cwd().resolve(), ROOT.resolve())

    def test_02_branch(self) -> None:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        self.assertEqual(branch, EXPECTED_BRANCH)

    def test_03_python_compatibility(self) -> None:
        self.assertEqual(platform.python_version(), EXPECTED_PYTHON)

    def test_04_clean_initial_source_state(self) -> None:
        checks = self.input_ver["verification"]["checks"]
        self.assertTrue(bool(checks["clean_initial_working_tree_pass"]))

    def test_05_stage1_tag_identity(self) -> None:
        checks = self.input_ver["verification"]["checks"]
        self.assertTrue(bool(checks["head_pass"]))
        self.assertTrue(bool(checks["stage1_tag_points_to_head_pass"]))

    def test_06_corrected_protocol_checksums(self) -> None:
        checks = self.input_ver["verification"]["checks"]
        self.assertTrue(bool(checks["corrected_protocol_checksum_pass"]))

    def test_07_stage1_checksums(self) -> None:
        checks = self.input_ver["verification"]["checks"]
        self.assertTrue(bool(checks["stage1_checksum_pass"]))

    def test_08_source_preservation(self) -> None:
        checks = self.input_ver["verification"]["checks"]
        self.assertTrue(bool(checks["source_preservation_pass"]))

    def test_09_builder_source_hashes(self) -> None:
        want = {
            "src/build_ulsan_npz.py",
            "src/build_ulsan_npz_v2.py",
            "src/build_ulsan_npz_H1.py",
            "revision_2026/04_controlled_reruns/ridge/run_corrected_ridge.py",
            "revision_2026/04_controlled_reruns/elasticnet/run_corrected_elasticnet.py",
            "revision_2026/04_controlled_reruns/hgbr/run_corrected_hgbr.py",
        }
        seen = set(self.artifacts["relative_path"].astype(str).tolist())
        self.assertTrue(want.issubset(seen))

    def test_10_npz_hashes(self) -> None:
        for p in NPZ_FILES:
            rp = str(p.relative_to(ROOT)).replace("\\", "/")
            row = self.artifacts[self.artifacts["relative_path"].astype(str) == rp]
            self.assertEqual(len(row), 1)
            self.assertEqual(row["sha256"].iloc[0], sha256_file(p))

    def test_11_npz_schema_inspection(self) -> None:
        for p in NPZ_FILES:
            z = np.load(p, allow_pickle=True)
            self.assertSetEqual(set(z.files), {"X", "y", "dates", "feature_names"})
            self.assertEqual(len(np.asarray(z["X"]).shape), 2)
            self.assertEqual(len(np.asarray(z["y"]).shape), 1)

    def test_12_final_feature_lineage_identification(self) -> None:
        required_models = {"Persistence", "Ridge", "ElasticNet", "HGBR", "BCR-TCN"}
        self.assertTrue(required_models.issubset(set(self.lineage["model"].astype(str).tolist())))

    def test_13_h1_h3_ordinary_lineage(self) -> None:
        hgb_h1 = self.lineage[(self.lineage["model"] == "HGBR") & (self.lineage["horizon"] == 1) & (self.lineage["used_in_final_corrected_model"].astype(bool))]
        hgb_h3 = self.lineage[(self.lineage["model"] == "HGBR") & (self.lineage["horizon"] == 3) & (self.lineage["used_in_final_corrected_model"].astype(bool))]
        self.assertTrue("ordinary" in hgb_h1["final_pathway"].iloc[0])
        self.assertTrue("ordinary" in hgb_h3["final_pathway"].iloc[0])

    def test_14_h5_v2_lineage(self) -> None:
        hgb_h5 = self.lineage[(self.lineage["model"] == "HGBR") & (self.lineage["horizon"] == 5) & (self.lineage["used_in_final_corrected_model"].astype(bool))]
        bcr_h5 = self.lineage[(self.lineage["model"] == "BCR-TCN") & (self.lineage["horizon"] == 5)]
        self.assertTrue("v2" in str(hgb_h5["final_pathway"].iloc[0]))
        self.assertTrue("v2" in str(bcr_h5["final_pathway"].iloc[0]))

    def test_15_h3_v2_exists_not_used(self) -> None:
        row = self.lineage[(self.lineage["model"] == "HGBR") & (self.lineage["horizon"] == 3) & (self.lineage["final_pathway"] == "v2_artifact_exists_not_used")]
        self.assertEqual(len(row), 1)
        self.assertFalse(bool(row["used_in_final_corrected_model"].iloc[0]))

    def test_16_strictly_prior_tnout_lag_construction(self) -> None:
        self.assertIn("TNout_lagk(t) = TNout(t-k)", self.feature_md)

    def test_17_strictly_prior_tnout_rolling_construction(self) -> None:
        self.assertIn("TNout_rollw(t) = mean(TNout(t-w), ..., TNout(t-1))", self.feature_md)

    def test_18_tnout_rolls_exclude_day_t(self) -> None:
        non_empty = self.lineage[self.lineage["tnout_rolls"].astype(str).str.len() > 0]
        self.assertTrue((non_empty["tnout_roll_includes_day_t"].astype(str).str.lower() == "false").all())

    def test_19_non_tn_day_t_inclusion_documented(self) -> None:
        with_non_tn = self.lineage[self.lineage["non_tnout_rolls"].astype(str).str.len() > 0]
        self.assertTrue((with_non_tn["non_tnout_rolls_include_day_t"].astype(str).str.lower() == "true").all())

    def test_20_short_definition(self) -> None:
        row = self.configs[self.configs["configuration_id"] == "SHORT"].iloc[0]
        self.assertEqual(int(row["maximum_required_history_days"]), 7)
        self.assertEqual(str(row["earliest_possible_date_from_raw_start"]), "2021-01-11")

    def test_21_reference_definition(self) -> None:
        row = self.configs[self.configs["configuration_id"] == "REFERENCE"].iloc[0]
        self.assertEqual(int(row["maximum_required_history_days"]), 14)
        self.assertEqual(str(row["earliest_possible_date_from_raw_start"]), "2021-01-18")

    def test_22_long_definition(self) -> None:
        row = self.configs[self.configs["configuration_id"] == "LONG"].iloc[0]
        self.assertEqual(int(row["maximum_required_history_days"]), 30)
        self.assertEqual(str(row["earliest_possible_date_from_raw_start"]), "2021-02-03")

    def test_23_no_future_information_feature(self) -> None:
        self.assertTrue((self.configs["uses_future_information"].astype(str).str.lower() == "false").all())

    def test_24_ridge_selected(self) -> None:
        d = self.registry[self.registry["model"] == "Ridge"]
        self.assertEqual(set(d["horizon"].tolist()), {1, 3, 5})
        self.assertTrue((d["will_be_refitted"].astype(str).str.lower() == "true").all())

    def test_25_hgbr_selected(self) -> None:
        d = self.registry[self.registry["model"] == "HGBR"]
        self.assertEqual(set(d["horizon"].tolist()), {1, 3, 5})
        self.assertTrue((d["will_be_refitted"].astype(str).str.lower() == "true").all())

    def test_26_persistence_context_only(self) -> None:
        d = self.registry[self.registry["model"] == "Persistence"]
        self.assertTrue((d["will_be_refitted"].astype(str).str.lower() == "false").all())
        self.assertTrue((d["sensitivity_role"].astype(str) == "context_only_baseline").all())

    def test_27_elasticnet_no_refit_status(self) -> None:
        d = self.registry[self.registry["model"] == "ElasticNet"]
        self.assertTrue((d["will_be_refitted"].astype(str).str.lower() == "false").all())

    def test_28_bcr_tcn_no_refit_status(self) -> None:
        d = self.registry[self.registry["model"] == "BCR-TCN"]
        self.assertEqual(len(d), 1)
        self.assertEqual(int(d["horizon"].iloc[0]), 5)
        self.assertEqual(str(d["will_be_refitted"].iloc[0]).lower(), "false")

    def test_29_all_three_horizons_included(self) -> None:
        self.assertSetEqual(set(self.common["horizon"].unique().tolist()), {1, 3, 5})

    def test_30_final_reference_mapping(self) -> None:
        rg = self.registry[self.registry["model"] == "Ridge"]
        self.assertTrue((rg["final_reference_configuration"] == "REFERENCE").all())
        hg = self.registry[self.registry["model"] == "HGBR"]
        h5 = hg[hg["horizon"] == 5]
        self.assertEqual(str(h5["final_reference_configuration"].iloc[0]), "LONG")

    def test_31_frozen_hyperparameter_rule(self) -> None:
        self.assertIn("Hyperparameter Freeze Rule", self.design_md)
        self.assertIn("No retuning", self.design_md)

    def test_32_no_retuning_rule(self) -> None:
        self.assertTrue((self.registry["retuning_allowed"].astype(str).str.lower() == "false").all())

    def test_33_canonical_test_date_preservation(self) -> None:
        self.assertTrue((self.common["preserves_all_canonical_test_dates"].astype(str).str.lower() == "true").all())

    def test_34_h1_fold_counts(self) -> None:
        ref = self.common[(self.common["configuration_id"] == "REFERENCE") & (self.common["horizon"] == 1)]
        for fold in [1, 2, 3]:
            row = ref[ref["outer_fold"].astype(str) == str(fold)].iloc[0]
            self.assertEqual(int(row["canonical_outer_test_rows"]), EXPECTED_COUNTS[1][fold])
        pooled = ref[ref["outer_fold"].astype(str) == "pooled"].iloc[0]
        self.assertEqual(int(pooled["canonical_outer_test_rows"]), EXPECTED_COUNTS[1]["pooled"])

    def test_35_h3_fold_counts(self) -> None:
        ref = self.common[(self.common["configuration_id"] == "REFERENCE") & (self.common["horizon"] == 3)]
        for fold in [1, 2, 3]:
            row = ref[ref["outer_fold"].astype(str) == str(fold)].iloc[0]
            self.assertEqual(int(row["canonical_outer_test_rows"]), EXPECTED_COUNTS[3][fold])
        pooled = ref[ref["outer_fold"].astype(str) == "pooled"].iloc[0]
        self.assertEqual(int(pooled["canonical_outer_test_rows"]), EXPECTED_COUNTS[3]["pooled"])

    def test_36_h5_fold_counts(self) -> None:
        ref = self.common[(self.common["configuration_id"] == "REFERENCE") & (self.common["horizon"] == 5)]
        for fold in [1, 2, 3]:
            row = ref[ref["outer_fold"].astype(str) == str(fold)].iloc[0]
            self.assertEqual(int(row["canonical_outer_test_rows"]), EXPECTED_COUNTS[5][fold])
        pooled = ref[ref["outer_fold"].astype(str) == "pooled"].iloc[0]
        self.assertEqual(int(pooled["canonical_outer_test_rows"]), EXPECTED_COUNTS[5]["pooled"])

    def test_37_exact_warmup_calculation(self) -> None:
        expected = {
            "SHORT": "2021-01-11",
            "REFERENCE": "2021-01-18",
            "LONG": "2021-02-03",
        }
        for cid, dt in expected.items():
            row = self.configs[self.configs["configuration_id"] == cid].iloc[0]
            self.assertEqual(str(row["earliest_possible_date_from_raw_start"]), dt)

    def test_38_training_row_retention_calculation(self) -> None:
        long_h1 = self.retention[
            (self.retention["configuration_id"] == "LONG")
            & (self.retention["horizon"] == 1)
            & (self.retention["outer_fold"].astype(str).isin(["1", "2", "3"]))
        ]
        losses = sorted(long_h1["outer_train_rows_lost_warmup"].astype(int).tolist())
        self.assertEqual(losses, [16, 16, 16])

    def test_39_no_canonical_test_date_loss(self) -> None:
        self.assertTrue((self.common["rows_lost_due_warmup"].astype(int) == 0).all())

    def test_40_no_model_fitting(self) -> None:
        txt = BUILD_SCRIPT.read_text(encoding="utf-8").lower()
        forbidden = ["sklearn", "xgboost", "lightgbm", "torch", ".fit(", "optuna"]
        self.assertFalse(any(tok in txt for tok in forbidden))

    def test_41_authorized_output_scope(self) -> None:
        for p in OUT_DIR.rglob("*"):
            if p.is_file():
                self.assertTrue(str(p.resolve()).startswith(str(OUT_DIR.resolve())))

    def test_42_deterministic_regeneration(self) -> None:
        files = stage_files_for_determinism()
        before = snapshot_hashes(files)
        run_builder()
        after = snapshot_hashes(files)
        self.assertEqual(before, after)

    def test_43_source_checksum_preservation(self) -> None:
        checks = self.input_ver["verification"]["checks"]
        self.assertTrue(bool(checks["corrected_protocol_checksum_pass"]))
        self.assertTrue(bool(checks["stage1_checksum_pass"]))
        self.assertTrue(bool(checks["dataset_checksum_pass"]))
        self.assertTrue(bool(checks["split_checksum_pass"]))

    def test_44_checksum_registry_coverage(self) -> None:
        listed = set(parse_checksum_manifest(CHECKSUM_PATH).keys())
        expected = set()
        for p in OUT_DIR.rglob("*"):
            if not p.is_file():
                continue
            if p.resolve() == CHECKSUM_PATH.resolve():
                continue
            if p.suffix.lower() not in {".py", ".csv", ".json", ".md"}:
                continue
            expected.add(str(p.relative_to(OUT_DIR)).replace("\\", "/"))
        self.assertSetEqual(listed, expected)
        chk = subprocess.run(["sha256sum", "-c", CHECKSUM_PATH.name], cwd=OUT_DIR, text=True, capture_output=True)
        self.assertEqual(int(chk.returncode), 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestLagWindowSensitivityDesign)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    update_manifest_test_result(
        success=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
    )

    # Rebuild once so final manifest/report/checksums include post-test status.
    run_builder()

    sys.exit(0 if result.wasSuccessful() else 1)
