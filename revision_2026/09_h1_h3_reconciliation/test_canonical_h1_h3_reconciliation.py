from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = OUT_DIR / "build_canonical_h1_h3_reconciliation.py"

EXPECTED_BRANCH = "controlled-reruns-v1"
EXPECTED_PYTHON = "3.8.10"
EXPECTED_DATASET_SHA256 = "6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb"
EXPECTED_SPLIT_SHA256 = "f83baf5732ffbf88b9a448a3d9ab9fa270f7495fe44d1559c1ac49795a30881a"
EXPECTED_MODELS = ["Persistence", "Ridge", "ElasticNet", "HGBR"]
EXPECTED_HORIZONS = [1, 3]
KEY_COLS = ["horizon", "outer_fold", "feature_date", "target_date"]
TOL = 1e-12
MASE_EPS = 1e-9
MASE_METHOD = "mean_absolute_scaled_error_over_pooled_rows_using_fold_local_train_denominator"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
DATASET_LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"
LOCKED_SHA_PATH = PROTOCOL_DIR / "locked_protocol_v1" / "corrected_protocol_v1.sha256"

SOURCE_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
MODEL_SLUG = {
    "Persistence": "persistence",
    "Ridge": "ridge",
    "ElasticNet": "elasticnet",
    "HGBR": "hgbr",
}
MODEL_EXPECTED_TEST_RESULT = {
    "Persistence": "pass_12_of_12",
    "Ridge": "pass_15_of_15",
    "ElasticNet": "pass_15_of_15",
    "HGBR": "pass_20_of_20",
}
EXPECTED_EVENT_COUNTS = {
    1: {"events_tau15": 141, "events_tau16": 74, "events_tau17": 30},
    3: {"events_tau15": 134, "events_tau16": 71, "events_tau17": 29},
}

H1_DIR = OUT_DIR / "h1"
H3_DIR = OUT_DIR / "h3"

H1_LONG = H1_DIR / "canonical_h1_predictions_long.csv"
H1_WIDE = H1_DIR / "canonical_h1_predictions_wide.csv"
H1_KEY_AUDIT = H1_DIR / "canonical_h1_key_audit.csv"
H1_EVENT_PREV = H1_DIR / "canonical_h1_event_prevalence.csv"
H1_METRICS_FOLD = H1_DIR / "canonical_h1_point_metrics_by_fold.csv"
H1_METRICS_POOL = H1_DIR / "canonical_h1_point_metrics_pooled.csv"
H1_RECON = H1_DIR / "h1_controlled_rerun_reconciliation.csv"

H3_LONG = H3_DIR / "canonical_h3_predictions_long.csv"
H3_WIDE = H3_DIR / "canonical_h3_predictions_wide.csv"
H3_KEY_AUDIT = H3_DIR / "canonical_h3_key_audit.csv"
H3_EVENT_PREV = H3_DIR / "canonical_h3_event_prevalence.csv"
H3_METRICS_FOLD = H3_DIR / "canonical_h3_point_metrics_by_fold.csv"
H3_METRICS_POOL = H3_DIR / "canonical_h3_point_metrics_pooled.csv"
H3_RECON = H3_DIR / "h3_controlled_rerun_reconciliation.csv"

TABLE1_PATH = OUT_DIR / "canonical_h1_h3_table1_reconciliation.csv"
TABLE2_PATH = OUT_DIR / "canonical_h1_h3_point_table2_source.csv"
SOURCE_REGISTRY_PATH = OUT_DIR / "canonical_h1_h3_source_registry.csv"
INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
MANIFEST_PATH = OUT_DIR / "canonical_h1_h3_reconciliation_manifest.json"
REPORT_PATH = OUT_DIR / "canonical_h1_h3_reconciliation_completion_report.md"
CHECKSUM_PATH = OUT_DIR / "canonical_h1_h3_reconciliation_checksums.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise").dt.strftime("%Y-%m-%d")


def run_build_script() -> None:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=ROOT, check=True)


def parse_checksum_manifest(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        out[parts[-1]] = parts[0]
    return out


def key_tuples(df: pd.DataFrame) -> set:
    return set(tuple(r) for r in df[KEY_COLS].itertuples(index=False, name=None))


def model_file_paths(model: str) -> Dict[str, Path]:
    slug = MODEL_SLUG[model]
    d = SOURCE_DIR / slug
    return {
        "prediction": d / "{}_predictions.csv".format(slug),
        "metrics_by_fold": d / "{}_metrics_by_fold.csv".format(slug),
        "metrics_pooled": d / "{}_metrics_pooled.csv".format(slug),
        "manifest": d / "{}_run_manifest.json".format(slug),
        "completion_report": d / "{}_completion_report.md".format(slug),
        "event_prevalence": d / "{}_event_prevalence.csv".format(slug),
    }


def compute_denoms_from_split_and_dataset(split_df: pd.DataFrame, dataset_df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    split = split_df.copy()
    split["horizon"] = pd.to_numeric(split["horizon"], errors="raise").astype(int)
    split["outer_fold"] = pd.to_numeric(split["outer_fold"], errors="raise").astype(int)
    split["original_row_index"] = pd.to_numeric(split["original_row_index"], errors="raise").astype(int)
    split["purged_outer_boundary"] = split["purged_outer_boundary"].astype(str).str.lower().map({"true": True, "false": False})
    split["target_date"] = pd.to_datetime(split["target_date"], errors="raise")

    vals = dataset_df["TNout"].astype(float).to_numpy()
    denoms: Dict[Tuple[int, int], float] = {}

    for h in [1, 3]:
        for fold in [1, 2, 3]:
            s = split[
                (split["horizon"] == int(h))
                & (split["outer_fold"] == int(fold))
                & (split["outer_role"].astype(str) == "outer_train")
                & (~split["purged_outer_boundary"].astype(bool))
            ].copy()
            s = s.sort_values(["target_date", "original_row_index"]).reset_index(drop=True)
            y = []
            for r in s.itertuples(index=False):
                idx = int(r.original_row_index) + int(h)
                y.append(float(vals[idx]))
            denoms[(int(h), int(fold))] = float(pd.Series(y, dtype=float).diff().abs().dropna().mean())

    return denoms


def stage_deterministic_files() -> List[Path]:
    files: List[Path] = []
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".py", ".csv", ".json", ".md", ".sha256"}:
            continue
        if "__pycache__" in str(p):
            continue
        files.append(p)
    return files


def snapshot_hashes(paths: List[Path]) -> Dict[str, str]:
    return {str(p.relative_to(OUT_DIR)).replace("\\", "/"): sha256_file(p) for p in paths}


def update_manifest_test_result(success: bool, tests_run: int, failures: int, errors: int) -> None:
    if not MANIFEST_PATH.exists():
        return
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["test_result"] = "pass_56_of_56" if success else "fail_{}_of_56".format(failures + errors)
    payload["test_command"] = "{} {}".format(sys.executable, Path(__file__).resolve())
    payload["test_summary"] = "Ran {} tests; failures={}; errors={}.".format(tests_run, failures, errors)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestCanonicalH1H3Reconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_build_script()

        cls.input_ver = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.report = REPORT_PATH.read_text(encoding="utf-8")
        cls.table1 = pd.read_csv(TABLE1_PATH)
        cls.table2 = pd.read_csv(TABLE2_PATH)
        cls.source_registry = pd.read_csv(SOURCE_REGISTRY_PATH)

        cls.split = pd.read_csv(SPLIT_PATH)
        cls.split["horizon"] = pd.to_numeric(cls.split["horizon"], errors="raise").astype(int)
        cls.split["outer_fold"] = pd.to_numeric(cls.split["outer_fold"], errors="raise").astype(int)
        cls.split["feature_date"] = normalize_date(cls.split["feature_date"])
        cls.split["target_date"] = normalize_date(cls.split["target_date"])
        cls.split["purged_outer_boundary"] = cls.split["purged_outer_boundary"].astype(str).str.lower().map({"true": True, "false": False})

        cls.h1_split = cls.split[(cls.split["horizon"] == 1) & (cls.split["outer_role"] == "outer_test") & (~cls.split["purged_outer_boundary"].astype(bool))][KEY_COLS].drop_duplicates()
        cls.h3_split = cls.split[(cls.split["horizon"] == 3) & (cls.split["outer_role"] == "outer_test") & (~cls.split["purged_outer_boundary"].astype(bool))][KEY_COLS].drop_duplicates()

        cls.h1_long = pd.read_csv(H1_LONG)
        cls.h1_wide = pd.read_csv(H1_WIDE)
        cls.h1_key_audit = pd.read_csv(H1_KEY_AUDIT)
        cls.h1_event_prev = pd.read_csv(H1_EVENT_PREV)
        cls.h1_metrics_fold = pd.read_csv(H1_METRICS_FOLD)
        cls.h1_metrics_pool = pd.read_csv(H1_METRICS_POOL)
        cls.h1_recon = pd.read_csv(H1_RECON)

        cls.h3_long = pd.read_csv(H3_LONG)
        cls.h3_wide = pd.read_csv(H3_WIDE)
        cls.h3_key_audit = pd.read_csv(H3_KEY_AUDIT)
        cls.h3_event_prev = pd.read_csv(H3_EVENT_PREV)
        cls.h3_metrics_fold = pd.read_csv(H3_METRICS_FOLD)
        cls.h3_metrics_pool = pd.read_csv(H3_METRICS_POOL)
        cls.h3_recon = pd.read_csv(H3_RECON)

        for df in [cls.h1_long, cls.h3_long, cls.h1_wide, cls.h3_wide]:
            df["horizon"] = pd.to_numeric(df["horizon"], errors="raise").astype(int)
            df["outer_fold"] = pd.to_numeric(df["outer_fold"], errors="raise").astype(int)
            df["feature_date"] = normalize_date(df["feature_date"])
            df["target_date"] = normalize_date(df["target_date"])

        cls.source_predictions = {}
        cls.source_by_fold = {}
        cls.source_pooled = {}
        cls.source_manifests = {}
        for model in EXPECTED_MODELS:
            paths = model_file_paths(model)
            cls.source_predictions[model] = pd.read_csv(paths["prediction"])
            cls.source_predictions[model]["horizon"] = pd.to_numeric(cls.source_predictions[model]["horizon"], errors="raise").astype(int)
            cls.source_predictions[model]["outer_fold"] = pd.to_numeric(cls.source_predictions[model]["outer_fold"], errors="raise").astype(int)
            cls.source_predictions[model]["feature_date"] = normalize_date(cls.source_predictions[model]["feature_date"])
            cls.source_predictions[model]["target_date"] = normalize_date(cls.source_predictions[model]["target_date"])
            cls.source_by_fold[model] = pd.read_csv(paths["metrics_by_fold"])
            cls.source_pooled[model] = pd.read_csv(paths["metrics_pooled"])
            cls.source_manifests[model] = json.loads(paths["manifest"].read_text(encoding="utf-8"))

        lock = json.loads(DATASET_LOCK_PATH.read_text(encoding="utf-8"))
        ds = pd.read_csv(Path(lock["absolute_path"]))
        ds = ds.drop(columns=[c for c in ds.columns if str(c).startswith("Unnamed")], errors="ignore")
        ds["Date"] = pd.to_datetime(ds["Date"], errors="raise")
        cls.dataset = ds.sort_values("Date").reset_index(drop=True)

    def test_01_correct_working_directory(self) -> None:
        self.assertEqual(Path.cwd().resolve(), ROOT.resolve())

    def test_02_correct_branch(self) -> None:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        self.assertEqual(branch, EXPECTED_BRANCH)

    def test_03_python_compatibility_3810(self) -> None:
        self.assertEqual(platform.python_version(), EXPECTED_PYTHON)

    def test_04_clean_or_authorized_only_working_tree_scope(self) -> None:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        paths = []
        for r in out.splitlines():
            if len(r) < 4:
                continue
            p = r[3:]
            if " -> " in p:
                p = p.split(" -> ")[-1]
            paths.append(p.strip())
        ok = (len(paths) == 0) or all(p == "revision_2026/09_h1_h3_reconciliation" or p.startswith("revision_2026/09_h1_h3_reconciliation/") for p in paths)
        self.assertTrue(ok)

    def test_05_dataset_checksum(self) -> None:
        lock = json.loads(DATASET_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sha256_file(Path(lock["absolute_path"])), EXPECTED_DATASET_SHA256)

    def test_06_split_checksum(self) -> None:
        self.assertEqual(sha256_file(SPLIT_PATH), EXPECTED_SPLIT_SHA256)

    def test_07_protocol_checksum_registries(self) -> None:
        checks = self.input_ver["verification"]["checks"]
        self.assertTrue(bool(checks["root_protocol_sha256sum_pass"]))
        self.assertTrue(bool(checks["locked_protocol_effective_pass"]))

    def test_08_root_vs_locked_protocol_identity(self) -> None:
        rows = self.input_ver["verification"]["protocol_root_locked_identity"]
        shared = [r for r in rows if r["root_exists"] and r["locked_exists"]]
        self.assertTrue(all(bool(r["identical"]) for r in shared))

    def test_09_source_manifest_checksum_identity(self) -> None:
        for model in EXPECTED_MODELS:
            manifest = self.source_manifests[model]
            mp = model_file_paths(model)
            self.assertEqual(sha256_file(mp["prediction"]), manifest["prediction_sha256"])

    def test_10_source_prediction_checksum_identity(self) -> None:
        for model in EXPECTED_MODELS:
            info = self.input_ver["verification"]["source_model_info"][model]
            self.assertTrue(bool(info["prediction_file_identity_pass"]))

    def test_11_source_metric_checksum_identity(self) -> None:
        for model in EXPECTED_MODELS:
            checks = self.input_ver["verification"]["source_model_info"][model]["metrics_file_checks"]
            self.assertTrue(all(bool(v["pass"]) for v in checks.values()))

    def test_12_source_test_decisions(self) -> None:
        for model in EXPECTED_MODELS:
            tr = self.input_ver["verification"]["source_model_info"][model]["test_result"]
            self.assertEqual(tr, MODEL_EXPECTED_TEST_RESULT[model])

    def test_13_source_completion_decisions(self) -> None:
        for model in EXPECTED_MODELS:
            self.assertEqual(self.input_ver["verification"]["source_model_info"][model]["completion_decision"], "A")

    def test_14_hgbr_issue_registry_resolved(self) -> None:
        issue = self.input_ver["verification"]["hgbr_issue_registry"]
        self.assertTrue(bool(issue["pass"]))

    def test_15_source_freeze_tag_identity(self) -> None:
        tags = self.input_ver["verification"]["local_tag_commits"]
        self.assertEqual(set(tags.keys()), {"corrected-protocol-v1", "persistence-corrected-v1", "ridge-corrected-v1", "elasticnet-corrected-v1", "hgbr-corrected-v1"})
        self.assertTrue(all(str(v) != "MISSING" for v in tags.values()))

    def test_16_source_directory_preservation(self) -> None:
        pres = self.input_ver["verification"]["source_directory_preservation"]
        self.assertTrue(all(bool(v["pass"]) for v in pres.values()))

    def test_17_authorized_models_only(self) -> None:
        models = set(self.h1_long["model"].unique().tolist()) | set(self.h3_long["model"].unique().tolist())
        self.assertSetEqual(models, set(EXPECTED_MODELS))

    def test_18_authorized_horizons_only(self) -> None:
        hs = set(self.h1_long["horizon"].unique().tolist()) | set(self.h3_long["horizon"].unique().tolist())
        self.assertSetEqual(hs, {1, 3})

    def test_19_no_h5_rows(self) -> None:
        self.assertFalse(bool(((self.h1_long["horizon"] == 5) | (self.h3_long["horizon"] == 5)).any()))

    def test_20_no_bcr_tcn(self) -> None:
        self.assertFalse(bool(self.h1_long["model"].astype(str).str.contains("BCR", case=False, regex=False).any()))
        self.assertFalse(bool(self.h3_long["model"].astype(str).str.contains("BCR", case=False, regex=False).any()))

    def test_21_no_tcn(self) -> None:
        self.assertFalse(bool(self.h1_long["model"].astype(str).str.contains("TCN", case=False).any()))
        self.assertFalse(bool(self.h3_long["model"].astype(str).str.contains("TCN", case=False).any()))

    def test_22_no_point_blend(self) -> None:
        self.assertFalse(bool(self.h1_long["model"].astype(str).str.contains("point", case=False).any()))
        self.assertFalse(bool(self.h3_long["model"].astype(str).str.contains("point", case=False).any()))

    def test_23_no_hybridrank(self) -> None:
        self.assertFalse(bool(self.h1_long["model"].astype(str).str.contains("hybrid", case=False).any()))
        self.assertFalse(bool(self.h3_long["model"].astype(str).str.contains("hybrid", case=False).any()))

    def test_24_no_fixed_rank_ensemble(self) -> None:
        self.assertFalse(bool(self.h1_long["model"].astype(str).str.contains("fixed", case=False).any()))
        self.assertFalse(bool(self.h3_long["model"].astype(str).str.contains("fixed", case=False).any()))

    def test_25_no_alarm_columns_or_metrics(self) -> None:
        cols = [c.lower() for c in self.h1_long.columns.tolist() + self.h3_long.columns.tolist()]
        forbidden_tokens = ["alarm", "precision", "recall", "false_positive_rate", "top_k"]
        self.assertFalse(any(any(tok in c for tok in forbidden_tokens) for c in cols))

    def test_26_h1_n_762(self) -> None:
        self.assertEqual(int(len(self.h1_wide)), 762)

    def test_27_h3_n_747(self) -> None:
        self.assertEqual(int(len(self.h3_wide)), 747)

    def test_28_exact_fold_counts(self) -> None:
        self.assertEqual(self.h1_wide.groupby("outer_fold").size().to_dict(), {1: 254, 2: 254, 3: 254})
        self.assertEqual(self.h3_wide.groupby("outer_fold").size().to_dict(), {1: 249, 2: 249, 3: 249})

    def test_29_long_table_row_counts(self) -> None:
        self.assertEqual(int(len(self.h1_long)), 3048)
        self.assertEqual(int(len(self.h3_long)), 2988)

    def test_30_wide_table_row_counts(self) -> None:
        self.assertEqual(int(len(self.h1_wide)), 762)
        self.assertEqual(int(len(self.h3_wide)), 747)

    def test_31_no_duplicate_canonical_keys(self) -> None:
        self.assertEqual(int(self.h1_long.duplicated(subset=["model"] + KEY_COLS).sum()), 0)
        self.assertEqual(int(self.h3_long.duplicated(subset=["model"] + KEY_COLS).sum()), 0)

    def test_32_identical_keys_across_models(self) -> None:
        for h_long, split_keys in [(self.h1_long, self.h1_split), (self.h3_long, self.h3_split)]:
            expected = key_tuples(split_keys)
            for m in EXPECTED_MODELS:
                d = h_long[h_long["model"] == m]
                self.assertSetEqual(key_tuples(d), expected)

    def test_33_identical_y_true_across_models(self) -> None:
        for h_long in [self.h1_long, self.h3_long]:
            ref = h_long[h_long["model"] == "Persistence"][KEY_COLS + ["y_true"]].rename(columns={"y_true": "y_ref"})
            for m in EXPECTED_MODELS:
                d = h_long[h_long["model"] == m][KEY_COLS + ["y_true"]]
                x = d.merge(ref, on=KEY_COLS, how="inner")
                self.assertEqual(len(x), len(ref))
                self.assertLessEqual(float((x["y_true"].astype(float) - x["y_ref"].astype(float)).abs().max()), TOL)

    def test_34_target_date_offset_equality(self) -> None:
        for h_long, h in [(self.h1_long, 1), (self.h3_long, 3)]:
            delta = (pd.to_datetime(h_long["target_date"]) - pd.to_datetime(h_long["feature_date"])).dt.days
            self.assertTrue(bool((delta == int(h)).all()))

    def test_35_source_prediction_row_identity(self) -> None:
        for model in EXPECTED_MODELS:
            src = self.source_predictions[model][self.source_predictions[model]["horizon"].isin([1, 3])].copy()
            can = pd.concat([self.h1_long, self.h3_long], axis=0)
            can = can[can["model"] == model][KEY_COLS + ["y_pred"]]
            m = src[KEY_COLS + ["y_pred"]].merge(can, on=KEY_COLS, how="inner", suffixes=("_src", "_can"))
            self.assertEqual(len(m), len(src))
            self.assertLessEqual(float((m["y_pred_src"].astype(float) - m["y_pred_can"].astype(float)).abs().max()), TOL)

    def test_36_finite_y_true_y_pred(self) -> None:
        for h_long in [self.h1_long, self.h3_long]:
            self.assertTrue(bool(np.isfinite(h_long["y_true"].astype(float).to_numpy()).all()))
            self.assertTrue(bool(np.isfinite(h_long["y_pred"].astype(float).to_numpy()).all()))

    def test_37_recomputed_event_label_equality(self) -> None:
        for h_long in [self.h1_long, self.h3_long]:
            self.assertTrue(bool(((h_long["y_true"].astype(float) >= 15).astype(int) == h_long["event_tau15"].astype(int)).all()))
            self.assertTrue(bool(((h_long["y_true"].astype(float) >= 16).astype(int) == h_long["event_tau16"].astype(int)).all()))
            self.assertTrue(bool(((h_long["y_true"].astype(float) >= 17).astype(int) == h_long["event_tau17"].astype(int)).all()))

    def test_38_expected_event_count_equality(self) -> None:
        for h, evdf in [(1, self.h1_event_prev), (3, self.h3_event_prev)]:
            pooled = evdf[evdf["outer_fold"].astype(str) == "pooled"].iloc[0]
            self.assertEqual(int(pooled["events_tau15"]), EXPECTED_EVENT_COUNTS[h]["events_tau15"])
            self.assertEqual(int(pooled["events_tau16"]), EXPECTED_EVENT_COUNTS[h]["events_tau16"])
            self.assertEqual(int(pooled["events_tau17"]), EXPECTED_EVENT_COUNTS[h]["events_tau17"])

    def test_39_date_range_reproduction(self) -> None:
        for h, wide, summary in [(1, self.h1_wide, self.h1_split), (3, self.h3_wide, self.h3_split)]:
            self.assertEqual(str(wide["feature_date"].min()), str(summary["feature_date"].min()))
            self.assertEqual(str(wide["feature_date"].max()), str(summary["feature_date"].max()))
            self.assertEqual(str(wide["target_date"].min()), str(summary["target_date"].min()))
            self.assertEqual(str(wide["target_date"].max()), str(summary["target_date"].max()))

    def test_40_fold_metric_reproduction(self) -> None:
        self.assertTrue(bool(self.h1_recon[self.h1_recon["outer_fold"].astype(str).isin(["1", "2", "3"])]["pass"].astype(bool).all()))
        self.assertTrue(bool(self.h3_recon[self.h3_recon["outer_fold"].astype(str).isin(["1", "2", "3"])]["pass"].astype(bool).all()))

    def test_41_pooled_mae_from_pooled_rows(self) -> None:
        for h_long, m_pool in [(self.h1_long, self.h1_metrics_pool), (self.h3_long, self.h3_metrics_pool)]:
            for model in EXPECTED_MODELS:
                d = h_long[h_long["model"] == model]
                mae = float(np.mean(np.abs(d["error"].astype(float).to_numpy())))
                obs = float(m_pool[m_pool["model"] == model]["MAE"].iloc[0])
                self.assertLessEqual(abs(mae - obs), TOL)

    def test_42_pooled_mse_from_pooled_rows(self) -> None:
        for h_long, m_pool in [(self.h1_long, self.h1_metrics_pool), (self.h3_long, self.h3_metrics_pool)]:
            for model in EXPECTED_MODELS:
                d = h_long[h_long["model"] == model]
                mse = float(np.mean(np.square(d["error"].astype(float).to_numpy())))
                obs = float(m_pool[m_pool["model"] == model]["MSE"].iloc[0])
                self.assertLessEqual(abs(mse - obs), TOL)

    def test_43_pooled_rmse_from_pooled_mse(self) -> None:
        for m_pool in [self.h1_metrics_pool, self.h3_metrics_pool]:
            rmse = np.sqrt(m_pool["MSE"].astype(float).to_numpy())
            obs = m_pool["RMSE"].astype(float).to_numpy()
            np.testing.assert_allclose(rmse, obs, rtol=0.0, atol=TOL)

    def test_44_mae_mse_terminology_protection(self) -> None:
        for h_long, m_fold in [(self.h1_long, self.h1_metrics_fold), (self.h3_long, self.h3_metrics_fold)]:
            for row in m_fold.itertuples(index=False):
                d = h_long[(h_long["model"] == row.model) & (h_long["outer_fold"].astype(int) == int(row.outer_fold))]
                mae = float(np.mean(np.abs(d["error"].astype(float).to_numpy())))
                mse = float(np.mean(np.square(d["error"].astype(float).to_numpy())))
                self.assertLessEqual(abs(float(row.MAE) - mae), TOL)
                self.assertLessEqual(abs(float(row.MSE) - mse), TOL)

    def test_45_fold_local_training_only_mase_denominator_reproduction(self) -> None:
        denoms = compute_denoms_from_split_and_dataset(self.split, self.dataset)
        src = pd.read_csv(model_file_paths("Persistence")["metrics_by_fold"])
        src = src[src["horizon"].isin([1, 3])].copy()
        for row in src.itertuples(index=False):
            exp = float(denoms[(int(row.horizon), int(row.outer_fold))])
            self.assertLessEqual(abs(exp - float(row.MASE_denom_train_only)), TOL)

    def test_46_pooled_row_level_mase_reproduction(self) -> None:
        denoms = compute_denoms_from_split_and_dataset(self.split, self.dataset)
        for h_long, m_pool, h in [(self.h1_long, self.h1_metrics_pool, 1), (self.h3_long, self.h3_metrics_pool, 3)]:
            for model in EXPECTED_MODELS:
                d = h_long[h_long["model"] == model].copy()
                scale = d["outer_fold"].astype(int).map(lambda f: denoms[(int(h), int(f))]).astype(float)
                mase = float(np.mean(np.abs(d["error"].astype(float).to_numpy()) / (scale.to_numpy(dtype=float) + MASE_EPS)))
                obs = float(m_pool[m_pool["model"] == model]["MASE"].iloc[0])
                self.assertLessEqual(abs(mase - obs), TOL)

    def test_47_controlled_rerun_metric_reconciliation(self) -> None:
        self.assertTrue(bool(self.h1_recon["pass"].astype(bool).all()))
        self.assertTrue(bool(self.h3_recon["pass"].astype(bool).all()))

    def test_48_all_four_models_in_table2(self) -> None:
        self.assertSetEqual(set(self.table2["model"].unique().tolist()), set(EXPECTED_MODELS))

    def test_49_exactly_eight_table2_rows(self) -> None:
        self.assertEqual(int(len(self.table2)), 8)

    def test_50_no_performance_based_model_filtering(self) -> None:
        counts = self.table2.groupby("horizon")["model"].nunique().to_dict()
        self.assertEqual(counts, {1: 4, 3: 4})

    def test_51_no_model_fitting(self) -> None:
        txt = BUILD_SCRIPT.read_text(encoding="utf-8").lower()
        forbidden = ["sklearn", "xgboost", "lightgbm", "fit(", "optuna"]
        self.assertFalse(any(tok in txt for tok in forbidden))

    def test_52_source_file_preservation(self) -> None:
        self.assertTrue(bool(self.input_ver["verification"]["checks"]["source_preservation_pass"]))

    def test_53_authorized_output_directory_only(self) -> None:
        for p in OUT_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".py", ".csv", ".json", ".md", ".sha256"}:
                self.assertTrue(str(p.resolve()).startswith(str(OUT_DIR.resolve())))

    def test_54_deterministic_regeneration(self) -> None:
        files = stage_deterministic_files()
        before = snapshot_hashes(files)
        run_build_script()
        after = snapshot_hashes(files)
        self.assertEqual(before, after)

    def test_55_stable_output_checksums(self) -> None:
        res = subprocess.run(["sha256sum", "-c", CHECKSUM_PATH.name], cwd=OUT_DIR, text=True, capture_output=True)
        self.assertEqual(int(res.returncode), 0)

    def test_56_checksum_registry_coverage(self) -> None:
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


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCanonicalH1H3Reconciliation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    update_manifest_test_result(
        success=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
    )

    # Rebuild once so manifest/report/checksums include final test_result deterministically.
    run_build_script()

    sys.exit(0 if result.wasSuccessful() else 1)
