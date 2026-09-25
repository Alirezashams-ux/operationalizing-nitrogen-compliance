from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = OUT_DIR / "build_final_computational_integration.py"

EXPECTED_REPOSITORY = "/home/alrezshams/acs_tnout_ulsan_revision"
EXPECTED_BRANCH = "controlled-reruns-v1"
EXPECTED_HEAD = "f61278b3e3fc2cb0b69564b346a4bf32e037b4fb"
EXPECTED_PARENT_TAG = "lag-window-sensitivity-execution-v1"
EXPECTED_PYTHON = "3.8.10"
EXPECTED_PARENT_TAG_OBJECT = "0ece3a84d51fcc80d29fa83a3ee4c08edce95006"

AUTHORIZED_REL_DIR = "revision_2026/11_final_integration"

TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REG_DIR = OUT_DIR / "registries"
AUDIT_DIR = OUT_DIR / "audits"

MANIFEST_PATH = OUT_DIR / "final_computational_integration_manifest.json"
REPORT_PATH = OUT_DIR / "final_computational_integration_completion_report.md"
CHECKSUM_PATH = OUT_DIR / "final_computational_integration_checksums.sha256"
DETERMINISTIC_AUDIT_PATH = AUDIT_DIR / "deterministic_rebuild_audit.json"

TABLE1_PATH = TABLE_DIR / "final_table1_canonical_event_prevalence.csv"
TABLE2_PATH = TABLE_DIR / "final_table2_point_model_comparison.csv"
TABLE3_COMPACT_PATH = TABLE_DIR / "final_table3_retrospective_alarm_budget_compact.csv"
TABLE3_FULL_PATH = TABLE_DIR / "si_retrospective_alarm_budget_full.csv"
SEQ_PATH = TABLE_DIR / "final_sequential_policy_table.csv"
SEQ_FOLD_PATH = TABLE_DIR / "si_sequential_policy_by_fold.csv"
LAG_PATH = TABLE_DIR / "final_lag_window_sensitivity_table.csv"
LAG_FULL_PATH = TABLE_DIR / "si_lag_window_metrics_full.csv"
SPLIT_SUMMARY_PATH = TABLE_DIR / "si_corrected_split_summary.csv"
FOLD_VAR_PATH = TABLE_DIR / "si_descriptive_fold_variability.csv"
SI_POINT_PATH = TABLE_DIR / "si_point_metrics_by_fold.csv"

AUDIT_TABLE1_PATH = AUDIT_DIR / "final_table1_reconciliation_audit.csv"
AUDIT_TABLE2_PATH = AUDIT_DIR / "final_table2_reconciliation_audit.csv"
AUDIT_TABLE3_PATH = AUDIT_DIR / "final_table3_reconciliation_audit.csv"
AUDIT_CROSS_PATH = AUDIT_DIR / "final_cross_package_consistency_audit.csv"
AUDIT_DATE_PATH = AUDIT_DIR / "final_date_alignment_audit.csv"
AUDIT_EMBARGO_PATH = AUDIT_DIR / "final_target_date_embargo_audit.csv"
AUDIT_ALARM_ARITH_PATH = AUDIT_DIR / "final_alarm_budget_arithmetic_audit.csv"
AUDIT_SEQ_PATH = AUDIT_DIR / "sequential_policy_reconciliation_audit.csv"
AUDIT_SEQ_CAP_PATH = AUDIT_DIR / "final_sequential_capacity_audit.csv"
AUDIT_FIG_DATA_PATH = AUDIT_DIR / "figure_data_reconciliation_audit.csv"
AUDIT_DOC_AVAIL_PATH = AUDIT_DIR / "document_scan_availability.json"
AUDIT_DOC_SCAN_PATH = AUDIT_DIR / "read_only_document_numeric_scan.csv"
AUDIT_EVENT_EQ_PATH = AUDIT_DIR / "final_event_count_equality_audit.csv"
AUDIT_MODEL_DATE_PATH = AUDIT_DIR / "final_model_date_equality_audit.csv"
AUDIT_METRIC_RECALC_PATH = AUDIT_DIR / "final_metric_recalculation_audit.csv"
AUDIT_LAG_PATH = AUDIT_DIR / "lag_window_integration_audit.csv"
AUDIT_LEAKAGE_PATH = AUDIT_DIR / "final_leakage_control_registry.csv"
AUDIT_HIST_DISPUTED_PATH = AUDIT_DIR / "historical_disputed_value_registry.csv"

REG_TAG_PATH = REG_DIR / "frozen_tag_registry.csv"
REG_PACKAGE_PATH = REG_DIR / "frozen_package_registry.csv"
REG_SOURCE_PATH = REG_DIR / "final_source_registry.csv"
REG_FIG_PATH = REG_DIR / "final_figure_registry.csv"
REG_EXCL_PATH = REG_DIR / "final_exclusion_registry.csv"
REG_TERM_PATH = REG_DIR / "final_terminology_registry.csv"
REG_CLAIMS_CSV_PATH = REG_DIR / "canonical_numeric_claim_registry.csv"
REG_CLAIMS_JSON_PATH = REG_DIR / "canonical_numeric_claim_registry.json"
REG_DOC_REQ_PATH = REG_DIR / "document_update_requirements.csv"

REPRO_TREE_JSON_PATH = OUT_DIR / "final_reproducibility_tree.json"
REPRO_TREE_MD_PATH = OUT_DIR / "final_reproducibility_tree.md"
INPUT_VERIFY_PATH = OUT_DIR / "input_verification.json"

TABLE1_EVENT_EXPECTED = {
    (1, 15): 141,
    (1, 16): 74,
    (1, 17): 30,
    (3, 15): 134,
    (3, 16): 71,
    (3, 17): 29,
    (5, 15): 134,
    (5, 16): 71,
    (5, 17): 29,
}

POINT_MODELS = {
    ("Persistence", 1),
    ("Persistence", 3),
    ("Persistence", 5),
    ("Ridge", 1),
    ("Ridge", 3),
    ("Ridge", 5),
    ("ElasticNet", 1),
    ("ElasticNet", 3),
    ("ElasticNet", 5),
    ("HGBR", 1),
    ("HGBR", 3),
    ("HGBR", 5),
    ("BCR-TCN v1.1", 5),
}

POINT_MAE_TOL = 1e-8
POINT_MSE_TOL = 2e-7
POINT_RMSE_TOL = 1e-8

FINAL_REF_MAP = {
    ("Ridge", 1): "REFERENCE",
    ("Ridge", 3): "REFERENCE",
    ("Ridge", 5): "REFERENCE",
    ("HGBR", 1): "REFERENCE",
    ("HGBR", 3): "REFERENCE",
    ("HGBR", 5): "LONG",
}

IMMUTABLE_DIRS = [
    ROOT / "revision_2026" / "03_corrected_protocol",
    ROOT / "revision_2026" / "04_controlled_reruns",
    ROOT / "revision_2026" / "05_canonical_predictions",
    ROOT / "revision_2026" / "06_corrected_hybridrank",
    ROOT / "revision_2026" / "07_retrospective_alarm_budget",
    ROOT / "revision_2026" / "08_sequential_alarm_policy",
    ROOT / "revision_2026" / "09_h1_h3_reconciliation",
    ROOT / "revision_2026" / "10_lag_window_sensitivity",
]

SNAPSHOT_EXT = {".py", ".csv", ".json", ".md", ".svg", ".png"}


@dataclass
class ChecksumValidation:
    registry_pass: bool
    coverage_pass: bool
    coverage_missing: List[str]
    coverage_extra: List[str]
    sha256sum_exit_code: int
    sha256sum_stdout: str
    sha256sum_stderr: str


class AssertionCollector:
    def __init__(self) -> None:
        self.total = 0
        self.failed = 0
        self.failures: List[str] = []

    def check(self, condition: bool, message: str) -> None:
        self.total += 1
        if not bool(condition):
            self.failed += 1
            self.failures.append(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_tree_hash(snapshot: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(snapshot.keys()):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(snapshot[rel].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def git_output(args: Sequence[str]) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def parse_git_status_paths() -> List[str]:
    raw = git_output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    paths: List[str] = []
    for line in raw.splitlines():
        if not line:
            continue
        payload = line[3:]
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        paths.append(payload.strip())
    return sorted(paths)


def is_authorized_git_path(path: str, authorized_rel: str) -> bool:
    p = path.strip().rstrip("/")
    a = authorized_rel.strip().rstrip("/")
    return bool(p) and (p.startswith(a) or a.startswith(p))


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def snapshot_tree_checksums(dirs: Sequence[Path]) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for d in dirs:
        if not d.exists():
            raise RuntimeError("immutable directory missing: {}".format(d))
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            snap[rel] = sha256_file(p)
    return snap


def snapshot_output_tree(out_dir: Path) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SNAPSHOT_EXT:
            continue
        rel = str(p.relative_to(out_dir)).replace("\\", "/")
        snap[rel] = sha256_file(p)
    return snap


def parse_checksum_manifest(path: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError("malformed checksum line in {}: {}".format(path, line))
        entries.append((parts[0].strip(), parts[1].strip()))
    return entries


def rewrite_checksums() -> None:
    lines: List[str] = []
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SNAPSHOT_EXT:
            continue
        if p.resolve() == CHECKSUM_PATH.resolve():
            continue
        rel = str(p.relative_to(OUT_DIR)).replace("\\", "/")
        lines.append("{}  {}".format(sha256_file(p), rel))
    CHECKSUM_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_checksums() -> ChecksumValidation:
    entries = parse_checksum_manifest(CHECKSUM_PATH)

    missing: List[str] = []
    mismatched: List[str] = []
    listed_files = []
    for expected_sha, rel in entries:
        listed_files.append(rel)
        p = OUT_DIR / rel
        if not p.exists():
            missing.append(rel)
            continue
        if sha256_file(p) != expected_sha:
            mismatched.append(rel)

    actual_files = sorted(
        str(p.relative_to(OUT_DIR)).replace("\\", "/")
        for p in OUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SNAPSHOT_EXT and p.name != CHECKSUM_PATH.name
    )
    listed_files_sorted = sorted(listed_files)

    coverage_missing = sorted(set(actual_files).difference(set(listed_files_sorted)))
    coverage_extra = sorted(set(listed_files_sorted).difference(set(actual_files)))

    proc = subprocess.run(
        ["sha256sum", "-c", CHECKSUM_PATH.name],
        cwd=OUT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    return ChecksumValidation(
        registry_pass=(len(missing) == 0 and len(mismatched) == 0),
        coverage_pass=(len(coverage_missing) == 0 and len(coverage_extra) == 0),
        coverage_missing=coverage_missing,
        coverage_extra=coverage_extra,
        sha256sum_exit_code=int(proc.returncode),
        sha256sum_stdout=proc.stdout,
        sha256sum_stderr=proc.stderr,
    )


def run_builder() -> None:
    subprocess.check_call([sys.executable, str(BUILD_SCRIPT)], cwd=ROOT)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_manifest_and_report(
    manifest: Dict[str, Any],
    test_result: str,
    deterministic_result: str,
    final_decision: str,
    deterministic_audit: Dict[str, Any],
    checksum: ChecksumValidation,
    source_preservation_pass: bool,
    source_preservation_changed: List[str],
    assertion_total: int,
    assertion_passed: int,
    assertion_failed: int,
    failures: List[str],
) -> None:
    manifest["test_result"] = test_result
    manifest["deterministic_result"] = deterministic_result
    manifest["final_decision"] = final_decision
    manifest["source_preservation_result"] = bool(source_preservation_pass)
    manifest["source_preservation_details"] = {
        "pass": bool(source_preservation_pass),
        "changed_paths": source_preservation_changed,
    }
    manifest["deterministic_rebuild_audit_file"] = str(DETERMINISTIC_AUDIT_PATH.relative_to(ROOT)).replace("\\", "/")
    manifest["checksum_result"] = {
        "registry_pass": checksum.registry_pass,
        "coverage_pass": checksum.coverage_pass,
        "coverage_missing": checksum.coverage_missing,
        "coverage_extra": checksum.coverage_extra,
        "sha256sum_exit_code": checksum.sha256sum_exit_code,
        "sha256sum_stdout": checksum.sha256sum_stdout,
        "sha256sum_stderr": checksum.sha256sum_stderr,
    }
    manifest["assertions_total"] = int(assertion_total)
    manifest["assertions_passed"] = int(assertion_passed)
    manifest["assertions_failed"] = int(assertion_failed)

    write_json(MANIFEST_PATH, manifest)
    write_json(DETERMINISTIC_AUDIT_PATH, deterministic_audit)

    gate_map = {
        "A": "A. Final computational integration package is validated and ready for audit transfer.",
        "B": "B. Integration outputs assembled, but unresolved review questions remain.",
        "C": "C. Source-lineage or numeric consistency conflict detected in integrated outputs.",
        "D": "D. Deterministic regeneration, checksum validation, test assertions, or source-preservation checks failed.",
    }

    lines = [
        "# Stage 3 Final Computational Integration Completion Report",
        "",
        "## Validation Summary",
        "- test_result: {}".format(test_result),
        "- deterministic_result: {}".format(deterministic_result),
        "- source_preservation_result: {}".format(bool(source_preservation_pass)),
        "- checksum_registry_pass: {}".format(checksum.registry_pass),
        "- checksum_coverage_pass: {}".format(checksum.coverage_pass),
        "- checksum_sha256sum_exit_code: {}".format(checksum.sha256sum_exit_code),
        "- assertions_total: {}".format(assertion_total),
        "- assertions_passed: {}".format(assertion_passed),
        "- assertions_failed: {}".format(assertion_failed),
        "",
        "## Deterministic Rebuild",
        "- first_build_package_hash: {}".format(deterministic_audit["first_build_package_hash"]),
        "- second_build_package_hash: {}".format(deterministic_audit["second_build_package_hash"]),
        "- files_compared: {}".format(deterministic_audit["files_compared"]),
        "- different_files: {}".format(deterministic_audit["different_files"]),
        "",
        "## Final Decision",
        gate_map[final_decision],
    ]

    if failures:
        lines.append("")
        lines.append("## Failed Assertions")
        for msg in failures:
            lines.append("- {}".format(msg))

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("run this test from workspace root: {}".format(ROOT))

    ac = AssertionCollector()

    # Snapshot immutable frozen packages before integration rebuild.
    pre_immutable = snapshot_tree_checksums(IMMUTABLE_DIRS)

    run_builder()
    first_snapshot = snapshot_output_tree(OUT_DIR)
    first_hash = stable_tree_hash(first_snapshot)

    run_builder()
    second_snapshot = snapshot_output_tree(OUT_DIR)
    second_hash = stable_tree_hash(second_snapshot)

    all_files = sorted(set(first_snapshot.keys()).union(set(second_snapshot.keys())))
    differing = [f for f in all_files if first_snapshot.get(f) != second_snapshot.get(f)]
    deterministic_pass = len(differing) == 0

    post_immutable = snapshot_tree_checksums(IMMUTABLE_DIRS)
    changed_immutable = sorted(
        p for p in sorted(set(pre_immutable.keys()).union(set(post_immutable.keys()))) if pre_immutable.get(p) != post_immutable.get(p)
    )
    source_preservation_pass = len(changed_immutable) == 0

    manifest = read_json(MANIFEST_PATH)
    input_verify = read_json(INPUT_VERIFY_PATH)
    repro_tree = read_json(REPRO_TREE_JSON_PATH)
    repro_tree_md = REPRO_TREE_MD_PATH.read_text(encoding="utf-8")
    report_text = REPORT_PATH.read_text(encoding="utf-8")

    table1 = read_csv(TABLE1_PATH)
    table2 = read_csv(TABLE2_PATH)
    table3_compact = read_csv(TABLE3_COMPACT_PATH)
    table3_full = read_csv(TABLE3_FULL_PATH)
    seq = read_csv(SEQ_PATH)
    seq_fold = read_csv(SEQ_FOLD_PATH)
    lag = read_csv(LAG_PATH)
    lag_full = read_csv(LAG_FULL_PATH)
    split_summary = read_csv(SPLIT_SUMMARY_PATH)
    fold_var = read_csv(FOLD_VAR_PATH)
    si_point = read_csv(SI_POINT_PATH)

    audit1 = read_csv(AUDIT_TABLE1_PATH)
    audit2 = read_csv(AUDIT_TABLE2_PATH)
    audit3 = read_csv(AUDIT_TABLE3_PATH)
    audit_cross = read_csv(AUDIT_CROSS_PATH)
    audit_date = read_csv(AUDIT_DATE_PATH)
    audit_embargo = read_csv(AUDIT_EMBARGO_PATH)
    audit_alarm = read_csv(AUDIT_ALARM_ARITH_PATH)
    audit_seq = read_csv(AUDIT_SEQ_PATH)
    audit_seq_cap = read_csv(AUDIT_SEQ_CAP_PATH)
    audit_fig = read_csv(AUDIT_FIG_DATA_PATH)
    audit_doc_scan = read_csv(AUDIT_DOC_SCAN_PATH)
    audit_event = read_csv(AUDIT_EVENT_EQ_PATH)
    audit_model_date = read_csv(AUDIT_MODEL_DATE_PATH)
    audit_metric = read_csv(AUDIT_METRIC_RECALC_PATH)
    audit_lag = read_csv(AUDIT_LAG_PATH)
    audit_leak = read_csv(AUDIT_LEAKAGE_PATH)
    audit_hist_disputed = read_csv(AUDIT_HIST_DISPUTED_PATH)
    doc_avail = read_json(AUDIT_DOC_AVAIL_PATH)

    reg_tag = read_csv(REG_TAG_PATH)
    reg_pkg = read_csv(REG_PACKAGE_PATH)
    reg_source = read_csv(REG_SOURCE_PATH)
    reg_fig = read_csv(REG_FIG_PATH)
    reg_excl = read_csv(REG_EXCL_PATH)
    reg_term = read_csv(REG_TERM_PATH)
    reg_claims = read_csv(REG_CLAIMS_CSV_PATH)
    reg_doc_req = read_csv(REG_DOC_REQ_PATH)
    reg_claims_json = read_json(REG_CLAIMS_JSON_PATH)

    # 1-14: environment and preflight identity
    ac.check(git_output(["git", "rev-parse", "--show-toplevel"]) == EXPECTED_REPOSITORY, "1. repository root mismatch")
    ac.check(git_output(["git", "branch", "--show-current"]) == EXPECTED_BRANCH, "2. branch mismatch")
    ac.check(git_output(["git", "rev-parse", "HEAD"]) == EXPECTED_HEAD, "3. HEAD mismatch")
    ac.check(sys.version.split()[0] == EXPECTED_PYTHON, "4. python version mismatch")
    ac.check(BUILD_SCRIPT.exists(), "5. build script missing")
    ac.check(TABLE_DIR.exists() and FIG_DIR.exists() and REG_DIR.exists() and AUDIT_DIR.exists(), "6. output subdirectories missing")
    ac.check(manifest.get("integration_id") == "stage3_final_computational_integration_v1", "7. integration_id mismatch")
    ac.check(manifest.get("input_parent_commit") == EXPECTED_HEAD, "8. parent commit mismatch in manifest")
    ac.check(manifest.get("input_parent_tag") == EXPECTED_PARENT_TAG, "9. parent tag mismatch in manifest")
    ac.check(manifest.get("input_parent_tag_object_id") == EXPECTED_PARENT_TAG_OBJECT, "10. parent tag object mismatch in manifest")
    ac.check(manifest.get("input_parent_tag_peeled_commit") == EXPECTED_HEAD, "11. parent peeled commit mismatch in manifest")
    ac.check(input_verify.get("head") == EXPECTED_HEAD, "12. input verification head mismatch")
    ac.check(bool(input_verify.get("parent_tag_registry_ok", False)), "13. input verification parent-tag registry check failed")
    ac.check(bool(input_verify.get("checksum_registry_all_pass", False)), "14. upstream checksum preflight failed")

    # 15-24: deterministic and source-preservation checks
    ac.check(deterministic_pass, "15. deterministic rebuild file hashes differ")
    ac.check(first_hash == second_hash, "16. deterministic package hash mismatch")
    ac.check(len(first_snapshot) > 40, "17. output snapshot unexpectedly small")
    ac.check(source_preservation_pass, "18. immutable frozen sources changed")
    ac.check(len(changed_immutable) == 0, "19. immutable changed path list is non-empty")
    ac.check((OUT_DIR / "build_final_computational_integration.py").exists(), "20. builder script absent in output directory")
    ac.check((OUT_DIR / "test_final_computational_integration.py").exists(), "21. test script absent in output directory")
    ac.check(CHECKSUM_PATH.exists(), "22. checksum file missing")
    ac.check(MANIFEST_PATH.exists(), "23. manifest file missing")
    ac.check(REPORT_PATH.exists(), "24. completion report missing")

    # 25-39: major row-count and schema checks
    ac.check(len(table1) == 9, "25. table1 row count mismatch")
    ac.check(len(table2) == 13, "26. table2 row count mismatch")
    ac.check(len(si_point) == 39, "27. si_point row count mismatch")
    ac.check(len(table3_compact) == 24, "28. table3 compact row count mismatch")
    ac.check(len(table3_full) == 36, "29. table3 full row count mismatch")
    ac.check(len(seq) == 72, "30. sequential pooled row count mismatch")
    ac.check(len(seq_fold) == 216, "31. sequential by-fold row count mismatch")
    ac.check(len(lag) == 18, "32. lag table row count mismatch")
    ac.check(len(lag_full) == 54, "33. lag full row count mismatch")
    ac.check(len(split_summary) == 9, "34. split summary row count mismatch")
    ac.check(len(fold_var) == 39, "35. fold variability row count mismatch")
    ac.check(len(audit_cross) == 27, "36. cross-package audit row count mismatch")
    ac.check(len(reg_tag) == 15, "37. frozen tag registry row count mismatch")
    ac.check(len(reg_pkg) == 15, "38. frozen package registry row count mismatch")
    ac.check(len(reg_source) == 36, "39. final source registry row count mismatch")

    # 40-56: table1 + date/embargo checks
    ac.check(set(table1["horizon_days"].astype(int).tolist()) == {1, 3, 5}, "40. table1 horizon set mismatch")
    ac.check(set(table1["threshold_mg_L"].astype(int).tolist()) == {15, 16, 17}, "41. table1 threshold set mismatch")
    hmap = table1.groupby("horizon_days")["canonical_test_n"].first().to_dict()
    ac.check(int(hmap.get(1, -1)) == 762, "42. H1 canonical N mismatch")
    ac.check(int(hmap.get(3, -1)) == 747, "43. H3 canonical N mismatch")
    ac.check(int(hmap.get(5, -1)) == 747, "44. H5 canonical N mismatch")

    ev_map = {(int(r.horizon_days), int(r.threshold_mg_L)): int(r.pooled_event_count) for r in table1.itertuples(index=False)}
    ac.check(ev_map == TABLE1_EVENT_EXPECTED, "45. table1 event-count map mismatch")

    fold_sum_ok = (table1["fold_1_n"] + table1["fold_2_n"] + table1["fold_3_n"] == table1["canonical_test_n"]).all()
    ac.check(bool(fold_sum_ok), "46. table1 fold sum != canonical_test_n")

    prev_ok = True
    for r in table1.itertuples(index=False):
        n = int(r.canonical_test_n)
        events = int(r.pooled_event_count)
        if abs(float(r.prevalence_fraction) - float(events / n)) > 1e-12:
            prev_ok = False
            break
    ac.check(prev_ok, "47. prevalence fraction mismatch in table1")

    rand_ok = True
    for r in table1.itertuples(index=False):
        n = int(r.canonical_test_n)
        f = [int(r.fold_1_n), int(r.fold_2_n), int(r.fold_3_n)]
        r05 = sum(int(math.ceil(0.05 * x)) for x in f) / n
        r10 = sum(int(math.ceil(0.10 * x)) for x in f) / n
        if abs(float(r.random_expected_recall_r05) - r05) > 1e-12 or abs(float(r.random_expected_recall_r10) - r10) > 1e-12:
            rand_ok = False
            break
    ac.check(rand_ok, "48. random expected recall fields mismatch")

    ac.check(audit1["event_column_equals_y_true"].map(normalize_bool).all(), "49. table1 event-column reconciliation failed")
    ac.check(audit1["cross_check_match"].map(normalize_bool).all(), "50. table1 cross-check mismatch")
    ac.check(audit_date["target_equals_feature_plus_horizon"].map(normalize_bool).all(), "51. date alignment audit failed")
    ac.check(audit_date.set_index("horizon")["row_count"].to_dict() == {1: 762, 3: 747, 5: 747}, "52. date alignment row_count mismatch")
    ac.check(len(audit_embargo) == 9, "53. embargo audit row count mismatch")
    ac.check(audit_embargo["strict_embargo_condition"].map(normalize_bool).all(), "54. strict target-date embargo condition failed")
    ac.check(audit_embargo["outer_embargo_pass"].map(normalize_bool).all(), "55. outer embargo pass failed")
    ac.check(audit_embargo["inner_embargo_pass"].map(normalize_bool).all(), "56. inner embargo pass failed")

    # 57-76: table2 + point-metric audits
    got_point_models = {(str(r.model), int(r.horizon_days)) for r in table2.itertuples(index=False)}
    ac.check(got_point_models == POINT_MODELS, "57. table2 model-horizon scope mismatch")
    ac.check((table2[table2["model"] == "BCR-TCN v1.1"]["horizon_days"] == 5).all(), "58. BCR-TCN appears outside H5")

    counts_ok = True
    for r in table2.itertuples(index=False):
        h = int(r.horizon_days)
        n = int(r.canonical_test_n)
        if h == 1 and n != 762:
            counts_ok = False
        if h in {3, 5} and n != 747:
            counts_ok = False
    ac.check(counts_ok, "59. table2 canonical N values mismatch")

    ac.check(len(audit2) == 13, "60. table2 reconciliation row count mismatch")
    ac.check(float(audit2["abs_diff_MAE"].max()) <= POINT_MAE_TOL, "61. table2 MAE recomputation tolerance exceeded")
    ac.check(float(audit2["abs_diff_MSE"].max()) <= POINT_MSE_TOL, "62. table2 MSE recomputation tolerance exceeded")
    ac.check(float(audit2["abs_diff_RMSE"].max()) <= POINT_RMSE_TOL, "63. table2 RMSE recomputation tolerance exceeded")
    ac.check(audit2["source_RMSE_equals_sqrt_source_MSE"].map(normalize_bool).all(), "64. source RMSE != sqrt(source MSE)")
    ac.check(float(audit2["abs_diff_MASE_weighted"].max()) <= 1e-12, "65. pooled MASE vs weighted fold MASE mismatch")
    ac.check(set(si_point["outer_fold"].astype(int).unique().tolist()) == {1, 2, 3}, "66. si_point outer fold set mismatch")

    per_combo_three = (si_point.groupby(["model", "horizon_days"]).size() == 3).all()
    ac.check(bool(per_combo_three), "67. si_point missing per-fold rows per model-horizon")

    ac.check(set(si_point["source_package"].unique().tolist()) == {"canonical_h1_h3_reconciliation", "canonical_h5_assembly"}, "68. si_point source_package set mismatch")
    ac.check((si_point["MASE"] >= 0).all(), "69. negative MASE in si_point")
    ac.check((table2["MAE_mg_L"] >= 0).all(), "70. negative MAE in table2")
    ac.check((table2["RMSE_mg_L"] >= 0).all(), "71. negative RMSE in table2")
    ac.check((table2["MSE_mg2_L2"] >= 0).all(), "72. negative MSE in table2")
    ac.check((table2["model_scope_note"].astype(str).str.len() > 0).all(), "73. empty model_scope_note in table2")
    ac.check((table2["mase_method"].astype(str).str.len() > 0).all(), "74. empty mase_method in table2")
    ac.check((table2["mase_denominator_scope"].astype(str).str.contains("fold_local_training_only")).all(), "75. unexpected MASE denominator scope")
    ac.check(audit_metric.shape[0] == 13, "76. metric recalculation audit row count mismatch")

    # 77-95: retrospective and sequential checks
    ac.check(set(table3_compact["threshold_mg_L"].astype(int).unique().tolist()) == {15, 16}, "77. compact retrospective threshold set mismatch")
    ac.check(set(table3_compact["nominal_budget_fraction"].astype(float).unique().tolist()) == {0.05, 0.10}, "78. compact retrospective budget set mismatch")
    ac.check(set(table3_full["threshold_mg_L"].astype(int).unique().tolist()) == {15, 16, 17}, "79. full retrospective threshold set mismatch")

    full_rows_per_model = table3_full.groupby("model").size().to_dict()
    compact_rows_per_model = table3_compact.groupby("model").size().to_dict()
    ac.check(all(v == 6 for v in full_rows_per_model.values()), "80. full retrospective rows/model mismatch")
    ac.check(all(v == 4 for v in compact_rows_per_model.values()), "81. compact retrospective rows/model mismatch")

    tie_ok = table3_compact["tie_break_rule"].astype(str).str.contains("desc_score").all()
    ac.check(bool(tie_ok), "82. retrospective tie-break rule mismatch")
    ac.check(audit3["pooled_k_matches_sum_fold_k"].map(normalize_bool).all(), "83. table3 pooled_K != sum fold-local k")
    ac.check(audit3["realized_fraction_match"].map(normalize_bool).all(), "84. table3 realized fraction mismatch")
    ac.check(set(table3_compact["k_per_fold"].astype(int).unique().tolist()) == {13, 25}, "85. table3 k_per_fold set mismatch")
    ac.check(table3_compact["retrospective_only"].map(normalize_bool).all(), "86. retrospective_only flag mismatch")
    ac.check(table3_compact["evaluation_type"].astype(str).str.contains("retrospective", case=False).all(), "87. retrospective evaluation_type mismatch")

    ac.check(set(seq["policy_type"].unique().tolist()) == {"past_only_sequential_threshold_simulation", "historical_quota_enforced_sequential_simulation"}, "88. sequential policy set mismatch")
    ac.check(set(seq["outer_fold_or_pooled"].astype(str).unique().tolist()) == {"pooled"}, "89. sequential pooled table contains non-pooled rows")
    ac.check(set(seq["startup_dates"].astype(int).unique().tolist()) == {90}, "90. sequential pooled startup_dates mismatch")
    ac.check(set(seq["eligible_dates"].astype(int).unique().tolist()) == {657}, "91. sequential pooled eligible_dates mismatch")
    ac.check(set(seq_fold["outer_fold_or_pooled"].astype(str).unique().tolist()) == {"1", "2", "3"}, "92. sequential by-fold outer set mismatch")
    ac.check(set(seq_fold["startup_dates"].astype(int).unique().tolist()) == {30}, "93. sequential by-fold startup_dates mismatch")
    ac.check(set(seq_fold["eligible_dates"].astype(int).unique().tolist()) == {219}, "94. sequential by-fold eligible_dates mismatch")
    ac.check(audit_seq_cap["prefix_capacity_pass"].map(normalize_bool).all(), "95. sequential prefix-capacity audit failed")

    # 96-110: sequential quota semantics + lag checks
    ac.check(audit_seq_cap["capacity_audit_pass"].map(normalize_bool).all(), "96. sequential capacity audit failed")
    ac.check(audit_seq_cap["candidate_provenance_pass"].map(normalize_bool).all(), "97. sequential candidate provenance audit failed")
    ac.check(audit_seq["pass"].map(normalize_bool).all(), "98. sequential policy reconciliation audit failed")

    quota_rows = seq[seq["policy_type"] == "historical_quota_enforced_sequential_simulation"].copy()
    past_rows = seq[seq["policy_type"] == "past_only_sequential_threshold_simulation"].copy()
    ac.check((quota_rows["issued_alarms"].astype(float) <= quota_rows["maximum_integer_capacity"].astype(float)).all(), "99. quota issued alarms exceed maximum integer capacity")
    ac.check((quota_rows["suppressed_candidates"].astype(float) >= 0).all(), "100. negative suppressed candidates in quota rows")
    ac.check((past_rows["suppressed_candidates"].fillna(0).astype(float) == 0).all(), "101. past-only rows have non-zero suppression")
    ac.check((past_rows["issued_alarms"].astype(float) == past_rows["candidate_alarms"].astype(float)).all(), "102. past-only issued alarms != candidate alarms")
    ac.check(seq["causal"].map(normalize_bool).all(), "103. sequential causal flag not true for all rows")
    ac.check((seq["prospective_field_validation"].map(normalize_bool) == False).all(), "104. sequential prospective_field_validation should be false")
    ac.check(quota_rows["hard_prefix_capacity_enforced"].map(normalize_bool).all(), "105. quota rows missing hard prefix enforcement flag")
    ac.check((past_rows["hard_prefix_capacity_enforced"].map(normalize_bool) == False).all(), "106. past-only rows incorrectly set hard-prefix flag")

    ac.check(len(lag) == 18 and set(lag["model"].unique().tolist()) == {"Ridge", "HGBR"}, "107. lag table model scope mismatch")
    ac.check(set(lag["configuration_id"].unique().tolist()) == {"SHORT", "REFERENCE", "LONG"}, "108. lag configuration set mismatch")
    lag_ref_map = {(str(r.model), int(r.horizon_days)): str(r.final_reference_configuration) for r in lag.drop_duplicates(subset=["model", "horizon_days"]).itertuples(index=False)}
    ac.check(lag_ref_map == FINAL_REF_MAP, "109. lag final reference mapping mismatch")
    ac.check(int(lag["is_final_reference"].map(normalize_bool).sum()) == 6, "110. lag final-reference row count mismatch")

    # 111-126: lag and figure registries
    ac.check(audit_lag["pass"].map(normalize_bool).all(), "111. lag integration audit failed")
    ref_rows = lag[lag["is_final_reference"].map(normalize_bool)]
    ac.check((ref_rows["configuration_id"].astype(str) == ref_rows["final_reference_configuration"].astype(str)).all(), "112. lag reference rows have mismatched configuration label")
    ac.check((ref_rows["MAE_absolute_difference"].astype(float).abs() <= 1e-12).all(), "113. reference rows should have zero MAE difference")
    ac.check((lag["feature_count"].astype(float) > 0).all(), "114. lag feature_count must be positive")
    ac.check((lag["training_rows"].astype(float) > 0).all(), "115. lag training_rows must be positive")
    ac.check((lag["interpretation_classification"].astype(str).str.len() > 0).all(), "116. lag interpretation_classification empty")
    ac.check((lag["TNout_memory_features"].astype(str).str.len() > 0).all(), "117. lag TNout memory feature string empty")

    ac.check(len(reg_fig) == 5, "118. figure registry row count mismatch")
    figure_sources_ok = True
    for r in reg_fig.itertuples(index=False):
        src = OUT_DIR / str(r.source_csv).replace("figures/", "figures/")
        fig_svg = OUT_DIR / str(r.figure_artifact)
        fig_png = fig_svg.with_suffix(".png")
        if (not src.exists()) or (not fig_svg.exists()) or (not fig_png.exists()):
            figure_sources_ok = False
            break
        if sha256_file(src) != str(r.source_sha256):
            figure_sources_ok = False
            break
    ac.check(figure_sources_ok, "119. figure registry source/file linkage or hashes failed")
    ac.check(len(audit_fig) == 5, "120. figure-data audit row count mismatch")
    ac.check(audit_fig["reconciliation_pass"].map(normalize_bool).all(), "121. figure-data reconciliation audit failed")

    # 127-143: registry integrity and document-scan checks
    ac.check(len(reg_excl) >= 14, "122. exclusion registry unexpectedly small")
    ac.check(set(reg_term["term_type"].unique().tolist()) == {"preferred", "restriction", "prohibited"}, "123. terminology registry term_type set mismatch")
    ac.check(not reg_claims["claim_id"].duplicated().any(), "124. canonical claim registry has duplicate claim_id")
    ac.check(len(reg_claims_json) == len(reg_claims), "125. canonical claim CSV/JSON length mismatch")

    claim_sha_ok = reg_claims["source_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    ac.check(bool(claim_sha_ok), "126. canonical claim source_sha256 format mismatch")
    ac.check(not reg_doc_req["claim_id"].duplicated().any(), "127. document update requirements duplicate claim_id")

    ac.check(reg_tag["local_exists"].map(normalize_bool).all(), "128. required local frozen tags missing")
    parent_tag_row = reg_tag[reg_tag["tag_name"] == EXPECTED_PARENT_TAG]
    ac.check(not parent_tag_row.empty, "129. parent frozen tag row missing")
    if not parent_tag_row.empty:
        ac.check(str(parent_tag_row.iloc[0]["peeled_commit_id"]) == EXPECTED_HEAD, "130. parent frozen tag peeled commit mismatch")
    else:
        ac.check(False, "130. parent frozen tag peeled commit mismatch")

    source_hash_ok = True
    for r in reg_source.itertuples(index=False):
        path = ROOT / str(r.source_file)
        if not path.exists() or sha256_file(path) != str(r.source_sha256):
            source_hash_ok = False
            break
    ac.check(source_hash_ok, "131. final source registry path/hash reconciliation failed")
    ac.check(not reg_source["source_file"].duplicated().any(), "132. final source registry duplicate source_file")
    ac.check((reg_source["used_for_outputs"].astype(str).str.len() > 0).all(), "133. final source registry used_for_outputs empty")

    ac.check(set(doc_avail.keys()) == {"submitted_manuscript", "submitted_si", "current_revised_manuscript_draft", "reviewer_response", "working_revision_record"}, "134. document availability keys mismatch")
    ac.check(doc_avail["reviewer_response"].get("availability_status") == "external_input_required", "135. reviewer response availability status mismatch")
    ac.check(len(audit_doc_scan) >= 1, "136. document numeric scan unexpectedly empty")
    ac.check(len(audit_hist_disputed) == 13, "137. historical disputed registry row count mismatch")

    # 144-158: tree, cross-audits, checksum, git-scope checks
    ac.check(repro_tree.get("root") == "raw_dataset_lock", "138. reproducibility tree root mismatch")
    node_names = [str(n.get("package_name")) for n in repro_tree.get("nodes", [])]
    ac.check("final_integration" in node_names, "139. reproducibility tree missing final_integration node")
    ac.check("Stage 3 Final Computational Integration Completion Report" in report_text, "140. completion report header mismatch")
    ac.check("Final Reproducibility Tree" in repro_tree_md, "141. reproducibility tree markdown header missing")
    ac.check(audit_cross["pass"].map(normalize_bool).all(), "142. cross-package consistency audit has failing checks")
    ac.check(audit_event["event_count_reconciled"].map(normalize_bool).all(), "143. event-count equality audit failed")
    ac.check(audit_model_date["all_models_identical_dates"].map(normalize_bool).all(), "144. model-date equality audit failed")
    ac.check(audit_metric["source_RMSE_equals_sqrt_source_MSE"].map(normalize_bool).all(), "145. metric audit source RMSE check failed")
    ac.check(audit_alarm["pooled_k_matches_sum_fold_k"].map(normalize_bool).all(), "146. alarm arithmetic pooled-k check failed")
    ac.check(audit_leak["pass"].map(normalize_bool).all(), "147. leakage control registry has failing checks")

    rewrite_checksums()
    checksum_result = validate_checksums()
    checksum_pass = bool(checksum_result.registry_pass and checksum_result.coverage_pass and checksum_result.sha256sum_exit_code == 0)
    ac.check(checksum_result.registry_pass, "148. final checksum registry hash validation failed")
    ac.check(checksum_result.coverage_pass, "149. final checksum registry coverage validation failed")
    ac.check(checksum_result.sha256sum_exit_code == 0, "150. sha256sum -c failed on final package")

    status_paths = parse_git_status_paths()
    clean_outside_authorized = all(is_authorized_git_path(p, AUTHORIZED_REL_DIR) for p in status_paths)
    ac.check(clean_outside_authorized, "151. git status includes modifications outside authorized stage3 directory")

    deterministic_audit = {
        "first_build_package_hash": first_hash,
        "second_build_package_hash": second_hash,
        "files_compared": len(all_files),
        "identical_files": len(all_files) - len(differing),
        "different_files": len(differing),
        "differences": differing,
        "deterministic_result": "pass" if deterministic_pass else "fail",
    }

    test_result = "pass_{}_of_{}".format(ac.total - ac.failed, ac.total) if ac.failed == 0 else "fail_{}_of_{}".format(ac.total - ac.failed, ac.total)
    deterministic_result = "pass" if deterministic_pass else "fail"

    # Decision policy:
    # A: all core checks passed.
    # B: outputs assembled but unresolved review questions.
    # C: source-lineage or numeric conflict in core consistency audits.
    # D: deterministic/checksum/assertion/source-preservation failure.
    source_conflict = not (
        audit_cross["pass"].map(normalize_bool).all()
        and audit_event["event_count_reconciled"].map(normalize_bool).all()
        and audit_model_date["all_models_identical_dates"].map(normalize_bool).all()
        and audit_alarm["pooled_k_matches_sum_fold_k"].map(normalize_bool).all()
        and audit_leak["pass"].map(normalize_bool).all()
    )

    questions_remaining = False

    if (not deterministic_pass) or (not checksum_pass) or (not source_preservation_pass) or (ac.failed > 0 and not source_conflict):
        final_decision = "D"
    elif source_conflict:
        final_decision = "C"
    elif questions_remaining:
        final_decision = "B"
    else:
        final_decision = "A"

    update_manifest_and_report(
        manifest=manifest,
        test_result=test_result,
        deterministic_result=deterministic_result,
        final_decision=final_decision,
        deterministic_audit=deterministic_audit,
        checksum=checksum_result,
        source_preservation_pass=source_preservation_pass,
        source_preservation_changed=changed_immutable,
        assertion_total=ac.total,
        assertion_passed=ac.total - ac.failed,
        assertion_failed=ac.failed,
        failures=ac.failures,
    )

    # Recompute checksums after manifest/report/audit updates.
    rewrite_checksums()
    final_checksum = validate_checksums()
    if final_checksum.sha256sum_exit_code != 0:
        raise RuntimeError("final checksum verification failed\n{}\n{}".format(final_checksum.sha256sum_stdout, final_checksum.sha256sum_stderr))

    # Persist updated checksum_result in manifest.
    manifest_latest = read_json(MANIFEST_PATH)
    manifest_latest["checksum_result"] = {
        "registry_pass": final_checksum.registry_pass,
        "coverage_pass": final_checksum.coverage_pass,
        "coverage_missing": final_checksum.coverage_missing,
        "coverage_extra": final_checksum.coverage_extra,
        "sha256sum_exit_code": final_checksum.sha256sum_exit_code,
        "sha256sum_stdout": final_checksum.sha256sum_stdout,
        "sha256sum_stderr": final_checksum.sha256sum_stderr,
    }
    write_json(MANIFEST_PATH, manifest_latest)

    # Checksums changed again due manifest update.
    rewrite_checksums()
    post_manifest_checksum = validate_checksums()
    if post_manifest_checksum.sha256sum_exit_code != 0:
        raise RuntimeError("post-manifest checksum verification failed\n{}\n{}".format(post_manifest_checksum.sha256sum_stdout, post_manifest_checksum.sha256sum_stderr))

    if ac.failed:
        detail = "\n".join(ac.failures)
        raise AssertionError("Stage 3 final integration validation failed ({} failed of {}):\n{}".format(ac.failed, ac.total, detail))

    print("Ran {} checks; failures=0".format(ac.total))
    print("deterministic_result={}".format(deterministic_result))
    print("final_decision={}".format(final_decision))


if __name__ == "__main__":
    main()
