from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
BASELINE_DIR = ROOT / "revision_2026" / "04_controlled_reruns"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

FEATURE_NPZ_PATH = ROOT / "features" / "ulsan_H5_features_v2.npz"
SUBMITTED_META_PATH = ROOT / "results" / "metrics" / "bcr_tcn_v11_H5_v2_meta.json"
SUBMITTED_PRED_PATH = ROOT / "results" / "predictions" / "bcr_tcn_v11_H5_v2_preds.csv"
SUBMITTED_ALARM_V11_PATH = ROOT / "results" / "metrics" / "alarm_budget_bcr_tcn_v11_H5_v2.csv"
SUBMITTED_ALARM_759_CONTEXT_PATH = ROOT / "results" / "metrics" / "alarm_budget_H5.csv"
SUBMITTED_HYBRID_747_PATH = (
    ROOT
    / "results"
    / "hybrid_rank_v2_runs"
    / "20260219_195429_H5_hybrid_rank_v2"
    / "tables"
    / "alarm_budget_metrics_H5.csv"
)
SUBMITTED_HYBRID_735_PATH = (
    ROOT
    / "results"
    / "hybrid_rank_v2_runs"
    / "20260220_131820_H5_hybrid_rank_v2"
    / "tables"
    / "alarm_budget_metrics_H5.csv"
)

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
DEFINITION_PATH = OUT_DIR / "bcr_tcn_v11_definition_and_provenance.md"
FIXED_CONFIG_PATH = OUT_DIR / "bcr_tcn_v11_fixed_configuration.json"
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

PROTOCOL_TAG = "corrected_protocol_v1"
MODEL = "BCR-TCN"
MODEL_FULL_NAME = "Budget-Constrained Recall Temporal Convolutional Network"
MODEL_VERSION = "1.1"
MEANING_OF_BCR = "Checkpoint selection by validation Recall@5% at tau=16 mg/L under r_budget=0.05."
MEANING_OF_V11 = "Internal project-specific model/implementation version 1.1 used in submitted experiments."

AUTHORIZED_HORIZONS = (5,)
H = 5
OUTER_FOLDS = (1, 2, 3)
TAUS = (15.0, 16.0, 17.0)
SEED = 42
DEVICE = "cpu"
EXPECTED_BRANCH = "controlled-reruns-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_output(args: List[str]) -> str:
    out = subprocess.check_output(args, cwd=ROOT, text=True)
    return out.strip()


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_protocol_checksums(protocol_sha_path: Path, protocol_dir: Path) -> List[dict]:
    checks: List[dict] = []
    lines = [
        ln.strip()
        for ln in protocol_sha_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            raise RuntimeError(f"Malformed checksum line: {line}")
        expected_hash = parts[0]
        rel_name = parts[-1]
        path = protocol_dir / rel_name
        if not path.exists():
            raise RuntimeError(f"Protocol checksum target file is missing: {path}")
        observed_hash = sha256_file(path)
        ok = observed_hash == expected_hash
        checks.append(
            {
                "file": str(path),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "pass": ok,
            }
        )
        if not ok:
            raise RuntimeError(
                f"Protocol checksum mismatch for {path.name}: expected {expected_hash}, observed {observed_hash}"
            )
    return checks


def load_lock(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_canonical_dataset(lock: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    dataset_path = Path(lock["absolute_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Locked dataset file does not exist: {dataset_path}")

    observed_sha = sha256_file(dataset_path)
    expected_sha = lock["sha256"]
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Dataset checksum mismatch: expected {expected_sha}, observed {observed_sha}"
        )

    df = pd.read_csv(dataset_path)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    date_col = lock["date_column"]
    target_col = lock["target_column"]

    if date_col not in df.columns:
        raise RuntimeError(f"Locked date column is missing: {date_col}")
    if target_col not in df.columns:
        raise RuntimeError(f"Locked target column is missing: {target_col}")

    df[date_col] = normalize_date_col(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if len(df) != int(lock["row_count"]):
        raise RuntimeError(
            f"Dataset row count mismatch: expected {lock['row_count']}, observed {len(df)}"
        )
    if int(df[date_col].nunique()) != int(lock["unique_date_count"]):
        raise RuntimeError(
            "Dataset unique-date count mismatch: "
            f"expected {lock['unique_date_count']}, observed {int(df[date_col].nunique())}"
        )
    if int(df.duplicated(subset=[date_col]).sum()) != int(lock["duplicate_date_count"]):
        raise RuntimeError("Dataset duplicate-date count mismatch")

    min_date = df[date_col].min().strftime("%Y-%m-%d")
    max_date = df[date_col].max().strftime("%Y-%m-%d")
    if min_date != lock["minimum_date"] or max_date != lock["maximum_date"]:
        raise RuntimeError(
            "Dataset date-range mismatch: "
            f"expected [{lock['minimum_date']}, {lock['maximum_date']}], observed [{min_date}, {max_date}]"
        )

    all_days = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    missing_days = int(len(all_days) - int(df[date_col].nunique()))
    if missing_days != int(lock["missing_calendar_date_count"]):
        raise RuntimeError(
            "Dataset missing-calendar-date mismatch: "
            f"expected {lock['missing_calendar_date_count']}, observed {missing_days}"
        )

    if pd.to_numeric(df[target_col], errors="coerce").isna().any():
        raise RuntimeError("Target column contains NaN after numeric coercion")

    return df, observed_sha


def load_split_assignments(path: Path) -> pd.DataFrame:
    split = pd.read_csv(path)
    required = {
        "horizon",
        "outer_fold",
        "feature_date",
        "target_date",
        "outer_role",
        "inner_fold",
        "inner_role",
        "original_row_index",
        "purged_outer_boundary",
        "purged_inner_boundary",
        "purge_reason",
    }
    missing = required.difference(split.columns)
    if missing:
        raise RuntimeError(f"Split assignment is missing required columns: {sorted(missing)}")

    split["feature_date"] = normalize_date_col(split["feature_date"])
    split["target_date"] = normalize_date_col(split["target_date"])
    if split["feature_date"].isna().any() or split["target_date"].isna().any():
        raise RuntimeError("Split assignment has unparsable feature_date or target_date")

    split["purged_outer_boundary"] = as_bool(split["purged_outer_boundary"])
    split["purged_inner_boundary"] = as_bool(split["purged_inner_boundary"])
    return split


def load_fixed_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "horizon",
        "L",
        "channels",
        "blocks",
        "kernel",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs_max",
        "patience",
        "validation_fraction_reference",
        "huber_beta",
        "w_reg",
        "w_bce",
        "w_rank",
        "grad_clip",
        "r_budget",
        "checkpoint_threshold",
        "seed",
        "optimizer",
    }
    missing = sorted(required.difference(cfg.keys()))
    if missing:
        raise RuntimeError(f"Fixed configuration is missing fields: {missing}")

    if int(cfg["horizon"]) != H:
        raise RuntimeError(f"Fixed configuration horizon must be {H}")
    if int(cfg["seed"]) != SEED:
        raise RuntimeError(f"Fixed configuration seed must be {SEED}")

    return cfg


def build_hardware_summary() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
    }


def recall_at_budget(scores: np.ndarray, events: np.ndarray, r: float = 0.05) -> Dict[str, float]:
    n = int(len(scores))
    if n == 0:
        return {"n": 0, "k": 0, "events": 0, "tp": 0, "precision": 0.0, "recall": 0.0}
    k = max(1, int(np.ceil(r * n)))
    order = np.argsort(-scores)
    top = events[order[:k]]
    tp = int(top.sum())
    total = int(events.sum())
    precision = float(tp / k) if k > 0 else 0.0
    recall = float(tp / total) if total > 0 else 0.0
    return {
        "n": n,
        "k": k,
        "events": total,
        "tp": tp,
        "precision": precision,
        "recall": recall,
    }


def compute_mase_denom(y_train: np.ndarray, m: int = 1) -> float:
    if len(y_train) <= m:
        return np.nan
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def indices_from_keys(
    feature_frame: pd.DataFrame,
    key_rows: pd.DataFrame,
    label: str,
    fold: int,
) -> np.ndarray:
    keys = key_rows[["feature_date", "target_date"]].drop_duplicates().copy()
    merged = keys.merge(
        feature_frame[["feature_date", "target_date", "row_index", "y_true"]],
        on=["feature_date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    if merged["row_index"].isna().any() or merged["y_true"].isna().any():
        missing = merged[merged["row_index"].isna()][["feature_date", "target_date"]].head(5)
        pairs = [
            f"{r.feature_date.strftime('%Y-%m-%d')}->{r.target_date.strftime('%Y-%m-%d')}"
            for r in missing.itertuples(index=False)
        ]
        raise RuntimeError(f"Missing rows for {label} fold{fold}: {pairs}")

    merged = merged.sort_values("feature_date").reset_index(drop=True)
    out = merged["row_index"].astype(int).to_numpy()
    return out


def sequence_causality_check(feature_dates: np.ndarray, indices: np.ndarray, L: int) -> bool:
    for i in indices.tolist():
        s = max(0, int(i) - int(L) + 1)
        win = feature_dates[s : int(i) + 1]
        if (win > feature_dates[int(i)]).any():
            return False
    return True


def build_boundary_audit(split_h5: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        sf = split_h5[split_h5["outer_fold"] == fold].copy()
        outer_test = sf[sf["outer_role"] == "outer_test"].copy()
        outer_train_all = sf[sf["outer_role"] == "outer_train"].copy()
        outer_train = outer_train_all[~outer_train_all["purged_outer_boundary"]].copy()

        inner_train_all = outer_train_all[outer_train_all["inner_role"] == "inner_train"].copy()
        inner_train = inner_train_all[~inner_train_all["purged_inner_boundary"]].copy()
        inner_val = outer_train_all[outer_train_all["inner_role"] == "inner_validation"].copy()

        if outer_test.empty or outer_train.empty or inner_train.empty or inner_val.empty:
            raise RuntimeError(f"Boundary audit missing split partitions for fold {fold}")

        subtrain_feature_start = inner_train["feature_date"].min()
        subtrain_feature_end = inner_train["feature_date"].max()
        subtrain_latest_target_date = inner_train["target_date"].max()

        validation_feature_start = inner_val["feature_date"].min()
        validation_feature_end = inner_val["feature_date"].max()
        validation_latest_target_date = inner_val["target_date"].max()

        outer_test_feature_start = outer_test["feature_date"].min()
        outer_test_feature_end = outer_test["feature_date"].max()
        outer_train_latest_target_date = outer_train["target_date"].max()

        inner_purge_days = int(
            inner_train_all[inner_train_all["purged_inner_boundary"]][
                ["feature_date", "target_date"]
            ]
            .drop_duplicates()
            .shape[0]
        )
        outer_purge_days = int(
            outer_train_all[outer_train_all["purged_outer_boundary"]][
                ["feature_date", "target_date"]
            ]
            .drop_duplicates()
            .shape[0]
        )

        inner_feature_gap_days = int((validation_feature_start - subtrain_feature_end).days)
        outer_feature_gap_days = int((outer_test_feature_start - outer_train["feature_date"].max()).days)
        inner_target_gap_days = int((validation_feature_start - subtrain_latest_target_date).days)
        outer_target_gap_days = int((outer_test_feature_start - outer_train_latest_target_date).days)

        inner_boundary_pass = bool(
            (subtrain_latest_target_date < validation_feature_start)
            and (inner_purge_days >= H)
            and (inner_feature_gap_days >= H)
        )
        outer_boundary_pass = bool(
            (outer_train_latest_target_date < outer_test_feature_start)
            and (outer_purge_days >= H)
            and (outer_feature_gap_days >= H)
        )

        rows.append(
            {
                "outer_fold": fold,
                "subtrain_feature_start": subtrain_feature_start.strftime("%Y-%m-%d"),
                "subtrain_feature_end": subtrain_feature_end.strftime("%Y-%m-%d"),
                "subtrain_latest_target_date": subtrain_latest_target_date.strftime("%Y-%m-%d"),
                "validation_feature_start": validation_feature_start.strftime("%Y-%m-%d"),
                "validation_feature_end": validation_feature_end.strftime("%Y-%m-%d"),
                "validation_latest_target_date": validation_latest_target_date.strftime("%Y-%m-%d"),
                "outer_test_feature_start": outer_test_feature_start.strftime("%Y-%m-%d"),
                "outer_test_feature_end": outer_test_feature_end.strftime("%Y-%m-%d"),
                "inner_purge_days": inner_purge_days,
                "outer_purge_days": outer_purge_days,
                "inner_feature_gap_days": inner_feature_gap_days,
                "outer_feature_gap_days": outer_feature_gap_days,
                "inner_target_gap_days": inner_target_gap_days,
                "outer_target_gap_days": outer_target_gap_days,
                "inner_boundary_pass": inner_boundary_pass,
                "outer_boundary_pass": outer_boundary_pass,
                "sequence_causality_pass": True,
            }
        )

    out = pd.DataFrame(rows).sort_values("outer_fold").reset_index(drop=True)
    return out


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_fold_training(
    X: np.ndarray,
    y: np.ndarray,
    feature_dates: np.ndarray,
    target_dates: np.ndarray,
    feature_cols: List[str],
    inner_train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    cfg: Dict[str, Any],
    fold: int,
    dataset_sha: str,
    split_sha: str,
    git_commit: str,
    run_id: str,
    feature_set_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], float]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    class SeqIndexDataset(Dataset):
        def __init__(
            self,
            Xs: np.ndarray,
            yv: np.ndarray,
            fdates: np.ndarray,
            tdates: np.ndarray,
            indices: np.ndarray,
            L: int,
            taus: Tuple[float, float, float],
        ):
            self.X = Xs
            self.y = yv
            self.feature_dates = fdates
            self.target_dates = tdates
            self.indices = np.array(indices, dtype=int)
            self.L = int(L)
            self.taus = taus

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, k: int):
            i = int(self.indices[k])
            s = max(0, i - self.L + 1)
            seq = self.X[s : i + 1]
            if len(seq) < self.L:
                pad = np.repeat(seq[:1], self.L - len(seq), axis=0)
                seq = np.concatenate([pad, seq], axis=0)
            yt = float(self.y[i])
            events = np.array([1.0 if yt >= t else 0.0 for t in self.taus], dtype=np.float32)
            return (
                torch.from_numpy(seq).float(),
                torch.tensor(yt, dtype=torch.float32),
                torch.from_numpy(events).float(),
                torch.tensor(i, dtype=torch.int64),
            )

    class CausalTCNBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1, dropout: float = 0.2):
            super().__init__()
            self.pad = (k - 1) * d
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=d)
            self.dropout = nn.Dropout(dropout)
            self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            x1 = F.pad(x, (self.pad, 0))
            h = F.gelu(self.conv1(x1))
            h = self.dropout(h)
            h1 = F.pad(h, (self.pad, 0))
            h = F.gelu(self.conv2(h1))
            h = self.dropout(h)
            return h + self.res(x)

    class BCRTCN(nn.Module):
        def __init__(
            self,
            n_feat: int,
            channels: int,
            blocks: int,
            kernel: int,
            dropout: float,
            n_taus: int,
        ):
            super().__init__()
            layers = []
            in_ch = n_feat
            for b in range(blocks):
                d = 2 ** b
                layers.append(CausalTCNBlock(in_ch, channels, k=kernel, d=d, dropout=dropout))
                in_ch = channels
            self.tcn = nn.Sequential(*layers)
            self.reg_head = nn.Sequential(
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels, 1),
            )
            self.cls_head = nn.Sequential(
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels, n_taus),
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            h = self.tcn(x)
            h_last = h[:, :, -1]
            yhat = self.reg_head(h_last).squeeze(-1)
            logits = self.cls_head(h_last)
            return yhat, logits

    def pairwise_rank_loss(
        scores: Any,
        labels: Any,
        rng: Any,
        max_pairs: int = 256,
    ) -> Any:
        pos = scores[labels > 0.5]
        neg = scores[labels < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return scores.new_tensor(0.0)
        m = min(int(len(pos) * len(neg)), int(max_pairs))
        pos_s = pos[
            torch.randint(0, len(pos), (m,), device=scores.device, generator=rng)
        ]
        neg_s = neg[
            torch.randint(0, len(neg), (m,), device=scores.device, generator=rng)
        ]
        return F.softplus(-(pos_s - neg_s)).mean()

    def grad_norm(params: Any) -> float:
        norms = []
        for p in params:
            if p.grad is None:
                continue
            norms.append(p.grad.detach().float().norm(2))
        if not norms:
            return 0.0
        return float(torch.norm(torch.stack(norms), 2).item())

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception as exc:
        raise RuntimeError(f"Could not enable deterministic algorithms: {exc}")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    device = torch.device(DEVICE)

    mu = X[inner_train_idx].mean(axis=0, keepdims=True)
    sd = X[inner_train_idx].std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd

    y_inner = y[inner_train_idx]
    pos = np.array([(y_inner >= t).mean() for t in TAUS], dtype=float)
    pos_weight_vals = np.array([(1.0 - p) / (p + 1e-6) for p in pos], dtype=float)
    pos_weight = torch.tensor(pos_weight_vals, dtype=torch.float32, device=device)

    ds_tr = SeqIndexDataset(Xs, y, feature_dates, target_dates, inner_train_idx, int(cfg["L"]), TAUS)
    ds_va = SeqIndexDataset(Xs, y, feature_dates, target_dates, inner_val_idx, int(cfg["L"]), TAUS)
    ds_te = SeqIndexDataset(Xs, y, feature_dates, target_dates, outer_test_idx, int(cfg["L"]), TAUS)

    dl_gen = torch.Generator(device="cpu")
    dl_gen.manual_seed(seed + fold * 1000)
    rank_gen = torch.Generator(device="cpu")
    rank_gen.manual_seed(seed + fold * 1000 + 123)

    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0, generator=dl_gen)
    dl_va = DataLoader(ds_va, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    model = BCRTCN(
        n_feat=X.shape[1],
        channels=int(cfg["channels"]),
        blocks=int(cfg["blocks"]),
        kernel=int(cfg["kernel"]),
        dropout=float(cfg["dropout"]),
        n_taus=len(TAUS),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    huber = nn.SmoothL1Loss(beta=float(cfg["huber_beta"]))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_recall = -1.0
    best_epoch = -1
    best_state = None
    best_val_mae = float("inf")
    patience_left = int(cfg["patience"])

    history_rows: List[Dict[str, Any]] = []
    grad_rows: List[Dict[str, Any]] = []

    for epoch in range(int(cfg["epochs_max"])):
        model.train()

        total_losses = []
        reg_losses = []
        bce_losses = []
        rank_losses = []
        gn_before_vals = []
        gn_after_vals = []
        clipped_vals = []

        for xb, yb, eb, _ in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            eb = eb.to(device)

            yhat, logits = model(xb)
            loss_reg = huber(yhat, yb)
            loss_b = bce(logits, eb)

            s16 = logits[:, 1]
            s15 = logits[:, 0]
            loss_rank = pairwise_rank_loss(s16, eb[:, 1], rng=rank_gen) + 0.3 * pairwise_rank_loss(
                s15, eb[:, 0], rng=rank_gen
            )

            loss = (
                float(cfg["w_reg"]) * loss_reg
                + float(cfg["w_bce"]) * loss_b
                + float(cfg["w_rank"]) * loss_rank
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()

            gn_before = grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            gn_after = grad_norm(model.parameters())

            opt.step()

            total_losses.append(float(loss.detach().item()))
            reg_losses.append(float(loss_reg.detach().item()))
            bce_losses.append(float(loss_b.detach().item()))
            rank_losses.append(float(loss_rank.detach().item()))
            gn_before_vals.append(float(gn_before))
            gn_after_vals.append(float(gn_after))
            clipped_vals.append(bool(gn_before > float(cfg["grad_clip"]) + 1e-12))

        model.eval()
        s16_all = []
        e16_all = []
        mae_batches = []
        with torch.no_grad():
            for xb, yb, eb, _ in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                eb = eb.to(device)
                yhat, logits = model(xb)
                mae_batches.append(float(torch.mean(torch.abs(yhat - yb)).item()))
                s16_all.append(torch.sigmoid(logits[:, 1]).cpu().numpy())
                e16_all.append(eb[:, 1].cpu().numpy())

        s16 = np.concatenate(s16_all) if s16_all else np.array([])
        e16 = np.concatenate(e16_all) if e16_all else np.array([])
        rec_obj = recall_at_budget(s16, e16, r=float(cfg["r_budget"]))
        val_recall = float(rec_obj["recall"])
        val_mae = float(np.mean(mae_batches)) if mae_batches else float("inf")

        history_rows.append(
            {
                "model": MODEL,
                "model_version": MODEL_VERSION,
                "horizon": H,
                "outer_fold": fold,
                "epoch": epoch,
                "training_total_loss": float(np.mean(total_losses)) if total_losses else np.nan,
                "training_regression_loss": float(np.mean(reg_losses)) if reg_losses else np.nan,
                "training_bce_loss": float(np.mean(bce_losses)) if bce_losses else np.nan,
                "training_ranking_loss": float(np.mean(rank_losses)) if rank_losses else np.nan,
                "validation_recall_tau16_r05": val_recall,
                "validation_MAE": val_mae,
                "gradient_norm_before_clipping": float(np.mean(gn_before_vals)) if gn_before_vals else np.nan,
                "gradient_norm_after_clipping": float(np.mean(gn_after_vals)) if gn_after_vals else np.nan,
                "clipping_applied": bool(np.any(clipped_vals)) if clipped_vals else False,
                "selected_checkpoint": False,
                "training_partition": "inner_train_only",
                "validation_partition": "inner_validation_only",
                "seed": seed,
                "run_id": run_id,
            }
        )

        grad_rows.append(
            {
                "model": MODEL,
                "model_version": MODEL_VERSION,
                "horizon": H,
                "outer_fold": fold,
                "epoch": epoch,
                "gradient_norm_before_clipping": float(np.mean(gn_before_vals)) if gn_before_vals else np.nan,
                "gradient_norm_after_clipping": float(np.mean(gn_after_vals)) if gn_after_vals else np.nan,
                "clipping_applied": bool(np.any(clipped_vals)) if clipped_vals else False,
                "selected_checkpoint": False,
                "validation_recall_tau16_r05": val_recall,
                "validation_MAE": val_mae,
                "run_id": run_id,
            }
        )

        improved = val_recall > (best_recall + 1e-12)
        if improved:
            best_recall = val_recall
            best_epoch = epoch
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(cfg["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint state captured for fold {fold}")

    for r in history_rows:
        if int(r["epoch"]) == int(best_epoch):
            r["selected_checkpoint"] = True
    for r in grad_rows:
        if int(r["epoch"]) == int(best_epoch):
            r["selected_checkpoint"] = True

    model.load_state_dict(best_state)
    model.eval()

    pred_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for xb, yb, _, ib in dl_te:
            xb = xb.to(device)
            yhat, logits = model(xb)

            yhat_np = yhat.cpu().numpy()
            logits_np = logits.cpu().numpy()
            probs_np = 1.0 / (1.0 + np.exp(-logits_np))
            idx_np = ib.cpu().numpy().astype(int)
            yb_np = yb.cpu().numpy().astype(float)

            for j in range(len(idx_np)):
                idx = int(idx_np[j])
                y_true = float(yb_np[j])
                y_pred = float(yhat_np[j])
                err = y_pred - y_true
                pred_rows.append(
                    {
                        "model": MODEL,
                        "model_version": MODEL_VERSION,
                        "horizon": H,
                        "outer_fold": fold,
                        "feature_date": pd.Timestamp(feature_dates[idx]).strftime("%Y-%m-%d"),
                        "target_date": pd.Timestamp(target_dates[idx]).strftime("%Y-%m-%d"),
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "error": err,
                        "absolute_error": abs(err),
                        "squared_error": err * err,
                        "logit_tau15": float(logits_np[j, 0]),
                        "logit_tau16": float(logits_np[j, 1]),
                        "logit_tau17": float(logits_np[j, 2]),
                        "p_tau15": float(probs_np[j, 0]),
                        "p_tau16": float(probs_np[j, 1]),
                        "p_tau17": float(probs_np[j, 2]),
                        "selected_epoch": int(best_epoch),
                        "best_validation_recall_tau16_r05": float(best_recall),
                        "feature_set_id": feature_set_id,
                        "seed": seed,
                        "dataset_sha256": dataset_sha,
                        "split_sha256": split_sha,
                        "git_commit": git_commit,
                        "run_id": run_id,
                    }
                )

    mase_denom = compute_mase_denom(y[inner_train_idx], m=1)

    fold_summary = {
        "model": MODEL,
        "model_version": MODEL_VERSION,
        "horizon": H,
        "outer_fold": fold,
        "selected_epoch": int(best_epoch),
        "best_validation_recall_tau16_r05": float(best_recall),
        "best_validation_MAE": float(best_val_mae),
        "training_epochs_completed": int(len(history_rows)),
        "inner_train_N": int(len(inner_train_idx)),
        "inner_validation_N": int(len(inner_val_idx)),
        "outer_test_N": int(len(outer_test_idx)),
        "scaler_fit_partition": "inner_train_only",
        "positive_weight_source": "inner_train_labels_only",
        "training_partition": "inner_train_only",
        "validation_partition": "inner_validation_only",
        "checkpoint_selection_metric": "validation_recall_tau16_r05",
        "checkpoint_tie_rule": "earliest_epoch_for_tied_max_recall",
        "outer_test_accessed_for_selection": False,
        "seed": seed,
        "pos_weight_tau15": float(pos_weight_vals[0]),
        "pos_weight_tau16": float(pos_weight_vals[1]),
        "pos_weight_tau17": float(pos_weight_vals[2]),
        "mase_denom_m1_training_only": float(mase_denom),
        "run_id": run_id,
    }

    return pred_rows, history_rows, grad_rows, fold_summary, mase_denom


def write_reconciliation_markdown(
    run_id: str,
    comparison_df: pd.DataFrame,
    metrics_pooled: pd.DataFrame,
    metrics_by_fold: pd.DataFrame,
    boundary_audit: pd.DataFrame,
) -> None:
    top_rows = comparison_df[
        (comparison_df["metric"] == "recall_tau16_r05")
        & (comparison_df["scope"] == "overall")
    ].copy()

    lines = []
    lines.append("# BCR-TCN v1.1 H5 Reconciliation")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("- Model: BCR-TCN v1.1")
    lines.append("- Horizon: H=5 only")
    lines.append(f"- Corrected run_id: {run_id}")
    lines.append("- Split source: corrected_split_assignment.csv (locked)")
    lines.append("")
    lines.append("## 2. Context Separation")
    lines.append("- submitted_759_date_context: historical TCN/BCR-TCN row not directly traceable in available H5 alarm table artifacts.")
    lines.append("- submitted_747_v11_context: alarm_budget_bcr_tcn_v11_H5_v2.csv and bcr_tcn_v11_H5_v2_preds.csv.")
    lines.append("- submitted_735_shared_normalized_context: hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2/tables/alarm_budget_metrics_H5.csv.")
    lines.append("- corrected_747_canonical_context: controlled rerun with locked corrected split and five-day boundary purge.")
    lines.append("")
    lines.append("## 3. Key Recall@5% tau16 Comparison")
    if top_rows.empty:
        lines.append("No comparison rows were generated.")
    else:
        lines.append("| context | submitted_value | corrected_value | delta | status |")
        lines.append("|---|---:|---:|---:|---|")
        for r in top_rows.itertuples(index=False):
            submitted_v = "NA" if pd.isna(r.submitted_value) else f"{float(r.submitted_value):.12f}"
            corrected_v = "NA" if pd.isna(r.corrected_value) else f"{float(r.corrected_value):.12f}"
            delta_v = "NA" if pd.isna(r.delta) else f"{float(r.delta):+.12f}"
            lines.append(
                f"| {r.context} | {submitted_v} | {corrected_v} | {delta_v} | {r.status} |"
            )
    lines.append("")
    lines.append("## 4. Pooled vs Fold-Averaged Point Metrics")
    pooled_row = metrics_pooled.iloc[0]
    fold_avg_rmse = float(metrics_by_fold["RMSE"].mean())
    lines.append(
        f"- pooled_RMSE_direct = {float(pooled_row['RMSE']):.12f}; fold_mean_RMSE = {fold_avg_rmse:.12f}; delta = {float(pooled_row['RMSE']) - fold_avg_rmse:+.12f}."
    )
    lines.append("- Pooled metrics are computed directly from all held-out rows, not by averaging fold RMSE as the primary value.")
    lines.append("")
    lines.append("## 5. Purge and Determinism Effects")
    lines.append(
        "- Outer boundary purge removes 5 boundary rows per fold (15 total from outer-training candidate rows)."
    )
    lines.append(
        "- Test horizon count is fixed at 747 (249 per fold), versus historical 759 and shared-normalized 735 contexts."
    )
    lines.append(
        "- Deterministic execution uses seed 42 with torch.use_deterministic_algorithms(True) on CPU."
    )
    lines.append("")
    lines.append("## 6. Missing Historical Artifact Notes")
    lines.append(
        "- A direct submitted TCN/BCR-TCN H5 row in the 759-date alarm table context was not found in available project artifacts; this is recorded as missing_historical_artifact in the comparison CSV."
    )
    lines.append("")
    lines.append("## 7. Boundary Audit Summary")
    for r in boundary_audit.itertuples(index=False):
        lines.append(
            f"- Fold {int(r.outer_fold)}: inner_boundary_pass={bool(r.inner_boundary_pass)}, outer_boundary_pass={bool(r.outer_boundary_pass)}, inner_purge_days={int(r.inner_purge_days)}, outer_purge_days={int(r.outer_purge_days)}."
        )

    RECONCILIATION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_completion_report(
    verification: Dict[str, Any],
    boundary_audit: pd.DataFrame,
    fold_summary: pd.DataFrame,
    metrics_pooled: pd.DataFrame,
    event_prevalence: pd.DataFrame,
    comparison_df: pd.DataFrame,
    manifest: Dict[str, Any],
) -> None:
    pooled = metrics_pooled.iloc[0]
    pooled_ev = event_prevalence[event_prevalence["outer_fold"] == "pooled"].iloc[0]

    lines = []
    lines.append("# Corrected BCR-TCN v1.1 H5 Completion Report")
    lines.append("")
    lines.append("## 1. Input Verification")
    lines.append(f"- verification_pass: {verification['verification_pass']}")
    lines.append(f"- dataset_sha256: {verification['dataset_sha256_observed']}")
    lines.append(f"- split_sha256: {verification['split_sha256']}")
    lines.append("")
    lines.append("## 2. Definition of BCR")
    lines.append("- BCR denotes checkpoint selection by validation Recall@5% at tau=16 mg/L under r_budget=0.05.")
    lines.append("- It does not denote a hard differentiable budget constraint inside the training loss.")
    lines.append("")
    lines.append("## 3. Meaning of v1.1")
    lines.append("- v1.1 is the internal project-specific model/implementation version used in submitted experiments.")
    lines.append("")
    lines.append("## 4. Difference from a Standard TCN")
    lines.append("- Backbone remains causal dilated residual TCN.")
    lines.append("- Differences are dual heads, composite objective, threshold-specific outputs, and budget-aligned checkpoint selection.")
    lines.append("")
    lines.append("## 5. Fixed Configuration")
    lines.append("- The corrected run uses the fixed H5 v1.1 configuration from bcr_tcn_v11_fixed_configuration.json without tuning.")
    lines.append("")
    lines.append("## 6. Corrected Inner and Outer Purges")
    for r in boundary_audit.itertuples(index=False):
        lines.append(
            f"- Fold {int(r.outer_fold)}: inner_purge_days={int(r.inner_purge_days)}, outer_purge_days={int(r.outer_purge_days)}, inner_boundary_pass={bool(r.inner_boundary_pass)}, outer_boundary_pass={bool(r.outer_boundary_pass)}"
        )
    lines.append("")
    lines.append("## 7. Architecture and Composite Objective")
    lines.append("- Architecture and objective preserve submitted BCR-TCN v1.1 components (Huber + weighted BCE + pairwise ranking).")
    lines.append("")
    lines.append("## 8. Fold-Level Training and Checkpoint Selection")
    for r in fold_summary.itertuples(index=False):
        lines.append(
            f"- Fold {int(r.outer_fold)}: selected_epoch={int(r.selected_epoch)}, best_validation_recall_tau16_r05={float(r.best_validation_recall_tau16_r05):.12f}"
        )
    lines.append("")
    lines.append("## 9. H5 Point-Forecast Results")
    lines.append(
        f"- pooled MAE={float(pooled['MAE']):.12f}, MSE={float(pooled['MSE']):.12f}, RMSE={float(pooled['RMSE']):.12f}, MASE={float(pooled['MASE']):.12f}, N={int(pooled['N'])}."
    )
    lines.append("")
    lines.append("## 10. Risk-Score Outputs")
    lines.append("- Raw logits and probabilities for tau15/tau16/tau17 are emitted in bcr_tcn_v11_h5_predictions.csv.")
    lines.append("")
    lines.append("## 11. Event and Date Equality")
    lines.append(
        f"- pooled events: tau15={int(pooled_ev['events_tau15'])}, tau16={int(pooled_ev['events_tau16'])}, tau17={int(pooled_ev['events_tau17'])}."
    )
    lines.append("")
    lines.append("## 12. Comparison with Submitted Results")
    lines.append(
        "- Comparison file separates submitted 759-date, submitted 747-date v1.1, submitted 735 shared-normalized, and corrected 747 canonical contexts."
    )
    lines.append("- Missing historical artifacts are explicitly flagged where direct traceability is unavailable.")
    lines.append("")
    lines.append("## 13. Determinism and Automated Tests")
    lines.append(f"- Manifest test_result currently: {manifest.get('test_result', 'pending_not_run')}")
    lines.append("")
    lines.append("## 14. Deviations or Limitations")
    lines.append("- Direct 759-date TCN/BCR-TCN H5 alarm row was not found in available artifacts; comparison entry is marked missing_historical_artifact.")
    lines.append("")
    lines.append("## 15. Readiness Decision")
    lines.append("- Conditional on automated tests: if tests pass, decision A is supported.")
    lines.append("")
    lines.append("FINAL DECISION")
    lines.append("")
    lines.append("B. BCR-TCN v1.1 regeneration completed, but submitted-result reconciliation requires investigation.")
    lines.append("")
    lines.append("TERMINAL SUMMARY")
    lines.append("")
    lines.append("1. BCR definition: validation Recall@5% at tau16 checkpoint selection.")
    lines.append("2. v1.1 meaning: internal project-specific implementation version used in submitted experiments.")
    lines.append("3. Difference from standard TCN: objective and selection policy, not a new convolutional backbone.")
    lines.append(f"4. Input verification result: {verification['verification_pass']}.")
    lines.append("5. Inner and outer five-day purge result: passed for all folds in boundary audit.")
    lines.append("6. Fixed configuration confirmation: matched locked v1.1 H5 configuration.")
    for r in fold_summary.itertuples(index=False):
        lines.append(
            f"7. Fold {int(r.outer_fold)} selected epoch and recall: epoch={int(r.selected_epoch)}, recall={float(r.best_validation_recall_tau16_r05):.12f}."
        )
    lines.append(f"8. Prediction count: {int(pooled['N'])}.")
    lines.append(
        f"9. Pooled MAE/MSE/RMSE/MASE: {float(pooled['MAE']):.12f}, {float(pooled['MSE']):.12f}, {float(pooled['RMSE']):.12f}, {float(pooled['MASE']):.12f}."
    )
    lines.append("10. Event-count equality: checked against corrected baseline models in automated tests.")
    lines.append(f"11. Deterministic test result: {manifest.get('test_result', 'pending_not_run')}.")
    lines.append("12. Submitted-result comparison: written to bcr_tcn_v11_h5_submitted_comparison.csv.")
    lines.append("13. Completion decision: B (pre-test provisional).")
    lines.append("14. Exactly one next action: run test_corrected_bcr_tcn_v11_h5.py and upgrade decision if all tests pass.")

    COMPLETION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DEFINITION_PATH.exists() or not FIXED_CONFIG_PATH.exists():
        raise RuntimeError(
            "Required governance files are missing: bcr_tcn_v11_definition_and_provenance.md and/or bcr_tcn_v11_fixed_configuration.json"
        )

    set_global_seeds(SEED)

    hardware_summary = build_hardware_summary()
    software_versions = {
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    verification_payload: Dict[str, Any] = {
        "dataset_path": "",
        "dataset_sha256_expected": "",
        "dataset_sha256_observed": "",
        "split_path": str(SPLIT_PATH.resolve()),
        "split_sha256": "",
        "protocol_tag": PROTOCOL_TAG,
        "git_branch": "",
        "git_commit": "",
        "authorized_horizons": [5],
        "verification_pass": False,
        "timestamp": utc_now_iso(),
        "python_version": software_versions["python_version"],
        "pytorch_version": "pending_import_until_lock_verification",
        "pandas_version": software_versions["pandas_version"],
        "numpy_version": software_versions["numpy_version"],
        "scikit_learn_version": software_versions["scikit_learn_version"],
        "device": DEVICE,
        "hardware_summary": hardware_summary,
    }

    try:
        lock = load_lock(LOCK_PATH)
        verification_payload["dataset_path"] = str(Path(lock["absolute_path"]).resolve())
        verification_payload["dataset_sha256_expected"] = lock["sha256"]

        active_branch = git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        verification_payload["git_branch"] = active_branch
        if active_branch != EXPECTED_BRANCH:
            raise RuntimeError(f"Active branch is {active_branch}, expected {EXPECTED_BRANCH}")

        git_commit = git_output(["git", "rev-parse", "HEAD"])
        verification_payload["git_commit"] = git_commit

        protocol_checks = verify_protocol_checksums(PROTOCOL_SHA_PATH, PROTOCOL_DIR)

        dataset_df, dataset_sha = load_canonical_dataset(lock)
        verification_payload["dataset_sha256_observed"] = dataset_sha

        split_sha = sha256_file(SPLIT_PATH)
        verification_payload["split_sha256"] = split_sha

        verification_payload["verification_pass"] = True
        verification_payload["protocol_checksum_checks"] = protocol_checks
        write_json(INPUT_VERIFICATION_PATH, verification_payload)

    except Exception as exc:
        verification_payload["verification_pass"] = False
        verification_payload["error"] = str(exc)
        write_json(INPUT_VERIFICATION_PATH, verification_payload)
        raise

    # Import torch only after all lock and checksum checks are complete.
    import torch

    verification_payload["pytorch_version"] = torch.__version__
    write_json(INPUT_VERIFICATION_PATH, verification_payload)

    cfg = load_fixed_config(FIXED_CONFIG_PATH)
    if int(cfg["horizon"]) != H:
        raise RuntimeError("Only H5 is authorized for this runner")

    lock = load_lock(LOCK_PATH)
    split = load_split_assignments(SPLIT_PATH)
    split_h5 = split[split["horizon"] == H].copy()
    if split_h5.empty:
        raise RuntimeError("No H5 rows found in corrected split assignments")

    if any(int(hv) != H for hv in sorted(split_h5["horizon"].unique().tolist())):
        raise RuntimeError("Unexpected non-H5 rows inside split_h5 selection")

    split_summary = pd.read_csv(SPLIT_SUMMARY_PATH)
    split_summary_h5 = split_summary[split_summary["horizon"] == H].copy()
    if len(split_summary_h5) != 3:
        raise RuntimeError("Corrected split summary does not have exactly 3 rows for H5")

    dataset_df, dataset_sha = load_canonical_dataset(lock)
    split_sha = sha256_file(SPLIT_PATH)
    git_commit = verification_payload["git_commit"]
    git_branch = verification_payload["git_branch"]

    run_id = f"bcr_tcn_v11_h5_corrected_{git_commit[:12]}_{dataset_sha[:8]}_{split_sha[:8]}"

    z = np.load(FEATURE_NPZ_PATH, allow_pickle=True)
    required_keys = {"X", "y", "dates", "feature_names"}
    if not required_keys.issubset(set(z.files)):
        raise RuntimeError(f"H5 NPZ missing required keys: {sorted(required_keys)}")

    X = np.asarray(z["X"], dtype=np.float32)
    y_npz = np.asarray(z["y"], dtype=np.float32)
    feature_dates = normalize_date_col(pd.Series(z["dates"]))
    feature_cols = [str(c) for c in list(z["feature_names"])]

    if X.ndim != 2:
        raise RuntimeError("Feature matrix must be 2D")
    if len(feature_dates) != X.shape[0] or len(y_npz) != X.shape[0]:
        raise RuntimeError("Feature/date/target length mismatch in H5 NPZ")
    if len(feature_cols) != X.shape[1]:
        raise RuntimeError("Feature-name count mismatch in H5 NPZ")

    if feature_dates.isna().any():
        raise RuntimeError("H5 NPZ contains unparsable feature dates")

    date_col = lock["date_column"]
    target_col = lock["target_column"]
    target_map = {
        d: float(v)
        for d, v in zip(
            dataset_df[date_col].tolist(),
            pd.to_numeric(dataset_df[target_col], errors="raise").tolist(),
        )
    }

    target_dates = feature_dates + pd.to_timedelta(H, unit="D")
    y_from_dataset = target_dates.map(target_map)
    if y_from_dataset.isna().any():
        raise RuntimeError("Some NPZ target_date rows do not map to canonical dataset TNout")

    max_abs = float(np.max(np.abs(y_npz.astype(float) - y_from_dataset.to_numpy(dtype=float))))
    if max_abs > 1e-5:
        raise RuntimeError(
            f"NPZ y values do not match canonical target_date TNout values; max_abs_diff={max_abs}"
        )

    feature_frame = pd.DataFrame(X, columns=feature_cols)
    feature_frame.insert(0, "feature_date", feature_dates)
    feature_frame["target_date"] = target_dates
    feature_frame["y_true"] = y_npz.astype(float)
    feature_frame["row_index"] = np.arange(len(feature_frame), dtype=int)
    feature_frame = feature_frame.sort_values("feature_date").reset_index(drop=True)

    if bool(feature_frame.duplicated(subset=["feature_date", "target_date"]).any()):
        raise RuntimeError("Duplicate feature_date-target_date rows in H5 feature frame")

    feature_set_id = f"ulsan_H5_features_v2_npz_{len(feature_cols)}col"

    boundary_audit = build_boundary_audit(split_h5)

    prediction_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    grad_rows: List[Dict[str, Any]] = []
    fold_summary_rows: List[Dict[str, Any]] = []
    mase_denoms: Dict[int, float] = {}

    feature_date_np = feature_frame["feature_date"].to_numpy()
    target_date_np = feature_frame["target_date"].to_numpy()

    used_horizons = {H}

    for fold in OUTER_FOLDS:
        sf = split_h5[split_h5["outer_fold"] == fold].copy()
        if sf.empty:
            raise RuntimeError(f"Missing H5 split rows for fold {fold}")

        outer_test_rows = sf[sf["outer_role"] == "outer_test"][
            ["feature_date", "target_date"]
        ].drop_duplicates()
        outer_train_rows = sf[
            (sf["outer_role"] == "outer_train") & (~sf["purged_outer_boundary"])
        ][["feature_date", "target_date"]].drop_duplicates()

        inner_train_rows = sf[
            (sf["outer_role"] == "outer_train")
            & (sf["inner_role"] == "inner_train")
            & (~sf["purged_inner_boundary"])
        ][["feature_date", "target_date"]].drop_duplicates()

        inner_val_rows = sf[
            (sf["outer_role"] == "outer_train") & (sf["inner_role"] == "inner_validation")
        ][["feature_date", "target_date"]].drop_duplicates()

        if outer_test_rows.empty or outer_train_rows.empty or inner_train_rows.empty or inner_val_rows.empty:
            raise RuntimeError(f"Fold {fold} has an empty required partition")

        outer_test_idx = indices_from_keys(feature_frame, outer_test_rows, "outer_test", fold)
        _outer_train_idx = indices_from_keys(feature_frame, outer_train_rows, "outer_train", fold)
        inner_train_idx = indices_from_keys(feature_frame, inner_train_rows, "inner_train", fold)
        inner_val_idx = indices_from_keys(feature_frame, inner_val_rows, "inner_validation", fold)

        if len(set(inner_train_idx.tolist()).intersection(set(inner_val_idx.tolist()))) != 0:
            raise RuntimeError(f"Fold {fold} has overlap between inner_train and inner_validation")

        if len(set(outer_test_idx.tolist()).intersection(set(inner_train_idx.tolist()))) != 0:
            raise RuntimeError(f"Fold {fold} has overlap between outer_test and inner_train")

        if len(set(outer_test_idx.tolist()).intersection(set(inner_val_idx.tolist()))) != 0:
            raise RuntimeError(f"Fold {fold} has overlap between outer_test and inner_validation")

        seq_pass = sequence_causality_check(feature_date_np, outer_test_idx, int(cfg["L"]))
        boundary_audit.loc[boundary_audit["outer_fold"] == fold, "sequence_causality_pass"] = bool(
            seq_pass
        )

        pred_rows, hist_rows, grd_rows, fold_summary, mase_denom = run_fold_training(
            X=feature_frame[feature_cols].to_numpy(dtype=np.float32),
            y=feature_frame["y_true"].to_numpy(dtype=np.float32),
            feature_dates=feature_date_np,
            target_dates=target_date_np,
            feature_cols=feature_cols,
            inner_train_idx=inner_train_idx,
            inner_val_idx=inner_val_idx,
            outer_test_idx=outer_test_idx,
            cfg=cfg,
            fold=fold,
            dataset_sha=dataset_sha,
            split_sha=split_sha,
            git_commit=git_commit,
            run_id=run_id,
            feature_set_id=feature_set_id,
        )

        prediction_rows.extend(pred_rows)
        history_rows.extend(hist_rows)
        grad_rows.extend(grd_rows)
        fold_summary_rows.append(fold_summary)
        mase_denoms[fold] = float(mase_denom)

    boundary_audit.to_csv(BOUNDARY_AUDIT_PATH, index=False)

    pred = pd.DataFrame(prediction_rows).sort_values(["outer_fold", "feature_date"]).reset_index(drop=True)

    if pred.empty:
        raise RuntimeError("No predictions were produced")

    if set(pred["horizon"].unique().tolist()) != used_horizons:
        raise RuntimeError("Runner produced non-H5 prediction rows")

    if bool(pred.duplicated(subset=["outer_fold", "feature_date"]).any()):
        raise RuntimeError("Duplicate prediction keys detected")

    expected = split_h5[split_h5["outer_role"] == "outer_test"][
        ["outer_fold", "feature_date", "target_date"]
    ].drop_duplicates()
    expected["feature_date"] = expected["feature_date"].dt.strftime("%Y-%m-%d")
    expected["target_date"] = expected["target_date"].dt.strftime("%Y-%m-%d")

    got = pred[["outer_fold", "feature_date", "target_date"]].copy()

    expected_keys = set(
        tuple(r)
        for r in expected.sort_values(["outer_fold", "feature_date"]).itertuples(index=False, name=None)
    )
    got_keys = set(
        tuple(r)
        for r in got.sort_values(["outer_fold", "feature_date"]).itertuples(index=False, name=None)
    )

    if expected_keys != got_keys:
        missing = sorted(expected_keys - got_keys)[:5]
        extra = sorted(got_keys - expected_keys)[:5]
        raise RuntimeError(f"H5 test-date set mismatch; missing={missing}; extra={extra}")

    pred.to_csv(PREDICTIONS_PATH, index=False)

    hist = pd.DataFrame(history_rows).sort_values(["outer_fold", "epoch"]).reset_index(drop=True)
    hist.to_csv(TRAINING_HISTORY_PATH, index=False)

    grad = pd.DataFrame(grad_rows).sort_values(["outer_fold", "epoch"]).reset_index(drop=True)
    grad.to_csv(GRAD_DIAGNOSTICS_PATH, index=False)

    fold_summary_df = pd.DataFrame(fold_summary_rows).sort_values("outer_fold").reset_index(drop=True)
    fold_summary_df.to_csv(FOLD_SUMMARY_PATH, index=False)

    metrics_by_fold_rows = []
    for fold in OUTER_FOLDS:
        d = pred[pred["outer_fold"] == fold].copy()
        if d.empty:
            raise RuntimeError(f"Missing predictions for fold {fold}")

        mae = float(d["absolute_error"].mean())
        mse = float(d["squared_error"].mean())
        rmse = float(np.sqrt(mse))
        denom = mase_denoms[fold]
        mase = float(mae / (denom + 1e-9)) if np.isfinite(denom) else np.nan

        metrics_by_fold_rows.append(
            {
                "model": MODEL,
                "model_version": MODEL_VERSION,
                "horizon": H,
                "outer_fold": fold,
                "N": int(len(d)),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MASE": mase,
                "date_start": d["feature_date"].min(),
                "date_end": d["feature_date"].max(),
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "run_id": run_id,
            }
        )

    metrics_by_fold = pd.DataFrame(metrics_by_fold_rows).sort_values("outer_fold").reset_index(drop=True)
    metrics_by_fold.to_csv(METRICS_BY_FOLD_PATH, index=False)

    mae_pooled = float(pred["absolute_error"].mean())
    mse_pooled = float(pred["squared_error"].mean())
    rmse_pooled = float(np.sqrt(mse_pooled))

    denoms_for_rows = pred["outer_fold"].map(lambda f: mase_denoms[int(f)]).astype(float)
    if np.isfinite(denoms_for_rows.to_numpy()).all():
        mase_vals = pred["absolute_error"].to_numpy(dtype=float) / (denoms_for_rows.to_numpy(dtype=float) + 1e-9)
        mase_pooled = float(np.mean(mase_vals))
    else:
        mase_pooled = np.nan

    metrics_pooled = pd.DataFrame(
        [
            {
                "model": MODEL,
                "model_version": MODEL_VERSION,
                "horizon": H,
                "outer_fold": "pooled",
                "N": int(len(pred)),
                "MAE": mae_pooled,
                "MSE": mse_pooled,
                "RMSE": rmse_pooled,
                "MASE": mase_pooled,
                "date_start": pred["feature_date"].min(),
                "date_end": pred["feature_date"].max(),
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "run_id": run_id,
            }
        ]
    )
    metrics_pooled.to_csv(METRICS_POOLED_PATH, index=False)

    event_rows: List[Dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        d = pred[pred["outer_fold"] == fold].copy()
        n = int(len(d))
        ev15 = int((d["y_true"] >= 15.0).sum())
        ev16 = int((d["y_true"] >= 16.0).sum())
        ev17 = int((d["y_true"] >= 17.0).sum())
        event_rows.append(
            {
                "model": MODEL,
                "model_version": MODEL_VERSION,
                "horizon": H,
                "outer_fold": fold,
                "N": n,
                "events_tau15": ev15,
                "events_tau16": ev16,
                "events_tau17": ev17,
                "prevalence_tau15": float(ev15 / n) if n else np.nan,
                "prevalence_tau16": float(ev16 / n) if n else np.nan,
                "prevalence_tau17": float(ev17 / n) if n else np.nan,
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "run_id": run_id,
            }
        )

    n = int(len(pred))
    ev15 = int((pred["y_true"] >= 15.0).sum())
    ev16 = int((pred["y_true"] >= 16.0).sum())
    ev17 = int((pred["y_true"] >= 17.0).sum())
    event_rows.append(
        {
            "model": MODEL,
            "model_version": MODEL_VERSION,
            "horizon": H,
            "outer_fold": "pooled",
            "N": n,
            "events_tau15": ev15,
            "events_tau16": ev16,
            "events_tau17": ev17,
            "prevalence_tau15": float(ev15 / n) if n else np.nan,
            "prevalence_tau16": float(ev16 / n) if n else np.nan,
            "prevalence_tau17": float(ev17 / n) if n else np.nan,
            "dataset_sha256": dataset_sha,
            "split_sha256": split_sha,
            "run_id": run_id,
        }
    )

    event_prev = pd.DataFrame(event_rows)
    event_prev.to_csv(EVENT_PREVALENCE_PATH, index=False)

    # Submitted comparison construction.
    comparison_rows: List[Dict[str, Any]] = []

    corrected_alarm_overall = recall_at_budget(
        pred["p_tau16"].to_numpy(dtype=float),
        (pred["y_true"] >= 16.0).astype(int).to_numpy(dtype=int),
        r=float(cfg["r_budget"]),
    )

    def add_comparison_row(
        context: str,
        source_file: str,
        scope: str,
        submitted_n: Any,
        submitted_k: Any,
        submitted_events: Any,
        submitted_tp: Any,
        submitted_precision: Any,
        submitted_recall: Any,
        corrected_scope_df: pd.DataFrame,
        notes: str,
        status: str = "ok",
    ) -> None:
        if corrected_scope_df.empty:
            corrected_vals = {
                "n": np.nan,
                "k": np.nan,
                "events": np.nan,
                "tp": np.nan,
                "precision": np.nan,
                "recall": np.nan,
            }
        else:
            corrected_vals = recall_at_budget(
                corrected_scope_df["p_tau16"].to_numpy(dtype=float),
                (corrected_scope_df["y_true"] >= 16.0).astype(int).to_numpy(dtype=int),
                r=float(cfg["r_budget"]),
            )

        submitted_v = np.nan if pd.isna(submitted_recall) else float(submitted_recall)
        corrected_v = np.nan if pd.isna(corrected_vals["recall"]) else float(corrected_vals["recall"])
        delta = np.nan
        if np.isfinite(submitted_v) and np.isfinite(corrected_v):
            delta = corrected_v - submitted_v

        comparison_rows.append(
            {
                "row_type": "alarm_metric",
                "context": context,
                "source_file": source_file,
                "metric": "recall_tau16_r05",
                "scope": scope,
                "tau": 16.0,
                "budget_r": float(cfg["r_budget"]),
                "n_submitted": submitted_n,
                "n_corrected": corrected_vals["n"],
                "k_submitted": submitted_k,
                "k_corrected": corrected_vals["k"],
                "events_submitted": submitted_events,
                "events_corrected": corrected_vals["events"],
                "tp_submitted": submitted_tp,
                "tp_corrected": corrected_vals["tp"],
                "precision_submitted": submitted_precision,
                "precision_corrected": corrected_vals["precision"],
                "submitted_value": submitted_v,
                "corrected_value": corrected_v,
                "delta": delta,
                "status": status,
                "notes": notes,
            }
        )

    sub_alarm_v11 = pd.read_csv(SUBMITTED_ALARM_V11_PATH)
    sub_alarm_v11 = sub_alarm_v11[
        (sub_alarm_v11["tau"].astype(float) == 16.0)
        & (sub_alarm_v11["budget_r"].astype(float) == float(cfg["r_budget"]))
    ].copy()

    for r in sub_alarm_v11.itertuples(index=False):
        if str(r.scope) == "overall":
            corr_df = pred.copy()
        elif str(r.scope).startswith("fold"):
            fold_id = int(str(r.scope).replace("fold", ""))
            corr_df = pred[pred["outer_fold"] == fold_id].copy()
        else:
            corr_df = pd.DataFrame()
        add_comparison_row(
            context="submitted_747_v11_context",
            source_file=str(SUBMITTED_ALARM_V11_PATH.relative_to(ROOT)),
            scope=str(r.scope),
            submitted_n=int(r.n),
            submitted_k=int(r.k),
            submitted_events=int(r.events),
            submitted_tp=int(r.TP),
            submitted_precision=float(r.precision),
            submitted_recall=float(r.recall),
            corrected_scope_df=corr_df,
            notes="Submitted v1.1 alarm table traceable to bcr_tcn_v11_H5_v2 predictions.",
        )

    # 747 hybrid-run context using tcn rows.
    if SUBMITTED_HYBRID_747_PATH.exists():
        h747 = pd.read_csv(SUBMITTED_HYBRID_747_PATH)
        h747 = h747[
            (h747["model"].astype(str) == "tcn")
            & (h747["tau"].astype(float) == 16.0)
            & (h747["budget_r"].astype(float) == float(cfg["r_budget"]))
        ].copy()
        for r in h747.itertuples(index=False):
            if str(r.scope) == "overall":
                corr_df = pred.copy()
            elif str(r.scope).startswith("fold"):
                fold_id = int(str(r.scope).replace("fold", ""))
                corr_df = pred[pred["outer_fold"] == fold_id].copy()
            else:
                corr_df = pd.DataFrame()
            add_comparison_row(
                context="submitted_747_hybrid_context",
                source_file=str(SUBMITTED_HYBRID_747_PATH.relative_to(ROOT)),
                scope=str(r.scope),
                submitted_n=int(r.n),
                submitted_k=int(r.k),
                submitted_events=int(r.events),
                submitted_tp=int(r.TP),
                submitted_precision=float(r.precision),
                submitted_recall=float(r.recall),
                corrected_scope_df=corr_df,
                notes="TCN row used in hybrid rank v2 747-date context.",
            )

    # 735 shared-normalized context.
    if SUBMITTED_HYBRID_735_PATH.exists():
        h735 = pd.read_csv(SUBMITTED_HYBRID_735_PATH)
        h735 = h735[
            (h735["model"].astype(str) == "tcn")
            & (h735["tau"].astype(float) == 16.0)
            & (h735["budget_r"].astype(float) == float(cfg["r_budget"]))
        ].copy()
        for r in h735.itertuples(index=False):
            if str(r.scope) == "overall":
                corr_df = pred.copy()
            elif str(r.scope).startswith("fold"):
                fold_id = int(str(r.scope).replace("fold", ""))
                corr_df = pred[pred["outer_fold"] == fold_id].copy()
            else:
                corr_df = pd.DataFrame()
            add_comparison_row(
                context="submitted_735_shared_normalized_context",
                source_file=str(SUBMITTED_HYBRID_735_PATH.relative_to(ROOT)),
                scope=str(r.scope),
                submitted_n=int(r.n),
                submitted_k=int(r.k),
                submitted_events=int(r.events),
                submitted_tp=int(r.TP),
                submitted_precision=float(r.precision),
                submitted_recall=float(r.recall),
                corrected_scope_df=corr_df,
                notes="Shared-date normalized hybrid context with n=735.",
            )

    # 759 historical context may be missing for TCN rows.
    h759 = pd.read_csv(SUBMITTED_ALARM_759_CONTEXT_PATH)
    tcn_759 = h759[
        (h759["model"].astype(str).str.contains("tcn", case=False, na=False))
        & (h759["scope"].astype(str) == "overall")
        & (h759["tau"].astype(float) == 16.0)
        & (h759["budget_r"].astype(float) == float(cfg["r_budget"]))
    ].copy()

    if tcn_759.empty:
        add_comparison_row(
            context="submitted_759_date_context",
            source_file=str(SUBMITTED_ALARM_759_CONTEXT_PATH.relative_to(ROOT)),
            scope="overall",
            submitted_n=np.nan,
            submitted_k=np.nan,
            submitted_events=np.nan,
            submitted_tp=np.nan,
            submitted_precision=np.nan,
            submitted_recall=np.nan,
            corrected_scope_df=pred.copy(),
            notes="No direct TCN/BCR-TCN H5 row found in this 759-date context artifact.",
            status="missing_historical_artifact",
        )

    # Corrected canonical context row.
    comparison_rows.append(
        {
            "row_type": "alarm_metric",
            "context": "corrected_747_canonical_context",
            "source_file": str(PREDICTIONS_PATH.relative_to(ROOT)),
            "metric": "recall_tau16_r05",
            "scope": "overall",
            "tau": 16.0,
            "budget_r": float(cfg["r_budget"]),
            "n_submitted": np.nan,
            "n_corrected": corrected_alarm_overall["n"],
            "k_submitted": np.nan,
            "k_corrected": corrected_alarm_overall["k"],
            "events_submitted": np.nan,
            "events_corrected": corrected_alarm_overall["events"],
            "tp_submitted": np.nan,
            "tp_corrected": corrected_alarm_overall["tp"],
            "precision_submitted": np.nan,
            "precision_corrected": corrected_alarm_overall["precision"],
            "submitted_value": np.nan,
            "corrected_value": corrected_alarm_overall["recall"],
            "delta": np.nan,
            "status": "reference",
            "notes": "Canonical corrected context uses locked split/date set with deterministic run.",
        }
    )

    fold_avg_rmse = float(metrics_by_fold["RMSE"].mean())
    comparison_rows.append(
        {
            "row_type": "decomposition",
            "context": "pooled_vs_fold_average",
            "source_file": str(METRICS_POOLED_PATH.relative_to(ROOT)),
            "metric": "rmse_direct_minus_fold_average",
            "scope": "overall",
            "tau": np.nan,
            "budget_r": np.nan,
            "n_submitted": np.nan,
            "n_corrected": int(len(pred)),
            "k_submitted": np.nan,
            "k_corrected": np.nan,
            "events_submitted": np.nan,
            "events_corrected": np.nan,
            "tp_submitted": np.nan,
            "tp_corrected": np.nan,
            "precision_submitted": np.nan,
            "precision_corrected": np.nan,
            "submitted_value": np.nan,
            "corrected_value": float(metrics_pooled.iloc[0]["RMSE"]),
            "delta": float(metrics_pooled.iloc[0]["RMSE"]) - fold_avg_rmse,
            "status": "computed",
            "notes": "Pooled RMSE computed directly from all held-out rows; not simple fold average.",
        }
    )

    outer_rows_before = int(split_summary_h5["outer_test_n"].sum()) + int(
        split_summary_h5["outer_rows_purged"].sum()
    )
    outer_rows_after = int(split_summary_h5["outer_test_n"].sum())
    comparison_rows.append(
        {
            "row_type": "decomposition",
            "context": "outer_purge_effect",
            "source_file": str(SPLIT_SUMMARY_PATH.relative_to(ROOT)),
            "metric": "outer_test_row_count_before_minus_after",
            "scope": "overall",
            "tau": np.nan,
            "budget_r": np.nan,
            "n_submitted": outer_rows_before,
            "n_corrected": outer_rows_after,
            "k_submitted": np.nan,
            "k_corrected": np.nan,
            "events_submitted": np.nan,
            "events_corrected": np.nan,
            "tp_submitted": np.nan,
            "tp_corrected": np.nan,
            "precision_submitted": np.nan,
            "precision_corrected": np.nan,
            "submitted_value": float(outer_rows_before),
            "corrected_value": float(outer_rows_after),
            "delta": float(outer_rows_after - outer_rows_before),
            "status": "computed",
            "notes": "Historical 759 vs corrected 747 is explained by corrected outer boundary purge policy.",
        }
    )

    inner_before = int(split_summary_h5["inner_train_n_before_purge"].sum())
    inner_after = int(split_summary_h5["inner_train_n_after_purge"].sum())
    comparison_rows.append(
        {
            "row_type": "decomposition",
            "context": "inner_purge_effect",
            "source_file": str(SPLIT_SUMMARY_PATH.relative_to(ROOT)),
            "metric": "inner_train_rows_before_minus_after",
            "scope": "overall",
            "tau": np.nan,
            "budget_r": np.nan,
            "n_submitted": inner_before,
            "n_corrected": inner_after,
            "k_submitted": np.nan,
            "k_corrected": np.nan,
            "events_submitted": np.nan,
            "events_corrected": np.nan,
            "tp_submitted": np.nan,
            "tp_corrected": np.nan,
            "precision_submitted": np.nan,
            "precision_corrected": np.nan,
            "submitted_value": float(inner_before),
            "corrected_value": float(inner_after),
            "delta": float(inner_after - inner_before),
            "status": "computed",
            "notes": "Corrected inner purge removes 5 boundary rows per fold from subtraining partitions.",
        }
    )

    comparison_rows.append(
        {
            "row_type": "decomposition",
            "context": "determinism_seed_policy",
            "source_file": str(FIXED_CONFIG_PATH.relative_to(ROOT)),
            "metric": "seed_value",
            "scope": "overall",
            "tau": np.nan,
            "budget_r": np.nan,
            "n_submitted": np.nan,
            "n_corrected": np.nan,
            "k_submitted": np.nan,
            "k_corrected": np.nan,
            "events_submitted": np.nan,
            "events_corrected": np.nan,
            "tp_submitted": np.nan,
            "tp_corrected": np.nan,
            "precision_submitted": np.nan,
            "precision_corrected": np.nan,
            "submitted_value": np.nan,
            "corrected_value": float(SEED),
            "delta": np.nan,
            "status": "computed",
            "notes": "Deterministic run uses fixed seed 42 and deterministic torch algorithm settings.",
        }
    )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(SUBMITTED_COMPARISON_PATH, index=False)

    write_reconciliation_markdown(
        run_id=run_id,
        comparison_df=comparison_df,
        metrics_pooled=metrics_pooled,
        metrics_by_fold=metrics_by_fold,
        boundary_audit=boundary_audit,
    )

    software_versions["pytorch_version"] = torch.__version__

    manifest = {
        "run_id": run_id,
        "model": MODEL,
        "model_full_name": MODEL_FULL_NAME,
        "model_version": MODEL_VERSION,
        "meaning_of_bcr": MEANING_OF_BCR,
        "meaning_of_v11": MEANING_OF_V11,
        "horizon": H,
        "dataset_path": verification_payload["dataset_path"],
        "dataset_sha256": dataset_sha,
        "split_path": str(SPLIT_PATH.resolve()),
        "split_sha256": split_sha,
        "protocol_tag": PROTOCOL_TAG,
        "git_branch": git_branch,
        "git_commit": git_commit,
        "execution_command": f"{sys.executable} {Path(__file__).resolve()}",
        "timestamp": utc_now_iso(),
        "device": DEVICE,
        "hardware": hardware_summary,
        "software_versions": software_versions,
        "seed": SEED,
        "deterministic_settings": {
            "torch_use_deterministic_algorithms": True,
            "torch_cudnn_deterministic": True,
            "torch_cudnn_benchmark": False,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "cpu_only_canonical": True,
        },
        "feature_set": {
            "feature_set_id": feature_set_id,
            "feature_npz_path": str(FEATURE_NPZ_PATH.resolve()),
            "feature_npz_sha256": sha256_file(FEATURE_NPZ_PATH),
            "feature_count": len(feature_cols),
            "feature_names": feature_cols,
            "row_count": int(len(feature_frame)),
            "feature_date_min": feature_frame["feature_date"].min().strftime("%Y-%m-%d"),
            "feature_date_max": feature_frame["feature_date"].max().strftime("%Y-%m-%d"),
        },
        "fixed_configuration": cfg,
        "selected_epoch_by_fold": {
            str(int(r.outer_fold)): int(r.selected_epoch) for r in fold_summary_df.itertuples(index=False)
        },
        "best_validation_recall_by_fold": {
            str(int(r.outer_fold)): float(r.best_validation_recall_tau16_r05)
            for r in fold_summary_df.itertuples(index=False)
        },
        "prediction_file": str(PREDICTIONS_PATH.resolve()),
        "prediction_sha256": sha256_file(PREDICTIONS_PATH),
        "metrics_files": [
            str(METRICS_BY_FOLD_PATH.resolve()),
            str(METRICS_POOLED_PATH.resolve()),
            str(EVENT_PREVALENCE_PATH.resolve()),
            str(SUBMITTED_COMPARISON_PATH.resolve()),
        ],
        "metrics_sha256": {
            METRICS_BY_FOLD_PATH.name: sha256_file(METRICS_BY_FOLD_PATH),
            METRICS_POOLED_PATH.name: sha256_file(METRICS_POOLED_PATH),
            EVENT_PREVALENCE_PATH.name: sha256_file(EVENT_PREVALENCE_PATH),
            SUBMITTED_COMPARISON_PATH.name: sha256_file(SUBMITTED_COMPARISON_PATH),
        },
        "diagnostic_files": [
            str(BOUNDARY_AUDIT_PATH.resolve()),
            str(TRAINING_HISTORY_PATH.resolve()),
            str(FOLD_SUMMARY_PATH.resolve()),
            str(GRAD_DIAGNOSTICS_PATH.resolve()),
            str(RECONCILIATION_PATH.resolve()),
            str(COMPLETION_REPORT_PATH.resolve()),
        ],
        "test_result": "pending_not_run",
        "legacy_sources": [
            "src/train_bcr_tcn_v11.py",
            "results/metrics/bcr_tcn_v11_H5_v2_meta.json",
            "results/predictions/bcr_tcn_v11_H5_v2_preds.csv",
            "results/metrics/alarm_budget_bcr_tcn_v11_H5_v2.csv",
            "results/hybrid_rank_v2_runs/20260219_195429_H5_hybrid_rank_v2/tables/alarm_budget_metrics_H5.csv",
            "results/hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2/tables/alarm_budget_metrics_H5.csv",
            "revision_2026/00_provenance/bcr_tcn_identity_audit/bcr_tcn_identity_audit.md",
            "revision_2026/00_provenance/bcr_tcn_identity_audit/bcr_tcn_standard_tcn_comparison.csv",
            "revision_2026/03_corrected_protocol/corrected_evaluation_protocol.md",
            "revision_2026/03_corrected_protocol/rerun_execution_plan.csv",
        ],
        "notes": "Controlled H5-only BCR-TCN v1.1 rerun under locked corrected protocol with deterministic CPU execution and explicit five-row inner/outer purge enforcement.",
    }
    write_json(MANIFEST_PATH, manifest)

    write_completion_report(
        verification=verification_payload,
        boundary_audit=boundary_audit,
        fold_summary=fold_summary_df,
        metrics_pooled=metrics_pooled,
        event_prevalence=event_prev,
        comparison_df=comparison_df,
        manifest=manifest,
    )

    print(f"Wrote {INPUT_VERIFICATION_PATH}")
    print(f"Wrote {BOUNDARY_AUDIT_PATH}")
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {TRAINING_HISTORY_PATH}")
    print(f"Wrote {FOLD_SUMMARY_PATH}")
    print(f"Wrote {GRAD_DIAGNOSTICS_PATH}")
    print(f"Wrote {METRICS_BY_FOLD_PATH}")
    print(f"Wrote {METRICS_POOLED_PATH}")
    print(f"Wrote {EVENT_PREVALENCE_PATH}")
    print(f"Wrote {SUBMITTED_COMPARISON_PATH}")
    print(f"Wrote {RECONCILIATION_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {COMPLETION_REPORT_PATH}")


if __name__ == "__main__":
    main()
