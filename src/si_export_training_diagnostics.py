from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset


TAUS = [15.0, 16.0, 17.0]
BUDGETS = [0.05, 0.10]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_seq(features: np.ndarray, index: int, seq_len: int) -> np.ndarray:
    start = max(0, index - seq_len + 1)
    seq = features[start : index + 1]
    if len(seq) < seq_len:
        pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
        seq = np.concatenate([pad, seq], axis=0)
    return seq


class SeqIndexDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, dates: np.ndarray, indices: np.ndarray, seq_len: int, taus: List[float]):
        self.x = x
        self.y = y
        self.dates = dates
        self.indices = np.asarray(indices, dtype=int)
        self.seq_len = seq_len
        self.taus = taus

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = int(self.indices[k])
        seq = make_seq(self.x, idx, self.seq_len)
        target = float(self.y[idx])
        events = np.asarray([1.0 if target >= tau else 0.0 for tau in self.taus], dtype=np.float32)
        return (
            torch.from_numpy(seq).float(),
            torch.tensor(target, dtype=torch.float32),
            torch.from_numpy(events).float(),
            self.dates[idx],
        )


class CausalTCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.pad(x, (self.pad, 0))
        h = F.gelu(self.conv1(x1))
        h = self.dropout(h)
        h1 = F.pad(h, (self.pad, 0))
        h = F.gelu(self.conv2(h1))
        h = self.dropout(h)
        return h + self.residual(x)


class BCRTCN(nn.Module):
    def __init__(self, n_feat: int, channels: int, blocks: int, kernel: int, dropout: float, n_taus: int):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = n_feat
        for block_idx in range(blocks):
            dilation = 2 ** block_idx
            layers.append(CausalTCNBlock(in_ch, channels, kernel=kernel, dilation=dilation, dropout=dropout))
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h_last = h[:, :, -1]
        yhat = self.reg_head(h_last).squeeze(-1)
        logits = self.cls_head(h_last)
        return yhat, logits


def pairwise_rank_loss(scores: torch.Tensor, labels: torch.Tensor, max_pairs: int = 256) -> torch.Tensor:
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return scores.new_tensor(0.0)
    n_pairs = min(len(pos) * len(neg), max_pairs)
    pos_s = pos[torch.randint(0, len(pos), (n_pairs,), device=scores.device)]
    neg_s = neg[torch.randint(0, len(neg), (n_pairs,), device=scores.device)]
    return F.softplus(-(pos_s - neg_s)).mean()


def recall_at_budget(scores: np.ndarray, events: np.ndarray, budget_r: float = 0.05) -> float:
    n = len(scores)
    k = max(1, int(np.ceil(budget_r * n)))
    order = np.argsort(-scores)
    top_events = events[order[:k]]
    tp = int(top_events.sum())
    total = int(events.sum())
    if total == 0:
        return 0.0
    return float(tp / total)


def grad_stats(model: nn.Module) -> Dict[str, float]:
    total_sq = 0.0
    max_abs = 0.0
    sum_abs = 0.0
    n_params = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += float(torch.sum(grad * grad).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(grad)).item()))
        sum_abs += float(torch.sum(torch.abs(grad)).item())
        n_params += int(grad.numel())
    mean_abs = sum_abs / max(n_params, 1)
    return {
        "global_l2": float(math.sqrt(total_sq)),
        "max_abs": float(max_abs),
        "mean_abs": float(mean_abs),
    }


def summarize_metric(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": np.nan, "max": np.nan, "p90": np.nan, "min": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "min": float(np.min(arr)),
    }


def select_score_column(df: pd.DataFrame, tau: int) -> str:
    col = f"p_tau{tau}"
    if col in df.columns:
        return col
    return "y_pred"


def failure_rows_for_df(model_name: str, df: pd.DataFrame, tau: float, budget_r: float, scope: str) -> Dict[str, float]:
    local = df.copy()
    score_col = select_score_column(local, int(tau))
    local["event"] = (local["y_true"] >= tau).astype(int)
    n = len(local)
    k = max(1, int(np.ceil(budget_r * n)))
    local = local.sort_values(score_col, ascending=False).reset_index(drop=True)
    local["alarm"] = 0
    local.loc[: k - 1, "alarm"] = 1

    tp = int(((local["alarm"] == 1) & (local["event"] == 1)).sum())
    fp = int(((local["alarm"] == 1) & (local["event"] == 0)).sum())
    fn = int(((local["alarm"] == 0) & (local["event"] == 1)).sum())
    tn = int(((local["alarm"] == 0) & (local["event"] == 0)).sum())
    events = int(local["event"].sum())
    non_events = int((local["event"] == 0).sum())

    return {
        "model": model_name,
        "scope": scope,
        "tau": float(tau),
        "budget_r": float(budget_r),
        "score_col": score_col,
        "n": int(n),
        "k": int(k),
        "events": int(events),
        "non_events": int(non_events),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "false_alarm_rate": float(fp / max(non_events, 1)),
        "miss_rate": float(fn / max(events, 1)),
    }


def build_failure_bar_table(root: Path, horizon: int) -> pd.DataFrame:
    pred_dir = root / "results" / "predictions"
    candidates = {
        f"persistence_v2_H{horizon}": pred_dir / f"persistence_v2_H{horizon}_preds.csv",
        f"enet_a0.1_l0.5_v2_H{horizon}": pred_dir / f"enet_a0.1_l0.5_v2_H{horizon}_preds.csv",
        f"hgbr_optuna_H{horizon}_v2": pred_dir / f"hgbr_optuna_H{horizon}_v2_preds.csv",
        f"bcr_tcn_v11_H{horizon}_v2": pred_dir / f"bcr_tcn_v11_H{horizon}_v2_preds.csv",
        f"hybrid_rank_ens_H{horizon}_v2": pred_dir / f"hybrid_rank_ens_H{horizon}_v2_preds.csv",
    }

    rows = []
    for model_name, path in candidates.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "fold" not in df.columns:
            df["fold"] = 0
        for tau in TAUS:
            for budget_r in BUDGETS:
                rows.append(failure_rows_for_df(model_name, df, tau, budget_r, "overall"))
                for fold in sorted(df["fold"].unique()):
                    dfx = df[df["fold"] == fold]
                    rows.append(failure_rows_for_df(model_name, dfx, tau, budget_r, f"fold{int(fold)}"))

    return pd.DataFrame(rows)


def replay_training_diagnostics(root: Path, horizon: int, seq_len: int, n_splits: int, epochs: int, device: str) -> pd.DataFrame:
    npz_path = root / "features" / f"ulsan_H{horizon}_features_v2.npz"
    z = np.load(npz_path, allow_pickle=True)
    x = z["X"].astype(np.float32)
    y = z["y"].astype(np.float32)
    dates = z["dates"].astype(str)

    cfg = dict(
        channels=48,
        blocks=5,
        kernel=3,
        dropout=0.25,
        lr=8e-4,
        wd=1e-3,
        batch=64,
        patience=25,
        val_frac=0.15,
        huber_beta=1.0,
        w_reg=1.0,
        w_bce=0.6,
        w_rank=1.4,
        grad_clip=1.0,
        r_budget=0.05,
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    torch_device = torch.device(device)
    all_rows: List[Dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(x), start=1):
        train_idx = np.asarray(train_idx, dtype=int)
        cut = int(math.floor(len(train_idx) * (1.0 - cfg["val_frac"])))
        sub_train_idx = train_idx[:cut]
        sub_val_idx = train_idx[cut:]

        mu = x[sub_train_idx].mean(axis=0, keepdims=True)
        sd = x[sub_train_idx].std(axis=0, keepdims=True) + 1e-6
        xs = (x - mu) / sd

        ds_train = SeqIndexDataset(xs, y, dates, sub_train_idx, seq_len, TAUS)
        ds_val = SeqIndexDataset(xs, y, dates, sub_val_idx, seq_len, TAUS)
        ds_test = SeqIndexDataset(xs, y, dates, np.asarray(test_idx, dtype=int), seq_len, TAUS)

        dl_train = DataLoader(ds_train, batch_size=cfg["batch"], shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=cfg["batch"], shuffle=False)

        model = BCRTCN(
            n_feat=x.shape[1],
            channels=cfg["channels"],
            blocks=cfg["blocks"],
            kernel=cfg["kernel"],
            dropout=cfg["dropout"],
            n_taus=len(TAUS),
        ).to(torch_device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        huber = nn.SmoothL1Loss(beta=cfg["huber_beta"])

        y_sub = y[sub_train_idx]
        pos = np.asarray([(y_sub >= tau).mean() for tau in TAUS], dtype=float)
        pos_weight = torch.tensor([(1 - p) / (p + 1e-6) for p in pos], dtype=torch.float32, device=torch_device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_recall = -1.0
        best_epoch = -1
        patience_left = cfg["patience"]

        for epoch in range(epochs):
            model.train()
            tr_total_losses = []
            tr_reg_losses = []
            tr_bce_losses = []
            tr_rank_losses = []
            tr_mae_batches = []
            grad_l2_pre = []
            grad_l2_post = []
            grad_max_abs_pre = []
            grad_mean_abs_pre = []
            clipped_batches = 0

            for xb, yb, eb, _ in dl_train:
                xb = xb.to(torch_device)
                yb = yb.to(torch_device)
                eb = eb.to(torch_device)

                yhat, logits = model(xb)
                loss_reg = huber(yhat, yb)
                loss_bce = bce(logits, eb)
                rank16 = pairwise_rank_loss(logits[:, 1], eb[:, 1])
                rank15 = pairwise_rank_loss(logits[:, 0], eb[:, 0])
                loss_rank = rank16 + 0.3 * rank15
                loss = cfg["w_reg"] * loss_reg + cfg["w_bce"] * loss_bce + cfg["w_rank"] * loss_rank

                optimizer.zero_grad()
                loss.backward()

                pre = grad_stats(model)
                returned_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
                post = grad_stats(model)
                if returned_norm > cfg["grad_clip"]:
                    clipped_batches += 1

                optimizer.step()

                tr_total_losses.append(float(loss.item()))
                tr_reg_losses.append(float(loss_reg.item()))
                tr_bce_losses.append(float(loss_bce.item()))
                tr_rank_losses.append(float(loss_rank.item()))
                tr_mae_batches.append(float(torch.mean(torch.abs(yhat - yb)).item()))
                grad_l2_pre.append(float(pre["global_l2"]))
                grad_l2_post.append(float(post["global_l2"]))
                grad_max_abs_pre.append(float(pre["max_abs"]))
                grad_mean_abs_pre.append(float(pre["mean_abs"]))

            model.eval()
            val_maes = []
            val_scores = []
            val_events = []
            with torch.no_grad():
                for xb, yb, eb, _ in dl_val:
                    xb = xb.to(torch_device)
                    yb = yb.to(torch_device)
                    eb = eb.to(torch_device)
                    yhat, logits = model(xb)
                    val_maes.append(float(torch.mean(torch.abs(yhat - yb)).item()))
                    val_scores.append(torch.sigmoid(logits[:, 1]).cpu().numpy())
                    val_events.append(eb[:, 1].cpu().numpy())

            if len(val_scores):
                val_scores_arr = np.concatenate(val_scores)
                val_events_arr = np.concatenate(val_events)
                val_recall = recall_at_budget(val_scores_arr, val_events_arr, budget_r=cfg["r_budget"])
                val_event_rate = float(val_events_arr.mean())
            else:
                val_recall = 0.0
                val_event_rate = 0.0

            val_mae = float(np.mean(val_maes)) if len(val_maes) else np.nan

            grad_pre_stats = summarize_metric(grad_l2_pre)
            grad_post_stats = summarize_metric(grad_l2_post)
            grad_maxabs_stats = summarize_metric(grad_max_abs_pre)

            exploding_risk = int(
                (not np.isnan(grad_pre_stats["max"]))
                and (
                    grad_pre_stats["max"] > 10.0 * cfg["grad_clip"]
                    or grad_maxabs_stats["max"] > 5.0
                )
            )
            vanishing_risk = int(
                (not np.isnan(grad_post_stats["p90"]))
                and grad_post_stats["p90"] < 1e-4
            )

            row = {
                "horizon": horizon,
                "fold": fold,
                "epoch": epoch,
                "n_train_sub": int(len(sub_train_idx)),
                "n_val_sub": int(len(sub_val_idx)),
                "n_test_fold": int(len(ds_test)),
                "train_loss_total": float(np.mean(tr_total_losses)),
                "train_loss_reg": float(np.mean(tr_reg_losses)),
                "train_loss_bce": float(np.mean(tr_bce_losses)),
                "train_loss_rank": float(np.mean(tr_rank_losses)),
                "train_mae": float(np.mean(tr_mae_batches)),
                "val_mae": float(val_mae),
                "val_recall_tau16_r05": float(val_recall),
                "val_event_rate_tau16": float(val_event_rate),
                "grad_l2_pre_mean": grad_pre_stats["mean"],
                "grad_l2_pre_p90": grad_pre_stats["p90"],
                "grad_l2_pre_max": grad_pre_stats["max"],
                "grad_l2_post_mean": grad_post_stats["mean"],
                "grad_l2_post_p90": grad_post_stats["p90"],
                "grad_l2_post_max": grad_post_stats["max"],
                "grad_max_abs_pre_max": grad_maxabs_stats["max"],
                "grad_mean_abs_pre_mean": float(np.mean(grad_mean_abs_pre)) if len(grad_mean_abs_pre) else np.nan,
                "grad_clip_threshold": cfg["grad_clip"],
                "clip_fraction": float(clipped_batches / max(len(grad_l2_pre), 1)),
                "is_exploding_risk": exploding_risk,
                "is_vanishing_risk": vanishing_risk,
            }
            all_rows.append(row)

            if val_recall > best_recall + 1e-4:
                best_recall = val_recall
                best_epoch = epoch
                patience_left = cfg["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        all_rows.append(
            {
                "horizon": horizon,
                "fold": fold,
                "epoch": -1,
                "n_train_sub": int(len(sub_train_idx)),
                "n_val_sub": int(len(sub_val_idx)),
                "n_test_fold": int(len(ds_test)),
                "train_loss_total": np.nan,
                "train_loss_reg": np.nan,
                "train_loss_bce": np.nan,
                "train_loss_rank": np.nan,
                "train_mae": np.nan,
                "val_mae": np.nan,
                "val_recall_tau16_r05": float(best_recall),
                "val_event_rate_tau16": np.nan,
                "grad_l2_pre_mean": np.nan,
                "grad_l2_pre_p90": np.nan,
                "grad_l2_pre_max": np.nan,
                "grad_l2_post_mean": np.nan,
                "grad_l2_post_p90": np.nan,
                "grad_l2_post_max": np.nan,
                "grad_max_abs_pre_max": np.nan,
                "grad_mean_abs_pre_mean": np.nan,
                "grad_clip_threshold": cfg["grad_clip"],
                "clip_fraction": np.nan,
                "is_exploding_risk": np.nan,
                "is_vanishing_risk": np.nan,
                "best_epoch_marker": int(best_epoch),
            }
        )

    return pd.DataFrame(all_rows)


def add_fit_regime(epoch_df: pd.DataFrame) -> pd.DataFrame:
    df = epoch_df.copy()
    df = df[df["epoch"] >= 0].copy()
    out_parts = []

    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold].sort_values("epoch").copy()
        fold_best_train = float(sub["train_mae"].min())
        fold_best_val = float(sub["val_mae"].min())
        sub["generalization_gap_mae"] = sub["val_mae"] - sub["train_mae"]
        sub["val_mae_above_fold_best"] = sub["val_mae"] - fold_best_val
        sub["train_mae_ratio_to_best"] = sub["train_mae"] / max(fold_best_train, 1e-8)
        sub["val_mae_ratio_to_best"] = sub["val_mae"] / max(fold_best_val, 1e-8)

        regime = []
        for _, row in sub.iterrows():
            if row["epoch"] < 8:
                regime.append("warmup")
                continue
            is_overfit = (
                row["generalization_gap_mae"] > 0.8
                and row["val_mae_above_fold_best"] > 0.2
                and row["train_mae_ratio_to_best"] <= 1.10
            )
            is_underfit = (
                row["train_mae_ratio_to_best"] >= 1.25
                and row["val_mae_ratio_to_best"] >= 1.25
            )
            if is_overfit:
                regime.append("overfitting_risk")
            elif is_underfit:
                regime.append("underfitting_risk")
            else:
                regime.append("stable_or_improving")
        sub["fit_regime"] = regime
        out_parts.append(sub)

    return pd.concat(out_parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SI diagnostics for blocked CV, gradients, fit regimes, and failure bars.")
    parser.add_argument("--h", type=int, default=5, help="Forecast horizon to replay TCN diagnostics on.")
    parser.add_argument("--L", type=int, default=60, help="Sequence length used for TCN.")
    parser.add_argument("--splits", type=int, default=3, help="Number of TimeSeriesSplit folds.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs for diagnostics replay.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for replay (cpu/cuda).")
    args = parser.parse_args()

    set_seed(42)
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_df = replay_training_diagnostics(
        root=root,
        horizon=args.h,
        seq_len=args.L,
        n_splits=args.splits,
        epochs=args.epochs,
        device=args.device,
    )

    raw_epoch_path = out_dir / f"si_training_epoch_diagnostics_H{args.h}.csv"
    epoch_df.to_csv(raw_epoch_path, index=False)

    fit_df = add_fit_regime(epoch_df)
    fit_path = out_dir / f"si_overfit_underfit_epoch_flags_H{args.h}.csv"
    fit_df.to_csv(fit_path, index=False)

    grad_cols = [
        "horizon",
        "fold",
        "epoch",
        "grad_l2_pre_mean",
        "grad_l2_pre_p90",
        "grad_l2_pre_max",
        "grad_l2_post_mean",
        "grad_l2_post_p90",
        "grad_l2_post_max",
        "grad_max_abs_pre_max",
        "grad_mean_abs_pre_mean",
        "grad_clip_threshold",
        "clip_fraction",
        "is_exploding_risk",
        "is_vanishing_risk",
    ]
    grad_df = fit_df[grad_cols].copy()
    grad_path = out_dir / f"si_gradient_health_epoch_flags_H{args.h}.csv"
    grad_df.to_csv(grad_path, index=False)

    grad_summary = (
        grad_df.groupby("fold", as_index=False)
        .agg(
            n_epochs=("epoch", "count"),
            exploding_epochs=("is_exploding_risk", "sum"),
            vanishing_epochs=("is_vanishing_risk", "sum"),
            clip_fraction_mean=("clip_fraction", "mean"),
            grad_l2_pre_max=("grad_l2_pre_max", "max"),
            grad_l2_post_max=("grad_l2_post_max", "max"),
        )
    )
    grad_summary["exploding_epoch_rate"] = grad_summary["exploding_epochs"] / grad_summary["n_epochs"].clip(lower=1)
    grad_summary["vanishing_epoch_rate"] = grad_summary["vanishing_epochs"] / grad_summary["n_epochs"].clip(lower=1)
    grad_summary_path = out_dir / f"si_gradient_ve_summary_H{args.h}.csv"
    grad_summary.to_csv(grad_summary_path, index=False)

    fit_regime_counts = (
        fit_df.groupby(["fold", "fit_regime"], as_index=False)
        .size()
        .rename(columns={"size": "count_epochs"})
    )
    fit_regime_counts_path = out_dir / f"si_overfit_underfit_counts_H{args.h}.csv"
    fit_regime_counts.to_csv(fit_regime_counts_path, index=False)

    failure_df = build_failure_bar_table(root=root, horizon=args.h)
    failure_path = out_dir / f"si_failure_bars_H{args.h}.csv"
    failure_df.to_csv(failure_path, index=False)

    failure_overall = failure_df[failure_df["scope"] == "overall"].copy()
    failure_overall_path = out_dir / f"si_failure_bars_overall_H{args.h}.csv"
    failure_overall.to_csv(failure_overall_path, index=False)

    summary = {
        "horizon": args.h,
        "epochs_requested": args.epochs,
        "splits": args.splits,
        "cv_strategy": "TimeSeriesSplit expanding-window blocked forward chaining",
        "outputs": {
            "training_epoch_diagnostics_csv": str(raw_epoch_path.relative_to(root)),
            "overfit_underfit_flags_csv": str(fit_path.relative_to(root)),
            "gradient_health_flags_csv": str(grad_path.relative_to(root)),
            "gradient_ve_summary_csv": str(grad_summary_path.relative_to(root)),
            "failure_bars_csv": str(failure_path.relative_to(root)),
            "failure_bars_overall_csv": str(failure_overall_path.relative_to(root)),
            "overfit_underfit_counts_csv": str(fit_regime_counts_path.relative_to(root)),
        },
    }
    summary_path = out_dir / f"si_diagnostics_manifest_H{args.h}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Saved:", raw_epoch_path)
    print("Saved:", fit_path)
    print("Saved:", grad_path)
    print("Saved:", grad_summary_path)
    print("Saved:", failure_path)
    print("Saved:", failure_overall_path)
    print("Saved:", fit_regime_counts_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
