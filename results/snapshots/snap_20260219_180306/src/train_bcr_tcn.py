import argparse, math, random, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit

TAUS = [15.0, 16.0, 17.0]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_seq(X, i, L):
    # leakage-safe: only uses indices <= i; left-pad with earliest available row
    s = max(0, i - L + 1)
    seq = X[s:i+1]
    if len(seq) < L:
        pad = np.repeat(seq[:1], L - len(seq), axis=0)
        seq = np.concatenate([pad, seq], axis=0)
    return seq

class SeqIndexDataset(Dataset):
    def __init__(self, X, y, dates, indices, L, taus):
        self.X = X
        self.y = y
        self.dates = dates
        self.indices = np.array(indices, dtype=int)
        self.L = L
        self.taus = taus

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])
        seq = make_seq(self.X, i, self.L)                      # (L, F)
        yt = float(self.y[i])
        events = np.array([1.0 if yt >= t else 0.0 for t in self.taus], dtype=np.float32)
        return (
            torch.from_numpy(seq).float(),                     # (L, F)
            torch.tensor(yt, dtype=torch.float32),             # scalar
            torch.from_numpy(events).float(),                  # (T,)
            self.dates[i]
        )

class CausalTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, dropout=0.2):
        super().__init__()
        self.k = k
        self.d = d
        self.pad = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=d)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, L) — causal padding on the left only
        x1 = F.pad(x, (self.pad, 0))
        h = F.gelu(self.conv1(x1))
        h = self.dropout(h)
        h1 = F.pad(h, (self.pad, 0))
        h = F.gelu(self.conv2(h1))
        h = self.dropout(h)
        return h + self.res(x)

class BCR_TCN(nn.Module):
    def __init__(self, n_feat, channels=32, blocks=4, kernel=3, dropout=0.2, n_taus=3):
        super().__init__()
        layers = []
        in_ch = n_feat
        for b in range(blocks):
            d = 2 ** b
            layers.append(CausalTCNBlock(in_ch, channels, k=kernel, d=d, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.reg_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, n_taus)   # logits for each tau
        )

    def forward(self, x):
        # x: (B, L, F) -> (B, F, L)
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)  # (B, C)
        yhat = self.reg_head(h).squeeze(-1)     # (B,)
        logits = self.cls_head(h)               # (B, T)
        return yhat, logits

def pairwise_rank_loss(scores, labels):
    """
    RankNet-style loss. scores: (B,), labels: (B,) in {0,1}.
    Encourages scores(event) > scores(non-event).
    """
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return scores.new_tensor(0.0)

    # sample pairs to keep it light
    m = min(len(pos), len(neg), 64)
    pos_s = pos[torch.randint(0, len(pos), (m,), device=scores.device)]
    neg_s = neg[torch.randint(0, len(neg), (m,), device=scores.device)]
    return F.softplus(-(pos_s - neg_s)).mean()

def fit_one_fold(X, y, dates, tr_idx, te_idx, L, device, cfg):
    # inner time-safe val split from training indices
    tr_idx = np.array(tr_idx, dtype=int)
    te_idx = np.array(te_idx, dtype=int)
    n_tr = len(tr_idx)
    cut = int(math.floor(n_tr * (1.0 - cfg["val_frac"])))
    sub_tr = tr_idx[:cut]
    sub_va = tr_idx[cut:]

    # standardize features using sub_tr only (no leakage)
    mu = X[sub_tr].mean(axis=0, keepdims=True)
    sd = X[sub_tr].std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd

    ds_tr = SeqIndexDataset(Xs, y, dates, sub_tr, L, TAUS)
    ds_va = SeqIndexDataset(Xs, y, dates, sub_va, L, TAUS)
    ds_te = SeqIndexDataset(Xs, y, dates, te_idx, L, TAUS)

    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch"], shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch"], shuffle=False, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=cfg["batch"], shuffle=False, drop_last=False)

    model = BCR_TCN(
        n_feat=X.shape[1],
        channels=cfg["channels"],
        blocks=cfg["blocks"],
        kernel=cfg["kernel"],
        dropout=cfg["dropout"],
        n_taus=len(TAUS)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    huber = nn.SmoothL1Loss(beta=cfg["huber_beta"])
    bce = nn.BCEWithLogitsLoss()

    best = {"val_mae": float("inf"), "state": None, "epoch": -1}
    patience_left = cfg["patience"]

    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, yb, eb, _ in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            eb = eb.to(device)

            yhat, logits = model(xb)

            # losses
            loss_reg = huber(yhat, yb)

            loss_bce = bce(logits, eb)

            # ranking + budget focus on tau=16 (index 1), plus mild on tau=15 (index 0)
            s16 = logits[:, 1]
            s15 = logits[:, 0]
            loss_rank = pairwise_rank_loss(s16, eb[:, 1]) + 0.3 * pairwise_rank_loss(s15, eb[:, 0])

            p16 = torch.sigmoid(s16)
            loss_budget = (p16.mean() - cfg["r_target"]) ** 2

            loss = (
                cfg["w_reg"] * loss_reg +
                cfg["w_bce"] * loss_bce +
                cfg["w_rank"] * loss_rank +
                cfg["w_budget"] * loss_budget
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

        # validation
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for xb, yb, eb, _ in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat, _ = model(xb)
                ys.append(yb.cpu().numpy())
                yh.append(yhat.cpu().numpy())
        ys = np.concatenate(ys) if len(ys) else np.array([])
        yh = np.concatenate(yh) if len(yh) else np.array([])

        val_mae = float(np.mean(np.abs(ys - yh))) if len(ys) else float("inf")

        improved = val_mae < best["val_mae"] - 1e-4
        if improved:
            best["val_mae"] = val_mae
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # test preds
    model.eval()
    rows = []
    with torch.no_grad():
        for xb, yb, eb, dateb in dl_te:
            xb = xb.to(device)
            yhat, logits = model(xb)
            yhat = yhat.cpu().numpy()
            logits = logits.cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            for j in range(len(dateb)):
                rows.append({
                    "date": dateb[j],
                    "y_true": float(yb[j].item()),
                    "y_pred": float(yhat[j]),
                    "logit_tau15": float(logits[j,0]),
                    "logit_tau16": float(logits[j,1]),
                    "logit_tau17": float(logits[j,2]),
                    "p_tau15": float(probs[j,0]),
                    "p_tau16": float(probs[j,1]),
                    "p_tau17": float(probs[j,2]),
                })

    return rows, best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--npz", type=str, default="")
    ap.add_argument("--L", type=int, default=30)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    set_seed(42)

    ROOT = Path(__file__).resolve().parents[1]
    npz = Path(args.npz) if args.npz else (ROOT / "features" / f"ulsan_H{args.h}_features_v2.npz")
    z = np.load(npz, allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(np.float32)
    dates = z["dates"].astype(str)

    pred_dir = ROOT / "results" / "predictions"
    met_dir  = ROOT / "results" / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        L=args.L,
        channels=32,
        blocks=4,
        kernel=3,
        dropout=0.25,
        lr=1e-3,
        wd=1e-3,
        batch=64,
        epochs=args.epochs,
        patience=20,
        val_frac=0.15,
        huber_beta=1.0,
        r_target=0.05,      # budget-aware regularizer
        w_reg=1.0,
        w_bce=0.5,
        w_rank=0.8,
        w_budget=0.2,
        grad_clip=1.0,
    )

    device = torch.device(args.device)

    tscv = TimeSeriesSplit(n_splits=args.splits)
    all_rows = []
    fold_summ = []

    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        rows, best = fit_one_fold(X, y, dates, tr, te, args.L, device, cfg)
        for r in rows:
            r["fold"] = fold
        all_rows.extend(rows)
        fold_summ.append({"fold": fold, "best_val_mae": best["val_mae"], "best_epoch": best["epoch"]})

    df = pd.DataFrame(all_rows).sort_values(["fold","date"]).reset_index(drop=True)
    out_csv = pred_dir / f"bcr_tcn_H{args.h}_v2_preds.csv"
    df.to_csv(out_csv, index=False)

    out_meta = met_dir / f"bcr_tcn_H{args.h}_v2_meta.json"
    out_meta.write_text(json.dumps({
        "h": args.h,
        "npz": str(npz),
        "cfg": cfg,
        "fold_summary": fold_summ,
        "n_preds": len(df),
    }, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_meta)
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
