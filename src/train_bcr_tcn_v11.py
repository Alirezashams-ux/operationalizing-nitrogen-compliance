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

    def __len__(self): return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])
        seq = make_seq(self.X, i, self.L)                      # (L, F)
        yt = float(self.y[i])
        events = np.array([1.0 if yt >= t else 0.0 for t in self.taus], dtype=np.float32)
        return (
            torch.from_numpy(seq).float(),
            torch.tensor(yt, dtype=torch.float32),
            torch.from_numpy(events).float(),
            self.dates[i],
        )

class CausalTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, dropout=0.2):
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

class BCR_TCN(nn.Module):
    def __init__(self, n_feat, channels=32, blocks=4, kernel=3, dropout=0.2, n_taus=3):
        super().__init__()
        layers=[]
        in_ch=n_feat
        for b in range(blocks):
            d=2**b
            layers.append(CausalTCNBlock(in_ch, channels, k=kernel, d=d, dropout=dropout))
            in_ch=channels
        self.tcn = nn.Sequential(*layers)

        # USE LAST TIMESTEP FEATURE (not mean pool)
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
        # x: (B, L, F) -> (B, F, L)
        x = x.transpose(1,2)
        h = self.tcn(x)                    # (B, C, L)
        h_last = h[:, :, -1]               # (B, C)
        yhat = self.reg_head(h_last).squeeze(-1)
        logits = self.cls_head(h_last)
        return yhat, logits

def pairwise_rank_loss(scores, labels, max_pairs=256):
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return scores.new_tensor(0.0)
    m = min(len(pos)*len(neg), max_pairs)
    pos_s = pos[torch.randint(0, len(pos), (m,), device=scores.device)]
    neg_s = neg[torch.randint(0, len(neg), (m,), device=scores.device)]
    return F.softplus(-(pos_s - neg_s)).mean()

def recall_at_budget(scores, events, r=0.05):
    # scores/events are 1D numpy arrays
    n = len(scores)
    k = max(1, int(np.ceil(r*n)))
    order = np.argsort(-scores)  # desc
    top = events[order[:k]]
    tp = int(top.sum())
    total = int(events.sum())
    return tp / total if total > 0 else 0.0

def fit_one_fold(X, y, dates, tr_idx, te_idx, L, device, cfg):
    tr_idx = np.array(tr_idx, int); te_idx = np.array(te_idx, int)

    # inner val split (time-safe)
    cut = int(math.floor(len(tr_idx) * (1.0 - cfg["val_frac"])))
    sub_tr = tr_idx[:cut]
    sub_va = tr_idx[cut:]

    # standardize from sub_tr only
    mu = X[sub_tr].mean(axis=0, keepdims=True)
    sd = X[sub_tr].std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd

    ds_tr = SeqIndexDataset(Xs, y, dates, sub_tr, L, TAUS)
    ds_va = SeqIndexDataset(Xs, y, dates, sub_va, L, TAUS)
    ds_te = SeqIndexDataset(Xs, y, dates, te_idx, L, TAUS)

    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch"], shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch"], shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=cfg["batch"], shuffle=False)

    model = BCR_TCN(
        n_feat=X.shape[1],
        channels=cfg["channels"],
        blocks=cfg["blocks"],
        kernel=cfg["kernel"],
        dropout=cfg["dropout"],
        n_taus=len(TAUS),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    huber = nn.SmoothL1Loss(beta=cfg["huber_beta"])

    # pos_weight per tau from sub_tr prevalence
    y_tr = y[sub_tr]
    pos = np.array([(y_tr >= t).mean() for t in TAUS], float)
    pos_weight = torch.tensor([(1-p)/(p+1e-6) for p in pos], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = {"val_recall": -1.0, "state": None, "epoch": -1}
    patience = cfg["patience"]

    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, yb, eb, _ in dl_tr:
            xb = xb.to(device); yb = yb.to(device); eb = eb.to(device)
            yhat, logits = model(xb)

            loss_reg = huber(yhat, yb)
            loss_bce = bce(logits, eb)

            # focus ranking on tau=16 (index 1), and mild tau=15
            s16 = logits[:, 1]; s15 = logits[:, 0]
            loss_rank = pairwise_rank_loss(s16, eb[:, 1]) + 0.3 * pairwise_rank_loss(s15, eb[:, 0])

            loss = cfg["w_reg"]*loss_reg + cfg["w_bce"]*loss_bce + cfg["w_rank"]*loss_rank

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

        # validate: maximize Recall@5% for tau=16 (exactly your evaluation)
        model.eval()
        s16_all, e16_all = [], []
        mae_all = []
        with torch.no_grad():
            for xb, yb, eb, _ in dl_va:
                xb = xb.to(device); yb = yb.to(device); eb = eb.to(device)
                yhat, logits = model(xb)
                mae_all.append(torch.mean(torch.abs(yhat - yb)).item())
                s16_all.append(torch.sigmoid(logits[:,1]).cpu().numpy())
                e16_all.append(eb[:,1].cpu().numpy())
        s16_all = np.concatenate(s16_all) if len(s16_all) else np.array([])
        e16_all = np.concatenate(e16_all) if len(e16_all) else np.array([])
        val_recall = recall_at_budget(s16_all, e16_all, r=cfg["r_budget"]) if len(s16_all) else 0.0
        val_mae = float(np.mean(mae_all)) if len(mae_all) else float("inf")

        improved = val_recall > best["val_recall"] + 1e-4
        if improved:
            best["val_recall"] = val_recall
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch
            patience = cfg["patience"]
        else:
            patience -= 1
            if patience <= 0:
                break

        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  val_recall@{cfg['r_budget']:.2f}(tau16)={val_recall:.3f}  val_MAE={val_mae:.3f}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # test preds
    model.eval()
    rows=[]
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
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=260)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    set_seed(42)
    ROOT = Path(__file__).resolve().parents[1]
    npz = ROOT / "features" / f"ulsan_H{args.h}_features_v2.npz"
    z = np.load(npz, allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(np.float32)
    dates = z["dates"].astype(str)

    pred_dir = ROOT/"results"/"predictions"
    met_dir  = ROOT/"results"/"metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        channels=48,
        blocks=5,
        kernel=3,
        dropout=0.25,
        lr=8e-4,
        wd=1e-3,
        batch=64,
        epochs=args.epochs,
        patience=25,
        val_frac=0.15,
        huber_beta=1.0,
        w_reg=1.0,
        w_bce=0.6,
        w_rank=1.4,
        grad_clip=1.0,
        r_budget=0.05,
    )

    device = torch.device(args.device)
    tscv = TimeSeriesSplit(n_splits=args.splits)

    all_rows=[]
    fold_summ=[]
    for fold,(tr,te) in enumerate(tscv.split(X),1):
        rows, best = fit_one_fold(X,y,dates,tr,te,args.L,device,cfg)
        for r in rows: r["fold"]=fold
        all_rows.extend(rows)
        fold_summ.append({"fold":fold, "best_val_recall_tau16_r05":best["val_recall"], "best_epoch":best["epoch"]})

    df = pd.DataFrame(all_rows).sort_values(["fold","date"]).reset_index(drop=True)
    out_csv = pred_dir / f"bcr_tcn_v11_H{args.h}_v2_preds.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "h": args.h,
        "L": args.L,
        "npz": str(npz),
        "cfg": cfg,
        "fold_summary": fold_summ,
        "n_preds": int(len(df)),
    }
    out_meta = met_dir / f"bcr_tcn_v11_H{args.h}_v2_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_meta)

if __name__ == "__main__":
    main()
