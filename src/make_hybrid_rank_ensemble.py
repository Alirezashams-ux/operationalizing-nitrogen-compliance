import argparse
import pandas as pd
import numpy as np

def rank01(x: pd.Series) -> np.ndarray:
    # rank to [0,1] where larger values => closer to 1
    r = x.rank(method="average", ascending=True).to_numpy(dtype=float)
    if len(r) <= 1:
        return np.zeros_like(r)
    return (r - 1.0) / (len(r) - 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcn", required=True)
    ap.add_argument("--enet", required=True)
    ap.add_argument("--persist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", default="0.5,0.25,0.25", help="weights: tcn,enet,persist")
    args = ap.parse_args()

    w_tcn, w_enet, w_persist = [float(s) for s in args.w.split(",")]
    s = w_tcn + w_enet + w_persist
    w_tcn, w_enet, w_persist = w_tcn/s, w_enet/s, w_persist/s

    tcn = pd.read_csv(args.tcn)
    enet = pd.read_csv(args.enet)
    per  = pd.read_csv(args.persist)

    # Normalize column names expected across files
    for df in (tcn, enet, per):
        if "date" not in df.columns and "Date" in df.columns:
            df.rename(columns={"Date":"date"}, inplace=True)
        if "y_true" not in df.columns and "y" in df.columns:
            df.rename(columns={"y":"y_true"}, inplace=True)

    key_cols = ["date"]
    if "fold" in tcn.columns and "fold" in enet.columns and "fold" in per.columns:
        key_cols = ["fold","date"]

    # keep only necessary columns
    tcn_keep = key_cols + ["y_true","p_tau15","p_tau16","p_tau17"]
    enet_keep = key_cols + ["y_pred"]
    per_keep  = key_cols + ["y_pred"]

    tcn = tcn[tcn_keep].copy()
    enet = enet[enet_keep].copy().rename(columns={"y_pred":"y_pred_enet"})
    per  = per[per_keep].copy().rename(columns={"y_pred":"y_pred_persist"})

    df = tcn.merge(enet, on=key_cols, how="inner").merge(per, on=key_cols, how="inner")

    # build rank-ensemble per tau; ranks computed within fold if fold exists, else overall
    out_rows = []
    group_iter = [("overall", df)] if "fold" not in df.columns else df.groupby("fold", sort=True)

    for gname, g in group_iter:
        g = g.copy()

        # Scores to ensemble:
        # - TCN: p_tauXX
        # - ENet/persistence: y_pred (monotone with risk); convert to rank and combine
        r_enet = rank01(g["y_pred_enet"])
        r_per  = rank01(g["y_pred_persist"])

        for tau in [15,16,17]:
            r_tcn = rank01(g[f"p_tau{tau}"])
            g[f"p_tau{tau}"] = (w_tcn*r_tcn + w_enet*r_enet + w_persist*r_per)

        out_rows.append(g)

    out = pd.concat(out_rows, axis=0).sort_values(key_cols).reset_index(drop=True)

    # For completeness (not used for alarms), give a single y_pred (use ENet)
    out["y_pred"] = out["y_pred_enet"]
    out.drop(columns=["y_pred_enet","y_pred_persist"], inplace=True)

    out.to_csv(args.out, index=False)
    print("Saved:", args.out, "rows:", len(out), "weights:", (w_tcn,w_enet,w_persist))

if __name__ == "__main__":
    main()
