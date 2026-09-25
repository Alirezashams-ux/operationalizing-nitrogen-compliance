import argparse
import pandas as pd
import numpy as np

def rank01(x: pd.Series) -> np.ndarray:
    r = x.rank(method="average", ascending=True).to_numpy(dtype=float)
    if len(r) <= 1:
        return np.zeros_like(r)
    return (r - 1.0) / (len(r) - 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcn", required=True)
    ap.add_argument("--persist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", default="0.7,0.3", help="weights: tcn,persist (sum to 1)")
    args = ap.parse_args()

    w_tcn, w_p = [float(s) for s in args.w.split(",")]
    s = w_tcn + w_p
    w_tcn, w_p = w_tcn/s, w_p/s

    tcn = pd.read_csv(args.tcn)
    per = pd.read_csv(args.persist)

    # normalize common column names
    for df in (tcn, per):
        if "date" not in df.columns and "Date" in df.columns:
            df.rename(columns={"Date":"date"}, inplace=True)
        if "y_true" not in df.columns and "y" in df.columns:
            df.rename(columns={"y":"y_true"}, inplace=True)

    key_cols = ["date"]
    if "fold" in tcn.columns and "fold" in per.columns:
        key_cols = ["fold","date"]

    tcn_keep = key_cols + ["y_true","p_tau15","p_tau16","p_tau17"]
    per_keep = key_cols + ["y_pred"]

    tcn = tcn[tcn_keep].copy()
    per = per[per_keep].copy().rename(columns={"y_pred":"y_pred_persist"})

    df = tcn.merge(per, on=key_cols, how="inner")

    out_rows = []
    group_iter = [("overall", df)] if "fold" not in df.columns else df.groupby("fold", sort=True)

    for _, g in group_iter:
        g = g.copy()
        r_p = rank01(g["y_pred_persist"])
        for tau in [15,16,17]:
            r_t = rank01(g[f"p_tau{tau}"])
            g[f"p_tau{tau}"] = (w_tcn*r_t + w_p*r_p)
        out_rows.append(g)

    out = pd.concat(out_rows, axis=0).sort_values(key_cols).reset_index(drop=True)
    out["y_pred"] = out["y_pred_persist"]   # not used for alarms; keep for completeness
    out.drop(columns=["y_pred_persist"], inplace=True)

    out.to_csv(args.out, index=False)
    print("Saved:", args.out, "rows:", len(out), "weights:", (w_tcn, w_p))

if __name__ == "__main__":
    main()
