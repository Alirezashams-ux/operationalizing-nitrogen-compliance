import argparse
import pandas as pd
import numpy as np
from pathlib import Path

TAUS = [15.0,16.0,17.0]
BUDGETS = [0.05,0.10]

def score_col_for_tau(df, tau):
    # if probabilistic scores exist, use them; else fallback to y_pred
    m = {15.0:"p_tau15", 16.0:"p_tau16", 17.0:"p_tau17"}
    c = m[tau]
    return c if c in df.columns else "y_pred"

def alarm_metrics(df, tau, r):
    df = df.copy()
    df["event"] = (df["y_true"] >= tau).astype(int)
    n = len(df)
    k = max(1, int(np.ceil(r*n)))
    sc = score_col_for_tau(df, tau)
    df = df.sort_values(sc, ascending=False)
    alarm = df.head(k)
    tp = int(alarm["event"].sum())
    total = int(df["event"].sum())
    return dict(n=n,k=k,events=total,TP=tp,
                precision=tp/k if k else np.nan,
                recall=tp/total if total else np.nan,
                score=sc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--model", type=str, default="bcr_tcn")
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    if "fold" not in df.columns:
        raise ValueError("pred file must contain fold column")

    rows=[]
    for scope, dfx in [("overall", df)] + [(f"fold{f}", df[df.fold==f]) for f in sorted(df.fold.unique())]:
        for tau in TAUS:
            for r in BUDGETS:
                m = alarm_metrics(dfx, tau, r)
                rows.append({"model": args.model, "scope": scope, "tau": tau, "budget_r": r, **m})

    out = pd.DataFrame(rows)
    out_path = args.out or str(Path(args.pred).with_suffix("").as_posix() + "_alarm.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    key = out[(out.scope=="overall") & (out.budget_r==0.05) & (out.tau.isin([15.0,16.0]))]
    print("\nKey decision rows (overall, r=5%, tau=15/16):")
    print(key[["model","tau","precision","recall","TP","events","k","score"]].sort_values(["tau","recall"], ascending=[True,False]).to_string(index=False))

if __name__ == "__main__":
    main()
