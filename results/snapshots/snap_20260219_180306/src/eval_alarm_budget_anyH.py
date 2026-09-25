import argparse
import pandas as pd
import numpy as np
from pathlib import Path

TAUS = [15.0, 16.0, 17.0]
BUDGETS = [0.05, 0.10]

def alarm_metrics(df, tau, r):
    df = df.copy()
    df["event"] = (df["y_true"] >= tau).astype(int)
    n = len(df)
    k = max(1, int(np.ceil(r * n)))
    df = df.sort_values("y_pred", ascending=False)
    alarm = df.head(k)
    tp = int(alarm["event"].sum())
    total_events = int(df["event"].sum())
    precision = tp / k if k > 0 else np.nan
    recall = tp / total_events if total_events > 0 else np.nan
    return dict(n=n, k=k, events=total_events, TP=tp, precision=precision, recall=recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    PRED = ROOT / "results" / "predictions"
    OUTD = ROOT / "results" / "metrics"
    OUTD.mkdir(parents=True, exist_ok=True)

    # models we expect for this horizon
    files = [
        PRED / f"persistence_H{args.h}_preds.csv",
        PRED / f"enet_a0.1_l0.5_H{args.h}_preds.csv",  # exists from run_linear_alarm
        PRED / f"hgbr_optuna_H{args.h}_preds.csv",
    ]
    files = [f for f in files if f.exists()]

    rows = []
    for f in files:
        model = f.name.replace(f"_H{args.h}_preds.csv","")
        df = pd.read_csv(f)
        scopes = [("overall", df)] + [(f"fold{fo}", df[df["fold"]==fo]) for fo in sorted(df["fold"].unique())]
        for scope, dfx in scopes:
            for tau in TAUS:
                for r in BUDGETS:
                    m = alarm_metrics(dfx, tau, r)
                    rows.append({"model": model, "scope": scope, "tau": tau, "budget_r": r, **m})

    out = pd.DataFrame(rows)
    out_path = OUTD / f"alarm_budget_compare_H{args.h}.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # print key lines for decision: H=5 tau=15/16 r=0.05 overall
    key = out[(out.scope=="overall") & (out.budget_r==0.05) & (out.tau.isin([15.0,16.0]))]
    print("\nKey decision rows (overall, r=5%, tau=15/16):")
    print(key[["model","tau","precision","recall","TP","events","k"]].sort_values(["tau","recall"], ascending=[True,False]).to_string(index=False))

if __name__ == "__main__":
    main()
