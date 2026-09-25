
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "predictions"
OUT_DIR  = ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAUS = [15.0, 16.0, 17.0]
BUDGETS = [0.05, 0.10]  # r = 5%, 10%

def alarm_metrics(df, tau, r):
    df = df.copy()
    df["event"] = (df["y_true"] >= tau).astype(int)

    n = len(df)
    k = max(1, int(np.ceil(r * n)))

    # score = predicted TNout (rank by risk)
    df = df.sort_values("y_pred", ascending=False)
    alarm = df.head(k)

    tp = int(alarm["event"].sum())
    total_events = int(df["event"].sum())

    precision = tp / k if k > 0 else np.nan
    recall = tp / total_events if total_events > 0 else np.nan

    return dict(n=n, k=k, events=total_events, TP=tp, precision=precision, recall=recall)

def main():
    all_rows = []

    for csv in sorted(PRED_DIR.glob("*_H1_preds.csv")):
        model_name = csv.name.replace("_H1_preds.csv","")
        df = pd.read_csv(csv)

        # fold-wise and overall
        for scope, dfx in [("overall", df)] + [(f"fold{f}", df[df["fold"]==f]) for f in sorted(df["fold"].unique())]:
            for tau in TAUS:
                for r in BUDGETS:
                    m = alarm_metrics(dfx, tau, r)
                    all_rows.append({
                        "model": model_name,
                        "scope": scope,
                        "tau": tau,
                        "budget_r": r,
                        **m
                    })

    out = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "alarm_budget_H1.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(out.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
