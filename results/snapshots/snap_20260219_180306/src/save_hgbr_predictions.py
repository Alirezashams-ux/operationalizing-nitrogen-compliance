import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    NPZ  = ROOT / "features" / f"ulsan_H{args.h}_features.npz"
    CFG  = ROOT / "results" / "hgbr" / f"hgbr_optuna_H{args.h}.json"
    OUTD = ROOT / "results" / "predictions"
    OUTD.mkdir(parents=True, exist_ok=True)

    z = np.load(NPZ, allow_pickle=True)
    X, y = z["X"], z["y"]
    dates = z["dates"].astype(str)

    cfg = json.loads(CFG.read_text())
    params = cfg["best_params"]
    # add fixed params
    params.update(dict(
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    ))

    tscv = TimeSeriesSplit(n_splits=args.splits)

    rows = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        m = HistGradientBoostingRegressor(**params)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])

        rows.append(pd.DataFrame({
            "date": dates[te],
            "fold": fold,
            "y_true": y[te],
            "y_pred": pred
        }))

    df = pd.concat(rows, ignore_index=True)
    out_path = OUTD / f"hgbr_optuna_H{args.h}_preds.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path, "rows:", len(df))

if __name__ == "__main__":
    main()
