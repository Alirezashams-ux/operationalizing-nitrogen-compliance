
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parents[1]
NPZ  = ROOT / "features" / "ulsan_H1_features.npz"
OUTD = ROOT / "results" / "predictions"
OUTD.mkdir(parents=True, exist_ok=True)

def persistence_pred(y_all, test_idx, lag=1):
    y_pred = []
    for i in test_idx:
        j = i - lag
        y_pred.append(y_all[j] if j >= 0 else np.nan)
    return np.array(y_pred, dtype=float)

def main():
    z = np.load(NPZ, allow_pickle=True)
    X = z["X"]
    y = z["y"]
    dates = z["dates"].astype(str)
    feats = z["feature_names"].tolist()

    # best model from your results
    enet = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=20000))
    ])

    models = {
        "enet_a0.1_l0.5": enet,
        "persistence_1": None,
    }

    tscv = TimeSeriesSplit(n_splits=3)

    for name, model in models.items():
        rows = []
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            y_true = y[te]
            d_te = dates[te]

            if name == "persistence_1":
                y_pred = persistence_pred(y, te, lag=1)
                ok = ~np.isnan(y_pred)
                y_true = y_true[ok]
                d_te = d_te[ok]
                y_pred = y_pred[ok]
            else:
                model.fit(X[tr], y[tr])
                y_pred = model.predict(X[te])

            df = pd.DataFrame({
                "date": d_te,
                "fold": fold,
                "y_true": y_true,
                "y_pred": y_pred,
            })
            rows.append(df)

        out = pd.concat(rows, ignore_index=True)
        out_path = OUTD / f"{name}_H1_preds.csv"
        out.to_csv(out_path, index=False)
        print("Saved:", out_path, "rows:", len(out))

    # also save feature names for traceability
    (OUTD / "feature_names_H1.txt").write_text("\n".join(feats))
    print("Saved feature list:", OUTD / "feature_names_H1.txt")

if __name__ == "__main__":
    main()
