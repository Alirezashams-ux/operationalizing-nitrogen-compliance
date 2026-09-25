from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "data" / "raw" / "Ulsan_Yongsan.csv"

def add_time_features(df):
    doy = df["Date"].dt.dayofyear.values
    df["sin_doy"] = np.sin(2*np.pi*doy/365.25)
    df["cos_doy"] = np.cos(2*np.pi*doy/365.25)
    return df

def build(h):
    df = pd.read_csv(CSV)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    needed = ["TNout","Inflow","TNin","TOCin","temp_mean_c","precip_total_mm"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available: {list(df.columns)}")

    if "BODin" not in df.columns:
        df["BODin"] = np.nan

    df["C_N"] = df["TOCin"] / (df["TNin"] + 1e-6)

    # target at t+h
    df["y"] = df["TNout"].shift(-h)

    # memory features (only past)
    df["TNout_lag1"]   = df["TNout"].shift(1)
    df["TNout_roll7"]  = df["TNout"].shift(1).rolling(7).mean()
    df["TNout_roll14"] = df["TNout"].shift(1).rolling(14).mean()

    for col in ["Inflow","TNin","TOCin","BODin"]:
        df[f"{col}_roll7"] = pd.to_numeric(df[col], errors="coerce").rolling(7).mean()

    df["temp_roll7"]  = pd.to_numeric(df["temp_mean_c"], errors="coerce").rolling(7).mean()
    df["precip_sum3"] = pd.to_numeric(df["precip_total_mm"], errors="coerce").rolling(3).sum()

    df = add_time_features(df)

    feature_cols = [
        "TNout_lag1","TNout_roll7","TNout_roll14",
        "Inflow","Inflow_roll7",
        "TNin","TNin_roll7",
        "TOCin","TOCin_roll7",
        "BODin","BODin_roll7",
        "C_N",
        "temp_mean_c","temp_roll7",
        "precip_total_mm","precip_sum3",
        "sin_doy","cos_doy"
    ]

    out = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    X = out[feature_cols].to_numpy(np.float32)
    y = out["y"].to_numpy(np.float32)
    dates = out["Date"].dt.strftime("%Y-%m-%d").to_numpy()

    out_path = ROOT / "features" / f"ulsan_H{h}_features.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, dates=dates, feature_names=np.array(feature_cols, dtype=object))
    print("Saved:", out_path, "X:", X.shape, "y:", y.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True, help="horizon (e.g., 1,3,5)")
    args = ap.parse_args()
    build(args.h)

if __name__ == "__main__":
    main()
