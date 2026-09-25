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

    # Stoichiometric proxy
    df["C_N"] = df["TOCin"] / (df["TNin"] + 1e-6)

    # Target at t+h
    df["y"] = df["TNout"].shift(-h)

    # ---- Memory (expanded) ----
    df["TNout_lag1"] = df["TNout"].shift(1)
    df["TNout_lag3"] = df["TNout"].shift(3)
    df["TNout_lag5"] = df["TNout"].shift(5)
    df["TNout_lag7"] = df["TNout"].shift(7)

    df["TNout_roll7"]  = df["TNout"].shift(1).rolling(7).mean()
    df["TNout_roll14"] = df["TNout"].shift(1).rolling(14).mean()
    df["TNout_roll30"] = df["TNout"].shift(1).rolling(30).mean()

    # ---- Load smoothers (expanded) ----
    for col in ["Inflow","TNin","TOCin","BODin"]:
        s = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_roll7"]  = s.rolling(7).mean()
        df[f"{col}_roll14"] = s.rolling(14).mean()

    # ---- Weather aggregates (expanded) ----
    temp = pd.to_numeric(df["temp_mean_c"], errors="coerce")
    prcp = pd.to_numeric(df["precip_total_mm"], errors="coerce")
    df["temp_roll7"]  = temp.rolling(7).mean()
    df["temp_roll14"] = temp.rolling(14).mean()
    df["temp_roll30"] = temp.rolling(30).mean()
    df["precip_sum3"]  = prcp.rolling(3).sum()
    df["precip_sum7"]  = prcp.rolling(7).sum()
    df["precip_sum14"] = prcp.rolling(14).sum()

    # ---- C/N smoother ----
    df["C_N_roll14"] = pd.to_numeric(df["C_N"], errors="coerce").rolling(14).mean()

    # ---- Interactions (help HGBR) ----
    df["Inflow_x_precip3"] = pd.to_numeric(df["Inflow"], errors="coerce") * df["precip_sum3"]
    df["temp_x_CN"]        = temp * df["C_N"]
    df["Inflow_x_TNin"]    = pd.to_numeric(df["Inflow"], errors="coerce") * pd.to_numeric(df["TNin"], errors="coerce")

    df = add_time_features(df)

    feature_cols = [
        # memory
        "TNout_lag1","TNout_lag3","TNout_lag5","TNout_lag7",
        "TNout_roll7","TNout_roll14","TNout_roll30",

        # loads + smoothers
        "Inflow","Inflow_roll7","Inflow_roll14",
        "TNin","TNin_roll7","TNin_roll14",
        "TOCin","TOCin_roll7","TOCin_roll14",
        "BODin","BODin_roll7","BODin_roll14",

        # stoichiometry
        "C_N","C_N_roll14",

        # weather
        "temp_mean_c","temp_roll7","temp_roll14","temp_roll30",
        "precip_total_mm","precip_sum3","precip_sum7","precip_sum14",

        # interactions
        "Inflow_x_precip3","temp_x_CN","Inflow_x_TNin",

        # seasonality
        "sin_doy","cos_doy"
    ]

    out = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    X = out[feature_cols].to_numpy(np.float32)
    y = out["y"].to_numpy(np.float32)
    dates = out["Date"].dt.strftime("%Y-%m-%d").to_numpy()

    out_path = ROOT / "features" / f"ulsan_H{h}_features_v2.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, dates=dates, feature_names=np.array(feature_cols, dtype=object))
    print("Saved:", out_path, "X:", X.shape, "y:", y.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True)
    args = ap.parse_args()
    build(args.h)

if __name__ == "__main__":
    main()
