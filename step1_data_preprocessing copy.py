import os
import glob
import warnings
import numpy as np
import pandas as pd
from config import (
    DATA_DIR, OUTPUT_DIR, HEADER_ROW, COLUMNS_MAP,
    PEAK_HOURS, SCARCITY_CLIP_MAX
)

warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load a single file ───────────────────────────────────────────
def load_monthly_file(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, header=HEADER_ROW)

    # Clean column names 
    df.columns = df.columns.str.strip()

    # Rename columns
    df = df.rename(columns=COLUMNS_MAP)

    # Keep required columns only
    required = list(COLUMNS_MAP.values())
    available = [c for c in required if c in df.columns]
    df = df[available].copy()

    # Convert numeric columns
    for col in ["purchase_bid_mw", "sell_bid_mw", "mcv_mw", "fsv_mw", "mcp", "hour"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "")
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["date", "mcp", "purchase_bid_mw", "sell_bid_mw"])

    return df


# ── 2. Load all files from your folder ───────────────────────────────
def load_all_files():
    folder = "DAM data - IEX"   # ← your actual folder name

    pattern = os.path.join(folder, "*.xlsx")
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"No Excel files found in '{folder}'")

    frames = []

    for file in files:
        print(f"Loading: {file}")
        df = load_monthly_file(file)   # ✅ pass filepath (correct)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    print(f"\n✅ Total rows loaded: {len(df_all):,}")

    return df_all

# ── 3. Feature engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time & market features."""
    df = df.copy()

    #df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Time features
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_peak"]     = df["hour"].isin(PEAK_HOURS).astype(int)

    # Parse 15-min slot index (0–95) from time_block string "HH:MM - HH:MM"
    def slot_index(tb):
        try:
            start = tb.split("-")[0].strip()
            h, m = map(int, start.split(":"))
            return h * 4 + m // 15
        except Exception:
            return np.nan

    df["slot"] = df["time_block"].apply(slot_index)

    # Market features
    df["scarcity_ratio"] = (
        df["purchase_bid_mw"] / df["sell_bid_mw"].replace(0, np.nan)
    ).clip(upper=SCARCITY_CLIP_MAX)

    df["supply_surplus"] = df["sell_bid_mw"] - df["purchase_bid_mw"]

    # Lag features (sorted so lags make sense)
    df = df.sort_values(["date", "slot"]).reset_index(drop=True)
    df["mcp_lag1"]  = df["mcp"].shift(1)
    df["mcp_lag4"]  = df["mcp"].shift(4)   # 1 hour ago
    df["mcp_lag96"] = df["mcp"].shift(96)  # 1 day ago (96 slots)

    df = df.dropna(subset=["mcp_lag1"]).reset_index(drop=True)
    return df


# ── 4. Main pipeline ──────────────────────────────────────────────────────────
def preprocess():
    print("=" * 60)
    print("STEP 1 – Data Preprocessing")
    print("=" * 60)

    df = load_all_files()
    df = engineer_features(df)

    # Basic quality report
    if not df.empty:
        print(f"\n  Date range : {df['date'].min()} → {df['date'].max()}")
        print(f"  Rows       : {len(df):,}")
        print(f"  MCP range  : {df['mcp'].min():.1f} – {df['mcp'].max():.1f}  Rs/MWh")
        print(f"  Avg MCP    : {df['mcp'].mean():.1f}  Rs/MWh")
    else:
        print("DataFrame is empty")

    
    out_path = os.path.join(OUTPUT_DIR, "processed_data.csv")
    df.to_csv(out_path, index=False)
    print(f"\n   Saved → {out_path}")

    return df


if __name__ == "__main__":
    preprocess()
