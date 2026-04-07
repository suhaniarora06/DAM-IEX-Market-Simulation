# =============================================================================
# step1_data_preprocessing.py  –  Clean & combine IEX DAM Excel files
# =============================================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
from config import OUTPUT_DIR, HEADER_ROW, COLUMNS_MAP, PEAK_HOURS, SCARCITY_CLIP_MAX

warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load & clean ONE file ───────────────────────────────────────────────
def load_monthly_file(filepath: str) -> pd.DataFrame:
    print(f"\nLoading: {filepath}")

    df = pd.read_excel(filepath, header=HEADER_ROW)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename columns using config mapping
    df = df.rename(columns=COLUMNS_MAP)

    # Check available columns
    required_cols = ["date", "mcp", "purchase_bid_mw", "sell_bid_mw"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"⚠️ Skipping file (missing columns): {missing}")
        return pd.DataFrame()

    # Keep only relevant columns
    keep_cols = list(COLUMNS_MAP.values())
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Convert numeric columns safely
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
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=required_cols)

    return df


# ── 2. Load ALL files from your folder ─────────────────────────────────────
def load_all_files():
    folder = "DAM data - IEX"   # your folder name

    pattern = os.path.join(folder, "*.xlsx")
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"No Excel files found in '{folder}'")

    frames = []

    for file in files:
        df = load_monthly_file(file)

        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No valid data loaded from any file.")

    df_all = pd.concat(frames, ignore_index=True)

    print(f"\n✅ Total rows loaded: {len(df_all):,}")

    return df_all


# ── 3. Feature Engineering ────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if "hour" in df.columns:
        df["is_peak"] = df["hour"].isin(PEAK_HOURS).astype(int)

    # Market features
    df["scarcity_ratio"] = (
        df["purchase_bid_mw"] /
        df["sell_bid_mw"].replace(0, np.nan)
    ).clip(upper=SCARCITY_CLIP_MAX)

    df["supply_surplus"] = df["sell_bid_mw"] - df["purchase_bid_mw"]

    return df


# ── 4. Main pipeline ──────────────────────────────────────────────────────
def preprocess():
    print("=" * 60)
    print("STEP 1 – Data Preprocessing")
    print("=" * 60)

    df = load_all_files()
    df = engineer_features(df)

    # ── Basic Report ──
    print("\n📊 Data Summary:")

    if not df.empty and df["date"].notna().any():
        print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    else:
        print("  Date range : Not available")

    print(f"  Rows       : {len(df):,}")

    if "mcp" in df.columns and df["mcp"].notna().any():
        print(f"  MCP range  : {df['mcp'].min():.1f} – {df['mcp'].max():.1f} Rs/MWh")
        print(f"  Avg MCP    : {df['mcp'].mean():.1f} Rs/MWh")
    else:
        print("  MCP stats  : Not available")

    # Save output
    out_path = os.path.join(OUTPUT_DIR, "processed_data.csv")
    df.to_csv(out_path, index=False)

    print(f"\n💾 Saved → {out_path}")

    return df


# ── Run standalone ────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocess()