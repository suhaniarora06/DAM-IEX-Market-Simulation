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


# ── Month mapping for sorting ───────────────────────────────────────────────
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}


def extract_month_order(filename):
    fname = filename.lower()
    for month, num in MONTH_MAP.items():
        if month in fname:
            return num
    return 13  # unknown files go last


# ── 1. Load & clean ONE file ───────────────────────────────────────────────
def load_monthly_file(filepath: str) -> pd.DataFrame:
    print(f"\nLoading: {filepath}")

    if "SEPTEMBER" in filepath:
        df = pd.read_excel(filepath, header=4)
    else:
        df = pd.read_excel(filepath, header=HEADER_ROW)

    #df = pd.read_excel(filepath, header=HEADER_ROW)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename columns
    df = df.rename(columns=COLUMNS_MAP)

    # Required columns check
    required_cols = ["date", "mcp", "purchase_bid_mw", "sell_bid_mw"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"⚠️ Skipping file (missing columns): {missing}")
        return pd.DataFrame()

    # Keep relevant columns
    keep_cols = list(COLUMNS_MAP.values())
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

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
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=required_cols)

    return df


# ── 2. Load ALL files in correct order ─────────────────────────────────────
def load_all_files():
    folder = "DAM data - IEX"

    pattern = os.path.join(folder, "*.xlsx")
    files = glob.glob(pattern)

    if not files:
        raise ValueError(f"No Excel files found in '{folder}'")

    # ✅ SORT BY MONTH (FIXED)
    files = sorted(files, key=extract_month_order)

    print("\n📂 Files in correct order:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    frames = []

    for file in files:
        df = load_monthly_file(file)

        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No valid data loaded from any file.")

    df_all = pd.concat(frames, ignore_index=True)

    # ✅ FINAL SORT (VERY IMPORTANT)
    #df_all = df_all.sort_values("date").reset_index(drop=True)

    print(f"\n✅ Total rows loaded: {len(df_all):,}")

    return df_all


# ── 3. Feature Engineering ────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── Time features ──
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if "hour" in df.columns:
        df["is_peak"] = df["hour"].isin(PEAK_HOURS).astype(int)

    # ── Create SLOT (CRITICAL FIX) ──
    def get_slot(tb):
        try:
            start = tb.split("-")[0].strip()
            h, m = map(int, start.split(":"))
            return h * 4 + m // 15
        except:
            return np.nan

    df["slot"] = df["time_block"].apply(get_slot)

    # ── FINAL SORT (MOST IMPORTANT FIX) ──
    df = df.sort_values(["date", "slot"]).reset_index(drop=True)

    # ── ADD LAG FEATURES (YOU ARE MISSING THIS) ──
    df["mcp_lag1"]  = df["mcp"].shift(1)
    df["mcp_lag4"]  = df["mcp"].shift(4)
    df["mcp_lag96"] = df["mcp"].shift(96)

    # Optional but recommended
    df = df.dropna(subset=["mcp_lag1"]).reset_index(drop=True)

    # ── Market features ──
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

    print("\n📊 Data Summary:")

    if not df.empty and df["date"].notna().any():
        print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    else:
        print("  Date range : Not available")

    print(f"  Rows       : {len(df):,}")

    if "mcp" in df.columns and df["mcp"].notna().any():
        print(f"  MCP range  : {df['mcp'].min():.1f} – {df['mcp'].max():.1f} Rs/MWh")
        print(f"  Avg MCP    : {df['mcp'].mean():.1f} Rs/MWh")

    out_path = os.path.join(OUTPUT_DIR, "processed_data.csv")
    df.to_csv(out_path, index=False)

    print(f"\n💾 Saved → {out_path}")

    return df


# ── Run standalone ────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocess()