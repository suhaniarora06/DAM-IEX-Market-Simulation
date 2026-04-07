# =============================================================================
# step2_demand_curve.py  –  Demand curve estimation
# =============================================================================
"""
Inverse linear demand per (slot, is_peak):   P = a - b * Q

Robust two-anchor estimation:
  Anchor 1: (Q = MCV_mean, P = MCP_mean)  → on the curve
  Anchor 2: (Q = purchase_bid_mean, P = 0) → max willingness-to-buy qty
  ⟹  b = MCP_mean / (purchase_bid_mean - MCV_mean)
       a = b * purchase_bid_mean * 1.30   (30 % headroom above equilibrium)

This guarantees a > MCP everywhere, so supply-demand always intersects.
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from config import OUTPUT_DIR, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)

DEMAND_FEATURES = [
    "purchase_bid_mw", "mcv_mw", "scarcity_ratio",
    "slot", "is_peak", "is_weekend", "month",
    "mcp_lag1", "mcp_lag4",
]


# ── 1.  MCP regression model (used in agent state) ────────────────────────────
def fit_demand_model(df: pd.DataFrame):
    feat_cols = [c for c in DEMAND_FEATURES if c in df.columns]
    sub = df[feat_cols + ["mcp"]].dropna()
    X, y = sub[feat_cols].values, sub["mcp"].values
    pl = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=10.0))])
    pl.fit(X, y)
    cv_rmse = -cross_val_score(pl, X, y, cv=5,
                               scoring="neg_root_mean_squared_error").mean()
    print(f"  MCP ridge model  5-fold CV RMSE : {cv_rmse:.1f}  Rs/MWh")
    return pl, feat_cols


# ── 2.  a / b per (slot, is_peak) group ──────────────────────────────────────
def estimate_ab_params(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (slot, is_peak), g in df.groupby(["slot", "is_peak"]):
        if len(g) < 3:
            continue
        mcp_m  = g["mcp"].mean()
        mcv_m  = g["mcv_mw"].mean()
        bid_m  = g["purchase_bid_mw"].mean()
        gap    = bid_m - mcv_m
        b      = mcp_m / gap if gap > 10 else mcp_m / max(mcv_m, 1) * 0.1
        a      = b * bid_m * 1.30          # 30 % headroom
        records.append(dict(slot=slot, is_peak=is_peak,
                            a=a, b=b,
                            mean_mcp=mcp_m, std_mcp=g["mcp"].std(),
                            mean_mcv=mcv_m, mean_bid=bid_m))
    params = pd.DataFrame(records)
    pct_ok = (params.a > params.mean_mcp).mean() * 100
    print(f"  Slots with a > mean_mcp : {pct_ok:.0f} %")
    return params


# ── 3.  DemandCurve object ────────────────────────────────────────────────────
class DemandCurve:
    """
    P(Q) = a - b·Q  (clamped ≥ 0)
    Q(P) = (a - P) / b
    """
    def __init__(self, params_df: pd.DataFrame):
        self._df = params_df.set_index(["slot", "is_peak"])
        self._global_mean = params_df["mean_mcp"].mean()

    def _row(self, slot: int, is_peak: int):
        for key in [(slot, is_peak), (slot, 1 - is_peak)]:
            if key in self._df.index:
                return self._df.loc[key]
        # fallback: nearest slot
        sub = self._df[self._df.index.get_level_values("is_peak") == is_peak]
        return sub.iloc[0] if len(sub) else self._df.iloc[0]

    def price(self, quantity: float, slot: int, is_peak: int, total_supply: float = None) -> float:
        r = self._row(slot, is_peak)
        return max(float(r["a"]) - float(r["b"]) * float(quantity), 0.0)

    def quantity_at_price(self, price: float, slot: int, is_peak: int) -> float:
        r = self._row(slot, is_peak)
        b = float(r["b"])
        if b <= 0:
            return 0.0
        return max((float(r["a"]) - float(price)) / b, 0.0)

    def equilibrium(self, slot: int, is_peak: int):
        r = self._row(slot, is_peak)
        return float(r["mean_mcp"]), float(r["mean_mcv"])


# ── 4.  Main ──────────────────────────────────────────────────────────────────
def estimate_demand(df: pd.DataFrame):
    print("=" * 60)
    print("STEP 2 – Demand Curve Estimation")
    print("=" * 60)

    model, feat_cols = fit_demand_model(df)
    with open(os.path.join(MODEL_DIR, "demand_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "features": feat_cols}, f)

    params_df = estimate_ab_params(df)
    params_df.to_csv(os.path.join(OUTPUT_DIR, "demand_params.csv"), index=False)
    print(f"  demand_params.csv  ({len(params_df)} slot-groups)")

    return DemandCurve(params_df), model, feat_cols


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "processed_data.csv"),
                     parse_dates=["date"])
    estimate_demand(df)
