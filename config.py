import os

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = "DAM data - IEX"          # folder containing monthly .xlsx files
OUTPUT_DIR  = "outputs"       # all artefacts written here
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")

# ── Data ingestion ──────────────────────────────────────────────────────────
HEADER_ROW  = 3               # 0-indexed row that contains column names in xlsx
COLUMNS_MAP = {               # rename raw xlsx headers → clean internal names
    "Date"                       : "date",
    "Hour"                       : "hour",
    "Time Block"                 : "time_block",
    "Purchase Bid (MW)"          : "purchase_bid_mw",
    "Sell Bid (MW)"              : "sell_bid_mw",
    "MCV (MW)"                   : "mcv_mw",
    "Final Scheduled Volume (MW)": "fsv_mw",
    "MCP (Rs/MWh) *"             : "mcp",
}

# ── Feature engineering ─────────────────────────────────────────────────────
PEAK_HOURS        = list(range(7, 23))   # hours 7–22 are peak
SCARCITY_CLIP_MAX = 5.0                  # cap demand/supply ratio