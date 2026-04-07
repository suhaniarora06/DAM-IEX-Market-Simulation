import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main():
    import config as cfg

    # Use default paths from config (inside your project folder)
    data_dir = cfg.DATA_DIR

    # Create output folders locally
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    t0 = time.time()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────
    banner("STEP 1 / 5  –  Data Preprocessing")
    from step1_data_preprocessing import preprocess
    df = preprocess(data_dir)

    # ── Step 2: Demand Curve ──────────────────────────────────────────────
    banner("STEP 2 / 5  –  Demand Curve Estimation")
    from step2_demand_curve import estimate_demand
    demand_curve, _, _ = estimate_demand(df)

    # ── Step 3: Agent Pool ────────────────────────────────────────────────
    banner("STEP 3 / 5  –  Market Simulator & Agent Pool")
    from step3_market_simulator import create_agents
    mean_supply = df["sell_bid_mw"].mean()
    agents_preview = create_agents(mean_supply, n=cfg.N_AGENTS)

    print(f"{cfg.N_AGENTS} agents created")

    # ── Step 4: RL Training ───────────────────────────────────────────────
    banner("STEP 4 / 5  –  RL Training")
    from step4_rl_training import train

    agents, log_df = train(df, demand_curve)

    # Load saved model info
    agent_path = os.path.join(cfg.MODEL_DIR, "agents.pkl")
    with open(agent_path, "rb") as f:
        data = pickle.load(f)

    base_caps   = data["base_caps"]
    mean_supply = data["mean_supply"]

    # ── Step 5: Evaluation ────────────────────────────────────────────────
    banner("STEP 5 / 5  –  Evaluation & Scenario Testing")
    from step5_evaluation import run_evaluation

    res_df, scen_df = run_evaluation(
        df, agents, demand_curve, log_df, base_caps, mean_supply
    )

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner("PIPELINE COMPLETE")
    print(f"Total runtime: {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    main()