"""
Smoke test for categorical decoding in serialize_batch.

Prints 3 rows from processed_DataCo_train.csv serialized two ways:
  1. With decoding  (raw_csv_path provided)
  2. Without decoding (raw_csv_path=None, pure numeric)

Usage:
    conda run -n simenv python3 test_serialization.py
"""

import sys
import os
import argparse
import types
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.llm_model import LLMAttnPoolNetwork
from tools import feature_list

RAW_CSV      = "datasets/DataCo/DataCo.csv"
PROCESSED    = "datasets/DataCo/processed_DataCo_train.csv"
DATASET      = "DataCo"
HF_MODEL     = "gpt2"          # smallest model; backbone weights don't matter for this test
N_ROWS       = 3


def make_env(dataset, device="cpu"):
    args = types.SimpleNamespace(
        dataset=dataset,
        use_gpu=0,
        device_id=0,
        seed=42,
        wandb=0,
        data_dir="datasets",
        ckpt_dir=None,
        pool_init="vocab",
        pool_type="attention",
    )
    # Minimal Env stand-in — only needs .args and .device
    env = types.SimpleNamespace(args=args, device=device)
    return env


def serialize_without_decoders(model, raw_state):
    """Call serialize_batch with decoders temporarily cleared."""
    saved = model.decoders
    model.decoders = {}
    texts = model.serialize_batch(raw_state)
    model.decoders = saved
    return texts


def main():
    if not os.path.exists(RAW_CSV):
        print(f"ERROR: raw CSV not found at {RAW_CSV}")
        sys.exit(1)
    if not os.path.exists(PROCESSED):
        print(f"ERROR: processed CSV not found at {PROCESSED}")
        sys.exit(1)

    print("=" * 70)
    print("Instantiating LLMAttnPoolNetwork with decoder (raw_csv_path provided)...")
    print("=" * 70)
    env = make_env(DATASET)
    model = LLMAttnPoolNetwork(env, model_name=HF_MODEL, raw_csv_path=RAW_CSV)

    # Load N_ROWS from the processed train file
    df = pd.read_csv(PROCESSED)
    feature_dim = len(
        feature_list.product_info[DATASET]
        + feature_list.order_info[DATASET]
        + feature_list.customer_info[DATASET]
        + feature_list.shipping_info[DATASET]
    )
    rows = df.iloc[:N_ROWS, :feature_dim].values
    raw_state = torch.FloatTensor(rows)

    decoded_texts  = model.serialize_batch(raw_state)
    numeric_texts  = serialize_without_decoders(model, raw_state)

    print()
    print("=" * 70)
    print(f"Side-by-side comparison for {N_ROWS} rows")
    print("=" * 70)
    for i, (dec, num) in enumerate(zip(decoded_texts, numeric_texts)):
        print(f"\n--- Row {i + 1} ---")
        print(f"[WITH DECODING]\n  {dec}")
        print(f"[NUMERIC ONLY ]\n  {num}")

    print("\nDone.")


if __name__ == "__main__":
    main()
