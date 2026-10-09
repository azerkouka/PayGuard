"""Transaction schema shared by the batch and streaming simulators."""

import json

import numpy as np
import pandas as pd

P2P = "p2p_transfer"
MERCHANT = "merchant_payment"
TOPUP = "mobile_topup"
TRANSACTION_TYPES = [P2P, MERCHANT, TOPUP]

ACCOUNT_TAKEOVER = "account_takeover"
CARD_TESTING = "card_testing"
NIGHT_DRAIN = "night_drain"
FRAUD_SCENARIOS = [ACCOUNT_TAKEOVER, CARD_TESTING, NIGHT_DRAIN]

COLUMNS = [
    "transaction_id",
    "timestamp",
    "user_id",
    "merchant_id",  # set for merchant payments and top-ups, null for P2P
    "recipient_id",  # set for P2P transfers, null otherwise
    "amount_tnd",
    "type",
    "device_id",
    "city",
    "is_fraud",
    "fraud_scenario",  # null if legit
]


def new_transaction_ids(n: int, rng: np.random.Generator) -> list[str]:
    return [f"{x:016x}" for x in rng.integers(0, 2**63, n)]


def finalize(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sort by time, assign transaction ids and enforce column order and dtypes."""
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["transaction_id"] = new_transaction_ids(len(df), rng)
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    df["amount_tnd"] = df["amount_tnd"].astype("float64")
    df["is_fraud"] = df["is_fraud"].astype(bool)
    return df[COLUMNS]


def encode_transaction(tx: dict) -> tuple[bytes, bytes]:
    """Kafka (key, value) for a transaction: keyed by user, JSON value, ISO timestamp."""
    value = {**tx, "timestamp": pd.Timestamp(tx["timestamp"]).isoformat()}
    return tx["user_id"].encode(), json.dumps(value).encode()


def decode_transaction(value: bytes) -> dict:
    tx = json.loads(value)
    tx["timestamp"] = pd.Timestamp(tx["timestamp"])
    return tx
