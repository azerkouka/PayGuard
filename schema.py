"""The transaction contract shared by every component.

This module depends on nothing else in the project: the simulator produces transactions in
this shape, and features, scoring, the API and the consumers read them.
"""

import json
from typing import Literal

import pandas as pd

P2P = "p2p_transfer"
MERCHANT = "merchant_payment"
TOPUP = "mobile_topup"
TRANSACTION_TYPES = [P2P, MERCHANT, TOPUP]
TransactionType = Literal["p2p_transfer", "merchant_payment", "mobile_topup"]

ACCOUNT_TAKEOVER = "account_takeover"
CARD_TESTING = "card_testing"
NIGHT_DRAIN = "night_drain"
FRAUD_SCENARIOS = [ACCOUNT_TAKEOVER, CARD_TESTING, NIGHT_DRAIN]

# timestamps are naive local times in this zone: the hour of day is a feature
LOCAL_TIMEZONE = "Africa/Tunis"

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
    "is_fraud",  # ground truth, known to the simulator only
    "fraud_scenario",  # null if legit
]


def encode_transaction(tx: dict) -> tuple[bytes, bytes]:
    """Kafka (key, value) for a transaction: keyed by user, JSON value, ISO timestamp."""
    value = {**tx, "timestamp": pd.Timestamp(tx["timestamp"]).isoformat()}
    return tx["user_id"].encode(), json.dumps(value).encode()


def decode_transaction(value: bytes) -> dict:
    tx = json.loads(value)
    tx["timestamp"] = pd.Timestamp(tx["timestamp"])
    return tx
