"""SQL behind the analyst dashboard. Everything is read from PostgreSQL.

None of these queries returns `is_fraud` or `fraud_scenario`: the analyst must not see the
simulator's ground truth.
"""

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from config import settings

LABELS = ("fraud", "false_positive")


def _fetch(sql: str, params: tuple = ()) -> list[dict]:
    with psycopg.connect(settings.postgres_dsn, row_factory=dict_row, autocommit=True) as conn:
        return conn.execute(sql, params).fetchall()


def kpis(window_minutes: int = 5) -> dict:
    """Traffic, alert rate and latency over the last minutes; analyst precision since the start."""
    traffic = _fetch(
        """
        SELECT count(*)                                              AS scored,
               count(*) FILTER (WHERE decision <> 'approve')         AS alerts,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
        FROM scores
        WHERE scored_at >= now() - make_interval(mins => %s)
        """,
        (window_minutes,),
    )[0]
    labels = _fetch(
        """
        SELECT count(*)                                AS labeled,
               count(*) FILTER (WHERE label = 'fraud') AS confirmed
        FROM analyst_labels
        """
    )[0]
    scored, labeled = traffic["scored"], labels["labeled"]
    return {
        "tx_per_min": scored / window_minutes,
        "alert_rate": traffic["alerts"] / scored if scored else None,
        "p95_latency_ms": traffic["p95_latency_ms"],
        "labeled": labeled,
        "confirmed": labels["confirmed"],
        "precision": labels["confirmed"] / labeled if labeled else None,
    }


def alerts(decisions: list[str], include_labeled: bool = False, limit: int = 200) -> pd.DataFrame:
    """Review / block decisions, newest first, with the transaction they refer to."""
    rows = _fetch(
        """
        SELECT s.transaction_id, s.scored_at, s.decision, s.score, s.top_reasons, s.latency_ms,
               t."timestamp", t.user_id, t.amount_tnd, t.type, t.city, t.device_id,
               t.merchant_id, t.recipient_id, l.label
        FROM scores s
        JOIN transactions t USING (transaction_id)
        LEFT JOIN analyst_labels l USING (transaction_id)
        WHERE s.decision = ANY(%s) AND (%s OR l.label IS NULL)
        ORDER BY s.scored_at DESC, s.score DESC
        LIMIT %s
        """,
        (decisions, include_labeled, limit),
    )
    columns = ["transaction_id", "scored_at", "decision", "score", "top_reasons", "latency_ms",
               "timestamp", "user_id", "amount_tnd", "type", "city", "device_id", "merchant_id",
               "recipient_id", "label"]  # fmt: skip
    df = pd.DataFrame(rows, columns=columns)
    df["amount_tnd"] = df["amount_tnd"].astype(float)
    return df


def user_history(user_id: str, until, limit: int = 15) -> pd.DataFrame:
    """The user's latest transactions up to `until` (included), newest first."""
    rows = _fetch(
        """
        SELECT t.transaction_id, t."timestamp", t.type, t.amount_tnd, t.city, t.device_id,
               coalesce(t.merchant_id, t.recipient_id) AS counterparty, s.decision
        FROM transactions t
        LEFT JOIN scores s USING (transaction_id)
        WHERE t.user_id = %s AND t."timestamp" <= %s
        ORDER BY t."timestamp" DESC
        LIMIT %s
        """,
        (user_id, until, limit),
    )
    columns = ["transaction_id", "timestamp", "type", "amount_tnd", "city", "device_id",
               "counterparty", "decision"]  # fmt: skip
    df = pd.DataFrame(rows, columns=columns)
    df["amount_tnd"] = df["amount_tnd"].astype(float)
    return df


def save_label(transaction_id: str, label: str) -> None:
    """Record the analyst's verdict. A second verdict on the same alert replaces the first."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    with psycopg.connect(settings.postgres_dsn, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO analyst_labels (transaction_id, label) VALUES (%s, %s)
            ON CONFLICT (transaction_id)
            DO UPDATE SET label = EXCLUDED.label, labeled_at = now()
            """,
            (transaction_id, label),
        )
