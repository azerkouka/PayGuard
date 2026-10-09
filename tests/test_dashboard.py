"""The dashboard page, rendered with canned query results (no PostgreSQL needed)."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import queries

ALERT_COLUMNS = ["transaction_id", "scored_at", "decision", "score", "top_reasons", "latency_ms",
                 "timestamp", "user_id", "amount_tnd", "type", "city", "device_id", "merchant_id",
                 "recipient_id", "label"]  # fmt: skip


def sample_alerts() -> pd.DataFrame:
    rows = [
        ("tx-1", pd.Timestamp("2026-10-09 10:00:05", tz="UTC"), "block", 0.997,
         [{"feature": "is_new_recipient", "description": "First transfer to this recipient",
           "impact": 3.7}],
         21.0, pd.Timestamp("2026-04-01 03:10:00"), "U00001", 1800.0, "p2p_transfer", "Tunis",
         "dev_a", None, "U04999", None),
        ("tx-2", pd.Timestamp("2026-10-09 10:00:01", tz="UTC"), "review", 0.61, [], 18.0,
         pd.Timestamp("2026-04-01 02:50:00"), "U00002", 2.5, "merchant_payment", "Sfax",
         "dev_b", "M0001", None, None),
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


def sample_history(user_id, until, limit=15) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["tx-1", "old-1"],
            "timestamp": [pd.Timestamp("2026-04-01 03:10:00"), pd.Timestamp("2026-03-30 14:00:00")],
            "type": ["p2p_transfer", "merchant_payment"],
            "amount_tnd": [1800.0, 22.0],
            "city": ["Tunis", "Tunis"],
            "device_id": ["dev_a", "dev_a"],
            "counterparty": ["U04999", "M0003"],
            "decision": ["block", None],
        }
    )


@pytest.fixture
def saved(monkeypatch):
    """Replace every query by canned data; returns the list of verdicts saved."""
    saved = []
    kpis = {"tx_per_min": 3000.0, "alert_rate": 0.036, "p95_latency_ms": 42.1, "labeled": 4,
            "confirmed": 3, "precision": 0.75}  # fmt: skip
    monkeypatch.setattr(queries, "kpis", lambda window_minutes=5: kpis)
    monkeypatch.setattr(queries, "alerts", lambda *args, **kwargs: sample_alerts())
    monkeypatch.setattr(queries, "user_history", sample_history)
    monkeypatch.setattr(queries, "save_label", lambda tx_id, label: saved.append((tx_id, label)))
    return saved


APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def run_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=30).run()


def test_page_shows_kpis_and_the_queue(saved):
    app = run_app()
    assert not app.exception
    assert [m.value for m in app.metric] == ["3,000", "3.6%", "75%", "42 ms"]
    queue = app.dataframe[0].value
    assert list(queue["Decision"]) == ["block", "review"]
    assert "is_fraud" not in queue.columns and "fraud_scenario" not in queue.columns


def test_page_without_alerts(saved, monkeypatch):
    monkeypatch.setattr(queries, "alerts", lambda *args, **kwargs: sample_alerts().iloc[:0])
    monkeypatch.setattr(
        queries, "kpis",
        lambda window_minutes=5: {"tx_per_min": 0.0, "alert_rate": None, "p95_latency_ms": None,
                                  "labeled": 0, "confirmed": 0, "precision": None},
    )  # fmt: skip
    app = run_app()
    assert not app.exception
    assert [m.value for m in app.metric] == ["0", "–", "–", "–"]
    assert "No alert to review" in app.info[0].value


def test_open_alert_and_confirm_fraud(saved):
    app = run_app()
    # simulate a click on the first row of the queue (a real browser keeps this selection
    # between runs; the test harness does not, so it is set again before the button click)
    selection = {"selection": {"rows": [0], "columns": []}}
    app.session_state["queue_0"] = selection
    app.run()
    assert not app.exception
    assert any("First transfer to this recipient" in m.value for m in app.markdown)

    app.session_state["queue_0"] = selection
    next(b for b in app.button if b.label == "Confirmed fraud").click().run()
    assert not app.exception
    assert saved == [("tx-1", "fraud")]
    assert app.session_state["queue_version"] == 1  # the alert is closed, the queue is live again
