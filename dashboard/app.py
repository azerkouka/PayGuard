"""Analyst dashboard: `streamlit run dashboard/app.py`.

KPIs on top, the live queue of alerts on the left, the selected alert on the right.
The page refreshes by itself; while an alert is open the queue is frozen so that rows do not
move under the analyst's eyes.
"""

import sys
from pathlib import Path

# `streamlit run` puts dashboard/ on the import path, not the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import psycopg
import streamlit as st

from dashboard import queries
from schema import LOCAL_TIMEZONE

KPI_WINDOW_MINUTES = 5
HISTORY_ROWS = 15
TYPE_NAMES = {
    "p2p_transfer": "P2P transfer",
    "merchant_payment": "Merchant payment",
    "mobile_topup": "Mobile top-up",
}
LABEL_NAMES = {"fraud": "Confirmed fraud", "false_positive": "False positive"}


def show_kpis() -> None:
    k = queries.kpis(KPI_WINDOW_MINUTES)
    tx, alert, precision, latency = st.columns(4)
    tx.metric("Transactions / min", f"{k['tx_per_min']:,.0f}")
    alert.metric("Alert rate", "–" if k["alert_rate"] is None else f"{k['alert_rate']:.1%}")
    precision.metric(
        "Confirmed-fraud precision",
        "–" if k["precision"] is None else f"{k['precision']:.0%}",
        help=f"{k['confirmed']} confirmed frauds out of {k['labeled']} alerts reviewed so far.",
    )
    latency.metric(
        "p95 latency",
        "–" if k["p95_latency_ms"] is None else f"{k['p95_latency_ms']:,.0f} ms",
        help="From publication in Kafka to the score.",
    )
    st.caption(
        f"Traffic, alert rate and latency over the last {KPI_WINDOW_MINUTES} minutes; "
        "precision over every alert reviewed."
    )


def show_queue(alerts: pd.DataFrame, key: str) -> None:
    table = pd.DataFrame(
        {
            "Scored at": pd.to_datetime(alerts["scored_at"], utc=True)
            .dt.tz_convert(LOCAL_TIMEZONE)
            .dt.strftime("%H:%M:%S"),
            "Decision": alerts["decision"],
            "Score": alerts["score"],
            "Amount (TND)": alerts["amount_tnd"],
            "User": alerts["user_id"],
            "Type": alerts["type"].map(TYPE_NAMES),
            "Verdict": alerts["label"].map(LABEL_NAMES).fillna(""),
        }
    )
    st.dataframe(
        table,
        key=key,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        height=560,
        column_config={
            "Score": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "Amount (TND)": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def show_alert(alert: pd.Series) -> bool:
    """Details of one alert and the verdict buttons. Returns True when a verdict was saved."""
    counterparty = alert["recipient_id"] or alert["merchant_id"]
    st.subheader(f"{TYPE_NAMES[alert['type']]} of {alert['amount_tnd']:,.3f} TND")
    st.caption(f"Transaction {alert['transaction_id']}")

    score, decision, when = st.columns(3)
    score.metric("Score", f"{alert['score']:.3f}")
    decision.metric("Decision", alert["decision"])
    when.metric("Transaction time", f"{alert['timestamp']:%d %b %H:%M}")
    st.markdown(
        f"**User** {alert['user_id']} → **{'Recipient' if alert['recipient_id'] else 'Merchant'}** "
        f"{counterparty}  \n**City** {alert['city']} · **Device** {alert['device_id']}"
    )

    st.markdown("**Why it was flagged**")
    reasons = alert["top_reasons"]
    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason['description']}  (+{reason['impact']:.2f})")
    else:
        st.caption("The model gave no specific reason.")

    saved = False
    fraud, false_positive = st.columns(2)
    if fraud.button("Confirmed fraud", type="primary", width="stretch"):
        queries.save_label(alert["transaction_id"], "fraud")
        saved = True
    if false_positive.button("False positive", width="stretch"):
        queries.save_label(alert["transaction_id"], "false_positive")
        saved = True
    if alert["label"]:
        st.caption(f"Current verdict: {LABEL_NAMES[alert['label']]}. A new click replaces it.")

    st.markdown(f"**This user's last {HISTORY_ROWS} transactions**")
    history = queries.user_history(alert["user_id"], alert["timestamp"], HISTORY_ROWS)
    history.insert(
        0, "", (history["transaction_id"] == alert["transaction_id"]).map({True: "▶", False: ""})
    )
    history["type"] = history["type"].map(TYPE_NAMES)
    history["decision"] = history["decision"].fillna("")
    st.dataframe(
        history.drop(columns="transaction_id"),
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="DD MMM HH:mm"),
            "type": "Type",
            "amount_tnd": st.column_config.NumberColumn("Amount (TND)", format="%.3f"),
            "city": "City",
            "device_id": "Device",
            "counterparty": "To",
            "decision": "Decision",
        },
    )
    return saved


def live_view(decisions: list[str], include_labeled: bool) -> None:
    state = st.session_state
    state.setdefault("queue_version", 0)
    key = f"queue_{state.queue_version}"

    # the row clicked in the queue, looked up in the data that was on screen when it was clicked
    selection = state.get(key)
    rows = selection["selection"]["rows"] if selection else []
    shown = state.get("alerts_shown")
    selected = shown.iloc[rows[0]] if rows and shown is not None and rows[0] < len(shown) else None

    show_kpis()
    if selected is None:
        alerts = queries.alerts(decisions, include_labeled)
        state.alerts_shown = alerts
    else:
        alerts = shown  # frozen while an alert is open

    queue, detail = st.columns([5, 4], gap="large")
    with queue:
        st.subheader(f"Alerts ({len(alerts)})")
        if alerts.empty:
            st.info("No alert to review. Start the stream and the consumer to see some.")
        else:
            show_queue(alerts, key)
    with detail:
        if selected is None:
            st.subheader("Alert details")
            st.caption("Click a row in the queue to open an alert.")
            return
        st.caption("The queue is paused while this alert is open.")
        closed = st.button("Close and resume live updates")
        if show_alert(selected) or closed:
            state.queue_version += 1  # new table: selection cleared, queue reloaded
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="PayGuard alerts", layout="wide")
    st.title("PayGuard · fraud alerts")

    with st.sidebar:
        st.header("Queue")
        decisions = st.multiselect("Decisions", ["block", "review"], default=["block", "review"])
        include_labeled = st.toggle("Show alerts already reviewed", value=False)
        st.header("Refresh")
        live = st.toggle("Live updates", value=True)
        seconds = st.slider("Every (seconds)", 2, 15, 3, disabled=not live)

    @st.fragment(run_every=seconds if live else None)
    def refreshing_view():
        try:
            live_view(decisions, include_labeled)
        except psycopg.OperationalError as exc:
            st.error(f"PostgreSQL is not reachable (is `make up` running?): {exc}")

    refreshing_view()


main()
