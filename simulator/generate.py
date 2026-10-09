"""Batch simulator: `python -m simulator.generate --days 90 --seed 42`."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from simulator.legit import BURST_INFLATION, add_bursts, build_legit, sample_users_and_times
from simulator.profiles import LifeEvents, Population, build_population, plan_life_events
from simulator.scenarios import Seen, inject_fraud
from simulator.schema import finalize

DEFAULT_OUTPUT = settings.data_dir / "transactions.parquet"
DEFAULT_START_DATE = "2026-01-01"
DEFAULT_DAYS = 90
DEFAULT_USERS = 5000
DEFAULT_MERCHANTS = 500
DEFAULT_TRANSACTIONS = 500_000
DEFAULT_FRAUD_RATE = 0.015


def generate_window(
    pop: Population,
    events: LifeEvents,
    start: np.datetime64,
    days: int,
    n_transactions: int,
    fraud_rate: float,
    rng: np.random.Generator,
    seen: Seen | None = None,
) -> pd.DataFrame:
    """Legit and fraudulent transactions between `start` and `start + days`, unsorted.

    Shared by the batch generator (one 90-day window) and the stream (one day at a time).
    A few transactions end shortly after the window: bursts and incidents started near its end.
    """
    n_base = round(n_transactions * (1 - fraud_rate) / BURST_INFLATION)  # bursts add rows
    u, ts = sample_users_and_times(pop, n_base, start, days, rng)
    legit = add_bursts(build_legit(pop, events, u, ts, rng), rng)

    n_fraud = round(len(legit) * fraud_rate / (1 - fraud_rate))
    fraud = inject_fraud(pop, events, n_fraud, start, days, rng, seen=seen)
    return pd.concat([legit, fraud], ignore_index=True)


def generate_dataset(
    days: int = DEFAULT_DAYS,
    seed: int = settings.seed,
    n_users: int = DEFAULT_USERS,
    n_merchants: int = DEFAULT_MERCHANTS,
    n_transactions: int = DEFAULT_TRANSACTIONS,
    fraud_rate: float = DEFAULT_FRAUD_RATE,
    start_date: str = DEFAULT_START_DATE,
) -> tuple[pd.DataFrame, Population]:
    """Generate a labelled transaction history. Same arguments give the same output."""
    rng = np.random.default_rng(seed)
    start = np.datetime64(start_date, "s")

    pop = build_population(n_users, n_merchants, rng)
    events = plan_life_events(pop, start, days, rng)

    df = generate_window(pop, events, start, days, n_transactions, fraud_rate, rng)
    return finalize(df, rng), pop


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic wallet transactions.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--merchants", type=int, default=DEFAULT_MERCHANTS)
    parser.add_argument("--transactions", type=int, default=DEFAULT_TRANSACTIONS)
    parser.add_argument("--fraud-rate", type=float, default=DEFAULT_FRAUD_RATE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    t0 = time.perf_counter()
    df, _ = generate_dataset(
        days=args.days,
        seed=args.seed,
        n_users=args.users,
        n_merchants=args.merchants,
        n_transactions=args.transactions,
        fraud_rate=args.fraud_rate,
        start_date=args.start_date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"{len(df):,} transactions from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"fraud rate: {df['is_fraud'].mean():.2%}")
    print(df["fraud_scenario"].value_counts().to_string())
    print(f"written to {args.output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
