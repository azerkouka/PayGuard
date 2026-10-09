import pytest

from ml.train import train
from simulator.generate import generate_dataset


@pytest.fixture(scope="session")
def trained_models(tmp_path_factory):
    """A small model trained once for the whole test session: (directory, metrics)."""
    out_dir = tmp_path_factory.mktemp("models")
    df, _ = generate_dataset(seed=3, days=30, n_users=300, n_merchants=60, n_transactions=20_000)
    df.to_parquet(out_dir / "transactions.parquet", index=False)
    metrics = train(out_dir / "transactions.parquet", out_dir, warmup_days=3)
    return out_dir, metrics
