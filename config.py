"""Central configuration: every setting is read from an environment variable, with a default.

Usage: `from config import settings`.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional; plain environment variables still work
    pass

ROOT_DIR = Path(__file__).resolve().parent


def _str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Paths
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", ROOT_DIR / "data"))
    models_dir: Path = field(default_factory=lambda: _path("MODELS_DIR", ROOT_DIR / "models"))

    # Simulator
    seed: int = field(default_factory=lambda: _int("SEED", 42))
    stream_rate: int = field(default_factory=lambda: _int("STREAM_RATE", 50))

    # Kafka
    kafka_bootstrap_servers: str = field(
        default_factory=lambda: _str("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    kafka_transactions_topic: str = field(
        default_factory=lambda: _str("KAFKA_TRANSACTIONS_TOPIC", "transactions")
    )
    kafka_alerts_topic: str = field(default_factory=lambda: _str("KAFKA_ALERTS_TOPIC", "alerts"))
    kafka_consumer_group: str = field(
        default_factory=lambda: _str("KAFKA_CONSUMER_GROUP", "payguard-consumer")
    )
    consumer_batch_size: int = field(default_factory=lambda: _int("CONSUMER_BATCH_SIZE", 200))

    # Redis
    redis_host: str = field(default_factory=lambda: _str("REDIS_HOST", "127.0.0.1"))
    redis_port: int = field(default_factory=lambda: _int("REDIS_PORT", 6379))
    # a user's known devices / cities / recipients are forgotten after this long without activity
    state_ttl_days: int = field(default_factory=lambda: _int("STATE_TTL_DAYS", 90))

    # PostgreSQL. 127.0.0.1, not localhost: on Windows with Docker Desktop, localhost goes
    # through IPv6 and batched inserts become ~25x slower
    postgres_host: str = field(default_factory=lambda: _str("POSTGRES_HOST", "127.0.0.1"))
    postgres_port: int = field(default_factory=lambda: _int("POSTGRES_PORT", 5434))
    postgres_db: str = field(default_factory=lambda: _str("POSTGRES_DB", "payguard"))
    postgres_user: str = field(default_factory=lambda: _str("POSTGRES_USER", "payguard"))
    postgres_password: str = field(default_factory=lambda: _str("POSTGRES_PASSWORD", "payguard"))

    # Decision threshold cost function (TND)
    cost_missed_fraud: float = field(default_factory=lambda: _float("COST_MISSED_FRAUD", 200.0))
    cost_blocked_legit: float = field(default_factory=lambda: _float("COST_BLOCKED_LEGIT", 5.0))
    # a transaction is blocked outright only above the score where this share of flags is fraud
    block_min_precision: float = field(
        default_factory=lambda: _float("BLOCK_MIN_PRECISION", 0.95)
    )

    # Scoring API
    api_host: str = field(default_factory=lambda: _str("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _int("API_PORT", 8010))

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
