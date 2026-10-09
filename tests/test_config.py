from config import Settings


def test_defaults():
    settings = Settings()
    assert settings.kafka_transactions_topic == "transactions"
    assert settings.data_dir.name == "data"
    assert settings.postgres_dsn.startswith("postgresql://")


def test_environment_override(monkeypatch):
    monkeypatch.setenv("REDIS_PORT", "6380")
    assert Settings().redis_port == 6380
