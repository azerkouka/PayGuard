import fakeredis
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api.main import create_app
from features import FEATURES, compute_features_batch
from features.store import FeatureStore
from ml.scoring import APPROVE, BLOCK, REVIEW, Scorer


@pytest.fixture(scope="module")
def scorer(trained_models):
    return Scorer(trained_models[0])


@pytest.fixture(scope="module")
def history(trained_models):
    return pd.read_parquet(trained_models[0] / "transactions.parquet")


@pytest.fixture
def store(history):
    store = FeatureStore(fakeredis.FakeRedis(decode_responses=True))
    store.bootstrap(history)
    return store


@pytest.fixture
def client(scorer, store):
    with TestClient(create_app(scorer, store)) as client:
        yield client


def usual_payment(history, **changes) -> dict:
    """A new merchant payment that looks like the user's last one, one day later."""
    last = history[history["type"] == "merchant_payment"].iloc[-1]
    tx = {
        "transaction_id": "new-1",
        "timestamp": str(last["timestamp"].normalize() + pd.Timedelta(days=1, hours=14)),
        "user_id": last["user_id"],
        "merchant_id": last["merchant_id"],
        "recipient_id": None,
        "amount_tnd": float(last["amount_tnd"]),
        "type": "merchant_payment",
        "device_id": last["device_id"],
        "city": last["city"],
    }
    return {**tx, **changes}


def night_drain(history) -> dict:
    return usual_payment(
        history,
        transaction_id="new-2",
        timestamp=str(history["timestamp"].max().normalize() + pd.Timedelta(days=1, hours=3)),
        type="p2p_transfer",
        merchant_id=None,
        recipient_id="U-never-seen",
        amount_tnd=5000.0,
        device_id="dev_unknown",
        city="Tataouine",
    )


def test_scorer_matches_the_trained_model(scorer, history, trained_models):
    """Same scores as the sklearn wrapper used in training (early stopping included)."""
    X = compute_features_batch(history).iloc[-300:]
    model = XGBClassifier()
    model.load_model(trained_models[0] / "model.json")
    expected = model.predict_proba(X)[:, 1]
    scores = [scorer.score(row).score for row in X.to_dict("records")]
    np.testing.assert_allclose(scores, expected, atol=1e-5)


def test_decision_thresholds(scorer):
    assert scorer.decide(scorer.review_threshold - 1e-6) == APPROVE
    assert scorer.decide(scorer.review_threshold) == REVIEW
    assert scorer.decide(scorer.block_threshold) == BLOCK


def test_score_usual_payment_is_approved(client, history):
    response = client.post("/score", json=usual_payment(history))
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "new-1"
    assert 0 <= body["score"] <= 1
    assert body["decision"] == APPROVE
    assert body["latency_ms"] > 0


def test_score_night_drain_is_flagged_with_reasons(client, history):
    usual = client.post("/score", json=usual_payment(history)).json()
    body = client.post("/score", json=night_drain(history)).json()
    assert body["score"] > usual["score"]
    assert body["decision"] in (REVIEW, BLOCK)
    reasons = body["reasons"]
    assert 1 <= len(reasons) <= 3
    assert all(r["impact"] > 0 and r["description"] for r in reasons)
    assert [r["impact"] for r in reasons] == sorted((r["impact"] for r in reasons), reverse=True)
    assert len({r["feature"] for r in reasons}) == len(reasons)


def test_score_does_not_modify_the_state(client, store, history):
    tx = usual_payment(history)
    first = client.post("/score", json=tx).json()
    second = client.post("/score", json=tx).json()
    assert first["score"] == second["score"]
    assert store.load({**tx, "timestamp": pd.Timestamp(tx["timestamp"])})[1] is False


def test_score_conflict_when_already_processed_by_the_stream(client, store, history):
    tx = usual_payment(history)
    store.apply({**tx, "timestamp": pd.Timestamp(tx["timestamp"])})
    assert client.post("/score", json=tx).status_code == 409


def test_score_unknown_user_is_scored(client, history):
    tx = usual_payment(history, user_id="U-brand-new", device_id="dev_x")
    assert client.post("/score", json=tx).status_code == 200


@pytest.mark.parametrize(
    "changes",
    [
        {"amount_tnd": -5},
        {"amount_tnd": 0},
        {"type": "wire"},
        {"merchant_id": None},  # a merchant payment without merchant
        {"timestamp": "yesterday"},
        {"device_id": ""},
    ],
)
def test_score_rejects_invalid_input(client, history, changes):
    assert client.post("/score", json=usual_payment(history, **changes)).status_code == 422


def test_score_accepts_extra_fields_and_timezones(client, history):
    tx = usual_payment(history, is_fraud=False, fraud_scenario=None)
    naive = client.post("/score", json=tx).json()
    aware = client.post("/score", json={**tx, "timestamp": tx["timestamp"] + "+01:00"}).json()
    assert naive["score"] == aware["score"]  # Tunisia is UTC+1 all year


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["redis"] is True


def test_model_info(client, scorer):
    body = client.get("/model-info").json()
    assert body["version"] == scorer.version
    assert body["features"] == FEATURES
    assert body["review_threshold"] <= body["block_threshold"]
    assert set(body["metrics"]) == {"logistic_regression", "xgboost"}
