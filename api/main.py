"""Scoring API: `uvicorn api.main:app --port 8010`.

POST /score      score one transaction (read-only: the user's state is not modified)
GET  /health     liveness, including Redis
GET  /model-info model version, thresholds and metrics
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import redis
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from config import settings
from features.online import compute_features
from features.store import FeatureStore
from ml.scoring import Scorer
from simulator.schema import MERCHANT, P2P, TOPUP

LOCAL_TZ = ZoneInfo("Africa/Tunis")


class Transaction(BaseModel):
    transaction_id: str = Field(min_length=1)
    timestamp: datetime
    user_id: str = Field(min_length=1)
    merchant_id: str | None = None
    recipient_id: str | None = None
    amount_tnd: float = Field(gt=0)
    type: Literal[P2P, MERCHANT, TOPUP]
    device_id: str = Field(min_length=1)
    city: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def to_local_naive(cls, value: datetime) -> datetime:
        """The model was trained on local Tunisian time without timezone (hour of day matters)."""
        if value.tzinfo is not None:
            value = value.astimezone(LOCAL_TZ).replace(tzinfo=None)
        return value

    @model_validator(mode="after")
    def counterparty_matches_type(self):
        if self.type == P2P and not self.recipient_id:
            raise ValueError("a p2p_transfer needs a recipient_id")
        if self.type != P2P and not self.merchant_id:
            raise ValueError(f"a {self.type} needs a merchant_id")
        return self


class Reason(BaseModel):
    feature: str
    description: str
    impact: float  # SHAP value, in log-odds: how much this pushed the score up


class ScoreResponse(BaseModel):
    transaction_id: str
    score: float
    decision: Literal["approve", "review", "block"]
    reasons: list[Reason]
    latency_ms: float
    model_version: str


def create_app(scorer: Scorer | None = None, store: FeatureStore | None = None) -> FastAPI:
    """Build the app. Pass `scorer` and `store` in tests; otherwise they are created at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # loaded once here, reused by every request
        app.state.scorer = scorer or Scorer(settings.models_dir)
        app.state.store = store or FeatureStore(
            redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True),
            state_ttl_seconds=settings.state_ttl_days * 86400,
        )
        yield

    app = FastAPI(title="PayGuard scoring API", lifespan=lifespan)

    @app.post("/score", response_model=ScoreResponse)
    def score(transaction: Transaction, request: Request) -> ScoreResponse:
        started = time.perf_counter()
        tx = transaction.model_dump()
        try:
            state, already_applied = request.app.state.store.load(tx)
        except redis.RedisError as exc:
            raise HTTPException(503, "feature store unavailable") from exc
        if already_applied:
            # its own effect is already in the user's state: features would no longer be "past only"
            raise HTTPException(409, "transaction already processed by the stream")

        scorer: Scorer = request.app.state.scorer
        result = scorer.score(compute_features(tx, state))
        return ScoreResponse(
            transaction_id=transaction.transaction_id,
            score=result.score,
            decision=result.decision,
            reasons=result.reasons,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            model_version=scorer.version,
        )

    @app.get("/health")
    def health(request: Request) -> dict:
        try:
            request.app.state.store.client.ping()
        except redis.RedisError as exc:
            raise HTTPException(503, "redis unreachable") from exc
        return {"status": "ok", "redis": True, "model_version": request.app.state.scorer.version}

    @app.get("/model-info")
    def model_info(request: Request) -> dict:
        scorer: Scorer = request.app.state.scorer
        return {
            "version": scorer.version,
            "review_threshold": scorer.review_threshold,
            "block_threshold": scorer.block_threshold,
            "features": scorer.features,
            "metrics": scorer.metrics["models"],
            "threshold_metrics": scorer.metrics["threshold"],
        }

    return app


app = create_app()
