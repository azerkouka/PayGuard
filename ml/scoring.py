"""Scoring library: features -> score, decision and reasons.

Used directly by the API and by the streaming consumer. Load a `Scorer` once, call `score` often.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from features.definitions import NIGHT_END_HOUR, TYPE_FLAGS

APPROVE, REVIEW, BLOCK = "approve", "review", "block"
N_REASONS = 3

TYPE_NAMES = {"type_p2p": "P2P transfer", "type_merchant": "merchant payment",
              "type_topup": "mobile top-up"}  # fmt: skip
WINDOW_NAMES = {"5m": "5 minutes", "1h": "hour", "24h": "24 hours"}

# features describing the same fact are explained as one reason
REASON_GROUPS = {**dict.fromkeys(TYPE_FLAGS, "type"), "hour": "time", "is_night": "time"}


def _duration(seconds: float) -> str:
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            n = seconds / size
            return f"{n:.0f} {unit}{'s' if round(n) != 1 else ''}"
    return f"{seconds:.0f} seconds"


def describe(group: str, f: Mapping[str, float]) -> str:
    """Plain-words description of a feature (or feature group) for an analyst."""
    if group == "type":
        return "Transaction type: " + next(TYPE_NAMES[k] for k in TYPE_FLAGS if f[k])
    if group == "time":
        hour = int(f["hour"])
        return f"Made at {hour:02d}:00" + (" (night)" if hour < NIGHT_END_HOUR else "")
    if group == "amount_tnd":
        return f"Amount of {f['amount_tnd']:,.3f} TND"
    if group == "amount_ratio":
        return f"Amount is {f['amount_ratio']:.1f}x this user's average"
    if group == "seconds_since_prev":
        return f"{_duration(f['seconds_since_prev'])} since this user's previous transaction"
    if group == "is_new_device":
        return "New device for this user" if f[group] else "Device already used by this user"
    if group == "is_new_city":
        return "New city for this user" if f[group] else "City already seen for this user"
    if group == "is_new_recipient":
        return "First transfer to this recipient" if f[group] else "No new recipient involved"
    stat, window = group.split("_")
    n = int(f[f"count_{window}"])
    window = WINDOW_NAMES[window]
    if n == 0:
        return f"No other transaction in the last {window}"
    if stat == "count":
        return f"{n} other transaction{'s' if n != 1 else ''} in the last {window}"
    if stat == "sum":
        return f"{f[group]:,.3f} TND already spent in the last {window}"
    return f"Average of {f[group]:,.3f} TND per transaction in the last {window}"


@dataclass(frozen=True)
class ScoreResult:
    score: float
    decision: str
    reasons: list[dict]  # [{"feature", "description", "impact"}], strongest first


class Scorer:
    def __init__(self, models_dir: Path):
        self.features: list[str] = json.loads((models_dir / "features.json").read_text())
        thresholds = json.loads((models_dir / "threshold.json").read_text())
        self.review_threshold: float = thresholds["review_threshold"]
        self.block_threshold: float = thresholds["block_threshold"]
        self.metrics: dict = json.loads((models_dir / "metrics.json").read_text())
        self.version: str = self.metrics["trained_at"]

        self.booster = xgb.Booster(model_file=models_dir / "model.json")
        self.booster.set_param({"nthread": 1})  # one row at a time: threads only add overhead
        # the saved model also holds the trees built after the early-stopping optimum
        self._trees = (0, self.booster.best_iteration + 1)

    def decide(self, score: float) -> str:
        if score >= self.block_threshold:
            return BLOCK
        if score >= self.review_threshold:
            return REVIEW
        return APPROVE

    def score(self, features: Mapping[str, float]) -> ScoreResult:
        row = np.array([[features[name] for name in self.features]], dtype=np.float32)
        matrix = xgb.DMatrix(row, feature_names=self.features)
        # SHAP values (exact TreeSHAP): one contribution per feature plus a bias, in log-odds.
        # Their sum is the model's raw output, so the score comes from the same call.
        contributions = self.booster.predict(
            matrix, pred_contribs=True, iteration_range=self._trees
        )[0]
        score = 1.0 / (1.0 + math.exp(-float(contributions.sum())))

        impact: dict[str, float] = {}
        for name, value in zip(self.features, contributions[:-1], strict=True):
            group = REASON_GROUPS.get(name, name)
            impact[group] = impact.get(group, 0.0) + float(value)
        top = sorted(impact.items(), key=lambda item: item[1], reverse=True)[:N_REASONS]
        reasons = [
            {"feature": group, "description": describe(group, features), "impact": round(value, 4)}
            for group, value in top
            if value > 0  # only what pushes the score towards fraud
        ]
        return ScoreResult(score=score, decision=self.decide(score), reasons=reasons)
