import json

import numpy as np
import pandas as pd

from ml.train import THRESHOLDS, cost_curve, pick_block_threshold, recall_at_fpr, temporal_split


def test_temporal_split_is_chronological():
    ts = pd.Series(pd.date_range("2026-01-01", periods=100 * 24, freq="h"))
    split, boundaries = temporal_split(ts, warmup_days=7)
    assert list(split.unique()) == ["warmup", "train", "validation", "test"]
    assert ts[split == "train"].max() < ts[split == "validation"].min()
    assert ts[split == "validation"].max() < ts[split == "test"].min()
    assert boundaries["train_start"] == "2026-01-08"


def test_recall_at_fpr():
    y = np.array([0] * 100 + [1] * 4)
    scores = np.r_[np.linspace(0, 0.5, 100), [0.9, 0.8, 0.7, 0.1]]
    assert recall_at_fpr(y, scores, max_fpr=0.01) == 0.75
    assert recall_at_fpr(y, scores, max_fpr=1.0) == 1.0


def test_cost_curve_picks_the_cheapest_threshold():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.5, 0.9])
    # cheap false alarms: catch both frauds, accept one blocked customer
    cheap_fp = cost_curve(y, scores, cost_missed_fraud=100, cost_blocked_legit=1)
    assert cheap_fp.min() == 1 and 0.2 < THRESHOLDS[cheap_fp.argmin()] <= 0.5
    # expensive false alarms: better to miss the low-score fraud
    costly_fp = cost_curve(y, scores, cost_missed_fraud=1, cost_blocked_legit=100)
    assert costly_fp.min() == 1 and 0.6 < THRESHOLDS[costly_fp.argmin()] <= 0.9


def test_block_threshold_reaches_the_precision_target():
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.3, 0.6, 0.7, 0.65, 0.8, 0.9])
    # at 0.5 precision is 3/5; only above 0.7 is everything flagged a fraud
    block = pick_block_threshold(y, scores, min_precision=0.95, review_threshold=0.5)
    assert 0.7 < block <= 0.8
    # never below the review threshold
    assert pick_block_threshold(y, scores, 0.5, review_threshold=0.75) >= 0.75


def test_train_writes_artifacts(trained_models):
    out_dir, metrics = trained_models
    for name in (
        "model.json",
        "features.json",
        "threshold.json",
        "metrics.json",
        "threshold_cost.png",
        "shap_importance.png",
    ):
        assert (out_dir / name).exists()
    assert json.loads((out_dir / "metrics.json").read_text()) == metrics
    thresholds = metrics["threshold"]
    assert 0 < thresholds["review_threshold"] <= thresholds["block_threshold"] <= 1
    xgb, baseline = metrics["models"]["xgboost"], metrics["models"]["logistic_regression"]
    assert xgb["test"]["pr_auc"] > baseline["test"]["pr_auc"] > 0.015  # both beat random
