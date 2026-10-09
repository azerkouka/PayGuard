"""Offline training: `python -m ml.train`.

Temporal split, logistic regression baseline, XGBoost, cost-based threshold, SHAP importance.
Writes model.json, features.json, threshold.json, metrics.json and two plots to models/.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import settings
from features import FEATURES, compute_features_batch
from ml.plots import plot_shap_importance, plot_threshold_cost

DEFAULT_DATA = settings.data_dir / "transactions.parquet"
THRESHOLDS = np.linspace(0, 1, 1001)
SHAP_SAMPLE = 20_000


def temporal_split(
    timestamps: pd.Series, train_frac: float = 0.70, val_frac: float = 0.15, warmup_days: int = 7
) -> tuple[pd.Series, dict[str, str]]:
    """Label each row warmup / train / validation / test by date, in chronological order.

    The first `warmup_days` are kept out of training: users have no history yet, so their
    features are not representative. They still count as history for later rows.
    """
    start, end = timestamps.min(), timestamps.max()
    train_start = start.normalize() + pd.Timedelta(days=warmup_days)
    train_end = (start + (end - start) * train_frac).normalize()
    val_end = (start + (end - start) * (train_frac + val_frac)).normalize()

    split = pd.Series("test", index=timestamps.index)
    split[timestamps < val_end] = "validation"
    split[timestamps < train_end] = "train"
    split[timestamps < train_start] = "warmup"
    boundaries = {"train_start": train_start, "train_end": train_end, "validation_end": val_end}
    return split, {k: str(v.date()) for k, v in boundaries.items()}


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> float:
    """Best recall reachable while flagging at most `max_fpr` of legit transactions."""
    fpr, tpr, _ = roc_curve(y, scores)
    return float(tpr[fpr <= max_fpr].max())


def evaluate(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "recall_at_1pct_fpr": recall_at_fpr(y, scores, 0.01),
    }


def cost_curve(
    y: np.ndarray, scores: np.ndarray, cost_missed_fraud: float, cost_blocked_legit: float
) -> np.ndarray:
    """Total cost for each value of THRESHOLDS, flagging a transaction when score >= threshold."""
    fraud, legit = np.sort(scores[y == 1]), np.sort(scores[y == 0])
    missed = np.searchsorted(fraud, THRESHOLDS, side="left")
    blocked = len(legit) - np.searchsorted(legit, THRESHOLDS, side="left")
    return missed * cost_missed_fraud + blocked * cost_blocked_legit


def pick_block_threshold(
    y: np.ndarray, scores: np.ndarray, min_precision: float, review_threshold: float
) -> float:
    """Lowest threshold above the review one where at least `min_precision` of flags are fraud.

    Falls back to the most precise threshold if the target is never reached.
    """
    fraud, legit = np.sort(scores[y == 1]), np.sort(scores[y == 0])
    tp = len(fraud) - np.searchsorted(fraud, THRESHOLDS, side="left")
    fp = len(legit) - np.searchsorted(legit, THRESHOLDS, side="left")
    precision = np.divide(tp, tp + fp, out=np.zeros(len(THRESHOLDS)), where=tp + fp > 0)
    allowed = (THRESHOLDS >= review_threshold) & (tp > 0)
    good = allowed & (precision >= min_precision)
    if good.any():
        return float(THRESHOLDS[good.argmax()])
    return float(THRESHOLDS[np.where(allowed, precision, -1).argmax()])


def confusion_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = scores >= threshold
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(int((y == 1).sum()), 1),
        "fpr": fp / max(int((y == 0).sum()), 1),
        "alert_rate": float(flagged.mean()),
    }


def train(
    data_path: Path = DEFAULT_DATA,
    out_dir: Path = settings.models_dir,
    seed: int = settings.seed,
    cost_missed_fraud: float = settings.cost_missed_fraud,
    cost_blocked_legit: float = settings.cost_blocked_legit,
    warmup_days: int = 7,
    block_min_precision: float = settings.block_min_precision,
) -> dict:
    df = pd.read_parquet(data_path)
    # features are computed on the full history BEFORE splitting: a test transaction may look
    # back into the training period (as it would online), never forward
    X = compute_features_batch(df)
    y = df["is_fraud"].to_numpy().astype(int)
    split, boundaries = temporal_split(df["timestamp"], warmup_days=warmup_days)
    parts = {name: (split == name).to_numpy() for name in ("train", "validation", "test")}
    X_train, y_train = X[parts["train"]], y[parts["train"]]
    X_val, y_val = X[parts["validation"]], y[parts["validation"]]

    baseline = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    baseline.fit(X_train, y_train)

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=seed,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    scores = {
        "logistic_regression": baseline.predict_proba(X)[:, 1],
        "xgboost": model.predict_proba(X)[:, 1],
    }
    model_metrics = {
        name: {part: evaluate(y[parts[part]], s[parts[part]]) for part in ("validation", "test")}
        for name, s in scores.items()
    }

    # thresholds: chosen on validation, reported on test
    # score >= review threshold -> an analyst looks at it; score >= block threshold -> refused
    xgb_scores = scores["xgboost"]
    val_scores = xgb_scores[parts["validation"]]
    costs = cost_curve(y_val, val_scores, cost_missed_fraud, cost_blocked_legit)
    threshold = float(THRESHOLDS[costs.argmin()])
    block_threshold = pick_block_threshold(y_val, val_scores, block_min_precision, threshold)
    test = parts["test"]
    flagged = xgb_scores >= threshold
    scenario = df["fraud_scenario"]
    recall_by_scenario = {
        name: float(flagged[test & (scenario == name).to_numpy()].mean())
        for name in sorted(scenario.dropna().unique())
    }

    # global importance: mean |SHAP value| per feature on a sample of the test set
    sample = X[test].sample(min(SHAP_SAMPLE, int(test.sum())), random_state=seed)
    shap_values = shap.TreeExplainer(model).shap_values(sample)
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
    importance = importance.sort_values(ascending=False)

    threshold_info = {
        "review_threshold": threshold,
        "block_threshold": block_threshold,
        "cost_missed_fraud": cost_missed_fraud,
        "cost_blocked_legit": cost_blocked_legit,
        "block_min_precision": block_min_precision,
    }
    metrics = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {
            "file": data_path.name,
            "rows": len(df),
            "fraud_rate": float(y.mean()),
            **boundaries,
            "splits": {
                name: {"rows": int(mask.sum()), "fraud_rate": float(y[mask].mean())}
                for name, mask in parts.items()
            },
        },
        "models": model_metrics,
        "xgboost_params": {
            "best_iteration": int(model.best_iteration),
            "scale_pos_weight": scale_pos_weight,
            "learning_rate": 0.05,
            "max_depth": 5,
        },
        "threshold": {
            **threshold_info,
            "validation_cost": float(costs.min()),
            "test": confusion_at(y[test], xgb_scores[test], threshold),
            "test_block": confusion_at(y[test], xgb_scores[test], block_threshold),
            "test_recall_by_scenario": recall_by_scenario,
        },
        "shap_importance": {k: float(v) for k, v in importance.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / "model.json")
    (out_dir / "features.json").write_text(json.dumps(FEATURES, indent=2))
    (out_dir / "threshold.json").write_text(json.dumps(threshold_info, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_threshold_cost(THRESHOLDS, costs, threshold, out_dir / "threshold_cost.png")
    plot_shap_importance(importance, out_dir / "shap_importance.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--cost-missed-fraud", type=float, default=settings.cost_missed_fraud)
    parser.add_argument("--cost-blocked-legit", type=float, default=settings.cost_blocked_legit)
    parser.add_argument("--warmup-days", type=int, default=7)
    args = parser.parse_args()

    m = train(
        args.data,
        args.out_dir,
        args.seed,
        args.cost_missed_fraud,
        args.cost_blocked_legit,
        args.warmup_days,
    )

    d = m["data"]
    print(f"train < {d['train_end']} <= validation < {d['validation_end']} <= test")
    for name, s in d["splits"].items():
        print(f"  {name:<11}{s['rows']:>8,} rows, fraud rate {s['fraud_rate']:.2%}")
    print(f"\n{'model':<21}{'split':<12}{'PR-AUC':>8}{'recall@1%FPR':>14}")
    for name, by_split in m["models"].items():
        for part, v in by_split.items():
            print(f"{name:<21}{part:<12}{v['pr_auc']:>8.3f}{v['recall_at_1pct_fpr']:>14.3f}")
    t = m["threshold"]
    print(
        f"\nreview threshold {t['review_threshold']:.3f} (missed fraud = "
        f"{t['cost_missed_fraud']:g} TND, blocked legit = {t['cost_blocked_legit']:g} TND), "
        "on test:"
    )
    print("  " + ", ".join(f"{k} {v:.3f}" for k, v in t["test"].items()))
    by_scenario = ", ".join(f"{k} {v:.3f}" for k, v in t["test_recall_by_scenario"].items())
    print(f"  recall by scenario: {by_scenario}")
    print(
        f"block threshold {t['block_threshold']:.3f} "
        f"(precision >= {t['block_min_precision']:g} on validation), on test:"
    )
    print("  " + ", ".join(f"{k} {v:.3f}" for k, v in t["test_block"].items()))
    print("\ntop features (mean |SHAP|):")
    for name, value in list(m["shap_importance"].items())[:6]:
        print(f"  {name:<20}{value:.3f}")
    print(f"\nartifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
