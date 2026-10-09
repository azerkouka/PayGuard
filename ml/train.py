"""Offline training: `python -m ml.train`.

Temporal split, logistic regression baseline, XGBoost, cost-based thresholds, SHAP importance.
Writes model.json, features.json, threshold.json, metrics.json and two plots to models/.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import settings
from features import FEATURES, compute_features_batch
from ml.plots import plot_shap_importance, plot_threshold_cost

DEFAULT_DATA = settings.data_dir / "transactions.parquet"
THRESHOLDS = np.linspace(0, 1, 1001)
SHAP_SAMPLE = 20_000
EVALUATED_SPLITS = ("validation", "test")

# scale_pos_weight (class imbalance) and random_state are added at training time
XGBOOST_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50,
    "tree_method": "hist",
}


@dataclass(frozen=True)
class Dataset:
    """Features and labels of every transaction, with its place in the temporal split."""

    X: pd.DataFrame
    y: np.ndarray
    scenario: pd.Series  # fraud scenario name, null for legit; used for evaluation only
    parts: dict[str, np.ndarray]  # split name -> boolean mask
    boundaries: dict[str, str]

    def split(self, name: str) -> tuple[pd.DataFrame, np.ndarray]:
        mask = self.parts[name]
        return self.X[mask], self.y[mask]


# --- metrics and thresholds -------------------------------------------------------------------


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


def _flagged_counts(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each value of THRESHOLDS: frauds flagged and legit flagged (score >= threshold)."""
    fraud, legit = np.sort(scores[y == 1]), np.sort(scores[y == 0])
    true_positives = len(fraud) - np.searchsorted(fraud, THRESHOLDS, side="left")
    false_positives = len(legit) - np.searchsorted(legit, THRESHOLDS, side="left")
    return true_positives, false_positives


def cost_curve(
    y: np.ndarray, scores: np.ndarray, cost_missed_fraud: float, cost_blocked_legit: float
) -> np.ndarray:
    """Total cost for each value of THRESHOLDS, flagging a transaction when score >= threshold."""
    true_positives, false_positives = _flagged_counts(y, scores)
    missed = int((y == 1).sum()) - true_positives
    return missed * cost_missed_fraud + false_positives * cost_blocked_legit


def pick_block_threshold(
    y: np.ndarray, scores: np.ndarray, min_precision: float, review_threshold: float
) -> float:
    """Lowest threshold above the review one where at least `min_precision` of flags are fraud.

    Falls back to the most precise threshold if the target is never reached.
    """
    tp, fp = _flagged_counts(y, scores)
    precision = np.divide(tp, tp + fp, out=np.zeros(len(THRESHOLDS)), where=tp + fp > 0)
    allowed = (review_threshold <= THRESHOLDS) & (tp > 0)
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


# --- training steps ---------------------------------------------------------------------------


def load_dataset(data_path: Path, warmup_days: int) -> tuple[Dataset, dict]:
    """Read the transactions, compute their features and split them by date."""
    df = pd.read_parquet(data_path)
    # features are computed on the full history BEFORE splitting: a test transaction may look
    # back into the training period (as it would online), never forward
    X = compute_features_batch(df)
    y = df["is_fraud"].to_numpy().astype(int)
    split, boundaries = temporal_split(df["timestamp"], warmup_days=warmup_days)
    parts = {name: (split == name).to_numpy() for name in ("train", "validation", "test")}
    description = {
        "file": data_path.name,
        "rows": len(df),
        "fraud_rate": float(y.mean()),
        **boundaries,
        "splits": {
            name: {"rows": int(mask.sum()), "fraud_rate": float(y[mask].mean())}
            for name, mask in parts.items()
        },
    }
    return Dataset(X, y, df["fraud_scenario"], parts, boundaries), description


def fit_baseline(data: Dataset) -> Pipeline:
    baseline = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    return baseline.fit(*data.split("train"))


def fit_xgboost(data: Dataset, seed: int) -> XGBClassifier:
    """Gradient-boosted trees, frauds weighted up, stopped when validation PR-AUC stalls."""
    X_train, y_train = data.split("train")
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(**XGBOOST_PARAMS, scale_pos_weight=scale_pos_weight, random_state=seed)
    return model.fit(X_train, y_train, eval_set=[data.split("validation")], verbose=False)


def choose_thresholds(
    data: Dataset,
    scores: np.ndarray,
    cost_missed_fraud: float,
    cost_blocked_legit: float,
    block_min_precision: float,
) -> tuple[dict, dict, np.ndarray]:
    """Pick both thresholds on validation and report them on test.

    score >= review threshold -> an analyst looks at it; score >= block threshold -> refused.
    Returns (thresholds to save, their metrics, validation cost curve).
    """
    validation, test = data.parts["validation"], data.parts["test"]
    costs = cost_curve(
        data.y[validation], scores[validation], cost_missed_fraud, cost_blocked_legit
    )
    review = float(THRESHOLDS[costs.argmin()])
    block = pick_block_threshold(
        data.y[validation], scores[validation], block_min_precision, review
    )

    flagged = scores >= review
    recall_by_scenario = {
        name: float(flagged[test & (data.scenario == name).to_numpy()].mean())
        for name in sorted(data.scenario.dropna().unique())
    }
    thresholds = {
        "review_threshold": review,
        "block_threshold": block,
        "cost_missed_fraud": cost_missed_fraud,
        "cost_blocked_legit": cost_blocked_legit,
        "block_min_precision": block_min_precision,
    }
    report = {
        **thresholds,
        "validation_cost": float(costs.min()),
        "test": confusion_at(data.y[test], scores[test], review),
        "test_block": confusion_at(data.y[test], scores[test], block),
        "test_recall_by_scenario": recall_by_scenario,
    }
    return thresholds, report, costs


def shap_importance(model: XGBClassifier, data: Dataset, seed: int) -> pd.Series:
    """Global importance: mean |SHAP value| per feature on a sample of the test set."""
    X_test, _ = data.split("test")
    sample = X_test.sample(min(SHAP_SAMPLE, len(X_test)), random_state=seed)
    shap_values = shap.TreeExplainer(model).shap_values(sample)
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
    return importance.sort_values(ascending=False)


def train(
    data_path: Path = DEFAULT_DATA,
    out_dir: Path = settings.models_dir,
    seed: int = settings.seed,
    cost_missed_fraud: float = settings.cost_missed_fraud,
    cost_blocked_legit: float = settings.cost_blocked_legit,
    warmup_days: int = 7,
    block_min_precision: float = settings.block_min_precision,
) -> dict:
    """Train, evaluate and save everything. Returns the metrics written to metrics.json."""
    data, description = load_dataset(data_path, warmup_days)
    baseline = fit_baseline(data)
    model = fit_xgboost(data, seed)

    scores = {
        "logistic_regression": baseline.predict_proba(data.X)[:, 1],
        "xgboost": model.predict_proba(data.X)[:, 1],
    }
    model_metrics = {
        name: {
            part: evaluate(data.y[data.parts[part]], s[data.parts[part]])
            for part in EVALUATED_SPLITS
        }
        for name, s in scores.items()
    }
    thresholds, threshold_report, costs = choose_thresholds(
        data, scores["xgboost"], cost_missed_fraud, cost_blocked_legit, block_min_precision
    )
    importance = shap_importance(model, data, seed)

    metrics = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": description,
        "models": model_metrics,
        "xgboost_params": {
            "best_iteration": int(model.best_iteration),
            "scale_pos_weight": float(model.get_params()["scale_pos_weight"]),
            "learning_rate": XGBOOST_PARAMS["learning_rate"],
            "max_depth": XGBOOST_PARAMS["max_depth"],
        },
        "threshold": threshold_report,
        "shap_importance": {k: float(v) for k, v in importance.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / "model.json")
    (out_dir / "features.json").write_text(json.dumps(FEATURES, indent=2))
    (out_dir / "threshold.json").write_text(json.dumps(thresholds, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    review = thresholds["review_threshold"]
    plot_threshold_cost(THRESHOLDS, costs, review, out_dir / "threshold_cost.png")
    plot_shap_importance(importance, out_dir / "shap_importance.png")
    return metrics


# --- command line -----------------------------------------------------------------------------


def print_summary(metrics: dict) -> None:
    data = metrics["data"]
    print(f"train < {data['train_end']} <= validation < {data['validation_end']} <= test")
    for name, part in data["splits"].items():
        print(f"  {name:<11}{part['rows']:>8,} rows, fraud rate {part['fraud_rate']:.2%}")

    print(f"\n{'model':<21}{'split':<12}{'PR-AUC':>8}{'recall@1%FPR':>14}")
    for name, by_split in metrics["models"].items():
        for part, values in by_split.items():
            print(
                f"{name:<21}{part:<12}{values['pr_auc']:>8.3f}{values['recall_at_1pct_fpr']:>14.3f}"
            )

    t = metrics["threshold"]
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
    for name, value in list(metrics["shap_importance"].items())[:6]:
        print(f"  {name:<20}{value:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--cost-missed-fraud", type=float, default=settings.cost_missed_fraud)
    parser.add_argument("--cost-blocked-legit", type=float, default=settings.cost_blocked_legit)
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--block-min-precision", type=float, default=settings.block_min_precision)
    args = parser.parse_args()

    metrics = train(
        data_path=args.data,
        out_dir=args.out_dir,
        seed=args.seed,
        cost_missed_fraud=args.cost_missed_fraud,
        cost_blocked_legit=args.cost_blocked_legit,
        warmup_days=args.warmup_days,
        block_min_precision=args.block_min_precision,
    )
    print_summary(metrics)
    print(f"\nartifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
