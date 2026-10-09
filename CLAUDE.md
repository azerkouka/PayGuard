# PayGuard

Real-time fraud detection for a mobile wallet (Tunisian market, amounts in TND).
A simulator produces transactions, a streaming pipeline computes features and scores them
with an XGBoost model, and analysts review alerts in a dashboard.

This is a learning / portfolio project: the owner must be able to explain every component
in an interview. Clarity beats cleverness.

## Architecture

```
simulator ──► Kafka `transactions` ──► feature + scoring consumer ──► PostgreSQL (transactions, scores)
                                              │        │
                                              │        └──► Kafka `alerts` ──► Streamlit dashboard ──► analyst_labels
                                              ▼
                                            Redis (per-user recent history, known devices/cities/recipients)

offline: simulator (batch, Parquet) ──► features (batch) ──► ml/train.py ──► models/
```

## Stack

- Python 3.13, pandas, scikit-learn, XGBoost, SHAP
- Kafka (KRaft, single broker), Redis, PostgreSQL, all via docker-compose
- FastAPI for the scoring API, Streamlit for the dashboard
- pytest, ruff, GitHub Actions

## Folder structure

```
config.py        central settings, read from environment variables with defaults
simulator/       user/merchant profiles, fraud scenarios, batch (Parquet) and streaming (Kafka) modes
features/        feature definitions, written ONCE and shared by offline training and online scoring
ml/              training, evaluation, threshold selection, SHAP
streaming/       Kafka consumers (features, scoring, alerts)
api/             FastAPI scoring service
dashboard/       Streamlit analyst dashboard
infra/           init.sql and other infrastructure files
scripts/         small operational scripts (health check, benchmark)
tests/           pytest unit tests (no Docker required)
docs/            plots shown in the README (copied from models/ after training)
data/            generated Parquet files (git-ignored)
models/          trained model, feature list, metrics, plots (git-ignored)
```

## Commands

```
make install     pip install -r requirements.txt (into the active virtual environment)
make up          start Kafka, Redis, PostgreSQL (docker compose up -d --wait)
make generate    python -m simulator.generate --days 90 --seed 42
make train       python -m ml.train
make stream      python -m scripts.run_pipeline (history load + consumer + simulator, Ctrl+C to stop)
make dashboard   streamlit run dashboard/app.py (second terminal)

make bootstrap   python -m streaming.bootstrap (load the history into Redis and PostgreSQL)
make consume     python -m streaming.scoring_consumer (features + scoring + alerts)
make simulate    python -m simulator.stream --rate 50 (simulator only)
make api         uvicorn api.main:app --port 8010
make benchmark   python -m scripts.benchmark (2 minutes, prints latency and throughput)
make check       python -m scripts.check_infra
make reset       wipe all volumes and start again
make test        pytest
make lint        ruff check .
```

From a clean clone the sequence is: `make install`, `make up`, `make generate`, `make train`,
`make stream`, `make dashboard`.

`make` is not installed by default on Windows; the underlying commands can be run directly.

## Conventions

- All settings come from `config.py`. No hard-coded hosts, ports, paths or thresholds elsewhere.
- Feature logic lives only in `features/`. Online and offline paths must give the same values
  (enforced by a parity test). A feature may only use data strictly before the transaction.
- All randomness is seeded and reproducible.
- Splits are temporal (train / validation / test by date), never random.
- Headline metrics are PR-AUC and recall at 1% FPR, not accuracy.
- Type hints on public functions, short docstrings, ruff-clean.
- Unit tests must run without Docker (use fakeredis and in-memory data).
- Keep memory usage low: this runs on a laptop with limited RAM.

## Working rules

- Build one step at a time. Propose the plan or design first, then implement.
- No business logic beyond the current step.
- Never invent numbers: metrics, latency and throughput in docs come from `models/` and real benchmark runs.
- When something breaks, explain the root cause in plain terms before changing code, then propose the smallest fix.
