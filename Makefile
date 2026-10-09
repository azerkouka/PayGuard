# From a clean clone:
#   python -m venv .venv, activate it, then: make install
#   make up -> make generate -> make train -> make stream -> make dashboard (second terminal)
# `make` is not installed by default on Windows: each target is one command you can run directly.

.PHONY: help install up down logs reset check generate train stream dashboard \
        bootstrap consume simulate api benchmark test lint

help:
	@echo "make install    install Python dependencies into the active environment"
	@echo "make up         start Kafka, Redis and PostgreSQL"
	@echo "make generate   create 90 days of transactions in data/"
	@echo "make train      train the model into models/"
	@echo "make stream     run the live pipeline (history load + consumer + simulator)"
	@echo "make dashboard  open the analyst dashboard (second terminal)"
	@echo "make test / lint / benchmark / api / check / down / reset / logs"

install:
	python -m pip install -r requirements.txt

# --- infrastructure: Kafka, Redis, PostgreSQL ---
up:
	docker compose up -d --wait

down:
	docker compose down

logs:
	docker compose logs -f

# deletes all data (volumes) and starts again from empty
reset:
	docker compose down -v
	docker compose up -d --wait

check:
	python -m scripts.check_infra

# --- offline: data and model ---
generate:
	python -m simulator.generate --days 90 --seed 42

train:
	python -m ml.train

# --- online ---
# everything needed for a live demo, in one process tree; Ctrl+C stops it cleanly
stream:
	python -m scripts.run_pipeline --rate 50

dashboard:
	python -m streamlit run dashboard/app.py

# the three parts of `make stream`, to run separately
bootstrap:
	python -m streaming.bootstrap

consume:
	python -m streaming.scoring_consumer

simulate:
	python -m simulator.stream --rate 50

api:
	python -m uvicorn api.main:app --host 127.0.0.1 --port 8010

# runs the pipeline for 2 minutes and prints latency and throughput
benchmark:
	python -m scripts.benchmark --duration 120 --rate 50

# --- quality ---
test:
	python -m pytest

lint:
	python -m ruff check .
