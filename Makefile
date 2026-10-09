.PHONY: install up down logs reset check generate train bootstrap consume api stream benchmark dashboard test lint

install:
	python -m pip install -r requirements.txt

# Infrastructure: Kafka, Redis, PostgreSQL
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

generate:
	python -m simulator.generate --days 90 --seed 42

train:
	python -m ml.train

# load the users' historical state into Redis (run after generate, before consume)
bootstrap:
	python -m streaming.bootstrap

# full online pipeline: features, scoring, alerts
consume:
	python -m streaming.scoring_consumer

# runs the pipeline for 2 minutes and prints latency and throughput
benchmark:
	python -m scripts.benchmark --duration 120 --rate 50

api:
	python -m uvicorn api.main:app --host $(or $(API_HOST),127.0.0.1) --port $(or $(API_PORT),8010)

stream:
	python -m simulator.stream --rate 50

dashboard:
	@echo "TODO (step 8): streamlit run dashboard/app.py"

test:
	python -m pytest

lint:
	python -m ruff check .
