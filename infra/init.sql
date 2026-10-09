-- Runs once, the first time the PostgreSQL volume is created (`make reset` to run it again).

-- Raw transactions as received from Kafka.
-- The primary key makes inserts idempotent: a transaction delivered twice is stored once.
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  TEXT PRIMARY KEY,
    "timestamp"     TIMESTAMP NOT NULL,          -- local Tunisian time, as produced by the simulator
    user_id         TEXT NOT NULL,
    merchant_id     TEXT,
    recipient_id    TEXT,
    amount_tnd      NUMERIC(12, 3) NOT NULL,
    type            TEXT NOT NULL,
    device_id       TEXT NOT NULL,
    city            TEXT NOT NULL,
    is_fraud        BOOLEAN,                     -- simulator ground truth, for evaluation only
    fraud_scenario  TEXT,                        -- never shown to analysts
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS transactions_user_time_idx ON transactions (user_id, "timestamp" DESC);
CREATE INDEX IF NOT EXISTS transactions_ingested_idx ON transactions (ingested_at);

-- One row per scored transaction.
-- No foreign key to transactions: both tables are written from the stream and order is not guaranteed.
CREATE TABLE IF NOT EXISTS scores (
    transaction_id  TEXT PRIMARY KEY,
    score           DOUBLE PRECISION NOT NULL,
    decision        TEXT NOT NULL CHECK (decision IN ('approve', 'review', 'block')),
    top_reasons     JSONB NOT NULL DEFAULT '[]',
    latency_ms      DOUBLE PRECISION NOT NULL,
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS scores_scored_at_idx ON scores (scored_at);
CREATE INDEX IF NOT EXISTS scores_alerts_idx ON scores (scored_at DESC) WHERE decision <> 'approve';

-- Analyst feedback on alerts.
CREATE TABLE IF NOT EXISTS analyst_labels (
    transaction_id  TEXT PRIMARY KEY,
    label           TEXT NOT NULL CHECK (label IN ('fraud', 'false_positive')),
    labeled_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
