"""Redis-backed user state for online features.

Keys per user (the {braces} keep a user's keys on the same node in a Redis cluster):
    u:{user}:recent      sorted set, score = epoch seconds, member = "transaction_id|amount"
    u:{user}:devices     set
    u:{user}:cities      set
    u:{user}:recipients  set
    u:{user}:stats       hash: tx_count, amount_sum, last_ts (all-time)

The feature values themselves come from `features.online.compute_features`: this module only
loads a `UserState` from Redis and writes a transaction back.
"""

from collections import deque
from collections.abc import Mapping

import pandas as pd
import redis

from config import settings
from features.definitions import HISTORY_SECONDS
from features.online import UserState, compute_features, epoch_seconds
from schema import P2P

RECENT_TTL_SECONDS = HISTORY_SECONDS  # wall clock; reads also filter by score

# Atomic and idempotent: if the transaction is already in `recent`, nothing changes.
# This is what makes replaying a Kafka message safe (at-least-once delivery).
# Limit: `recent` only keeps 24 h, so a transaction replayed after its user has moved more than
# 24 h ahead (in transaction time) would be counted twice. Kafka only replays the last
# uncommitted batch, a few seconds of traffic, so this does not happen in practice.
APPLY_LUA = """
if redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1]) == 0 then
    return 0
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', '(' .. ARGV[7])
redis.call('EXPIRE', KEYS[1], ARGV[8])
redis.call('SADD', KEYS[2], ARGV[4])
redis.call('SADD', KEYS[3], ARGV[5])
if ARGV[6] ~= '' then
    redis.call('SADD', KEYS[4], ARGV[6])
end
redis.call('HINCRBY', KEYS[5], 'tx_count', 1)
redis.call('HINCRBYFLOAT', KEYS[5], 'amount_sum', ARGV[3])
redis.call('HSET', KEYS[5], 'last_ts', ARGV[2])
for i = 2, 5 do
    redis.call('EXPIRE', KEYS[i], ARGV[9])
end
if ARGV[10] ~= '' then
    redis.call('SET', KEYS[6], ARGV[10], 'EX', ARGV[11])
end
return 1
"""

# how long the result computed for a transaction is kept, to be reused if Kafka replays it
RESULT_TTL_SECONDS = 600


def result_key(user_id: str, transaction_id: str) -> str:
    return f"u:{{{user_id}}}:result:{transaction_id}"


def user_keys(user_id: str) -> list[str]:
    prefix = f"u:{{{user_id}}}"
    return [f"{prefix}:{name}" for name in ("recent", "devices", "cities", "recipients", "stats")]


def recent_member(transaction_id: str, amount: float) -> str:
    return f"{transaction_id}|{float(amount)!r}"


class FeatureStore:
    def __init__(self, client: redis.Redis, state_ttl_seconds: int = 90 * 86400):
        """`client` must be created with decode_responses=True (see `from_settings`)."""
        self.client = client
        self.state_ttl_seconds = state_ttl_seconds
        self._apply = client.register_script(APPLY_LUA)

    @classmethod
    def from_settings(cls) -> "FeatureStore":
        """The store on the Redis instance described by the configuration."""
        client = redis.Redis(
            host=settings.redis_host, port=settings.redis_port, decode_responses=True
        )
        return cls(client, state_ttl_seconds=settings.state_ttl_days * 86400)

    def load(self, tx: Mapping) -> tuple[UserState, bool]:
        """The user's state, and whether `tx` is already part of it (a replayed message)."""
        recent, devices, cities, recipients, stats = user_keys(tx["user_id"])
        ts = epoch_seconds(tx["timestamp"])
        pipe = self.client.pipeline(transaction=False)  # one round trip
        pipe.zrangebyscore(recent, ts - HISTORY_SECONDS, "+inf", withscores=True)
        pipe.smembers(devices)
        pipe.smembers(cities)
        pipe.smembers(recipients)
        pipe.hmget(stats, "tx_count", "amount_sum", "last_ts")
        pipe.zscore(recent, recent_member(tx["transaction_id"], tx["amount_tnd"]))
        entries, devices, cities, recipients, (count, total, last_ts), own_score = pipe.execute()

        state = UserState(
            recent=deque((score, float(member.rsplit("|", 1)[1])) for member, score in entries),
            devices=devices,
            cities=cities,
            recipients=recipients,
            tx_count=int(count or 0),
            amount_sum=float(total or 0.0),
            last_ts=None if last_ts is None else float(last_ts),
        )
        return state, own_score is not None

    def apply(self, tx: Mapping, result: str = "") -> bool:
        """Add `tx` to the user's state. Returns False if it was already there.

        `result` (for example the score, as JSON) is saved in the same atomic step and can be
        read back with `saved_result` if the transaction is replayed.
        """
        ts = epoch_seconds(tx["timestamp"])
        amount = float(tx["amount_tnd"])
        recipient = tx["recipient_id"] if tx["type"] == P2P else ""
        args = [
            recent_member(tx["transaction_id"], amount),
            repr(ts),
            repr(amount),
            tx["device_id"],
            tx["city"],
            recipient,
            repr(ts - HISTORY_SECONDS),
            RECENT_TTL_SECONDS,
            self.state_ttl_seconds,
            result,
            RESULT_TTL_SECONDS,
        ]
        keys = [*user_keys(tx["user_id"]), result_key(tx["user_id"], tx["transaction_id"])]
        return bool(self._apply(keys=keys, args=args))

    def saved_result(self, tx: Mapping) -> str | None:
        return self.client.get(result_key(tx["user_id"], tx["transaction_id"]))

    def process(self, tx: Mapping) -> dict[str, float] | None:
        """Features of `tx` from the past only, then `tx` joins the past.

        Returns None for a transaction that was already applied: its state can no longer be
        seen "from before", so features are not recomputed.
        """
        state, already_applied = self.load(tx)
        if already_applied:
            return None
        features = compute_features(tx, state)
        self.apply(tx)
        return features

    def clear(self) -> int:
        """Delete every user's state."""
        deleted = 0
        batch = []
        for key in self.client.scan_iter(match="u:*", count=1000):
            batch.append(key)
            if len(batch) >= 1000:
                deleted += self.client.delete(*batch)
                batch = []
        if batch:
            deleted += self.client.delete(*batch)
        return deleted

    def bootstrap(self, df: pd.DataFrame) -> int:
        """Load the state every user would have after the transactions of `df`.

        Same result as calling `process` on each row, but in bulk. Replaces existing state.
        """
        self.clear()
        df = df.sort_values("timestamp", kind="stable")
        df = df.assign(_ts=df["timestamp"].astype("datetime64[ns]").astype("int64") / 1e9)
        pipe = self.client.pipeline(transaction=False)
        n_users = 0
        for user_id, g in df.groupby("user_id", sort=False):
            recent, devices, cities, recipients, stats = user_keys(user_id)
            last_ts = float(g["_ts"].iloc[-1])
            tail = g[g["_ts"] >= last_ts - HISTORY_SECONDS]
            pipe.zadd(
                recent,
                {
                    recent_member(tx_id, amount): ts
                    for tx_id, amount, ts in zip(
                        tail["transaction_id"], tail["amount_tnd"], tail["_ts"], strict=True
                    )
                },
            )
            pipe.sadd(devices, *g["device_id"].unique())
            pipe.sadd(cities, *g["city"].unique())
            known_recipients = g.loc[g["type"] == P2P, "recipient_id"].unique()
            if len(known_recipients):
                pipe.sadd(recipients, *known_recipients)
            pipe.hset(
                stats,
                mapping={
                    "tx_count": len(g),
                    "amount_sum": repr(float(g["amount_tnd"].sum())),
                    "last_ts": repr(last_ts),
                },
            )
            pipe.expire(recent, RECENT_TTL_SECONDS)
            for key in (devices, cities, recipients, stats):
                pipe.expire(key, self.state_ttl_seconds)
            n_users += 1
            if n_users % 500 == 0:
                pipe.execute()
        pipe.execute()
        return n_users
