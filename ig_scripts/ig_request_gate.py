"""Coordinate IG REST calls across bot + mobile API to avoid rate-limit collisions."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GATE_FILE = PROJECT_ROOT / "runtime" / "ig_request_gate.json"
LOCK_FILE = PROJECT_ROOT / "runtime" / "ig_request_gate.lock"

# Stagger consumers on different seconds each minute (UTC).
CONSUMER_SECONDS: dict[str, frozenset[int]] = {
    "bot_fetch": frozenset(range(5, 8)),        # :05–:07 gold IG fetch → cache
    "bot_trade": frozenset(range(5, 8)),        # :05–:07 gold score / orders
    "bot_oil": frozenset(range(8, 11)),         # :08–:10 oil fetch + trades
    "bot_sync": frozenset(range(10, 12)),       # :10–:11 gold position sync
    "bot_oil_sync": frozenset(range(14, 16)),   # :14–:15 oil position sync
    "bot_db": frozenset(range(32, 38)),         # :32–:37 db price backfill
    "mobile_api": frozenset(range(45, 55)),    # :45–:54 journal IG reconcile
}

MIN_GAP_SEC = float(os.environ.get("IG_REQUEST_MIN_GAP_SEC", "2.0"))
MAX_WAIT_SEC = float(os.environ.get("IG_REQUEST_MAX_WAIT_SEC", "58.0"))


def ig_second_slot_open(consumer: str, now: datetime | None = None) -> bool:
    """True when this consumer is allowed to make IG calls right now."""
    now = now or datetime.now(timezone.utc)
    allowed = CONSUMER_SECONDS.get(consumer)
    if allowed is None:
        return True
    return now.second in allowed


def acquire_ig_request_slot(consumer: str | None = None) -> None:
    """Block until consumer's second-window and global min-gap are both satisfied."""
    name = (consumer or os.environ.get("IG_REQUEST_CONSUMER") or "default").strip()
    allowed = CONSUMER_SECONDS.get(name)
    deadline = time.monotonic() + MAX_WAIT_SEC
    while time.monotonic() < deadline:
        now = datetime.now(timezone.utc)
        if allowed is not None and now.second not in allowed:
            time.sleep(0.2)
            continue
        if _reserve_global_gap(name):
            return
        time.sleep(0.2)
    logger.warning("IG request gate timeout consumer=%s", name)


def _reserve_global_gap(consumer: str) -> bool:
    GATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp, fcntl.LOCK_EX)
        now = time.time()
        last_ts = 0.0
        if GATE_FILE.exists():
            try:
                data = json.loads(GATE_FILE.read_text(encoding="utf-8"))
                last_ts = float(data.get("last_ts") or 0.0)
            except Exception:
                last_ts = 0.0
        if now - last_ts < MIN_GAP_SEC:
            return False
        GATE_FILE.write_text(
            json.dumps(
                {
                    "last_ts": now,
                    "consumer": consumer,
                    "second": datetime.now(timezone.utc).second,
                }
            ),
            encoding="utf-8",
        )
        return True
