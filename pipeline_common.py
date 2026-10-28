"""
pipeline_common.py — shared pieces of the ingest / transform pipeline.

    - week math (one definition of "Week N" for every script)
    - the full PostgreSQL schema, created idempotently by ensure_schema()
    - connect(): psycopg2 connection using db_config settings (session time zone = UTC)
    - copy_upsert(): fast bulk load (COPY into a temp table, then INSERT ... ON CONFLICT)
    - logging setup

Lives in the project root next to db_config.py and is importable from any script after
`pip install -e .` (listed in pyproject.toml's py-modules).
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Sequence

import psycopg2

from db_config import PG_CONFIG

# =============================================================================
# WEEKS
# =============================================================================
# Week N = the N-th 7-day block since ranked launch, [start, start + 7 days) in UTC.
# Running any script "for" a date processes the most recent COMPLETED week before it:
#   2026-10-26 -> 88 days since launch -> 88 // 7 = 12 -> Week 12 = Oct 15 00:00 – Oct 22 00:00 UTC
RANKED_START_DATE = date(2026, 7, 30)


def week_number(target: date | None = None) -> int:
    target = target or date.today()
    days = (target - RANKED_START_DATE).days
    if days < 7:
        raise ValueError(f"{target} is before the end of Week 1 (ranked launched {RANKED_START_DATE}).")
    return days // 7


def week_bounds(week: int) -> tuple[datetime, datetime]:
    """Half-open [start, end) in UTC. Use `>= start AND < end` everywhere, never BETWEEN."""
    start = datetime.combine(RANKED_START_DATE + timedelta(weeks=week - 1), datetime.min.time(),
                             tzinfo=timezone.utc)
    return start, start + timedelta(weeks=1)


def period_name(week: int) -> str:
    return f"Week_{week}"


def parse_date(s: str | None) -> date | None:
    return datetime.strptime(s[:10], "%Y-%m-%d").date() if s else None


# =============================================================================
# SCHEMA  (single definition; every script calls ensure_schema before writing)
# =============================================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS heroes (
    hero_id     INT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    class_name  VARCHAR(100),
    hero_type   VARCHAR(50),
    complexity  INT
);

CREATE TABLE IF NOT EXISTS match_player (
    -- columns ordered widest-first so rows pack without alignment padding
    match_id            BIGINT NOT NULL,
    account_id          BIGINT NOT NULL,
    start_time          TIMESTAMPTZ NOT NULL,
    duration_s          INT,
    net_worth           INT,
    player_damage       INT,
    player_damage_taken INT,
    damage_absorbed     INT,
    damage_mitigated    INT,
    player_healing      INT,
    heal_lost           INT,
    max_health          INT,
    self_healing        INT,
    self_damage         INT,
    heal_prevented      INT,
    teammate_healing    INT,
    player_barriering   INT,
    absorption_provided INT,
    hero_id             SMALLINT,
    kills               SMALLINT,
    deaths              SMALLINT,
    assists             SMALLINT,
    last_hits           SMALLINT,
    denies              SMALLINT,
    assigned_lane       SMALLINT,
    average_badge_team0 SMALLINT,
    average_badge_team1 SMALLINT,
    mvp_rank            SMALLINT,
    won                 BOOLEAN,
    match_mode          VARCHAR(20),
    game_mode           VARCHAR(20),
    winning_team        VARCHAR(10),
    team                VARCHAR(10),
    PRIMARY KEY (match_id, account_id)
);
-- every weekly query filters on start_time; without this each one scans the whole table
CREATE INDEX IF NOT EXISTS match_player_start_time_idx ON match_player (start_time);

CREATE TABLE IF NOT EXISTS match_hero_roster (
    match_id            BIGINT PRIMARY KEY,
    start_time          TIMESTAMPTZ NOT NULL,
    match_duration_s    INT,
    winning_team        VARCHAR(20),
    team0               SMALLINT[],      -- hero ids, sorted, so equal comps compare equal
    team1               SMALLINT[],
    average_badge_team0 SMALLINT,
    average_badge_team1 SMALLINT
);
CREATE INDEX IF NOT EXISTS match_hero_roster_start_time_idx ON match_hero_roster (start_time);

CREATE TABLE IF NOT EXISTS patch_periods (
    period_id   SMALLINT PRIMARY KEY,
    period_name VARCHAR(100) UNIQUE,
    start_time  TIMESTAMPTZ NOT NULL,
    end_time    TIMESTAMPTZ,             -- NULL = still open (e.g. the current patch)
    description VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS items (
    id        BIGINT PRIMARY KEY,
    name      VARCHAR(100),
    tier      INT,
    type      VARCHAR(50),
    slot_type VARCHAR(50),
    cost      INT
);
"""


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


# =============================================================================
# CONNECTIONS / BULK LOAD
# =============================================================================
def connect():
    """psycopg2 connection from db_config (session time zone from PG_TIMEZONE, default UTC)."""
    return psycopg2.connect(**PG_CONFIG)


def _csv_value(v):
    if v is None:
        return None              # written as an unquoted empty field -> NULL
    if isinstance(v, (list, tuple)):
        return "{" + ",".join("NULL" if x is None else str(x) for x in v) + "}"   # Postgres array literal
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def copy_upsert(conn, table: str, columns: Sequence[str], rows: Iterable[Sequence],
                conflict: str = "DO NOTHING", key: Sequence[str] = ()) -> int:
    """Bulk-load rows: COPY into a temp table (fast path), then INSERT ... ON CONFLICT into
    `table`. ~2–3x faster than execute_values on a local DB (more over a network). Returns rows attempted."""
    buf = io.StringIO()
    w = csv.writer(buf)
    n = 0
    for r in rows:
        w.writerow([_csv_value(v) for v in r])
        n += 1
    if not n:
        return 0
    buf.seek(0)
    cols = ", ".join(columns)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE IF NOT EXISTS _stage_{table} (LIKE {table} INCLUDING DEFAULTS)"
                    " ON COMMIT DELETE ROWS")
        cur.copy_expert(f"COPY _stage_{table} ({cols}) FROM STDIN WITH (FORMAT csv)", buf)
        on_conflict = f"ON CONFLICT ({', '.join(key)}) {conflict}" if key else f"ON CONFLICT {conflict}"
        cur.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _stage_{table} {on_conflict}")
    conn.commit()
    return n


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("pipeline")
