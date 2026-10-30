"""
ingest_dl_comm_api_parquet.py — weekly ranked-match ingest (main pipeline entry point).

    1. register the week in patch_periods
    2. stream that week's RANKED player rows from the Deadlock API's public parquet files
       (DuckDB reads them over HTTP, pushing the date filter down so only matching row
       groups are downloaded) into match_player, in COPY batches
       plus each match's hero bans into match_bans, when the API recorded them
    3. rebuild match_hero_roster for that week only, from complete matches

Re-running a week is safe: inserts skip rows that already exist (primary keys), and the
roster rebuild for the week is a replace. The API's dump stores many recent rows twice (base +
residual files, identical on every column used here); the primary key keeps one copy.

Usage:
    python ingest_dl_comm_api_parquet.py                      # last completed week
    python ingest_dl_comm_api_parquet.py --date 2026-10-26    # week containing the day before
    python ingest_dl_comm_api_parquet.py --week 12
    python ingest_dl_comm_api_parquet.py --week 12 --with-stats         # also damage/healing stats
    python ingest_dl_comm_api_parquet.py --period Release_Nurse_Harrow_d1-7   # see register_period.py
    python ingest_dl_comm_api_parquet.py --week 12 --source-files ./sample/*.parquet   # offline/dev
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import duckdb
import requests

from pipeline_common import (connect, copy_upsert, ensure_schema, get_period, parse_date, period_name,
                             setup_logging, week_bounds, week_number)

MANIFEST_URL = "https://data.deadlock-api.com/v1/manifest.json"
BATCH_SIZE = 20_000
log = setup_logging()

# (postgres column, DuckDB expression). Order here = insert order.
# The 13 end-of-match stats (damage, healing, ...) exist in the API only as per-minute arrays
# ("stats.*"), whose max is the end-of-match value. Those arrays are ~90% of the bytes this script
# would download and no current analysis uses them, so they are only read with --with-stats;
# otherwise they stay NULL. Missing stats are NULL, never 0 (0 would drag averages down).
STAT_COLUMNS = ["heal_lost", "max_health", "self_healing", "self_damage", "player_damage",
                "heal_prevented", "player_healing", "damage_absorbed", "damage_mitigated",
                "teammate_healing", "player_barriering", "absorption_provided", "player_damage_taken"]
COLUMNS: list[tuple[str, str]] = [
    ("match_id", "mp.match_id"),
    ("account_id", "mp.account_id"),
    ("start_time", "mp.start_time"),
    ("duration_s", "mp.duration_s"),
    ("match_mode", "CAST(mp.match_mode AS VARCHAR)"),
    ("game_mode", "CAST(mp.game_mode AS VARCHAR)"),
    ("winning_team", "CAST(mp.winning_team AS VARCHAR)"),
    ("team", "CAST(mp.team AS VARCHAR)"),
    ("won", "mp.won"),
    ("hero_id", "mp.hero_id"),
    ("kills", "mp.kills"),
    ("deaths", "mp.deaths"),
    ("assists", "mp.assists"),
    ("net_worth", "mp.net_worth"),
    ("last_hits", "mp.last_hits"),
    ("denies", "mp.denies"),
    ("assigned_lane", "mp.assigned_lane"),
    ("average_badge_team0", "mp.average_badge_team0"),
    ("average_badge_team1", "mp.average_badge_team1"),
    ("mvp_rank", "mp.mvp_rank"),
]
# rank columns: optional per file set (older dumps may lack them); re-ingesting a week fills them in
RANK_COLUMNS = [("average_badge", "average_badge"), ("player_rank", "player_rank_initial_display_rank")]
RANK_UPDATE = "DO UPDATE SET average_badge = EXCLUDED.average_badge, player_rank = EXCLUDED.player_rank"
STAT_EXPRESSIONS = [(c, f'CAST(list_max(mp."stats.{c}") AS INTEGER)') for c in STAT_COLUMNS]
DUCKDB_THREADS = int(os.environ.get("DUCKDB_THREADS", "16"))   # remote reads wait on the network:
                                                                # more parallel requests than CPU cores helps


# --- SOURCE ------------------------------------------------------------------------
def parquet_urls(table: str, retries: int = 3) -> list[str]:
    """Current parquet files for a table, from the API's manifest."""
    for attempt in range(1, retries + 1):
        try:
            res = requests.get(MANIFEST_URL, timeout=30)
            res.raise_for_status()
            manifest = res.json()
            break
        except requests.RequestException as e:
            if attempt == retries:
                raise
            log.warning("manifest fetch failed (%s), retrying in %ss", e, 5 * attempt)
            time.sleep(5 * attempt)
    meta = manifest["tables"][table]
    gen = meta.get("generation", 0)
    return [f"{manifest['public_url']}/{f['key']}" for f in meta["files"]
            if meta.get("policy") == "snapshot" or f.get("generation", 0) == gen]


def duckdb_source(files: list[str]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")            # compare / return timestamps in UTC
    if any(f.startswith("http") for f in files):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET http_retries = 5; SET http_retry_wait_ms = 1000;")
        con.execute(f"SET threads = {DUCKDB_THREADS}")          # parallel range requests
        con.execute("SET enable_http_metadata_cache = true")    # read each file's footer once per run
    # views can't take query parameters, so the file list is inlined as a quoted SQL literal
    file_list = "[" + ", ".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
    con.execute(f"CREATE VIEW match_player AS SELECT * FROM read_parquet({file_list}, union_by_name = true)")
    return con


# --- LOAD --------------------------------------------------------------------------
def register_week(conn, week: int, start: datetime, end: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO patch_periods (period_id, period_name, start_time, end_time, description)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (period_id) DO UPDATE SET period_name = EXCLUDED.period_name,
                start_time = EXCLUDED.start_time, end_time = EXCLUDED.end_time,
                description = EXCLUDED.description
        """, (week, period_name(week), start, end, f"Ranked Week {week}"))
    conn.commit()


def load_match_players(src, conn, start: datetime, end: datetime, with_stats: bool = False) -> int:
    # Cast the PARAMETERS to the column's own type so DuckDB never has to cast the column itself:
    # a cast on the column disables row-group skipping (the filter can't use the files' min/max
    # stats). The API's files currently store TIMESTAMP WITH TIME ZONE; older dumps used TIMESTAMP.
    cols = {r[0]: r[1] for r in src.execute("DESCRIBE match_player").fetchall()}
    col_type = cols["start_time"]
    rank = [(c, f"mp.{src_col}" if src_col in cols else "NULL") for c, src_col in RANK_COLUMNS]
    columns = COLUMNS + rank + (STAT_EXPRESSIONS if with_stats else [])
    has_bans = "banned_hero_ids" in cols
    log.info("Damage/healing stats: %s  |  bans: %s",
             "included (--with-stats)" if with_stats else "skipped (NULL)",
             "loading" if has_bans else "not in these files")
    select = ",\n            ".join(expr for _, expr in columns)
    # bans ride along as one extra trailing column (empty list -> NULL = not recorded)
    bans_expr = ("CASE WHEN len(mp.banned_hero_ids) > 0 THEN CAST(mp.banned_hero_ids AS SMALLINT[]) END"
                 if has_bans else "NULL")
    query = f"""
        SELECT {select},
            {bans_expr} AS _bans
        FROM match_player mp
        WHERE mp.start_time >= CAST(? AS {col_type}) AND mp.start_time < CAST(? AS {col_type})
          AND (CAST(mp.match_mode AS VARCHAR) = 'Ranked' OR TRY_CAST(mp.match_mode AS INTEGER) = 4)
    """
    params = [start, end] if "ZONE" in col_type.upper() else [start.replace(tzinfo=None), end.replace(tzinfo=None)]
    cur = src.execute(query, params)
    names = [c for c, _ in columns]
    i_match, i_start = names.index("match_id"), names.index("start_time")
    total = n_ban_matches = 0
    t0 = time.time()
    fetch_s = load_s = 0.0
    while True:
        t = time.time()
        rows = cur.fetchmany(BATCH_SIZE)          # time spent here = downloading + decoding parquet
        fetch_s += time.time() - t
        if not rows:
            break
        t = time.time()
        bans = {r[i_match]: (r[i_match], r[i_start], r[-1]) for r in rows if r[-1]}   # one row per match
        # existing rows only get their rank columns refreshed (backfills weeks ingested before they existed)
        total += copy_upsert(conn, "match_player", names, (r[:-1] for r in rows), key=["match_id", "account_id"],
                             conflict=RANK_UPDATE)
        if bans:
            n_ban_matches += copy_upsert(conn, "match_bans", ["match_id", "start_time", "banned_hero_ids"],
                                         bans.values(), key=["match_id"])
        load_s += time.time() - t                 # time spent here = writing into PostgreSQL
        log.info("  match_player: %s rows (%.0f rows/s)  |  source %.0fs, postgres %.0fs so far",
                 f"{total:,}", total / max(time.time() - t0, 1e-6), fetch_s, load_s)
    log.info("Source (download/decode) %.0fs = %.0f%%, PostgreSQL writes %.0fs = %.0f%%",
             fetch_s, 100 * fetch_s / max(fetch_s + load_s, 1e-6), load_s, 100 * load_s / max(fetch_s + load_s, 1e-6))
    if has_bans:
        log.info("Bans recorded for %s matches (a match can span two batches, so this may count a few twice)",
                 f"{n_ban_matches:,}")
    return total


def rebuild_rosters(conn, start: datetime, end: datetime) -> tuple[int, int]:
    """Replace this week's match_hero_roster rows. Only complete matches are kept:
    12 players, 6 per team, exactly one winning side. Returns (kept, skipped)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM match_hero_roster WHERE start_time >= %s AND start_time < %s", (start, end))
        cur.execute("""
            WITH per_match AS (
                SELECT
                    match_id,
                    MIN(start_time)  AS start_time,
                    MAX(duration_s)  AS match_duration_s,
                    MAX(team) FILTER (WHERE won)                                         AS winning_team,
                    ARRAY_AGG(hero_id ORDER BY hero_id) FILTER (WHERE LOWER(team) = 'team0') AS team0,
                    ARRAY_AGG(hero_id ORDER BY hero_id) FILTER (WHERE LOWER(team) = 'team1') AS team1,
                    MAX(average_badge_team0) AS average_badge_team0,
                    MAX(average_badge_team1) AS average_badge_team1,
                    MAX(average_badge)       AS average_badge,
                    COUNT(*)                                       AS n_players,
                    COUNT(DISTINCT team) FILTER (WHERE won)        AS n_winning_teams
                FROM match_player
                WHERE start_time >= %s AND start_time < %s
                GROUP BY match_id
            ), inserted AS (
                INSERT INTO match_hero_roster (match_id, start_time, match_duration_s, winning_team,
                                               team0, team1, average_badge_team0, average_badge_team1, average_badge)
                SELECT match_id, start_time, match_duration_s, winning_team,
                       team0, team1, average_badge_team0, average_badge_team1, average_badge
                FROM per_match
                WHERE n_players = 12 AND n_winning_teams = 1
                  AND cardinality(team0) = 6 AND cardinality(team1) = 6
                RETURNING 1
            )
            SELECT (SELECT COUNT(*) FROM inserted), (SELECT COUNT(*) FROM per_match)
        """, (start, end))
        kept, total = cur.fetchone()
    conn.commit()
    return kept, total - kept


# --- MAIN --------------------------------------------------------------------------
def resolve_range(args) -> tuple[str, datetime, datetime, int | None]:
    """(what, start, end, week): a week by number/date, or a registered period plus the period it
    is compared against (both are loaded, so the comparison is ready too)."""
    if not args.period:
        week = args.week or week_number(parse_date(args.date))
        start, end = week_bounds(week)
        return f"Week {week}", start, end, week
    conn = connect()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            p = get_period(cur, args.period)
            start, end = p["start"], p["end"] or datetime.now(timezone.utc)
            if p["compare_to"]:
                start = min(start, get_period(cur, p["compare_to"])["start"])
    finally:
        conn.close()
    return f"{args.period} (+ its comparison period)", start, end, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest one ranked week (or a registered period) from the Deadlock API")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--date", "--start-date", dest="date",
                   help="YYYY-MM-DD; ingests the last week completed before this date (default today)")
    g.add_argument("--week", type=int, help="week number to ingest")
    g.add_argument("--period", help="a period from register_period.py (also loads its comparison period)")
    ap.add_argument("--source-files", nargs="+", help="local/remote parquet files instead of the manifest")
    ap.add_argument("--with-stats", action="store_true",
                    help="also load the 13 damage/healing stats (~10x more download)")
    args = ap.parse_args()

    what, start, end, week = resolve_range(args)
    log.info("%s: %s -> %s (UTC, end exclusive)", what, f"{start:%Y-%m-%d %H:%M}", f"{end:%Y-%m-%d %H:%M}")

    conn = connect()
    try:
        ensure_schema(conn)                      # before anything writes (fresh-DB safe)
        if week is not None:
            register_week(conn, week, start, end)
        files = args.source_files or parquet_urls("match_player")
        log.info("Reading %s parquet file(s)", len(files))
        src = duckdb_source(files)
        n = load_match_players(src, conn, start, end, args.with_stats)
        if n == 0:
            log.error("No ranked rows found for %s — source not published yet, or wrong dates?", what)
            return 1
        kept, skipped = rebuild_rosters(conn, start, end)
        log.info("Done: %s player rows, %s matches (%s incomplete matches skipped from rosters)",
                 f"{n:,}", f"{kept:,}", f"{skipped:,}")
        return 0
    except Exception:
        log.exception("Ingest failed for %s", what)   # full traceback, non-zero exit
        conn.rollback()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())