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
    python ingest_dl_comm_api_parquet.py --ranks-only --weeks 1-13      # backfill only the rank columns

--ranks-only updates the rank columns of rows already in match_player and nothing else: it reads ~10 small
columns instead of ~20, narrows the files by the week's match_id range (from your database), and with
--weeks reads the file list once for every week.

Every run opens only the files that can hold the requested dates. The manifest files are partitioned by
match_id // 1,000,000, and the match ids around the dates come from your database, so no file has to be
opened just to find out what's in it.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime, timezone

import csv
import io
import warnings
import json
from datetime import timedelta
from pathlib import Path

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
RANK_COLUMNS = [("average_badge", "average_badge"), ("player_rank", "player_rank_initial_display_rank"),
                ("rank_progress_initial", "player_rank_initial_flat_progress"),
                ("rank_progress_final", "player_rank_final_flat_progress"),
                ("rank_change_desired", "player_rank_desired_progress_change"),
                ("shield_consumed", "player_rank_consumed_demotion_protection"),
                ("shields_initial", "player_rank_initial_demotion_protection_games"),
                ("win_streak_initial", "player_rank_initial_win_streak"),
                ("calibration_games", "player_rank_initial_calibration_games"),
                ("rank_outcome", "CAST(player_match_outcome AS VARCHAR)")]
RANK_UPDATE = "DO UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}" for c, _ in RANK_COLUMNS)
STAT_EXPRESSIONS = [(c, f'CAST(list_max(mp."stats.{c}") AS INTEGER)') for c in STAT_COLUMNS]
DUCKDB_THREADS = int(os.environ.get("DUCKDB_THREADS", "16"))   # remote reads wait on the network:
                                                                # more parallel requests than CPU cores helps


# --- SOURCE ------------------------------------------------------------------------
def parquet_urls(table: str, retries: int = 3) -> list[dict]:
    """Current parquet files for a table, from the API's manifest: [{url, partition, bytes}]."""
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
    return [{"url": f"{manifest['public_url']}/{f['key']}", "partition": f.get("partition"), "bytes": f.get("bytes")}
            for f in meta["files"] if meta.get("policy") == "snapshot" or f.get("generation", 0) == gen]


def source_entries(args) -> list[dict]:
    if args.source_files:                     # local / dev files: no partition info, all are read
        return [{"url": f, "partition": None, "bytes": None} for f in args.source_files]
    return parquet_urls("match_player")


# --- FILE SELECTION ----------------------------------------------------------------
# The manifest lists ~400 files, each in a partition = match_id // 1,000,000 (its "partition_expr").
# Match ids grow with time, so a date range only needs the partitions its match ids fall in: picking
# them from the manifest costs nothing, while letting DuckDB check every file's footer takes many minutes.
PARTITION_SIZE = 1_000_000
ID_MARGIN = 200_000           # match ids aren't perfectly in start_time order near a range's edges


def files_in_partitions(entries: list[dict], lo: int | None, hi: int | None) -> list[str]:
    """URLs of the files in partitions lo..hi (None = open-ended). Entries without a partition are kept."""
    out = [e["url"] for e in entries
           if e.get("partition") is None
           or ((lo is None or e["partition"] >= lo) and (hi is None or e["partition"] <= hi))]
    gb = sum(e.get("bytes") or 0 for e in entries if e["url"] in set(out)) / 1e9
    log.info("Files: partitions %s..%s -> %s of %s files (%.1f GB)", "first" if lo is None else lo,
             "last" if hi is None else hi, len(out), len(entries), gb)
    return out


# match_id ranges already known: checked against the API (weeks 1-3, 9-10) or remembered from earlier runs.
# Saved ranges live in assets/match_id_ranges.json; these seed it so no run depends on the API being free.
RANGES_PATH = Path(__file__).resolve().parents[2] / "assets" / "match_id_ranges.json"
KNOWN_RANGES = {
    "2026-07-30": (96_324_895, 97_719_121),     # week 1
    "2026-08-06": (97_719_125, 99_104_080),     # week 2
    "2026-08-13": (99_104_085, 100_493_667),    # week 3
    "2026-09-24": (107_379_203, 109_156_453),   # week 9
    "2026-10-01": (109_156_458, 112_660_851),   # week 10
}


def _range_key(start: datetime, end: datetime) -> str:
    return f"{start:%Y-%m-%dT%H:%M}/{end:%Y-%m-%dT%H:%M}"


def remembered_range(start: datetime, end: datetime) -> tuple[int, int] | None:
    try:
        saved = json.loads(RANGES_PATH.read_text())
    except (OSError, ValueError):
        saved = {}
    if _range_key(start, end) in saved:
        return tuple(saved[_range_key(start, end)])
    if end - start == timedelta(days=7) and f"{start:%Y-%m-%d}" in KNOWN_RANGES and start.hour == 0:
        return KNOWN_RANGES[f"{start:%Y-%m-%d}"]
    return None


def remember_range(start: datetime, end: datetime, rng: tuple[int, int]) -> None:
    if end > datetime.now(timezone.utc):        # an unfinished range can still grow
        return
    try:
        saved = json.loads(RANGES_PATH.read_text())
    except (OSError, ValueError):
        saved = {}
    saved[_range_key(start, end)] = list(rng)
    RANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RANGES_PATH.write_text(json.dumps(saved, indent=1, sort_keys=True))


def source_match_range(start: datetime, end: datetime) -> tuple[int, int] | None:
    """(min, max) match_id for a date range, asked from the API's SQL endpoint (DuckDB over the same parquet
    files; only match_player has start_time). A min/max over a whole week runs out of memory there, so it
    takes the lowest id in the range's first hour and the highest in its last hour: match ids grow with
    time, and ID_MARGIN covers the small jitter. None if the API can't answer (the database is used)."""
    try:
        from ingest_dl_comm_api_database import query_api

        def ts(t):
            return f"TIMESTAMPTZ '{t:%Y-%m-%d %H:%M:%S}+00'"
        sql = (f"SELECT (SELECT min(match_id) FROM match_player WHERE start_time >= {ts(start)} "
               f"AND start_time < {ts(start + timedelta(hours=1))}) AS lo, "
               f"(SELECT max(match_id) FROM match_player WHERE start_time >= {ts(end - timedelta(hours=1))} "
               f"AND start_time < {ts(end)}) AS hi")
        rows = query_api(sql, max_attempts=3)
        if isinstance(rows, dict):                       # {"data": [...]} shape, just in case
            rows = rows.get("data", [])
        lo, hi = rows[0]["lo"], rows[0]["hi"]
        if lo is None or hi is None:                     # an edge hour with no matches (e.g. a future end)
            return None
        lo, hi = int(lo), int(hi)
        return (lo, hi) if 0 < lo <= hi else None
    except Exception as e:
        log.warning("  couldn't get the match_id range from the API (%s); using your database", str(e)[:160])
        return None


def partitions_for_dates(conn, start: datetime, end: datetime) -> tuple[int | None, int | None]:
    """Partition bounds for a date range. First choice: the exact match ids from the API. Fallback: match
    ids in your database (the range's own rows, the last match before it, the first after it); a side
    with nothing to go on stays open-ended."""
    rng, where = remembered_range(start, end), "remembered"
    if not rng:
        rng, where = source_match_range(start, end), "from the API"
        if rng:
            remember_range(start, end, rng)
    if rng:
        log.info("  match_id %s..%s for these dates (%s)", f"{rng[0]:,}", f"{rng[1]:,}", where)
        return max(0, rng[0] - ID_MARGIN) // PARTITION_SIZE, (rng[1] + ID_MARGIN) // PARTITION_SIZE
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(match_id) FROM match_player WHERE start_time < %s", (start,))
        before = cur.fetchone()[0]
        cur.execute("SELECT MIN(match_id) FROM match_player WHERE start_time >= %s", (end,))
        after = cur.fetchone()[0]
        cur.execute("SELECT MIN(match_id), MAX(match_id) FROM match_player "
                    "WHERE start_time >= %s AND start_time < %s", (start, end))
        lo_in, hi_in = cur.fetchone()
    lo = before if before is not None else (lo_in - 3 * PARTITION_SIZE if lo_in is not None else None)
    hi = after if after is not None else None
    return (None if lo is None else max(0, lo - ID_MARGIN) // PARTITION_SIZE,
            None if hi is None else (hi + ID_MARGIN) // PARTITION_SIZE)


def duckdb_source(files: list[str], same_schema: bool = True) -> duckdb.DuckDBPyConnection:
    """same_schema: every file has the same columns (true for the manifest's files: one generation).
    union_by_name makes DuckDB read every file's footer, one after another, before the first row; without
    it only the first file's schema is read up front and the rest are opened in parallel during the scan."""
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")            # compare / return timestamps in UTC
    if any(f.startswith("http") for f in files):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET http_retries = 5; SET http_retry_wait_ms = 1000;")
        con.execute(f"SET threads = {DUCKDB_THREADS}")          # parallel range requests
        con.execute("SET enable_http_metadata_cache = true")    # read each file's footer once per run
    # views can't take query parameters, so the file list is inlined as a quoted SQL literal
    file_list = "[" + ", ".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
    con.execute("SET preserve_insertion_order = false")    # hand rows over as soon as any file has them
    union = "" if same_schema else ", union_by_name = true"
    con.execute(f"CREATE VIEW match_player AS SELECT * FROM read_parquet({file_list}{union})")
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
    rank = rank_exprs(cols)
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
    with Heartbeat("waiting for the first rows"):
        cur = RowStream(src, query, params)
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


class RowStream:
    """Query results in batches as DuckDB produces them (Arrow record batches), so rows reach PostgreSQL
    and the log while the download is still going, instead of after the whole query has finished."""

    def __init__(self, src, query: str, params: list):
        res = src.execute(query, params)
        warnings.filterwarnings("ignore", message="Arrow output on a DB-API result")   # harmless extra copy
        # DuckDB 2.x renamed fetch_record_batch to to_arrow_reader; support both
        self.reader = (res.to_arrow_reader(BATCH_SIZE) if hasattr(res, "to_arrow_reader")
                       else res.fetch_record_batch(BATCH_SIZE))

    def fetchmany(self, _n: int | None = None) -> list[tuple]:
        try:
            batch = self.reader.read_next_batch()
        except StopIteration:
            return []
        return list(zip(*(col.to_pylist() for col in batch.columns)))


def rank_exprs(cols: dict) -> list[tuple[str, str]]:
    """(column, DuckDB expression) for the rank columns; NULL for any the files don't have."""
    out = []
    for c, src_col in RANK_COLUMNS:
        name = src_col.split("(")[-1].split(" ")[0]          # the source column inside a CAST(...)
        out.append((c, src_col.replace(name, f"mp.{name}") if name in cols else "NULL"))
    return out


class Heartbeat:
    """Logs every 30 s while a slow step (remote parquet scan) hasn't produced anything yet."""

    def __init__(self, what: str, every: float = 30.0):
        self.what, self.every, self.stop = what, every, threading.Event()

    def __enter__(self):
        t0 = time.time()

        def beat():
            while not self.stop.wait(self.every):
                log.info("  still %s... %.0fs", self.what, time.time() - t0)
        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()


def week_match_range(conn, start: datetime, end: datetime) -> tuple[int, int, int] | None:
    """(min match_id, max match_id, rows) already in match_player for this range."""
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(match_id), MAX(match_id), COUNT(*) FROM match_player "
                    "WHERE start_time >= %s AND start_time < %s", (start, end))
        lo, hi, n = cur.fetchone()
    return (lo, hi, n) if n else None


def backfill_ranks(src, conn, start: datetime, end: datetime) -> int:
    """Fill the rank columns of rows already in match_player for [start, end); nothing else is touched.
    Returns rows updated."""
    rng = week_match_range(conn, start, end)
    if not rng:
        log.warning("  no rows in match_player for this range yet: run the normal ingest for it first")
        return 0
    lo, hi, n_have = rng
    cols = {r[0]: r[1] for r in src.execute("DESCRIBE match_player").fetchall()}
    rank = rank_exprs(cols)
    missing = [c for c, e in rank if e == "NULL"]
    if missing:
        log.warning("  these files don't have: %s (left NULL)", ", ".join(missing))
    col_type = cols["start_time"]
    names = ["match_id", "account_id"] + [c for c, _ in rank]
    # match_id bounds skip whole files / row groups fast (files are split by match_id); start_time keeps it exact
    query = f"""
        SELECT mp.match_id, mp.account_id, {", ".join(e for _, e in rank)}
        FROM match_player mp
        WHERE mp.match_id BETWEEN ? AND ?
          AND mp.start_time >= CAST(? AS {col_type}) AND mp.start_time < CAST(? AS {col_type})
          AND (CAST(mp.match_mode AS VARCHAR) = 'Ranked' OR TRY_CAST(mp.match_mode AS INTEGER) = 4)
    """
    params = [lo, hi] + ([start, end] if "ZONE" in col_type.upper()
                         else [start.replace(tzinfo=None), end.replace(tzinfo=None)])
    log.info("  %s rows to fill, match_id %s..%s", f"{n_have:,}", lo, hi)
    t0 = time.time()
    with Heartbeat("waiting for the first rows"):
        cur = RowStream(src, query, params)
        first = cur.fetchmany(BATCH_SIZE)
    total = 0
    rows = first
    sets = ", ".join(f"{c} = s.{c}" for c in names[2:])
    with conn.cursor() as pc:
        pc.execute("CREATE TEMP TABLE IF NOT EXISTS _stage_ranks AS SELECT " + ", ".join(names)
                   + " FROM match_player LIMIT 0")
    conn.commit()
    while rows:
        buf = io.StringIO()
        w = csv.writer(buf)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
        buf.seek(0)
        with conn.cursor() as pc:
            pc.execute("TRUNCATE _stage_ranks")
            pc.copy_expert(f"COPY _stage_ranks ({', '.join(names)}) FROM STDIN WITH (FORMAT csv)", buf)
            # the dump stores some rows twice (identical): DISTINCT ON keeps UPDATE to one source row per target
            pc.execute(f"""UPDATE match_player m SET {sets}
                           FROM (SELECT DISTINCT ON (match_id, account_id) * FROM _stage_ranks) s
                           WHERE m.match_id = s.match_id AND m.account_id = s.account_id""")
            total += pc.rowcount
        conn.commit()
        log.info("  updated %s / %s rows (%.0fs)", f"{total:,}", f"{n_have:,}", time.time() - t0)
        rows = cur.fetchmany(BATCH_SIZE)
    return total


def parse_weeks(spec: str) -> list[int]:
    """"1-13" or "1,3,5-7" -> [1, ..., 13]."""
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


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
    g.add_argument("--weeks", help="several weeks in one run, e.g. 1-13 (with --ranks-only)")
    ap.add_argument("--source-files", nargs="+", help="local/remote parquet files instead of the manifest")
    ap.add_argument("--with-stats", action="store_true",
                    help="also load the 13 damage/healing stats (~10x more download)")
    ap.add_argument("--ranks-only", action="store_true",
                    help="only fill the rank columns of rows already loaded (fast backfill)")
    args = ap.parse_args()

    if args.ranks_only:
        return ranks_only(args)
    if args.weeks:
        raise SystemExit("--weeks is for --ranks-only; for full ingests run one --week at a time")
    what, start, end, week = resolve_range(args)
    log.info("%s: %s -> %s (UTC, end exclusive)", what, f"{start:%Y-%m-%d %H:%M}", f"{end:%Y-%m-%d %H:%M}")

    conn = connect()
    try:
        ensure_schema(conn)                      # before anything writes (fresh-DB safe)
        if week is not None:
            register_week(conn, week, start, end)
        files = files_in_partitions(source_entries(args), *partitions_for_dates(conn, start, end))
        if not files:
            log.error("No source file covers %s yet", what)
            return 1
        with Heartbeat("reading file footers"):
            src = duckdb_source(files, same_schema=not args.source_files)
            src.execute("DESCRIBE match_player").fetchall()
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


def ranks_only(args) -> int:
    ranges = []
    if args.weeks:
        ranges = [(f"Week {w}", *week_bounds(w)) for w in parse_weeks(args.weeks)]
    else:
        what, start, end, _ = resolve_range(args)
        ranges = [(what, start, end)]
    conn = connect()
    try:
        ensure_schema(conn)                      # adds the rank columns if this DB predates them
        entries = source_entries(args)
        done = 0
        for what, start, end in ranges:
            log.info("%s: %s -> %s, rank columns only", what, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")
            rng = week_match_range(conn, start, end)
            if not rng:
                log.warning("  no rows in match_player for this range yet: run the normal ingest for it first")
                continue
            files = files_in_partitions(entries, max(0, rng[0] - ID_MARGIN) // PARTITION_SIZE,
                                        (rng[1] + ID_MARGIN) // PARTITION_SIZE)
            with Heartbeat("reading file footers"):
                src = duckdb_source(files, same_schema=not args.source_files)
                src.execute("DESCRIBE match_player").fetchall()
            n = backfill_ranks(src, conn, start, end)
            src.close()
            log.info("%s: %s rows updated", what, f"{n:,}")
            done += n
        log.info("Done: %s rows updated across %s range(s)", f"{done:,}", len(ranges))
        return 0 if done else 1
    except Exception:
        log.exception("Rank backfill failed")
        conn.rollback()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())