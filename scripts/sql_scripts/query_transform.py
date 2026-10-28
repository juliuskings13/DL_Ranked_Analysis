"""
query_transform.py — weekly SQL transforms -> the CSV/Parquet files the chart scripts read.

How it works
    1. look the period up in patch_periods (written by the ingest)
    2. build two temp tables on one connection, once:
         wk_matches  complete matches of the period (12 players, 6 per side, one winner)
         wk_players  their 12 player rows, joined to heroes
       plus wk_type_counts (heroes of each archetype per team, zeros included)
    3. run every export against those small tables, instead of re-filtering the full
       match_player table 15 times
    4. write <project>/week<N>_analysis/<name>.csv (+ period_meta.json)

Using only complete matches keeps every count consistent (12 hero rows per match), which the
charts rely on when they turn hero appearances into match counts.

Usage:
    python query_transform.py                       # last completed week
    python query_transform.py --date 2026-10-26
    python query_transform.py --week 12
    python query_transform.py --period Patch_1_2    # any named row in patch_periods
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline_common import connect, parse_date, period_name, setup_logging, week_number

PROJECT_ROOT = Path(__file__).resolve().parents[2]       # scripts/sql_scripts/ -> project root
MIN_DUO_GAMES = 1000          # duos below this are not exported (the chart also filters)
TOP_COMPS = 20
log = setup_logging()

# 5-minute bracket label from a duration in seconds; NULL under 20 minutes (same buckets as before)
BRACKET_SQL = """CASE WHEN {d} < 1200 THEN NULL
                      WHEN {d} >= 3600 THEN '60m+'
                      ELSE (20 + 5 * (({d} / 60 - 20) / 5))::text || '-'
                           || (25 + 5 * (({d} / 60 - 20) / 5))::text || 'm' END"""

# =============================================================================
# STAGING (temp tables, built once per run)
# =============================================================================
STAGING = [
    """CREATE TEMP TABLE wk_matches AS
       SELECT match_id, start_time, match_duration_s AS duration_s, winning_team, team0, team1
       FROM match_hero_roster
       WHERE start_time >= %(start)s AND (%(end)s::timestamptz IS NULL OR start_time < %(end)s)
         AND cardinality(team0) = 6 AND cardinality(team1) = 6""",
    """CREATE TEMP TABLE wk_players AS
       SELECT mp.match_id, mp.team, mp.won, mp.hero_id, m.duration_s,
              h.name AS hero_name, LOWER(h.hero_type) AS hero_type
       FROM match_player mp
       JOIN wk_matches m USING (match_id)
       JOIN heroes h ON h.hero_id = mp.hero_id""",
    "CREATE INDEX ON wk_players (match_id, team)",
    # heroes of each archetype per team, INCLUDING zero (a team with no mystics has a row with n = 0)
    """CREATE TEMP TABLE wk_type_counts AS
       WITH teams AS (SELECT match_id, team, BOOL_OR(won) AS won FROM wk_players GROUP BY 1, 2),
            types AS (SELECT DISTINCT hero_type FROM wk_players WHERE hero_type IS NOT NULL)
       SELECT t.match_id, t.team, t.won, ty.hero_type, COUNT(p.hero_id) AS n
       FROM teams t CROSS JOIN types ty
       LEFT JOIN wk_players p ON p.match_id = t.match_id AND p.team = t.team AND p.hero_type = ty.hero_type
       GROUP BY 1, 2, 3, 4""",
    "ANALYZE wk_matches", "ANALYZE wk_players", "ANALYZE wk_type_counts",
]

# =============================================================================
# EXPORTS: name -> (SQL, formats). Column names are the contract with the chart scripts.
# =============================================================================
EXPORTS: dict[str, tuple[str, tuple[str, ...]]] = {
    # pick_rate_winrate_scatter, hero_wr_total_matches_split_column, balance_wr_distribution, ...
    "hero_winrates": ("""
        SELECT hero_name, hero_type, COUNT(*) AS total_matches, SUM(won::int) AS wins,
               ROUND(100.0 * AVG(won::int), 2) AS winrate
        FROM wk_players GROUP BY hero_name, hero_type ORDER BY winrate DESC""", ("csv",)),

    # hero_duo_synergy_table: ALL duos with enough games (the chart ranks by lift, both directions)
    "hero_duo_synergies": ("""
        SELECT a.hero_name AS hero_name_a, b.hero_name AS hero_name_b,
               COUNT(*) AS matches_together, SUM(a.won::int) AS wins,
               ROUND(100.0 * AVG(a.won::int), 2) AS duo_winrate
        FROM wk_players a
        JOIN wk_players b ON b.match_id = a.match_id AND b.team = a.team AND a.hero_id < b.hero_id
        GROUP BY a.hero_name, b.hero_name
        HAVING COUNT(*) >= %(min_duo)s
        ORDER BY duo_winrate DESC""", ("csv",)),

    # archetype_stacking: one row per match per archetype, from the side with MORE of it.
    # Ties (diff 0) are counted once, from the lexically first team — the old query counted them twice.
    "hero_type_diff_winrate": ("""
        SELECT a.hero_type, a.n - b.n AS type_differential, COUNT(*) AS match_frequency,
               ROUND(100.0 * AVG(a.won::int), 2) AS winrate_of_surplus_team
        FROM wk_type_counts a
        JOIN wk_type_counts b ON b.match_id = a.match_id AND b.hero_type = a.hero_type AND b.team <> a.team
        WHERE a.n > b.n OR (a.n = b.n AND a.team < b.team)
        GROUP BY a.hero_type, a.n - b.n ORDER BY 1, 2""", ("csv",)),

    # matchup_bubble_chart: every (your count, enemy count) pair from BOTH sides of each match;
    # split per archetype into <type>_comp_wr.csv below
    "_type_matchups": ("""
        SELECT a.hero_type, a.n AS t0_count, b.n AS t1_count, COUNT(*) AS total_matchups,
               ROUND(100.0 * AVG(a.won::int), 2) AS t0_winrate
        FROM wk_type_counts a
        JOIN wk_type_counts b ON b.match_id = a.match_id AND b.hero_type = a.hero_type AND b.team <> a.team
        GROUP BY 1, 2, 3 ORDER BY 1, total_matchups DESC""", ()),

    # match_duration (fallback buckets) + hero_strength_and_scaling column totals
    "avg_duration": (f"""
        SELECT {BRACKET_SQL.format(d='duration_s')} AS duration_bracket, COUNT(*) AS total_games
        FROM wk_matches WHERE duration_s >= 1200
        GROUP BY 1 ORDER BY MIN(duration_s)""", ("csv",)),

    # match_duration: exact 1-minute histogram (quantiles, IQR, median split)
    "match_duration_minutes": ("""
        SELECT duration_s / 60 AS duration_min, COUNT(*) AS games
        FROM wk_matches GROUP BY 1 ORDER BY 1""", ("csv",)),

    # hero_strength_and_scaling
    "hero_duration_scaling": (f"""
        SELECT hero_name, {BRACKET_SQL.format(d='duration_s')} AS duration_bracket,
               COUNT(*) AS total_games, SUM(won::int) AS wins,
               ROUND(100.0 * AVG(won::int), 2) AS winrate
        FROM wk_players WHERE duration_s >= 1200
        GROUP BY hero_name, 2 ORDER BY hero_name, MIN(duration_s)""", ("parquet",)),

    # comp_uniqueness
    "unique_comp_freq_match": ("""
        WITH comps AS (SELECT team0 AS comp FROM wk_matches UNION ALL SELECT team1 FROM wk_matches),
        freq AS (SELECT comp, COUNT(*) AS freq FROM comps GROUP BY comp),
        bucketed AS (
            SELECT freq,
                   CASE WHEN freq = 1 THEN 1 WHEN freq < 5 THEN 2 WHEN freq < 10 THEN 3 WHEN freq < 20 THEN 4
                        WHEN freq < 30 THEN 5 WHEN freq < 40 THEN 6 ELSE 7 END AS sort_order
            FROM freq),
        agg AS (SELECT sort_order, COUNT(*) AS unique_team_comps, SUM(freq) AS total_games_played
                FROM bucketed GROUP BY sort_order)
        SELECT (ARRAY['single-occurrence', '2 - 4 occurrences', '5 - 9 occurrences', '10 - 19 occurrences',
                      '20 - 29 occurrences', '30 - 39 occurrences', '40+ occurrences'])[sort_order]
                   AS frequency_bucket,
               unique_team_comps, total_games_played,
               SUM(unique_team_comps) OVER (ORDER BY sort_order DESC) AS cumulative_comps_n_plus,
               SUM(total_games_played) OVER (ORDER BY sort_order DESC) AS cumulative_games_n_plus
        FROM agg ORDER BY sort_order""", ("csv",)),

    # most_freq_comps
    "top_20_freq_comps": ("""
        WITH sides AS (
            SELECT team0 AS comp, LOWER(winning_team) = 'team0' AS won FROM wk_matches
            UNION ALL
            SELECT team1, LOWER(winning_team) = 'team1' FROM wk_matches),
        top AS (SELECT comp, COUNT(*) AS total_matches, ROUND(100.0 * AVG(won::int), 2) AS winrate
                FROM sides GROUP BY comp ORDER BY total_matches DESC, winrate DESC LIMIT %(top_comps)s)
        SELECT (SELECT STRING_AGG(h.name, ', ' ORDER BY h.name)
                FROM UNNEST(top.comp) AS u(hero_id) JOIN heroes h ON h.hero_id = u.hero_id) AS comp_names,
               total_matches, winrate
        FROM top ORDER BY total_matches DESC, winrate DESC""", ("csv",)),
}


# =============================================================================
# RUN
# =============================================================================
def resolve_period(cur, name: str):
    cur.execute("SELECT start_time, end_time FROM patch_periods WHERE period_name = %s", (name,))
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"'{name}' is not in patch_periods — run the ingest for that week first.")
    return row


def query_df(conn, sql: str, params: dict) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        # coerce_float: NUMERIC results (e.g. ROUND) arrive as Decimal; parquet would keep them as
        # decimals and the charts' float math would fail
        return pd.DataFrame.from_records(cur.fetchall(), columns=[c.name for c in cur.description],
                                         coerce_float=True)


def write(df: pd.DataFrame, out_dir: Path, name: str, formats: tuple[str, ...]) -> None:
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        df.to_csv(path, index=False) if fmt == "csv" else df.to_parquet(path, index=False)
        log.info("  wrote %-34s %s rows", path.name, f"{len(df):,}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the weekly SQL transforms")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--date", "--start-date", dest="date", help="YYYY-MM-DD (default today): uses the last completed week")
    g.add_argument("--week", type=int)
    g.add_argument("--period", help="any period_name in patch_periods, e.g. a patch window")
    ap.add_argument("--out-dir", help="override the output folder")
    args = ap.parse_args()

    if args.period:
        name, folder = args.period, args.period.lower()
    else:
        week = args.week or week_number(parse_date(args.date))
        name, folder = period_name(week), f"week{week}_analysis"
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect()
    failed = []
    try:
        with conn.cursor() as cur:
            start, end = resolve_period(cur, name)
            log.info("%s: %s -> %s  =>  %s", name, start, end or "open", out_dir)
            params = {"start": start, "end": end, "min_duo": MIN_DUO_GAMES, "top_comps": TOP_COMPS}
            t0 = time.time()
            for sql in STAGING:
                cur.execute(sql, params)
            cur.execute("SELECT COUNT(*) FROM wk_matches")
            n_matches = cur.fetchone()[0]
            cur.execute("""SELECT COUNT(*) FROM match_player WHERE start_time >= %(start)s
                           AND (%(end)s::timestamptz IS NULL OR start_time < %(end)s)""", params)
            n_rows_all = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM wk_players")
            n_players = cur.fetchone()[0]
            log.info("Staged %s complete matches (%s player rows) in %.1fs", f"{n_matches:,}",
                     f"{n_players:,}", time.time() - t0)
            if n_matches == 0:
                log.error("No complete matches in %s — nothing to export.", name)
                return 1
            if n_players != 12 * n_matches:
                log.warning("%s player rows dropped by the heroes join: run hero_ingest_deadlock.py",
                            f"{12 * n_matches - n_players:,}")
        conn.commit()          # keep the temp tables: a failed export's rollback must not drop them

        for export, (sql, formats) in EXPORTS.items():
            try:
                t = time.time()
                df = query_df(conn, sql, params)
                log.info("%-26s %.2fs", export, time.time() - t)
                if export == "_type_matchups":
                    for htype, g_ in df.groupby("hero_type"):
                        write(g_.drop(columns="hero_type"), out_dir, f"{htype}_comp_wr", ("csv",))
                else:
                    write(df, out_dir, export, formats)
            except Exception:
                log.exception("export %s failed", export)
                conn.rollback()
                failed.append(export)

        meta = {"period": name, "start": str(start), "end": str(end) if end else None,
                "complete_matches": n_matches, "player_rows_in_period": n_rows_all,
                "incomplete_rows_excluded": n_rows_all - 12 * n_matches}
        (out_dir / "period_meta.json").write_text(json.dumps(meta, indent=2))
    finally:
        conn.close()

    if failed:
        log.error("%s export(s) failed: %s", len(failed), ", ".join(failed))
        return 1
    log.info("All %s exports written to %s", len(EXPORTS), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())