"""
query_transform.py — weekly SQL transforms -> the CSV/Parquet files the chart scripts read.

How it works
    1. look the period up in patch_periods (written by the ingest)
    2. build two temp tables on one connection, once:
         wk_matches  complete matches of the period (12 players, 6 per side, one winner, 12 different heroes)
         wk_players  their 12 player rows, joined to heroes
       plus wk_type_counts (heroes of each archetype per team, zeros included)
       and wk_lanes (who won each lane: the team that took the OTHER team's Tier 1 first; needs the
       ingest's --with-objectives / --objectives-only, otherwise the lane-win columns come out empty)
    3. run every export against those small tables, instead of re-filtering the full
       match_player table 15 times
    4. write <project>/period_analyses/week<N>_analysis/<name>.csv (or .../<period name>/ for other periods),
       plus period_meta.json: the label charts print and the folder they compare against

Using only complete matches keeps every count consistent (12 hero rows per match), which the
charts rely on when they turn hero appearances into match counts.

Usage:
    python query_transform.py                       # last completed week
    python query_transform.py --date 2026-10-26
    python query_transform.py --week 12
    python query_transform.py --period Release_Nurse_Harrow_d1-7   # any period from register_period.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline_common import connect, get_period, parse_date, period_folder, period_name, setup_logging, week_number

PROJECT_ROOT = Path(__file__).resolve().parents[2]       # scripts/sql_scripts/ -> project root
MIN_DUO_GAMES = 1000          # duos below this are not exported (the chart also filters)
MIN_LANE_PAIR_GAMES = 100     # lane partners / lane opponents below this are not exported
# hero_per_min: (export column, wk_players expression); hero_spotlight picks which ones each hero shows
PER_MIN_STATS = [
    ("souls_pm", "net_worth"), ("damage_pm", "player_damage"), ("obj_damage_pm", "boss_damage"),
    ("tanked_pm", "player_damage_taken"), ("mitigated_pm", "damage_mitigated"),
    ("ally_heal_pm", "teammate_healing"), ("ally_barrier_pm", "teammate_barriering"),
    ("barrier_pm", "player_barriering"),                       # all barriers, own shields included
    ("heal_pm", "player_healing"),                            # all healing, self included
    ("self_heal_pm", "self_healing"),
    ("self_shield_pm", "(player_barriering - teammate_barriering)"),   # the hero's shields on itself
    ("creep_pm", "(creep_damage + neutral_damage)"),           # lane creeps + jungle camps
    ("heal_denied_pm", "heal_prevented"), ("kills_pm", "kills"), ("assists_pm", "assists"),
]
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
       SELECT match_id, start_time, match_duration_s AS duration_s, winning_team, team0, team1,
              average_badge / 10 AS rank_tier          -- 1 Initiate ... 11 Eternus; NULL/0 = unknown
       FROM match_hero_roster
       WHERE start_time >= %(start)s AND (%(end)s::timestamptz IS NULL OR start_time < %(end)s)
         AND cardinality(team0) = 6 AND cardinality(team1) = 6
         -- no hero twice in one match (rare API glitch; validate_week reports how many)
         AND cardinality(ARRAY(SELECT DISTINCT h FROM unnest(team0 || team1) AS h)) = 12""",
    # lane winner per match and lane (lane = assigned_lane numbering); NULL = undecided (both Tier 1s stood,
    # or fell the same second)
    """CREATE TEMP TABLE wk_lanes AS
       SELECT l.match_id, l.lane,
              CASE WHEN l.team1_t1_lost_s IS NOT NULL
                        AND (l.team0_t1_lost_s IS NULL OR l.team1_t1_lost_s < l.team0_t1_lost_s) THEN 'team0'
                   WHEN l.team0_t1_lost_s IS NOT NULL
                        AND (l.team1_t1_lost_s IS NULL OR l.team0_t1_lost_s < l.team1_t1_lost_s) THEN 'team1'
              END AS lane_winner
       FROM match_lanes l JOIN wk_matches m USING (match_id)""",
    """CREATE TEMP TABLE wk_players AS
       SELECT mp.match_id, mp.team, mp.won, mp.hero_id, m.duration_s, m.rank_tier,
              h.name AS hero_name, LOWER(h.hero_type) AS hero_type,
              NULLIF(mp.assigned_lane, 0) AS lane, mp.mvp_rank, mp.net_worth, mp.player_damage,
              mp.boss_damage, mp.player_healing, mp.teammate_healing, mp.player_barriering, mp.player_damage_taken,
              mp.damage_mitigated, mp.self_healing, mp.heal_prevented, mp.last_hits, mp.denies, mp.kills,
              mp.assists, mp.teammate_barriering, mp.creep_damage, mp.neutral_damage,
              CASE WHEN l.lane_winner IS NOT NULL THEN l.lane_winner = LOWER(mp.team) END AS lane_won
       FROM match_player mp
       JOIN wk_matches m USING (match_id)
       JOIN heroes h ON h.hero_id = mp.hero_id
       LEFT JOIN wk_lanes l ON l.match_id = mp.match_id AND l.lane = mp.assigned_lane""",
    "CREATE INDEX ON wk_players (match_id, team)",
    "CREATE INDEX ON wk_players (match_id, lane)",
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

    # power score: how often each hero is banned, among complete matches whose bans were recorded.
    # Heroes banned in every match never appear in wk_players, so heroes come from the bans too.
    "hero_bans": ("""
        WITH recorded AS (SELECT b.match_id, b.banned_hero_ids FROM match_bans b JOIN wk_matches m USING (match_id)),
             n AS (SELECT COUNT(*) AS n FROM recorded),
             banned AS (SELECT u.hero_id, COUNT(DISTINCT r.match_id) AS banned_matches
                        FROM recorded r CROSS JOIN LATERAL UNNEST(r.banned_hero_ids) AS u(hero_id) GROUP BY 1)
        SELECT h.name AS hero_name, COALESCE(b.banned_matches, 0) AS banned_matches,
               n.n AS matches_with_bans,
               ROUND(100.0 * COALESCE(b.banned_matches, 0) / NULLIF(n.n, 0), 2) AS ban_rate
        FROM heroes h CROSS JOIN n LEFT JOIN banned b USING (hero_id)
        WHERE b.hero_id IS NOT NULL OR h.hero_id IN (SELECT DISTINCT hero_id FROM wk_players)
        ORDER BY ban_rate DESC NULLS LAST, hero_name""", ("csv",)),

    # rank_performance: every hero's record in each rank tier of the match (average badge).
    # One row per hero and tier, so the chart script can group tiers into brackets freely.
    "hero_rank_winrates": ("""
        SELECT hero_name, hero_type, rank_tier, COUNT(*) AS games, SUM(won::int) AS wins,
               ROUND(100.0 * AVG(won::int), 2) AS winrate
        FROM wk_players WHERE rank_tier >= 1
        GROUP BY hero_name, hero_type, rank_tier ORDER BY hero_name, rank_tier""", ("csv",)),
    "rank_tier_matches": ("""
        SELECT rank_tier, COUNT(*) AS matches FROM wk_matches WHERE rank_tier >= 1
        GROUP BY rank_tier ORDER BY rank_tier""", ("csv",)),

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

    # hero_spotlight: balance patches and hero releases up to the end of the period (dashed lines on the
    # weekly trend). All of them, not just this period's: the trend reaches back several weeks.
    "patches": ("""
        SELECT start_time, kind, name, description
        FROM patch_events
        WHERE confirmed AND kind IN ('patch', 'minor_patch', 'hero_release')
          AND start_time < COALESCE(%(end)s::timestamptz, 'infinity')
        ORDER BY start_time""", ("csv",)),

    # ---- hero_spotlight wide/tall cards ------------------------------------------------------------
    # Lanes: where each hero starts and how it does there. lane = assigned_lane (1, 4, 6).
    # lane_games / lane_wins only count lanes that were decided (a Tier 1 fell), so lane_winrate is
    # "how often this hero's lane took the enemy Tier 1 first".
    "hero_lanes": ("""
        SELECT hero_name, lane, COUNT(*) AS games, SUM(won::int) AS wins,
               ROUND(100.0 * AVG(won::int), 2) AS winrate,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY hero_name), 2) AS lane_share,
               COUNT(lane_won) AS lane_games, SUM(lane_won::int) AS lane_wins,
               ROUND(100.0 * AVG(lane_won::int), 2) AS lane_winrate
        FROM wk_players WHERE lane IS NOT NULL
        GROUP BY hero_name, lane ORDER BY hero_name, lane""", ("csv",)),

    # Lane partners: same team, same starting lane. Both directions (one row per hero and partner).
    "hero_lane_partners": ("""
        SELECT a.hero_name, b.hero_name AS partner_name, COUNT(*) AS games, SUM(a.won::int) AS wins,
               ROUND(100.0 * AVG(a.won::int), 2) AS winrate,
               COUNT(a.lane_won) AS lane_games, ROUND(100.0 * AVG(a.lane_won::int), 2) AS lane_winrate
        FROM wk_players a
        JOIN wk_players b ON b.match_id = a.match_id AND b.team = a.team AND b.lane = a.lane
                         AND b.hero_id <> a.hero_id
        WHERE a.lane IS NOT NULL
        GROUP BY a.hero_name, b.hero_name
        HAVING COUNT(*) >= %(min_lane)s
        ORDER BY a.hero_name, winrate DESC""", ("csv",)),

    # Lane matchups: the enemies who started in the same lane (assigned_lane matches across teams).
    "hero_lane_matchups": ("""
        SELECT a.hero_name, b.hero_name AS opponent_name, COUNT(*) AS games, SUM(a.won::int) AS wins,
               ROUND(100.0 * AVG(a.won::int), 2) AS winrate,
               COUNT(a.lane_won) AS lane_games, ROUND(100.0 * AVG(a.lane_won::int), 2) AS lane_winrate
        FROM wk_players a
        JOIN wk_players b ON b.match_id = a.match_id AND b.team <> a.team AND b.lane = a.lane
        WHERE a.lane IS NOT NULL
        GROUP BY a.hero_name, b.hero_name
        HAVING COUNT(*) >= %(min_lane)s
        ORDER BY a.hero_name, winrate DESC""", ("csv",)),

    # MVP: mvp_rank 1 = the match MVP (always on the winning side), 2-3 = the two "key players" (at least one
    # of them on the losing side). top3_loss_rate = share of the hero's LOSSES where it was still top 3.
    # Only matches where the API recorded MVPs count (a match with none has mvp_rank NULL for all 12).
    "hero_mvp": ("""
        WITH rec AS (SELECT match_id FROM wk_players GROUP BY match_id HAVING COUNT(mvp_rank) > 0)
        SELECT hero_name, COUNT(*) AS games,
               SUM((mvp_rank = 1)::int) AS mvp, SUM((mvp_rank <= 3)::int) AS top3,
               ROUND(100.0 * AVG(COALESCE(mvp_rank = 1, false)::int), 2) AS mvp_rate,   -- NULL = not top 3 = 0
               ROUND(100.0 * AVG(COALESCE(mvp_rank <= 3, false)::int), 2) AS top3_rate,
               COUNT(*) FILTER (WHERE NOT won) AS losses,
               SUM((mvp_rank <= 3)::int) FILTER (WHERE NOT won) AS top3_in_losses,
               -- key player while losing: still swinging games the team couldn't win
               ROUND(100.0 * AVG(COALESCE(mvp_rank <= 3, false)::int) FILTER (WHERE NOT won), 2) AS top3_loss_rate
        FROM wk_players JOIN rec USING (match_id)
        GROUP BY hero_name ORDER BY mvp_rate DESC""", ("csv",)),

    # Per-minute output, macro level: total over total minutes (ratio of sums), each stat over the games
    # that have it (the stats.* ones need --with-stats, objective damage --with-objectives). Plus kill
    # participation, which is per match, not per minute.
    "hero_per_min": ("""
        SELECT hero_name, COUNT(*) AS games,
""" + ",\n".join(f"               ROUND(60.0 * SUM({c}) / NULLIF(SUM(duration_s) FILTER (WHERE {c} IS NOT NULL), 0), 3)"
                  f" AS {name}" for name, c in PER_MIN_STATS) + """,
               -- kill participation: (kills + assists) / the team's kills, per match, averaged over matches
               ROUND(100.0 * AVG((kills + assists)::numeric / NULLIF(team_kills, 0)), 2) AS kill_participation,
               COUNT(player_damage) AS games_with_stats, COUNT(boss_damage) AS games_with_obj
        FROM (SELECT *, SUM(kills) OVER (PARTITION BY match_id, team) AS team_kills FROM wk_players) p
        GROUP BY hero_name ORDER BY hero_name""", ("csv",)),

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
def period_info(cur, name: str) -> dict:
    """Start/end plus what charts need: a display label and the folder to compare against."""
    p = get_period(cur, name)
    if name.startswith("Week_") and name[5:].isdigit():
        n = int(name[5:])
        p["label"], p["compare_dir"] = f"Week {n}", (f"week{n - 1}_analysis" if n > 1 else None)
    else:
        p["label"] = p["description"] or name.replace("_", " ")
        # folder NAME only: the comparison folder sits next to this one (chart_stats.previous_week_dir)
        p["compare_dir"] = Path(period_folder(p["compare_to"])).name if p["compare_to"] else None
    return p


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
    g.add_argument("--period", help="any period in patch_periods, e.g. one from register_period.py")
    ap.add_argument("--with-compare", action="store_true",
                    help="also export the period it is compared against (patch periods' 'before' period)")
    ap.add_argument("--out-dir", help="override the output folder")
    args = ap.parse_args()

    name = args.period or period_name(args.week or week_number(parse_date(args.date)))
    names = [name]
    if args.with_compare:
        conn = connect()
        with conn.cursor() as cur:
            compare = get_period(cur, name)["compare_to"]
        conn.close()
        names += [compare] if compare else []
    rc = 0
    for n in names:
        out_dir = Path(args.out_dir) / Path(period_folder(n)).name if (args.out_dir and len(names) > 1) else \
            Path(args.out_dir) if args.out_dir else PROJECT_ROOT / period_folder(n)
        rc |= transform(n, out_dir)
    return rc


def transform(name: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect()
    failed = []
    try:
        with conn.cursor() as cur:
            info = period_info(cur, name)
            start, end = info["start"], info["end"]
            if end is not None and end > datetime.now(timezone.utc):
                log.error("%s isn't over yet (ends %s UTC): analyze it after that", name, f"{end:%Y-%m-%d %H:%M}")
                return 1
            log.info("%s: %s -> %s  =>  %s", name, start, end or "open", out_dir)
            params = {"start": start, "end": end, "min_duo": MIN_DUO_GAMES, "min_lane": MIN_LANE_PAIR_GAMES, "top_comps": TOP_COMPS}
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
            cur.execute("SELECT COUNT(*) FROM match_bans b JOIN wk_matches m USING (match_id)")
            n_ban_matches = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT match_id) FROM wk_lanes")
            n_lane_matches = cur.fetchone()[0]
            if n_lane_matches < 0.5 * max(n_matches, 1):
                log.warning("Lane towers recorded for only %s of %s matches: run the ingest with "
                            "--objectives-only for this range to fill lane wins", f"{n_lane_matches:,}",
                            f"{n_matches:,}")
            log.info("Bans recorded for %s of %s complete matches (%.0f%%)", f"{n_ban_matches:,}",
                     f"{n_matches:,}", 100 * n_ban_matches / max(n_matches, 1))
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

        meta = {"period": name, "label": info["label"], "compare_dir": info["compare_dir"],
                "start": str(start), "end": str(end) if end else None,
                "complete_matches": n_matches, "player_rows_in_period": n_rows_all,
                "incomplete_rows_excluded": n_rows_all - 12 * n_matches,
                "matches_with_bans": n_ban_matches, "matches_with_lanes": n_lane_matches,
                "ban_coverage_pct": round(100 * n_ban_matches / max(n_matches, 1), 1)}
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