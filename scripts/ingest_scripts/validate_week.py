"""
validate_week.py — data-quality checks for one ingested week (replaces ingest_missing_rosters.py).

Run after ingest, before query_transform. Each check prints PASS / WARN / FAIL with counts;
the script exits 1 if anything FAILs, so a scheduled pipeline stops before charts are built
from bad data.

Checks
    rows_loaded         the week has player rows at all
    unknown_heroes      hero_ids missing from `heroes` (would vanish from every analysis join)
    incomplete_matches  matches without exactly 12 players / 6 per team
    winner_consistency  matches where won flags don't mark exactly one team of 6 winners
    duplicate_heroes    the same hero twice in one match
    duration_mismatch   players of one match disagreeing on match length
    roster_coverage     complete matches missing from match_hero_roster
    ban_coverage        share of complete matches with recorded bans (info only: the API
                        records bans only for matches whose replay it processed)

Usage: python validate_week.py [--week N | --date YYYY-MM-DD] [--max-incomplete-pct 2.0]
"""
from __future__ import annotations

import argparse
import sys

from pipeline_common import connect, parse_date, setup_logging, week_bounds, week_number

log = setup_logging()

CHECKS = {
    "rows_loaded": ("FAIL", """
        SELECT COUNT(*) = 0, COUNT(*) FROM wk"""),
    "unknown_heroes": ("FAIL", """
        SELECT COUNT(DISTINCT hero_id) > 0, COUNT(DISTINCT hero_id)
        FROM wk LEFT JOIN heroes h USING (hero_id) WHERE h.hero_id IS NULL"""),
    "incomplete_matches": ("WARN", """
        WITH m AS (SELECT match_id, COUNT(*) n, COUNT(*) FILTER (WHERE LOWER(team) = 'team0') n0
                   FROM wk GROUP BY match_id)
        SELECT 100.0 * COUNT(*) FILTER (WHERE n <> 12 OR n0 <> 6) / NULLIF(COUNT(*), 0) > %(max_incomplete)s,
               COUNT(*) FILTER (WHERE n <> 12 OR n0 <> 6)
        FROM m"""),
    "winner_consistency": ("WARN", """
        WITH m AS (SELECT match_id, COUNT(*) FILTER (WHERE won) n_won,
                          COUNT(DISTINCT team) FILTER (WHERE won) teams_won,
                          BOOL_OR(won IS DISTINCT FROM (team = winning_team)) mismatch
                   FROM wk GROUP BY match_id HAVING COUNT(*) = 12)
        SELECT COUNT(*) FILTER (WHERE n_won <> 6 OR teams_won <> 1 OR mismatch) > 0,
               COUNT(*) FILTER (WHERE n_won <> 6 OR teams_won <> 1 OR mismatch)
        FROM m"""),
    "duplicate_heroes": ("FAIL", """
        SELECT COUNT(*) > 0, COUNT(*) FROM (
            SELECT 1 FROM wk GROUP BY match_id, hero_id HAVING COUNT(*) > 1) d"""),
    "duration_mismatch": ("WARN", """
        SELECT COUNT(*) > 0, COUNT(*) FROM (
            SELECT 1 FROM wk GROUP BY match_id HAVING MAX(duration_s) - MIN(duration_s) > 5) d"""),
    "roster_coverage": ("WARN", """
        WITH complete AS (SELECT match_id FROM wk GROUP BY match_id
                          HAVING COUNT(*) = 12 AND COUNT(*) FILTER (WHERE LOWER(team) = 'team0') = 6)
        SELECT COUNT(*) > 0, COUNT(*)
        FROM complete c LEFT JOIN match_hero_roster r USING (match_id) WHERE r.match_id IS NULL"""),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--week", type=int)
    g.add_argument("--date", "--start-date", dest="date")
    ap.add_argument("--max-incomplete-pct", type=float, default=2.0,
                    help="incomplete matches above this share of the week are flagged")
    args = ap.parse_args()

    week = args.week or week_number(parse_date(args.date))
    start, end = week_bounds(week)
    conn = connect()
    worst = "PASS"
    try:
        with conn.cursor() as cur:
            cur.execute("""CREATE TEMP TABLE wk AS SELECT match_id, account_id, team, won, winning_team,
                           hero_id, duration_s FROM match_player WHERE start_time >= %s AND start_time < %s""",
                        (start, end))
            cur.execute("SELECT COUNT(DISTINCT match_id), COUNT(*) FROM wk")
            n_matches, n_rows = cur.fetchone()
            log.info("Week %s: %s matches, %s player rows", week, f"{n_matches:,}", f"{n_rows:,}")
            for name, (severity, sql) in CHECKS.items():
                cur.execute(sql, {"max_incomplete": args.max_incomplete_pct})
                bad, count = cur.fetchone()
                status = severity if bad else "PASS"
                if status == "FAIL" or (status == "WARN" and worst == "PASS"):
                    worst = status
                log.info("%-5s %-20s %s", status, name, f"{count:,}")
            cur.execute("""
                WITH complete AS (SELECT match_id FROM wk GROUP BY match_id
                                  HAVING COUNT(*) = 12 AND COUNT(*) FILTER (WHERE LOWER(team) = 'team0') = 6)
                SELECT COUNT(*), COUNT(b.match_id) FROM complete c LEFT JOIN match_bans b USING (match_id)""")
            n_complete, n_bans = cur.fetchone()
            log.info("INFO  %-20s %s of %s complete matches (%.1f%%)", "ban_coverage", f"{n_bans:,}",
                     f"{n_complete:,}", 100 * n_bans / max(n_complete, 1))
    finally:
        conn.close()
    log.info("Result: %s", worst)
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())