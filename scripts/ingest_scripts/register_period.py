"""
register_period.py — define analysis periods that follow patches and hero releases, not weeks.

Weekly periods (Week_N, Thursday to Thursday since ranked launch) are created by the ingest.
This script adds named periods of any length to patch_periods, each paired with the period of
the same length right before it, which the charts and insights.py compare against.

    python register_period.py --releases
        list each hero's first ranked match in your data (= its release time), newest first

    python register_period.py --release "Nurse Harrow"
        Release_Nurse_Harrow_d1-7         first 7 days after the hero's first ranked match
        Release_Nurse_Harrow_d1-7_before  the 7 days before it (comparison)

    python register_period.py --release "Nurse Harrow" --skip-days 1 --days 7    # days 2-8
    python register_period.py --name Patch_2026-09-29 --start "2026-09-29 20:25" --days 7 \\
                              --label "Sept 29 patch, first week"

    python register_period.py --list

Then run it like a week:
    python scripts/run_pipeline.py --period Release_Nurse_Harrow_d1-7

Times are UTC. Use the patch's Steam post time (or the in-game release time) for --start.
Re-registering a name updates it. A period that hasn't finished yet is registered with its
future end; the pipeline refuses to analyze it until it is complete.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone

from pipeline_common import connect, ensure_schema, period_folder, setup_logging

log = setup_logging()
FIRST_CUSTOM_ID = 1000


def parse_utc(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise SystemExit(f"Can't read '{s}': use YYYY-MM-DD HH:MM (UTC)")


def releases(cur) -> list[tuple[str, datetime]]:
    """Each hero's first ranked match, for heroes that first appear 2+ days after the earliest data
    you have (so heroes that existed before you started ingesting aren't counted as releases)."""
    cur.execute("""
        WITH firsts AS (SELECT hero_id, MIN(start_time) AS first_match FROM match_player GROUP BY hero_id)
        SELECT h.name, f.first_match FROM firsts f JOIN heroes h USING (hero_id)
        WHERE f.first_match >= (SELECT MIN(first_match) FROM firsts) + INTERVAL '2 days'
        ORDER BY f.first_match DESC""")
    return cur.fetchall()


def upsert(cur, name: str, start: datetime, end: datetime, label: str, compare_to: str | None) -> None:
    cur.execute("SELECT period_id FROM patch_periods WHERE period_name = %s", (name,))
    row = cur.fetchone()
    if row:
        cur.execute("""UPDATE patch_periods SET start_time = %s, end_time = %s, description = %s,
                       compare_to = %s WHERE period_id = %s""", (start, end, label[:100], compare_to, row[0]))
        return
    cur.execute("SELECT GREATEST(COALESCE(MAX(period_id), 0) + 1, %s) FROM patch_periods WHERE period_id >= %s",
                (FIRST_CUSTOM_ID, FIRST_CUSTOM_ID))
    pid = cur.fetchone()[0]
    cur.execute("""INSERT INTO patch_periods (period_id, period_name, start_time, end_time, description, compare_to)
                   VALUES (%s, %s, %s, %s, %s, %s)""", (pid, name, start, end, label[:100], compare_to))


def register(cur, name: str, start: datetime, days: float, label: str, before: bool) -> None:
    end = start + timedelta(days=days)
    compare = f"{name}_before" if before else None
    if before:
        before_label = "the week before" if days == 7 else f"the {days:g} days before"
        upsert(cur, compare, start - timedelta(days=days), start, before_label, None)
    upsert(cur, name, start, end, label, compare)
    now = datetime.now(timezone.utc)
    log.info("Registered %s: %s -> %s UTC (%s)%s", name, f"{start:%Y-%m-%d %H:%M}", f"{end:%Y-%m-%d %H:%M}",
             label, "" if end <= now else f"  [complete in {(end - now).total_seconds() / 3600:.0f} h]")
    if before:
        log.info("  compared against %s (%s -> %s)", compare, f"{start - timedelta(days=days):%Y-%m-%d %H:%M}",
                 f"{start:%Y-%m-%d %H:%M}")
    log.info("  output folder: %s/   run: python scripts/run_pipeline.py --period %s", period_folder(name), name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Register patch / hero-release analysis periods")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--releases", action="store_true", help="list hero release times found in the data")
    what.add_argument("--list", action="store_true", help="list registered periods")
    what.add_argument("--release", metavar="HERO", help="period starting at this hero's first ranked match")
    what.add_argument("--name", help="custom period name, e.g. Patch_2026-09-29 (use with --start)")
    ap.add_argument("--start", help="custom period start, UTC: 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--days", type=float, default=7, help="period length (default 7)")
    ap.add_argument("--skip-days", type=float, default=0, help="start this many days after the event")
    ap.add_argument("--label", help="how charts name the period (default derived from the name)")
    ap.add_argument("--no-before", action="store_true", help="don't register the comparison period")
    args = ap.parse_args()

    conn = connect()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            if args.releases:
                for hero, t in releases(cur):
                    print(f"{t:%Y-%m-%d %H:%M} UTC ({t:%a})  {hero}")
                return 0
            if args.list:
                cur.execute("""SELECT period_name, start_time, end_time, description, compare_to FROM patch_periods
                               WHERE period_id >= %s ORDER BY start_time DESC""", (FIRST_CUSTOM_ID,))
                for name, s, e, d, c in cur.fetchall():
                    print(f"{name:40s} {s:%Y-%m-%d %H:%M} -> {e:%Y-%m-%d %H:%M}  {d}" + (f"  (vs {c})" if c else ""))
                return 0

            first = args.skip_days + 1
            last = args.skip_days + args.days
            span = f"d{first:g}-{last:g}"
            if args.release:
                found = {h.lower(): (h, t) for h, t in releases(cur)}
                if args.release.lower() not in found:
                    raise SystemExit(f"No release found for '{args.release}'. Heroes with a release in your "
                                     f"data: {', '.join(h for h, _ in found.values()) or 'none'}")
                hero, t = found[args.release.lower()]
                event = t.replace(second=0, microsecond=0)
                name = f"Release_{re.sub(r'[^A-Za-z0-9]+', '_', hero).strip('_')}_{span}"
                label = args.label or f"{hero}, days {first:g}–{last:g}"
            else:
                if not args.start:
                    raise SystemExit("--name needs --start 'YYYY-MM-DD HH:MM' (UTC)")
                event = parse_utc(args.start)
                name = args.name
                label = args.label or f"{name.replace('_', ' ')}, days {first:g}–{last:g}"
            register(cur, name, event + timedelta(days=args.skip_days), args.days, label, not args.no_before)
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())