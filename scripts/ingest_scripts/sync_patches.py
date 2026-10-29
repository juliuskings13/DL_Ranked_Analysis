"""
sync_patches.py — keep a table of patches and hero releases, and turn confirmed ones into
analysis periods (days 1-7 plus the week before, see register_period.py).

Sources
    feed   the Deadlock API's patch feed: the official forum changelog and Steam news posts
    data   hero releases found in your own match data (a hero's first ranked match)

Every run re-reads the feed; nothing is stored twice:
    - a post is stored once (its link is the key)
    - posts within GROUP_HOURS of each other become ONE patch event (forum + Steam copies of the same
      patch, a hotfix a few hours later, a hero shipping with a balance patch)
    - a forum post only says the date (its timestamp is the last edit), so it joins an event on that
      date or the next day (Valve dates posts in Pacific time; an evening patch is the next day in UTC)

Classification
    hero release (from data)            -> confirmed, named Release_<Hero>
    has an official forum changelog     -> confirmed, named Patch_<YYYY-MM-DD>
    Steam "Minor Update"                -> minor_patch, waiting for you
    anything else (news, spotlights)    -> unknown, waiting for you
Confirmed events get a period registered automatically: <name>_d1-7 and <name>_d1-7_before.

Usage
    python sync_patches.py                          # fetch + group + register periods, then show pending
    python sync_patches.py --list                   # every event
    python sync_patches.py --confirm 12 [--name Patch_6_0] [--description "Big October patch"]
    python sync_patches.py --rename 12 Patch_6_0 [--description ...]
    python sync_patches.py --ignore 13              # not a patch (news post, spotlight, ...)
    python sync_patches.py --merge 12 13            # same patch: move 13's posts into 12
    python sync_patches.py --feed-file feed.json    # offline / testing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from pipeline_common import RANKED_START_DATE, connect, ensure_schema, setup_logging
from register_period import register, releases

log = setup_logging()
FEED_URL = "https://api.deadlock-api.com/v2/patches"
GROUP_HOURS = 12          # posts this close together are one patch (hotfixes, forum + Steam copies)
PERIOD_DAYS = 7


# --- FEED --------------------------------------------------------------------------
def _pick(d: dict, *keys):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def _parse_time(v) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)
    s = str(v).strip().replace("Z", "+00:00")
    for fmt in (None, "%a, %d %b %Y %H:%M:%S %z"):            # ISO 8601, then RSS-style dates
        try:
            t = datetime.fromisoformat(s) if fmt is None else datetime.strptime(s, fmt)
            return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def fetch_feed(feed_file: str | None) -> list[dict]:
    """[{source, title, link, published_at}] from the API (or a saved JSON file)."""
    if feed_file:
        raw = json.loads(open(feed_file).read())
    else:
        for attempt in range(1, 4):
            try:
                r = requests.get(FEED_URL, timeout=30)
                r.raise_for_status()
                raw = r.json()
                break
            except (requests.RequestException, ValueError) as e:
                if attempt == 3:
                    raise
                log.warning("patch feed failed (%s), retrying in %ss", e, 5 * attempt)
                time.sleep(5 * attempt)
    items = raw if isinstance(raw, list) else _pick(raw, "patches", "items", "data") or []
    out = []
    for it in items:
        link = _pick(it, "link", "url", "guid")
        title = _pick(it, "title", "name")
        published = _parse_time(_pick(it, "pub_date", "pubDate", "published_at", "date"))
        source = (_pick(it, "source", "category") or "").lower()
        if not source and link:
            source = "steam" if "steampowered" in link else "forum" if "forums.playdeadlock" in link else ""
        if link and title and published:
            out.append({"source": source or "unknown", "title": title, "link": link, "published_at": published})
    if items and not out:
        raise SystemExit("The patch feed's format wasn't recognized. First entry's fields: "
                         f"{sorted(items[0])}. Send this line to update the parser.")
    return out


def effective_time(post: dict) -> tuple[datetime, bool]:
    """(when the patch happened, whether the time of day is known)."""
    if post["source"] == "forum":
        m = re.search(r"(\d{2})-(\d{2})-(\d{4})", post["title"])     # forum titles: MM-DD-YYYY
        if m:
            month, day, year = map(int, m.groups())
            return datetime(year, month, day, tzinfo=timezone.utc), False
        return post["published_at"], False                          # last-edit time: date-ish only
    return post["published_at"], True


# --- GROUPING ----------------------------------------------------------------------
def store_posts(cur, posts: list[dict]) -> int:
    new = 0
    for p in posts:
        t, known = effective_time(p)
        cur.execute("""INSERT INTO patch_posts (link, source, title, published_at, effective_at, time_known)
                       VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (link) DO NOTHING""",
                    (p["link"], p["source"], p["title"][:300], p["published_at"], t, known))
        new += cur.rowcount
    return new


def store_releases(cur) -> int:
    """Hero releases from the match data, stored as posts so they group like everything else."""
    new = 0
    for hero, first in releases(cur):
        cur.execute("""INSERT INTO patch_posts (link, source, title, published_at, effective_at, time_known)
                       VALUES (%s, 'data', %s, %s, %s, TRUE) ON CONFLICT (link) DO NOTHING""",
                    (f"data:hero_release:{hero}", f"{hero}: first ranked match", first,
                     first.replace(second=0, microsecond=0)))
        new += cur.rowcount
    return new


def find_event(cur, t: datetime, time_known: bool) -> int | None:
    if time_known:
        cur.execute("""SELECT event_id FROM patch_events
                       WHERE start_time BETWEEN %s AND %s AND kind <> 'ignore'
                       ORDER BY abs(extract(epoch FROM start_time - %s)) LIMIT 1""",
                    (t - timedelta(hours=GROUP_HOURS), t + timedelta(hours=GROUP_HOURS), t))
    else:   # date only: an event that day or the next (UTC)
        cur.execute("""SELECT event_id FROM patch_events
                       WHERE start_time >= %s AND start_time < %s AND kind <> 'ignore'
                       ORDER BY start_time LIMIT 1""", (t, t + timedelta(days=2)))
    row = cur.fetchone()
    return row[0] if row else None


def group_posts(cur) -> int:
    """Attach every unassigned post to an event, creating events as needed (timed posts first,
    so a date-only forum post can join the Steam post that has the real time)."""
    cur.execute("""SELECT link, effective_at, time_known FROM patch_posts WHERE event_id IS NULL
                   ORDER BY time_known DESC, effective_at""")
    created = 0
    for link, t, known in cur.fetchall():
        eid = find_event(cur, t, known)
        if eid is None:
            cur.execute("INSERT INTO patch_events (start_time, kind) VALUES (%s, 'unknown') RETURNING event_id", (t,))
            eid = cur.fetchone()[0]
            created += 1
        elif known:   # a timed post can only move an event earlier
            cur.execute("UPDATE patch_events SET start_time = LEAST(start_time, %s) WHERE event_id = %s", (t, eid))
        cur.execute("UPDATE patch_posts SET event_id = %s WHERE link = %s", (eid, link))
    return created


def classify(cur) -> None:
    """Fill kind / name / description of events you haven't confirmed or named yourself."""
    cur.execute("""SELECT event_id, start_time FROM patch_events
                   WHERE NOT confirmed AND kind <> 'ignore' AND name IS NULL""")
    for eid, start in cur.fetchall():
        cur.execute("SELECT source, title, effective_at, time_known FROM patch_posts WHERE event_id = %s "
                    "ORDER BY effective_at", (eid,))
        posts = cur.fetchall()
        data = [p for p in posts if p[0] == "data"]
        forum = [p for p in posts if p[0] == "forum"]
        steam = [p for p in posts if p[0] == "steam"]
        titles = "; ".join(p[1] for p in posts if p[0] != "data")[:200] or None
        if data:
            hero = data[0][1].split(":")[0]
            kind, name, confirmed, start = "hero_release", "Release_" + re.sub(r"[^A-Za-z0-9]+", "_", hero).strip("_"), True, data[0][2]
            desc = f"{hero} release" + (f" ({titles})" if titles else "")
        elif forum:
            kind, confirmed = "patch", True
            name, desc = f"Patch_{forum[0][2]:%Y-%m-%d}", titles
            if not any(p[3] for p in posts):       # only a date: keep the day, flag it
                desc = (desc or "") + " [time unknown: start set to 00:00 UTC]"
        elif any("update" in p[1].lower() for p in steam):
            kind, name, confirmed, desc = "minor_patch", f"Patch_{start:%Y-%m-%d}", False, titles
        else:
            kind, name, confirmed, desc = "unknown", None, False, titles
        if name:   # keep names unique (two events the same day)
            cur.execute("SELECT 1 FROM patch_events WHERE name = %s AND event_id <> %s", (name, eid))
            if cur.fetchone():
                name = f"{name}_{eid}"
        cur.execute("""UPDATE patch_events SET kind = %s, name = %s, description = %s, confirmed = %s,
                       start_time = %s WHERE event_id = %s""",
                    (kind, name, desc[:200] if desc else None, confirmed, start, eid))


def register_periods(cur) -> list[str]:
    # only events after ranked launch: earlier patches have no ranked data to analyze
    cur.execute("""SELECT event_id, name, start_time, description FROM patch_events
                   WHERE confirmed AND kind <> 'ignore' AND name IS NOT NULL AND period_name IS NULL
                     AND start_time >= %s + INTERVAL '7 days'
                   ORDER BY start_time""", (RANKED_START_DATE,))
    done = []
    for eid, name, start, desc in cur.fetchall():
        period = f"{name}_d1-{PERIOD_DAYS}"
        label = (desc.split(" (")[0] if desc and name.startswith("Release_") else name.replace("_", " "))
        register(cur, period, start, PERIOD_DAYS, f"{label}, days 1–{PERIOD_DAYS}", before=True)
        cur.execute("UPDATE patch_events SET period_name = %s WHERE event_id = %s", (period, eid))
        done.append(period)
    return done


# --- REVIEW ------------------------------------------------------------------------
def show(cur, pending_only: bool) -> None:
    cur.execute(f"""SELECT e.event_id, e.start_time, e.kind, e.name, e.confirmed, e.period_name, e.description,
                           count(p.link) FROM patch_events e LEFT JOIN patch_posts p USING (event_id)
                    {"WHERE NOT e.confirmed AND e.kind <> 'ignore'" if pending_only else ""}
                    GROUP BY e.event_id ORDER BY e.start_time DESC""")
    rows = cur.fetchall()
    if pending_only and not rows:
        log.info("Nothing waiting for review.")
        return
    if pending_only:
        log.info("Waiting for you (confirm, rename, ignore or merge):")
    for eid, start, kind, name, ok, period, desc, n in rows:
        state = "✓" if ok else "ignored" if kind == "ignore" else "pending"
        print(f"  #{eid:<4} {start:%Y-%m-%d %H:%M} UTC  {kind:12s} {state:8s} {name or '(no name)':30s} "
              f"{n} post(s)  {desc or ''}" + (f"  -> {period}" if period else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync patches and hero releases; register analysis periods")
    act = ap.add_mutually_exclusive_group()
    act.add_argument("--list", action="store_true", help="show every event")
    act.add_argument("--confirm", type=int, metavar="ID", help="confirm an event (registers its periods)")
    act.add_argument("--rename", nargs=2, metavar=("ID", "NAME"))
    act.add_argument("--ignore", type=int, metavar="ID", help="mark an event as not a patch")
    act.add_argument("--merge", nargs=2, type=int, metavar=("KEEP", "DROP"), help="merge DROP's posts into KEEP")
    ap.add_argument("--name", help="with --confirm: name to use, e.g. Patch_6_0")
    ap.add_argument("--description", help="with --confirm / --rename")
    ap.add_argument("--feed-file", help="read the feed from a JSON file instead of the API")
    ap.add_argument("--no-feed", action="store_true", help="only detect hero releases from the data")
    args = ap.parse_args()

    conn = connect()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            if args.list:
                show(cur, pending_only=False)
                return 0
            if args.rename or args.confirm is not None:
                eid = int(args.rename[0]) if args.rename else args.confirm
                name = args.rename[1] if args.rename else args.name
                if name:   # a confirmed event gets a period under its new name on the next step
                    cur.execute("UPDATE patch_events SET name = %s, period_name = NULL WHERE event_id = %s",
                                (name, eid))
                if args.description:
                    cur.execute("UPDATE patch_events SET description = %s WHERE event_id = %s",
                                (args.description[:200], eid))
                if args.confirm is not None:
                    cur.execute("SELECT name FROM patch_events WHERE event_id = %s", (eid,))
                    row = cur.fetchone()
                    if not row or not row[0]:
                        raise SystemExit(f"Event #{eid} has no name yet: add --name, e.g. --name Patch_6_0")
                    cur.execute("UPDATE patch_events SET confirmed = TRUE, kind = CASE WHEN kind IN "
                                "('unknown', 'minor_patch') THEN 'patch' ELSE kind END WHERE event_id = %s", (eid,))
                    for p in register_periods(cur):
                        log.info("Registered %s", p)
                if args.rename:
                    for p in register_periods(cur):
                        log.info("Registered %s (the old period stays until you remove it)", p)
            elif args.ignore is not None:
                cur.execute("UPDATE patch_events SET kind = 'ignore', confirmed = FALSE WHERE event_id = %s",
                            (args.ignore,))
            elif args.merge:
                keep, drop = args.merge
                cur.execute("UPDATE patch_posts SET event_id = %s WHERE event_id = %s", (keep, drop))
                cur.execute("""UPDATE patch_events SET start_time = LEAST(start_time,
                                   (SELECT start_time FROM patch_events WHERE event_id = %s)) WHERE event_id = %s""",
                            (drop, keep))
                cur.execute("DELETE FROM patch_events WHERE event_id = %s", (drop,))
                # a confirmed event's period follows its (possibly earlier) start
                cur.execute("UPDATE patch_events SET period_name = NULL WHERE event_id = %s AND confirmed", (keep,))
                for p in register_periods(cur):
                    log.info("Updated %s", p)
            else:
                new_posts = 0
                if not args.no_feed:
                    try:
                        new_posts = store_posts(cur, fetch_feed(args.feed_file))
                    except (requests.RequestException, ValueError) as e:
                        log.warning("Patch feed unavailable (%s); using hero releases from the data only", e)
                new_rel = store_releases(cur)
                created = group_posts(cur)
                classify(cur)
                periods = register_periods(cur)
                log.info("Patch feed: %s new post(s), %s new hero release(s) -> %s new event(s), %s new period(s)",
                         new_posts, new_rel, created, len(periods))
                show(cur, pending_only=True)
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
