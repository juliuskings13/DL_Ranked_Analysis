"""
hero_ingest_deadlock.py — sync the heroes dimension table from the Deadlock API's assets endpoint.

Run it whenever a hero is added or renamed. Every analysis query joins on heroes, so a
hero missing here is silently dropped from all results (validate_week.py checks for this).
hero_type is stored lower-case so queries never need LOWER().

The API leaves hero_type out for some heroes (since the "City Never Sleeps" update: Dynamo, Kelvin, Ivy,
Paige, Rem, Nurse Harrow). Those get their class from HERO_TYPE_FALLBACK below, and a hero_type already
in your table is never overwritten with an empty one, so classes you set by hand survive every sync.

Usage: python hero_ingest_deadlock.py
"""
from __future__ import annotations

import sys

import requests

from pipeline_common import connect, copy_upsert, ensure_schema, setup_logging

# the old assets host (assets.deadlock-api.com/v2) was retired; same data now lives under /v1/assets
HEROES_URL = "https://api.deadlock-api.com/v1/assets/heroes?only_active=true"
log = setup_logging()

# class for heroes the API returns without one (only used when the API's value is empty)
HERO_TYPE_FALLBACK = {
    "Dynamo": "mystic",
    "Paige": "mystic",
    "Kelvin": "brawler",
    "Ivy": "marksman",            # fill in: brawler / marksman / assassin / mystic
    "Rem": "mystic",
    "Nurse Harrow": "mystic",
}


def fetch_heroes() -> list[tuple]:
    res = requests.get(HEROES_URL, timeout=30)
    res.raise_for_status()
    rows = []
    for h in res.json():
        if h.get("id") is None:
            continue
        rows.append((h["id"], h.get("name") or f"Hero_{h['id']}", h.get("class_name"), h.get("hero_type"),
                     h.get("complexity")))
    return rows


def normalize(rows: list[tuple]) -> list[tuple]:
    """Lower-case, trimmed hero_type (None if blank), applied right before writing."""
    return [(i, name, cls, (t or "").strip().lower() or HERO_TYPE_FALLBACK.get(name) or None, cx)
            for i, name, cls, t, cx in rows]


def main() -> int:
    try:
        rows = normalize(fetch_heroes())
    except requests.RequestException:
        log.exception("Could not fetch heroes from %s", HEROES_URL)
        return 1
    conn = connect()
    try:
        ensure_schema(conn)
        copy_upsert(conn, "heroes", ["hero_id", "name", "class_name", "hero_type", "complexity"], rows,
                    key=["hero_id"],
                    conflict="DO UPDATE SET name = EXCLUDED.name, class_name = EXCLUDED.class_name, "
                             # an empty class from the API never wipes one already stored (e.g. set by hand)
                             "hero_type = COALESCE(EXCLUDED.hero_type, heroes.hero_type), "
                             "complexity = EXCLUDED.complexity")
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM heroes WHERE hero_type IS NULL AND hero_id = ANY(%s) ORDER BY name",
                        ([r[0] for r in rows],))
            missing_type = [n for (n,) in cur.fetchall()]
        log.info("Synced %s heroes", len(rows))
        if missing_type:
            log.warning("No hero_type for: %s (they won't appear in archetype charts; add them to "
                        "HERO_TYPE_FALLBACK)", ", ".join(missing_type))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())