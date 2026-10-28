"""
hero_ingest_deadlock.py — sync the heroes dimension table from the Deadlock assets API.

Run it whenever a hero is added or renamed. Every analysis query joins on heroes, so a
hero missing here is silently dropped from all results (validate_week.py checks for this).
hero_type is stored lower-case so queries never need LOWER().

Usage: python hero_ingest_deadlock.py
"""
from __future__ import annotations

import sys

import requests

from pipeline_common import connect, copy_upsert, ensure_schema, setup_logging

HEROES_URL = "https://api.deadlock-api.com/v1/assets/heroes"
log = setup_logging()


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
    return [(i, name, cls, (t or "").strip().lower() or None, cx) for i, name, cls, t, cx in rows]


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
                             "hero_type = EXCLUDED.hero_type, complexity = EXCLUDED.complexity")
        missing_type = [r[1] for r in rows if r[3] is None]
        log.info("Synced %s heroes", len(rows))
        if missing_type:
            log.warning("No hero_type for: %s (they won't appear in archetype charts)", ", ".join(missing_type))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())