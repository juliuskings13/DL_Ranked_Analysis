"""
ingest_item_details.py — everything the assets API knows about every item, ability and weapon, with history.

ingest_game_assets.py keeps names, ids, tiers and costs. This keeps the rest (stats, wording, cooldowns,
the API's whole record) and a new version whenever a patch changes an item:

  item_versions   one row per version of an item: the main columns, the wording as plain text, and the API's
                  full record as JSONB. valid_from / valid_to say when it was current (valid_to NULL = now).
  item_stats      the item's stats in that version, one row per stat: label, value as shown ("14m") and as
                  a number (14), unit, tooltip section, whether the card shows it
  item_current    view: the current version of every item

A version is "changed" when the game data changes. Image links, icon links and the API's own timestamps
don't count, so a CDN move doesn't create versions. Items gone from the API get their version closed.
History starts with the first run: run it right after each patch (and weekly with the pipeline) so
valid_from lines up with the patch.

    -- how Extra Health's bonus health moved across patches
    SELECT v.valid_from, s.value_text
    FROM item_stats s JOIN item_versions v USING (item_id, valid_from)
    WHERE v.name = 'Extra Health' AND s.property = 'BonusHealth' ORDER BY v.valid_from;

Usage: python ingest_item_details.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from datetime import datetime, timezone

import requests
from psycopg2.extras import Json, execute_values

from pipeline_common import connect, ensure_schema, setup_logging

ITEMS_URL = "https://api.deadlock-api.com/v1/assets/items?language=english"
HEADERS = {"User-Agent": "Deadlock Analytics (personal project)"}
VOLATILE = ("image", "icon", "update_time")             # keys containing these are not game data
log = setup_logging()


def game_data(x):
    """The record without image / icon links and timestamps, and without <img> tags in the wording."""
    if isinstance(x, dict):
        return {k: game_data(v) for k, v in x.items() if not any(w in k.lower() for w in VOLATILE)}
    if isinstance(x, list):
        return [game_data(v) for v in x]
    if isinstance(x, str):
        return re.sub(r"<img[^>]*>", "", x)
    return x


def data_hash(it: dict) -> str:
    return hashlib.sha256(json.dumps(game_data(it), sort_keys=True).encode()).hexdigest()[:16]


def plain(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n")).strip()


def description(it: dict) -> str | None:
    """The item's wording as plain text: its description parts (desc, passive, active ...), or for abilities
    without one, their tooltip sections."""
    d = it.get("description")
    parts = [v for v in d.values() if isinstance(v, str) and v.strip()] if isinstance(d, dict) else \
        [d] if isinstance(d, str) and d.strip() else []
    if not parts:
        for sec in (it.get("tooltip_details") or {}).get("info_sections") or []:
            if isinstance(sec, dict) and isinstance(sec.get("loc_string"), str):
                parts.append(sec["loc_string"])
    text = "\n\n".join(plain(p) for p in parts)
    return text or None


def shown_properties(it: dict) -> set[str]:
    """Stats the item's tooltip lists (any *properties list inside tooltip_sections), plus a real cooldown."""
    shown = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.endswith("properties") and isinstance(v, list):
                    shown.update(str(p) for p in v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(it.get("tooltip_sections") or [])
    cd = (it.get("properties") or {}).get("AbilityCooldown") or {}
    if str(cd.get("value", "0")) not in ("0", "0.0"):
        shown.add("AbilityCooldown")
    return shown


def number(v) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def stats(it: dict) -> list[tuple]:
    """(property, label, value_text, value, unit, section, shown) for the stats worth keeping: everything the
    tooltip shows, and any other labelled stat that isn't zero / switched off."""
    shown = shown_properties(it)
    rows = []
    for prop, p in (it.get("properties") or {}).items():
        if not isinstance(p, dict) or "value" not in p:
            continue
        text = str(p["value"]).strip()
        off = text in ("", "0", "0.0", "-1", "-1.0") or text == str(p.get("disable_value"))
        if prop not in shown and (off or not p.get("label")):
            continue
        rows.append((prop[:80], (p.get("label") or None), text[:50], number(text), (p.get("postfix") or "").strip()[:20] or None,
                     p.get("tooltip_section"), prop in shown))
    return rows


def version_row(it: dict, h: str) -> dict:
    return {"class_name": it.get("class_name"), "name": it.get("name"), "type": it.get("type"),
            "slot_type": it.get("item_slot_type"), "tier": it.get("item_tier"), "cost": it.get("cost"),
            "shopable": it.get("shopable"), "activation": it.get("activation"), "description": description(it),
            "data_hash": h}


def main() -> int:
    ap = argparse.ArgumentParser(description="Item stats, wording and history from the Deadlock assets API")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()
    try:
        res = requests.get(ITEMS_URL, headers=HEADERS, timeout=60)
        res.raise_for_status()
        items = list({it["id"]: it for it in res.json() if it.get("id") is not None}.values())
    except requests.RequestException:
        log.exception("Could not reach the assets API")
        return 1
    run_at = datetime.now(timezone.utc)

    conn = connect()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT item_id, data_hash, name FROM item_versions WHERE valid_to IS NULL")
            current = {r[0]: (r[1].strip(), r[2]) for r in cur.fetchall()}
            seen, new, changed = set(), [], []
            for it in items:
                h = data_hash(it)
                seen.add(it["id"])
                if it["id"] not in current:
                    new.append((it, h))
                elif current[it["id"]][0] != h:
                    changed.append((it, h))
            gone = [i for i in current if i not in seen]
            log.info("%s items: %s new, %s changed, %s gone, %s unchanged", len(items), len(new), len(changed),
                     len(gone), len(items) - len(new) - len(changed))
            for it, _ in changed:
                log.info("  changed: %s (%s)", it.get("name") or it.get("class_name"), it.get("type"))
            for i in gone:
                log.info("  gone:    %s", current[i][1] or i)
            if args.dry_run:
                return 0

            closing = [it["id"] for it, _ in changed] + gone
            if closing:
                cur.execute("UPDATE item_versions SET valid_to = %s WHERE valid_to IS NULL AND item_id = ANY(%s)",
                            (run_at, closing))
            todo = new + changed
            cols = ["item_id", "valid_from", "class_name", "name", "type", "slot_type", "tier", "cost", "shopable",
                    "activation", "description", "data_hash", "data"]
            execute_values(cur, f"INSERT INTO item_versions ({', '.join(cols)}) VALUES %s",
                           [(it["id"], run_at, *version_row(it, h).values(), Json(it)) for it, h in todo],
                           page_size=200)
            stat_rows = [(it["id"], run_at, *s) for it, _ in todo for s in stats(it)]
            execute_values(cur, "INSERT INTO item_stats (item_id, valid_from, property, label, value_text, value, "
                                "unit, section, shown) VALUES %s", stat_rows, page_size=1000)
        conn.commit()                                     # all or nothing: versions, closes and stats together
        log.info("Wrote %s item versions and %s stats", len(todo), len(stat_rows))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
