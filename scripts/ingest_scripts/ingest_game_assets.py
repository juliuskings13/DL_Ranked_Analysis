"""
ingest_game_assets.py — items, hero abilities and their icons from the Deadlock assets API.

What it does (safe to re-run; only new or changed things are written / downloaded):
  1. items           every shop item, ability and weapon: id, name, class, type, slot, tier, cost,
                     shopable, owning hero (for abilities), image URL  -> Postgres `items`
  2. hero_abilities  each hero's four abilities in slot order (1-3, 4 = ultimate)
                     -> Postgres `hero_abilities`
  3. icons           shop item icons   -> assets/item_icons/<item>.png
                     ability icons     -> assets/ability_icons/<hero>/<slot>_<ability>.png
  4. game_names.json hero / ability (with slot) / item (with slot, tier) names for scripts that
                     don't touch the database, e.g. patch_notes.py -> assets/game_names.json

Overrides: if assets/ability_slots.csv exists (columns hero,slot,ability), its rows replace the
API's ability names / slots. Use it if a hero's slots ever come back wrong or missing.

Usage: python ingest_game_assets.py [--no-icons] [--force-icons] [--no-db]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from pipeline_common import connect, copy_upsert, ensure_schema, setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]       # scripts/ingest_scripts/ -> project root
ASSETS = PROJECT_ROOT / "assets"
# the old assets host (assets.deadlock-api.com/v2) was retired; same data now lives under /v1/assets
HEROES_URL = "https://api.deadlock-api.com/v1/assets/heroes?only_active=true"
ITEMS_URL = "https://api.deadlock-api.com/v1/assets/items?language=english"
HEADERS = {"User-Agent": "Deadlock Analytics (personal project)"}
log = setup_logging()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fetch(url: str):
    res = requests.get(url, headers=HEADERS, timeout=60)
    res.raise_for_status()
    return res.json()


def build(heroes: list[dict], items: list[dict]) -> dict:
    by_class = {it.get("class_name"): it for it in items if it.get("class_name")}
    hero_name = {h["id"]: h["name"] for h in heroes if h.get("id") is not None and h.get("name")}
    playable = [h for h in heroes if h.get("player_selectable", True) and not h.get("disabled")
                and h.get("id") in hero_name]

    abilities = {}                                       # hero -> [{slot, name, id, class_name, image}]
    for h in playable:
        slots = []
        for k in range(1, 5):
            cls = (h.get("items") or {}).get(f"signature{k}")
            it = by_class.get(cls)
            if it and it.get("name"):
                slots.append({"slot": k, "name": it["name"], "id": it.get("id"), "class_name": cls,
                              "image": it.get("image") or it.get("image_webp")})
        if slots:
            abilities[h["name"]] = slots

    override = ASSETS / "ability_slots.csv"
    if override.exists():
        with open(override, newline="") as f:
            for row in csv.DictReader(f):
                hero, slot, name = row.get("hero", "").strip(), int(row.get("slot") or 0), row.get("ability", "").strip()
                if hero and 1 <= slot <= 4 and name:
                    cur = [a for a in abilities.get(hero, []) if a["slot"] != slot]
                    old = next((a for a in abilities.get(hero, []) if a["slot"] == slot), {})
                    abilities[hero] = sorted(cur + [{**old, "slot": slot, "name": name}], key=lambda a: a["slot"])
        log.info("Applied ability overrides from %s", override)

    shop = {}
    for it in items:
        if it.get("type") == "upgrade" and it.get("shopable") and it.get("name"):
            shop[it["name"]] = {"id": it.get("id"), "slot": it.get("item_slot_type"), "tier": it.get("item_tier"),
                                "cost": it.get("cost"), "image": it.get("shop_image") or it.get("image")}
    return {"heroes": sorted(hero_name[h["id"]] for h in playable), "abilities": abilities, "items": shop,
            "hero_ids": {h["name"]: h["id"] for h in playable}}


def write_db(items: list[dict], names: dict) -> None:
    conn = connect()
    try:
        ensure_schema(conn)
        cols = ["id", "name", "tier", "type", "slot_type", "cost", "class_name", "shopable", "hero_id", "image_url"]
        rows = []
        for it in items:
            if it.get("id") is None:
                continue
            hero = it.get("hero") or ((it.get("heroes") or [None])[0] if it.get("type") != "upgrade" else None)
            rows.append((it["id"], it.get("name"), it.get("item_tier"), it.get("type"), it.get("item_slot_type"),
                         it.get("cost"), it.get("class_name"), it.get("shopable"), hero,
                         it.get("shop_image") or it.get("image")))
        n = copy_upsert(conn, "items", cols, rows, key=["id"],
                        conflict="DO UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}" for c in cols[1:]))
        ab_rows = [(names["hero_ids"][h], a["slot"], a.get("id"), a.get("class_name"), a["name"])
                   for h, slots in names["abilities"].items() for a in slots if h in names["hero_ids"]]
        m = copy_upsert(conn, "hero_abilities", ["hero_id", "slot", "ability_id", "class_name", "name"], ab_rows,
                        key=["hero_id", "slot"],
                        conflict="DO UPDATE SET ability_id = EXCLUDED.ability_id, class_name = EXCLUDED.class_name, "
                                 "name = EXCLUDED.name")
        log.info("Upserted %s items and %s hero abilities", n, m)
    finally:
        conn.close()


def download_icons(names: dict, force: bool) -> None:
    jobs = []
    for name, it in names["items"].items():
        if it.get("image"):
            dest = ASSETS / "item_icons" / f"{slug(name)}.png"
            it["icon"] = str(dest.relative_to(PROJECT_ROOT))
            jobs.append((it["image"], dest))
    for hero, slots in names["abilities"].items():
        for a in slots:
            if a.get("image"):
                dest = ASSETS / "ability_icons" / slug(hero) / f"{a['slot']}_{slug(a['name'])}.png"
                a["icon"] = str(dest.relative_to(PROJECT_ROOT))
                jobs.append((a["image"], dest))
    todo = [(u, d) for u, d in jobs if force or not d.exists()]
    if not todo:
        log.info("All %s icons already on disk", len(jobs))
        return

    def get(url: str, dest: Path) -> None:
        res = requests.get(url, headers=HEADERS, timeout=30)
        res.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".webp") or res.headers.get("content-type", "").endswith("webp"):
            from io import BytesIO
            from PIL import Image
            Image.open(BytesIO(res.content)).save(dest)
        else:
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(res.content)
            tmp.replace(dest)

    failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(get, u, d): d for u, d in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:                          # one missing image never stops the run
                failed += 1
                log.warning("icon %s: %s", futures[fut].name, e)
    log.info("Downloaded %s icons (%s failed, %s already on disk)", len(todo) - failed, failed, len(jobs) - len(todo))


def main() -> int:
    ap = argparse.ArgumentParser(description="Items, abilities and icons from the Deadlock assets API")
    ap.add_argument("--no-icons", action="store_true", help="skip downloading icons")
    ap.add_argument("--force-icons", action="store_true", help="re-download icons that already exist")
    ap.add_argument("--no-db", action="store_true", help="only write icons and game_names.json")
    args = ap.parse_args()
    try:
        heroes, items = fetch(HEROES_URL), fetch(ITEMS_URL)
    except requests.RequestException:
        log.exception("Could not reach the assets API")
        return 1
    names = build(heroes, items)
    missing = [h for h in names["heroes"] if len(names["abilities"].get(h, [])) < 4]
    if missing:
        log.warning("Fewer than 4 abilities found for: %s (add them to assets/ability_slots.csv)", ", ".join(missing))
    if not args.no_icons:
        download_icons(names, args.force_icons)
    if not args.no_db:
        write_db(items, names)
    out = ASSETS / "game_names.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(names, indent=1))
    log.info("%s heroes, %s shop items -> %s", len(names["heroes"]), len(names["items"]), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())