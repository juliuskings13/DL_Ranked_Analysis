"""
patch_notes.py — turn a Deadlock balance patch into buff / nerf / rework cards.

Two steps, so you can fix what the parser gets wrong before anything is drawn:

  1. parse   reads the patch notes and writes patches/<date>_<name>/changes.csv, one row per
             change line, with the parser's call (auto) and why it made it.
  2. cards   draws the cards from changes.csv, using your corrections where you made them.

Usage
    python patch_notes.py parse --latest                     # newest post in the API's patch feed
    python patch_notes.py parse --link https://store.steampowered.com/news/app/1422450/view/...
    python patch_notes.py parse --file notes.txt --name 10-30-2026   # notes pasted into a text file
    python patch_notes.py review patches/2026-10-30_10-30-2026       # list lines that need a look
    python patch_notes.py cards  patches/2026-10-30_10-30-2026
    python animate.py patches/2026-10-30_10-30-2026 --all            # animated versions

Cards (in this order for a video):
    patch_opener              thumbnail / opener: big hero icons under BUFFED / NERFED / REWORKED
    patch_heroes              per hero, number of buffs vs nerfs
    patch_heroes_size         per hero, how big the changes were (% changes added up)
    patch_hero_<hero>         every change for one hero, grouped "1: Ability" ... "4: Ultimate"
    patch_items / patch_items_size  items overview (count / size)
    patch_items_<weapon|vitality|spirit>[_N]   every item change of that type, two columns per card
    patch_general_N           map, economy and system changes

Names, ability slot numbers and icons come from assets/game_names.json, written by
scripts/ingest_scripts/ingest_game_assets.py (run it once, and after patches that add heroes or items).
To fix or add an ability number by hand, put hero,slot,ability rows in assets/ability_slots.csv.

Corrections (changes.csv, edit in any spreadsheet app; re-parsing keeps them, matched by the id column):
    auto      the call: buff / nerf / rework / change / fix. Overwrite it where the parser is wrong
              (the parser column keeps what the script said, so the two can be compared).
    change, subject, ability   fix typos or wrong matches; your text is kept on re-parse.
    weight    how big the change is (default 1). e.g. 3 for an ultimate cooldown change,
              0.5 for a tiny number tweak. The overview bars add these up.
    include   "no" drops the line from every card (mode-only changes start as "no").
    note      anything; shown nowhere, just for you.

How the parser decides (column `why`):
    numbers   "X from A to B" / "X increased by N%": the direction of the change, then whether more
              of that stat is good for the hero or item (cooldowns, costs, cast times and penalties are
              better lower; damage, health, duration, range, slows and resist shred are better higher;
              more gravity is a nerf unless it's put on enemies, e.g. Knockdown).
    wording   lines without numbers: "no longer grants" -> nerf, "now also applies" -> buff,
              "fixed" -> fix, "instead of" / "reworked" -> rework.
    unclear   nothing matched; it must be set by hand (cards show it as a gray "change").
    learned   you corrected the same (or a 90%+ similar) line in an earlier patch: your call is reused.

Learning: every correction (auto differs from parser) goes into patches/_learned.csv when you parse or
make cards. `python patch_notes.py lessons` lists the rules you override most, with examples: the
to-do list for improving the rules themselves.
    Lines with negative numbers or wording-only calls are marked confidence=low. Check those first.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme as T  # noqa: E402
from chart_stats import Insight  # noqa: E402

FEED_URL = "https://api.deadlock-api.com/v2/patches"
# the old assets host (assets.deadlock-api.com/v2) was retired; same data now lives under /v1/assets
HEROES_URL = "https://api.deadlock-api.com/v1/assets/heroes?only_active=true"
ITEMS_URL = "https://api.deadlock-api.com/v1/assets/items?language=english"
PATCH_ROOT = T.PROJECT_ROOT / "patches"
MODE_SUBJECTS = {"street brawl", "hero labs", "sandbox", "coop", "bot match", "brawl"}
# columns you may edit in changes.csv; re-parsing keeps your edits ("verdict" is the older override column)
EDITABLE = ["auto", "change", "subject", "ability", "weight", "include", "note", "verdict"]
CSV_FIRST = ["subject", "ability", "auto", "change", "from", "to", "why"]   # column order in changes.csv, for reviewing
# order used to read a changes.csv that lost its header row (e.g. saved without it from a spreadsheet)
CSV_ORDER = CSV_FIRST + ["id", "order", "section", "kind", "slot", "tier", "confidence", "verdict", "weight",
                         "include", "note", "parser"]
LEARNED = "_learned.csv"     # every correction you've made, across patches (in the patches folder)
VERDICTS = ["buff", "nerf", "rework", "change", "fix", "unclear"]

VERDICT_STYLE = {   # glyph, color, label
    "buff":    ("▲", T.GREEN, "BUFF"),
    "nerf":    ("▼", T.RED, "NERF"),
    "rework":  ("◆", T.GOLD, "REWORK"),
    "change":  ("●", "#9aa5b1", "CHANGE"),
    "unclear": ("●", "#9aa5b1", "CHANGE"),
    "fix":     ("✓", "#6b7480", "FIX"),
    "system":  ("●", "#9aa5b1", "SYSTEM"),
}
SLOT_COLORS = {"weapon": "#f0a43b", "vitality": "#5fcf6b", "spirit": "#b46cf0"}
# item cards: title and item names in the slot's color (same as the matching hero class colors)
SLOT_ACCENT = {"weapon": T.ARCHETYPE_COLORS["marksman"], "vitality": T.ARCHETYPE_COLORS["brawler"],
               "spirit": T.ARCHETYPE_COLORS["mystic"]}
ITEM_MIN_FONT = 21          # item cards never shrink lines below this (phone legibility); they split instead
ITEM_HEAD_SCALE = 1.75      # item name band, icon and name size vs. a normal group heading (icon ~2x)
ITEMS_PER_CARD_MAX = 8
ITEM_MAX_FONT = 27          # cards with few items grow their text up to this instead of leaving space empty

# stats where LESS is better for whoever owns them (compared on the raw, signed value)
LOWER_BETTER = [r"cool ?downs?", r"\bcd\b", r"cast (time|delay)", r"\bdelay\b", r"\bcosts?\b", r"\bprice\b",
                r"spin[- ]?up", r"reload time", r"wind[- ]?up", r"charge time", r"lockout",
                r"time to (charge|cast|activate|spawn)", r"(channel|recovery|initial|deploy|post ?cast|buildup|"
                r"brew|cycle|fuse|travel) times?", r"fuse timers?"]
# stats where a BIGGER magnitude is worse (compared on the absolute value)
MAG_WORSE = [r"penalty", r"self[- ]?damage", r"drawback", r"damage taken", r"health cost"]

UP_WORDS = r"increased|increase|raised|more|faster|higher|longer|extended|boosted|improved"
DOWN_WORDS = r"reduced|reduce|decreased|decrease|lowered|less|slower|lower|shorter|weaker"
NUM = r"(?<![\w.])[+-]?\d+(?:\.\d+)?"


# --- SOURCES -------------------------------------------------------------------------
def _get_json(url: str):
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_patch(link: str | None) -> dict:
    """{title, link, published_at, content} from the API's patch feed: the newest post, or the one at link."""
    raw = _get_json(FEED_URL)
    items = raw if isinstance(raw, list) else next((raw[k] for k in ("patches", "items", "data") if k in raw), [])
    pick = lambda d, *ks: next((d[k] for k in ks if d.get(k)), None)  # noqa: E731
    posts = [{"title": pick(it, "title", "name"), "link": pick(it, "link", "url", "guid"),
              "published_at": pick(it, "pub_date", "pubDate", "published_at", "date"),
              "content": pick(it, "content", "content_encoded", "description", "body")} for it in items]
    if link:
        posts = [p for p in posts if p["link"] == link]
        if not posts:
            raise SystemExit(f"{link} isn't in the patch feed (yet). Paste the notes into a file and use --file.")
    else:
        posts = [p for p in posts if p["content"] and re.search(r"(^|\n|>)\s*-\s*\w[^:\n]{0,40}:", p["content"])]
        if not posts:
            raise SystemExit("No patch with change lines in the feed. Paste the notes into a file and use --file.")
        posts.sort(key=lambda p: str(p["published_at"]), reverse=True)
    if not posts[0]["content"]:
        raise SystemExit(f"The feed has no text for {posts[0]['link']}. Paste the notes into a file and use --file.")
    return posts[0]


def html_to_lines(text: str) -> list[str]:
    """Patch post (Steam/forum HTML or plain text) -> list of text lines."""
    text = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</div>|</h\d>", "\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "\n- ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\\[", "[").replace("\\]", "]").replace("\r", "")
    return [ln.strip() for ln in text.split("\n")]


GAME_NAMES = T.ASSETS_BASE / "game_names.json"      # written by ingest_game_assets.py


def _normalize_names(raw: dict) -> dict:
    """Accepts game_names.json (abilities with slots/icons) or the older plain cache; returns
    {heroes, abilities: hero -> [names], slots: hero -> {name.lower(): slot},
     ability_icons: hero -> {name.lower(): path}, items: name -> slot, item_icons: name -> path}."""
    out = {"heroes": raw.get("heroes", []), "abilities": {}, "slots": {}, "ability_icons": {}, "items": {},
           "item_icons": {}}
    for hero, lst in (raw.get("abilities") or {}).items():
        for a in lst:
            if isinstance(a, str):
                out["abilities"].setdefault(hero, []).append(a)
                continue
            out["abilities"].setdefault(hero, []).append(a["name"])
            if a.get("slot"):
                out["slots"].setdefault(hero, {})[a["name"].lower()] = int(a["slot"])
            if a.get("icon"):
                out["ability_icons"].setdefault(hero, {})[a["name"].lower()] = str(T.PROJECT_ROOT / a["icon"])
    for name, it in (raw.get("items") or {}).items():
        if isinstance(it, dict):
            out["items"][name] = str(it.get("slot") or "").lower()
            if it.get("icon"):
                out["item_icons"][name] = str(T.PROJECT_ROOT / it["icon"])
        else:
            out["items"][name] = str(it or "").lower()
    override = next((p for p in (T.ASSETS_BASE / "ability_slots.csv", PATCH_ROOT / "ability_slots.csv") if p.exists()),
                    None)
    if override is not None:                     # hero,slot,ability rows you maintain by hand win
        for row in pd.read_csv(override, dtype=str).fillna("").itertuples():
            if row.hero and row.ability and str(row.slot).isdigit():
                out["abilities"].setdefault(row.hero, [])
                if row.ability not in out["abilities"][row.hero]:
                    out["abilities"][row.hero].append(row.ability)
                out["slots"].setdefault(row.hero, {})[row.ability.lower()] = int(row.slot)
    return out


def load_names(online: bool = True) -> dict:
    """Hero, ability (with slot number) and item names. Order of preference: assets/game_names.json
    (from ingest_game_assets.py), the assets API, the last cached copy, hero names from a week folder."""
    if GAME_NAMES.exists():
        return _normalize_names(json.loads(GAME_NAMES.read_text()))
    cache = PATCH_ROOT / "_names_cache.json"
    if online:
        try:
            heroes, items = _get_json(HEROES_URL), _get_json(ITEMS_URL)
            by_class = {it.get("class_name"): it for it in items}
            raw = {"heroes": sorted(h["name"] for h in heroes if h.get("name") and h.get("player_selectable", True)),
                   "abilities": {}, "items": {}}
            for h in heroes:
                for k in range(1, 5):
                    it = by_class.get((h.get("items") or {}).get(f"signature{k}"))
                    if it and it.get("name") and h.get("name"):
                        raw["abilities"].setdefault(h["name"], []).append({"slot": k, "name": it["name"]})
            for it in items:
                if it.get("type") == "upgrade" and it.get("shopable") and it.get("name"):
                    raw["items"][it["name"]] = {"slot": it.get("item_slot_type")}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw, indent=1))
            return _normalize_names(raw)
        except Exception as e:
            print(f"  (couldn't load names from the assets API: {e})")
    if cache.exists():
        return _normalize_names(json.loads(cache.read_text()))
    print("  (no item/ability names: run ingest_game_assets.py; hero detection only)")
    weeks = sorted(T.PROJECT_ROOT.glob("week*_analysis/hero_winrates.csv"))
    heroes = sorted(pd.read_csv(weeks[-1])["hero_name"].unique()) if weeks else []
    return _normalize_names({"heroes": heroes})


# --- PARSING -------------------------------------------------------------------------
def _match_name(text: str, names, prefix_ok: bool = False) -> str | None:
    t = text.strip().lower()
    for n in names:
        nl = n.lower()
        if t == nl or t == nl.removeprefix("the ") or "the " + t == nl:
            return n
    if prefix_ok and t:                       # "Nurse" -> "Nurse Harrow" when only one name starts that way
        hits = [n for n in names if n.lower().removeprefix("the ").startswith(t + " ")]
        if len(hits) == 1:
            return hits[0]
    return None


def _nums(s: str) -> list[float]:
    return [float(x) for x in re.findall(NUM, s)]


def judge(text: str) -> tuple[str, str, str, float | None, float | None]:
    """(verdict, why, confidence, from, to) for one change line (subject and ability already removed)."""
    t = text.strip()
    tl = t.lower()
    main = re.sub(r"\([^)]*\)", "", tl)          # parentheticals are explanations, not the change
    if re.search(r"^fixed\b|\bbugs?\b", main):
        return "fix", "wording: fixed", "high", None, None
    if re.search(r"\b(performance|warning effects?|to make it clear|fall through geometry|under the map)\b", main):
        return "fix", "wording: performance / clarity fix", "low", None, None
    if re.search(r"\b(ui|visual|indicator|sounds?|audio|animations?|tooltip|minimaps?|reticle|aim poses|vfx)\b",
                 main):
        return "change", "wording: visual / quality of life", "low", None, None
    if re.search(r"\brework|redesign|\breplaced\b|completely", main):
        return "rework", "wording: rework", "high", None, None
    if re.search(r"\b(gain\w*|bonus\w*|buffs?|stacks?|stolen|steal\w*)\b.*\b(is|are|gets?)? ?(also |now )?removed\b",
                 main):
        return "nerf", "wording: gains / bonus removed", "low", None, None
    if re.search(r"\brefund", main):
        return "buff", "wording: refund", "low", None, None
    if re.search(r"\b(now )?random\b", main):
        return "change", "wording: random", "low", None, None
    if re.search(r"instead of|\bchanged? (from|to)\b", main) and not re.search(rf"from\s*{NUM}", main):
        return "change", "wording: changed / instead of", "low", None, None
    lower_better = any(re.search(p, main) for p in LOWER_BETTER)
    mag_worse = any(re.search(p, main) for p in MAG_WORSE)
    if "gravity" in main:          # more gravity hurts whoever has it, unless it's put on enemies (e.g. Knockdown)
        up = bool(re.search(r"increase|more|higher|adds|heavier|now (also )?increases", main))
        on_enemy = bool(re.search(r"enem|target|debuff|victim", main))
        good = up if on_enemy else not up
        return ("buff" if good else "nerf"), "gravity: more is worse unless on enemies", "low", None, None

    # "X from A to B" (A and B may be ranges like 25-61 or 18m->54m, or quoted bundles: sum them)
    m = re.search(r"\bfrom\b(.+?)\bto\b(.+)$", main) or re.search(rf"({NUM}[^\d]*?)\bto\b(\s*{NUM}.*)$", main)
    if m:
        a_side, b_side = m.group(1), m.group(2)
        # stop B at the next clause ("... to 15 and charged shots from 44 to 48" -> both pairs)
        pairs = re.findall(r"\bfrom\b\s*(.+?)\s*\bto\b\s*(\"[^\"]*\"|\S+)", main)
        if len(pairs) > 1:
            a = sum(sum(_nums(p[0])) for p in pairs)
            b = sum(sum(_nums(p[1])) for p in pairs)
        else:
            a_n, b_n = _nums(a_side), _nums(b_side)
            n = min(len(a_n), len(b_n)) or 1
            a, b = (sum(a_n[:n]), sum(b_n[:n])) if a_n and b_n else (None, None)
        if a is not None and b is not None and a != b:
            raw_up = b > a
            mag_up = abs(b) > abs(a)
            negatives = a < 0 or b < 0
            if lower_better:
                good, why = not raw_up, "numbers: lower is better"
            elif mag_worse:
                good, why = not mag_up, "numbers: bigger is worse"
            else:
                good, why = mag_up, "numbers: higher is better"
            conf = "low" if negatives and not lower_better else "high"
            return ("buff" if good else "nerf"), why, conf, a, b

    lb_word = lower_better or re.search(r"\b(delay|time)\b", main)
    # strong wording first: these phrases decide the line even if it also says "increased" etc.
    if re.search(r"no longer builds from", main):            # loses the component's stats
        return "nerf", "wording: loses a component", "low", None, None
    if re.search(r"now builds from", main):
        return "change", "wording: recipe change", "low", None, None
    nl = re.search(rf"no longer (has|gives|grants|provides)\s*\"?({NUM})?", main)
    if nl:
        if lb_word:                                 # losing a delay is good, losing a reduction of it is bad
            neg = nl.group(2) is not None and float(nl.group(2)) < 0
            return ("nerf" if neg else "buff"), "wording: no longer has (time/delay)", "low", None, None
        return "nerf", "wording: no longer + benefit", "low", None, None
    if re.search(r"no longer (pauses?|interrupt\w*|cancels?|resets?|stops?|slows?|loses?|costs?|penali\w*|"
                 r"destroy\w*|consum\w*)", main):
        return "buff", "wording: no longer + drawback", "low", None, None
    if re.search(r"no longer (scales?|gets?|increases?|appl(y|ies)|deals?|benefits?|affects?|counts?|builds?|"
                 r"works?|stacks?|lifts?|gains?|triggers?|refresh(es)?)", main):
        return "nerf", "wording: no longer + benefit", "low", None, None
    if re.search(r"now stacks?\b", main):
        return "buff", "wording: now stacks", "low", None, None
    if re.search(r"now only|is now required|now requires|now ends|now gets interrupted|gets interrupted|"
                 r"now goes on cooldown|can only|half damage|diminishing|can no longer|no longer be", main):
        return "nerf", "wording: restriction", "low", None, None
    nh = re.search(rf"now (?:also )?(?:has|gives?|grants?|deals?|provides?)\s*({NUM})", main)
    if nh:
        positive = float(nh.group(1)) > 0
        good = (not positive) if lb_word else positive          # "-2s Charge Delay" is good
        return ("buff" if good else "nerf"), "wording: now has +/-", "low", None, None

    # "X increased by 20%" / "Buildup is 35% slower" / "reduced by ~20%"
    up = re.search(rf"\b({UP_WORDS})\b", main)
    down = re.search(rf"\b({DOWN_WORDS})\b", main)
    if (up or down) and not (up and down):
        went_up = bool(up)
        if lower_better:
            good, why = not went_up, "wording: lower is better"
        elif mag_worse:
            good, why = not went_up, "wording: bigger is worse"
        else:
            good, why = went_up, "wording: more is better"
        return ("buff" if good else "nerf"), why, "low", None, None

    if re.search(r"no longer", main):
        return "nerf", "wording: no longer", "low", None, None
    if re.search(r"(now (also )?(grants|gives|applies|scales|increases|works with|deals|counts|heals|pierces|"
                 r"bounces|can|has|returns|benefits?|stacks?|refunds|lets)|can now|now also)", main):
        return "buff", "wording: now also / can now", "low", None, None
    return "unclear", "nothing matched", "low", None, None


def parse_notes(lines: list[str], names: dict) -> pd.DataFrame:
    heroes = names["heroes"]
    items = names["items"]
    rows, section = [], ""
    entries: list[tuple[str, str]] = []
    for ln in lines:
        if not ln:
            continue
        if not re.search(r"[A-Za-z]", ln):
            continue
        sec = re.match(r"^\[\s*(.+?)\s*\]$", ln) or re.match(r"^([A-Za-z][\w &'/-]{1,30}):$", ln)
        if sec:
            section = sec.group(1).lower()
            continue
        if re.match(r"^[-*•]\s+", ln):
            entries.append((section, re.sub(r"^[-*•]\s+", "", ln)))
        elif entries:                                   # wrapped continuation of the previous line
            entries[-1] = (entries[-1][0], entries[-1][1] + " " + ln)
    for i, (section, line) in enumerate(entries):
        subject, rest, kind = "", line, "general"
        m = re.match(r"^([^:]{1,40}):\s*(.+)$", line)
        if m:
            cand = m.group(1).strip()
            hero = _match_name(cand, heroes, prefix_ok=True)
            if section in MODE_SUBJECTS:               # whole section is for another mode (e.g. Street Brawl)
                subject, rest, kind = (hero or cand), m.group(2), "mode"
            elif hero:
                subject, rest, kind = hero, m.group(2), "hero"
            elif cand.lower() in MODE_SUBJECTS:
                subject, rest, kind = cand, m.group(2), "mode"
            elif _match_name(cand, items) or "item" in section or (not items and "general" not in section):
                subject, rest, kind = (_match_name(cand, items) or cand), m.group(2), "item"
        if section in MODE_SUBJECTS:
            kind = "mode"
        ability, change = "", rest
        if kind == "hero":
            parts = re.split(r"\s+-\s+", rest, maxsplit=1)
            known = sorted(names["abilities"].get(subject, []), key=len, reverse=True)
            if len(parts) == 2 and len(parts[0]) <= 40 and not re.search(r"\bfrom\b|\bto\b|%", parts[0]):
                ability, change = parts
            else:
                hit = next((a for a in known if rest.lower().startswith(a.lower() + " ")), None)
                if hit:
                    ability, change = hit, rest[len(hit):].strip()
            known_hit = _match_name(ability, known) if ability else None
            ability = known_hit or ability
            if re.fullmatch(r"(?i)gun|weapon", ability or ""):
                ability = "Gun"
        tier = re.search(r"\bT([123])\b", change)
        verdict, why, conf, a, b = judge(change)
        if subject.lower() == "knockdown" and "gravity" in change.lower() and verdict in ("buff", "nerf"):
            verdict = "buff" if verdict == "nerf" else "nerf"      # Knockdown's gravity is its whole point
            why += " (flipped: Knockdown)"
        if kind == "hero" and verdict in ("buff", "nerf") and re.search(
                rf"\bagainst (the )?({'|'.join(re.escape(w) for w in subject.split() if len(w) > 2)}|him|her|them)\b",
                change, re.I):
            verdict, why, conf = ("nerf" if verdict == "buff" else "buff"), why + " (flipped: works against the hero)", "low"
        if kind in ("general", "mode") and verdict not in ("fix",):
            verdict, why = ("system", "general change") if kind == "general" else (verdict, why + " (mode only)")
        rows.append({"id": hashlib.sha1(line.encode()).hexdigest()[:10], "order": i, "section": section,
                     "kind": kind, "subject": subject, "slot": items.get(subject, "") if kind == "item" else "",
                     "ability": ability, "tier": f"T{tier.group(1)}" if tier else "", "change": change.strip(),
                     "from": a, "to": b, "auto": verdict, "why": why, "confidence": conf,
                     "weight": "", "include": "no" if kind == "mode" else "", "note": "", "parser": verdict})
    df = pd.DataFrame(rows)
    if len(df):
        tier_moves(df)
        apply_learned(df)
    if len(df):                     # "Close the Deal" / "Close The Deal": one spelling per hero ability
        first = {}
        for i, r in df.iterrows():
            k = (r["subject"], r["ability"].lower())
            df.at[i, "ability"] = first.setdefault(k, r["ability"])
    return df


TIER_LOST = re.compile(r"^(T[123]) no longer (?:has|gives|grants|provides) (.+)$", re.I)
TIER_GAINED = re.compile(r"^(T[123]) now (?:also )?(?:has|gives|grants|provides) (.+)$", re.I)


def _stat(text: str) -> str:
    """The effect a tier line talks about, numbers and signs removed ("-0.3s Time To Spawn" -> "time to spawn")."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", re.sub(NUM + r"[a-z%]*", " ", text.lower()))).strip()


def tier_moves(df: pd.DataFrame) -> None:
    """An effect moving between tiers is ONE change: the lower tier's line gets the call and the other becomes
    "change" (not counted twice). Moving to a higher tier costs more points: nerf. Moving to a lower tier:
    buff (low confidence if the value also changed size)."""
    for (subj, ab), g in df[df["kind"] == "hero"].groupby(["subject", "ability"]):
        lost = [(i, TIER_LOST.match(c)) for i, c in g["change"].items() if TIER_LOST.match(c)]
        gained = [(i, TIER_GAINED.match(c)) for i, c in g["change"].items() if TIER_GAINED.match(c)]
        for i, ml in lost:
            for j, mg in gained:
                if _stat(ml.group(2)) != _stat(mg.group(2)):
                    continue
                t_from, t_to = int(ml.group(1)[1]), int(mg.group(1)[1])
                if t_from == t_to:
                    continue
                a = [abs(float(x)) for x in re.findall(NUM, ml.group(2))]
                b = [abs(float(x)) for x in re.findall(NUM, mg.group(2))]
                shrank = bool(a and b and b[0] < a[0])
                grew = bool(a and b and b[0] > a[0])
                call = "nerf" if t_to > t_from else "buff"
                why = f"tier move T{t_from} -> T{t_to}" + (" (value also shrank: check)" if shrank else
                                                          " (value also grew: check)" if grew else "")
                # the call goes on the lower tier's line (the one players feel first), the other is "change"
                main_i, other_i = (i, j) if t_from < t_to else (j, i)
                df.at[main_i, "auto"] = df.at[main_i, "parser"] = call
                df.at[main_i, "why"] = why
                df.at[main_i, "confidence"] = "low" if (shrank or grew) else "high"
                df.at[other_i, "auto"] = df.at[other_i, "parser"] = "change"
                df.at[other_i, "why"] = f"tier move T{t_from} -> T{t_to} (counted on the other line)"


def patch_dir_for(title: str, published) -> Path:
    when = pd.to_datetime(published, utc=True, errors="coerce")
    day = (when if pd.notna(when) else pd.Timestamp.now(tz="UTC")).strftime("%Y-%m-%d")
    name = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "patch"
    return PATCH_ROOT / f"{day}_{name}"


def read_changes_csv(path: Path) -> pd.DataFrame:
    """changes.csv as strings. Copes with a file saved without its header row or without trailing
    empty columns (some spreadsheet apps do that), using the column order recorded at parse time."""
    path = Path(path)
    first = path.read_text().split("\n", 1)[0]
    if "id" in [c.strip() for c in first.split(",")]:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        meta = path.with_name("patch_meta.json")
        order = json.loads(meta.read_text()).get("columns", CSV_ORDER) if meta.exists() else CSV_ORDER
        df = pd.read_csv(path, dtype=str, keep_default_na=False, header=None)
        df.columns = list(order[:df.shape[1]]) + [f"extra_{i}" for i in range(df.shape[1] - len(order))]
    for c in CSV_ORDER:
        if c not in df.columns:
            df[c] = ""
    return df


def text_key(text: str) -> str:
    """A change line with numbers and punctuation removed, so the same wording matches across patches."""
    t = re.sub(NUM, "#", str(text).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z# ]+", " ", t)).strip()


def apply_learned(df: pd.DataFrame) -> None:
    """Calls you made in earlier reviews, reused for lines worded the same (or almost the same) way."""
    path = PATCH_ROOT / LEARNED
    if not path.exists():
        return
    import difflib
    learned = pd.read_csv(path, dtype=str, keep_default_na=False)
    if learned.empty:
        return
    by_key = {(r.kind, r.key): r for r in learned.itertuples()}
    for i, r in df.iterrows():
        if r["kind"] not in ("hero", "item"):
            continue
        k = text_key(r["change"])
        hit = by_key.get((r["kind"], k))
        how = "same wording"
        if hit is None:
            cands = learned[learned["kind"] == r["kind"]]
            best, score = None, 0.0
            for c in cands.itertuples():
                sc = difflib.SequenceMatcher(None, k, c.key).ratio()
                if sc > score:
                    best, score = c, sc
            if best is not None and score >= 0.9:
                hit, how = best, f"{score:.0%} similar"
        if hit is not None and hit.call != r["auto"]:
            df.at[i, "auto"] = hit.call
            df.at[i, "parser"] = hit.call
            df.at[i, "why"] = f"learned ({how}): {hit.change[:60]}"
            df.at[i, "confidence"] = "low"


def learn_from(patch_dir: Path) -> int:
    """Add this patch's corrections (rows where your call differs from the parser's) to the learned list."""
    path = Path(patch_dir) / "changes.csv"
    if not path.exists():
        return 0
    df = read_changes_csv(path)
    df["call"] = np.where(df["verdict"].str.strip() != "", df["verdict"].str.strip().str.lower(), df["auto"].str.lower())
    fixed = df[df["kind"].isin(["hero", "item"]) & (df["parser"] != "") & (df["call"] != df["parser"].str.lower())]
    rows = pd.DataFrame({"key": fixed["change"].map(text_key), "kind": fixed["kind"], "call": fixed["call"],
                         "parser": fixed["parser"], "rule": fixed["why"], "subject": fixed["subject"],
                         "change": fixed["change"], "patch": Path(patch_dir).name, "id": fixed["id"]})
    lp = PATCH_ROOT / LEARNED
    old = pd.read_csv(lp, dtype=str, keep_default_na=False) if lp.exists() else pd.DataFrame(columns=rows.columns)
    old = old[~((old["patch"] == Path(patch_dir).name) & ~old["id"].isin(rows["id"]))]   # un-corrected since
    allr = pd.concat([old[~old["id"].isin(rows["id"])], rows], ignore_index=True)
    lp.parent.mkdir(parents=True, exist_ok=True)
    allr.to_csv(lp, index=False)
    return len(rows)


def backup(path: Path, keep: int = 10) -> None:
    """Copy changes.csv to backups/changes_<time>.csv before it's rewritten; keeps the last `keep` copies."""
    import shutil
    bdir = path.parent / "backups"
    bdir.mkdir(exist_ok=True)
    shutil.copy2(path, bdir / f"changes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    for oldb in sorted(bdir.glob("changes_*.csv"))[:-keep]:
        oldb.unlink()


def write_changes(df: pd.DataFrame, out: Path, meta: dict) -> pd.DataFrame:
    """Write changes.csv, keeping your edits from an earlier parse of the same lines (matched by id)."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "changes.csv"
    if path.exists():
        learn_from(out)
        backup(path)
        old = read_changes_csv(path).drop_duplicates("id").set_index("id")
        new_ids = df["id"]
        for col in EDITABLE:
            if col not in old.columns or col not in df.columns:
                continue
            prev = new_ids.map(old[col])
            if col == "auto":
                # your call wins if you changed it from what the parser said back then (older files have no
                # parser column: then any difference from today's parse counts as your edit)
                prev_parser = new_ids.map(old["parser"]) if "parser" in old.columns else pd.Series("", index=df.index)
                legacy = prev_parser.fillna("") == ""
                edited = prev.notna() & (prev != "") & np.where(legacy, prev != df["auto"], prev != prev_parser)
                df["auto"] = np.where(edited, prev, df["auto"])
            else:
                df[col] = np.where(prev.notna() & (prev != "") & (prev != df[col].astype(str)), prev, df[col])
        if "verdict" in df.columns:                  # fold the old override column into auto
            v = df["verdict"].fillna("").str.strip()
            df["auto"] = np.where(v != "", v.str.lower(), df["auto"])
            df = df.drop(columns="verdict")
    first = [c for c in CSV_FIRST if c in df.columns]
    cols = first + [c for c in df.columns if c not in first]
    df[cols].to_csv(path, index=False)
    (out / "patch_meta.json").write_text(json.dumps({**meta, "columns": cols}, indent=2, default=str))
    learn_from(out)
    return df


# --- LOADING FOR CARDS --------------------------------------------------------------
def load_changes(patch_dir: Path) -> tuple[pd.DataFrame, dict]:
    path = Path(patch_dir) / "changes.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = read_changes_csv(path)
    meta = json.loads((Path(patch_dir) / "patch_meta.json").read_text()) if (Path(patch_dir) / "patch_meta.json").exists() else {}
    df = df[df["include"].str.lower() != "no"].copy()
    df["final"] = np.where(df["verdict"].str.strip() != "", df["verdict"].str.strip().str.lower(),
                           df["auto"].str.strip().str.lower())
    df["w"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    df["order"] = pd.to_numeric(df["order"], errors="coerce")
    df["pct"] = [change_pct(r) for _, r in df.iterrows()]
    return df.sort_values("order"), meta


PCT_CAP = 50.0          # one change counts at most this many % toward a hero's total (0.04 -> 0.01 is "75%")


def change_pct(r) -> float:
    """Size of one change in %: |to - from| / |from|, or the "by N%" in the text. NaN when the line has no numbers."""
    a, b = pd.to_numeric(r.get("from"), errors="coerce"), pd.to_numeric(r.get("to"), errors="coerce")
    if pd.notna(a) and pd.notna(b) and a != b:
        pct = abs(b - a) / abs(a) * 100 if a != 0 else PCT_CAP
    else:
        m = re.search(r"\bby\s*~?(\d+(?:\.\d+)?)\s*%", str(r.get("change", "")), re.I)
        if not m:
            return float("nan")
        pct = float(m.group(1))
    return min(pct, PCT_CAP)


def subject_summary(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Per hero (or item): weighted buff and nerf totals, rework flag, net, and the overall call."""
    d = df[df["kind"] == kind]
    if d.empty:
        return pd.DataFrame(columns=["subject", "buff", "nerf", "other", "rework", "net", "call", "slot"])
    g = d.groupby("subject", sort=False)
    s = pd.DataFrame({
        "buff": g.apply(lambda x: x.loc[x["final"] == "buff", "w"].sum()),
        "nerf": g.apply(lambda x: x.loc[x["final"] == "nerf", "w"].sum()),
        "other": g.apply(lambda x: x.loc[x["final"].isin(["change", "unclear", "fix"]), "w"].sum()),
        "rework": g.apply(lambda x: (x["final"] == "rework").any()),
        "slot": g["slot"].first(),
        # size = sum of each change's % (times its weight); n_sized = how many changes had numbers
        "buff_size": g.apply(lambda x: (x.loc[x["final"] == "buff", "pct"] * x.loc[x["final"] == "buff", "w"]).sum()),
        "nerf_size": g.apply(lambda x: (x.loc[x["final"] == "nerf", "pct"] * x.loc[x["final"] == "nerf", "w"]).sum()),
        "buff_sized": g.apply(lambda x: x.loc[x["final"] == "buff", "pct"].notna().sum()),
        "nerf_sized": g.apply(lambda x: x.loc[x["final"] == "nerf", "pct"].notna().sum()),
    }).reset_index()
    s["net"] = s["buff"] - s["nerf"]
    s["net_size"] = s["buff_size"] - s["nerf_size"]
    s["buff_avg"] = s["buff_size"] / s["buff_sized"].replace(0, np.nan)
    s["nerf_avg"] = s["nerf_size"] / s["nerf_sized"].replace(0, np.nan)
    s["call"] = np.select(
        [s["rework"], (s["buff"] > 0) & (s["nerf"] == 0), (s["nerf"] > 0) & (s["buff"] == 0),
         (s["buff"] > 0) & (s["nerf"] > 0)],
        ["REWORKED", "BUFFED", "NERFED", np.where(s["net"] > 0, "MIXED, MOSTLY BUFFED",
                                                  np.where(s["net"] < 0, "MIXED, MOSTLY NERFED", "MIXED"))],
        "CHANGED")
    return s.sort_values(["net", "buff"], ascending=[False, False]).reset_index(drop=True)


# --- CARDS -------------------------------------------------------------------------
def _meta(meta: dict) -> T.ChartMeta:
    label = meta.get("title") or "Patch"
    return T.ChartMeta(f"Patch {label}", None, source="Deadlock patch notes")


def _blank_card():
    fig = T.plt.figure(figsize=(T.CARD_IN, T.CARD_IN), dpi=T.DPI, facecolor=T.BG)
    return fig


def _img(path: str | None, size: int, alpha: float = 1.0) -> np.ndarray | None:
    """An icon file as an RGBA array, size x size px (aspect kept, centered), faded by alpha."""
    if not path or not Path(path).exists():
        return None
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    arr = np.asarray(canvas).astype(np.float32) / 255
    arr[..., 3] *= alpha
    return arr


def render_overview(summary: pd.DataFrame, kind: str, meta: dict, progress: float = 1.0,
                    highlight: set[str] | None = None, measure: str = "count",
                    icons: dict | None = None) -> "T.Figure":
    """Every hero (or item) touched by the patch: nerfs as a red bar to the left, buffs green to the right.
    measure="count": bar = number of changes (or their weights). measure="size": bar = total % change."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    p_bars = T.ease_out_cubic(T.phase(progress, 0.15, 0.85))
    size = measure == "size"
    bcol, ncol = ("buff_size", "nerf_size") if size else ("buff", "nerf")
    s = summary.copy()
    if size:
        s = s.sort_values(["net_size", "buff_size"], ascending=[False, False]).reset_index(drop=True)
    nb, nn = int((s["call"] == "BUFFED").sum()), int((s["call"] == "NERFED").sum())
    what = "HEROES" if kind == "hero" else "ITEMS"
    sub = (f"{len(s)} {what.lower()} changed · {nb} buffed · {nn} nerfed · "
           f"{len(s) - nb - nn} mixed or reworked")
    fig, ax = T.new_table_card()
    T.draw_header(fig, f"{what}: HOW BIG WERE THE CHANGES" if size else f"{what}: BUFFS & NERFS", sub, alpha=p_frame)
    T.draw_footer(fig, _meta(meta), alpha=p_frame)
    ncols = 1 if len(s) <= 15 else 2
    per_col = int(np.ceil(len(s) / ncols)) or 1
    row_px = T.table_rows(fig, ax, per_col, header_rows=0.9, footer_rows=0.55)
    vmax = max(float(s[[bcol, ncol]].to_numpy().max()) if len(s) else 1, 1)
    col_w = 1 / ncols
    icon_px = int(np.clip(row_px * 0.78, 22, 54))
    font = T.fit_font_px(s["subject"], col_w * 1000 * (0.30 if ncols == 2 else 0.24), row_px * 0.5,
                         max_px=22, min_px=11)
    for c in range(ncols):
        x0 = c * col_w + (0.012 if c else 0)
        name_x = x0 + (icon_px + 14) / 1000 if kind == "hero" else x0 + 0.02
        bar_c = x0 + col_w * (0.66 if ncols == 2 else 0.62)
        half = col_w * (0.26 if ncols == 2 else 0.30)
        ax.text(bar_c - half, -0.75, "◀ NERFS", ha="left", va="center", color=T.RED, fontsize=T.px_to_pt(13),
                fontweight="bold", alpha=p_frame)
        ax.text(bar_c + half, -0.75, "BUFFS ▶", ha="right", va="center", color=T.GREEN, fontsize=T.px_to_pt(13),
                fontweight="bold", alpha=p_frame)
        ax.plot([bar_c, bar_c], [-0.45, per_col - 0.5], color=T.DIVIDER, lw=1.2, alpha=p_frame)
    for i, r in s.iterrows():
        c, row = divmod(i, per_col)
        x0 = c * col_w + (0.012 if c else 0)
        bar_c = x0 + col_w * (0.66 if ncols == 2 else 0.62)
        half = col_w * (0.26 if ncols == 2 else 0.30)
        hl = highlight is None or r["subject"].lower() in highlight
        a = 1.0 if hl else 0.25
        t = T.ease_out_cubic(T.phase(p_bars, i / (len(s) + 2), i / (len(s) + 2) + 0.35))
        if kind == "hero":
            T.add_icon(ax, T.icon_any(r["subject"], icon_px, None, a * p_frame), (x0 + icon_px / 2 / 1000 + 0.004, row))
            name_x = x0 + (icon_px + 12) / 1000
        else:
            ic = _img((icons or {}).get(r["subject"]), icon_px, a * p_frame)
            if ic is not None:
                T.add_icon(ax, ic, (x0 + icon_px / 2 / 1000 + 0.004, row))
                name_x = x0 + (icon_px + 12) / 1000
            else:
                ax.scatter([x0 + 0.008], [row], s=60, color=SLOT_COLORS.get(r["slot"], T.NEUTRAL), alpha=a * p_frame)
                name_x = x0 + 0.022
        ax.text(name_x, row, r["subject"], ha="left", va="center", color=T.TEXT, alpha=a * p_frame,
                fontsize=T.px_to_pt(font), fontweight="bold")
        bh = 0.56
        lab = (lambda v: f"{v:.0f}%") if size else (lambda v: f"{v:g}")  # noqa: E731
        if r[ncol] > 0:
            w = half * r[ncol] / vmax * t
            ax.add_patch(T.plt.Rectangle((bar_c - w, row - bh / 2), w, bh, color=T.RED, alpha=0.9 * a, lw=0))
            ax.text(bar_c - w - 0.006, row, lab(r[ncol]), ha="right", va="center", color=T.RED,
                    fontsize=T.px_to_pt(font * 0.85), fontweight="bold", alpha=a * t)
        if r[bcol] > 0:
            w = half * r[bcol] / vmax * t
            ax.add_patch(T.plt.Rectangle((bar_c, row - bh / 2), w, bh, color=T.GREEN, alpha=0.9 * a, lw=0))
            ax.text(bar_c + w + 0.006, row, lab(r[bcol]), ha="left", va="center", color=T.GREEN,
                    fontsize=T.px_to_pt(font * 0.85), fontweight="bold", alpha=a * t)
        if r["rework"]:
            ax.text(bar_c, row, "◆", ha="center", va="center", color=T.GOLD, fontsize=T.px_to_pt(font * 1.1),
                    alpha=a * t, zorder=6)
    note = (f"bar = all the % changes added up (each capped at {PCT_CAP:.0f}%; lines without numbers aren't "
            "counted)" if size else "bar length = number of changes (or their weights, if set)")
    ax.text(0, per_col - 0.5 + 0.45, note + "  ·  ◆ = rework" +
            ("  ·  dot = item slot (weapon / vitality / spirit)" if kind == "item" and not icons else ""),
            ha="left", va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(13), alpha=p_frame)
    return fig


def opener_groups(summary: pd.DataFrame) -> list[tuple[str, str, list[str]]]:
    """(label, color, heroes) for the opener: buffed (net > 0), nerfed (net < 0), reworked (+ even mixes).
    Heroes that only got bug fixes are left out. Biggest changes first."""
    s = summary[(summary["buff"] + summary["nerf"] > 0) | summary["rework"]].copy()
    s["mag"] = s["net"].abs()
    rew = s[s["rework"]]
    rest = s[~s["rework"]]
    even = rest[rest["net"] == 0]
    out = [("BUFFED", T.GREEN, list(rest[rest["net"] > 0].sort_values("mag", ascending=False)["subject"])),
           ("NERFED", T.RED, list(rest[rest["net"] < 0].sort_values("mag", ascending=False)["subject"]))]
    third = list(rew["subject"]) + list(even["subject"])
    if third:
        out.append(("REWORKED" if even.empty else ("MIXED" if rew.empty else "REWORKED & MIXED"), T.GOLD, third))
    return [g for g in out if g[2]]


CLOSER_TOP = 3              # closer card: this many heroes and items on each side


def closer_picks(summary: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Biggest winners and losers: ranked by the net size of their changes (buff % minus nerf %, the
    patch_*_size cards), with the net number of changes as the tie-breaker."""
    s = summary.copy()
    s["score"] = s["net_size"].fillna(0) + s["net"] * 1e-3
    return (s[s["net"] > 0].nlargest(n, "score"), s[s["net"] < 0].nsmallest(n, "score"))


def render_closer(heroes: pd.DataFrame, items: pd.DataFrame, meta: dict, item_icons: dict, progress: float = 1.0,
                  highlight: set[str] | None = None) -> "T.Figure":
    """Closing card: the patch's biggest winners and losers, heroes on top, items below."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.12))
    fig = _blank_card()
    T.draw_header(fig, "WINNERS & LOSERS", f"Patch {meta.get('title', '')} · biggest buffs and nerfs by size",
                  alpha=p_frame)
    T.draw_footer(fig, _meta(meta), alpha=p_frame)
    top = (1 - fig._content_top) * T.CARD_PX + 20
    bottom = (1 - T.FOOTER_LINE_Y) * T.CARD_PX - 14
    x0 = T.MARGIN_X * T.CARD_PX
    col_w = (T.CARD_PX - 2 * x0 - 30) / 2
    sections = [("HEROES", heroes, True)] + ([("ITEMS", items, False)] if len(items) else [])
    sec_h = (bottom - top) / len(sections)
    k, n_all = 0, sum(min(len(d), 2 * CLOSER_TOP) for _, d, _ in sections) or 1
    for si, (label, summ, is_hero) in enumerate(sections):
        if summ is None or summ.empty:
            continue
        wins, losses = closer_picks(summ, CLOSER_TOP)
        y0 = top + si * sec_h
        fig.text(x0 / T.CARD_PX, 1 - y0 / T.CARD_PX, label, color=T.GOLD, fontsize=T.px_to_pt(24),
                 fontweight="bold", va="top", ha="left", alpha=p_frame)
        row_h = (sec_h - 82) / CLOSER_TOP
        icon = int(min(row_h * 0.82, 104))
        for c, (head, rows, color, glyph) in enumerate((("BIGGEST BUFFS", wins, T.GREEN, "▲"),
                                                         ("BIGGEST NERFS", losses, T.RED, "▼"))):
            cx = x0 + c * (col_w + 30)
            fig.text(cx / T.CARD_PX, 1 - (y0 + 40) / T.CARD_PX, f"{glyph} {head}", color=color,
                     fontsize=T.px_to_pt(19), fontweight="bold", va="top", ha="left", alpha=p_frame)
            for j, (_, r) in enumerate(rows.iterrows()):
                t = T.ease_out_cubic(T.phase(progress, 0.12 + 0.8 * k / n_all, 0.12 + 0.8 * k / n_all + 0.15))
                k += 1
                hl = highlight is None or r["subject"].lower() in highlight
                a = t * (1.0 if hl else 0.3)
                yy = y0 + 76 + j * row_h
                arr = T.icon_any(r["subject"], icon, color, a) if is_hero else _img(item_icons.get(r["subject"]), icon, a)
                if arr is not None:
                    fig.figimage(arr if arr.dtype != np.float32 else arr, xo=int(cx), yo=int(T.CARD_PX - yy - icon),
                                 origin="upper", zorder=3)
                tx = cx + icon + 14
                name_px = T.fit_font_px([r["subject"]], col_w - icon - 18, max_px=26, min_px=14)
                fig.text(tx / T.CARD_PX, 1 - (yy + icon * 0.32) / T.CARD_PX, r["subject"], color=T.TEXT,
                         fontsize=T.px_to_pt(name_px), fontweight="bold", va="center", ha="left", alpha=a)
                bits = []
                if r["buff"]:
                    bits.append(f"{r['buff']:g} buff" + ("s" if r["buff"] != 1 else ""))
                if r["nerf"]:
                    bits.append(f"{r['nerf']:g} nerf" + ("s" if r["nerf"] != 1 else ""))
                size = r["net_size"]
                sub = " · ".join(bits) + (f" · net {size:+.0f}%" if pd.notna(size) and size else "")
                fig.text(tx / T.CARD_PX, 1 - (yy + icon * 0.72) / T.CARD_PX, sub, color=color,
                         fontsize=T.px_to_pt(16), va="center", ha="left", alpha=a)
    return fig


def render_opener(summary: pd.DataFrame, meta: dict, types: dict, progress: float = 1.0,
                  highlight: set[str] | None = None) -> "T.Figure":
    """Thumbnail / opener: just the hero icons, big, sorted into BUFFED / NERFED / REWORKED."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    p_icons = T.ease_out_cubic(T.phase(progress, 0.1, 0.9))
    groups = opener_groups(summary)
    fig = _blank_card()
    title = f"PATCH {meta.get('title', '')}".upper().replace(" GAMEPLAY UPDATE", "")
    tpx = T.fit_font_px([title], T.CARD_PX * (1 - 2 * T.MARGIN_X), max_px=62, min_px=34)
    t = fig.text(T.MARGIN_X, 0.945, title, color=T.GOLD, fontsize=T.px_to_pt(tpx), fontweight="bold",
                 va="center", ha="left", alpha=p_frame)
    T._mark_header_bottom(fig, t)
    T.draw_footer(fig, _meta(meta), alpha=p_frame)
    top = (1 - fig._content_top) * T.CARD_PX + 26
    bottom = (1 - T.FOOTER_LINE_Y) * T.CARD_PX - 16
    width = T.CARD_PX * (1 - 2 * T.MARGIN_X)
    head_h, gap = 52, 12
    for size in range(170, 39, -2):                    # biggest icons that fit every section
        per_row = max(1, int((width + gap) // (size + gap)))
        need = sum(head_h + int(np.ceil(len(g[2]) / per_row)) * (size + gap) + 18 for g in groups)
        if need <= bottom - top:
            break
    y = top
    k, n_all = 0, sum(len(g[2]) for g in groups)
    for label, color, heroes in groups:
        fig.text(T.MARGIN_X, 1 - y / T.CARD_PX, f"{label}  {len(heroes)}", color=color, fontsize=T.px_to_pt(34),
                 fontweight="bold", va="top", ha="left", alpha=p_frame)
        y += head_h
        for i, h in enumerate(heroes):
            r, c = divmod(i, per_row)
            x = T.MARGIN_X * T.CARD_PX + c * (size + gap)
            yy = y + r * (size + gap)
            tt = T.ease_out_cubic(T.phase(p_icons, k / (n_all + 3), k / (n_all + 3) + 0.25))
            k += 1
            hl = highlight is None or h.lower() in highlight
            a = tt * (1.0 if hl else 0.25)
            if a <= 0:
                continue
            ax = fig.add_axes([x / T.CARD_PX, 1 - (yy + size) / T.CARD_PX, size / T.CARD_PX, size / T.CARD_PX])
            ax._no_fit = True
            ax.imshow(T.icon_any(h, size, color, a))
            ax.axis("off")
        y += int(np.ceil(len(heroes) / per_row)) * (size + gap) + 18
    return fig


def _wrap(text: str, size_px: float, width_px: float) -> list[str]:
    """Word-wrap to width_px. Widths are measured once per word (cached) and added up, so laying out
    and paginating a long patch stays fast."""
    space = T._text_w_per_px("a a", False) - T._text_w_per_px("aa", False)
    lines, cur, cur_w = [], [], 0.0
    for w in text.split():
        ww = T._text_w_per_px(w, False)
        add = ww + (space if cur else 0.0)
        if cur and (cur_w + add) * size_px > width_px:
            lines.append(" ".join(cur))
            cur, cur_w = [w], ww
        else:
            cur.append(w)
            cur_w += add
    return lines + ([" ".join(cur)] if cur else [])


def _layout_list(groups: list[tuple[str, list[dict]]], width_px: float, height_px: float, min_font: float = 13,
                 max_font: float = 24, head_scale: float = 1.0):
    """Pick the largest font at which all groups fit; returns (font_px, line plan, fits)."""
    for font in np.arange(max_font, min_font - 0.01, -0.5):
        plan, h = [], 0.0
        for gname, rows in groups:
            if gname:
                h += font * 1.55 * head_scale
                plan.append(("group", gname, None, font))
            for r in rows:
                lines = _wrap(r["text"], font, width_px - font * 1.6)
                plan.append(("row", lines, r, font))
                h += font * 1.32 * len(lines) + font * 0.45
        if h <= height_px:
            return font, plan, True
    return min_font, plan, False


ROW_SECONDS = 0.7           # text cards: each line gets this long on screen before the next one starts ...
ROW_FADE_S = 0.35           # ... of which this much is its fade-in (the rest is the pause before the next line)
INTRO_S = 0.6               # title, icon and footer fade in first


def list_build_s(groups) -> float:
    """Suggested build-in time for a text card: an intro, then ~1s per line."""
    return INTRO_S + ROW_SECONDS * sum(len(rows) for _, rows in groups)


HERO_TITLE_PX = 63          # hero change cards: 50% bigger than the standard 42px title
HERO_ICON_PX = 165          # and 50% bigger icon (standard 110)


def _layout_columns(groups, col_w: float, height_px: float, ncols: int, min_font: float = 13,
                    head_scale: float = 1.0, max_font: float = 24):
    """Like _layout_list, but whole groups (never split) are spread over ncols columns of about equal
    height. Returns (font, [plan per column], fits)."""
    for font in np.arange(max_font, min_font - 0.01, -0.5):
        blocks = []
        for g in groups:
            _, plan, _ = _layout_list([g], col_w, 1e9, font, max_font=font)
            gh = sum((font * 1.55 * head_scale) if k == "group" else (font * 1.32 * len(b) + font * 0.45) for k, b, _, _ in plan)
            blocks.append((gh, plan))
        target = sum(h for h, _ in blocks) / ncols
        cols, heights = [[]], [0.0]
        for gh, plan in blocks:
            # next column once this one is about its share of the total
            if heights[-1] > 0 and len(cols) < ncols and heights[-1] + gh / 2 > target:
                cols.append([])
                heights.append(0.0)
            cols[-1].extend(plan)
            heights[-1] += gh
        if max(heights) <= height_px:
            return font, cols, True
    return min_font, cols, False


def render_list(title: str, sub: str, groups: list[tuple[str, list[dict]]], meta: dict, progress: float = 1.0,
                highlight: set[str] | None = None, icon: str | None = None, ring: str | None = None,
                badge: tuple[str, str] | None = None, group_icons: dict | None = None,
                ncols: int = 1, head_scale: float = 1.0, accent: str | None = None,
                min_font: float | None = None, max_font: float = 24) -> "T.Figure":
    """A card listing changes: optional group headings (abilities, items), each line with a colored marker.
    With icon= (hero cards) the title and icon are drawn large. head_scale > 1 draws bigger group headings
    and icons (item cards); accent colors the title and group headings (item slot colors); min_font is the
    smallest allowed line font."""
    total_s = list_build_s(groups)
    p_frame = T.ease_out_cubic(T.phase(progress, 0, INTRO_S / total_s))
    head, slot = INTRO_S / total_s, ROW_SECONDS / total_s       # progress units: intro, then one slot per line
    row_t = lambda k: T.ease_out_cubic(T.phase(progress, head + k * slot,  # noqa: E731
                                               head + k * slot + slot * ROW_FADE_S / ROW_SECONDS))
    fig = _blank_card()
    T.draw_footer(fig, _meta(meta), alpha=p_frame)
    if icon:
        size = HERO_ICON_PX
        iax = fig.add_axes([1 - T.MARGIN_X - size / T.CARD_PX, 1 - 12 / T.CARD_PX - size / T.CARD_PX,
                            size / T.CARD_PX, size / T.CARD_PX])
        iax._no_fit = True
        iax.imshow(T.icon_any(icon, size, ring, p_frame))
        iax.axis("off")
        room = T.CARD_PX * (1 - 2 * T.MARGIN_X) - size - 24
        tpx = T.fit_font_px([title], room, max_px=HERO_TITLE_PX, min_px=34)
        t = fig.text(T.MARGIN_X, 1 - 52 / T.CARD_PX, title, color=T.GOLD, fontsize=T.px_to_pt(tpx),
                     fontweight="bold", va="center", ha="left", alpha=p_frame)
        st = fig.text(T.MARGIN_X, 1 - 106 / T.CARD_PX, sub, color=T.TEXT_MUTED, fontsize=T.px_to_pt(T.SIZE_SUBTITLE),
                      va="center", ha="left", alpha=p_frame)
        T._mark_header_bottom(fig, t)
        T._mark_header_bottom(fig, st)
    else:
        T.draw_header(fig, title, sub, alpha=p_frame)
        if accent:
            for t in fig.texts:
                if t.get_text() == title:
                    t.set_color(accent)
    top_px = (1 - getattr(fig, "_content_top", 0.88) + 0.03) * T.CARD_PX
    if badge:
        label, color = badge
        fig.text(T.MARGIN_X, 1 - top_px / T.CARD_PX, label, color=color, fontsize=T.px_to_pt(30), fontweight="bold",
                 va="top", ha="left", alpha=p_frame)
        top_px += 52
    if icon:
        top_px = max(top_px, 12 + HERO_ICON_PX + 14)      # list starts below the big icon
    bottom_px = (1 - T.FOOTER_LINE_Y) * T.CARD_PX - 20
    x0_px = T.MARGIN_X * T.CARD_PX
    width_px = T.CARD_PX - 2 * x0_px
    if os.environ.get("PATCH_DEBUG"):
        print(f"  list area {title[:20]}: {bottom_px - top_px:.0f}px")
    if ncols > 1:
        col_gap = 36
        col_w = (width_px - col_gap * (ncols - 1)) / ncols
        font, col_plans, fits = _layout_columns(groups, col_w, bottom_px - top_px, ncols,
                                                (min_font or MIN_LIST_FONT) - 2, head_scale, max_font)
        starts = [(x0_px + c * (col_w + col_gap), cp) for c, cp in enumerate(col_plans)]
        width_px = col_w
    else:
        font, plan, fits = _layout_list(groups, width_px, bottom_px - top_px, (min_font or MIN_LIST_FONT) - 2,
                                        head_scale=head_scale)
        starts = [(x0_px, plan)]
    n_rows = sum(1 for _, cp in starts for p in cp if p[0] == "row")
    k = 0
    for x0_px, plan in starts:
        y = top_px
        for kind, body, r, f in plan:
            if kind == "group":
                gh = f * 1.55 * head_scale                    # heading band height
                gx = x0_px
                isz = int(f * 1.35 * head_scale)
                gi = _img((group_icons or {}).get(body), isz, row_t(k))
                if gi is not None:
                    fig.figimage(gi, xo=int(gx), yo=int(T.CARD_PX - y - (gh + isz) / 2 + f * 0.1),
                                 origin="upper", zorder=3)
                    gx += isz + f * 0.35
                hpx = f * (0.92 if head_scale == 1 else 1.15)
                hpx = min(hpx, T.fit_font_px([body.upper()], width_px - (gx - x0_px) - 4, max_px=hpx, min_px=10))
                fig.text(gx / T.CARD_PX, 1 - (y + gh / 2 - f * 0.1) / T.CARD_PX, body.upper(), color=accent or T.GOLD,
                         fontsize=T.px_to_pt(hpx), fontweight="bold",
                         va="center", ha="left", alpha=row_t(k))
                y += gh
                continue
            t = row_t(k)                 # one line at a time, each held before the next appears
            k += 1
            hl = highlight is None or (r.get("key") or "").lower() in highlight
            a = t * (1.0 if hl else 0.3)
            glyph, color, _ = VERDICT_STYLE.get(r["final"], VERDICT_STYLE["change"])
            fig.text((x0_px + f * 0.15) / T.CARD_PX, 1 - y / T.CARD_PX, glyph, color=color,
                     fontsize=T.px_to_pt(f * 0.95), va="top", ha="left", alpha=a, fontweight="bold")
            for j, line in enumerate(body):
                fig.text((x0_px + f * 1.6) / T.CARD_PX, 1 - (y + j * f * 1.32) / T.CARD_PX, line,
                         color=T.TEXT if r["final"] not in ("fix",) else T.TEXT_MUTED,
                         fontsize=T.px_to_pt(f), va="top", ha="left", alpha=a)
            y += f * 1.32 * len(body) + f * 0.45
    if not fits:
        fig.text(0.5, T.FOOTER_LINE_Y + 0.01, "(more on the next card)", color=T.TEXT_MUTED, ha="center",
                 va="bottom", fontsize=T.px_to_pt(12), alpha=p_frame)
    return fig


def _row(r) -> dict:
    return {"text": r["change"][0].upper() + r["change"][1:], "final": r["final"], "key": r["ability"]}


LIST_W_PX = T.CARD_PX * (1 - 2 * T.MARGIN_X)
LIST_H_PX = {"hero": 785, "other": 840}     # room for the list under the header (hero cards: big icon + badge)
MIN_LIST_FONT = 18                         # below this a card splits into pages instead of shrinking further


def _paginate(groups: list[tuple[str, list[dict]]], height_px: float, ncols: int = 1,
              keep_groups: bool = False, min_font: float | None = None, head_scale: float = 1.0,
              max_groups: int | None = None) -> list[list[tuple[str, list[dict]]]]:
    """Split into as few cards as possible with every line at MIN_LIST_FONT or larger, then even out
    the cards so the last one isn't a near-empty page in a much bigger font. keep_groups: never split
    one group (e.g. one item) across two cards."""
    if ncols > 1:
        col_w = (LIST_W_PX - 36 * (ncols - 1)) / ncols
        fits0 = lambda page: _layout_columns(page, col_w, height_px, ncols, mf, head_scale)[2]  # noqa: E731
    else:
        fits0 = lambda page: _layout_list(page, LIST_W_PX, height_px, mf, head_scale=head_scale)[2]  # noqa: E731
    mf = min_font or MIN_LIST_FONT
    fits = lambda page: (max_groups is None or len(page) <= max_groups) and fits0(page)  # noqa: E731
    if keep_groups:
        def greedy(cap_lines: float):
            pages, cur, n = [], [], 0
            for g in groups:
                if cur and (n + len(g[1]) > cap_lines or not fits(cur + [g])):
                    pages.append(cur)
                    cur, n = [], 0
                cur.append(g)
                n += len(g[1])
            return pages + ([cur] if cur else [])
        total = sum(len(g[1]) for g in groups)
        pages = greedy(float("inf"))
        if len(pages) > 1:                    # even out: about the same number of lines on every card
            for slack in (0, 1, 2, 3, 4):
                even = greedy(np.ceil(total / len(pages)) + slack)
                if len(even) == len(pages):
                    return even
        return pages
    total = sum(len(rows) for _, rows in groups)

    def greedy(cap: int):
        pages, cur, n = [], [], 0
        for g, rows in groups:
            for r in rows:
                trial = [(gg, list(rr)) for gg, rr in cur]
                if trial and trial[-1][0] == g:
                    trial[-1][1].append(r)
                else:
                    trial.append((g, [r]))
                if cur and (n >= cap or not fits(trial)):
                    pages.append(cur)
                    cur, n = [(g, [r])], 1
                else:
                    cur, n = trial, n + 1
        return pages + ([cur] if cur else [])

    pages = greedy(total)
    if len(pages) > 1:
        even = greedy(int(np.ceil(total / len(pages))))
        if len(even) == len(pages):
            pages = even
    return pages


def hero_groups_of(df: pd.DataFrame, hero: str, names: dict | None = None) -> list[tuple[str, list[dict]]]:
    """Changes grouped under Base stats, Gun, then each ability in slot order, headed "4: Develop Queen"."""
    d = df[(df["kind"] == "hero") & (df["subject"] == hero)]
    slots = (names or {}).get("slots", {}).get(hero, {})
    order = []
    for _, r in d.iterrows():
        g = r["ability"] or "Base stats"
        if g not in order:
            order.append(g)
    order.sort(key=lambda g: (g != "Base stats", g != "Gun", slots.get(g.lower(), 9)))
    label = lambda g: f"{slots[g.lower()]}: {g}" if g.lower() in slots else g  # noqa: E731
    return [(label(g), [_row(r) for _, r in d[(d["ability"].replace("", "Base stats")) == g].iterrows()])
            for g in order]


def cards(data_dir: Path, **_) -> dict:
    """{name: draw(progress, highlight)} for a patch folder (one with changes.csv)."""
    df, meta = load_changes(data_dir)
    names = load_names(online=False)
    out = {}
    heroes = subject_summary(df, "hero")
    items = subject_summary(df, "item")
    item_icons = names.get("item_icons", {})
    from hero_spotlight import slug
    types = {}
    try:
        weeks = sorted(T.PROJECT_ROOT.glob("week*_analysis/hero_winrates.csv"))
        if weeks:
            hw = pd.read_csv(weeks[-1])
            types = dict(zip(hw["hero_name"], hw["hero_type"].str.lower()))
    except Exception:
        pass
    if len(heroes):
        out["patch_closer"] = (lambda progress=1.0, highlight=None:
                               render_closer(heroes, items, meta, item_icons, progress, highlight))
        out["patch_opener"] = (lambda progress=1.0, highlight=None:
                               render_opener(heroes, meta, types, progress, highlight))
        out["patch_heroes"] = (lambda progress=1.0, highlight=None:
                               render_overview(heroes, "hero", meta, progress, highlight))
        out["patch_heroes_size"] = (lambda progress=1.0, highlight=None:
                                    render_overview(heroes, "hero", meta, progress, highlight, measure="size"))
    if len(items):
        out["patch_items"] = (lambda progress=1.0, highlight=None:
                              render_overview(items, "item", meta, progress, highlight, icons=item_icons))
        out["patch_items_size"] = (lambda progress=1.0, highlight=None:
                                   render_overview(items, "item", meta, progress, highlight, measure="size",
                                                   icons=item_icons))
    for _, h in heroes.iterrows():
        name = h["subject"]
        pages = _paginate(hero_groups_of(df, name, names), LIST_H_PX["hero"])
        ab_icons = {}
        for ab, path in names.get("ability_icons", {}).get(name, {}).items():
            slot = names["slots"].get(name, {}).get(ab)
            proper = next((a for a in names["abilities"].get(name, []) if a.lower() == ab), ab)
            ab_icons[f"{slot}: {proper}" if slot else proper] = path
        color = {"BUFFED": T.GREEN, "NERFED": T.RED, "REWORKED": T.GOLD}.get(h["call"], "#c9b88a")
        plural = lambda n, w: f"{n:g} {w}" + ("" if n == 1 else "s")  # noqa: E731
        avg = lambda v: f" (avg {v:.0f}%)" if pd.notna(v) else ""  # noqa: E731
        sub = (f"{plural(h['buff'], 'buff')}{avg(h['buff_avg'])} · {plural(h['nerf'], 'nerf')}{avg(h['nerf_avg'])}"
               + (" · reworked" if h["rework"] else ""))
        for p, groups in enumerate(pages, 1):
            key = f"patch_hero_{slug(name)}" + (f"_{p}" if len(pages) > 1 else "")
            out[key] = (lambda progress=1.0, highlight=None, g=groups, n=name, s=sub, c=color, call=h["call"], pg=p,
                        total=len(pages), gi=ab_icons:
                        render_list(n.upper() + (f" ({pg}/{total})" if total > 1 else ""), s, g, meta,
                                    progress, highlight, icon=n, ring=T.archetype_color(types.get(n, "")),
                                    badge=(call, c), group_icons=gi))
    if len(items):
        slot_names = {"weapon": "WEAPON", "vitality": "VITALITY", "spirit": "SPIRIT"}
        slot_of = items["slot"].where(items["slot"].isin(list(slot_names)), "")
        for slot in ["weapon", "vitality", "spirit", ""]:
            sub_items = items[slot_of == slot]
            if sub_items.empty:
                continue
            groups = []
            for _, it in sub_items.iterrows():
                d = df[(df["kind"] == "item") & (df["subject"] == it["subject"])]
                groups.append((it["subject"], [{**_row(r), "key": it["subject"]} for _, r in d.iterrows()]))
            pages = _paginate(groups, LIST_H_PX["other"], ncols=2, keep_groups=True, min_font=ITEM_MIN_FONT,
                              head_scale=ITEM_HEAD_SCALE, max_groups=ITEMS_PER_CARD_MAX)
            nb, nn = int((sub_items["net"] > 0).sum()), int((sub_items["net"] < 0).sum())
            label = slot_names.get(slot, "OTHER")
            for p, g in enumerate(pages, 1):
                key = f"patch_items_{slot or 'other'}" + (f"_{p}" if len(pages) > 1 else "")
                out[key] = (lambda progress=1.0, highlight=None, g=g, p=p, total=len(pages), label=label,
                            sub=f"{len(sub_items)} items · {nb} mostly buffed · {nn} mostly nerfed":
                            render_list(f"{label} ITEMS" + (f" ({p}/{total})" if total > 1 else ""), sub, g, meta,
                                        progress, highlight, group_icons=item_icons, ncols=2,
                                        head_scale=ITEM_HEAD_SCALE, accent=SLOT_ACCENT.get(label.lower()),
                                        min_font=ITEM_MIN_FONT, max_font=ITEM_MAX_FONT))
    gen = df[df["kind"] == "general"]
    if len(gen):
        rows = [{"text": r["change"], "final": r["final"] if r["final"] in ("fix",) else "system", "key": ""}
                for _, r in gen.iterrows()]
        gpages = _paginate([("", rows)], LIST_H_PX["other"])
        for p, g in enumerate(gpages, 1):
            out[f"patch_general_{p}"] = (lambda progress=1.0, highlight=None, g=g, p=p, total=len(gpages):
                                         render_list("GENERAL CHANGES" + (f" ({p}/{total})" if total > 1 else ""),
                                                     "map, economy and system changes", g, meta,
                                                     progress, highlight))
    for name, draw in out.items():          # text cards reveal line by line: give animate.py their pace
        groups = getattr(draw, "__defaults__", None) and next(
            (d for d in draw.__defaults__ if isinstance(d, list) and d and isinstance(d[0], tuple)), None)
        if groups:
            draw.build_s = list_build_s(groups)
    return out


def insights(data_dir: Path, **_) -> list[Insight]:
    df, meta = load_changes(data_dir)
    out = []
    h = subject_summary(df, "hero")
    if len(h):
        for _, r in h.head(3).iterrows():
            if r["net"] > 0:
                out.append(Insight("patch_heroes", "patch_buff", hero=r["subject"], highlight=(r["subject"],),
                                   headline=f"{r['subject']}: {r['call'].lower()} ({r['buff']:g} buffs, {r['nerf']:g} nerfs)",
                                   effect=float(r["net"]), confidence=1.0))
        for _, r in h.iloc[::-1].head(3).iterrows():
            if r["net"] < 0:
                out.append(Insight("patch_heroes", "patch_nerf", hero=r["subject"], highlight=(r["subject"],),
                                   headline=f"{r['subject']}: {r['call'].lower()} ({r['nerf']:g} nerfs, {r['buff']:g} buffs)",
                                   effect=float(-r["net"]), confidence=1.0))
        for _, r in h[h["rework"]].iterrows():
            out.append(Insight("patch_heroes", "patch_rework", hero=r["subject"], highlight=(r["subject"],),
                               headline=f"{r['subject']} gets a rework", effect=5.0, confidence=1.0))
    return out


# --- CLI ---------------------------------------------------------------------------
def review(patch_dir: Path) -> None:
    shift_warning(patch_dir)
    df = read_changes_csv(Path(patch_dir) / "changes.csv")
    edited = df["parser"].ne("") & df["auto"].ne(df["parser"])
    todo = df[~edited & (df["include"].str.lower() != "no") & df["kind"].isin(["hero", "item"]) &
              ((df["confidence"] == "low") | (df["auto"] == "unclear"))]
    print(f"{len(df)} lines · {int(edited.sum())} corrected by you · {len(todo)} more worth a look "
          f"(low confidence or unclear):\n")
    for _, r in todo.iterrows():
        who = r["subject"] + (f" · {r['ability']}" if r["ability"] else "")
        print(f"  [{r['id']}] {r['auto'].upper():8s} {who}: {r['change']}\n             ({r['why']})")
    counts = df[df["include"].str.lower() != "no"].groupby(["kind", "auto"]).size().unstack(fill_value=0)
    print("\nCalls by kind:\n" + counts.to_string())


def shift_warning(patch_dir: Path) -> None:
    """Spot an `auto` column that got shuffled in a spreadsheet (e.g. one column sorted on its own): your
    calls then disagree with lots of plain numbers-based calls ("Cooldown increased ..." marked buff)."""
    df = read_changes_csv(Path(patch_dir) / "changes.csv")
    sure = df[(df["parser"] != "") & (df["confidence"] == "high") & df["why"].str.startswith("numbers")]
    off = sure[sure["auto"].str.lower() != sure["parser"].str.lower()]
    if len(off) >= 4 and len(off) >= 0.1 * max(len(sure), 1):
        print(f"\n  WARNING: your call differs from a plain numbers-based call on {len(off)} of {len(sure)} lines.")
        print("  If you sorted or pasted the auto column on its own in a spreadsheet, it may be out of line")
        print("  with the change text. Examples:")
        for r in off.head(8).itertuples():
            print(f"    {r.auto:>7s} (parser: {r.parser})  {r.subject}: {r.change[:70]}")
        print()


def lessons() -> None:
    """Which parser rules you override most, with examples: the list to turn into rule fixes."""
    lp = PATCH_ROOT / LEARNED
    if not lp.exists():
        print("No corrections saved yet.")
        return
    d = pd.read_csv(lp, dtype=str, keep_default_na=False)
    d["rule"] = d["rule"].str.replace(r"^learned.*", "learned from an earlier review", regex=True)
    print(f"{len(d)} corrections across {d['patch'].nunique()} patch(es)\n")
    for rule, g in sorted(d.groupby("rule"), key=lambda x: -len(x[1])):
        print(f"{len(g):3d} x  {rule}")
        for r in g.head(4).itertuples():
            print(f"        {r.parser:>7s} -> {r.call:<7s} {r.subject}: {r.change[:80]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Patch notes -> buff / nerf cards")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--latest", action="store_true")
    src.add_argument("--link")
    src.add_argument("--file", type=Path)
    p.add_argument("--name", help="patch name for the folder and card footers (default: post title or file name)")
    p.add_argument("--date", help="YYYY-MM-DD (default: post date, or today for --file)")
    p.add_argument("--out", type=Path, help="patch folder (default: patches/<date>_<name>)")
    r = sub.add_parser("review")
    r.add_argument("patch_dir", type=Path)
    sub.add_parser("lessons", help="rules you override most often, across all reviewed patches")
    c = sub.add_parser("cards")
    c.add_argument("patch_dir", type=Path)
    c.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()

    if args.cmd == "parse":
        if args.file:
            post = {"title": args.name or args.file.stem, "link": str(args.file), "content": args.file.read_text(),
                    "published_at": args.date or datetime.now(timezone.utc).date().isoformat()}
        else:
            post = fetch_patch(args.link)
            post["title"] = args.name or post["title"]
            if args.date:
                post["published_at"] = args.date
        names = load_names()
        df = parse_notes(html_to_lines(post["content"]), names)
        if df.empty:
            raise SystemExit("No change lines found (expected lines starting with '- ').")
        out = args.out or patch_dir_for(post["title"], post["published_at"])
        meta = {k: post[k] for k in ("title", "link", "published_at")}
        df = write_changes(df, out, meta)
        print(f"{len(df)} change lines -> {out / 'changes.csv'}")
        review(out)
        print(f"\nCorrect the verdict column where needed, then: python patch_notes.py cards {out}")
    elif args.cmd == "review":
        review(args.patch_dir)
    elif args.cmd == "lessons":
        lessons()
    else:
        shift_warning(args.patch_dir)
        backup(args.patch_dir / "changes.csv")
        for old in (args.patch_dir / "graphs").glob("patch_*.png"):     # drop cards a previous run made
            old.unlink()
        learn_from(args.patch_dir)
        saved = T.save_cards(cards(args.patch_dir), args.patch_dir / "graphs", args.progress)
        print(f"Saved {len(saved)} cards to {args.patch_dir / 'graphs'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())