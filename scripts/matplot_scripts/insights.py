"""
insights.py — rank the week's findings and write a plan for each video.

Every chart script has an insights(data_dir) function that returns its findings as data
(chart_stats.Insight). This script collects them, scores each one by

    score = KIND_WEIGHT[kind] × min(effect, EFFECT_CAP) × confidence_factor(confidence)

and builds a running order for every video in VIDEOS: the video's cards in order, each with
its own findings ranked (several findings about one hero merge into one). One card = one
animate.py clip that highlights each finding in turn.

Outputs, in <week folder>/plans/:
    01_meta_report.md           voiceover outline: card by card, what to say, how sure we are
    01_meta_report_scenes.csv   the matching clips:  python animate.py <week> --scenes plans/01_meta_report_scenes.csv
    ...one pair per video this week...
    shorts.md / shorts_scenes.csv   one-card, one-hero stories for shorts
    hero_compendium_scenes.csv      every hero card, 30 s each, tier list first (no-voiceover video)
    hero_compendium_chapters.txt    YouTube chapters for it
and in the week folder:
    insights.md                 index: this week's videos, their top stories, and what to double-check
    insights.json               everything as data

Rotation (by week number): "When heroes win" and "Teams & synergy" alternate, the class
deep dive alternates between two class pairs, and one of four hero-spotlight groups is covered
(--hero-groups all for all four; each hero card doubles as a short). Force a lineup with --videos.

Tune what makes the cut with KIND_WEIGHT, MIN_CONFIDENCE / STORY_CONFIDENCE and the per-card
limits in VIDEOS.

Usage: python insights.py [week_dir | --week N] [--videos meta_report teams classes_a ...] [--base DIR]
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme as T  # noqa: E402
from chart_stats import Insight, confidence_label, load_hero_meta  # noqa: E402

MODULES = ["week_over_week", "pick_rate_winrate_scatter", "balance_wr_distribution",
           "hero_wr_total_matches_split_column", "archetype_stacking", "matchup_bubble_chart",
           "hero_strength_and_scaling", "hero_duo_synergy_table", "most_freq_comps", "comp_uniqueness",
           "match_duration", "hero_spotlight", "rank_performance"]

# How much each kind of story matters (edit freely). Effects are in win-rate points, except
# duration_change (minutes). Within a video, this decides which findings each card leads with.
KIND_WEIGHT = {
    "tier_change": 1.4, "new_hero": 1.3, "nerf_watch": 1.2, "buff_watch": 1.1, "mover": 1.0,
    "popular_trap": 1.0, "hidden_gem": 0.9, "late_scaler": 0.9, "early_hero": 0.8,
    "duo_synergy": 0.8, "duo_anti": 0.7, "stacking": 0.7, "duration_change": 1.0,
    "class_best": 0.8, "class_worst": 0.8, "class_count": 0.6, "class_late": 0.8, "class_early": 0.8,
    "class_late_best": 0.6, "rank_best": 0.7, "rank_worst": 0.7, "class_rank": 0.9,
}
EFFECT_CAP = 8.0            # one huge outlier shouldn't bury everything else
MIN_CONFIDENCE = 0.80       # below this a finding is dropped (listed under "check before you say it")
STORY_CONFIDENCE = 0.95     # findings between MIN and this are listed as "maybe mention", not animated
BUILD_S, HOLD_S, BEAT_S = 3.5, 1.5, 4.0     # animate.py timing per clip
CLASSES = ["brawler", "marksman", "assassin", "mystic"]
CLASS_PAIRS = [("brawler", "mystic"), ("marksman", "assassin")]


def _plural(cls: str) -> str:
    return "marksmen" if cls == "marksman" else f"{cls}s"


def _per_class(*prefixes: str, classes=CLASSES) -> list[str]:
    return [f"{p}_{c}" for c in classes for p in prefixes]


# Each video: its cards in running order, and how many findings each card may highlight.
VIDEOS = {
    "meta_report": {
        "title": "Weekly meta report",
        "cards": ["tier_list", "hero_movers", "pick_rate_winrate_scatter", "deadlock_balance_chart",
                  "hero_winrates_top_bottom"],
        "per_card": {"tier_list": 8, "hero_movers": 4}, "default_per_card": 3,
    },
    "when_heroes_win": {
        "title": "When heroes win: game length", "rotation": 0,
        "cards": ["match_duration", "hero_early_vs_late",
                  *_per_class("hero_duration_winrate", "hero_duration_scaling")],
        "default_per_card": 2,
    },
    "teams": {
        "title": "Teams & synergy", "rotation": 1,
        "cards": ["archetype_stacking", *_per_class("archetype_count_matrix"), "hero_duo_synergy_best",
                  "hero_duo_synergy_worst", "most_played_comps", "comp_uniqueness"],
        "per_card": {"hero_duo_synergy_best": 3, "hero_duo_synergy_worst": 3}, "default_per_card": 2,
    },
    **{f"classes_{'ab'[i]}": {
        "title": f"Class deep dive: {_plural(pair[0])} & {_plural(pair[1])}",
        "rotation": i, "classes": pair,
        "cards": ["tier_list", *_per_class("hero_winrates", "archetype_count_matrix", "hero_duration_winrate",
                                           "hero_duration_scaling", classes=pair)],
        "default_per_card": 2,
    } for i, pair in enumerate(CLASS_PAIRS)},
}
for _g in range(1, 5):
    VIDEOS[f"heroes_{_g}"] = {"title": f"Hero spotlights, group {_g} of 4", "hero_group": _g, "cards": []}
VIDEOS["rank"] = {
    "title": "Performance by rank",
    "cards": ["rank_best_heroes", "rank_worst_heroes", *_per_class("rank_curve")],
    "per_card": {"rank_best_heroes": 4, "rank_worst_heroes": 4}, "default_per_card": 2,
}
SHORTS = 12
COMPENDIUM_BUILD_S, COMPENDIUM_HOLD_S = 4.0, 26.0     # 30 s per card in the no-voiceover hero video


def confidence_factor(c: float) -> float:
    """0 at MIN_CONFIDENCE, 1 at 99%+."""
    return max(0.0, min(1.0, (c - MIN_CONFIDENCE) / (0.99 - MIN_CONFIDENCE)))


def score(i: Insight) -> float:
    return KIND_WEIGHT.get(i.kind, 0.5) * min(i.effect, EFFECT_CAP) * confidence_factor(i.confidence)


# --- COLLECT -----------------------------------------------------------------------
def collect(data_dir: Path, **opts) -> list[Insight]:
    found = []
    for name in MODULES:
        try:
            found += importlib.import_module(name).insights(data_dir, **opts)
        except FileNotFoundError as e:
            print(f"  skipping {name}: {Path(str(e.filename or e)).name} not found")
    return found


def merge_by_hero(stories: list[Insight]) -> list[Insight]:
    """Keep the best-scoring story per hero; the others become extra lines under it."""
    best: dict[str, Insight] = {}
    out = []
    for i in sorted(stories, key=score, reverse=True):
        if i.hero is None:
            out.append(i)
        elif i.hero not in best:
            best[i.hero] = Insight(**{**asdict(i), "also": []})   # copy: one finding can sit in several videos
            out.append(best[i.hero])
        else:
            best[i.hero].also.append(f"{i.headline} ({i.detail})" if i.detail else i.headline)
    return out


def week_number(data_dir: Path) -> int | None:
    m = re.search(r"week(\d+)", data_dir.name)
    return int(m.group(1)) if m else None


def lineup(data_dir: Path, requested: list[str] | None, all_groups: bool = False) -> list[str]:
    """This week's videos: the meta report, one of the alternating pair, one class pair, and one
    hero-spotlight group (all four with all_groups)."""
    if requested:
        unknown = [v for v in requested if v not in VIDEOS]
        if unknown:
            raise SystemExit(f"Unknown video(s): {', '.join(unknown)}. Choose from: {', '.join(VIDEOS)}")
        return requested
    n = week_number(data_dir)
    turn = (n or 0) % 2                       # patch periods (no week number) use the first turn
    heroes = [f"heroes_{g}" for g in range(1, 5)] if all_groups else [f"heroes_{(n or 0) % 4 + 1}"]
    return ["meta_report",
            next(k for k in ("when_heroes_win", "teams") if VIDEOS[k]["rotation"] == turn),
            next(k for k in VIDEOS if k.startswith("classes_") and VIDEOS[k]["rotation"] == turn),
            "rank", *heroes]


def plan_video(key: str, found: list[Insight], data_dir: Path, available: set[str]) -> dict:
    v = VIDEOS[key]
    if "hero_group" in v:
        return plan_hero_group(key, v, found, data_dir, available)
    cards = [c for c in v["cards"] if c in available]
    sections = []
    featured: dict[str, str] = {}             # hero -> card where the video already covers them
    for card in cards:
        mine = [i for i in found if i.card == card]
        if card == "tier_list" and "classes" in v:
            # class deep dive: open on the tier list with this video's heroes highlighted, nothing else
            types = load_hero_meta(data_dir).set_index("hero_name")["hero_type"]
            mine = [Insight("tier_list", "class_intro", context=True,
                            headline=f"Where the {_plural(cls)} sit on the tier list",
                            highlight=tuple(types[types == cls].index)) for cls in v["classes"]]
        context = [i for i in mine if i.context]
        stories = merge_by_hero([i for i in mine if not i.context and i.confidence >= MIN_CONFIDENCE])
        limit = v.get("per_card", {}).get(card, v.get("default_per_card", 3))
        solid, repeats = [], []
        for i in stories:
            if i.confidence < STORY_CONFIDENCE:
                continue
            if i.hero and i.hero in featured:         # one headline per hero per video
                i.tags = (*i.tags, f"already covered on `{featured[i.hero]}`")
                repeats.append(i)
            else:
                solid.append(i)
        for i in solid[:limit]:
            if i.hero:
                featured[i.hero] = card
        sections.append({"card": card, "context": context, "stories": solid[:limit],
                         "maybe": [i for i in stories if i.confidence < STORY_CONFIDENCE] + solid[limit:] + repeats,
                         "dropped": [i for i in mine if not i.context and i.confidence < MIN_CONFIDENCE]})
    return {"key": key, "title": v["title"], "sections": sections}


def plan_hero_group(key: str, v: dict, found: list[Insight], data_dir: Path, available: set[str]) -> dict:
    """One card per hero in the group, strongest first; each card carries the hero's profile line
    and every finding about that hero from any chart (shown on the hero card, so no highlights)."""
    import hero_spotlight as HS
    data = HS.load_all(data_dir)
    group = HS.hero_groups(data_dir, load_hero_meta(data_dir))[v["hero_group"] - 1]
    heroes = sorted((h for h in group if h in data["d"].index), key=lambda h: -data["d"].loc[h, "power"])
    sections = []
    for hero in heroes:
        card = f"hero_{HS.slug(hero)}"
        if card not in available:
            continue
        context = [i for i in found if i.card == card and i.context]
        stories = merge_by_hero([i for i in found if i.hero == hero and not i.context
                                 and i.confidence >= MIN_CONFIDENCE])
        sections.append({"card": card, "context": context, "stories": stories[:4], "maybe": stories[4:],
                          "dropped": [], "no_highlight": True})
    return {"key": key, "title": v["title"], "sections": sections}


def plan_shorts(found: list[Insight]) -> list[Insight]:
    """Single-hero stories, strongest first, one per hero: each is one card with one highlight."""
    stories = [i for i in found if not i.context and i.hero and i.confidence >= STORY_CONFIDENCE]
    return merge_by_hero(stories)[:SHORTS]


# --- WRITE -------------------------------------------------------------------------
def _meta_line(data_dir: Path) -> str:
    try:
        m = json.loads((data_dir / "period_meta.json").read_text())
    except (OSError, ValueError):
        return ""
    bits = [f"{m.get('complete_matches', 0):,} complete matches"]
    if "ban_coverage_pct" in m:
        bits.append(f"bans recorded for {m['ban_coverage_pct']}% of them")
    return " · ".join(bits)


def _item(i: Insight, n: int | None = None, show_card: bool = False) -> str:
    lines = [f"{n}. **{i.headline}**" if n else f"- **{i.headline}**"]
    if i.detail:
        lines.append(f"   {i.detail}.")
    for a in i.also:
        lines.append(f"   Also: {a}.")
    meta = [f"card `{i.card}`"] if show_card else []
    if i.highlight:
        meta.append("highlight " + " + ".join(i.highlight[:6]) + (" + …" if len(i.highlight) > 6 else ""))
    if not i.context:
        meta.append(f"{confidence_label(i.confidence)} ({i.confidence:.0%})")
    if meta:
        lines.append("   " + " · ".join(meta))
    lines += [f"   ⚠ {t}" for t in i.tags]
    return "\n".join(lines)


def write_video(p: dict, n: int, data_dir: Path, out_dir: Path) -> tuple[Path, int]:
    label = T.week_label_from_dir(data_dir)
    stem = f"{n:02d}_{p['key']}"
    md = [f"# {p['title']} — {label}", "",
          f"Generated {datetime.now():%Y-%m-%d %H:%M} · {_meta_line(data_dir)}".rstrip(" ·"), "",
          f"Running order: one clip per card, in `{stem}_scenes.csv`. " + (
              "Each clip holds on the hero's card long enough to read its findings; each clip also works as a short."
              if p["key"].startswith("heroes_") else
              "Each numbered finding is one highlight in that clip, in this order."), ""]
    rows = []
    for k, s in enumerate(p["sections"], 1):
        md += [f"## {k}. `{s['card']}`", ""]
        md += [_item(i) for i in s["context"]]
        md += [_item(i, j) for j, i in enumerate(s["stories"], 1)]
        if not (s["context"] or s["stories"]):
            md.append("_No standout findings this week: describe the card, or skip it._")
        if s["maybe"]:
            md += ["", "Maybe mention (less certain, or past this card's limit; not animated):"]
            md += [_item(i) for i in s["maybe"]]
        md.append("")
        if s.get("no_highlight"):                 # hero cards: hold while the findings are read out
            beats, hold = [], HOLD_S + BEAT_S * (len(s["context"]) + len(s["stories"]))
        else:
            beats = [f"{'+'.join(i.highlight)}:{BEAT_S:g}" for i in s["context"] + s["stories"] if i.highlight]
            if beats:
                beats.append("-:1")
            hold = HOLD_S if beats else HOLD_S + BEAT_S
        rows.append([s["card"], BUILD_S, hold, "; ".join(beats), f"{p['key']}_{k:02d}_{s['card']}"])
    dropped = [i for s in p["sections"] for i in s["dropped"]]
    flagged = [i for s in p["sections"] for i in s["stories"] if i.tags]
    if flagged or dropped:
        md += ["## Check before you say it", ""]
        md += [f"- {i.headline}: " + "; ".join(i.tags) for i in flagged]
        md += [f"- Dropped as possible noise ({i.confidence:.0%} sure): {i.headline}" for i in dropped]
    (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n")
    with open(out_dir / f"{stem}_scenes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "build", "hold", "highlight", "name"])
        w.writerows(rows)
    return out_dir / f"{stem}.md", len(rows)


def write_shorts(shorts: list[Insight], data_dir: Path, out_dir: Path) -> None:
    md = [f"# Shorts — {T.week_label_from_dir(data_dir)}", "",
          "One finding per short: one card, one hero highlighted. Clips in `shorts_scenes.csv`.", ""]
    md += [_item(i, n, show_card=True) for n, i in enumerate(shorts, 1)]
    (out_dir / "shorts.md").write_text("\n".join(md) + "\n")
    with open(out_dir / "shorts_scenes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "build", "hold", "highlight", "name"])
        for n, i in enumerate(shorts, 1):
            slug = re.sub(r"[^a-z0-9]+", "_", (i.hero or "").lower()).strip("_")
            w.writerow([i.card, BUILD_S, HOLD_S, f"{'+'.join(i.highlight)}:{BEAT_S + 2:g}",
                        f"short_{n:02d}_{slug}"])


def write_compendium(data_dir: Path, out_dir: Path, available: set[str], order: str) -> int:
    """Every hero card back to back (tier list first), plus YouTube chapters so viewers can jump to
    their hero. Render and join:  python animate.py <week> --scenes plans/hero_compendium_scenes.csv
    --concat hero_compendium.mp4 --fps 30"""
    import hero_spotlight as HS
    d = HS.load_all(data_dir)["d"]
    heroes = sorted(d.index) if order == "alpha" else list(d.sort_values("power", ascending=False).index)
    cards = (["tier_list"] if "tier_list" in available else []) + \
            [f"hero_{HS.slug(h)}" for h in heroes if f"hero_{HS.slug(h)}" in available]
    if not cards:
        return 0
    names = {f"hero_{HS.slug(h)}": h for h in heroes}
    per = COMPENDIUM_BUILD_S + COMPENDIUM_HOLD_S
    chapters = []
    with open(out_dir / "hero_compendium_scenes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "build", "hold", "highlight", "name"])
        for k, card in enumerate(cards):
            w.writerow([card, COMPENDIUM_BUILD_S, COMPENDIUM_HOLD_S, "", f"compendium_{k:02d}_{card}"])
            t = int(k * per)
            stamp = f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"
            chapters.append(f"{stamp} {names.get(card, 'Tier list')}")
    label = T.week_label_from_dir(data_dir)
    (out_dir / "hero_compendium_chapters.txt").write_text(
        f"Every hero, {label} ({'A–Z' if order == 'alpha' else 'strongest to weakest'})\n\n" + "\n".join(chapters) + "\n")
    return len(cards)


def write_index(videos: list[dict], files: list[Path], shorts: list[Insight], data_dir: Path) -> Path:
    md = [f"# {T.week_label_from_dir(data_dir)} — video plan", "",
          f"Generated {datetime.now():%Y-%m-%d %H:%M} · {_meta_line(data_dir)}".rstrip(" ·"), ""]
    for p, f in zip(videos, files):
        top = sorted((i for s in p["sections"] for i in s["stories"]), key=score, reverse=True)[:3]
        md += [f"## {p['title']}", f"`plans/{f.name}` · {len(p['sections'])} cards", ""]
        md += [f"- {i.headline}" for i in top] or ["- (context only this week)"]
        md.append("")
    md += ["## Shorts", f"`plans/shorts.md` · {len(shorts)} shorts", ""] + [f"- {i.headline}" for i in shorts[:5]]
    path = data_dir / "insights.md"
    path.write_text("\n".join(md) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank this week's findings and write a plan per video")
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--week", type=int, help="week number (instead of week_dir)")
    ap.add_argument("--videos", nargs="+", help=f"override this week's lineup: {', '.join(VIDEOS)}")
    ap.add_argument("--base", default=None, help="folder to compare against (default: last week)")
    ap.add_argument("--hero-groups", choices=["rotate", "all"], default="rotate",
                    help="one hero-spotlight group per week (default) or all four")
    ap.add_argument("--compendium-order", choices=["tier", "alpha"], default="tier",
                    help="hero order in the no-voiceover all-heroes video (default: strongest first)")
    args = ap.parse_args()

    data_dir = (T.PROJECT_ROOT / f"week{args.week}_analysis") if args.week else T.resolve_target_dir(args.week_dir)
    if not (data_dir / "hero_winrates.csv").exists():
        print(f"No hero_winrates.csv in {data_dir}: run query_transform.py first")
        return 1
    import animate                                       # card names this week (some exports may be missing)
    available = set(animate.collect_cards(data_dir, base=args.base))
    found = collect(data_dir, base=args.base)

    out_dir = data_dir / "plans"
    out_dir.mkdir(exist_ok=True)
    videos = [plan_video(k, found, data_dir, available)
              for k in lineup(data_dir, args.videos, args.hero_groups == "all")]
    skipped = [p["title"] for p in videos if not p["sections"]]
    videos = [p for p in videos if p["sections"]]
    for t in skipped:
        print(f"{t}: skipped, none of its cards exist this week (missing exports?)")
    files = []
    for n, p in enumerate(videos, 1):
        f, clips = write_video(p, n, data_dir, out_dir)
        files.append(f)
        print(f"{p['title']}: {f.name} ({clips} clips)")
    shorts = plan_shorts(found)
    write_shorts(shorts, data_dir, out_dir)
    n_comp = write_compendium(data_dir, out_dir, available, args.compendium_order)
    if n_comp:
        mins = n_comp * (COMPENDIUM_BUILD_S + COMPENDIUM_HOLD_S) / 60
        print(f"Hero compendium: {n_comp} cards, {mins:.0f} min (plans/hero_compendium_scenes.csv + chapters)")
    index = write_index(videos, files, shorts, data_dir)
    (data_dir / "insights.json").write_text(json.dumps(
        {"videos": [{"key": p["key"], "title": p["title"], "sections": [
            {k: ([asdict(i) | {"score": round(score(i), 2)} for i in v] if isinstance(v, list) else v)
             for k, v in s.items()} for s in p["sections"]]} for p in videos],
         "shorts": [asdict(i) for i in shorts]}, indent=2, default=str))
    print(f"Shorts: {len(shorts)}\nIndex: {index}\nRender a video: python animate.py {data_dir} "
          f"--scenes {out_dir / (files[0].stem + '_scenes.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())