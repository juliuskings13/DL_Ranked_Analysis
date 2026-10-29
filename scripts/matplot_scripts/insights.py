"""
insights.py — rank the week's findings and write the video plan.

Every chart script has an insights(data_dir) function that returns its findings as data
(chart_stats.Insight). This script collects them, scores each one by

    score = KIND_WEIGHT[kind] × min(effect, EFFECT_CAP) × confidence_factor(confidence)

merges findings about the same hero into one story, and writes three files to the week folder:

    insights.md            voiceover outline: opening context, ranked stories, closing context,
                           plus a "check before you say it" list (noise, new heroes, ban-pool changes)
    insights_scenes.csv    one animate.py clip per item, in outline order, with the right heroes
                           highlighted:  python animate.py --scenes week13_analysis/insights_scenes.csv
    insights.json          everything above as data (for later automation: titles, shorts, ...)

Tune what makes the cut with KIND_WEIGHT (how much each kind of story matters to you),
EFFECT_CAP and MIN_CONFIDENCE.

Usage: python insights.py [week_dir | --week N] [--top 10] [--base DIR]
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme as T  # noqa: E402
from chart_stats import Insight, confidence_label  # noqa: E402

MODULES = ["week_over_week", "pick_rate_winrate_scatter", "balance_wr_distribution", "archetype_stacking",
           "hero_strength_and_scaling", "hero_duo_synergy_table", "comp_uniqueness", "match_duration"]

# How much each kind of story matters (edit freely). Effects are in win-rate points, except
# duration_change (minutes). Tier changes and nerf/buff watch drive the most content.
KIND_WEIGHT = {
    "tier_change": 1.4, "new_hero": 1.3, "nerf_watch": 1.2, "buff_watch": 1.1, "mover": 1.0,
    "popular_trap": 1.0, "hidden_gem": 0.9, "late_scaler": 0.9, "early_hero": 0.8,
    "duo_synergy": 0.8, "duo_anti": 0.7, "stacking": 0.7, "duration_change": 1.0,
}
EFFECT_CAP = 8.0            # one huge outlier shouldn't bury everything else
MIN_CONFIDENCE = 0.80       # below this a finding is dropped (listed under "check before you say it")
STORY_CONFIDENCE = 0.95     # the main story list needs at least this; 80-95% go to "also worth a mention"
KIND_LIMIT = {"stacking": 1, "duo_synergy": 1, "duo_anti": 1, "late_scaler": 1, "early_hero": 1}
DEFAULT_KIND_LIMIT = 3      # per kind in the main list, so one chart doesn't take over the video
OPENING = ("tier_top", "balance")      # context shown before the stories
CLOSING = ("duration", "comps")                          # context shown after them

# animate.py timing per clip
BUILD_S, HOLD_S, BEAT_S = 3.5, 1.5, 4.0


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
            mod = importlib.import_module(name)
            found += mod.insights(data_dir, **opts)
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
            best[i.hero] = i
            out.append(i)
        else:
            best[i.hero].also.append(f"{i.headline} ({i.detail})" if i.detail else i.headline)
    return out


def plan(data_dir: Path, top: int, **opts) -> dict:
    found = collect(data_dir, **opts)
    context = [i for i in found if i.context]
    stories = [i for i in found if not i.context]
    kept = merge_by_hero([i for i in stories if i.confidence >= MIN_CONFIDENCE and score(i) > 0])
    kept.sort(key=score, reverse=True)
    main, more, per_kind = [], [], {}
    for i in kept:
        room = per_kind.get(i.kind, 0) < KIND_LIMIT.get(i.kind, DEFAULT_KIND_LIMIT)
        if len(main) < top and room and i.confidence >= STORY_CONFIDENCE:
            main.append(i)
            per_kind[i.kind] = per_kind.get(i.kind, 0) + 1
        else:
            more.append(i)
    dropped = [i for i in stories if i.confidence < MIN_CONFIDENCE]
    flagged = [i for i in main if i.tags]
    order = lambda kinds: sorted((i for i in context if i.kind in kinds), key=lambda i: kinds.index(i.kind))
    return {"opening": order(OPENING), "stories": main, "more": more[:12],
            "closing": order(CLOSING), "flagged": flagged, "dropped": dropped}


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


def _item(i: Insight, n: int | None = None) -> str:
    head = f"{n}. **{i.headline}**" if n else f"- **{i.headline}**"
    lines = [head]
    if i.detail:
        lines.append(f"   {i.detail}.")
    for a in i.also:
        lines.append(f"   Also: {a}.")
    meta = [f"card `{i.card}`"]
    if i.highlight:
        meta.append("highlight " + " + ".join(i.highlight))
    if not i.context:
        meta.append(f"{confidence_label(i.confidence)} ({i.confidence:.0%})")
        meta.append(f"score {score(i):.1f}")
    lines.append("   " + " · ".join(meta))
    for t in i.tags:
        lines.append(f"   ⚠ {t}")
    return "\n".join(lines)


def write_markdown(p: dict, data_dir: Path, path: Path) -> None:
    week = T.week_label_from_dir(data_dir)
    out = [f"# {week} insights", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M} · {_meta_line(data_dir)}".rstrip(" ·"), "",
           "Suggested order for the video. Every item has a matching clip in `insights_scenes.csv`.", ""]
    if p["opening"]:
        out += ["## Opening: the state of the meta", ""] + [_item(i) for i in p["opening"]] + [""]
    if p["stories"]:
        out += ["## Stories, strongest first", ""]
        out += [_item(i, k) for k, i in enumerate(p["stories"], 1)] + [""]
    if p["closing"]:
        out += ["## Closing context", ""] + [_item(i) for i in p["closing"]] + [""]
    if p["more"]:
        out += ["## Also worth a mention (shorts, hero spotlights)", ""] + [_item(i) for i in p["more"]] + [""]
    if p["flagged"] or p["dropped"]:
        out += ["## Check before you say it", ""]
        out += [f"- {i.headline}: " + "; ".join(i.tags) for i in p["flagged"]]
        out += [f"- Dropped as possible noise ({i.confidence:.0%} sure): {i.headline}" for i in p["dropped"]]
        out.append("")
    path.write_text("\n".join(out))


def write_scenes(p: dict, path: Path) -> int:
    rows = []
    for i in p["opening"] + p["stories"] + p["closing"]:
        beats = f"{'+'.join(i.highlight)}:{BEAT_S:g}" if i.highlight else ""
        hold = HOLD_S if beats else HOLD_S + BEAT_S
        rows.append([i.card, BUILD_S, hold, beats, f"{len(rows) + 1:02d}_{i.card}"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "build", "hold", "highlight", "name"])
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank this week's findings and write the video plan")
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--week", type=int, help="week number (instead of week_dir)")
    ap.add_argument("--top", type=int, default=10, help="stories in the main list")
    ap.add_argument("--base", default=None, help="folder to compare against (default: last week)")
    args = ap.parse_args()

    data_dir = (T.PROJECT_ROOT / f"week{args.week}_analysis") if args.week else T.resolve_target_dir(args.week_dir)
    if not (data_dir / "hero_winrates.csv").exists():
        print(f"No hero_winrates.csv in {data_dir}: run query_transform.py first")
        return 1
    p = plan(data_dir, args.top, base=args.base)
    write_markdown(p, data_dir, data_dir / "insights.md")
    n = write_scenes(p, data_dir / "insights_scenes.csv")
    (data_dir / "insights.json").write_text(json.dumps(
        {k: [asdict(i) | {"score": round(score(i), 2)} for i in v] for k, v in p.items()}, indent=2, default=str))
    print((data_dir / "insights.md").read_text())
    print(f"\nWrote insights.md, insights.json and insights_scenes.csv ({n} clips) to {data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())