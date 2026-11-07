"""
season_reset.py — season-change cards: the new season's rules vs the old one, and what the soft reset did
to every player's rank (last rank of the old season vs first rank of the new one).

Data comes straight from the Deadlock Community API's SQL endpoint (ClickHouse), so it works before the
new season's matches are in your database: one aggregated query, a few hundred rows.

    python season_reset.py export <out dir>            # -> reset_steps.csv + season_reset_meta.json
    python season_reset.py cards  <out dir>            # -> graphs/s2_*.png
    python animate.py <out dir>                        # animated clips (the folder is detected by its meta file)

Rank "steps": every rank has six tiers (I-VI); step = (rank - 1) * 6 + (tier - 1), 0 = Initiate I,
60 = Eternus I. A step is computed from the flat progress on each season's own ladder (7,000 points per
rank in season 1, 6,500 in season 2), so they compare like-for-like badges. Within Eternus the tier is a
percentile cut, taken from the displayed badge.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import theme as T

# =============================================================================
# SEASON RULES  (from the game's ranked-season data, /v1/assets/ranked-seasons)
# =============================================================================
SEASONS = {
    1: {"name": "Season 1", "start": 1785430800, "end": 1793908800, "points": 300,
        "tier_steps": [1000, 1000, 1000, 1000, 1000, 2000], "tier_shields": 2, "rank_shields": 5,
        "shields_until": 111, "placement_cap": 86, "placements": 8, "min_wins": 60},
    2: {"name": "Season 2", "start": 1793908800, "end": 1804802400, "points": 350,
        "tier_steps": [1000, 1000, 1000, 1000, 1000, 1500], "tier_shields": 1, "rank_shields": 3,
        "shields_until": 91, "placement_cap": 101, "placements": 8, "min_wins": 60},
}
OLD, NEW = 1, 2
S1_LOOKBACK_DAYS = 36       # a player's old-season rank = after their last match in this many days before the end

RANK_NAMES = ["Initiate", "Seeker", "Acolyte", "Sentinel", "Mystic", "Ritualist", "Emissary", "Oracle",
              "Phantom", "Ascendant", "Eternus"]
TIER_ROMAN = ["I", "II", "III", "IV", "V", "VI"]
UP_COLOR, DOWN_COLOR, SAME_COLOR = "#51cf66", "#ff6b6b", "#adb5bd"

SQL_URL = "https://api.deadlock-api.com/v1/sql"
HEADERS = {"User-Agent": "Deadlock Analytics (personal project)"}


def badge_name(badge: int) -> str:
    """86 -> 'Oracle VI'."""
    return f"{RANK_NAMES[badge // 10 - 1]} {TIER_ROMAN[badge % 10 - 1]}"


def rank_size(season: int) -> int:
    return sum(SEASONS[season]["tier_steps"])


def step_sql(progress: str, season: int, display: str | None = None) -> str:
    """SQL (DuckDB) expression: flat progress -> step (0 = Initiate I) on that season's ladder."""
    size = rank_size(season)
    eternus = size * 10
    within = f"least(({progress} % {size}) // 1000, 5)"
    top = (f"60 + CASE WHEN {display} // 10 = 11 THEN {display} % 10 - 1 ELSE 0 END" if display else "60")
    return f"CASE WHEN {progress} >= {eternus} THEN {top} ELSE ({progress} // {size}) * 6 + {within} END"


def export_sql() -> str:
    """The API's SQL endpoint runs DuckDB over its parquet snapshots. Progress values are cast to BIGINT
    (they're unsigned there, and the step math subtracts)."""
    s_old, s_new = SEASONS[OLD], SEASONS[NEW]
    lookback = s_old["end"] - S1_LOOKBACK_DAYS * 86400
    return f"""
WITH new AS (
    SELECT account_id,
           arg_min(player_rank_initial_flat_progress, start_time)::BIGINT AS p2,
           arg_min(player_rank_initial_calibration_games, start_time) AS cal2
    FROM match_player
    WHERE start_time >= to_timestamp({s_new['start']}) AND rank_interval = {NEW}
    GROUP BY account_id),
old AS (
    SELECT account_id,
           arg_max(player_rank_final_flat_progress, start_time)::BIGINT AS pf,
           arg_max(player_rank_initial_display_rank, start_time)::BIGINT AS b1,
           arg_max(player_rank_initial_calibration_games, start_time) AS cal1
    FROM match_player
    WHERE start_time >= to_timestamp({lookback}) AND start_time < to_timestamp({s_new['start'] + 4 * 3600})
      AND rank_interval = {OLD} AND account_id IN (SELECT account_id FROM new)
    GROUP BY account_id)
SELECT {step_sql('pf', OLD, 'b1')} AS s1_step, {step_sql('p2', NEW)} AS s2_step, count(*) AS players
FROM new INNER JOIN old USING (account_id)
WHERE cal1 = 0 AND cal2 = 0
GROUP BY ALL ORDER BY 1, 2"""


def query_api(sql: str, attempts: int = 6) -> list[dict]:
    import requests
    for k in range(attempts):
        try:
            res = requests.get(SQL_URL, params={"query": sql, "format": "json"}, headers=HEADERS, timeout=180)
            if res.status_code == 200:
                return res.json()
            if res.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"API error {res.status_code}: {res.text[:300]}")
        except requests.RequestException as e:
            print(f"  connection error: {e}")
        time.sleep(min(5 * 2 ** k, 120))
    raise RuntimeError("API still failing")


def export(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = query_api(export_sql())
    if isinstance(rows, dict):                       # {"data": [...]} style responses
        rows = rows.get("data") or rows.get("rows") or []
    df = pd.DataFrame(rows)
    if list(df.columns) != ["s1_step", "s2_step", "players"]:
        df.columns = ["s1_step", "s2_step", "players"]
    df = df.apply(pd.to_numeric).astype(int)
    df.to_csv(out / "reset_steps.csv", index=False)
    meta = {"old": OLD, "new": NEW, "players": int(df["players"].sum()),
            "exported": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "lookback_days": S1_LOOKBACK_DAYS}
    (out / "season_reset_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{meta['players']:,} players -> {out}")


# =============================================================================
# DATA
# =============================================================================
def load(data_dir: Path) -> dict:
    df = pd.read_csv(data_dir / "reset_steps.csv")
    meta = json.loads((data_dir / "season_reset_meta.json").read_text())
    df["change"] = df["s2_step"] - df["s1_step"]
    df["r1"] = df["s1_step"] // 6 + 1
    df["r2"] = df["s2_step"] // 6 + 1
    return {"steps": df, "meta": meta}


def by_rank(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r, x in df.groupby("r1"):
        n = x["players"].sum()
        rows.append({"rank": r, "n": n, "mean": np.average(x["change"], weights=x["players"]),
                     "up": 100 * x.loc[x["change"] > 0, "players"].sum() / n,
                     "down": 100 * x.loc[x["change"] < 0, "players"].sum() / n})
    return pd.DataFrame(rows).set_index("rank").reindex(range(1, 12))


def chart_meta(d: dict) -> T.ChartMeta:
    m = d["meta"]
    return T.ChartMeta(f"{SEASONS[OLD]['name']} → {SEASONS[NEW]['name']}", None,
                       date_range=f"{m['players']:,} players  ·  data to {m['exported']}")


def _phases(progress: float):
    return (T.ease_out_cubic(T.phase(progress, 0, 0.15)), T.ease_out_cubic(T.phase(progress, 0.15, 0.85)),
            T.ease_out_cubic(T.phase(progress, 0.8, 0.98)))


def _note(fig, text: str, alpha: float) -> None:
    y = T.FOOTER_LINE_Y + 9 / T.CARD_PX
    fig.text(T.margin_x(fig), y, text, ha="left", va="bottom", color=T.TEXT_MUTED, fontsize=T.px_to_pt(14),
             alpha=alpha)
    fig._content_bottom = y + 28 / T.CARD_PX


def _style_ticks(ax, alpha: float, size: float | None = None) -> None:
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_alpha(alpha)
        tl.set_color(T.TEXT)
        tl.set_fontsize(T.px_to_pt(size or T.SIZE_TICK))


# =============================================================================
# CARDS
# =============================================================================
def pct_change(old: float, new: float) -> str:
    """'+16.7%' (relative to the old value); '' when unchanged."""
    if old == new or not old:
        return ""
    return f"{(new - old) / abs(old) * 100:+.1f}%".replace(".0%", "%")


def rules_rows() -> list[tuple[str, str, str, str]]:
    """(label, old, new, % change) per rule; % change is blank for rules that aren't numbers."""
    a, b = SEASONS[OLD], SEASONS[NEW]
    d = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%b %-d")  # noqa: E731
    weeks = lambda s: (s["end"] - s["start"]) / 86400 / 7  # noqa: E731
    wins = lambda k: rank_size(k) / SEASONS[k]["points"]  # noqa: E731
    return [
        ("Points per win / loss", f"{a['points']}", f"{b['points']}", pct_change(a["points"], b["points"])),
        ("Last tier of a rank (VI)", f"{a['tier_steps'][-1]:,} pts", f"{b['tier_steps'][-1]:,} pts",
         pct_change(a["tier_steps"][-1], b["tier_steps"][-1])),
        ("Points per rank", f"{rank_size(OLD):,}", f"{rank_size(NEW):,}", pct_change(rank_size(OLD), rank_size(NEW))),
        ("Net wins to climb a rank", f"{wins(OLD):.1f}", f"{wins(NEW):.1f}", pct_change(wins(OLD), wins(NEW))),
        ("Shields per tier", f"{a['tier_shields']}", f"{b['tier_shields']}",
         pct_change(a["tier_shields"], b["tier_shields"])),
        ("Shields per rank", f"{a['rank_shields']}", f"{b['rank_shields']}",
         pct_change(a["rank_shields"], b["rank_shields"])),
        ("Shields stop at", RANK_NAMES[a['shields_until'] // 10 - 1], RANK_NAMES[b['shields_until'] // 10 - 1], ""),
        ("Highest placement", badge_name(a['placement_cap']), badge_name(b['placement_cap']), ""),
        ("Placements (returning players)", f"{a['placements']}", "none", "-100%"),
        ("Season length", f"{weeks(a):.0f} weeks", f"~{weeks(b):.0f} weeks", pct_change(round(weeks(a)), round(weeks(b)))),
    ]


def render_rules(d: dict, progress: float = 1.0, highlight=None):
    p_frame, p_rows, _ = _phases(progress)
    fig = T.plt.figure(figsize=(T.CARD_IN, T.CARD_IN), dpi=T.DPI, facecolor=T.BG)
    a, b = SEASONS[OLD], SEASONS[NEW]
    faster = 100 * (1 - (rank_size(NEW) / b["points"]) / (rank_size(OLD) / a["points"]))
    T.draw_header(fig, f"WHAT CHANGED IN {b['name'].upper()}",
                  f"ranks take {faster:.0f}% fewer net wins, but shields are much rarer", alpha=p_frame)
    T.draw_footer(fig, chart_meta(d), alpha=p_frame)
    _note(fig, f"from the game's ranked-season data  ·  'shields stop at': no shields from that rank up  ·  "
          f"{b['name']} ends {datetime.fromtimestamp(b['end'], timezone.utc):%b %-d, %Y}",
          p_frame)
    rows = rules_rows()
    x0 = T.MARGIN_X
    col = [x0, 0.47, 0.66, 1 - x0]
    top, row_h = 0.80, 0.064
    for j, txt in enumerate(["", a["name"].upper(), b["name"].upper(), "CHANGE"]):
        fig.text(col[j], top + 0.01, txt, color=T.TEXT_MUTED, fontsize=T.px_to_pt(20), fontweight="bold",
                 ha="right" if j == 3 else "left", va="bottom", alpha=p_frame)
    for i, (label, v1, v2, pct) in enumerate(rows):
        t = T.ease_out_cubic(T.phase(p_rows, i / len(rows) * 0.8, i / len(rows) * 0.8 + 0.2))
        y = top - (i + 0.5) * row_h - 0.008
        if i % 2 == 0:
            fig.patches.append(T.plt.Rectangle((x0 - 0.008, y - row_h / 2), 1 - 2 * x0 + 0.016, row_h,
                                               transform=fig.transFigure, color=T.PANEL, alpha=0.7 * t, zorder=0))
        changed = v1 != v2
        fig.text(col[0], y, label, color=T.TEXT, fontsize=T.px_to_pt(23), ha="left", va="center", alpha=t)
        fig.text(col[1], y, v1, color=T.TEXT_MUTED if changed else T.TEXT, fontsize=T.px_to_pt(23),
                 ha="left", va="center", alpha=t)
        fig.text(col[2], y, v2, color=T.GOLD if changed else T.TEXT, fontsize=T.px_to_pt(23 if len(v2) < 14 else 19),
                 fontweight="bold" if changed else "normal", ha="left", va="center", alpha=t)
        if pct:
            fig.text(col[3], y, pct, color=T.TEXT, fontsize=T.px_to_pt(23), fontweight="bold", ha="right",
                     va="center", alpha=t)
    return fig


def render_by_rank(d: dict, progress: float = 1.0, highlight=None):
    df = d["steps"]
    t = by_rank(df)
    tot = df["players"].sum()
    up = 100 * df.loc[df["change"] > 0, "players"].sum() / tot
    down = 100 * df.loc[df["change"] < 0, "players"].sum() / tot
    p_frame, p_bars, p_lab = _phases(progress)
    fig, ax = T.new_card()
    T.draw_header(fig, "THE SOFT RESET PULLED EVERYONE TO THE MIDDLE",
                  f"{up:.0f}% of players started {SEASONS[NEW]['name']} higher than they ended "
                  f"{SEASONS[OLD]['name']}, {down:.0f}% lower", alpha=p_frame)
    T.draw_footer(fig, chart_meta(d), alpha=p_frame)
    _note(fig, f"average tiers moved (6 tiers = 1 rank)  ·  last {SEASONS[OLD]['name']} rank (last "
          f"{S1_LOOKBACK_DAYS} days) vs first {SEASONS[NEW]['name']} rank  ·  players who already played both",
          p_frame)
    y = np.arange(len(t))
    vals = t["mean"].to_numpy(float)
    colors = [UP_COLOR if v > 0 else DOWN_COLOR for v in vals]
    ax.barh(y, np.nan_to_num(vals) * p_bars, 0.68, color=colors, zorder=3)
    ax.axvline(0, color=T.TEXT_MUTED, lw=1.5, zorder=4)
    lim = np.nanmax(np.abs(vals)) * 1.15
    ax.set_xlim(-lim, lim)
    for yi, (r, row) in zip(y, t.iterrows()):
        if np.isnan(row["mean"]) or p_lab <= 0:
            continue
        v = row["mean"]
        share = row["up"] if v > 0 else row["down"]
        txt = f"{v:+.1f} tiers  ({share:.0f}% {'up' if v > 0 else 'down'})"
        # on the other side of zero from the bar, so long bars never run into the rank names
        ax.text(-0.3 if v > 0 else 0.3, yi, txt, ha="right" if v > 0 else "left", va="center",
                color=UP_COLOR if v > 0 else DOWN_COLOR, fontsize=T.px_to_pt(17), fontweight="bold",
                alpha=p_lab, path_effects=T.STROKE, zorder=5)
    ax.set_yticks(y, [f"{RANK_NAMES[r - 1]}  ({int(n):,})" for r, n in zip(t.index, t["n"].fillna(0))])
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
    ax.set_xlabel(f"tiers moved going into {SEASONS[NEW]['name']}", color=T.TEXT_MUTED,
                  fontsize=T.px_to_pt(17), alpha=p_frame)
    ax.set_ylabel(f"rank at the end of {SEASONS[OLD]['name']}  (players)", color=T.TEXT_MUTED,
                  fontsize=T.px_to_pt(17), alpha=p_frame)
    ax.grid(axis="y", visible=False)
    _style_ticks(ax, p_frame)
    return fig


def render_flow(d: dict, progress: float = 1.0, highlight=None):
    df = d["steps"]
    m = df.pivot_table(index="r1", columns="r2", values="players", aggfunc="sum", fill_value=0)
    m = m.reindex(index=range(1, 12), columns=range(1, 12), fill_value=0)
    top_new = int(df.loc[df["players"] > 0, "r2"].max())
    m = m.loc[:, m.columns <= top_new]                 # no empty columns above the highest new-season rank
    pct = m.div(m.sum(axis=1).replace(0, np.nan), axis=0) * 100
    p_frame, p_cells, p_lab = _phases(progress)
    fig, ax = T.new_card()
    same = 100 * df.loc[df["r1"] == df["r2"], "players"].sum() / df["players"].sum()
    T.draw_header(fig, "WHERE EVERY RANK LANDED",
                  f"only {same:.0f}% kept their rank  ·  nobody starts {SEASONS[NEW]['name']} above "
                  f"{RANK_NAMES[top_new - 1]}", alpha=p_frame)
    T.draw_footer(fig, chart_meta(d), alpha=p_frame)
    _note(fig, f"row = rank at the end of {SEASONS[OLD]['name']}  ·  cell = share of that rank starting "
          f"{SEASONS[NEW]['name']} in each rank  ·  outlined = same rank", p_frame)
    rows, cols = list(pct.index), list(pct.columns)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("gold", [T.PANEL, T.GOLD])
    ax.imshow(np.nan_to_num(pct.to_numpy()) * p_cells, cmap=cmap, vmin=0, vmax=80, aspect="auto",
              origin="lower", zorder=2)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = pct.loc[r, c]
            if r == c:
                ax.add_patch(T.plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=T.TEXT, lw=1.6,
                                             alpha=p_frame, zorder=4))
            if not np.isnan(v) and v >= 2 and p_lab > 0:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=T.px_to_pt(15),
                        fontweight="bold", color=T.BG if v >= 35 else T.TEXT, alpha=p_lab, zorder=5)
    ax.set_xticks(range(len(cols)), [RANK_NAMES[c - 1] for c in cols], rotation=35, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(len(rows)), [RANK_NAMES[r - 1] for r in rows])
    ax.set_xlabel(f"rank at the start of {SEASONS[NEW]['name']}", color=T.TEXT_MUTED,
                  fontsize=T.px_to_pt(17), alpha=p_frame)
    ax.set_ylabel(f"rank at the end of {SEASONS[OLD]['name']}", color=T.TEXT_MUTED,
                  fontsize=T.px_to_pt(17), alpha=p_frame)
    ax.grid(visible=False)
    for s in ax.spines.values():
        s.set_visible(False)
    _style_ticks(ax, p_frame, 15)
    return fig


def _shares(df: pd.DataFrame) -> tuple[float, float, float]:
    """% of players who started the new season higher / lower / in the same rank."""
    tot = df["players"].sum()
    up = 100 * df.loc[df["change"] > 0, "players"].sum() / tot
    down = 100 * df.loc[df["change"] < 0, "players"].sum() / tot
    same_rank = 100 * df.loc[df["r1"] == df["r2"], "players"].sum() / tot
    return up, down, same_rank


def _tile(fig, x: float, y: float, w: float, h: float, value: str, label: str, color: str, alpha: float,
          value_px: float, label_px: float = 24) -> None:
    from matplotlib.patches import FancyBboxPatch
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                  transform=fig.transFigure, fc=T.PANEL, ec=color, lw=2.5, alpha=alpha, zorder=1))
    vpx = min(value_px, w * T.CARD_PX * 0.86 / max(T.display_w_per_px(value), 1e-6))
    T.outlined_text(fig, x + w / 2, y + h * 0.60, value, vpx, color, alpha, display=True, ha="center",
                    va="center", width=4)
    lines = textwrap.fill(label, 26)
    fig.text(x + w / 2, y + h * 0.20, lines, ha="center", va="center", color=T.TEXT, alpha=alpha,
             fontsize=T.px_to_pt(label_px), linespacing=1.15)


def render_opener(d: dict, progress: float = 1.0, highlight=None):
    """Hook card: the one number that sells the video."""
    p_frame, p_main, _ = _phases(progress)
    up, down, _ = _shares(d["steps"])
    fig = T.plt.figure(figsize=(T.CARD_IN, T.CARD_IN), dpi=T.DPI, facecolor=T.BG)
    T.draw_header(fig, f"{SEASONS[NEW]['name'].upper()} IS LIVE",
                  f"new rules, a soft reset, and a balance patch hours before it", alpha=p_frame)
    T.draw_footer(fig, chart_meta(d), alpha=p_frame)
    T.outlined_text(fig, 0.5, 0.56, f"{up:.0f}%", 300 * (0.85 + 0.15 * p_main), UP_COLOR, p_main, display=True,
                    ha="center", va="center", width=6)
    fig.text(0.5, 0.33, f"of players started {SEASONS[NEW]['name']}\nHIGHER than they ended {SEASONS[OLD]['name']}",
             ha="center", va="center", color=T.TEXT, fontsize=T.px_to_pt(40), fontweight="bold", alpha=p_main,
             linespacing=1.25)
    fig.text(0.5, 0.21, f"({down:.0f}% started lower)", ha="center", va="center", color=T.TEXT_MUTED,
             fontsize=T.px_to_pt(26), alpha=p_main)
    return fig


def render_closer(d: dict, progress: float = 1.0, highlight=None):
    """Wrap-up card for the whole video: the season's four takeaways."""
    p_frame, p_tiles, _ = _phases(progress)
    df = d["steps"]
    up, down, same_rank = _shares(df)
    a, b = SEASONS[OLD], SEASONS[NEW]
    wins = lambda k: rank_size(k) / SEASONS[k]["points"]  # noqa: E731
    t = by_rank(df)
    top = t["mean"].dropna().iloc[-1]
    tiles = [
        (f"{(wins(NEW) / wins(OLD) - 1) * 100:+.0f}%", "net wins needed to climb a rank", UP_COLOR),
        (f"{b['tier_shields']} / {b['rank_shields']}",
         f"shields per tier / rank (was {a['tier_shields']} / {a['rank_shields']})", DOWN_COLOR),
        (f"{up:.0f}%", f"started higher than they ended {a['name']}", UP_COLOR),
        (f"{top:+.0f}", f"tiers for the average {RANK_NAMES[-1]} player", DOWN_COLOR),
    ]
    fig = T.plt.figure(figsize=(T.CARD_IN, T.CARD_IN), dpi=T.DPI, facecolor=T.BG)
    T.draw_header(fig, f"{b['name'].upper()} IN FOUR NUMBERS",
                  "faster climbing, fewer safety nets, and a ladder squeezed to the middle", alpha=p_frame)
    T.draw_footer(fig, chart_meta(d), alpha=p_frame)
    x0, gap = T.MARGIN_X, 24 / T.CARD_PX
    w = (1 - 2 * x0 - gap) / 2
    y_top, y_bot = 0.84, 0.10
    h = (y_top - y_bot - gap) / 2
    for k, (value, label, color) in enumerate(tiles):
        tk = T.ease_out_cubic(T.phase(p_tiles, k * 0.18, k * 0.18 + 0.35))
        col, row = k % 2, k // 2
        _tile(fig, x0 + col * (w + gap), y_top - (row + 1) * h - row * gap, w, h, value, label, color, tk,
              value_px=150)
    return fig


def cards(data_dir: Path, **_) -> dict:
    data_dir = Path(data_dir)
    if not (data_dir / "season_reset_meta.json").exists():
        raise FileNotFoundError(data_dir / "season_reset_meta.json")
    d = load(data_dir)
    out = {"s2_opener": lambda progress=1.0, highlight=None: render_opener(d, progress),
           "s2_closer": lambda progress=1.0, highlight=None: render_closer(d, progress),
           "s2_rules": lambda progress=1.0, highlight=None: render_rules(d, progress),
           "s2_reset_by_rank": lambda progress=1.0, highlight=None: render_by_rank(d, progress),
           "s2_reset_flow": lambda progress=1.0, highlight=None: render_flow(d, progress)}
    for name, secs in BUILD_S.items():          # simple cards build faster than animate's 6 s default
        out[name].build_s = secs
    return out


BUILD_S = {"s2_opener": 2.5, "s2_closer": 3.0, "s2_rules": 4.0}


def summary(data_dir: Path) -> str:
    d = load(Path(data_dir))
    t = by_rank(d["steps"])
    lines = [f"{RANK_NAMES[r - 1]:>10}: {row['mean']:+.1f} tiers  ({row['up']:.0f}% up, {row['down']:.0f}% down, "
             f"n={int(row['n']):,})" for r, row in t.dropna().iterrows()]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "cards", "summary"])
    ap.add_argument("dir", type=Path, nargs="?", default=T.ANALYSES_ROOT / "season_2_start")
    a = ap.parse_args()
    if a.command == "export":
        export(a.dir)
    elif a.command == "summary":
        print(summary(a.dir))
    else:
        out = a.dir / "graphs"
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("s2_*.png"):
            old.unlink()
        for name, draw in cards(a.dir).items():
            T.save_card(draw(1.0), out / f"{name}.png")
        print(f"Saved cards to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())