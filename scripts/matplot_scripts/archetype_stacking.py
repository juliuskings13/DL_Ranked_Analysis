"""
archetype_stacking.py — 1080x1080 card: what happens to win rate when your team has
N more heroes of an archetype than the enemy team. One line per archetype, 95% CI bands.

Input: hero_type_diff_winrate.csv (hero_type, type_differential, match_frequency,
       winrate_of_surplus_team) from query_transform.py.

Usage:
    python3 archetype_stacking.py [week_dir] [--highlight marksman,assassin] [--progress 0.5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

import theme as T
from chart_stats import wilson_ci
from chart_stats import Insight, wr_confidence

# --- CONFIG ------------------------------------------------------------------
CSV_NAME = "hero_type_diff_winrate.csv"
OUTPUT_NAME = "archetype_stacking.png"
MIN_BUCKET_N = 200          # buckets with fewer matches are hidden (footer says so)
PLURAL = {"brawler": "BRAWLERS", "marksman": "MARKSMEN",
          "assassin": "ASSASSINS", "mystic": "MYSTICS"}
LINE_W = 4.5
DIM_ALPHA = 0.15
LABEL_GAP_PX = 30           # min vertical spacing between end-of-line labels
BAND_PAD_PCT = 2.0          # y room past the most extreme point (bands may run off)
END_LABEL_PX = 24           # "Assassins 52.5%" labels at the end of each line
ZONE_LABEL_PX = 20          # WINS MORE / LOSES MORE / band key


# --- DATA --------------------------------------------------------------------
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"type_differential": "diff", "match_frequency": "n",
                            "winrate_of_surplus_team": "wr"})
    df["hero_type"] = df["hero_type"].str.lower()
    df = df[df["diff"] > 0]                       # +0 is 50% by definition
    lo, hi = wilson_ci(df["wr"] / 100 * df["n"], df["n"])
    df["wr_lo"], df["wr_hi"] = lo * 100, hi * 100
    df["shown"] = df["n"] >= MIN_BUCKET_N
    return df.sort_values(["hero_type", "diff"]).reset_index(drop=True)


def headline(df: pd.DataFrame) -> tuple[str, str]:
    """Title = the most *confident* effect: rank by the CI bound nearest 50%, so a
    noisy small bucket can't win the headline just by having an extreme point."""
    s = df[df["shown"]].copy()
    s["dev"] = np.where(s["wr"] >= 50, (s["wr_lo"] - 50).clip(lower=0),
                        (s["wr_hi"] - 50).clip(upper=0))
    worst, best = s.loc[s["dev"].idxmin()], s.loc[s["dev"].idxmax()]
    lead = worst if abs(worst["dev"]) >= abs(best["dev"]) else best
    verb = "LOSES" if lead["dev"] < 0 else "WINS"
    title = f"STACKING {PLURAL.get(lead['hero_type'], lead['hero_type'].upper())} {verb} GAMES"
    sub = (f"+{int(best['diff'])} {best['hero_type']}s: {best['wr']:.1f}% win rate  ·  "
           f"+{int(worst['diff'])} {worst['hero_type']}s: {worst['wr']:.1f}%")
    sub = sub.replace("marksmans", "marksmen")
    return title, sub


# --- RENDER ------------------------------------------------------------------
def end_label(htype: str, wr: float) -> str:
    return f"{PLURAL.get(htype, htype.upper()).capitalize()}  {wr:.1f}%"


def _spread_labels(ys_px: list[float], gap: float) -> list[float]:
    """Push end labels apart vertically so they never overlap (keeps order)."""
    order = np.argsort(ys_px)
    out = np.array(ys_px, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out.tolist()


def render(df: pd.DataFrame, meta: T.ChartMeta, progress: float = 1.0,
           highlight: set[str] | None = None, title: str | None = None,
           subtitle: str | None = None) -> "T.Figure":
    """progress: frame fades in, then each archetype's line draws left→right (staggered)."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0.0, 0.15))
    auto_title, auto_sub = headline(df)

    fig, ax = T.new_card()
    T.draw_header(fig, title or auto_title, subtitle or auto_sub, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    shown = df[df["shown"]]
    max_diff = int(shown["diff"].max())
    # Limits follow the points, not the bands: wide bands simply run off the card,
    # which still reads as "uncertain" without squashing the confident buckets.
    y_lo = min(shown["wr"].min(), 50) - BAND_PAD_PCT
    y_hi = max(shown["wr"].max(), 50) + BAND_PAD_PCT
    ax.set_ylim(y_lo, y_hi)
    x_min, x_max = -0.15, max_diff + 0.15
    renderer = fig.canvas.get_renderer()
    ax_w = ax.get_window_extent(renderer).width
    for htype, g in shown.groupby("hero_type"):     # widen until every end label fits
        last = g.iloc[-1]
        probe = ax.text(0, 0, end_label(htype, last["wr"]),
                        fontsize=T.px_to_pt(END_LABEL_PX), fontweight="bold")
        c = (probe.get_window_extent(renderer).width + 18 + 14) / ax_w
        probe.remove()
        x_max = max(x_max, (float(last["diff"]) - c * x_min) / (1 - c))
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1 if (y_hi - y_lo) <= 10 else 2))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "even" if v == 0 else (f"+{v:.0f}" if 0 < v <= max_diff else "")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("More heroes of that class than the enemy team", labelpad=12)
    ax.set_ylabel("Win rate of the stacked team", labelpad=10)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_alpha(p_frame)

    # Advantage / liability zones
    ax.axhspan(50, y_hi, color=T.GREEN, alpha=0.05 * p_frame, zorder=0.5, lw=0)
    ax.axhspan(y_lo, 50, color=T.RED, alpha=0.05 * p_frame, zorder=0.5, lw=0)
    ax.axhline(50, color=T.TEXT_MUTED, lw=1.6, alpha=0.6 * p_frame, zorder=1)
    ax.text(0.012, 0.985, "WINS MORE", transform=ax.transAxes, color=T.GREEN, fontweight="bold",
            fontsize=T.px_to_pt(ZONE_LABEL_PX), ha="left", va="top", alpha=0.8 * p_frame)
    ax.text(0.012, 0.015, "LOSES MORE", transform=ax.transAxes, color=T.RED, fontweight="bold",
            fontsize=T.px_to_pt(ZONE_LABEL_PX), ha="left", va="bottom", alpha=0.8 * p_frame)
    # what the shaded bands mean, for viewers (see README / video script)
    ax.text(0.988, 0.015, "shaded band = where the true win rate likely is (95%)\n"
            "wider band = fewer games at that stack size", transform=ax.transAxes, color=T.TEXT_MUTED,
            fontsize=T.px_to_pt(ZONE_LABEL_PX * 0.8), ha="right", va="bottom", alpha=0.9 * p_frame,
            linespacing=1.25)

    types = [t for t in T.ARCHETYPE_COLORS if t in set(df["hero_type"])]
    ends = []                                       # (type, x, y, alpha, text)
    for k, htype in enumerate(types):
        g = shown[shown["hero_type"] == htype]
        if g.empty:
            continue
        s = 0.15 + 0.12 * k
        t = T.ease_out_cubic(T.phase(progress, s, s + 0.45))
        if t <= 0:
            continue
        hl = highlight is None or htype in highlight
        a = 1.0 if hl else DIM_ALPHA
        col = T.archetype_color(htype)

        xs = np.concatenate([[0], g["diff"].to_numpy(float)])
        ys = np.concatenate([[50], g["wr"].to_numpy(float)])
        lo = np.concatenate([[50], g["wr_lo"].to_numpy(float)])
        hi = np.concatenate([[50], g["wr_hi"].to_numpy(float)])

        # draw up to x_end (partial last segment during animation)
        x_end = xs[-1] * t
        keep = xs <= x_end
        px, py, plo, phi = xs[keep], ys[keep], lo[keep], hi[keep]
        if x_end > px[-1] and keep.sum() < len(xs):
            j = keep.sum()
            f = (x_end - xs[j - 1]) / (xs[j] - xs[j - 1])
            px = np.append(px, x_end)
            py = np.append(py, ys[j - 1] + f * (ys[j] - ys[j - 1]))
            plo = np.append(plo, lo[j - 1] + f * (lo[j] - lo[j - 1]))
            phi = np.append(phi, hi[j - 1] + f * (hi[j] - hi[j - 1]))

        ax.fill_between(px, plo, phi, color=col, alpha=0.16 * a, lw=0, zorder=2)
        ax.plot(px, py, color=col, lw=LINE_W, alpha=a, solid_capstyle="round",
                zorder=6 if hl else 3)
        dots = xs[1:][xs[1:] <= x_end]
        ax.scatter(dots, ys[1:][: len(dots)], s=T.px_to_pt(16) ** 2, color=col,
                   edgecolors=T.BG, linewidths=2, alpha=a, zorder=7 if hl else 4)

        if t >= 1.0:
            last = g.iloc[-1]
            ends.append((htype, float(last["diff"]), float(last["wr"]), a,
                         end_label(htype, last["wr"])))

    # End-of-line labels, spread so they never collide
    if ends:
        to_px = ax.transData.transform
        ys_px = [to_px((x, y))[1] for _, x, y, _, _ in ends]
        spread = _spread_labels(ys_px, LABEL_GAP_PX)
        inv = ax.transData.inverted()
        for (htype, x, y, a, text), y_px in zip(ends, spread):
            x_px = to_px((x, y))[0] + 18
            lx, ly = inv.transform((x_px, y_px))
            ax.text(lx, ly, text, color=T.archetype_color(htype), fontweight="bold",
                    fontsize=T.px_to_pt(END_LABEL_PX), ha="left", va="center",
                    alpha=a, zorder=9, path_effects=T.STROKE)
    return fig


# --- CLI ---------------------------------------------------------------------
def insights(data_dir: Path, **_) -> list[Insight]:
    """The most extreme reliable stacking result per archetype; insights.py keeps the strongest."""
    df = prepare(pd.read_csv(data_dir / CSV_NAME))
    df = df[df["shown"]].assign(conf=lambda d: wr_confidence(d["wr"], d["n"]))
    df = df[(df["wr_hi"] < 50) | (df["wr_lo"] > 50)]
    out = []
    for htype, g in df.groupby("hero_type"):
        r = g.loc[(g["wr"] - 50).abs().idxmax()]
        plural = "marksmen" if htype == "marksman" else f"{htype}s"
        out.append(Insight(Path(OUTPUT_NAME).stem, "stacking", highlight=(htype,),
                           headline=f"Teams with {int(r['diff'])} more {plural} than the enemy win {r['wr']:.1f}%",
                           detail=f"{int(r['n']):,} matches; 95% range {r['wr_lo']:.1f}–{r['wr_hi']:.1f}%",
                           effect=abs(r["wr"] - 50), confidence=float(r["conf"])))
    return out

def cards(data_dir: Path, **_) -> dict:
    """{output name: draw(progress=1.0, highlight=None) -> Figure} for every card this script makes."""
    df = prepare(pd.read_csv(data_dir / CSV_NAME))
    meta = T.ChartMeta(week_label=T.week_label_from_dir(data_dir), min_n=MIN_BUCKET_N,
                       min_n_unit="matches per point")
    return {Path(OUTPUT_NAME).stem: lambda progress=1.0, highlight=None: render(df, meta, progress=progress,
                                                                                 highlight=highlight)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--highlight", default="", help='archetypes, e.g. "marksman,assassin"')
    ap.add_argument("--progress", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    highlight = T.parse_names(args.highlight)
    (name, draw), = cards(data_dir).items()
    out = Path(args.out) if args.out else data_dir / "graphs" / f"{name}.png"
    T.save_card(draw(args.progress, highlight), out)
    df = prepare(pd.read_csv(data_dir / CSV_NAME))
    hidden = df[~df["shown"]]
    print(f"Saved {out}  | hidden buckets (<{MIN_BUCKET_N} matches): "
          + ", ".join(f"{r.hero_type}+{int(r['diff'])}({int(r.n)})" for _, r in hidden.iterrows()))


if __name__ == "__main__":
    main()