"""
hero_duo_synergy_table.py — duo synergy cards (1080x1080).

A duo of two strong heroes wins a lot even with zero synergy, so raw duo win rate mostly
re-measures solo strength. Here each duo is compared to what its two heroes' solo win rates
predict (log-odds added, see chart_stats.expected_pair_wr):

    lift = actual duo WR − expected duo WR

Outputs:
    hero_duo_synergy_best.png    10 duos with the most positive lift
    hero_duo_synergy_worst.png   10 duos with the most negative lift

Selection vs display:
    which 10 duos appear = the 10 with the best *pessimistic* lift (the end of the 95% range
    nearest zero), so a small-sample pair can't make the list on luck alone;
    the order they're shown in = --sort (default: lift, so the column reads top to bottom).
Color = confidence that the synergy is real, P(true lift > 0) for the best card (< 0 for the
worst): gray at 50% (coin flip) fading to full green / red at 97.5%+. With ~700 duos, a few
will look confident by chance alone, so a duo that stays strong several weeks running is the
real finding.
Needs hero_duo_synergies.csv with ALL duos ≥ MIN_DUO_N matches (see README note / SQL).

Usage: python3 hero_duo_synergy_table.py [week_dir] [--highlight "Haze,Paige"]
"""
from __future__ import annotations

import argparse
from math import erf, sqrt

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import theme as T
from chart_stats import expected_pair_wr, load_hero_meta, total_matches_from_hero_rows

CSV_NAME = "hero_duo_synergies.csv"
MIN_DUO_N = 1000
ROWS = 10
DIM_ALPHA = 0.25
MIN_LIFT_COL_PX = 240        # shrink fonts if the lift column would get narrower than this
WRAP_BELOW_PX = 21           # ...and wrap duo names onto two lines rather than go below this
HEADERS = {"games": "GAMES", "wr": "WIN %", "exp": "EXPECTED", "lift": "LIFT (pts)"}
HDR_PX = 17
CONF_FULL = 0.975            # confidence at which the color reaches full green / red
CONF_CURVE = 1.8             # >1 keeps weak evidence grayer (75% ≈ 30% color, 90% ≈ 70%)
SORT_KEYS = {"lift": "lift", "confidence": "conf", "games": "n", "wr": "wr"}


def prepare(data_dir) -> pd.DataFrame:
    heroes = load_hero_meta(data_dir).set_index("hero_name")
    df = pd.read_csv(data_dir / CSV_NAME)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"matches_together": "n", "duo_winrate": "wr"})
    df = df[df["n"] >= MIN_DUO_N].copy()
    df["wr_a"] = df["hero_name_a"].map(heroes["winrate"])
    df["wr_b"] = df["hero_name_b"].map(heroes["winrate"])
    df["type_a"] = df["hero_name_a"].map(heroes["hero_type"])
    df["type_b"] = df["hero_name_b"].map(heroes["hero_type"])
    df = df.dropna(subset=["wr_a", "wr_b"])
    df["expected"] = expected_pair_wr(df["wr_a"], df["wr_b"])
    df["lift"] = df["wr"] - df["expected"]
    se = np.sqrt(df["wr"] / 100 * (1 - df["wr"] / 100) / df["n"]) * 100
    df["lift_lo"], df["lift_hi"] = df["lift"] - 1.96 * se, df["lift"] + 1.96 * se
    # confidence that the true lift is on the same side of zero as the measured lift
    df["conf"] = [0.5 * (1 + erf(abs(l) / s_ / sqrt(2))) for l, s_ in zip(df["lift"], se)]
    return df


def conf_color(lift: float, conf: float) -> str:
    """Gray at 50% confidence, full green (positive) / red (negative) at CONF_FULL and above."""
    t = float(np.clip((conf - 0.5) / (CONF_FULL - 0.5), 0, 1)) ** CONF_CURVE
    return T.blend(T.TEXT_MUTED, T.GREEN if lift >= 0 else T.RED, t)


def duo_label(a: str, b: str, wrap: bool) -> str:
    return f"{a} +\n{b}" if wrap else f"{a} + {b}"


def _name_width(rows: pd.DataFrame, font: float, wrap: bool) -> float:
    if wrap:
        return max(max(T.text_px(f"{a} +", font), T.text_px(b, font))
                   for a, b in zip(rows["hero_name_a"], rows["hero_name_b"]))
    return max(T.text_px(duo_label(a, b, False), font) for a, b in zip(rows["hero_name_a"], rows["hero_name_b"]))


def _columns(rows: pd.DataFrame, W: float, font: float, icon_px: float, wrap: bool) -> tuple[dict, float]:
    num = lambda col_vals, hdr: max(T.text_px(HEADERS[hdr], HDR_PX),
                                    max(T.text_px(v, font) for v in col_vals))
    widths = {
        "rank": T.text_px("10", font),
        "icons": 2 * icon_px + 6,
        "names": _name_width(rows, font, wrap),
        "games": num([f"{v / 1000:.1f}k" for v in rows["n"]], "games"),
        "wr": num([f"{v:.1f}" for v in rows["wr"]], "wr"),
        "exp": num([f"{v:.1f}" for v in rows["expected"]], "exp"),
        "lift": None,
    }
    cols = T.layout_columns(widths, W, gap_px=20)
    return cols, (cols["lift"][1] - cols["lift"][0]) * W


def render(rows: pd.DataFrame, meta: T.ChartMeta, title: str, subtitle: str,
           progress: float = 1.0, highlight: set[str] | None = None) -> "T.Figure":
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    fig, ax = T.new_table_card()
    T.draw_header(fig, title, subtitle, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)
    n = len(rows)
    row = T.table_rows(fig, ax, n, header_rows=1.0, footer_rows=0.55)
    W, _ = T.content_px(fig)

    # biggest font/icons that still leave the lift column at least MIN_LIFT_COL_PX wide.
    # If one-line names would force the font below WRAP_BELOW_PX, put "A +" / "B" on two lines.
    icon_px = min(row * 0.82, 66)
    wrap = False
    for wrap in (False, True):
        font = min(row * (0.40 if not wrap else 0.33), 26)
        cols, lift_px = _columns(rows, W, font, icon_px, wrap)
        while lift_px < MIN_LIFT_COL_PX and font > 13:
            font -= 1
            cols, lift_px = _columns(rows, W, font, icon_px, wrap)
        if font >= WRAP_BELOW_PX or wrap:
            break

    hdr = dict(color=T.GOLD, fontsize=T.px_to_pt(HDR_PX), fontweight="bold", va="center", alpha=p_frame)
    ax.text(cols["names"][0], -1.0, "DUO", ha="left", **hdr)
    for key in ("games", "wr", "exp"):
        ax.text(cols[key][1], -1.0, HEADERS[key], ha="right", **hdr)
    L0, L1 = cols["lift"]
    ax.text((L0 + L1) / 2, -1.0, HEADERS["lift"], ha="center", **hdr)

    # lift scale for THIS card: map the full data range (0, bars and whiskers, including any
    # whisker that crosses zero) onto the column, leaving room for the labels on the outer side.
    lab_font = min(max(font, 18) * 1.05, 26)
    label_w = (T.text_px("+10.0", lab_font) + 10) / W
    v_lo = min(0.0, float(rows["lift_lo"].min()))
    v_hi = max(0.0, float(rows["lift_hi"].max()))
    x_lo = L0 + (label_w if (rows["lift"] < 0).any() else 0.004)
    x_hi = L1 - (label_w if (rows["lift"] > 0).any() else 0.004)

    def lift_x(v):
        return x_lo + (np.asarray(v) - v_lo) / (v_hi - v_lo) * (x_hi - x_lo)

    x_mid = float(lift_x(0.0))

    ax.plot([x_mid] * 2, [-0.5, n - 0.5], color=T.TEXT_MUTED, lw=1.4, alpha=0.7 * p_frame, zorder=1)
    icx = (cols["icons"][0] + cols["icons"][1]) / 2
    icon_off = (icon_px / 2 + 3) / W

    for i, (_, r) in enumerate(rows.iterrows()):
        s = 0.15 + 0.5 * i / max(n - 1, 1)
        t = T.ease_out_cubic(T.phase(progress, s, s + 0.3))
        pair = {r["hero_name_a"].lower(), r["hero_name_b"].lower()}
        hl = highlight is None or bool(pair & highlight)
        a = (1.0 if hl else DIM_ALPHA) * t
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, i - 0.5), 1, 1, color=T.PANEL, alpha=0.55 * p_frame, zorder=0, lw=0))
        if t <= 0:
            continue
        fs = T.px_to_pt(font)
        ax.text(cols["rank"][0], i, f"{i + 1}", ha="left", va="center", color=T.TEXT_MUTED,
                fontsize=fs, fontweight="bold", alpha=a)
        for dx, name, ht in ((-icon_off, r["hero_name_a"], r["type_a"]), (icon_off, r["hero_name_b"], r["type_b"])):
            T.add_icon(ax, T.icon_any(name, icon_px, T.archetype_color(ht), a, wr=r["wr"]),   # duo WR
                       (icx + dx, i), zorder=6)
        ax.text(cols["names"][0], i, duo_label(r["hero_name_a"], r["hero_name_b"], wrap), ha="left",
                va="center", color=T.TEXT, fontsize=fs, fontweight="bold", alpha=a, linespacing=1.05)
        ax.text(cols["games"][1], i, f"{r['n'] / 1000:.1f}k", ha="right", va="center", color=T.TEXT_MUTED,
                fontsize=fs, alpha=a)
        ax.text(cols["wr"][1], i, f"{r['wr']:.1f}", ha="right", va="center", color=T.TEXT,
                fontsize=fs, fontweight="bold", alpha=a)
        ax.text(cols["exp"][1], i, f"{r['expected']:.1f}", ha="right", va="center", color=T.TEXT_MUTED,
                fontsize=fs, alpha=a)

        col = conf_color(r["lift"], r["conf"])
        lv = r["lift"] * t
        ax.add_patch(Rectangle((min(x_mid, lift_x(lv)), i - 0.3), abs(lift_x(lv) - x_mid), 0.6,
                               color=col, alpha=0.9 * a, lw=0, zorder=3))
        if t >= 1:
            ax.plot([lift_x(r["lift_lo"]), lift_x(r["lift_hi"])], [i, i], color=T.TEXT, lw=2.4,
                    alpha=0.45 * a, zorder=4)
            right = r["lift"] >= 0
            end = lift_x(r["lift_hi"] if right else r["lift_lo"])
            ax.text(end + (6 if right else -6) / W, i,
                    f"{r['lift']:+.1f}", ha="left" if right else "right", va="center", color=col,
                    fontsize=T.px_to_pt(lab_font), fontweight="bold", alpha=a, zorder=5, path_effects=T.STROKE)
    # key: what the shading means (gray = coin flip, full color = confident it's real)
    pos = (rows["lift"] >= 0).all()
    ky = n - 0.5 + 0.38
    steps = 24
    k0 = L0 + (L1 - L0) * 0.30
    k1 = L1
    for k in range(steps):
        c = T.blend(T.TEXT_MUTED, T.GREEN if pos else T.RED, (k / (steps - 1)) ** CONF_CURVE)
        ax.add_patch(Rectangle((k0 + (k1 - k0) * k / steps, ky - 0.11), (k1 - k0) / steps + 0.0005, 0.22,
                               color=c, lw=0, alpha=p_frame, zorder=3))
    kfs = T.px_to_pt(14)
    ax.text(k0 - 8 / W, ky, "confidence it's real:", ha="right", va="center", color=T.TEXT_MUTED,
            fontsize=kfs, alpha=p_frame)
    ax.text(k0, ky + 0.2, "50%", ha="left", va="top", color=T.TEXT_MUTED, fontsize=kfs, alpha=p_frame)
    ax.text(k1, ky + 0.2, f"{CONF_FULL * 100:g}%+", ha="right", va="top", color=T.TEXT_MUTED,
            fontsize=kfs, alpha=p_frame)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--highlight", default="")
    ap.add_argument("--progress", type=float, default=1.0)
    ap.add_argument("--sort", choices=list(SORT_KEYS), default="lift",
                    help="display order of the 10 selected duos")
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    df = prepare(data_dir)
    heroes = load_hero_meta(data_dir)
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), total_matches_from_hero_rows(heroes["total_matches"]),
                       min_n=MIN_DUO_N, min_n_unit="games per duo")
    hl = {h.strip() for h in args.highlight.split(",") if h.strip()} or None
    if len(df) < 2 * ROWS:
        print(f"Warning: only {len(df)} duos in {CSV_NAME}; export all duos ≥{MIN_DUO_N} games "
              "(drop the WR ≥ 50 / LIMIT 20 filters in extract_hero_pairings) for a fair ranking.")

    corr = np.corrcoef(df["expected"], df["lift"])[0, 1]
    if corr < -0.4:
        print(f"Warning: lift correlates with expected WR (r={corr:.2f}). Strong pairs look "
              "anti-synergistic across the board, so the expected-WR baseline is biased; "
              "check that hero_winrates.csv covers the same week/filters as the duo export.")

    # pick by the pessimistic end of the range (no ties), then display in the chosen order
    key = SORT_KEYS[args.sort]
    best = df[df["lift"] > 0].nlargest(ROWS, "lift_lo")
    best = best.sort_values(key, ascending=False)
    worst = df[df["lift"] < 0].nsmallest(ROWS, "lift_hi")
    worst = worst.sort_values(key, ascending=(key == "lift"))   # most negative lift first
    out = data_dir / "graphs"
    b0 = best.loc[best["lift"].idxmax()]
    T.save_card(render(best, meta, "DUOS THAT ACTUALLY SYNERGIZE",
                       f"{b0['hero_name_a']} + {b0['hero_name_b']} win {b0['lift']:+.1f} pts more than their solo win rates predict",
                       args.progress, hl), out / "hero_duo_synergy_best.png")
    w0 = worst.loc[worst["lift"].idxmin()]
    T.save_card(render(worst, meta, "DUOS THAT DRAG EACH OTHER DOWN",
                       f"{w0['hero_name_a']} + {w0['hero_name_b']}: {w0['lift']:+.1f} pts below what their solo win rates predict",
                       args.progress, hl), out / "hero_duo_synergy_worst.png")
    print(f"Saved duo synergy cards to {out}  ({len(df)} duos considered)")


if __name__ == "__main__":
    main()