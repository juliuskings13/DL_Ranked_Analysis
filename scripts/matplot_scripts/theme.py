"""
theme.py — shared look, paths and 1080x1080 card frame for every Deadlock chart.

Every chart script should:
    from theme import *            # or import theme as T
    fig, ax = new_card()
    draw_header(fig, title, subtitle)
    ... plot on ax ...
    draw_footer(fig, meta)
    save_card(fig, path)

Chart functions should accept `progress` (0..1) and `highlight` (set of names)
so the same code renders static PNGs (progress=1) and animation frames later.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# =============================================================================
# CANVAS
# =============================================================================
CARD_PX = 1080          # every chart is a 1080x1080 card
DPI = 100
CARD_IN = CARD_PX / DPI

# Figure-fraction layout of the card (0,0 = bottom-left)
HEADER_TITLE_Y = 0.950
HEADER_SUB_Y = 0.908
HEADER_LEGEND_Y = 0.872
PLOT_RECT = (0.115, 0.125, 0.850, 0.715)   # starting box only; fit_card() resizes it to fill the card
FOOTER_LINE_Y = 0.025                       # credits strip: ~27px tall
FOOTER_TEXT_Y = 0.0115
MARGIN_X = 0.037                            # ~40px side margins


def px_to_pt(px: float) -> float:
    """Matplotlib sizes are points; we think in card pixels."""
    return px * 72.0 / DPI


# =============================================================================
# PALETTE  (hex for matplotlib; use rgba() for Pillow)
# =============================================================================
BG = "#0f1717"            # deep teal-slate background
PANEL = "#162422"         # stripes / panels
GRID = "#1e2a2a"
DIVIDER = "#2d3c3c"
TEXT = "#f4f0e2"          # cream
TEXT_MUTED = "#b4b9c3"
FOOTER = "#8c919b"
GOLD = "#ffd073"          # titles / accents
GREEN = "#41d67e"         # above 50% / advantage
RED = "#ef6363"           # below 50% / liability
NEUTRAL = "#8a8f98"       # insufficient sample

# Background zone tints (scatter quadrants, balance tiers, duration bands). One value so every
# chart's zones read at the same strength; 0.15 is the level that's visible on video.
ZONE_TINT_ALPHA = 0.15
# Tier colors: (tint color, tint strength x ZONE_TINT_ALPHA, label color). Red deepens below 50%,
# green deepens above, gold marks balanced. Tier boundaries live in chart_stats.TIER_BANDS.
TIER_STYLE = {"TRASH":    ("#ef6363", 1.6, "#ef6363"),
              "WEAK":     ("#ef6363", 0.95, "#ef6363"),
              "UNDER":    ("#ef6363", 0.45, "#f4a3a3"),
              "BALANCED": ("#ffd073", 1.0, "#ffd073"),
              "OVER":     ("#41d67e", 0.45, "#9be8bb"),
              "STRONG":   ("#41d67e", 0.95, "#41d67e"),
              "GOD TIER": ("#41d67e", 1.6, "#41d67e")}

ARCHETYPE_COLORS = {
    "brawler": "#22c55e",
    "marksman": "#f97316",
    "assassin": "#06b6d4",
    "mystic": "#d21ff0",
}


def rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha


def blend(hex_a: str, hex_b: str, t: float) -> str:
    """Color t of the way from hex_a (t=0) to hex_b (t=1)."""
    a, b = np.array(rgba(hex_a)[:3], float), np.array(rgba(hex_b)[:3], float)
    r, g, bl = (a + (b - a) * float(np.clip(t, 0, 1))).round().astype(int)
    return f"#{r:02x}{g:02x}{bl:02x}"


def archetype_color(hero_type: str) -> str:
    return ARCHETYPE_COLORS.get(str(hero_type).strip().lower(), NEUTRAL)


# =============================================================================
# TYPOGRAPHY  (sizes in card pixels)
# =============================================================================
FONT_FAMILY = "DejaVu Sans"
SIZE_TITLE = 42
SIZE_SUBTITLE = 22
SIZE_AXIS = 21
SIZE_TICK = 18
SIZE_LABEL = 14
SIZE_LABEL_BIG = 19
SIZE_FOOTER = 14

STROKE = [pe.withStroke(linewidth=3, foreground=BG)]  # outline for text over data


def apply_mpl_theme() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": DIVIDER,
        "axes.labelcolor": TEXT,
        "axes.linewidth": 1.2,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": px_to_pt(SIZE_TICK),
        "ytick.labelsize": px_to_pt(SIZE_TICK),
        "axes.labelsize": px_to_pt(SIZE_AXIS),
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "text.color": TEXT,
    })


apply_mpl_theme()

# =============================================================================
# PATHS  (one copy of the week/dir logic instead of one per script)
# =============================================================================
RANKED_START_DATE = date(2026, 7, 30)
PROJECT_ROOT = Path(os.environ.get("DL_PROJECT_ROOT", Path.home() / "Projects" / "DL_ranked_analysis"))
ASSETS_BASE = Path(os.environ.get("DL_ASSETS_DIR", PROJECT_ROOT / "assets"))

# Hero icon variant chosen by the win rate the icon sits next to (solo, duo or team):
#   below ICON_WR_LOW            -> critical_icons/
#   ICON_WR_LOW … ICON_WR_HIGH   -> base_icons/      (inclusive, so exactly 50% is "base")
#   above ICON_WR_HIGH           -> gloat_icons/
ICON_WR_LOW, ICON_WR_HIGH = 49.0, 51.0
ICON_FOLDERS = {"critical": "critical_icons", "base": "base_icons", "gloat": "gloat_icons"}
ICON_DIR = ASSETS_BASE / ICON_FOLDERS["base"]


def get_current_week_number(target: date | None = None) -> int:
    return ((target or date.today()) - RANKED_START_DATE).days // 7


def resolve_target_dir(arg: str | None = None) -> Path:
    """CLI dir if given and valid, else the current week's folder."""
    candidate = arg if arg is not None else (sys.argv[1] if len(sys.argv) > 1 else None)
    if candidate and Path(candidate).is_dir():
        return Path(candidate)
    return PROJECT_ROOT / f"week{get_current_week_number()}_analysis"


def week_label_from_dir(data_dir: Path) -> str:
    """How charts name the period: period_meta.json's label (e.g. "Nurse Harrow, days 1–7"),
    else "Week N" from the folder name."""
    import json
    try:
        label = json.loads((Path(data_dir) / "period_meta.json").read_text()).get("label")
        if label:
            return label
    except (OSError, ValueError):
        pass
    m = re.search(r"week(\d+)", Path(data_dir).name)
    return f"Week {m.group(1)}" if m else Path(data_dir).name


# =============================================================================
# METADATA FOOTER
# =============================================================================
@dataclass
class ChartMeta:
    week_label: str
    n_matches: int | None = None
    min_n: int | None = None
    min_n_unit: str = "games"              # e.g. "matches per point"
    date_range: str | None = None          # e.g. "Oct 19–25" once the query exports it
    source: str = "Deadlock Community API"
    handle: str = "@eReys13"

    def left_text(self) -> str:
        parts = [self.source, self.week_label]
        if self.date_range:
            parts.append(self.date_range)
        if self.n_matches:
            parts.append(f"{self.n_matches:,} matches")
        if self.min_n:
            parts.append(f"min {self.min_n:,} {self.min_n_unit}")
        return "  ·  ".join(parts)


# =============================================================================
# CARD FRAME
# =============================================================================
def margin_x(fig: Figure) -> float:
    """Side margin as a figure fraction; a card can tighten it with fig._margin_x."""
    return getattr(fig, "_margin_x", MARGIN_X)


def new_card() -> tuple[Figure, plt.Axes]:
    fig = plt.figure(figsize=(CARD_IN, CARD_IN), dpi=DPI, facecolor=BG)
    ax = fig.add_axes(PLOT_RECT)
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)
    return fig, ax


def _mark_header_bottom(fig: Figure, artist) -> None:
    """Remember the lowest header element so fit_card() knows where content may start."""
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    bottom = bb.y0 / fig.bbox.height
    fig._content_top = min(getattr(fig, "_content_top", 1.0), bottom)


def draw_header(fig: Figure, title: str, subtitle: str | None = None, alpha: float = 1.0) -> None:
    mx = margin_x(fig)
    width = (1 - 2 * mx) * CARD_PX
    title_px = fit_font_px([title], width, max_px=SIZE_TITLE, min_px=26)   # long titles shrink to fit
    t = fig.text(mx, HEADER_TITLE_Y, title, color=GOLD, fontsize=px_to_pt(title_px),
                 fontweight="bold", va="center", ha="left", alpha=alpha)
    _mark_header_bottom(fig, t)
    if subtitle:
        s = fig.text(mx, HEADER_SUB_Y, subtitle, color=TEXT_MUTED,
                     fontsize=px_to_pt(SIZE_SUBTITLE), va="center", ha="left", alpha=alpha)
        _mark_header_bottom(fig, s)


def text_width_frac(fig: Figure, artist) -> float:
    """Measured width of a text artist as a fraction of the card width."""
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    return bb.width / fig.bbox.width


def draw_archetype_legend(fig: Figure, y: float = HEADER_LEGEND_Y, alpha: float = 1.0) -> float:
    """Draws the archetype key; returns the x (figure fraction) where it ends."""
    x = MARGIN_X
    size = px_to_pt(SIZE_LABEL + 2)
    for name, color in ARCHETYPE_COLORS.items():
        dot = fig.text(x, y, "●", color=color, fontsize=size + 2, va="center", ha="left", alpha=alpha)
        x += text_width_frac(fig, dot) + 0.006
        label = fig.text(x, y, name.capitalize(), color=TEXT, fontsize=size,
                         fontweight="bold", va="center", ha="left", alpha=alpha)
        _mark_header_bottom(fig, label)
        x += text_width_frac(fig, label) + 0.025
    return x


def draw_kpi_row(fig: Figure, items: list[tuple], height_px: float | None = None,
                 alpha: float = 1.0) -> None:
    """Row of stat tiles directly under the header.
    items = [(value, label, color), ...] or (value, label, color, sub) for a second caption
    line (e.g. the change vs last week). Fonts are sized to the tile; content placed
    afterwards starts below the row."""
    from matplotlib.patches import FancyBboxPatch
    has_sub = any(len(it) > 3 and it[3] for it in items)
    height_px = height_px or (140 if has_sub else 118)
    top = getattr(fig, "_content_top", 0.9) - GAP_BELOW_HEADER
    h = height_px / CARD_PX
    gap = 16 / CARD_PX
    n = len(items)
    w = (1 - 2 * MARGIN_X - gap * (n - 1)) / n
    w_px = w * CARD_PX
    v_px = fit_font_px([it[0] for it in items], w_px * 0.84, height_px * 0.36, max_px=52, min_px=20)
    l_px = fit_font_px([it[1] for it in items], w_px * 0.88, height_px * 0.17, max_px=20,
                       min_px=12, bold=False)
    subs = [it[3] for it in items if len(it) > 3 and it[3]]
    s_px = fit_font_px(subs, w_px * 0.88, height_px * 0.15, max_px=18, min_px=11) if subs else 0
    y_val, y_lab, y_sub = (0.34, 0.66, 0.86) if has_sub else (0.40, 0.80, None)
    for k, it in enumerate(items):
        value, label, color = it[:3]
        sub = it[3] if len(it) > 3 else None
        x = MARGIN_X + k * (w + gap)
        fig.add_artist(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                      transform=fig.transFigure, facecolor=PANEL, edgecolor="none",
                                      alpha=alpha))
        fig.text(x + w / 2, top - h * y_val, value, ha="center", va="center", color=color,
                 fontsize=px_to_pt(v_px), fontweight="bold", alpha=alpha)
        fig.text(x + w / 2, top - h * y_lab, label, ha="center", va="center", color=TEXT_MUTED,
                 fontsize=px_to_pt(l_px), alpha=alpha)
        if sub:
            fig.text(x + w / 2, top - h * y_sub, sub, ha="center", va="center", color=TEXT,
                     fontsize=px_to_pt(s_px), fontweight="bold", alpha=alpha)
    fig._content_top = top - h


def draw_footer(fig: Figure, meta: ChartMeta, alpha: float = 1.0) -> None:
    mx = margin_x(fig)
    fig.add_artist(plt.Line2D([mx, 1 - mx], [FOOTER_LINE_Y] * 2,
                              color=DIVIDER, linewidth=1, transform=fig.transFigure, alpha=alpha))
    fs = px_to_pt(SIZE_FOOTER)
    fig.text(mx, FOOTER_TEXT_Y, meta.left_text(), color=FOOTER, fontsize=fs,
             va="center", ha="left", alpha=alpha)
    fig.text(1 - mx, FOOTER_TEXT_Y, f"Analysis by {meta.handle}", color=FOOTER,
             fontsize=fs, va="center", ha="right", alpha=alpha)


GAP_BELOW_HEADER = 0.022   # ~24px between the header block and the content
GAP_ABOVE_FOOTER = 0.018   # ~19px between the content and the footer rule


def content_bounds(fig: Figure) -> tuple[float, float]:
    """(bottom, top) of the content area as figure fractions. A card can override either edge:
    fig._content_bottom (e.g. to put its own notes right above the footer rule) and
    fig._gap_below_header (a tighter gap under the header)."""
    top = getattr(fig, "_content_top", 0.97) - getattr(fig, "_gap_below_header", GAP_BELOW_HEADER)
    bottom = getattr(fig, "_content_bottom", FOOTER_LINE_Y + GAP_ABOVE_FOOTER)
    return bottom, top


def fit_card(fig: Figure, iterations: int = 4) -> None:
    """Grow/shrink the plot area so everything it draws (tick labels, axis titles, icons,
    out-of-axes text) exactly fills the space between the side margins, header and footer.

    Replaces the old fixed PLOT_RECT, which left ~125px empty on the left of every card
    that didn't need y-axis labels. Runs automatically in save_card() / fig_to_image()."""
    renderer = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    left, right = margin_x(fig), 1 - margin_x(fig)
    bottom, top = content_bounds(fig)
    for ax in fig.axes:
        if getattr(ax, "_no_fit", False):          # multi-panel cards place their axes themselves
            continue
        for _ in range(iterations):
            pos = ax.get_position()
            tb = ax.get_tightbbox(renderer)
            if tb is None:
                break
            # how far the decorations stick out past the axes box, as figure fractions
            pad_l = pos.x0 - tb.x0 / W
            pad_r = tb.x1 / W - pos.x1
            pad_b = pos.y0 - tb.y0 / H
            pad_t = tb.y1 / H - pos.y1
            new = [left + pad_l, bottom + pad_b,
                   (right - pad_r) - (left + pad_l), (top - pad_t) - (bottom + pad_b)]
            if new[2] <= 0.1 or new[3] <= 0.1:
                break
            if np.allclose(new, [pos.x0, pos.y0, pos.width, pos.height], atol=5e-4):
                break
            ax.set_position(new)


def save_card(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fit_card(fig)
    fig.savefig(path, dpi=DPI, facecolor=BG)   # no bbox_inches: card must stay exactly 1080x1080
    plt.close(fig)
    return path


def parse_names(text: str | None) -> set[str] | None:
    """'Haze, Victor' -> {'Haze', 'Victor'}; empty -> None (no highlight)."""
    names = {n.strip() for n in (text or "").split(",") if n.strip()}
    return names or None


def save_cards(cards: dict, out_dir: Path, progress: float = 1.0, highlight: set[str] | None = None) -> list[Path]:
    """Save every card from a chart script's cards() dict as <out_dir>/<name>.png."""
    return [save_card(draw(progress, highlight), out_dir / f"{name}.png") for name, draw in cards.items()]


def card_layout(fig: Figure) -> list:
    """Fit the card, then return every axes' position. Animation frames reuse the finished
    card's layout: fitting each frame separately would make the plot jump around as labels
    and icons appear (fit_card measures whatever is drawn at that moment)."""
    fit_card(fig)
    return [ax.get_position().frozen() for ax in fig.axes]


def fig_to_array(fig: Figure, layout: list | None = None) -> np.ndarray:
    """Render a figure to an HxWx3 uint8 array (animation frames / ffmpeg piping).
    layout: positions from card_layout() of the finished card; None = fit this figure."""
    if layout is not None and len(layout) == len(fig.axes):
        for ax, pos in zip(fig.axes, layout):
            ax.set_position(pos)
    else:
        fit_card(fig)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def fig_to_image(fig: Figure, layout: list | None = None) -> Image.Image:
    """Render a figure to a PIL image."""
    return Image.fromarray(fig_to_array(fig, layout))


# =============================================================================
# HERO ICONS
# =============================================================================
def icon_state(wr: float | None) -> str:
    """'critical' / 'base' / 'gloat' for a win rate in percent (None -> 'base')."""
    if wr is None or (isinstance(wr, float) and np.isnan(wr)):
        return "base"
    if wr < ICON_WR_LOW:
        return "critical"
    if wr > ICON_WR_HIGH:
        return "gloat"
    return "base"


def icon_path(name: str, wr: float | None = None) -> Path | None:
    """Icon file for this hero and win rate, falling back to base_icons/ if the variant is missing."""
    for folder in (ICON_FOLDERS[icon_state(wr)], ICON_FOLDERS["base"]):
        p = ASSETS_BASE / folder / f"{name}.png"
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1024)
def hero_icon(name: str, size_px: int, ring_hex: str | None = None,
              alpha: float = 1.0, ring_px: int = 4, state: str = "base") -> np.ndarray | None:
    """Circular-cropped hero icon with optional colored ring. None if icon missing.
    `state` picks the variant folder (see icon_state); pass wr to icon_any() instead."""
    path = icon_path(name, {"critical": 0.0, "base": 50.0, "gloat": 100.0}[state])
    if path is None:
        return None
    src = Image.open(path).convert("RGBA")
    side = min(src.size)
    left, top = (src.width - side) // 2, (src.height - side) // 2
    src = src.crop((left, top, left + side, top + side))

    ss = 4  # supersample for smooth edges
    big = size_px * ss
    inner = big - 2 * ring_px * ss if ring_hex else big
    icon = src.resize((inner, inner), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    if ring_hex:
        ImageDraw.Draw(canvas).ellipse([0, 0, big - 1, big - 1], fill=rgba(ring_hex))
    disc = Image.new("RGBA", (inner, inner), rgba(BG))
    disc.alpha_composite(icon)
    mask = Image.new("L", (inner, inner), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, inner - 1, inner - 1], fill=255)
    offset = (big - inner) // 2
    canvas.paste(disc, (offset, offset), mask)

    out = canvas.resize((size_px, size_px), Image.Resampling.LANCZOS)
    arr = np.asarray(out).astype(np.float32) / 255.0
    arr[..., 3] *= alpha
    return arr


@lru_cache(maxsize=512)
def placeholder_icon(name: str, size_px: int, ring_hex: str | None = None,
                     alpha: float = 1.0, ring_px: int = 4) -> np.ndarray:
    """Disc with the hero's initials, used when the real icon file is missing."""
    from matplotlib import font_manager
    from PIL import ImageFont

    ss = 4
    big = size_px * ss
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ring = ring_hex or NEUTRAL
    d.ellipse([0, 0, big - 1, big - 1], fill=rgba(ring))
    pad = ring_px * ss
    d.ellipse([pad, pad, big - 1 - pad, big - 1 - pad], fill=rgba(PANEL))
    initials = "".join(w[0] for w in name.replace("&", " ").split()[:2]).upper() or "?"
    font_path = font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY, weight="bold"))
    font = ImageFont.truetype(font_path, int(big * 0.36))
    d.text((big / 2, big / 2), initials, font=font, fill=rgba(ring), anchor="mm")
    out = img.resize((size_px, size_px), Image.Resampling.LANCZOS)
    arr = np.asarray(out).astype(np.float32) / 255.0
    arr[..., 3] *= alpha
    return arr


def icon_any(name: str, size_px: int, ring_hex: str | None = None, alpha: float = 1.0,
             wr: float | None = None) -> np.ndarray:
    """Real icon if available, otherwise the initials placeholder.
    wr = the win rate this icon is shown next to (solo, duo or team), in percent;
    it selects critical_icons / base_icons / gloat_icons."""
    size_px = max(int(round(size_px)), 8)
    arr = hero_icon(name, size_px, ring_hex, round(alpha, 2), state=icon_state(wr))
    return arr if arr is not None else placeholder_icon(name, size_px, ring_hex, round(alpha, 2))


# =============================================================================
# TEXT MEASURING / DYNAMIC SIZING
# =============================================================================
@lru_cache(maxsize=4096)
@lru_cache(maxsize=100_000)
def _text_w_per_px(s: str, bold: bool) -> float:
    """Rendered width of s, in pixels, per 1px of font size."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    prop = FontProperties(family=FONT_FAMILY, weight="bold" if bold else "normal")
    return TextPath((0, 0), s, size=100, prop=prop).get_extents().width / 100.0 * 1.05  # + side bearings


def text_px(s: str, size_px: float, bold: bool = True) -> float:
    """Width in card pixels of s drawn at size_px."""
    return _text_w_per_px(str(s), bold) * size_px


def fit_font_px(strings, max_w_px: float, max_h_px: float | None = None, max_px: float = 30,
                min_px: float = 11, bold: bool = True) -> float:
    """Largest font size (px) at which every string fits max_w_px (and max_h_px tall)."""
    widest = max((_text_w_per_px(str(s), bold) for s in strings), default=0.0)
    size = max_px
    if widest > 0:
        size = min(size, max_w_px / widest)
    if max_h_px:
        size = min(size, max_h_px)
    return max(min_px, size)


def content_px(fig: Figure) -> tuple[float, float]:
    """Width/height in pixels that fit_card() will give the content (call after the header)."""
    bottom, top = content_bounds(fig)
    return (1 - 2 * MARGIN_X) * CARD_PX, (top - bottom) * CARD_PX


def table_rows(fig: Figure, ax, n_rows: float, header_rows: float = 1.0, footer_rows: float = 0.0) -> float:
    """Set a table's y-range so n_rows (+ header/footer bands) fill the card; returns the
    resulting row height in px (use it to size icons and fonts)."""
    _, h = content_px(fig)
    ax.set_ylim(n_rows - 0.5 + footer_rows, -0.5 - header_rows)
    return h / (n_rows + header_rows + footer_rows)


def layout_columns(widths_px: dict, total_px: float, gap_px: float = 22) -> dict:
    """Place table columns left to right. widths_px: name -> width in px, or None for a
    flexible column that takes the leftover space (shared equally if several).
    Returns name -> (x0, x1) in 0..1 table units."""
    fixed = sum(w for w in widths_px.values() if w is not None)
    n_flex = sum(1 for w in widths_px.values() if w is None)
    leftover = total_px - fixed - gap_px * (len(widths_px) - 1)
    flex = max(leftover / n_flex, 40.0) if n_flex else 0.0
    out, x = {}, 0.0
    for name, w in widths_px.items():
        w = flex if w is None else w
        out[name] = (x / total_px, (x + w) / total_px)
        x += w + gap_px
    return out


def add_icon(ax, arr: np.ndarray, xy: tuple[float, float], zorder: float = 5):
    """Place an icon array at data coords on ax (pixel size = array size)."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    ab = AnnotationBbox(OffsetImage(arr, zoom=1, dpi_cor=False), xy, frameon=False,
                        zorder=zorder, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)
    return ab


def new_table_card() -> tuple[Figure, plt.Axes]:
    """Card whose plot area is a blank 0..1 canvas (x) for tables; set ylim per row count."""
    fig, ax = new_card()
    ax.axis("off")
    ax.set_xlim(0, 1)
    return fig, ax


ARCHETYPE_ORDER = list(ARCHETYPE_COLORS)   # brawler, marksman, assassin, mystic


def archetype_rank(hero_type: str) -> int:
    t = str(hero_type).strip().lower()
    return ARCHETYPE_ORDER.index(t) if t in ARCHETYPE_ORDER else len(ARCHETYPE_ORDER)


def wr_color(wr: float, lo: float | None = None, hi: float | None = None) -> str:
    """Green above 50, red below; neutral when the CI straddles 50%."""
    if lo is not None and hi is not None and lo <= 50 <= hi:
        return TEXT_MUTED
    return GREEN if wr >= 50 else RED


# =============================================================================
# ANIMATION HELPERS
# =============================================================================
def clamp01(t: float) -> float:
    return max(0.0, min(1.0, t))


def phase(progress: float, start: float, end: float) -> float:
    """Local 0..1 progress of a sub-animation running between start and end."""
    if end <= start:
        return 1.0 if progress >= end else 0.0
    return clamp01((progress - start) / (end - start))


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3