import sys
from datetime import date
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as patheffects

# --- PATH RESOLUTION & CONFIGURATION ---
RANKED_START_DATE = date(2026, 7, 30)


def get_current_week_number() -> int:
    """Calculates the target week number (0-indexed offset for completed week)."""
    days_elapsed = (date.today() - RANKED_START_DATE).days
    return days_elapsed // 7


def resolve_target_dir() -> Path:
    """Gets target dir from CLI args or defaults to current week's analysis folder."""
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            return target

    week_num = get_current_week_number()
    return Path.home() / "Projects" / "DL_ranked_analysis" / f"week{week_num}_analysis"


DATA_DIR = resolve_target_dir()
OUTPUT_DIR = DATA_DIR / "graphs"
MATCH_CSV_FILE_PATH = DATA_DIR / "avg_duration.csv"
HERO_DURATION_WINRATES_PARQUET_FILE_PATH = DATA_DIR / "hero_duration_scaling.parquet"

# Output filenames
OUTPUT_FOCUSED_WR = OUTPUT_DIR / "hero_focused_winrate_duration_heatmap.png"
OUTPUT_FOCUSED_WINS = OUTPUT_DIR / "hero_focused_total_wins_duration_heatmap.png"
OUTPUT_RAW_WR = OUTPUT_DIR / "hero_raw_winrate_duration_heatmap.png"
OUTPUT_SCALING = OUTPUT_DIR / "hero_scaling_weighted_winrate_duration_heatmap.png"

# Target brackets for focused mid-to-late game heatmaps
TARGET_FOCUSED_BRACKETS = ["30-35m", "35-40m", "40-45m"]

# --- THEME COLORS ---
BG_COLOR = (15, 23, 23, 255)
TEXT_COLOR = (244, 240, 226, 255)
TITLE_COLOR = (255, 208, 115, 255)
FOOTER_COLOR = (140, 145, 155, 255)
GRID_COLOR = (25, 35, 35, 255)
BASELINE_COLOR = (100, 115, 120, 255)

COLOR_GAIN = (72, 205, 122, 255)
COLOR_NEUTRAL_ABS = (150, 150, 150, 255)
COLOR_NEUTRAL_DARK = (70, 85, 90, 255)
COLOR_LOSS = (214, 80, 74, 255)
COLOR_TEAL = (45, 125, 125, 180)
HEX_GRID_COLOR = "#192323"

# FONT SIZES
HERO_NAME_SIZE = 16
AXES_PADDING = 8
AXES_SIZES = 18
TITLE_SIZES = 24
TITLE_PADDING = 10


text_outline = [patheffects.withStroke(linewidth=2.5, foreground=HEX_GRID_COLOR)]

text_shadow = [
    patheffects.SimpleLineShadow(offset=(1, -1), shadowcolor="black", alpha=0.8),
    patheffects.Normal(),
]

def rgba_to_mpl(color_tuple: tuple) -> tuple:
    """Converts 0-255 RGBA tuple to 0-1 Matplotlib RGBA tuple."""
    return tuple(c / 255.0 for c in color_tuple)


mpl_bg = rgba_to_mpl(BG_COLOR)
mpl_neutral_abs = rgba_to_mpl(COLOR_NEUTRAL_ABS)
mpl_neutral_dark = rgba_to_mpl(COLOR_NEUTRAL_DARK)
mpl_text = rgba_to_mpl(TEXT_COLOR)
mpl_title = rgba_to_mpl(TITLE_COLOR)
mpl_footer = rgba_to_mpl(FOOTER_COLOR)
mpl_grid = rgba_to_mpl(GRID_COLOR)
mpl_gain = rgba_to_mpl(COLOR_GAIN)
mpl_loss = rgba_to_mpl(COLOR_LOSS)
mpl_teal = rgba_to_mpl(COLOR_TEAL)
mpl_base = rgba_to_mpl(BASELINE_COLOR)

# --- CUSTOM COLORMAPS ---
cmap_focused_wr = mcolors.LinearSegmentedColormap.from_list(
    "deadlock_focused_wr", [mpl_loss, mpl_neutral_abs, mpl_gain]
)

cmap_abs_wr = mcolors.LinearSegmentedColormap.from_list(
    "deadlock_abs_wr", list(zip([0.0, 0.5, 1.0], [mpl_loss, mpl_neutral_abs, mpl_gain]))
)

cmap_scaling = mcolors.LinearSegmentedColormap.from_list(
    "deadlock_scaling", list(zip([0.0, 0.5, 1.0], [mpl_loss, mpl_neutral_abs, mpl_gain]))
)


def apply_shared_mpl_style() -> None:
    """Sets standard canvas styling parameters across all generated plots."""
    plt.rcParams.update({
        "figure.facecolor": mpl_bg,
        "axes.facecolor": mpl_bg,
        "text.color": mpl_text,
        "axes.labelcolor": mpl_title,
        "xtick.color": mpl_text,
        "ytick.color": mpl_text,
        "axes.edgecolor": mpl_grid,
        "font.family": "sans-serif",
    })


def get_bracket_start(b_str: str) -> int:
    """Extracts integer start minute from duration bracket string for chronological sorting."""
    clean = str(b_str).replace("m+", "").split("-")[0]
    return int(clean) if clean.isdigit() else 999


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validates input file existence and returns raw datasets with calculated fields."""
    if not MATCH_CSV_FILE_PATH.exists() or not HERO_DURATION_WINRATES_PARQUET_FILE_PATH.exists():
        print(f"Error: Missing input files in {DATA_DIR}")
        sys.exit(1)

    df_dist = pd.read_csv(MATCH_CSV_FILE_PATH)
    df_hero = pd.read_parquet(HERO_DURATION_WINRATES_PARQUET_FILE_PATH)

    if float(np.max(df_hero["winrate"])) <= 1.0:
        df_hero["winrate"] = df_hero["winrate"] * 100.0

    df_hero["total_wins"] = (
        df_hero["total_games"] * (df_hero["winrate"] / 100.0)
    ).round().astype(int)

    return df_dist, df_hero


def render_top_histogram(
    ax: plt.Axes, df_dist: pd.DataFrame, global_total_games: int, title_text: str
) -> None:
    """Renders top panel histogram using global_total_games for absolute global %."""
    bars = ax.bar(
        df_dist["duration_bracket"],
        df_dist["total_games"],
        color=mpl_teal,
        edgecolor=mpl_base,
        width=1.0,
    )

    for bar in bars:
        height = bar.get_height()
        pct = (height / global_total_games) * 100 if global_total_games > 0 else 0
        count_str = f"{height/1000:.1f}k" if height >= 1000 else str(int(height))


        ax.annotate(
            f"{count_str}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, -8),  
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize= 14,
            fontweight="bold",
            color=mpl_text,
        )

    ax.set_title(title_text, color=mpl_title, fontsize=TITLE_SIZES, fontweight="bold", pad=TITLE_PADDING)
    ax.set_ylabel("Games", fontsize=AXES_SIZES, color=mpl_text)
    ax.grid(axis="y", color=mpl_grid, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)
    ax.tick_params(axis="x", labelbottom=False, bottom=False, labelsize= HERO_NAME_SIZE)


def add_footer(fig: plt.Figure) -> None:
    fig.add_artist(
        plt.Line2D(
            [0.04, 0.96], [0.02, 0.02],
            transform=fig.transFigure,
            color=mpl_grid,
            linewidth=1,
        )
    )

    fig.text(
        0.04, 0.012, "Data Source: Deadlock Community API",
        fontsize=9,
        color=mpl_footer,
        ha="left",
        va="center",
        fontweight="bold",
        path_effects=text_outline,
    )

    fig.text(
        0.96, 0.012, "Analysis by @eReys13",
        fontsize=9,
        color=mpl_footer,
        ha="right",
        va="center",
        fontweight="bold",
        path_effects=text_outline,
    )


# --- GENERATOR 1: FOCUSED WIN RATE HEATMAP ---
def generate_focused_winrate_heatmap(
    df_dist_raw: pd.DataFrame, df_hero_raw: pd.DataFrame, global_total_games: int
) -> None:
    apply_shared_mpl_style()

    heatmap_data = df_hero_raw.pivot(
        index="hero_name", columns="duration_bracket", values="winrate"
    )
    available_buckets = [b for b in TARGET_FOCUSED_BRACKETS if b in heatmap_data.columns]
    heatmap_data = heatmap_data[available_buckets]

    df_dist = df_dist_raw[df_dist_raw["duration_bracket"].isin(available_buckets)].copy()
    df_dist["duration_bracket"] = pd.Categorical(
        df_dist["duration_bracket"], categories=available_buckets, ordered=True
    )
    df_dist = df_dist.sort_values("duration_bracket")

    heatmap_data["mean_wr"] = heatmap_data.mean(axis=1)
    sorted_ilocs = np.argsort(np.asarray(heatmap_data["mean_wr"]))[::-1]
    heatmap_data = heatmap_data.iloc[sorted_ilocs].drop(columns=["mean_wr"])

    norm = mcolors.TwoSlopeNorm(
        vmin=46, vcenter=50.0, vmax= 54
    )

    fig = plt.figure(figsize=(9, 16))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 6], width_ratios=[15, 1], hspace=0.08, wspace=0.04, bottom=0.055,
        top=0.97,
        left=0.08,
        right=0.96
    )

    ax_hist = fig.add_subplot(gs[0, 0])
    render_top_histogram(ax_hist, df_dist, global_total_games, "CORE MATCH DURATION DISTRIBUTION")

    ax_heat = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    sns.heatmap(
        heatmap_data,
        cmap=cmap_focused_wr,
        norm=norm,
        ax=ax_heat,
        cbar_ax=cax,
        cbar_kws={"label": "Win Rate (%)"},
        linewidths=1,
        linecolor=HEX_GRID_COLOR,
        annot=True,
        fmt=".1f",
        annot_kws={"size": HERO_NAME_SIZE - 4, "color": mpl_text},
    )

    ax_heat.set_title(
        "HERO WIN RATE (30m - 45m MATCHES)", color=mpl_title, fontsize=TITLE_SIZES, fontweight="bold", pad=TITLE_PADDING
    )
    ax_heat.set_xlabel("Match Duration Bracket", fontsize=AXES_SIZES, color=mpl_title, labelpad=AXES_PADDING - 6)
    ax_heat.set_ylabel(
        "Hero (Sorted by Mid-to-Late Game WR)", fontsize=AXES_SIZES, color=mpl_title, labelpad=AXES_PADDING
    )

    plt.setp(ax_heat.get_xticklabels(), fontsize= HERO_NAME_SIZE, rotation=0, ha="center")
    plt.setp(ax_heat.get_yticklabels(), fontsize=HERO_NAME_SIZE)

    cax.yaxis.label.set_color(mpl_text)
    cax.yaxis.label.set_fontsize(14)
    cax.tick_params(colors=mpl_text)    


    for text in ax_heat.texts:
        text.set_path_effects(text_outline)

    for label in ax_heat.get_yticklabels() + ax_heat.get_xticklabels():
        label.set_path_effects(text_outline)
    
    add_footer(fig)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FOCUSED_WR, dpi=300, bbox_inches="tight", facecolor=mpl_bg)
    plt.close()
    print(f"✅ Generated Focused Win Rate Heatmap: {OUTPUT_FOCUSED_WR}")


# --- GENERATOR 2: FOCUSED TOTAL WINS (COLORED BY WIN RATE) ---
def generate_focused_total_wins_heatmap(
    df_dist_raw: pd.DataFrame, df_hero_raw: pd.DataFrame, global_total_games: int
) -> None:
    apply_shared_mpl_style()

    heatmap_wr = df_hero_raw.pivot(
        index="hero_name", columns="duration_bracket", values="winrate"
    )
    heatmap_wins = df_hero_raw.pivot(
        index="hero_name", columns="duration_bracket", values="total_wins"
    )

    available_buckets = [b for b in TARGET_FOCUSED_BRACKETS if b in heatmap_wr.columns]
    heatmap_wr = heatmap_wr[available_buckets]
    heatmap_wins = heatmap_wins[available_buckets]

    df_dist = df_dist_raw[df_dist_raw["duration_bracket"].isin(available_buckets)].copy()
    df_dist["duration_bracket"] = pd.Categorical(
        df_dist["duration_bracket"], categories=available_buckets, ordered=True
    )
    df_dist = df_dist.sort_values("duration_bracket")

    heatmap_wins["sum_wins"] = heatmap_wins.sum(axis=1)
    sorted_ilocs = np.argsort(np.asarray(heatmap_wins["sum_wins"]))[::-1]

    heatmap_wins = heatmap_wins.iloc[sorted_ilocs].drop(columns=["sum_wins"])
    heatmap_wr = heatmap_wr.iloc[sorted_ilocs]

    annot_text = heatmap_wins.map(
        lambda v: f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}"
    )

    norm = mcolors.TwoSlopeNorm(
        vmin=46, vcenter=50.0, vmax=54
    )

    fig = plt.figure(figsize=(9, 16))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 6], width_ratios=[15, 1], hspace=0.08, wspace=0.04, bottom=0.055,
        top=0.97,
        left=0.08,
        right=0.96
    )

    ax_hist = fig.add_subplot(gs[0, 0])
    render_top_histogram(ax_hist, df_dist, global_total_games, "CORE MATCH DURATION DISTRIBUTION")

    ax_heat = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    sns.heatmap(
        heatmap_wr,
        cmap=cmap_focused_wr,
        norm=norm,
        ax=ax_heat,
        cbar_ax=cax,
        cbar_kws={"label": "Win Rate (%)"},
        linewidths=1,
        linecolor=HEX_GRID_COLOR,
        annot=annot_text,
        fmt="",
        annot_kws={"size": HERO_NAME_SIZE - 4, "color": mpl_text},
    )

    ax_heat.set_title(
        "TOTAL MATCHES WON (30m - 45m MATCHES)", color=mpl_title, fontsize=TITLE_SIZES, fontweight="bold", pad=TITLE_PADDING
    )
    ax_heat.set_xlabel("Match Duration Bracket", fontsize=AXES_SIZES, color=mpl_title, labelpad=AXES_PADDING  - 6)
    ax_heat.set_ylabel(
        "Hero (Sorted by Total Mid-to-Late Game Wins)", fontsize=AXES_SIZES, color=mpl_title, labelpad=AXES_PADDING
    )

    plt.setp(ax_heat.get_xticklabels(), fontsize= HERO_NAME_SIZE, rotation=0, ha="center")
    plt.setp(ax_heat.get_yticklabels(), fontsize=HERO_NAME_SIZE)

    cax.yaxis.label.set_color(mpl_text)
    cax.yaxis.label.set_fontsize(14)
    cax.tick_params(colors=mpl_text)    


    for text in ax_heat.texts:
        text.set_path_effects(text_outline)

    for label in ax_heat.get_yticklabels() + ax_heat.get_xticklabels():
        label.set_path_effects(text_outline)


    add_footer(fig)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FOCUSED_WINS, dpi=300, bbox_inches="tight", facecolor=mpl_bg)
    plt.close()
    print(f"✅ Generated Focused Total Wins Heatmap (colored by WR): {OUTPUT_FOCUSED_WINS}")


# --- GENERATOR 3: ABSOLUTE WIN RATE HEATMAP ---
def generate_absolute_winrate_heatmap(
    df_dist_raw: pd.DataFrame, df_hero_raw: pd.DataFrame, global_total_games: int
) -> None:
    apply_shared_mpl_style()

    heatmap_data = df_hero_raw.pivot(
        index="hero_name", columns="duration_bracket", values="winrate"
    )

    bucket_order = sorted(list(heatmap_data.columns), key=get_bracket_start)
    heatmap_data = heatmap_data[bucket_order]

    df_dist = df_dist_raw.copy()
    df_dist["sort_key"] = df_dist["duration_bracket"].apply(get_bracket_start)
    df_dist = df_dist.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])

    heatmap_data["mean_wr"] = heatmap_data.mean(axis=1)
    sorted_ilocs = np.argsort(np.asarray(heatmap_data["mean_wr"]))[::-1]
    heatmap_data = heatmap_data.iloc[sorted_ilocs].drop(columns=["mean_wr"])

    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1, 5.5],
    width_ratios=[30, 1],
    hspace=0.08,
    wspace=0.03,
    bottom=0.055,
    top=0.97,
    left=0.08,
    right=0.96
)

    ax_hist = fig.add_subplot(gs[0, 0])
    render_top_histogram(ax_hist, df_dist, global_total_games, "OVERALL MATCH DURATION DISTRIBUTION")

    ax_heat = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    sns.heatmap(
        heatmap_data,
        cmap=cmap_abs_wr,
        vmin=44.0,
        vmax=56.0,
        center=50.0,
        ax=ax_heat,
        cbar_ax=cax,
        cbar_kws={"label": "Win Rate (%)"},
        linewidths=1,
        linecolor=HEX_GRID_COLOR,
    )

    ax_heat.set_title(
        "HERO WIN RATE BY MATCH DURATION", color=mpl_title, fontsize=TITLE_SIZES, fontweight="bold", pad=TITLE_PADDING
    )
    ax_heat.set_xlabel("Match Duration", fontsize=AXES_SIZES, color=mpl_title, labelpad=AXES_PADDING)
    ax_heat.set_ylabel("Hero (Sorted by Overall Win Rate)", fontsize=AXES_SIZES, color=mpl_title, labelpad=AXES_PADDING)

    plt.setp(ax_heat.get_xticklabels(), fontsize= HERO_NAME_SIZE, rotation=0, ha="center")
    plt.setp(ax_heat.get_yticklabels(), fontsize=HERO_NAME_SIZE)
   
    cax.yaxis.label.set_color(mpl_text)
    cax.yaxis.label.set_fontsize(14)
    cax.tick_params(colors=mpl_text)    


    for text in ax_heat.texts:
        text.set_path_effects(text_outline)

    for label in ax_heat.get_yticklabels() + ax_heat.get_xticklabels():
        label.set_path_effects(text_outline)

    add_footer(fig)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_RAW_WR, dpi=300, bbox_inches="tight", facecolor=mpl_bg)
    plt.close()
    print(f"✅ Generated Absolute Raw Win Rate Heatmap: {OUTPUT_RAW_WR}")


# --- GENERATOR 4: RELATIVE SCALING HEATMAP ---
def generate_scaling_heatmap(
    df_dist_raw: pd.DataFrame, df_hero_raw: pd.DataFrame, global_total_games: int
) -> None:
    apply_shared_mpl_style()

    heatmap_data = df_hero_raw.pivot(
        index="hero_name", columns="duration_bracket", values="winrate"
    )

    bucket_order = sorted(list(heatmap_data.columns), key=get_bracket_start)
    heatmap_data = heatmap_data[bucket_order]

    df_dist = df_dist_raw.copy()
    df_dist["sort_key"] = df_dist["duration_bracket"].apply(get_bracket_start)
    df_dist = df_dist.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])

    bracket_weights = df_dist.set_index("duration_bracket")["total_games"].astype(float)
    bracket_weights = bracket_weights.reindex(heatmap_data.columns)

    valid_mask = heatmap_data.notna()
    hero_actual_weights = valid_mask.mul(bracket_weights, axis=1)
    weighted_sums = heatmap_data.mul(bracket_weights, axis=1).sum(axis=1)
    weighted_mean_series = weighted_sums / hero_actual_weights.sum(axis=1)

    scaling_data = heatmap_data.sub(weighted_mean_series, axis=0)
    sorted_ilocs = np.argsort(np.asarray(weighted_mean_series, dtype=float))[::-1]
    scaling_data = scaling_data.iloc[sorted_ilocs]

    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 5.5], width_ratios=[30, 1], hspace=0.10, wspace=0.03,bottom=0.055,
        top=0.97,
        left=0.08,
        right=0.96
    )

    ax_hist = fig.add_subplot(gs[0, 0])
    render_top_histogram(ax_hist, df_dist, global_total_games, "OVERALL MATCH DURATION DISTRIBUTION")

    ax_heat = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    sns.heatmap(
        scaling_data,
        cmap=cmap_scaling,
        vmin=-5.0,
        vmax=5.0,
        center=0.0,
        ax=ax_heat,
        cbar_ax=cax,
        cbar_kws={"label": "Scaling Delta (% Shift from Mean WR)"},
        linewidths=1,
        linecolor=HEX_GRID_COLOR,
    )

    ax_heat.set_title(
        "HERO WIN RATE SCALING BY MATCH DURATION",
        color=mpl_title,
        fontsize=TITLE_SIZES,
        fontweight="bold",
        pad=TITLE_PADDING,
    )
    ax_heat.set_xlabel("Match Duration", fontsize=AXES_SIZES, color=mpl_title, labelpad=AXES_PADDING)
    ax_heat.set_ylabel(
        "Hero (Sorted by Overall Win Rate)", fontsize=AXES_SIZES, color=mpl_title, labelpad=AXES_PADDING
    )

    plt.setp(ax_heat.get_xticklabels(), fontsize= HERO_NAME_SIZE, rotation=0, ha="center")
    plt.setp(ax_heat.get_yticklabels(), fontsize=HERO_NAME_SIZE)

    cax.yaxis.label.set_color(mpl_text)
    cax.yaxis.label.set_fontsize(14)
    cax.tick_params(colors=mpl_text)    

    for text in ax_heat.texts:
        text.set_path_effects(text_outline)

    for label in ax_heat.get_yticklabels() + ax_heat.get_xticklabels():
        label.set_path_effects(text_outline)

    add_footer(fig)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_SCALING, dpi=300, bbox_inches="tight", facecolor=mpl_bg)
    plt.close()
    print(f"✅ Generated Hero Scaling Heatmap: {OUTPUT_SCALING}")


# --- MAIN PIPELINE ---
def main() -> None:
    df_dist, df_hero = load_datasets()

    global_total_games = int(df_dist["total_games"].sum())

    print("🚀 Starting graphics generation pipeline...")
    generate_focused_winrate_heatmap(df_dist, df_hero, global_total_games)
    generate_focused_total_wins_heatmap(df_dist, df_hero, global_total_games)
    generate_absolute_winrate_heatmap(df_dist, df_hero, global_total_games)
    generate_scaling_heatmap(df_dist, df_hero, global_total_games)
    print("✨ All 4 charts successfully generated in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()