import re
import sys
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# --- CONFIGURATION & PATH RESOLUTION ---
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
CSV_FILE_PATH = DATA_DIR / "unique_comp_freq_match.csv"

# Output image paths
EXCLUSIVE_OUTPUT = OUTPUT_DIR / "unique_comp_freq_match_exclusive.png"
CUMULATIVE_OUTPUT = OUTPUT_DIR / "unique_comp_freq_match_cumulative.png"


def human_readable_ticks(x: float, pos: int) -> str:
    """Helper function to turn log ticks into clean shorthand (e.g. 120k)."""
    if x >= 1000:
        return f"{int(x/1000)}k"
    return f"{int(x)}"


def parse_threshold_label(bucket_str: str) -> str:
    """
    Converts frequency bucket names into minimum threshold labels.
    Examples: 'single-occurrence' -> '1+', '2 - 4 occurrences' -> '2+', '40+ occurrences' -> '40+'
    """
    bucket_str = str(bucket_str).strip()
    if bucket_str.lower() in ("single-occurrence", "single occurrence", "1"):
        return "1+"

    match = re.search(r"(\d+)", bucket_str)
    if match:
        return f"{match.group(1)}+"

    return bucket_str


def plot_decay_chart(
    x_labels: list,
    unique_comps: np.ndarray,
    total_matches: np.ndarray,
    title: str,
    output_path: Path,
):
    """Generates a dark-themed dual-axis decay chart with dynamic baselines."""
    num_buckets = len(x_labels)
    x_indices = np.arange(num_buckets)

    # Calculate Theoretical Baselines Dynamically
    start_comp_val = unique_comps[0]
    end_comp_val = max(unique_comps[-1], 1.0)  # Avoid log(0) for exponential decay

    linear_comps = np.linspace(start_comp_val, end_comp_val, num_buckets)
    exponential_comps = np.logspace(
        np.log10(start_comp_val), np.log10(end_comp_val), num_buckets
    )

    # Plot Setup
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    formatter = FuncFormatter(human_readable_ticks)

    # LEFT Y-Axis: Unique Team Comps & Baselines
    color_comps = "#00f5d4"  # Cyan
    ax1.set_xlabel("Minimum Ocurrences", fontsize=12, labelpad=12)
    ax1.set_ylabel("Number of Unique Team Comps", color=color_comps, fontsize=12)

    line_linear = ax1.plot(
        x_indices,
        linear_comps,
        color="#3ac348",  # Brawler Green
        linestyle=(0, (5, 3)),
        linewidth=1.75,
        label="Theoretical Linear Decrease",
        zorder=2,
    )

    line_exponential = ax1.plot(
        x_indices,
        exponential_comps,
        color="#ffaa00",  # Amber / Gold
        linestyle="-.",
        linewidth=1.75,
        label="Theoretical Exponential Decrease",
        zorder=2,
    )

    line1 = ax1.plot(
        x_indices,
        unique_comps,
        color=color_comps,
        linewidth=2.5,
        marker="o",
        markersize=7,
        label="Actual Unique Team Comps",
        zorder=4,
    )

    ax1.tick_params(axis="y", labelcolor=color_comps, labelsize=11)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(formatter)

    # Value Labels below cyan markers
    for x, y in zip(x_indices, unique_comps):
        ha = "right" if x == num_buckets - 1 else "center"
        ax1.text(
            float(x),
            float(y * 0.65),
            f"{int(y):,}",
            color=color_comps,
            fontsize=9,
            fontweight="bold",
            ha=ha,
            va="top",
        )

    # RIGHT Y-Axis: Total Matches Played
    ax2 = ax1.twinx()
    color_matches = "#ff006e"  # Hot Pink
    ax2.set_ylabel("Total Matches Played", color=color_matches, fontsize=12)

    line2 = ax2.plot(
        x_indices,
        total_matches,
        color=color_matches,
        linewidth=2.5,
        marker="s",
        markersize=7,
        label="Total Matches Played",
        zorder=4,
    )

    ax2.tick_params(axis="y", labelcolor=color_matches, labelsize=11)
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(formatter)

    # Value Labels above pink markers
    for x, y in zip(x_indices, total_matches):
        ha = "left" if x == num_buckets - 1 else "center"
        ax2.text(
            float(x),
            float(y * 1.45),
            f"{int(y):,}",
            color=color_matches,
            fontsize=9,
            fontweight="bold",
            ha=ha,
            va="bottom",
        )

    # Formatting & Dynamic Y-Limits
    plt.xticks(
        x_indices,
        x_labels,
        fontsize=10,
        rotation=15 if num_buckets > 8 or len(str(x_labels[0])) > 8 else 0,
    )
    ax1.set_title(title, fontsize=15, pad=25, fontweight="bold")
    ax1.grid(True, which="major", linestyle=":", alpha=0.15)

    min_comp_val = max(min(unique_comps), 1.0)
    max_comp_val = max(unique_comps)
    ax1.set_ylim(bottom=min_comp_val * 0.2, top=max_comp_val * 3.0)

    min_match_val = max(min(total_matches), 1.0)
    max_match_val = max(total_matches)
    ax2.set_ylim(bottom=min_match_val * 0.2, top=max_match_val * 3.0)

    # Combined Legend
    lines = line_linear + line_exponential + line1 + line2
    labels = [str(l.get_label()) for l in lines]

    ax1.legend(
        lines,
        labels,
        loc="upper right",
        frameon=True,
        facecolor="#222222",
        edgecolor="none",
        fontsize=11,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated: {output_path.name}")


def main():
    if not CSV_FILE_PATH.exists():
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE_PATH)
    df.columns = df.columns.str.strip().str.lower()

    # --- 1. EXCLUSIVE BUCKETS GRAPH ---
    exclusive_labels = df["frequency_bucket"].astype(str).tolist()
    exclusive_comps = df["unique_team_comps"].astype(float).to_numpy()
    exclusive_matches = df["total_games_played"].astype(float).to_numpy()

    plot_decay_chart(
        x_labels=exclusive_labels,
        unique_comps=exclusive_comps,
        total_matches=exclusive_matches,
        title="EXCLUSIVE BUCKETS: UNIQUE COMPS & MATCH VOLUME VS DECAY BASELINES",
        output_path=EXCLUSIVE_OUTPUT,
    )

    # --- 2. CUMULATIVE / INCLUSIVE THRESHOLDS GRAPH ---
    cumulative_labels = [
        parse_threshold_label(b) for b in df["frequency_bucket"]
    ]
    cumulative_comps = df["cumulative_comps_n_plus"].astype(float).to_numpy()
    cumulative_matches = df["cumulative_games_n_plus"].astype(float).to_numpy()

    plot_decay_chart(
        x_labels=cumulative_labels,
        unique_comps=cumulative_comps,
        total_matches=cumulative_matches,
        title="CUMULATIVE THRESHOLDS (N+): UNIQUE COMPS & MATCH VOLUME VS DECAY BASELINES",
        output_path=CUMULATIVE_OUTPUT,
    )


if __name__ == "__main__":
    main()