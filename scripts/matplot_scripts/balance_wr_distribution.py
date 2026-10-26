import sys
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# --- CONFIGURATION & PATH RESOLUTION ---
RANKED_START_DATE = date(2026, 7, 30)


def get_current_week_number() -> int:
    days_elapsed = (date.today() - RANKED_START_DATE).days
    return days_elapsed // 7


def resolve_target_dir() -> Path:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            return target

    week_num = get_current_week_number()
    return Path.home() / "Projects" / "DL_ranked_analysis" / f"week{week_num}_analysis"


DATA_DIR = resolve_target_dir()
OUTPUT_DIR = DATA_DIR / "graphs"
CSV_FILE_PATH = DATA_DIR / "hero_winrates.csv"


def load_and_transform_data(csv_path: Path) -> tuple[pd.DataFrame, int]:
    """Reads hero_winrates.csv and dynamically aggregates hero counts by bracket & archetype."""
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    df_raw = pd.read_csv(csv_path)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    # Standardize archetype names (e.g., 'brawler' -> 'Brawler')
    df_raw["hero_type"] = df_raw["hero_type"].str.strip().str.capitalize()

    # Define bracket boundaries (-inf to 46%, 46-48%, 48-49%, 49-51%, 51-52%, 52-54%, 54%+ inf)
    bins = [-float("inf"), 46.0, 48.0, 49.0, 51.0, 52.0, 54.0, float("inf")]
    bracket_labels = [
        "Trash\n(<46%)",
        "Weak\n(46-48%)",
        "Underperforming\n(48-49%)",
        "Balanced\n(49-51%)",
        "Overperforming\n(51-52%)",
        "Strong\n(52-54%)",
        "God Tier\n(>54%)",
    ]

    # Assign each hero to a win rate bracket
    df_raw["bracket"] = pd.cut(
        df_raw["winrate"], bins=bins, labels=bracket_labels, right=False
    )

    # Cross-tabulate archetype counts across brackets
    ct = pd.crosstab(df_raw["bracket"], df_raw["hero_type"])

    # Ensure all 4 archetypes exist in expected order
    archetype_order = ["Assassin", "Brawler", "Marksman", "Mystic"]
    for arch in archetype_order:
        if arch not in ct.columns:
            ct[arch] = 0

    ct = ct[archetype_order].reindex(bracket_labels, fill_value=0)
    total_heroes = len(df_raw)

    return ct, total_heroes


def main():
    # 1. Load and dynamically transform CSV data
    df, total_heroes = load_and_transform_data(CSV_FILE_PATH)
    brackets = df.index.tolist()

    # 2. Set up dark themed styling
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7))

    # 3. Plot empirical data as stacked columns
    # Matching hex colors to your existing archetype palette
    colors = ["#06b6d4", "#22c55e", "#f97316", "#D61FFF"]  # Assassin, Brawler, Marksman, Mystic
    df.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.35, alpha=0.85, zorder=3)

    # Empirical trend line (sums height of each stack)
    stacked_totals = df.sum(axis=1).values
    x_indices = np.arange(len(brackets))

    ax.plot(
        x_indices,
        stacked_totals,
        color="#00f5d4",  # Neon cyan trend line
        linestyle="-",
        linewidth=2.0,
        marker="o",
        markersize=6,
        label="Actual Hero Distribution",
        zorder=4,
    )

    # 4. Generate smooth theoretical bell curve dynamically centered on Balanced (index 3)
    x_smooth = np.linspace(0, len(brackets) - 1, 300)
    bell_y_smooth = norm.pdf(x_smooth, loc=3, scale=1.3)

    # Normalize bell curve peak height based on total hero count relative to 38
    peak_target = (total_heroes / 38.0) * 8.5
    bell_y_smooth = (bell_y_smooth / max(bell_y_smooth)) * peak_target

    ax.plot(
        x_smooth,
        bell_y_smooth,
        color="#ffffff",
        linestyle="--",
        linewidth=2.5,
        label="Theoretical Perfect Balance",
        zorder=5,
    )

    # 5. Styling & aesthetics
    ax.set_title(
        "Deadlock hero_type Balance vs. Theoretical Bell Curve",
        fontsize=16,
        pad=20,
        fontweight="bold",
    )
    ax.set_xlabel("Winrate Brackets", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Heroes", fontsize=12, labelpad=10)

    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)

    ax.grid(axis="y", linestyle=":", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        title="hero_types & Models",
        title_fontsize="11",
        fontsize="10",
        frameon=True,
        facecolor="#222222",
        edgecolor="none",
    )

    plt.tight_layout()

    # Save chart to designated graph output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "deadlock_balance_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Graph successfully saved to: {output_path}")


if __name__ == "__main__":
    main()