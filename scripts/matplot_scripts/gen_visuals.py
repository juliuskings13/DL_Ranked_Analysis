import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

script_dir = Path(__file__).parent
RANKED_START_DATE = date(2026, 7, 30)

parser = argparse.ArgumentParser(description="Matplotlib Visualization Suite")
parser.add_argument(
    "--start-date", type=str, default=None, help="YYYY-MM-DD target date"
)
args = parser.parse_args()

# Determine target date (CLI arg or today)
if args.start_date:
    target_d = datetime.strptime(args.start_date, "%Y-%m-%d").date()
else:
    target_d = date.today()

# Calculate completed week number
days_elapsed = (target_d - RANKED_START_DATE).days
week_num = days_elapsed // 7

# Construct target week directory path
target_dir = (
    Path.home()
    / "Projects"
    / "DL_ranked_analysis"
    / f"week{week_num}_analysis"
)

# Every script renders 1080x1080 cards via theme.py / chart_stats.py
plotting_scripts = [
    "pick_rate_winrate_scatter.py",            # replaces hero_wr_matches_representation.py
    "hero_wr_total_matches_split_column.py",   # top/bottom 10 + per-archetype win-rate cards
    "balance_wr_distribution.py",
    "archetype_stacking.py",                   # replaces hero_type_differentials.py
    "matchup_bubble_chart.py",                 # archetype count matrices
    "hero_strength_and_scaling.py",            # early-vs-late scatter + win-rate and scaling heatmaps
    "hero_duo_synergy_table.py",               # best / worst synergy lift
    "most_freq_comps.py",
    "comp_uniqueness.py",                      # replaces the pie + decay charts
    "match_duration.py",                       # match-length IQR (defines early / late game)
    "week_over_week.py",                       # tier list + biggest movers, writes hero_deltas.csv
]

failed = []
for script in plotting_scripts:
    script_path = script_dir / script
    print(f"running {script} for directory: {target_dir}")
    # sys.executable = the same Python (and venv) that launched this runner
    result = subprocess.run([sys.executable, str(script_path), str(target_dir)])
    if result.returncode != 0:
        failed.append(script)

if failed:
    print(f"\n{len(failed)} chart script(s) failed: {', '.join(failed)}")
    sys.exit(1)
print(f"\nAll {len(plotting_scripts)} chart scripts finished. Cards are in {target_dir / 'graphs'}")