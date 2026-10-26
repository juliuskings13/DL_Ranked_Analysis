import argparse
import subprocess
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

plotting_scripts = [
    "balance_wr_distribution.py",
    "global_pres_wr_dumbbell_chart.py",
    "hero_duo_synergy_table.py",
    "hero_duration_heatmaps.py",
    "hero_freq_on_most_freq_comps.py",
    "hero_type_differentials.py",
    "hero_wr_matches_representation.py",
    "hero_wr_total_matches_split_column.py",
    "matchup_bubble_chart.py",
    "most_freq_comps.py",
    "unique_comps_pie_charts.py",
    "unique_comps_vs_match_volume_decay.py",
]

for script in plotting_scripts:
    script_path = script_dir / script
    print(f"running {script} for directory: {target_dir}")
    # Pass target_dir as positional arg sys.argv[1]
    subprocess.run(["python3", str(script_path), str(target_dir)], check=True)