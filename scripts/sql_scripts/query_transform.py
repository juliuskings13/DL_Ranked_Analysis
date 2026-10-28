import logging
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Union
import pandas as pd
from sqlalchemy import TextClause, create_engine, text
from db_config import PG_CONFIG                  
from db_config import PG_CONFIG as DB_Params     
from db_config import DB_URI   

# script to run sql queries and transform output into readable csv files for the graphing scripts

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s %(message)s"
)

# ==========================================
# CONFIGURATION
# ==========================================

DB_URI = (
    f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}"
    f"@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
)

# Global engine instance for connection pooling across query functions
engine = create_engine(DB_URI)

OUTPUT_DIR = Path.home() / "Projects" / "DL_ranked_analysis"
# Ranked Launch Anchor Date: Thursday, July 30, 2026
RANKED_START_DATE = date(2026, 7, 30)


def get_current_week_number(target_date: Optional[date] = None) -> int:
    """Calculates the 1-indexed ranked week number based on a 7-day cadence."""
    if target_date is None:
        target_date = date.today()

    days_elapsed = (target_date - RANKED_START_DATE).days

    if days_elapsed < 0:
        raise ValueError("Target date cannot be before Ranked Launch (2026-07-30).")

    # Integer division by 7 
    return (days_elapsed // 7)


def get_weekly_output_dir(week_num:int) -> Path:
    """Generates and creates the weekly directory structure (e.g., ./week3_analysis)."""
    weekly_dir = OUTPUT_DIR / f"week{week_num}_analysis"

    # Create directory tree automatically
    weekly_dir.mkdir(parents=True, exist_ok=True)
    return weekly_dir


def verify_tables_exist(required_tables: list[str]) -> bool:
    check_query = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = ANY(:tables);
    """)
    with engine.connect() as conn:
        result = conn.execute(check_query, {"tables": required_tables})
        found_tables = {row[0] for row in result.fetchall()}

    missing = set(required_tables) - found_tables
    if missing:
        logging.error(f"aborting etl, missing required tables: {missing}")
        return False

    logging.info("all required database tables verified")
    return True


def run_query(
    query: Union[str, TextClause], params: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Executes a SQL query via SQLAlchemy connection and returns a pandas DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params=params)  # type: ignore


# ==========================================
# QUERY EXTRACTION METHODS
# ==========================================


def extract_hero_winrates(week_label: str) -> pd.DataFrame:
    """Aggregates overall Hero win rates, pick rates, and match counts."""
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time
            FROM patch_periods
            WHERE period_name = :week_label
        )
        SELECT
            h.name AS hero_name,
            h.hero_type,
            COUNT(mp.match_id) AS total_matches,
            ROUND((COUNT(mp.match_id) FILTER (WHERE mp.won = TRUE)::NUMERIC / COUNT(mp.match_id)) * 100, 2) AS winrate
        FROM match_player mp
        JOIN heroes h ON mp.hero_id = h.hero_id
        CROSS JOIN target_period tp
        WHERE mp.start_time >= tp.start_time
          AND (tp.end_time IS NULL OR mp.start_time <= tp.end_time)
        GROUP BY h.name, h.hero_type
        ORDER BY winrate DESC;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_hero_pairings(week_label: str) -> pd.DataFrame:
    """Calculates win rates for the top 20 duo synergies for 50% wr and +5k matches using the table."""
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time
            FROM patch_periods
            WHERE period_name = :week_label
        ),
        period_matches AS (
            SELECT 
                mp.match_id,
                mp.team,
                mp.hero_id,
                mp.won
            FROM match_player mp
            JOIN target_period tp 
              ON mp.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR mp.start_time < tp.end_time)
        ),
        duo_pairs AS (
            SELECT 
                m1.hero_id AS hero_a,
                m2.hero_id AS hero_b,
                m1.won
            FROM period_matches m1
            JOIN period_matches m2 
              ON m1.match_id = m2.match_id 
             AND m1.team = m2.team 
             AND m1.hero_id < m2.hero_id
        ),
        duo_winrates AS (
            SELECT 
                dp.hero_a,
                dp.hero_b,
                COUNT(*) AS matches_together,
                COUNT(*) FILTER (WHERE dp.won = TRUE) AS wins_together,
                ROUND(
                    (COUNT(*) FILTER (WHERE dp.won = TRUE)::NUMERIC / COUNT(*)) * 100, 2
                ) AS duo_win_rate
            FROM duo_pairs dp
            GROUP BY dp.hero_a, dp.hero_b
            HAVING COUNT(*) >= 20
        )
        SELECT 
            h1.name AS hero_name_a,
            h2.name AS hero_name_b,
            dw.matches_together,
            dw.duo_win_rate AS duo_winrate
        FROM duo_winrates dw
        JOIN heroes h1 ON dw.hero_a = h1.hero_id
        JOIN heroes h2 ON dw.hero_b = h2.hero_id
        WHERE dw.duo_win_rate >= 50 AND dw.matches_together >= 5000
        ORDER BY dw.duo_win_rate DESC
        LIMIT 20;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_global_presence_winrate(week_label: str) -> pd.DataFrame:
    """Extracts hero's global presence and winrate."""
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time FROM patch_periods WHERE period_name = :week_label
        ),
        week_meta AS (
            SELECT (COUNT(*)::NUMERIC / 12) AS total_games
            FROM match_player mp
            JOIN target_period tp 
              ON mp.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR mp.start_time < tp.end_time)
        ),
        period_players AS (
            SELECT mp.match_id, mp.hero_id, mp.won
            FROM match_player mp
            JOIN target_period tp 
              ON mp.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR mp.start_time < tp.end_time)
        )
        SELECT
            h.name AS hero_name,
            ROUND((COUNT(pp.match_id)::NUMERIC / wm.total_games) * 100, 2) AS global_presence,
            ROUND((COUNT(pp.match_id) FILTER (WHERE pp.won = TRUE)::NUMERIC / wm.total_games) * 100, 2) AS global_winrate
        FROM period_players pp
        CROSS JOIN week_meta wm
        JOIN heroes h ON pp.hero_id = h.hero_id
        GROUP BY h.name, wm.total_games
        ORDER BY global_presence DESC;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_unique_comp_freq_match(week_label: str) -> pd.DataFrame:
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time FROM patch_periods WHERE period_name = :week_label
        ),
        team_comps AS (
            SELECT m.team0 AS comp
            FROM match_hero_roster m
            JOIN target_period tp 
              ON m.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
            
            UNION ALL
            
            SELECT m.team1 AS comp
            FROM match_hero_roster m
            JOIN target_period tp 
              ON m.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
        ),
        comp_frequencies AS (
            SELECT comp, COUNT(*) AS freq
            FROM team_comps
            GROUP BY comp
        ),
        bucketed_comps AS (
            SELECT 
                freq,
                CASE 
                    WHEN freq = 1 THEN 'single-occurrence'
                    WHEN freq BETWEEN 2 AND 4 THEN '2 - 4 occurrences'
                    WHEN freq BETWEEN 5 AND 9 THEN '5 - 9 occurrences'
                    WHEN freq BETWEEN 10 AND 19 THEN '10 - 19 occurrences'
                    WHEN freq BETWEEN 20 AND 29 THEN '20 - 29 occurrences'
                    WHEN freq BETWEEN 30 AND 39 THEN '30 - 39 occurrences'
                    WHEN freq >= 40 THEN '40+ occurrences'
                END AS frequency_bucket,
                CASE 
                    WHEN freq = 1 THEN 1
                    WHEN freq BETWEEN 2 AND 4 THEN 2
                    WHEN freq BETWEEN 5 AND 9 THEN 3
                    WHEN freq BETWEEN 10 AND 19 THEN 4
                    WHEN freq BETWEEN 20 AND 29 THEN 5
                    WHEN freq BETWEEN 30 AND 39 THEN 6
                    WHEN freq >= 40 THEN 7
                END AS sort_order
            FROM comp_frequencies
        ),
        bucket_aggregates AS (
            SELECT 
                frequency_bucket,
                sort_order,
                COUNT(*) AS unique_team_comps,
                SUM(freq) AS total_games_played
            FROM bucketed_comps
            GROUP BY frequency_bucket, sort_order
        )
        SELECT 
            frequency_bucket,
            unique_team_comps,
            total_games_played,
            SUM(unique_team_comps) OVER (
                ORDER BY sort_order DESC 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_comps_N_plus,
            SUM(total_games_played) OVER (
                ORDER BY sort_order DESC 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_games_N_plus
        FROM bucket_aggregates
        ORDER BY sort_order ASC;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_top_20_freq_comps(week_label: str) -> pd.DataFrame:
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time FROM patch_periods WHERE period_name = :week_label
        ),
        all_teams AS (
            SELECT m.team0 AS comp_ids, CASE WHEN LOWER(m.winning_team) = 'team0' THEN 1 ELSE 0 END AS won
            FROM match_hero_roster m
            JOIN target_period tp ON m.start_time >= tp.start_time AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
            
            UNION ALL
            
            SELECT m.team1 AS comp_ids, CASE WHEN LOWER(m.winning_team) = 'team1' THEN 1 ELSE 0 END AS won
            FROM match_hero_roster m
            JOIN target_period tp ON m.start_time >= tp.start_time AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
        ),
        top_comps AS (
            SELECT 
                comp_ids,
                COUNT(*) AS total_matches,
                ROUND((SUM(won)::NUMERIC / COUNT(*)) * 100, 2) AS winrate
            FROM all_teams
            GROUP BY comp_ids
            ORDER BY total_matches DESC
            LIMIT 20
        )
        SELECT 
            (
                SELECT STRING_AGG(h.name, ', ' ORDER BY h.name)
                FROM UNNEST(tc.comp_ids) AS u(hero_id)
                JOIN heroes h ON h.hero_id = u.hero_id
            ) AS comp_names,
            tc.total_matches,
            tc.winrate
        FROM top_comps tc
        ORDER BY tc.total_matches DESC;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_top_20_freq_win_comps(week_label: str) -> pd.DataFrame:
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time FROM patch_periods WHERE period_name = :week_label
        ),
        all_teams AS (
            SELECT m.team0 AS comp_ids, CASE WHEN LOWER(m.winning_team) = 'team0' THEN 1 ELSE 0 END AS won
            FROM match_hero_roster m
            JOIN target_period tp ON m.start_time >= tp.start_time AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
            
            UNION ALL
            
            SELECT m.team1 AS comp_ids, CASE WHEN LOWER(m.winning_team) = 'team1' THEN 1 ELSE 0 END AS won
            FROM match_hero_roster m
            JOIN target_period tp ON m.start_time >= tp.start_time AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
        ),
        top_comps AS (
            SELECT 
                comp_ids,
                COUNT(*) AS total_matches,
                ROUND((SUM(won)::NUMERIC / COUNT(*)) * 100, 2) AS winrate
            FROM all_teams
            GROUP BY comp_ids
            ORDER BY total_matches DESC
        )
        SELECT 
            (
                SELECT STRING_AGG(h.name, ', ' ORDER BY h.name)
                FROM UNNEST(tc.comp_ids) AS u(hero_id)
                JOIN heroes h ON h.hero_id = u.hero_id
            ) AS comp_names,
            tc.total_matches,
            tc.winrate
        FROM top_comps tc
        WHERE winrate > 50
        ORDER BY total_matches DESC
        LIMIT 20;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_top_20_freq_lose_comps(week_label: str) -> pd.DataFrame:
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time FROM patch_periods WHERE period_name = :week_label
        ),
        all_teams AS (
            SELECT m.team0 AS comp_ids, CASE WHEN LOWER(m.winning_team) = 'team0' THEN 1 ELSE 0 END AS won
            FROM match_hero_roster m
            JOIN target_period tp ON m.start_time >= tp.start_time AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
            
            UNION ALL
            
            SELECT m.team1 AS comp_ids, CASE WHEN LOWER(m.winning_team) = 'team1' THEN 1 ELSE 0 END AS won
            FROM match_hero_roster m
            JOIN target_period tp ON m.start_time >= tp.start_time AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
        ),
        top_comps AS (
            SELECT 
                comp_ids,
                COUNT(*) AS total_matches,
                ROUND((SUM(won)::NUMERIC / COUNT(*)) * 100, 2) AS winrate
            FROM all_teams
            GROUP BY comp_ids
            ORDER BY total_matches DESC
        )
        SELECT 
            (
                SELECT STRING_AGG(h.name, ', ' ORDER BY h.name)
                FROM UNNEST(tc.comp_ids) AS u(hero_id)
                JOIN heroes h ON h.hero_id = u.hero_id
            ) AS comp_names,
            tc.total_matches,
            tc.winrate
        FROM top_comps tc
        WHERE winrate < 50
        ORDER BY total_matches DESC
        LIMIT 20;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_hero_freq_top_comps(week_label: str) -> pd.DataFrame:
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time FROM patch_periods WHERE period_name = :week_label
        ),
        team_comps AS (
            SELECT 
                mp.match_id, 
                mp.team,
                ARRAY_AGG(mp.hero_id ORDER BY mp.hero_id) AS comp_ids
            FROM match_player mp
            JOIN target_period tp 
              ON mp.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR mp.start_time < tp.end_time)
            GROUP BY mp.match_id, mp.team
        ), 
        top_comps AS (
            SELECT
                comp_ids,
                COUNT(*) AS total_matches
            FROM team_comps
            GROUP BY comp_ids
            ORDER BY total_matches DESC 
            LIMIT 20
        )
        SELECT 
            h.name AS hero_name,
            h.hero_type,
            COUNT(*) AS frequency
        FROM top_comps tc
        CROSS JOIN LATERAL UNNEST(tc.comp_ids) AS unnested(hero_id)
        JOIN heroes h ON h.hero_id = unnested.hero_id
        GROUP BY h.hero_id, h.name, h.hero_type
        ORDER BY frequency DESC;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_hero_freq_top_win_comps(week_label: str) -> pd.DataFrame:
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time FROM patch_periods WHERE period_name = :week_label
        ),
        team_comps AS (
            SELECT 
                mp.match_id, 
                mp.team,
                bool_or(mp.won) AS won,
                ARRAY_AGG(mp.hero_id ORDER BY mp.hero_id) AS comp_ids
            FROM match_player mp
            JOIN target_period tp 
              ON mp.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR mp.start_time < tp.end_time)
            GROUP BY mp.match_id, mp.team
        ), 
        top_comps AS (
            SELECT
                comp_ids,
                COUNT(*) AS total_matches
            FROM team_comps
            GROUP BY comp_ids
            HAVING (COUNT(*) FILTER (WHERE won = TRUE)::NUMERIC / COUNT(*)) > 0.50
            ORDER BY total_matches DESC 
            LIMIT 20
        )
        SELECT 
            h.name AS hero_name,
            h.hero_type,
            COUNT(*) AS frequency
        FROM top_comps tc
        CROSS JOIN LATERAL UNNEST(tc.comp_ids) AS unnested(hero_id)
        JOIN heroes h ON h.hero_id = unnested.hero_id
        GROUP BY h.hero_id, h.name, h.hero_type
        ORDER BY frequency DESC;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_hero_freq_top_lose_comps(week_label: str) -> pd.DataFrame:
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time FROM patch_periods WHERE period_name = :week_label
        ),
        team_comps AS (
            SELECT 
                mp.match_id, 
                mp.team,
                bool_or(mp.won) AS won,
                ARRAY_AGG(mp.hero_id ORDER BY mp.hero_id) AS comp_ids
            FROM match_player mp
            JOIN target_period tp 
              ON mp.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR mp.start_time < tp.end_time)
            GROUP BY mp.match_id, mp.team
        ), 
        top_comps AS (
            SELECT
                comp_ids,
                COUNT(*) AS total_matches
            FROM team_comps
            GROUP BY comp_ids
            HAVING (COUNT(*) FILTER (WHERE won = TRUE)::NUMERIC / COUNT(*)) < 0.50
            ORDER BY total_matches DESC 
            LIMIT 20
        )
        SELECT 
            h.name AS hero_name,
            h.hero_type,
            COUNT(*) AS frequency
        FROM top_comps tc
        CROSS JOIN LATERAL UNNEST(tc.comp_ids) AS unnested(hero_id)
        JOIN heroes h ON h.hero_id = unnested.hero_id
        GROUP BY h.hero_id, h.name, h.hero_type
        ORDER BY frequency DESC;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_hero_type_diff_winrate(week_label: str) -> pd.DataFrame:
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time FROM patch_periods WHERE period_name = :week_label
        ),
        period_matches AS (
            SELECT mp.*
            FROM match_player mp
            JOIN target_period tp 
              ON mp.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR mp.start_time < tp.end_time)
        ),
        team_type_counts AS (
            SELECT
                pm.match_id,
                pm.team,
                pm.won,
                h.hero_type,
                COUNT(pm.hero_id) AS type_count
            FROM period_matches pm
            JOIN heroes h ON pm.hero_id = h.hero_id
            GROUP BY pm.match_id, pm.team, pm.won, h.hero_type
        ),
        differentials AS (
            SELECT
                t1.match_id,
                t1.hero_type,
                t1.won AS surplus_team_won,
                t1.type_count - COALESCE(t2.type_count, 0) AS diff
            FROM team_type_counts t1
            LEFT JOIN team_type_counts t2 
                ON t1.match_id = t2.match_id 
                AND t1.hero_type = t2.hero_type 
                AND t1.team != t2.team
            WHERE t1.type_count >= COALESCE(t2.type_count, 0)
               OR (t1.type_count = COALESCE(t2.type_count, 0) AND t1.team = 'team0')
        )
        SELECT 
            hero_type,
            diff AS type_differential,
            COUNT(*) AS match_frequency,
            ROUND((COUNT(*) FILTER (WHERE surplus_team_won)::NUMERIC / COUNT(*)) * 100, 2) AS winrate_of_surplus_team
        FROM differentials
        GROUP BY hero_type, diff
        ORDER BY hero_type, diff;
    """)
    return run_query(query, params={"week_label": week_label})


def extract_herotype_comp_wr(hero_type: str, week_label: str) -> pd.DataFrame:
    query = text("""
        WITH match_counts AS (
            SELECT 
                m.match_id,
                m.winning_team,
                (
                    SELECT COUNT(*) 
                    FROM unnest(m.team0) AS t0_id
                    JOIN heroes h0 ON h0.hero_id = t0_id
                    WHERE LOWER(h0.hero_type) = :hero_type
                ) AS team0_type_count,
                (
                    SELECT COUNT(*) 
                    FROM unnest(m.team1) AS t1_id
                    JOIN heroes h1 ON h1.hero_id = t1_id
                    WHERE LOWER(h1.hero_type) = :hero_type
                ) AS team1_type_count
            FROM match_hero_roster m
            JOIN patch_periods tp ON tp.period_name = :week_label
              AND m.start_time >= tp.start_time 
              AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
        ),
        exact_matchups AS (
            SELECT 
                team0_type_count AS t0_count,
                team1_type_count AS t1_count,
                CASE WHEN LOWER(winning_team) = 'team0' THEN 1 ELSE 0 END AS win
            FROM match_counts
            UNION ALL
            SELECT 
                team1_type_count AS t0_count,
                team0_type_count AS t1_count,
                CASE WHEN LOWER(winning_team) = 'team1' THEN 1 ELSE 0 END AS win
            FROM match_counts
        )
        SELECT 
            t0_count,
            t1_count,
            COUNT(*) AS total_matchups,
            ROUND(AVG(win) * 100, 2) AS t0_winrate
        FROM exact_matchups
        GROUP BY t0_count, t1_count
        ORDER BY total_matchups DESC, t0_winrate DESC;
    """)
    ht = hero_type.lower()
    return run_query(query, params={"hero_type": ht, "week_label": week_label})


def extract_avg_duration(week_label: str) -> pd.DataFrame:
    """Segments match duration into 5-min buckets to analyze avg match duration."""
    query = text("""
        SELECT 
            CASE WIDTH_BUCKET(m.match_duration_s / 60, 20, 60, 8)
                WHEN 1 THEN '20-25m'
                WHEN 2 THEN '25-30m'
                WHEN 3 THEN '30-35m'
                WHEN 4 THEN '35-40m'
                WHEN 5 THEN '40-45m'
                WHEN 6 THEN '45-50m'
                WHEN 7 THEN '50-55m'
                WHEN 8 THEN '55-60m'
                ELSE '60m+'
            END AS duration_bracket,
            COUNT(*) AS total_games
        FROM match_hero_roster m
        JOIN patch_periods tp ON tp.period_name = :week_label
          AND m.start_time >= tp.start_time 
          AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
        WHERE m.match_duration_s >= 1200
        GROUP BY duration_bracket
        ORDER BY duration_bracket;
    """)
    return run_query(query, params={"week_label": week_label})

def extract_duration_minutes(week_label: str) -> pd.DataFrame:
    """Matches per whole minute of length (1-minute bins, no 20-minute floor).
    Feeds match_duration.py and the IQR-based early/late cut-offs in chart_stats.py."""
    query = text("""
        SELECT
            (m.match_duration_s / 60) AS duration_min,   -- integer division: 37:42 -> 37
            COUNT(*) AS games
        FROM match_hero_roster m
        JOIN patch_periods tp ON tp.period_name = :week_label
          AND m.start_time >= tp.start_time
          AND (tp.end_time IS NULL OR m.start_time < tp.end_time)
        GROUP BY duration_min
        ORDER BY duration_min;
    """)
    return run_query(query, params={"week_label": week_label})



def extract_duration_vs_winrate(week_label: str) -> pd.DataFrame:
    """Segments match duration into 5-minute buckets to analyze win rates over time."""
    query = text("""
        WITH target_period AS (
            SELECT start_time, end_time
            FROM patch_periods
            WHERE period_name = :week_label
        ),
        period_matches AS (
            SELECT 
                mp.match_id,
                mp.team,
                mp.hero_id,
                mp.duration_s,
                mp.won
            FROM match_player mp
            JOIN target_period tp 
              ON mp.start_time >= tp.start_time 
             AND (tp.end_time IS NULL OR mp.start_time < tp.end_time)
        )
        SELECT 
            h."name" AS hero_name,
            CASE WIDTH_BUCKET(pm.duration_s / 60, 20, 60, 8)
                WHEN 1 THEN '20-25m'
                WHEN 2 THEN '25-30m'
                WHEN 3 THEN '30-35m'
                WHEN 4 THEN '35-40m'
                WHEN 5 THEN '40-45m'
                WHEN 6 THEN '45-50m'
                WHEN 7 THEN '50-55m'
                WHEN 8 THEN '55-60m'
                ELSE '60m+'
            END AS duration_bracket,
            COUNT(*) AS total_games,
            ROUND(
                (COUNT(*) FILTER (WHERE won = TRUE)::NUMERIC / COUNT(*)) * 100, 2
            ) AS winrate
        FROM period_matches pm
        JOIN heroes h ON pm.hero_id = h.hero_id
        WHERE pm.duration_s >= 1200
        GROUP BY hero_name, duration_bracket
        ORDER BY hero_name, duration_bracket;
    """)
    return run_query(query, params={"week_label": week_label})


# ==========================================
# 3. OUTPUT FORMATTERS / EXPORTERS
# ==========================================

def export_dataset(
    df: pd.DataFrame,
    filename: str,
    output_dir: Path,
    formats: list[str] = ["csv", "parquet"],
):
    """Writes DataFrames out to disk in specified file formats."""
    base_path = output_dir / filename

    if "csv" in formats:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved CSV     -> {csv_path}")

    if "parquet" in formats:
        parquet_path = base_path.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        logging.info(f"Saved Parquet -> {parquet_path}")


# ==========================================
# 4. ETL RUNNER
# ==========================================
def run_etl(target_date: Optional[date] = None):
    logging.info("Starting SQL Query ETL pipeline...")

    # Calculate target week dynamically from target_date
    week_num = get_current_week_number(target_date)
    week_label = f"Week_{week_num}"
    logging.info(f"Targeting pipeline for {week_label} (Date: {target_date or date.today()})")

    required_tables = ["match_player", "heroes", "match_hero_roster", "patch_periods"]
    if not verify_tables_exist(required_tables):
        return

    # Pass target week/label into directory resolver
    curr_week_output_dir = get_weekly_output_dir(week_num=week_num)
    logging.info(f"Output directory set to: {curr_week_output_dir}")

    # Pass week_label/week_num to extraction functions if needed
    queries_to_run = [
        ("hero_winrates", lambda: extract_hero_winrates(week_label), ["csv"]),
        ("hero_duo_synergies", lambda: extract_hero_pairings(week_label), ["csv"]),
        ("global_presence_winrate", lambda: extract_global_presence_winrate(week_label), ["csv"]),
        ("unique_comp_freq_match", lambda: extract_unique_comp_freq_match(week_label), ["csv"]),
        ("top_20_freq_comps", lambda: extract_top_20_freq_comps(week_label), ["csv"]),
        ("top_20_freq_win_comps", lambda: extract_top_20_freq_win_comps(week_label), ["csv"]),
        ("top_20_freq_lose_comps", lambda: extract_top_20_freq_lose_comps(week_label), ["csv"]),
        ("hero_freq_top_comps", lambda: extract_hero_freq_top_comps(week_label), ["csv"]),
        ("hero_freq_top_win_comps", lambda: extract_hero_freq_top_win_comps(week_label), ["csv"]),
        ("hero_freq_top_lose_comps", lambda: extract_hero_freq_top_lose_comps(week_label), ["csv"]),
        ("hero_type_diff_winrate", lambda: extract_hero_type_diff_winrate(week_label), ["csv"]),
        ("brawler_comp_wr", lambda: extract_herotype_comp_wr("brawler", week_label), ["csv"]),
        ("mystic_comp_wr", lambda: extract_herotype_comp_wr("mystic", week_label), ["csv"]),
        ("marksman_comp_wr", lambda: extract_herotype_comp_wr("marksman", week_label), ["csv"]),
        ("assassin_comp_wr", lambda: extract_herotype_comp_wr("assassin", week_label), ["csv"]),
        ("avg_duration", lambda: extract_avg_duration(week_label), ["csv"]),
        ("match_duration_minutes", lambda: extract_duration_minutes(week_label), ["csv"]),
        ("hero_duration_scaling", lambda: extract_duration_vs_winrate(week_label), ["parquet"]),
    ]

    for name, extract_fn, formats in queries_to_run:
        logging.info(f"Running extract function: {name}...")
        try:
            df = extract_fn()
            if df.empty:
                logging.warning(f"Query {name} returned no rows. Skipping export.")
                continue

            export_dataset(df, filename=name, output_dir=curr_week_output_dir, formats=formats)
        except Exception as e:
            logging.error(f"Failed executing {name}: {e}")

    logging.info("ETL pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQL Query ETL Pipeline")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD target date")
    args = parser.parse_args()

    parsed_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    run_etl(target_date=parsed_date)