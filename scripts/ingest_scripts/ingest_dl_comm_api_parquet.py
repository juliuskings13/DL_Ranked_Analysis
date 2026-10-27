import argparse
import json
import duckdb
import psycopg2
from psycopg2.extras import execute_values
from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple 
import requests
from db_config import PG_CONFIG                  
from db_config import PG_CONFIG as DB_Params     
from db_config import DB_URI                     
# CONFIGURATION

DEFAULT_START_TIME = "2026-08-20 00:00:00"
BATCH_SIZE = 10000
DUCKLAKE_URL = "ducklake:https://s3-cache.deadlock-api.com/fast/db_snapshot.ducklake"
MANIFEST_URL = "https://data.deadlock-api.com/v1/manifest.json"

RANKED_START_DATE = date(2026, 7, 30)
def get_parquet_urls(table_name: str) -> list[str]:
    """Fetches active parquet file URLs for a given table from the API manifest."""
    res = requests.get(MANIFEST_URL, timeout=10)
    res.raise_for_status()
    manifest = res.json()
    
    base_url = manifest["public_url"]
    table_meta = manifest["tables"][table_name]
    gen = table_meta.get("generation", 0)
    
    urls = [
        f"{base_url}/{f['key']}"
        for f in table_meta["files"]
        if table_meta.get("policy") == "snapshot" or f.get("generation", 0) == gen
    ]
    return urls

def get_current_week_number(target_date: Optional[date] = None) -> int:
    if target_date is None:
        target_date = date.today()

    days_elapsed = (target_date - RANKED_START_DATE).days
    if days_elapsed < 0:
        raise ValueError("Target date cannot be before Ranked Launch (2026-07-30).")

    return (days_elapsed // 7)

def get_week_bounds(target_date: Optional[date] = None) -> Tuple[int, datetime, datetime]:
    week_num = get_current_week_number(target_date)

    week_start_date = RANKED_START_DATE + timedelta(days=(week_num - 1) * 7)
    week_end_date = week_start_date + timedelta(days=6)

    if week_num == 1:
        start_time = datetime.combine(week_start_date, time(13, 14, 0))
    else:
        start_time = datetime.combine(week_start_date, time.min)

    end_time = datetime.combine(week_end_date, time.max)

    return week_num, start_time, end_time

# 1. LOCAL POSTGRES SCHEMA SETUP
def setup_local_postgres():
    """Ensures local Postgres tables and primary keys exist."""
    with psycopg2.connect(**PG_CONFIG) as conn:
        with conn.cursor() as cur:
            print("🔧 Ensuring PostgreSQL schema is ready...")

            # Updated table schema with explicit stat columns
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS match_player (
                    -- 8-Byte Fields (Align 8)
                    match_id BIGINT,
                    account_id BIGINT,
                    start_time TIMESTAMP WITH TIME ZONE,

                    -- 4-Byte Fields (Align 4)
                    duration_s INT,
                    net_worth INT,
                    player_damage INT,
                    player_damage_taken INT,
                    damage_absorbed INT,
                    damage_mitigated INT,
                    player_healing INT,
                    heal_lost INT,
                    max_health INT,
                    self_healing INT,
                    self_damage INT,
                    heal_prevented INT,
                    teammate_healing INT,
                    player_barriering INT,
                    absorption_provided INT,

                    -- 2-Byte Fields (Align 2) -- Downcasted from INT4
                    hero_id SMALLINT,
                    kills SMALLINT,
                    deaths SMALLINT,
                    assists SMALLINT,
                    last_hits SMALLINT,
                    denies SMALLINT,
                    assigned_lane SMALLINT,
                    average_badge_team0 SMALLINT,
                    average_badge_team1 SMALLINT,
                    mvp_rank SMALLINT,

                    -- Enums / Strings / Booleans (Align 1 or Var)
                    won BOOLEAN,
                    match_mode VARCHAR(20),
                    game_mode VARCHAR(20),
                    winning_team VARCHAR(10),
                    team VARCHAR(10),

                    PRIMARY KEY (match_id, account_id)
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS match_hero_roster (
                    match_id BIGINT,
                    start_time TIMESTAMP WITH TIME ZONE,
                    match_duration_s INT,
                    winning_team VARCHAR(20),
                    team0 SMALLINT[],
                    team1 SMALLINT[],
                    average_badge_team0 SMALLINT NULL,
                    average_badge_team1 SMALLINT NULL,
                    PRIMARY KEY (match_id)
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS patch_periods (
                    period_id SMALLINT PRIMARY KEY,
                    period_name VARCHAR(100),
                    start_time TIMESTAMP WITH TIME ZONE,
                    end_time TIMESTAMP WITH TIME ZONE,
                    description VARCHAR(100)
                );
                """
            )

    print("✅ Schema ready.\n")


# 2. INITIALIZE DUCKLAKE CONNECTION
def get_ducklake_connection():
    print("🦆 Connecting to remote DuckDB via API Manifest...")
    con = duckdb.connect()
    
    con.execute("SET http_retries = 5;")
    con.execute("SET http_retry_wait_ms = 1000;")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # Load Parquet file paths dynamically into DuckDB views
    mp_urls = get_parquet_urls("match_player")
    con.execute(f"CREATE VIEW match_player AS SELECT * FROM read_parquet({mp_urls!r}, union_by_name = true);")

    return con


# 3. BULK INGESTION WITH CONFLICT HANDLING
def stream_ducklake_to_postgres(con, query, insert_sql, transform_fn, label=""):
    print(f"🚀 Initializing S3 stream for '{label}'...")

    cursor = con.execute(query)

    total_inserted = 0
    with psycopg2.connect(**PG_CONFIG) as conn:
        with conn.cursor() as cur:
            while True:
                rows = cursor.fetchmany(BATCH_SIZE)
                if not rows:
                    break

                records = [transform_fn(r) for r in rows]

                execute_values(cur, insert_sql, records)
                conn.commit()

                total_inserted += len(records)
                print(f"   -> Processed {total_inserted} rows...")

    print(f"🎉 Finished '{label}'! Successfully processed {total_inserted} records.\n")


# 4. POPULATE MATCH_HERO_ROSTER FROM MATCH_PLAYER
def populate_mhr():
    with psycopg2.connect(**PG_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO match_hero_roster (
                    match_id, 
                    start_time, 
                    match_duration_s,
                    winning_team, 
                    team0, 
                    team1,
                    average_badge_team0,
                    average_badge_team1
                )
                SELECT 
                    mp.match_id,
                    MIN(mp.start_time) AS start_time,
                    MAX(mp.duration_s) AS match_duration_s,
                    MAX(CASE WHEN mp.won = TRUE THEN mp.team END) AS winning_team,
                    ARRAY_AGG(mp.hero_id ORDER BY mp.hero_id) FILTER (WHERE LOWER(mp.team) = 'team0') AS team0,
                    ARRAY_AGG(mp.hero_id ORDER BY mp.hero_id) FILTER (WHERE LOWER(mp.team) = 'team1') AS team1,
                    MAX(mp.average_badge_team0) AS average_badge_team0,
                    MAX(mp.average_badge_team1) AS average_badge_team1
                FROM match_player mp
                GROUP BY mp.match_id
                ON CONFLICT (match_id) DO UPDATE 
                SET 
                    start_time = EXCLUDED.start_time,
                    match_duration_s = EXCLUDED.match_duration_s,
                    winning_team = EXCLUDED.winning_team,
                    team0 = EXCLUDED.team0,
                    team1 = EXCLUDED.team1,
                    average_badge_team0 = EXCLUDED.average_badge_team0,
                    average_badge_team1 = EXCLUDED.average_badge_team1;
                """
            )
    print("finished populating match_hero_roster!\n")


# 5. INSERT CURRENT WEEK INTO PATCH_PERIODS
def insert_curr_week(week_num: Optional[int] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
    period_name = f"Week_{week_num}"
    description = f"Ranked Week {week_num}"

    print(f"🗓️ Syncing {period_name}: {start_time} to {end_time}")

    upsert_sql = """
        INSERT INTO patch_periods (period_id, period_name, start_time, end_time, description)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (period_id) DO UPDATE 
        SET 
            period_name = EXCLUDED.period_name,
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            description = EXCLUDED.description;
    """

    with psycopg2.connect(**PG_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(upsert_sql, (week_num, period_name, start_time, end_time, description))
        conn.commit()
    print(f" Successfully registered {period_name} in patch_periods.")


# 6. MAIN WORKFLOW
def main():
    parser = argparse.ArgumentParser(description="Deadlock DuckLake Telemetry Ingestion")
    parser.add_argument("--start-date", type=str, default=str(date.today()))
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    target_d = datetime.strptime(args.start_date[:10], "%Y-%m-%d").date()
    week_num, start_time, end_time = get_week_bounds(target_d)

    insert_curr_week(week_num, start_time, end_time)

    print(f"📅 Running STRICT RANKED ingest for Week {week_num}")
    print(f"   Start: {start_time} | End: {end_time}")

    setup_local_postgres()
    con = get_ducklake_connection()

    # Print the exact column names and their data types
    #schema_info = con.execute("DESCRIBE dl_db.main.match_player;").fetchall()
    #for col in schema_info:
     #   if "stats" in col[0]:
     #       print(f"Column: {col[0]:<35} Type: {col[1]}")
    #return

    try:
        # Fixed Query: Aliased source table as mp to resolve stats struct references
        # Fixed Query: Double-quote column names containing dots
        match_player_query = f"""
            SELECT 
                mp.match_id, 
                mp.account_id, 
                mp.start_time, 
                mp.duration_s, 
                CAST(mp.match_mode AS VARCHAR) AS match_mode, 
                CAST(mp.game_mode AS VARCHAR) AS game_mode, 
                CAST(mp.winning_team AS VARCHAR) AS winning_team, 
                CAST(mp.team AS VARCHAR) AS team, 
                mp.won, 
                mp.hero_id, 
                mp.kills, 
                mp.deaths, 
                mp.assists, 
                mp.net_worth, 
                mp.last_hits, 
                mp.denies, 
                mp.assigned_lane,
                mp.average_badge_team0, 
                mp.average_badge_team1, 
                mp.mvp_rank,

                -- Double quotes around flat dotted column names:
                COALESCE(CAST(list_max(mp."stats.heal_lost") AS INT), 0)           AS heal_lost,
                COALESCE(CAST(list_max(mp."stats.max_health") AS INT), 0)          AS max_health,
                COALESCE(CAST(list_max(mp."stats.self_healing") AS INT), 0)        AS self_healing,
                COALESCE(CAST(list_max(mp."stats.self_damage") AS INT), 0)         AS self_damage,
                COALESCE(CAST(list_max(mp."stats.player_damage") AS INT), 0)       AS player_damage,
                COALESCE(CAST(list_max(mp."stats.heal_prevented") AS INT), 0)      AS heal_prevented,
                COALESCE(CAST(list_max(mp."stats.player_healing") AS INT), 0)      AS player_healing,
                COALESCE(CAST(list_max(mp."stats.damage_absorbed") AS INT), 0)     AS damage_absorbed,
                COALESCE(CAST(list_max(mp."stats.damage_mitigated") AS INT), 0)    AS damage_mitigated,
                COALESCE(CAST(list_max(mp."stats.teammate_healing") AS INT), 0)    AS teammate_healing,
                COALESCE(CAST(list_max(mp."stats.player_barriering") AS INT), 0)   AS player_barriering,
                COALESCE(CAST(list_max(mp."stats.absorption_provided") AS INT), 0) AS absorption_provided,
                COALESCE(CAST(list_max(mp."stats.player_damage_taken") AS INT), 0) AS player_damage_taken
            FROM match_player mp
            WHERE mp.start_time >= '{start_time}'
              AND mp.start_time <= '{end_time}'
              AND (CAST(mp.match_mode AS VARCHAR) = 'Ranked' OR TRY_CAST(mp.match_mode AS INT) = 4)
        """
        
        # Fixed Insert statement mapping all 33 fields
        match_player_insert = """
            INSERT INTO match_player (
                match_id, account_id, start_time, duration_s, match_mode, game_mode,
                winning_team, team, won, hero_id, kills, deaths, assists,
                net_worth, last_hits, denies, assigned_lane,
                average_badge_team0, average_badge_team1, mvp_rank,
                heal_lost, max_health, self_healing, self_damage, player_damage,
                heal_prevented, player_healing, damage_absorbed, damage_mitigated,
                teammate_healing, player_barriering, absorption_provided, player_damage_taken
            ) VALUES %s
            ON CONFLICT (match_id, account_id) DO NOTHING;
        """

        def match_player_transform(r):
            return (
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19],
                r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29],
                r[30], r[31], r[32]
            )

        stream_ducklake_to_postgres(
            con=con,
            query=match_player_query,
            insert_sql=match_player_insert,
            transform_fn=match_player_transform,
            label="match_player",
        )

    except Exception as e:
        print(f"s3 stream encountered error : {e}")

    finally:
        populate_mhr()


if __name__ == "__main__":
    main()