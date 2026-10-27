import json
import time
import urllib.parse
import psycopg2
from psycopg2.extras import execute_values
import requests
from db_config import PG_CONFIG                  
from db_config import PG_CONFIG as DB_Params     
from db_config import DB_URI   

# ==========================================
# CONFIGURATION
# ==========================================
DEADLOCK_API_SQL_URL = "https://api.deadlock-api.com/v1/sql"

# Patch timestamp (July 30, 2026 Ranked update)
PATCH_START_TIME = "2026-07-30 00:00:00"

# Fetch batch size
BATCH_SIZE = 2500

HEADERS = {"User-Agent": "Mozilla/5.0 (Deadlock Analytics Local Dev)"}


# ==========================================
# 1. LOCAL POSTGRES SETUP
# ==========================================
def setup_local_postgres():
    """Sets up local tables corresponding to ClickHouse schemas."""
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    print("🔧 Setting up PostgreSQL schema...")

    # Items dimension table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id BIGINT PRIMARY KEY,
            name VARCHAR(100),
            tier INT,
            type VARCHAR(50),
            slot_type VARCHAR(50),
            cost INT
        );
    """
    )

    # Roster context table (Strictly Ranked matches)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_match_roster (
            match_id BIGINT,
            account_id BIGINT,
            start_time TIMESTAMP WITH TIME ZONE,
            game_mode VARCHAR(50),
            match_duration_s INT,
            won BOOLEAN,
            mate_ids BIGINT[],
            enemy_ids BIGINT[],
            PRIMARY KEY (match_id, account_id)
        );
    """
    )

    # Core telemetry fact table (Strictly Ranked players/matches)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS match_player (
            match_id BIGINT,
            account_id BIGINT,
            start_time TIMESTAMP WITH TIME ZONE,
            duration_s INT,
            match_mode VARCHAR(50),
            game_mode VARCHAR(50),
            winning_team VARCHAR(20),
            team VARCHAR(20),
            won BOOLEAN,
            hero_id INT,
            kills INT,
            deaths INT,
            assists INT,
            net_worth INT,
            last_hits INT,
            denies INT,
            assigned_lane INT,
            final_stats JSONB,
            PRIMARY KEY (match_id, account_id)
        );
    """
    )

    # Derived View: Matches table
    cur.execute(
        """
        CREATE OR REPLACE VIEW matches AS
        SELECT 
            match_id,
            MIN(start_time) AS start_time,
            MAX(duration_s) AS duration_s,
            MAX(winning_team) AS winning_team,
            MAX(match_mode) AS match_mode,
            MAX(game_mode) AS game_mode
        FROM match_player
        GROUP BY match_id;
    """
    )

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Local PostgreSQL schema ready.\n")


# ==========================================
# 2. CLICKHOUSE QUERY RUNNER
# ==========================================
def execute_clickhouse_query(sql_query, attempt=1):
    encoded_query = urllib.parse.quote(sql_query)
    url = f"{DEADLOCK_API_SQL_URL}?query={encoded_query}&format=json"
    
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 429:
        # Calculate exponential backoff: 5s, 10s, 20s, 40s...
        raw_wait = 5 * (2 ** (attempt - 1))

        max_wait_secs = 120

        wait_time = min(raw_wait, max_wait_secs)
        print(f"  ⚠️ Throttled (429)! Exponential backoff: Waiting {wait_time}s (Attempt #{attempt})...")
        time.sleep(wait_time)
        return execute_clickhouse_query(sql_query, attempt=attempt + 1)
    elif response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    return response.json()

# ==========================================
# 3. INGESTION ENGINE
# ==========================================
def ingest_items():
    """Syncs static dimension items table."""
    print("📦 Syncing items table...")
    query = "SELECT id, name, tier, type, slot_type, cost FROM items"
    items_data = execute_clickhouse_query(query)

    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    insert_query = """
        INSERT INTO items (id, name, tier, type, slot_type, cost)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            tier = EXCLUDED.tier,
            type = EXCLUDED.type,
            slot_type = EXCLUDED.slot_type,
            cost = EXCLUDED.cost;
    """

    records = [
        (
            r["id"],
            r["name"],
            r.get("tier"),
            r.get("type"),
            r.get("slot_type"),
            r.get("cost"),
        )
        for r in items_data
    ]
    execute_values(cur, insert_query, records)
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Synced {len(records)} items.\n")


def batch_ingest_table(
    table_name, select_cols, insert_sql, filter_condition, record_transform_fn, start_offset
):
    """Generic batch pull from ClickHouse -> Postgres insert."""
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    offset = start_offset
    total_inserted = 0
    print(
        f"🚀 Starting Ranked-only ingestion for '{table_name}' since {PATCH_START_TIME}..."
    )

    while True:
        ch_sql = f"""
            SELECT {select_cols}
            FROM {table_name}
            WHERE {filter_condition}
            ORDER BY match_id ASC, account_id ASC
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """

        print(
            f"  -> Fetching {table_name} rows {offset} to {offset + BATCH_SIZE}..."
        )
        rows = execute_clickhouse_query(ch_sql)

        if not rows:
            print(
                f"🎉 Completed {table_name}! Inserted total of {total_inserted} records.\n"
            )
            break

        records = [record_transform_fn(r) for r in rows]
        execute_values(cur, insert_sql, records)
        conn.commit()

        count = len(rows)
        total_inserted += count
        offset += BATCH_SIZE

        # Rest between query batches to respect rate limits
        time.sleep(10)

    cur.close()
    conn.close()


# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    setup_local_postgres()

    # 1. Sync items
    ingest_items()
    time.sleep(3)

    # 2. Ingest match_player (Strictly Ranked matches & players)
    # Checks both string 'Ranked' and numeric code 4 for compatibility
    match_player_select = """
        match_id, account_id, start_time, duration_s, match_mode, game_mode, 
        winning_team, team, won, hero_id, kills, deaths, assists, 
        net_worth, last_hits, denies, assigned_lane, final_stats
    """
    match_player_insert = """
        INSERT INTO match_player (
            match_id, account_id, start_time, duration_s, match_mode, game_mode,
            winning_team, team, won, hero_id, kills, deaths, assists,
            net_worth, last_hits, denies, assigned_lane, final_stats
        ) VALUES %s
        ON CONFLICT (match_id, account_id) DO NOTHING;
    """
    match_player_filter = f"""
        (match_mode = 'Ranked' OR match_mode = '4') 
        AND start_time >= '{PATCH_START_TIME}'
    """
    match_player_transform = lambda r: (
        r["match_id"],
        r["account_id"],
        r["start_time"],
        r["duration_s"],
        r["match_mode"],
        r["game_mode"],
        r["winning_team"],
        r["team"],
        r["won"],
        r["hero_id"],
        r["kills"],
        r["deaths"],
        r["assists"],
        r["net_worth"],
        r["last_hits"],
        r["denies"],
        r["assigned_lane"],
        json.dumps(r["final_stats"]) if r.get("final_stats") else None,
    )

    batch_ingest_table(
        "match_player",
        match_player_select,
        match_player_insert,
        match_player_filter,
        match_player_transform,
        start_offset=15000
    )

    # 3. Ingest player_match_roster (Filtered ONLY to matches in match_player)
    roster_select = "match_id, account_id, start_time, game_mode, match_duration_s, won, mate_ids, enemy_ids"
    roster_insert = """
        INSERT INTO player_match_roster 
        (match_id, account_id, start_time, game_mode, match_duration_s, won, mate_ids, enemy_ids)
        VALUES %s
        ON CONFLICT (match_id, account_id) DO NOTHING;
    """
    # Dynamic ClickHouse subquery: Only fetch rosters for matches that are in Ranked mode
    roster_filter = f"""
        start_time >= '{PATCH_START_TIME}' 
        AND match_id IN (
            SELECT DISTINCT match_id 
            FROM match_player 
            WHERE (match_mode = 'Ranked' OR match_mode = '4') 
              AND start_time >= '{PATCH_START_TIME}'
        )
    """
    roster_transform = lambda r: (
        r["match_id"],
        r["account_id"],
        r["start_time"],
        r["game_mode"],
        r["match_duration_s"],
        r["won"],
        r["mate_ids"],
        r["enemy_ids"],
    )

    batch_ingest_table(
        "player_match_roster",
        roster_select,
        roster_insert,
        roster_filter,
        roster_transform,
        start_offset=0
    )


if __name__ == "__main__":
    main()