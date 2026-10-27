import json
from psycopg2.extras import execute_values
import psycopg2
import requests
from db_config import PG_CONFIG                  
from db_config import PG_CONFIG as DB_Params     
from db_config import DB_URI   


#Database config
base_api_url = "https://assets.deadlock-api.com/v2"

# Schema setup

create_tables_sql = """

create table if not exists heroes (
	hero_id int primary key,
	name varchar(100) not null,
	class_name varchar(100),
	hero_type varchar(50),
	complexity int
);
"""

def setup_database(conn):
	""" ensure destination tables exist before ingesting data"""
	with conn.cursor() as cursor:
		cursor.execute(create_tables_sql)
	conn.commit()
	print("Schema verified/created successfully")


# Sync hero table

def sync_heroes(conn):
	"fetch hero dictionary from API and insert into postgres"
	print("syncing hero metadata...")

	try:
		res = requests.get(f"{base_api_url}/heroes", timeout = 10)
		res.raise_for_status()
		heroes_data = res.json()
	except requests.RequestException as e:
		print(f"failed to fetch hero assets: {e}")
		return

	hero_rows = []
	for h in heroes_data:
		hero_id = h.get("id");

		if hero_id is not None:
			class_name = h.get("class_name")
			name = h.get("name") or f"Hero_{hero_id}"
			hero_type = h.get("hero_type")
			complexity = h.get("complexity")

			hero_rows.append(
				(hero_id, class_name, name, hero_type, complexity)
				)

	if hero_rows:
		with conn.cursor() as cursor:
			execute_values(
				cursor,
				"""
				insert into heroes (hero_id, class_name, name, hero_type, complexity)
				values %s
				on conflict (hero_id) do update set
				class_name = excluded.class_name,
				name = excluded.name,
				hero_type = excluded.hero_type,
				complexity = excluded.complexity;
				""",
				hero_rows
				)
		conn.commit()
		print(f"successfully updated {len(hero_rows)} heroes in database")

if __name__ == "__main__":
	try:
		connection = psycopg2.connect(**DB_Params)
		setup_database(connection)
		sync_heroes(connection)
		connection.close()
		print("hero sync complete!")
	except Exception as e:
		print(f"database connection failed {e}")