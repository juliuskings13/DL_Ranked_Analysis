import pandas as pd 
from sqlalchemy import create_engine
from db_config import DB_URI

def patch_missing_data():

	engine = create_engine(DB_URI)

	print("scanning database for missing match ids")
	missing_ids_df = pd.read_sql("""
		select distinct match_id from match_player
		except
		select distinct match_id from player_match_roster;
		""", engine)

	missing_set = set(missing_ids_df['match_id'])
	print(f"found {len(missing_set)} matches missing rosters")

	if not missing_set:
		print("no missing records found. database fully synced")
		return

	print("loading parquet data archive")
	df = pd.read_parquet('...')

	print("filtering parquet rows")

	missing_rows_df = df[df['match_id'].isin(missing_set)]

	if not missing_rows_df.empty:
		print(f"appending {len(missing_rows_df)} recovered records to player_match_roster")
		missing_rows_df.to_sql('player_match_roster', engine, if_exists='append', index = False)
		print("patch complete!")
	else:
		print("warning, missing id's not found in provided parquet files"
			"may need to target REST API for specific matches")

if __name__ == "__main__":
	print("this is just for practice im not running this shit cuz i dont feel like downloading the entire player roster parquet to my pc")