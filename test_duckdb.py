import sys; sys.path.insert(0, "scripts/ingest_scripts")
from ingest_dl_comm_api_parquet import parquet_urls, duckdb_source
from pipeline_common import week_bounds
s, e = (d.replace(tzinfo=None) for d in week_bounds(12))
con = duckdb_source(parquet_urls("match_player"))
print(con.execute("DESCRIBE SELECT start_time FROM match_player").fetchall())
plan = con.execute("EXPLAIN SELECT count(*) FROM match_player WHERE start_time >= ? AND start_time < ?", [s, e]).fetchall()[0][1]
print(plan)