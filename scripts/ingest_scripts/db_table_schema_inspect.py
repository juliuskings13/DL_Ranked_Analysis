import json
import requests
import time


base_url = f"https://api.deadlock-api.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0 (Deadlock Analytics Dev)"}
OUTPUT_FILE = "schema_results.json"


def inspect_database_struct():
	print("fetching table list from /v1/sql/tables")
	resp = requests.get(f"{base_url}/sql/tables", headers=HEADERS)

	if resp.status_code != 200:
		print(f"failed to fetch tables: {resp.status_code} - {resp.text}")
		return

	tables = resp.json()
	print(f"found {len(tables)}. fetching schemas safely under rate limits\n")

	all_schemas = {}

	for index, table in enumerate(tables, start = 1):
		print(f"[{index}/{len(tables)} fetching schema for: {table}]")

		schema_resp = requests.get(f"{base_url}/sql/tables/{table}/schema", headers=HEADERS)

		if schema_resp.status_code == 200:
			all_schemas[table] = schema_resp.json()
		elif schema_resp.status_code == 429:
			print(f"rate limited (429) on {table}! waiting 30s before retrying")
			time.sleep(60)
			schema_resp = requests.get(f"{base_url}/sql/tables/{table}/schema", headers=HEADERS)
			if schema_resp.status_code == 200:
				all_schemas[table] = schema_resp.json()
		else:
			all_schemas[table] = {"error": schema_resp.status_code}
			time.sleep(6)

	with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
		json.dump(all_schemas, f, indent = 2)

	print(f"\n done! full schemas saved to {OUTPUT_FILE}")

if __name__ == "__main__":
	inspect_database_struct()