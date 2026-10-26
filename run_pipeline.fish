#!/usr/bin/env fish

# Get the directory where this script lives
set -g SCRIPT_DIR (cd (dirname (status filename)); and pwd)

# Set PROJECT_DIR to the parent directory (or wherever your project root is)
set -g PROJECT_DIR $SCRIPT_DIR

# 1. Parse optional date argument
set -l DATE_FLAG
if test (count $argv) -ge 1
    set DATE_FLAG "--start-date" "$argv[1]"
    echo "🗓️ Overriding pipeline start date to: $argv[1]"
else
    echo "📅 No date provided. Running for default/current week..."
end

echo "Starting Deadlock Ranked Analytics Pipeline..."
echo "Project Root: $PROJECT_DIR"
echo "--------------------------------------------------"

# 1. Ingestion
echo "[1/3] Running database ingestion..."
python -u "$PROJECT_DIR/scripts/ingest_scripts/ingest_dl_comm_api_parquet.py" $DATE_FLAG
if test $status -ne 0
    echo "❌ Ingestion failed! Aborting pipeline."
    return 1
end

# 2. SQL ETL
echo "[2/3] Executing SQL ETL scripts..."
python -u "$PROJECT_DIR/scripts/sql_scripts/query_transform.py" $DATE_FLAG
if test $status -ne 0
    echo "❌ SQL ETL failed! Aborting pipeline."
    return 1
end

# 3. Visualizations
echo "[3/3] Generating Matplotlib plotting suites..."
python -u "$PROJECT_DIR/scripts/matplot_scripts/gen_visuals.py" $DATE_FLAG
if test $status -ne 0
    echo "❌ Visualization generation failed!"
    return 1
end

echo "--------------------------------------------------"
echo "✅ Pipeline completed successfully!"