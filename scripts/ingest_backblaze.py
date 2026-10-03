"""Convert Backblaze quarterly zips into one slim parquet in the schema driveew expects.

    # ~1 GB per quarter, from https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
    curl -O https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q4_2024.zip
    curl -O https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2025.zip
    python scripts/ingest_backblaze.py data_Q4_2024.zip data_Q1_2025.zip --out data/backblaze.parquet \
        --models "ST12000NM0008,HGST HUH721212ALN604"    # optional: restrict to a few models
    python -m driveew.run --parquet data/backblaze.parquet

Notes: older quarters lack some SMART columns; this script keeps only the columns it needs and
casts missing ones to 0. Run on >= 2 quarters so there is a future to hold out.
"""
import argparse, shutil, tempfile, zipfile
from pathlib import Path

import duckdb

NEEDED = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw",
          "smart_9_raw", "smart_194_raw"]

ap = argparse.ArgumentParser()
ap.add_argument("zips", nargs="+")
ap.add_argument("--out", default="data/backblaze.parquet")
ap.add_argument("--models", help="comma-separated model whitelist")
ap.add_argument("--sample-mod", type=int, default=1, help="keep drives with hash(serial_number) %% N == 0 (whole drive histories, so prevalence is preserved)")
a = ap.parse_args()

tmp = Path(tempfile.mkdtemp())
for z in a.zips:
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp)
con = duckdb.connect()
cols = ", ".join(f"coalesce(try_cast({c} AS DOUBLE), 0) AS {c}" for c in NEEDED)
conds = []
if a.models:
    conds.append("model IN (" + ",".join("'" + m.strip().replace("'", "''") + "'" for m in a.models.split(",")) + ")")
if a.sample_mod > 1:
    conds.append(f"hash(serial_number) % {a.sample_mod} = 0")
where = ("WHERE " + " AND ".join(conds)) if conds else ""
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
con.execute(f"""
COPY (SELECT CAST(date AS DATE) AS date, serial_number, model, CAST(capacity_bytes AS BIGINT) AS capacity_bytes,
             CAST(failure AS INT) AS failure, {cols}
      FROM read_csv('{tmp}/**/*.csv', union_by_name=true, header=true, ignore_errors=true, all_varchar=true)
      {where}) TO '{a.out}' (FORMAT parquet)""")
n = con.execute(f"SELECT count(*), count(DISTINCT serial_number), sum(failure) FROM '{a.out}'").fetchone()
print("rows, drives, failures:", n)
shutil.rmtree(tmp, ignore_errors=True)  # the extracted CSVs are ~20 GB for two quarters
