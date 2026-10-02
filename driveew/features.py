"""Feature + label construction in DuckDB (SQL window functions, so it scales to the real dataset)."""
from __future__ import annotations

import duckdb
import pandas as pd

SMART = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]
FEATURES = (SMART + [f"{c}_d7" for c in SMART] + [f"{c}_d14" for c in SMART]
            + ["power_on_days", "temp", "temp_d7", "model_code", "capacity_tb"])


def build(df: pd.DataFrame, horizon: int = 30, stride: int = 7) -> pd.DataFrame:
    """One row per drive every `stride` days; label = fails within `horizon` days after the row.

    Rows whose horizon runs past the end of the data are dropped (unknown label), and the failure
    day's own row is dropped (nothing left to predict).
    """
    con = duckdb.connect()
    con.register("raw", df)
    lag_cols = ",\n".join(
        f"{c} - lag({c}, 7) OVER w AS {c}_d7, {c} - lag({c}, 14) OVER w AS {c}_d14" for c in SMART)
    sql = f"""
    WITH b AS (
      SELECT *, {lag_cols},
             smart_194_raw - lag(smart_194_raw, 7) OVER w AS temp_d7,
             max(failure) OVER (PARTITION BY serial_number ORDER BY date
                                ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING) AS label,
             dense_rank() OVER (ORDER BY model) - 1 AS model_code,
             date_diff('day', min(date) OVER (), date) AS day_idx,
             max(date) OVER () AS last_date
      FROM raw WINDOW w AS (PARTITION BY serial_number ORDER BY date)
    )
    SELECT date, serial_number, model, pod, rack, coalesce(label, 0) AS label, day_idx,
           smart_9_raw / 24.0 AS power_on_days, smart_194_raw AS temp, capacity_bytes / 1e12 AS capacity_tb,
           model_code, {", ".join(SMART)},
           {", ".join(f"{c}_d7, {c}_d14" for c in SMART)}, temp_d7
    FROM b
    WHERE failure = 0 AND date <= last_date - INTERVAL {horizon} DAY AND day_idx % {stride} = 0
          AND {SMART[0]}_d14 IS NOT NULL
    """
    return con.execute(sql).df().fillna(0)
