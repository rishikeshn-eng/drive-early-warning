"""Schema-faithful synthetic fleet in Backblaze's daily-snapshot layout.

NOT real data. Failures follow a per-model hazard that rises with age; ~70% of failing
drives show a SMART 5/187/197/198 ramp before dying and ~30% fail with no warning
(Backblaze's own write-ups report that a sizeable minority of failures show none of
these attributes). Any accuracy measured on this data reflects this generator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODELS = {  # name: (capacity_tb, annualised failure rate)
    "ST12000NM0008": (12, 0.012),
    "HGST HUH721212ALN604": (12, 0.006),
    "ST4000DM000": (4, 0.035),
    "TOSHIBA MG07ACA14TA": (14, 0.010),
    "WDC WUH721816ALE6L4": (16, 0.008),
}
SMART_COLS = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw",
              "smart_9_raw", "smart_194_raw"]


def generate(n_drives: int = 6000, days: int = 365, seed: int = 7, start="2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(MODELS)
    model_idx = rng.choice(len(names), n_drives, p=[0.3, 0.25, 0.1, 0.2, 0.15])
    afr = np.array([MODELS[names[m]][1] for m in model_idx])
    age0 = rng.integers(0, 1500, n_drives)  # age in days at day 0
    install = np.where(rng.random(n_drives) < 0.25, rng.integers(0, days - 60, n_drives), 0)
    age_mult = 1 + (age0 > 900) * 1.5
    p_fail = 1 - np.exp(-afr * age_mult * (days - install) / 365)
    fails = rng.random(n_drives) < p_fail
    fail_day = np.where(fails, install + 30 + (rng.random(n_drives) * (days - install - 31)).astype(int), -1)
    signal = fails & (rng.random(n_drives) < 0.7)
    lead = np.clip(rng.exponential(25, n_drives).astype(int) + 4, 4, 90)
    noisy_healthy = (~fails) & (rng.random(n_drives) < 0.04)  # stable small nonzero counts
    pod = rng.integers(0, 12, n_drives)
    rack = rng.integers(0, 10, n_drives)
    base_temp = rng.normal(31, 3, n_drives)

    frames = []
    d0 = pd.Timestamp(start)
    for i in range(n_drives):
        last = fail_day[i] if fails[i] else days - 1
        t = np.arange(install[i], last + 1)
        n = len(t)
        cols = {c: np.zeros(n) for c in SMART_COLS[:5]}
        if signal[i]:
            ramp = np.clip((t - (fail_day[i] - lead[i])) / lead[i], 0, None)
            for c, scale in zip(SMART_COLS[:4], (60, 30, 4, 40)):
                if rng.random() < 0.8:
                    cols[c] = np.floor(ramp ** 1.5 * scale * rng.uniform(0.3, 1.5)
                                       * np.cumsum(rng.random(n) * 0.2 + 0.9) / n * 1.2)
            cols["smart_198_raw"] = np.floor(cols["smart_197_raw"] * rng.uniform(0.3, 1.0))
        elif noisy_healthy[i]:
            cols["smart_5_raw"] = np.full(n, rng.integers(1, 4), dtype=float)
        df = pd.DataFrame(cols)
        df["smart_9_raw"] = (age0[i] + t) * 24.0
        df["smart_194_raw"] = np.round(base_temp[i] + rng.normal(0, 1.2, n) + (3 * (cols["smart_197_raw"] > 0)))
        df["date"] = d0 + pd.to_timedelta(t, unit="D")
        df["serial_number"] = f"SYN{i:06d}"
        df["model"] = names[model_idx[i]]
        df["capacity_bytes"] = MODELS[names[model_idx[i]]][0] * 10 ** 12
        df["failure"] = 0
        if fails[i]:
            df.loc[df.index[-1], "failure"] = 1
        df["pod"] = pod[i]
        df["rack"] = rack[i]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "serial_number", "model", "capacity_bytes", "failure", *SMART_COLS, "pod", "rack"]]
