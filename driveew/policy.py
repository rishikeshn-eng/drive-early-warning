"""Turn risk scores into a replacement policy priced in rupees.

All cost inputs are ASSUMPTIONS (defaults below are placeholders): replace them with your fleet's numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    replace_inr: float = 14000.0   # drive + swap labour for a proactive replacement
    failure_inr: float = 60000.0   # unplanned failure: rebuild traffic, degraded redundancy, on-call time
    salvage_inr: float = 0.0       # residual value of a healthy drive pulled early


def simulate(scored: pd.DataFrame, threshold: float, costs: Costs, horizon: int = 30):
    """Replay the policy over a scored weekly panel (columns: serial_number, day_idx, label, risk).

    A drive flagged at risk >= threshold is replaced once (and leaves the fleet). Flagged drive with
    label=1 -> failure avoided; label=0 -> false alarm. Unflagged drives with label=1 -> still fail
    (counted once per drive; repeated weekly rows of the same drive are deduplicated by first hit).
    """
    df = scored.sort_values(["serial_number", "day_idx"])
    flagged = df[df.risk >= threshold].groupby("serial_number").first()
    avoided = int(flagged.label.sum())
    false_alarms = int((flagged.label == 0).sum())
    failing = df[df.label == 1].serial_number.nunique()
    missed = failing - avoided
    # a flagged-then-false-alarm drive could still be one that fails later; ignore (conservative: it is
    # counted as false alarm at flag time)
    spend = (avoided + false_alarms) * costs.replace_inr - false_alarms * costs.salvage_inr
    saved = avoided * costs.failure_inr
    return {
        "threshold": float(threshold), "replacements": avoided + false_alarms, "avoided_failures": avoided,
        "false_alarms": false_alarms, "missed_failures": missed, "failing_drives": int(failing),
        "recall": avoided / max(1, failing),
        "precision": avoided / max(1, avoided + false_alarms),
        "net_inr": saved - spend,
        "inr_per_avoided_failure": spend / avoided if avoided else None,
    }


def best_threshold(scored: pd.DataFrame, costs: Costs, grid=None):
    grid = np.unique(np.quantile(scored.risk, np.linspace(0.5, 0.999, 60))) if grid is None else grid
    results = [simulate(scored, t, costs) for t in grid]
    return max(results, key=lambda r: r["net_inr"]), results


def per_drive(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per drive: flagged (bool), avoided (flagged & label), fails (ever label=1)."""
    df = scored.sort_values(["serial_number", "day_idx"])
    g = df.groupby("serial_number")
    first_flag = df[df.risk >= threshold].groupby("serial_number").label.first()
    out = pd.DataFrame({"fails": g.label.max().astype(bool)})
    out["flagged"] = out.index.isin(first_flag.index)
    out["avoided"] = out.index.map(first_flag).fillna(0).astype(bool) & out.flagged
    return out


def bootstrap_net(scored: pd.DataFrame, threshold: float, costs: Costs, n: int = 500, seed: int = 0):
    """95% interval of net rupees, resampling drives (the unit that fails)."""
    d = per_drive(scored, threshold)
    rng = np.random.default_rng(seed)
    avoided, flagged = d.avoided.to_numpy(), d.flagged.to_numpy()
    nets = []
    for _ in range(n):
        idx = rng.integers(0, len(d), len(d))
        a, f = avoided[idx].sum(), flagged[idx].sum()
        nets.append(a * costs.failure_inr - f * costs.replace_inr + (f - a) * costs.salvage_inr)
    return [float(np.percentile(nets, 2.5)), float(np.percentile(nets, 97.5))]


def curve(scored: pd.DataFrame, points: int = 40):
    """Threshold sweep the UI uses to re-price the policy under different rupee assumptions."""
    grid = np.unique(np.quantile(scored.risk, np.linspace(0.6, 0.9995, points)))
    rows = []
    for t in grid:
        r = simulate(scored, t, Costs())
        rows.append({"t": round(float(t), 4), "repl": r["replacements"], "avoided": r["avoided_failures"],
                     "fa": r["false_alarms"], "failing": r["failing_drives"]})
    return rows
