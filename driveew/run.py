"""End-to-end: build panel -> temporal split -> model -> policy -> site payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import features, synth
from .policy import Costs, best_threshold, bootstrap_net, curve, simulate


def fit_eval(df: pd.DataFrame, horizon: int, costs: Costs, seed: int = 0, train_frac: float = 0.5,
             val_frac: float = 0.7, val_test_gap: bool = True):
    panel = features.build(df, horizon=horizon)
    cut_train = panel.day_idx.quantile(train_frac)
    cut_val = panel.day_idx.quantile(val_frac)
    tr = panel[panel.day_idx <= cut_train]
    # A horizon-long gap after training keeps the model's labels from overlapping what it is scored on. The
    # val->test gap only matters for the alert threshold, so short real datasets can switch it off.
    va = panel[(panel.day_idx > cut_train + horizon) & (panel.day_idx <= cut_val)]
    te = panel[panel.day_idx > cut_val + (horizon if val_test_gap else 0)]
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200,
                                         class_weight="balanced", random_state=seed)
    clf.fit(tr[features.FEATURES], tr.label)
    va = va.assign(risk=clf.predict_proba(va[features.FEATURES])[:, 1])
    te = te.assign(risk=clf.predict_proba(te[features.FEATURES])[:, 1])
    # baseline: "any SMART 5/187/197/198 > 0" rule
    rule = (te[["smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw"]].sum(axis=1) > 0).astype(float)
    best, _ = best_threshold(va, costs)  # threshold chosen on validation, applied to test
    test_policy = simulate(te, best["threshold"], costs)
    rule_policy = simulate(te.assign(risk=rule), 0.5, costs)
    metrics = {
        "horizon_days": horizon, "train_rows": len(tr), "test_rows": len(te),
        "test_positive_rate": float(te.label.mean()),
        "roc_auc": float(roc_auc_score(te.label, te.risk)),
        "pr_auc": float(average_precision_score(te.label, te.risk)),
        "rule_baseline_pr_auc": float(average_precision_score(te.label, rule)),
        "policy_test": test_policy, "rule_policy_test": rule_policy,
        "policy_net_ci95": bootstrap_net(te, best["threshold"], costs),
        "curve": curve(te),
        "no_policy_test_failing_drives": test_policy["failing_drives"],
        "no_policy_cost_inr": test_policy["failing_drives"] * costs.failure_inr,
    }
    return clf, te, metrics


def site_payload(te: pd.DataFrame, metrics_by_h: dict, costs: Costs, synthetic: bool) -> dict:
    last = te[te.day_idx == te.day_idx.max()]
    queue = last.sort_values("risk", ascending=False).head(25)
    cols = ["serial_number", "model", "pod", "rack", "risk", "smart_5_raw", "smart_187_raw",
            "smart_197_raw", "smart_198_raw", "power_on_days"]
    cut = last.risk.quantile(0.98)  # fleet-wide top-2% line, not per rack
    pods = (last.assign(is_high=last.risk >= cut).groupby(["pod", "rack"])
            .agg(drives=("serial_number", "count"), risk=("risk", "mean"), high=("is_high", "sum"))
            .reset_index())
    return {"synthetic": synthetic, "costs": costs.__dict__, "metrics": metrics_by_h,
            "queue": json.loads(queue[cols].round(3).to_json(orient="records")),
            "map": json.loads(pods.round(4).to_json(orient="records")),
            "as_of_day": int(last.day_idx.iloc[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", help="Backblaze-schema parquet (real data). Omit to use synthetic.")
    ap.add_argument("--drives", type=int, default=6000)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.7)
    ap.add_argument("--no-val-test-gap", action="store_true", help="for short real datasets (see README)")
    ap.add_argument("--out", default="docs/results.json")
    a = ap.parse_args()
    costs = Costs()
    if a.parquet:
        df = pd.read_parquet(a.parquet)
        if "pod" not in df:  # real data has no physical layout: bucket by serial hash for the map only
            h = pd.util.hash_array(df.serial_number.to_numpy())
            df["pod"] = (h % 12).astype(int)
            df["rack"] = ((h // 12) % 10).astype(int)  # independent of pod (h % 12 and h % 10 would share factors)
        synthetic = False
    else:
        df = synth.generate(a.drives)
        synthetic = True
    by_h, te_main = {}, None
    for h in (7, 30):
        _, te, m = fit_eval(df, h, costs, train_frac=a.train_frac, val_frac=a.val_frac, val_test_gap=not a.no_val_test_gap)
        by_h[str(h)] = m
        if h == 30:
            te_main = te
        print(f"H={h}: AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} (rule {m['rule_baseline_pr_auc']:.3f}) "
              f"policy net=₹{m['policy_test']['net_inr']:,.0f} recall={m['policy_test']['recall']:.2f}")
    Path(a.out).write_text(json.dumps(site_payload(te_main, by_h, costs, synthetic)))


if __name__ == "__main__":
    main()
