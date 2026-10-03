import numpy as np
import pandas as pd

from driveew import features, policy, synth
from driveew.policy import Costs


def small():
    return synth.generate(n_drives=400, days=240, seed=3)


def test_synth_schema_and_failures_are_last_row():
    df = small()
    assert {"date", "serial_number", "model", "failure", "smart_197_raw"} <= set(df.columns)
    f = df[df.failure == 1]
    assert len(f) > 0
    last = df.groupby("serial_number").date.max()
    # a failed drive disappears the day it fails
    assert (f.set_index("serial_number").date == last[f.serial_number]).all()


def test_labels_look_forward_only_and_drop_incomplete_horizon():
    df = small()
    p = features.build(df, horizon=30)
    assert p.date.max() <= df.date.max() - pd.Timedelta(days=30)
    pos = p[p.label == 1]
    fail_day = df[df.failure == 1].set_index("serial_number").date
    gap = (fail_day[pos.serial_number].values - pos.date.values) / np.timedelta64(1, "D")
    assert ((gap >= 1) & (gap <= 30)).all()  # positives are 1-30 days before the failure


def test_policy_accounting():
    scored = pd.DataFrame({
        "serial_number": ["a", "b", "c", "d"], "day_idx": [0, 0, 0, 0],
        "label": [1, 0, 1, 0], "risk": [0.9, 0.8, 0.1, 0.2]})
    c = Costs(replace_inr=10, failure_inr=100)
    r = policy.simulate(scored, 0.5, c)
    assert (r["avoided_failures"], r["false_alarms"], r["missed_failures"]) == (1, 1, 1)
    assert r["net_inr"] == 100 - 20 and r["inr_per_avoided_failure"] == 20


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(0)
    n = 500
    label = (rng.random(n) < 0.1).astype(int)
    scored = pd.DataFrame({"serial_number": [str(i) for i in range(n)], "day_idx": 0, "label": label,
                           "risk": label * 0.5 + rng.random(n) * 0.6})
    c = Costs(replace_inr=10, failure_inr=100)
    lo, hi = policy.bootstrap_net(scored, 0.6, c)
    point = policy.simulate(scored, 0.6, c)["net_inr"]
    assert lo <= point <= hi


def test_split_fractions_and_gap_options_keep_short_datasets_usable():
    from driveew import run
    df = synth.generate(n_drives=500, days=200, seed=5)
    _, te_gap, _ = run.fit_eval(df, 30, Costs(), train_frac=0.35, val_frac=0.7, val_test_gap=True)
    _, te_nogap, _ = run.fit_eval(df, 30, Costs(), train_frac=0.35, val_frac=0.7, val_test_gap=False)
    assert len(te_nogap) > len(te_gap) > 0
