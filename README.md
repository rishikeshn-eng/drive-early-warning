# drive-early-warning

Predict hard-drive failure 7 and 30 days out from daily SMART readings, then turn the predictions into a replacement policy priced in rupees per avoided failure.

**Live UI:** https://rishikeshn-eng.github.io/drive-early-warning/ (fleet map, risk queue, editable rupee assumptions)

> **Status: the live UI now shows a run on real Backblaze data** (Q4 2024 + Q1 2025 quarterly zips, a deterministic 1-in-4 sample of drives by `hash(serial_number)`: 13.75M daily readings, 80,150 drives, 509 failures). The earlier synthetic run is kept as `docs/results_synthetic.json` and summarised below as pipeline validation only. The full ~354k-drive history since 2013 was not used.

## Results on real Backblaze data (test window)

Temporal split on a short dataset: train on the first 35% of dates, validate (pick the alert threshold) on 35-70% after a horizon-long gap, test on the last 30% (no val-to-test gap, which only matters for the threshold). Gradient boosting on SMART 5/187/188/197/198 levels and 7/14-day deltas, temperature, age and model.

| horizon | ROC-AUC | PR-AUC (base rate) | SMART-rule PR-AUC | failing drives in test | policy at validation threshold | net at placeholder costs (95% bootstrap) |
|---|---|---|---|---|---|---|
| 7 days | 0.865 | 0.040 (0.025%) | 0.003 | 113 | 14 avoided, 258 false alarms | −₹29.7 lakh (−35.4 to −24.3 lakh) |
| 30 days | 0.865 | 0.090 (0.10%) | 0.012 | 150 | 23 avoided, 154 false alarms | −₹11.0 lakh (−16.2 to −5.5 lakh) |

What this says:
- **The model ranks drives well and beats the naive rule by 8-14x on PR-AUC**, but PR-AUC of 0.09 on a 0.1% base rate means most flagged drives are healthy: precision is 13% at 30 days and 5% at 7 days.
- **At the placeholder costs the policy loses money**, with a bootstrap interval entirely below zero. Break-even needs precision of replace cost / failure cost = 14,000 / 60,000 = 23%. At the 30-day operating point it pays only if an unplanned failure costs more than about ₹1.08 lakh (1.8x the placeholder). Both costs are my placeholders, not Backblaze or Indian market numbers; set yours in `policy.Costs` or with the sliders.
- Recall is low (15% of failing drives caught at 30 days at the validation-chosen threshold). Many real failures show no SMART warning, as Backblaze itself reports.
- Caveats: ~6 months of data (so only ~150 failing drives to test on), a sample of the fleet, one split, and the pod/rack map is a hash bucket for display (Backblaze publishes no locations).

## Pipeline validation on a synthetic fleet

`synth.py` generates 6,000 drives in Backblaze's layout with a planted failure model, which is how the pipeline was first verified (30-day: ROC-AUC 0.78, PR-AUC 0.41, net +₹4.1 lakh). Those numbers describe the generator, and they are far rosier than the real ones above: a useful reminder of how much a hand-built simulator flatters a model.

## Pipeline

1. `features.py`: DuckDB window functions build one row per drive per week: SMART 5/187/188/197/198 levels, 7- and 14-day deltas, temperature, age, model. Label = fails within H days *after* the row. Rows whose horizon runs past the end of the data are dropped (unknown label), as is the failure day itself.
2. `run.py`: temporal split with a gap of H days between train / validation / test (no label leakage). Gradient boosting (`HistGradientBoostingClassifier`, balanced classes). The alert threshold is chosen on validation, then applied once to test.
3. `policy.py`: replays the policy on the test weeks. A drive flagged at risk ≥ threshold is replaced once. Flagged and would have failed within H days = failure avoided; otherwise false alarm. Net rupees = avoided × failure cost − replacements × replacement cost. A bootstrap over drives (the unit that fails) gives a 95% interval.

## Run

```bash
pip install numpy pandas scikit-learn duckdb pyarrow pytest
python -m pytest -q
python -m driveew.run            # synthetic, ~15 s, writes docs/results.json
python scripts/build_site.py     # docs/index.html
```

### Run on real data (what produced the numbers above)

```bash
curl -O https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q4_2024.zip   # ~1.0 GB
curl -O https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2025.zip   # ~1.0 GB
python scripts/ingest_backblaze.py data_Q4_2024.zip data_Q1_2025.zip --sample-mod 4 --out data/backblaze.parquet   # ~1 min
python -m driveew.run --parquet data/backblaze.parquet --train-frac 0.35 --val-frac 0.7 --no-val-test-gap
python scripts/build_site.py
```

Keep `--sample-mod` at 1 for the whole fleet (more RAM). The pandas model step needs the weekly panel in memory; DuckDB does the feature windows.
