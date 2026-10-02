# drive-early-warning

Predict hard-drive failure 7 and 30 days out from daily SMART readings, then turn the predictions into a replacement policy priced in rupees per avoided failure.

**Live UI:** https://rishikeshn-eng.github.io/drive-early-warning/ (fleet map, risk queue, editable rupee assumptions)

> **Status: the published numbers come from a synthetic fleet, not real Backblaze data.**
> I generated 6,000 drives in Backblaze's exact column layout (`synth.py`), so the numbers below show that the pipeline works end to end and that it beats a naive rule on data with the failure patterns I put in. They say nothing about how real drives behave. The real run needs ~2 GB of Backblaze zips; see "Run on real data".

## Pipeline

1. `features.py`: DuckDB window functions build one row per drive per week: SMART 5/187/188/197/198 levels, 7- and 14-day deltas, temperature, age, model. Label = fails within H days *after* the row. Rows whose horizon runs past the end of the data are dropped (unknown label), as is the failure day itself.
2. `run.py`: temporal split with a gap of H days between train / validation / test (no label leakage). Gradient boosting (`HistGradientBoostingClassifier`, balanced classes). The alert threshold is chosen on validation, then applied once to test.
3. `policy.py`: replays the policy on the test weeks. A drive flagged at risk ≥ threshold is replaced once. Flagged and would have failed within H days = failure avoided; otherwise false alarm. Net rupees = avoided × failure cost − replacements × replacement cost. A bootstrap over drives (the unit that fails) gives a 95% interval.

## Results on the synthetic fleet (test window)

| horizon | ROC-AUC | PR-AUC | rule baseline PR-AUC* | failing drives in test | policy net at default costs (95% bootstrap) |
|---|---|---|---|---|---|
| 7 days | 0.81 | 0.31 | 0.004 | 33 | −₹5.2 lakh (−8.1 to −2.2 lakh) |
| 30 days | 0.78 | 0.41 | 0.011 | 31 | +₹4.1 lakh (+0.5 to +8.2 lakh) |

\*Rule: replace any drive whose SMART 5/187/197/198 is nonzero. It also over-flags: 4% of the synthetic healthy drives have a stable small nonzero SMART 5, as real fleets do.

Read these carefully:
- At 7 days the model loses money at the default costs (₹14,000 to replace, ₹60,000 per unplanned failure). With few days of warning and ~30% of failures showing no SMART ramp, precision is too low to pay for itself. At 30 days it breaks even-to-positive, but the interval is wide because the test window holds only ~31 failing drives.
- **Cost inputs are placeholders**, not Backblaze or Indian market figures. Edit them in `policy.Costs` or with the sliders in the UI. The UI sliders choose the best threshold on the test set itself, so they are optimistic; the table above uses the validation-chosen threshold.
- The fleet map's pod/rack layout is random: Backblaze publishes no physical locations. On real data it is a hash bucket for display only.

## Run

```bash
pip install numpy pandas scikit-learn duckdb pyarrow pytest
python -m pytest -q
python -m driveew.run            # synthetic, ~15 s, writes docs/results.json
python scripts/build_site.py     # docs/index.html
```

### Run on real data

```bash
curl -O https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q4_2024.zip   # ~1.0 GB
curl -O https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2025.zip   # ~1.0 GB
python scripts/ingest_backblaze.py data_Q4_2024.zip data_Q1_2025.zip --out data/backblaze.parquet
python -m driveew.run --parquet data/backblaze.parquet && python scripts/build_site.py
```

The full history (~354k drives since 2013) is the same code with more zips; DuckDB does the feature windows out of core, but the pandas model step expects the weekly panel to fit in memory.
