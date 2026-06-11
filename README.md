# Ghost Engine

Ghost Engine is a Python package for structural anomaly verification on segmented image-like tensors. The project is intentionally scoped as a research/engineering artifact: claims below are classified by implementation and evidence status.

## Claim discipline

### Implemented

- Boundary signatures are extracted with an active contour and compared with circular/reflection alignment.
- Baseline construction phase-aligns every baseline profile to a reference before computing `mean_profile`.
- Interior topology descriptors are included: hole count, Euler number, connected component count, skeleton length, distance-transform statistics, and Hu moments.
- `object_mode` supports `largest`, `all`, and `indexed`; `all`/`indexed` modes compare component counts in addition to signature distance.
- Fitted engine state can be saved and loaded as JSON, including `mean_profile`, `variance_cutoff`, resolution, spline points, engine parameters, channel weights, version, and training metadata.
- A CLI is available through `ghost-engine fit`, `ghost-engine verify`, and `ghost-engine benchmark`.
- `ghost_engine.benchmark` generates synthetic nuisance-variable benchmarks across brightness, blur, translation, rotation, occlusion, clutter, background noise, and mutation type, then exports CSV/JSON results, ROC/AUC metrics, plots, and a visual failure gallery.
- Benchmark runs support `max_cases` plus `first`, `random`, and `stratified` sampling so the full Cartesian grid can be bounded before active-contour evaluation.
- Benchmark ROC/AUC metrics are swept over thresholds derived from observed distance ratios in addition to the configured reporting sensitivities.
- Benchmark result rows flag high-nuisance combinations (`high_occlusion`, `dense_clutter`, `high_noise`, `heavy_blur`) where synthetic labels may be less clean.

### Experimentally supported in this repository

- Unit tests cover deterministic signature extraction, channel handling, translation tolerance, ordered profile alignment, phase-aligned baseline construction, internal-hole topology detection, multi-object mismatch detection, and model save/load.
- `run_validation.py` exercises a small synthetic ellipse-vs-rectangle regression check.

### Planned

- Validation on a public real-world anomaly dataset such as industrial inspection, medical masks, satellite segmentation, or quality-control imagery.
- Broader calibration of sensitivity defaults after real-data evaluation.
- Richer object-set matching beyond count mismatch for dense multi-object scenes.
- Public real-dataset benchmark wiring and checked-in dataset-specific evaluation recipes.

### Unsupported

- Production-readiness claims.
- Claims of real-world anomaly-detection accuracy.
- Claims that synthetic nuisance robustness transfers to medical, industrial, or satellite imagery without dataset-specific validation.

## CLI examples

```bash
ghost-engine fit --baseline data/baseline --out model.json
ghost-engine verify --model model.json --input samples/ --out audit_report.json --save-overlays
ghost-engine benchmark --config benchmark.yaml --out results/ --max-cases 512 --sampling stratified
```

## Benchmark outputs

`ghost-engine benchmark` and `python run_benchmark.py` write these files. By default, the synthetic grid is stratified down to 512 cases; set `sampling: full` in a config file to run the full Cartesian grid.

- `results.csv`
- `results.json`
- `metrics.json`
- `roc_curve.png`
- `distance_distribution.png`
- `failure_gallery/`
- `model.json`

## Real dataset validation requirement

Do not describe Ghost Engine as production-ready or empirically validated on real imagery until a public dataset evaluation is added and the resulting metrics and failures are reported.
