import csv
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, shift
from skimage.draw import disk, ellipse, polygon, rectangle
from skimage.transform import rotate

from ghost_engine.core import TopologicalGhostEngine

DEFAULT_CONFIG = {
    "resolution": 64,
    "baseline_count": 12,
    "sensitivity_values": [0.8, 1.0, 1.3, 1.6, 2.0],
    "threshold_count": 101,
    "max_cases": 512,
    "sampling": "stratified",
    "random_seed": 42,
    "brightness": [0.3, 0.5, 0.8, 1.0],
    "blur_sigma": [0, 1, 2, 3],
    "translation": [0, 4, 8, 12],
    "rotation": [0, 15, 30, 45, 90],
    "occlusion": [0.0, 0.10, 0.25, 0.40],
    "clutter_objects": [0, 2, 5, 10],
    "background_noise": [0.0, 0.05, 0.15, 0.30],
    "mutation_type": ["none", "rectangle", "dented_ellipse", "notched_object", "internal_hole", "detached_fragment"],
}


def load_config(config_path):
    if config_path is None:
        return dict(DEFAULT_CONFIG)
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    config = dict(DEFAULT_CONFIG)
    config.update(loaded or {})
    return config


def base_ellipse(resolution, brightness=1.0):
    image = np.zeros((resolution, resolution), dtype=float)
    rr, cc = ellipse(resolution // 2, resolution // 2, resolution // 4, resolution // 3, shape=image.shape)
    image[rr, cc] = brightness
    return image


def mutated_shape(resolution, mutation_type, brightness):
    image = base_ellipse(resolution, brightness)
    if mutation_type == "none":
        return image
    image[:] = 0.0
    if mutation_type == "rectangle":
        start = (resolution // 4, resolution // 4)
        rr, cc = rectangle(start=start, extent=(resolution // 2, resolution // 2), shape=image.shape)
        image[rr, cc] = brightness
    elif mutation_type == "dented_ellipse":
        image = base_ellipse(resolution, brightness)
        rr, cc = disk((resolution // 2, resolution // 2 + resolution // 4), resolution // 8, shape=image.shape)
        image[rr, cc] = 0.0
    elif mutation_type == "notched_object":
        image = base_ellipse(resolution, brightness)
        rr, cc = rectangle(start=(resolution // 2 - 5, resolution // 2), extent=(10, resolution // 3), shape=image.shape)
        image[rr, cc] = 0.0
    elif mutation_type == "internal_hole":
        image = base_ellipse(resolution, brightness)
        rr, cc = disk((resolution // 2, resolution // 2), resolution // 9, shape=image.shape)
        image[rr, cc] = 0.0
    elif mutation_type == "detached_fragment":
        image = base_ellipse(resolution, brightness)
        rr, cc = disk((resolution // 5, resolution // 5), resolution // 12, shape=image.shape)
        image[rr, cc] = brightness
    else:
        raise ValueError(f"Unsupported mutation_type: {mutation_type}")
    return image


def apply_nuisances(image, blur_sigma, translation, rotation, occlusion, clutter_objects, background_noise, rng):
    result = image.copy()
    if rotation:
        result = rotate(result, rotation, resize=False, preserve_range=True, mode="constant")
    if translation:
        result = shift(result, shift=(translation, -translation), order=1, mode="constant", cval=0.0)
    if occlusion:
        side = max(1, int(np.sqrt(result.size * occlusion)))
        start = (result.shape[0] // 2 - side // 2, result.shape[1] // 2 - side // 2)
        rr, cc = rectangle(start=start, extent=(side, side), shape=result.shape)
        result[rr, cc] = 0.0
    for _ in range(int(clutter_objects)):
        row = int(rng.integers(0, result.shape[0]))
        col = int(rng.integers(0, result.shape[1]))
        radius = int(rng.integers(2, 5))
        rr, cc = disk((row, col), radius, shape=result.shape)
        result[rr, cc] = max(float(np.max(result)), 0.8)
    if background_noise:
        result = result + rng.normal(0.0, background_noise, result.shape)
    if blur_sigma:
        result = gaussian_filter(result, sigma=blur_sigma)
    return np.clip(result, 0.0, 1.0)


def _case_dict(keys, values):
    return dict(zip(keys, values))


def _sample_indices(indices, take, rng):
    if take <= 0:
        return []
    if take >= len(indices):
        return list(indices)
    return sorted(rng.choice(indices, size=take, replace=False).tolist())


def benchmark_cases(config, rng=None):
    keys = ["brightness", "blur_sigma", "translation", "rotation", "occlusion", "clutter_objects", "background_noise", "mutation_type"]
    values_by_key = [config[key] for key in keys]
    all_values = list(product(*values_by_key))
    total_cases = len(all_values)
    max_cases = config.get("max_cases")
    sampling = config.get("sampling", "full")
    rng = np.random.default_rng(config.get("random_seed", 42)) if rng is None else rng

    if max_cases is None or int(max_cases) >= total_cases or sampling == "full":
        selected_indices = list(range(total_cases))
    else:
        max_cases = max(1, int(max_cases))
        if sampling == "first":
            selected_indices = list(range(max_cases))
        elif sampling == "random":
            selected_indices = _sample_indices(list(range(total_cases)), max_cases, rng)
        elif sampling == "stratified":
            mutation_idx = keys.index("mutation_type")
            stable = [idx for idx, values in enumerate(all_values) if values[mutation_idx] == "none"]
            mutated = [idx for idx, values in enumerate(all_values) if values[mutation_idx] != "none"]
            stable_take = min(len(stable), max_cases // 2)
            mutated_take = min(len(mutated), max_cases - stable_take)
            selected_indices = _sample_indices(stable, stable_take, rng) + _sample_indices(mutated, mutated_take, rng)
            if len(selected_indices) < max_cases:
                remaining = sorted(set(range(total_cases)) - set(selected_indices))
                selected_indices.extend(_sample_indices(remaining, max_cases - len(selected_indices), rng))
            selected_indices = sorted(selected_indices)
        else:
            raise ValueError("sampling must be one of: 'full', 'first', 'random', 'stratified'.")

    for idx in selected_indices:
        yield _case_dict(keys, all_values[idx])


def confusion(records, sensitivity):
    tp = fp = tn = fn = 0
    for item in records:
        predicted = item["combined_distance"] > item["threshold_base"] * sensitivity
        actual = item["is_actual_mutation"]
        tp += int(predicted and actual)
        fp += int(predicted and not actual)
        tn += int(not predicted and not actual)
        fn += int(not predicted and actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"sensitivity": sensitivity, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "false_positive_rate": fpr, "false_negative_rate": fnr}


def sensitivity_grid(records, config):
    threshold_base = float(records[0]["threshold_base"]) if records else 0.0
    observed = [float(item["combined_distance"]) / threshold_base for item in records if threshold_base > 0]
    explicit = [float(value) for value in config.get("sensitivity_values", [])]
    if not observed:
        return sorted(set(explicit))
    count = max(2, int(config.get("threshold_count", 101)))
    low = max(0.0, min(observed) - 1e-9)
    high = max(observed) + 1e-9
    dynamic = np.linspace(low, high, count).tolist()
    return sorted(set(round(value, 12) for value in [*dynamic, *explicit]))


def auc(points):
    ordered = sorted(points, key=lambda item: (item["false_positive_rate"], item["recall"]))
    x = [item["false_positive_rate"] for item in ordered]
    y = [item["recall"] for item in ordered]
    return float(np.trapezoid(y, x)) if len(x) > 1 else 0.0


def nuisance_limitation(case):
    reasons = []
    if float(case["occlusion"]) >= 0.25:
        reasons.append("high_occlusion")
    if int(case["clutter_objects"]) >= 5:
        reasons.append("dense_clutter")
    if float(case["background_noise"]) >= 0.15:
        reasons.append("high_noise")
    if int(case["blur_sigma"]) >= 3:
        reasons.append("heavy_blur")
    return reasons


def save_failure_overlay(path, image, verdict, actual, case):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    diag = verdict.get("diagnostics", {})
    snake = np.asarray(diag.get("snake_points", []), dtype=float)
    if snake.size:
        ax.plot(snake[:, 1], snake[:, 0], color="cyan", linewidth=1)
    if "center_col" in diag and "center_row" in diag:
        ax.scatter([diag["center_col"]], [diag["center_row"]], c="yellow", s=20)
    ax.set_title(f"actual={actual} predicted={verdict['is_mutation']}\nd={verdict['combined_distance']:.3f} t={verdict['threshold_cutoff']:.3f}")
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps({"case": case, "verdict": verdict}, indent=2), encoding="utf-8")


def run_benchmark(config_path=None, out_dir="results", max_cases=None, sampling=None):
    config = load_config(config_path)
    if max_cases is not None:
        config["max_cases"] = max_cases
    if sampling is not None:
        config["sampling"] = sampling
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config.get("random_seed", 42)))
    resolution = int(config["resolution"])
    baseline = [apply_nuisances(base_ellipse(resolution, 0.85), 0, int(rng.integers(-1, 2)), 0, 0, 0, 0.01, rng) for _ in range(int(config["baseline_count"]))]
    engine = TopologicalGhostEngine(resolution=resolution)
    engine.fit_baseline(baseline, metadata={"benchmark_config": config})
    records = []
    failure_dir = out / "failure_gallery"
    for idx, case in enumerate(benchmark_cases(config, rng=rng)):
        image = mutated_shape(resolution, case["mutation_type"], case["brightness"])
        image = apply_nuisances(image, case["blur_sigma"], case["translation"], case["rotation"], case["occlusion"], case["clutter_objects"], case["background_noise"], rng)
        verdict = engine.verify(image, sensitivity=1.3)
        actual = case["mutation_type"] != "none"
        predicted = verdict["is_mutation"]
        limitation_reasons = nuisance_limitation(case)
        record = {**case, "is_actual_mutation": actual, "is_predicted_mutation": predicted, "combined_distance": verdict["combined_distance"], "threshold_base": engine.variance_cutoff, "threshold_cutoff": verdict["threshold_cutoff"], "nuisance_limitation": ";".join(limitation_reasons), "is_high_nuisance_case": bool(limitation_reasons)}
        records.append(record)
        if predicted != actual:
            save_failure_overlay(failure_dir / f"failure_{idx:06d}.png", image, verdict, actual, case)
    csv_path = out / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if not records:
            raise RuntimeError("Benchmark produced no records; check benchmark configuration.")
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (out / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    thresholds = sensitivity_grid(records, config)
    metrics = [confusion(records, float(value)) for value in thresholds]
    explicit_metrics = [confusion(records, float(value)) for value in config["sensitivity_values"]]
    metrics_json = {"auc": auc(metrics), "case_count": len(records), "threshold_count": len(thresholds), "sampling": config.get("sampling"), "max_cases": config.get("max_cases"), "explicit_operating_points": explicit_metrics, "operating_points": metrics}
    (out / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    plt.figure()
    plt.plot([item["false_positive_rate"] for item in metrics], [item["recall"] for item in metrics], marker="o")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate / recall")
    plt.title(f"ROC AUC={metrics_json['auc']:.3f}")
    plt.savefig(out / "roc_curve.png", bbox_inches="tight")
    plt.close()
    plt.figure()
    stable = [item["combined_distance"] for item in records if not item["is_actual_mutation"]]
    mutated = [item["combined_distance"] for item in records if item["is_actual_mutation"]]
    plt.hist(stable, bins=30, alpha=0.6, label="stable")
    plt.hist(mutated, bins=30, alpha=0.6, label="mutation")
    plt.xlabel("Combined distance")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(out / "distance_distribution.png", bbox_inches="tight")
    plt.close()
    engine.save(out / "model.json")
    return {"results_csv": str(csv_path), "metrics_json": str(out / "metrics.json"), "failure_gallery": str(failure_dir)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--out", default="results")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--sampling", choices=["full", "first", "random", "stratified"])
    args = parser.parse_args()
    run_benchmark(args.config, args.out, max_cases=args.max_cases, sampling=args.sampling)
