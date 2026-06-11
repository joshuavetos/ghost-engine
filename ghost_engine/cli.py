import argparse
import json
from pathlib import Path

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize

from .core import TopologicalGhostEngine
from .benchmark import run_benchmark


def _load_image(path, resolution):
    image = imread(path)
    if image.dtype.kind in {"u", "i"}:
        image = image.astype(float) / np.iinfo(image.dtype).max
    else:
        image = image.astype(float)
    if image.shape[0] != resolution or image.shape[1] != resolution:
        image = resize(image, (resolution, resolution), preserve_range=True, anti_aliasing=True)
    return np.clip(image, 0.0, 1.0)


def _image_paths(path):
    source = Path(path)
    if source.is_file():
        return [source]
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
    return sorted(item for item in source.iterdir() if item.suffix.lower() in extensions)


def _load_inputs(path, resolution):
    images = []
    for item in _image_paths(path):
        image = np.load(item) if item.suffix.lower() == ".npy" else _load_image(item, resolution)
        images.append((item, image))
    if not images:
        raise ValueError(f"No supported images found at {path}.")
    return images


def fit_command(args):
    engine = TopologicalGhostEngine(resolution=args.resolution, object_mode=args.object_mode)
    baseline = [image for _, image in _load_inputs(args.baseline, args.resolution)]
    engine.fit_baseline(baseline, percentile=args.percentile, metadata={"baseline": str(args.baseline)})
    engine.save(args.out)


def _save_verify_overlay(path, image, verdict, overlay_dir):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    diag = verdict.get("diagnostics", {})
    snake = np.asarray(diag.get("snake_points", []), dtype=float)
    if snake.size:
        ax.plot(snake[:, 1], snake[:, 0], color="cyan", linewidth=1)
    if "center_col" in diag and "center_row" in diag:
        ax.scatter([diag["center_col"]], [diag["center_row"]], c="yellow", s=20)
    ax.set_title(f"mutation={verdict['is_mutation']} d={verdict['combined_distance']:.3f} t={verdict['threshold_cutoff']:.3f}")
    ax.axis("off")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(overlay_dir / f"{path.stem}_overlay.png", bbox_inches="tight")
    plt.close(fig)


def verify_command(args):
    engine = TopologicalGhostEngine.load(args.model)
    results = {}
    out = Path(args.out)
    overlay_dir = out.parent / f"{out.stem}_overlays"
    for path, image in _load_inputs(args.input, engine.resolution):
        verdict = engine.verify(image, sensitivity=args.sensitivity)
        results[str(path)] = verdict
        if args.save_overlays:
            _save_verify_overlay(path, image, verdict, overlay_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


def benchmark_command(args):
    run_benchmark(config_path=args.config, out_dir=args.out, max_cases=args.max_cases, sampling=args.sampling)


def build_parser():
    parser = argparse.ArgumentParser(prog="ghost-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--baseline", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--resolution", type=int, default=64)
    fit.add_argument("--percentile", type=float, default=95)
    fit.add_argument("--object-mode", choices=["largest", "all", "indexed"], default="largest")
    fit.set_defaults(func=fit_command)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--model", required=True)
    verify.add_argument("--input", required=True)
    verify.add_argument("--out", required=True)
    verify.add_argument("--sensitivity", type=float, default=1.3)
    verify.add_argument("--save-overlays", action="store_true", help="Save diagnostic overlays next to the audit report.")
    verify.set_defaults(func=verify_command)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--config")
    benchmark.add_argument("--out", required=True)
    benchmark.add_argument("--max-cases", type=int, help="Limit synthetic benchmark cases after sampling.")
    benchmark.add_argument("--sampling", choices=["full", "first", "random", "stratified"], help="Case selection strategy when max_cases is below the full Cartesian grid.")
    benchmark.set_defaults(func=benchmark_command)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
