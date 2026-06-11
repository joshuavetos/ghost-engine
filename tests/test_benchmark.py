import pytest

from ghost_engine.benchmark import DEFAULT_CONFIG, benchmark_cases, confusion, sensitivity_grid
from ghost_engine.cli import build_parser


def _small_config(**overrides):
    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "brightness": [1.0],
            "blur_sigma": [0],
            "translation": [0],
            "rotation": [0],
            "occlusion": [0.0],
            "clutter_objects": [0],
            "background_noise": [0.0],
            "mutation_type": ["none", "rectangle", "internal_hole"],
            "random_seed": 7,
        }
    )
    config.update(overrides)
    return config


def test_benchmark_cases_respects_max_cases_with_stratified_sampling():
    config = _small_config(max_cases=2, sampling="stratified")

    cases = list(benchmark_cases(config))

    assert len(cases) == 2
    assert any(case["mutation_type"] == "none" for case in cases)
    assert any(case["mutation_type"] != "none" for case in cases)


def test_default_benchmark_grid_is_bounded_before_active_contours():
    cases = list(benchmark_cases(DEFAULT_CONFIG))

    assert len(cases) == DEFAULT_CONFIG["max_cases"]
    assert any(case["mutation_type"] == "none" for case in cases)
    assert any(case["mutation_type"] != "none" for case in cases)


def test_benchmark_cases_rejects_unknown_sampling_mode():
    config = _small_config(max_cases=2, sampling="unsupported")

    with pytest.raises(ValueError, match="sampling must be one of"):
        list(benchmark_cases(config))


def test_sensitivity_grid_uses_observed_distances_not_only_explicit_values():
    records = [
        {"combined_distance": 0.5, "threshold_base": 1.0, "is_actual_mutation": False},
        {"combined_distance": 1.0, "threshold_base": 1.0, "is_actual_mutation": True},
        {"combined_distance": 2.0, "threshold_base": 1.0, "is_actual_mutation": True},
    ]
    config = _small_config(sensitivity_values=[1.3], threshold_count=11)

    thresholds = sensitivity_grid(records, config)

    assert 1.3 in thresholds
    assert len(thresholds) > len(config["sensitivity_values"])
    assert min(thresholds) <= 0.5
    assert max(thresholds) >= 2.0


def test_confusion_sweeps_dynamic_thresholds():
    records = [
        {"combined_distance": 0.5, "threshold_base": 1.0, "is_actual_mutation": False},
        {"combined_distance": 1.5, "threshold_base": 1.0, "is_actual_mutation": True},
    ]

    permissive = confusion(records, 0.4)
    strict = confusion(records, 2.0)

    assert permissive["tp"] == 1
    assert permissive["fp"] == 1
    assert strict["fn"] == 1
    assert strict["tn"] == 1


def test_cli_benchmark_accepts_sampling_controls():
    parser = build_parser()

    args = parser.parse_args(["benchmark", "--out", "results", "--max-cases", "32", "--sampling", "random"])

    assert args.max_cases == 32
    assert args.sampling == "random"
