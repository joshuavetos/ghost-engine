import sys
from pathlib import Path

import numpy as np
import pytest
from skimage.draw import ellipse, rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import validate_input_tensor
from ghost_engine.core import TopologicalGhostEngine


def _ellipse_image(row=32, col=32, r_radius=16, c_radius=16, brightness=1.0, noise_level=0.0):
    img = np.random.normal(0, noise_level, (64, 64))
    rr, cc = ellipse(row, col, r_radius, c_radius, shape=(64, 64))
    img[rr, cc] = brightness
    return np.clip(img, 0, 1)


def _rectangle_image(brightness=1.0):
    img = np.zeros((64, 64))
    rr, cc = rectangle(start=(16, 16), extent=(32, 32), shape=(64, 64))
    img[rr, cc] = brightness
    return img


def test_config_normalization_guardrail():
    """Ensures input tensors outside [0, 1] boundaries trip a ValueError."""
    invalid_tensor = np.array([[0.0, 1.5], [0.0, 0.0]])
    with pytest.raises(ValueError, match="FAIL-CLOSED: Input data breaks normalization bounds"):
        validate_input_tensor(invalid_tensor)


def test_config_accepts_channel_last_images():
    """Ensures normalized channel-last image tensors are accepted."""
    valid_tensor = np.zeros((64, 64, 3))
    assert validate_input_tensor(valid_tensor) is True


def test_config_rejects_non_image_rank():
    """Ensures non-image-shaped tensors trip a ValueError."""
    invalid_tensor = np.zeros((64,))
    with pytest.raises(ValueError, match="FAIL-CLOSED: Expected a 2D matrix or channel-last 3D image tensor"):
        validate_input_tensor(invalid_tensor)


def test_engine_uninitialized_state():
    """Ensures verify() raises RuntimeError if called before fit_baseline()."""
    engine = TopologicalGhostEngine(resolution=64)
    sample = np.zeros((64, 64))
    with pytest.raises(RuntimeError, match="Engine state uninitialized"):
        engine.verify(sample)


def test_engine_insufficient_baseline():
    """Ensures fit_baseline() rejects populations with fewer than 10 samples."""
    engine = TopologicalGhostEngine(resolution=64)
    insufficient_baseline = [np.zeros((64, 64)) for _ in range(5)]
    with pytest.raises(ValueError, match="Baseline dataset is too small"):
        engine.fit_baseline(insufficient_baseline)


def test_deterministic_signature_extraction():
    """Verifies that extract_signature yields identical distance profiles for identical inputs."""
    engine = TopologicalGhostEngine(resolution=64)
    np.random.seed(42)
    sample = np.random.uniform(0, 1, (64, 64))

    profile_1 = engine.extract_signature(sample)
    profile_2 = engine.extract_signature(sample)

    assert np.array_equal(profile_1, profile_2), "Signature extraction must be purely deterministic."


def test_multichannel_signature_matches_grayscale_intensity():
    """Verifies RGB inputs are reduced to luminance-equivalent grayscale geometry."""
    engine = TopologicalGhostEngine(resolution=64)
    grayscale = _ellipse_image(brightness=1.0)
    rgb = np.repeat(grayscale[:, :, np.newaxis], 3, axis=2)

    grayscale_profile = engine.extract_signature(grayscale)
    rgb_profile = engine.extract_signature(rgb)

    assert np.allclose(grayscale_profile, rgb_profile)


def test_ordered_distance_rejects_unordered_wasserstein_blind_spot():
    """Ensures equal radius histograms with different angular order are not treated as identical."""
    engine = TopologicalGhostEngine(resolution=64)
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    permuted = np.array([1.0, 3.0, 2.0, 4.0])

    assert engine._profile_distance(permuted, reference) > 0.0


def test_translation_uses_estimated_object_centroid():
    """Ensures object-centroid estimation reduces sensitivity to off-center placement."""
    engine = TopologicalGhostEngine(resolution=64)
    centered_profile, centered_diag = engine.extract_signature(_ellipse_image(row=32, col=32), return_diagnostics=True)
    shifted_profile, shifted_diag = engine.extract_signature(_ellipse_image(row=36, col=28), return_diagnostics=True)

    assert centered_diag["center_row"] == pytest.approx(32.0, abs=0.75)
    assert centered_diag["center_col"] == pytest.approx(32.0, abs=0.75)
    assert shifted_diag["center_row"] == pytest.approx(36.0, abs=0.75)
    assert shifted_diag["center_col"] == pytest.approx(28.0, abs=0.75)
    assert engine._profile_distance(shifted_profile, centered_profile) < 1.0


def test_brightness_variance_ignored_but_structural_mutation_detected():
    """Covers the central promise that brightness variance is stable while shape mutation is flagged."""
    np.random.seed(42)
    engine = TopologicalGhostEngine(resolution=64)
    baseline = [
        _ellipse_image(
            r_radius=np.random.randint(15, 18),
            c_radius=np.random.randint(15, 18),
            brightness=np.random.uniform(0.7, 1.0),
        )
        for _ in range(30)
    ]
    engine.fit_baseline(baseline)

    dim_same_shape = _ellipse_image(r_radius=16, c_radius=16, brightness=0.45)
    rectangle_mutation = _rectangle_image(brightness=0.8)

    stable_verdict = engine.verify(dim_same_shape)
    mutation_verdict = engine.verify(rectangle_mutation)

    assert stable_verdict["is_mutation"] is False
    assert mutation_verdict["is_mutation"] is True


def test_non_rgb_channels_require_explicit_weights():
    """Ensures arbitrary channel stacks fail closed unless channel semantics are supplied."""
    engine = TopologicalGhostEngine(resolution=64)
    hyperspectral_like = np.zeros((64, 64, 5))

    with pytest.raises(ValueError, match="Non-RGB channel-last tensors require explicit channel_weights"):
        engine.extract_signature(hyperspectral_like)


def test_weighted_non_rgb_channels_match_selected_structural_band():
    """Ensures explicit channel weights are used instead of blind channel averaging."""
    grayscale = _ellipse_image(brightness=1.0)
    multiband = np.zeros((64, 64, 5))
    multiband[:, :, 2] = grayscale

    grayscale_engine = TopologicalGhostEngine(resolution=64)
    weighted_engine = TopologicalGhostEngine(resolution=64, channel_weights=[0, 0, 1, 0, 0])

    grayscale_profile = grayscale_engine.extract_signature(grayscale)
    weighted_profile = weighted_engine.extract_signature(multiband)

    assert np.allclose(weighted_profile, grayscale_profile)


def test_segmentation_uses_largest_component_to_ignore_small_clutter():
    """Ensures center estimation is not captured by small bright nuisance objects."""
    engine = TopologicalGhostEngine(resolution=64)
    cluttered = _ellipse_image(row=34, col=29, r_radius=13, c_radius=17, brightness=0.85)
    cluttered[2:5, 54:58] = 1.0
    cluttered[57:60, 4:7] = 1.0

    _, diagnostics = engine.extract_signature(cluttered, return_diagnostics=True)

    assert diagnostics["center_row"] == pytest.approx(34.0, abs=1.25)
    assert diagnostics["center_col"] == pytest.approx(29.0, abs=1.25)
    assert diagnostics["foreground_area"] > 500


def test_fft_profile_distance_matches_exhaustive_ordered_alignment():
    """Ensures optimized circular alignment is numerically equivalent to exhaustive shifts."""
    engine = TopologicalGhostEngine(resolution=64)
    reference = np.array([2.0, 1.0, 4.0, 5.0, 3.0, 8.0])
    candidate = np.array([3.0, 8.0, 2.0, 1.0, 4.0, 5.0])

    exhaustive = min(
        np.sqrt(np.mean((np.roll(option, shift) - reference) ** 2))
        for option in (candidate, candidate[::-1])
        for shift in range(candidate.size)
    )

    assert engine._profile_distance(candidate, reference) == pytest.approx(exhaustive)


def test_baseline_profiles_are_phase_aligned_before_mean_profile():
    """Ensures asymmetric references are not smeared by raw circular averaging."""
    engine = TopologicalGhostEngine(resolution=64)
    reference = np.array([9.0, 1.0, 2.0, 1.0, 3.0, 1.0])
    profiles = [np.roll(reference, shift) for shift in [0, 1, 2, 3, 4, 5, 0, 2, 4, 1]]
    iterator = iter(profiles)
    engine.extract_signature = lambda _image: next(iterator)
    engine.extract_topology_descriptors = lambda _image: {
        "hole_count": 0.0,
        "euler_number": 1.0,
        "connected_component_count": 1.0,
        "skeleton_length": 1.0,
        "distance_mean": 1.0,
        "distance_std": 0.0,
        "distance_max": 1.0,
        "hu_1": 0.0,
        "hu_2": 0.0,
        "hu_3": 0.0,
        "hu_4": 0.0,
        "hu_5": 0.0,
        "hu_6": 0.0,
        "hu_7": 0.0,
    }
    engine.extract_object_signatures = lambda _image: [object()]

    engine.fit_baseline([np.zeros((64, 64)) for _ in profiles])

    raw_mean = np.mean(profiles, axis=0)
    assert np.max(raw_mean) < np.max(reference)
    assert np.allclose(engine.mean_profile, reference)


def test_internal_hole_changes_topology_descriptors():
    engine = TopologicalGhostEngine(resolution=64)
    solid = _ellipse_image(brightness=1.0)
    holed = solid.copy()
    rr, cc = ellipse(32, 32, 5, 5, shape=holed.shape)
    holed[rr, cc] = 0.0

    solid_topology = engine.extract_topology_descriptors(solid)
    holed_topology = engine.extract_topology_descriptors(holed)

    assert solid_topology["hole_count"] == 0.0
    assert holed_topology["hole_count"] >= 1.0
    assert solid_topology["euler_number"] != holed_topology["euler_number"]


def test_same_outer_boundary_with_internal_hole_is_rejected():
    engine = TopologicalGhostEngine(resolution=64)
    baseline = [_ellipse_image(brightness=0.9) for _ in range(10)]
    engine.fit_baseline(baseline)
    holed = _ellipse_image(brightness=0.9)
    rr, cc = ellipse(32, 32, 5, 5, shape=holed.shape)
    holed[rr, cc] = 0.0

    verdict = engine.verify(holed, sensitivity=1.0)

    assert verdict["is_mutation"] is True
    assert verdict["topology_distance"] > 0.0


def test_multi_object_all_mode_rejects_detached_fragment():
    engine = TopologicalGhostEngine(resolution=64, object_mode="all")
    baseline = [_ellipse_image(brightness=0.9) for _ in range(10)]
    engine.fit_baseline(baseline)
    fragmented = _ellipse_image(brightness=0.9)
    rr, cc = ellipse(8, 8, 4, 4, shape=fragmented.shape)
    fragmented[rr, cc] = 0.9

    verdict = engine.verify(fragmented, sensitivity=10.0)

    assert verdict["object_count_mismatch"] is True
    assert verdict["is_mutation"] is True


def test_model_save_load_round_trip(tmp_path):
    engine = TopologicalGhostEngine(resolution=64, object_mode="largest")
    baseline = [_ellipse_image(brightness=0.9) for _ in range(10)]
    engine.fit_baseline(baseline, metadata={"suite": "unit"})
    model_path = tmp_path / "model.json"

    engine.save(model_path)
    loaded = TopologicalGhostEngine.load(model_path)

    assert np.allclose(loaded.mean_profile, engine.mean_profile)
    assert loaded.variance_cutoff == pytest.approx(engine.variance_cutoff)
    assert loaded.training_metadata["suite"] == "unit"
