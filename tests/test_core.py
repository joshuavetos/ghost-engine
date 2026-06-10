import pytest
import numpy as np
from ghost_engine.core import TopologicalGhostEngine
from config import validate_input_tensor

def test_config_normalization_guardrail():
    """Ensures input tensors outside [0, 1] boundaries trip a ValueError."""
    invalid_tensor = np.array([[0.0, 1.5], [0.0, 0.0]])
    with pytest.raises(ValueError, match="FAIL-CLOSED: Input data breaks normalization bounds"):
        validate_input_tensor(invalid_tensor)

def test_config_dimension_guardrail():
    """Ensures multi-channel (3D) tensors trip a ValueError."""
    invalid_tensor = np.zeros((64, 64, 3))
    with pytest.raises(ValueError, match="FAIL-CLOSED: Expected a 2D grayscale matrix"):
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
