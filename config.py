import numpy as np

def validate_input_tensor(tensor):
    """Ensures input images are normalized [0, 1] and strictly single-channel."""
    if np.max(tensor) > 1.0 or np.min(tensor) < 0.0:
        raise ValueError("FAIL-CLOSED: Input data breaks normalization bounds. Must be scaled to [0, 1].")
        
    if len(tensor.shape) != 2:
        raise ValueError(f"FAIL-CLOSED: Expected a 2D grayscale matrix, got shape {tensor.shape}")
    return True
