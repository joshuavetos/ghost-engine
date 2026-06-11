import numpy as np


def validate_input_tensor(tensor):
    """Ensures input images are finite, normalized [0, 1], and image-shaped."""
    tensor = np.asarray(tensor)

    if tensor.size == 0:
        raise ValueError("FAIL-CLOSED: Input data is empty.")

    if not np.all(np.isfinite(tensor)):
        raise ValueError("FAIL-CLOSED: Input data contains NaN or infinite values.")

    if np.max(tensor) > 1.0 or np.min(tensor) < 0.0:
        raise ValueError("FAIL-CLOSED: Input data breaks normalization bounds. Must be scaled to [0, 1].")

    if len(tensor.shape) == 2:
        return True

    if len(tensor.shape) == 3 and tensor.shape[2] > 0:
        return True

    raise ValueError(f"FAIL-CLOSED: Expected a 2D matrix or channel-last 3D image tensor, got shape {tensor.shape}")
