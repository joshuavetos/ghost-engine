import logging

import numpy as np
from scipy.ndimage import center_of_mass
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import active_contour

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TopologicalGhostEngine:
    """
    Validates structural anomalies by separating geometric mutations
    from intensity, lighting, and preprocessing noise.
    """

    def __init__(
        self,
        resolution=64,
        num_spline_points=100,
        alpha=0.015,
        beta=3.0,
        w_line=0.0,
        w_edge=1.0,
        gamma=0.01,
        max_px_move=1.0,
        max_num_iter=2500,
        convergence=0.1,
        radius_factor=0.25,
    ):
        self.resolution = resolution
        self.num_spline_points = num_spline_points
        self.alpha = alpha
        self.beta = beta
        self.w_line = w_line
        self.w_edge = w_edge
        self.gamma = gamma
        self.max_px_move = max_px_move
        self.max_num_iter = max_num_iter
        self.convergence = convergence
        self.radius_factor = radius_factor
        self.default_center = np.array([resolution // 2, resolution // 2], dtype=float)
        self.mean_profile = None
        self.variance_cutoff = None
        self.last_diagnostics = None

    def _to_grayscale(self, image):
        image = np.asarray(image, dtype=float)
        expected_2d = (self.resolution, self.resolution)

        if not np.all(np.isfinite(image)):
            raise ValueError("Input image must contain only finite values.")

        if image.shape == expected_2d:
            return image

        if image.ndim == 3 and image.shape[:2] == expected_2d:
            channels = image.shape[2]
            if channels == 1:
                return image[:, :, 0]
            if channels in (3, 4):
                return rgb2gray(image[:, :, :3])
            return np.mean(image, axis=2)

        raise ValueError(
            f"Input image must be explicitly reshaped to ({self.resolution}, {self.resolution}) "
            f"or ({self.resolution}, {self.resolution}, channels); got {image.shape}"
        )

    def _estimate_center(self, image):
        if np.allclose(image, image.flat[0]):
            return self.default_center.copy()

        try:
            cutoff = threshold_otsu(image)
            mask = image > cutoff
        except ValueError:
            mask = image > np.mean(image)

        if not np.any(mask):
            return self.default_center.copy()

        row, col = center_of_mass(mask.astype(float))
        if not np.isfinite(row) or not np.isfinite(col):
            return self.default_center.copy()
        return np.array([row, col], dtype=float)

    def _get_init_spline(self, center=None, radius_factor=None):
        center = self.default_center if center is None else np.asarray(center, dtype=float)
        radius_factor = self.radius_factor if radius_factor is None else radius_factor
        radius = self.resolution * radius_factor
        s = np.linspace(0, 2 * np.pi, self.num_spline_points, endpoint=False)
        r = center[0] + radius * np.sin(s)
        c = center[1] + radius * np.cos(s)
        r = np.clip(r, 0, self.resolution - 1)
        c = np.clip(c, 0, self.resolution - 1)
        return np.array([r, c]).T

    def extract_signature(self, image, sigma=1.5, return_diagnostics=False):
        grayscale = self._to_grayscale(image)
        img_smoothed = gaussian(grayscale, sigma=sigma, preserve_range=True)
        center = self._estimate_center(img_smoothed)
        init_spline = self._get_init_spline(center=center)

        # beta is the active-contour bending-rigidity term; higher values produce smoother snakes.
        snake = active_contour(
            img_smoothed,
            init_spline,
            alpha=self.alpha,
            beta=self.beta,
            w_line=self.w_line,
            w_edge=self.w_edge,
            gamma=self.gamma,
            max_px_move=self.max_px_move,
            max_num_iter=self.max_num_iter,
            convergence=self.convergence,
        )

        profile = np.linalg.norm(snake - center, axis=1)
        diagnostics = {
            "center_row": float(center[0]),
            "center_col": float(center[1]),
            "snake_min_row": float(np.min(snake[:, 0])),
            "snake_max_row": float(np.max(snake[:, 0])),
            "snake_min_col": float(np.min(snake[:, 1])),
            "snake_max_col": float(np.max(snake[:, 1])),
        }
        self.last_diagnostics = diagnostics

        if return_diagnostics:
            return profile, diagnostics
        return profile

    def _profile_distance(self, profile, reference_profile):
        profile = np.asarray(profile, dtype=float)
        reference_profile = np.asarray(reference_profile, dtype=float)
        if profile.shape != reference_profile.shape:
            raise ValueError("Profile shapes must match before distance comparison.")

        distances = []
        for candidate in (profile, profile[::-1]):
            for shift in range(candidate.size):
                aligned = np.roll(candidate, shift)
                distances.append(np.sqrt(np.mean((aligned - reference_profile) ** 2)))
        return float(np.min(distances))

    def fit_baseline(self, baseline_images, percentile=95):
        if len(baseline_images) < 10:
            raise ValueError("Baseline dataset is too small to build a stable statistical profile.")

        logging.info(f"Profiling baseline variance across {len(baseline_images)} samples...")
        profiles = [self.extract_signature(img) for img in baseline_images]
        self.mean_profile = np.mean(profiles, axis=0)

        intra_dists = [self._profile_distance(p, self.mean_profile) for p in profiles]
        self.variance_cutoff = np.percentile(intra_dists, percentile)
        logging.info(f"Baseline established. Empirical Variance Cutoff: {self.variance_cutoff:.4f}")

    def verify(self, sample_image, sensitivity=1.3):
        if self.mean_profile is None or self.variance_cutoff is None:
            raise RuntimeError("Engine state uninitialized. Call fit_baseline() before verification.")

        sample_profile = self.extract_signature(sample_image)
        distance = self._profile_distance(sample_profile, self.mean_profile)

        threshold = self.variance_cutoff * sensitivity
        is_mutation = distance > threshold

        return {
            "is_mutation": bool(is_mutation),
            "ordered_profile_distance": float(distance),
            "threshold_cutoff": float(threshold),
            "structural_deviation_ratio": float(distance / self.variance_cutoff),
            "diagnostics": dict(self.last_diagnostics),
        }
