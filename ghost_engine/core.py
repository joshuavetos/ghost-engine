import inspect
import logging

import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects
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
        channel_weights=None,
        min_component_area_ratio=0.01,
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
        self.channel_weights = self._normalize_channel_weights(channel_weights)
        self.min_component_area_ratio = min_component_area_ratio
        self.default_center = np.array([resolution // 2, resolution // 2], dtype=float)
        self.mean_profile = None
        self.variance_cutoff = None
        self.last_diagnostics = None

    def _normalize_channel_weights(self, channel_weights):
        if channel_weights is None:
            return None

        weights = np.asarray(channel_weights, dtype=float)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("Channel weights must be a non-empty 1D sequence.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("Channel weights must contain only finite values.")

        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            raise ValueError("Channel weights must have a positive sum.")
        return weights / weight_sum

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
            if self.channel_weights is None:
                raise ValueError(
                    "Non-RGB channel-last tensors require explicit channel_weights; "
                    f"got {channels} channels."
                )
            if self.channel_weights.size != channels:
                raise ValueError(
                    f"channel_weights length ({self.channel_weights.size}) must match image channels ({channels})."
                )
            return np.tensordot(image, self.channel_weights, axes=([2], [0]))

        raise ValueError(
            f"Input image must be explicitly reshaped to ({self.resolution}, {self.resolution}) "
            f"or ({self.resolution}, {self.resolution}, channels); got {image.shape}"
        )


    def _remove_small_holes(self, mask, max_hole_size):
        params = inspect.signature(remove_small_holes).parameters
        if "max_size" in params:
            return remove_small_holes(mask, max_size=max_hole_size)
        return remove_small_holes(mask, area_threshold=max_hole_size)

    def _remove_small_objects(self, mask, min_object_size):
        params = inspect.signature(remove_small_objects).parameters
        if "max_size" in params:
            return remove_small_objects(mask, max_size=min_object_size)
        return remove_small_objects(mask, min_size=min_object_size)

    def _segment_foreground(self, image):
        if np.allclose(image, image.flat[0]):
            return np.zeros_like(image, dtype=bool)

        try:
            cutoff = threshold_otsu(image)
            mask = image > cutoff
        except ValueError:
            mask = image > np.mean(image)

        inverse_mask = ~mask
        if np.mean(mask) > 0.5 and np.any(inverse_mask):
            mask = inverse_mask

        min_size = max(1, int(self.resolution * self.resolution * self.min_component_area_ratio))
        mask = closing(mask, disk(1))
        mask = self._remove_small_holes(mask, min_size)
        mask = self._remove_small_objects(mask, min_size)

        labeled = label(mask)
        regions = regionprops(labeled)
        if not regions:
            return np.zeros_like(mask, dtype=bool)

        largest_region = max(regions, key=lambda region: region.area)
        return labeled == largest_region.label

    def _estimate_center(self, image, mask=None):
        mask = self._segment_foreground(image) if mask is None else mask
        if not np.any(mask):
            return self.default_center.copy()

        labeled = label(mask)
        regions = regionprops(labeled)
        if not regions:
            return self.default_center.copy()

        region = max(regions, key=lambda item: item.area)
        min_row, min_col, max_row, max_col = region.bbox
        bbox_center = np.array([(min_row + max_row - 1) / 2, (min_col + max_col - 1) / 2], dtype=float)

        distances = distance_transform_edt(mask)
        max_distance = np.max(distances)
        if max_distance > 0:
            medial_points = np.argwhere(np.isclose(distances, max_distance))
            medial_center = np.mean(medial_points, axis=0)
        else:
            medial_center = bbox_center

        row, col = center_of_mass(mask.astype(float))
        if np.isfinite(row) and np.isfinite(col):
            centroid = np.array([row, col], dtype=float)
        else:
            centroid = bbox_center

        center = np.mean([bbox_center, medial_center, centroid], axis=0)
        return center.astype(float)

    def _estimate_initial_radius_factor(self, image, mask=None):
        mask = self._segment_foreground(image) if mask is None else mask
        if not np.any(mask):
            return self.radius_factor

        rows, cols = np.where(mask)
        object_radius = max(np.ptp(rows) + 1, np.ptp(cols) + 1) / 2
        expanded_radius = object_radius * 1.15
        radius_factor = expanded_radius / self.resolution
        return float(np.clip(radius_factor, 0.1, 0.45))

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
        foreground_mask = self._segment_foreground(img_smoothed)
        center = self._estimate_center(img_smoothed, mask=foreground_mask)
        radius_factor = self._estimate_initial_radius_factor(img_smoothed, mask=foreground_mask)
        init_spline = self._get_init_spline(center=center, radius_factor=radius_factor)

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
            "initial_radius_factor": float(radius_factor),
            "foreground_area": int(np.sum(foreground_mask)),
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
        if profile.ndim != 1:
            raise ValueError("Profiles must be 1D before distance comparison.")
        if not np.all(np.isfinite(profile)) or not np.all(np.isfinite(reference_profile)):
            raise ValueError("Profiles must contain only finite values.")

        n = profile.size
        if n == 0:
            raise ValueError("Profiles must not be empty.")

        reference_energy = np.sum(reference_profile ** 2)
        profile_energy = np.sum(profile ** 2)
        reference_fft = np.fft.fft(reference_profile)

        min_distance_sq = np.inf
        for candidate in (profile, profile[::-1]):
            correlations = np.fft.ifft(np.fft.fft(candidate) * np.conj(reference_fft)).real
            distance_sq = (profile_energy + reference_energy - 2 * correlations) / n
            min_distance_sq = min(min_distance_sq, np.min(distance_sq))

        return float(np.sqrt(max(min_distance_sq, 0.0)))

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
