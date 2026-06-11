import inspect
import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, moments_central, moments_hu, moments_normalized, regionprops
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects, skeletonize
from skimage.segmentation import active_contour

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ENGINE_STATE_VERSION = "0.2.0"


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
        object_mode="largest",
        topology_weight=1.0,
    ):
        if object_mode not in {"largest", "all", "indexed"}:
            raise ValueError("object_mode must be one of: 'largest', 'all', 'indexed'.")
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
        self.object_mode = object_mode
        self.topology_weight = float(topology_weight)
        self.default_center = np.array([resolution // 2, resolution // 2], dtype=float)
        self.mean_profile = None
        self.mean_topology = None
        self.variance_cutoff = None
        self.baseline_object_count = None
        self.baseline_profiles = None
        self.baseline_topologies = None
        self.training_metadata = None
        self.last_diagnostics = None

    def _engine_parameters(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "w_line": self.w_line,
            "w_edge": self.w_edge,
            "gamma": self.gamma,
            "max_px_move": self.max_px_move,
            "max_num_iter": self.max_num_iter,
            "convergence": self.convergence,
            "radius_factor": self.radius_factor,
            "min_component_area_ratio": self.min_component_area_ratio,
            "object_mode": self.object_mode,
            "topology_weight": self.topology_weight,
        }

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

    def _threshold_foreground(self, image):
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
        return mask.astype(bool)

    def _clean_foreground(self, mask, fill_holes=True):
        min_size = max(1, int(self.resolution * self.resolution * self.min_component_area_ratio))
        mask = closing(mask, disk(1))
        if fill_holes:
            mask = self._remove_small_holes(mask, min_size)
        return self._remove_small_objects(mask, min_size)

    def _component_masks(self, image, mode=None, fill_holes=True):
        mode = self.object_mode if mode is None else mode
        mask = self._clean_foreground(self._threshold_foreground(image), fill_holes=fill_holes)
        labeled = label(mask)
        regions = sorted(regionprops(labeled), key=lambda region: (region.bbox[0], region.bbox[1]))
        if not regions:
            return []
        if mode == "largest":
            largest_region = max(regions, key=lambda region: region.area)
            return [labeled == largest_region.label]
        if mode in {"all", "indexed"}:
            return [labeled == region.label for region in regions]
        raise ValueError("object_mode must be one of: 'largest', 'all', 'indexed'.")

    def _segment_foreground(self, image):
        masks = self._component_masks(image, mode="largest", fill_holes=True)
        if not masks:
            return np.zeros_like(image, dtype=bool)
        return masks[0]

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
        medial_center = np.mean(np.argwhere(np.isclose(distances, max_distance)), axis=0) if max_distance > 0 else bbox_center
        row, col = center_of_mass(mask.astype(float))
        centroid = np.array([row, col], dtype=float) if np.isfinite(row) and np.isfinite(col) else bbox_center
        center = np.mean([bbox_center, medial_center, centroid], axis=0)
        return center.astype(float)

    def _estimate_initial_radius_factor(self, image, mask=None):
        mask = self._segment_foreground(image) if mask is None else mask
        if not np.any(mask):
            return self.radius_factor
        rows, cols = np.where(mask)
        object_radius = max(np.ptp(rows) + 1, np.ptp(cols) + 1) / 2
        return float(np.clip(object_radius * 1.15 / self.resolution, 0.1, 0.45))

    def _get_init_spline(self, center=None, radius_factor=None):
        center = self.default_center if center is None else np.asarray(center, dtype=float)
        radius_factor = self.radius_factor if radius_factor is None else radius_factor
        radius = self.resolution * radius_factor
        s = np.linspace(0, 2 * np.pi, self.num_spline_points, endpoint=False)
        r = center[0] + radius * np.sin(s)
        c = center[1] + radius * np.cos(s)
        return np.array([np.clip(r, 0, self.resolution - 1), np.clip(c, 0, self.resolution - 1)]).T

    def _topology_descriptors_from_mask(self, mask):
        mask = np.asarray(mask, dtype=bool)
        labeled = label(mask)
        props = regionprops(labeled)
        component_count = len(props)
        euler_number = int(sum(region.euler_number for region in props)) if props else 0
        hole_count = int(component_count - euler_number)
        skeleton_length = int(np.sum(skeletonize(mask))) if np.any(mask) else 0
        distances = distance_transform_edt(mask)
        positive_distances = distances[distances > 0]
        raw_moments = mask.astype(float)
        hu = moments_hu(moments_normalized(moments_central(raw_moments))) if np.any(mask) else np.zeros(7)
        return {
            "hole_count": float(max(hole_count, 0)),
            "euler_number": float(euler_number),
            "connected_component_count": float(component_count),
            "skeleton_length": float(skeleton_length),
            "distance_mean": float(np.mean(positive_distances)) if positive_distances.size else 0.0,
            "distance_std": float(np.std(positive_distances)) if positive_distances.size else 0.0,
            "distance_max": float(np.max(positive_distances)) if positive_distances.size else 0.0,
            **{f"hu_{idx + 1}": float(value) for idx, value in enumerate(hu)},
        }

    def extract_topology_descriptors(self, image):
        grayscale = self._to_grayscale(image)
        img_smoothed = gaussian(grayscale, sigma=1.0, preserve_range=True)
        masks = self._component_masks(img_smoothed, mode=self.object_mode, fill_holes=False)
        if not masks:
            return self._topology_descriptors_from_mask(np.zeros((self.resolution, self.resolution), dtype=bool))
        combined = np.logical_or.reduce(masks)
        return self._topology_descriptors_from_mask(combined)

    def extract_signature(self, image, sigma=1.5, return_diagnostics=False, component_mask=None):
        grayscale = self._to_grayscale(image)
        img_smoothed = gaussian(grayscale, sigma=sigma, preserve_range=True)
        foreground_mask = self._segment_foreground(img_smoothed) if component_mask is None else component_mask
        center = self._estimate_center(img_smoothed, mask=foreground_mask)
        radius_factor = self._estimate_initial_radius_factor(img_smoothed, mask=foreground_mask)
        init_spline = self._get_init_spline(center=center, radius_factor=radius_factor)
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
            "snake_points": snake.tolist(),
        }
        self.last_diagnostics = diagnostics
        if return_diagnostics:
            return profile, diagnostics
        return profile

    def extract_object_signatures(self, image):
        grayscale = self._to_grayscale(image)
        img_smoothed = gaussian(grayscale, sigma=1.5, preserve_range=True)
        masks = self._component_masks(img_smoothed, mode=self.object_mode, fill_holes=True)
        objects = []
        for idx, mask in enumerate(masks):
            profile, diagnostics = self.extract_signature(image, return_diagnostics=True, component_mask=mask)
            objects.append({
                "index": idx,
                "profile": profile,
                "topology": self._topology_descriptors_from_mask(mask),
                "diagnostics": diagnostics,
            })
        return objects

    def _best_profile_alignment(self, profile, reference_profile):
        profile = np.asarray(profile, dtype=float)
        reference_profile = np.asarray(reference_profile, dtype=float)
        if profile.shape != reference_profile.shape:
            raise ValueError("Profile shapes must match before alignment.")
        if profile.ndim != 1 or profile.size == 0:
            raise ValueError("Profiles must be non-empty 1D arrays before alignment.")
        if not np.all(np.isfinite(profile)) or not np.all(np.isfinite(reference_profile)):
            raise ValueError("Profiles must contain only finite values.")
        n = profile.size
        reference_fft = np.fft.fft(reference_profile)
        best = None
        for reflected, candidate in ((False, profile), (True, profile[::-1])):
            correlations = np.fft.ifft(np.fft.fft(candidate) * np.conj(reference_fft)).real
            shift = int(np.argmax(correlations))
            aligned = np.roll(candidate, -shift)
            distance = float(np.sqrt(np.mean((aligned - reference_profile) ** 2)))
            if best is None or distance < best["distance"]:
                best = {"profile": aligned, "shift": shift, "reflected": reflected, "distance": distance}
        return best

    def _align_profile_to_reference(self, profile, reference_profile):
        return self._best_profile_alignment(profile, reference_profile)["profile"]

    def _profile_distance(self, profile, reference_profile):
        return self._best_profile_alignment(profile, reference_profile)["distance"]

    def _topology_vector(self, descriptors):
        keys = [
            "hole_count", "euler_number", "connected_component_count", "skeleton_length",
            "distance_mean", "distance_std", "distance_max", "hu_1", "hu_2", "hu_3",
            "hu_4", "hu_5", "hu_6", "hu_7",
        ]
        return np.array([float(descriptors.get(key, 0.0)) for key in keys], dtype=float)

    def _topology_distance(self, topology, reference_topology):
        vec = self._topology_vector(topology)
        ref = self._topology_vector(reference_topology)
        scale = np.maximum(np.abs(ref), 1.0)
        return float(np.sqrt(np.mean(((vec - ref) / scale) ** 2)))

    def fit_baseline(self, baseline_images, percentile=95, metadata=None):
        if len(baseline_images) < 10:
            raise ValueError("Baseline dataset is too small to build a stable statistical profile.")
        logging.info(f"Profiling baseline variance across {len(baseline_images)} samples...")
        profiles = [self.extract_signature(img) for img in baseline_images]
        reference_profile = profiles[0]
        aligned_profiles = [self._align_profile_to_reference(profile, reference_profile) for profile in profiles]
        self.mean_profile = np.mean(aligned_profiles, axis=0)
        self.baseline_profiles = np.asarray(aligned_profiles)
        topologies = [self.extract_topology_descriptors(img) for img in baseline_images]
        topology_vectors = np.asarray([self._topology_vector(item) for item in topologies])
        mean_topology_vector = np.mean(topology_vectors, axis=0)
        topology_keys = list(topologies[0].keys())
        self.mean_topology = {key: float(value) for key, value in zip(topology_keys, mean_topology_vector)}
        self.baseline_topologies = topologies
        object_counts = [len(self.extract_object_signatures(img)) for img in baseline_images]
        self.baseline_object_count = int(round(float(np.median(object_counts))))
        intra_dists = [
            self._combined_distance(profile, topology, self.mean_profile, self.mean_topology)
            for profile, topology in zip(aligned_profiles, topologies)
        ]
        self.variance_cutoff = float(np.percentile(intra_dists, percentile))
        self.training_metadata = {
            "sample_count": len(baseline_images),
            "percentile": percentile,
            **({} if metadata is None else dict(metadata)),
        }
        logging.info(f"Baseline established. Empirical Variance Cutoff: {self.variance_cutoff:.4f}")

    def _combined_distance(self, profile, topology, reference_profile, reference_topology):
        profile_distance = self._profile_distance(profile, reference_profile)
        topology_distance = self._topology_distance(topology, reference_topology) if reference_topology is not None else 0.0
        return float(profile_distance + self.topology_weight * topology_distance)

    def verify(self, sample_image, sensitivity=1.3):
        if self.mean_profile is None or self.variance_cutoff is None:
            raise RuntimeError("Engine state uninitialized. Call fit_baseline() before verification.")
        sample_profile = self.extract_signature(sample_image)
        sample_topology = self.extract_topology_descriptors(sample_image)
        profile_distance = self._profile_distance(sample_profile, self.mean_profile)
        topology_distance = self._topology_distance(sample_topology, self.mean_topology)
        distance = float(profile_distance + self.topology_weight * topology_distance)
        object_count = len(self.extract_object_signatures(sample_image))
        object_count_mismatch = self.object_mode in {"all", "indexed"} and object_count != self.baseline_object_count
        threshold = float(self.variance_cutoff * sensitivity)
        is_mutation = distance > threshold or object_count_mismatch
        return {
            "is_mutation": bool(is_mutation),
            "ordered_profile_distance": float(profile_distance),
            "topology_distance": float(topology_distance),
            "combined_distance": distance,
            "threshold_cutoff": threshold,
            "structural_deviation_ratio": float(distance / self.variance_cutoff) if self.variance_cutoff else float("inf"),
            "object_count": int(object_count),
            "baseline_object_count": int(self.baseline_object_count) if self.baseline_object_count is not None else None,
            "object_count_mismatch": bool(object_count_mismatch),
            "topology_descriptors": sample_topology,
            "diagnostics": dict(self.last_diagnostics),
        }

    def save(self, path):
        if self.mean_profile is None or self.variance_cutoff is None:
            raise RuntimeError("Engine state uninitialized. Call fit_baseline() before saving.")
        state = {
            "version": ENGINE_STATE_VERSION,
            "resolution": self.resolution,
            "num_spline_points": self.num_spline_points,
            "engine_parameters": self._engine_parameters(),
            "channel_weights": None if self.channel_weights is None else self.channel_weights.tolist(),
            "mean_profile": self.mean_profile.tolist(),
            "mean_topology": self.mean_topology,
            "variance_cutoff": float(self.variance_cutoff),
            "baseline_object_count": self.baseline_object_count,
            "training_metadata": self.training_metadata,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        params = dict(state.get("engine_parameters", {}))
        engine = cls(
            resolution=state["resolution"],
            num_spline_points=state["num_spline_points"],
            channel_weights=state.get("channel_weights"),
            **params,
        )
        engine.mean_profile = np.asarray(state["mean_profile"], dtype=float)
        engine.mean_topology = state.get("mean_topology")
        engine.variance_cutoff = float(state["variance_cutoff"])
        engine.baseline_object_count = state.get("baseline_object_count")
        engine.training_metadata = state.get("training_metadata")
        return engine
