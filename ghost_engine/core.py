import numpy as np
import logging
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from scipy.stats import wasserstein_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TopologicalGhostEngine:
    """
    Validates structural anomalies by separating geometric mutations 
    from intensity, lighting, and preprocessing noise.
    """
    def __init__(self, resolution=64, num_spline_points=100):
        self.resolution = resolution
        self.num_spline_points = num_spline_points
        self.center = np.array([resolution // 2, resolution // 2])
        self.mean_profile = None
        self.variance_cutoff = None

    def _get_init_spline(self, radius_factor=0.25):
        s = np.linspace(0, 2 * np.pi, self.num_spline_points)
        r = self.center[0] + (self.resolution * radius_factor) * np.sin(s)
        c = self.center[1] + (self.resolution * radius_factor) * np.cos(s)
        return np.array([r, c]).T

    def extract_signature(self, image, sigma=1.5):
        if image.shape != (self.resolution, self.resolution):
            raise ValueError(f"Input image must be explicitly reshaped to ({self.resolution}, {self.resolution})")
            
        img_smoothed = gaussian(image, sigma=sigma)
        init_spline = self._get_init_spline()
        
        # Rigidity set to beta=3 to facilitate non-circular corner evaluation
        snake = active_contour(
            img_smoothed,
            init_spline,
            alpha=0.015,
            beta=3
        )
        
        return np.linalg.norm(snake - self.center, axis=1)

    def fit_baseline(self, baseline_images, percentile=95):
        if len(baseline_images) < 10:
            raise ValueError("Baseline dataset is too small to build a stable statistical profile.")
            
        logging.info(f"Profiling baseline variance across {len(baseline_images)} samples...")
        profiles = [self.extract_signature(img) for img in baseline_images]
        self.mean_profile = np.mean(profiles, axis=0)
        
        intra_dists = [wasserstein_distance(p, self.mean_profile) for p in profiles]
        self.variance_cutoff = np.percentile(intra_dists, percentile)
        logging.info(f"Baseline established. Empirical Variance Cutoff: {self.variance_cutoff:.4f}")

    def verify(self, sample_image, sensitivity=1.3):
        if self.mean_profile is None or self.variance_cutoff is None:
            raise RuntimeError("Engine state uninitialized. Call fit_baseline() before verification.")
            
        sample_profile = self.extract_signature(sample_image)
        w_dist = wasserstein_distance(sample_profile, self.mean_profile)
        
        threshold = self.variance_cutoff * sensitivity
        is_mutation = w_dist > threshold
        
        return {
            "is_mutation": bool(is_mutation),
            "wasserstein_distance": float(w_dist),
            "threshold_cutoff": float(threshold),
            "structural_deviation_ratio": float(w_dist / self.variance_cutoff)
        }
