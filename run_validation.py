import json
import numpy as np
from skimage.draw import ellipse, rectangle
from ghost_engine.core import TopologicalGhostEngine
from config import validate_input_tensor

def generate_clean_ellipse(r_radius, c_radius, noise_level=0.05, brightness=1.0):
    """Generates a structured geometric primitive with illumination noise."""
    img = np.random.normal(0, noise_level, (64, 64))
    rr, cc = ellipse(32, 32, r_radius, c_radius, shape=(64, 64))
    img[rr, cc] = brightness
    return np.clip(img, 0, 1)

def generate_rectangle_mutation(brightness=1.0):
    """
    Generates a true structural mutation using a convex rectangle 
    to guarantee clean rendering and prevent snake bridging.
    """
    img = np.random.normal(0, 0.05, (64, 64))
    start = (16, 16)
    extent = (32, 32)
    rr, cc = rectangle(start=start, extent=extent, shape=(64, 64))
    img[rr, cc] = brightness
    return np.clip(img, 0, 1)

def main():
    # Enforce global random seed for deterministic execution bounds
    np.random.seed(42)
    
    print("Generating structured geometric dataset...")
    baseline_mock = []
    for _ in range(100):
        r_rad = np.random.randint(14, 18)
        c_rad = np.random.randint(14, 18)
        brightness = np.random.uniform(0.7, 1.0)
        baseline_mock.append(generate_clean_ellipse(r_rad, c_rad, brightness=brightness))
    
    test_pool = [generate_clean_ellipse(16, 16, brightness=0.6) for _ in range(9)]
    test_pool.insert(3, generate_rectangle_mutation(brightness=0.8))

    # Guardrails
    for img in baseline_mock: validate_input_tensor(img)
    for img in test_pool: validate_input_tensor(img)

    # Initialize engine
    engine = TopologicalGhostEngine(resolution=64)
    engine.fit_baseline(baseline_mock)
    
    audit_log = {}
    print("\nExecuting batch structural auditing...")
    
    for idx, sample in enumerate(test_pool):
        verdict = engine.verify(sample)
        audit_log[f"sample_{idx}"] = verdict
        
        if verdict["is_mutation"]:
            print(f" -> [ALERT] Sample {idx}: True Structural Mutation Found. Dist: {verdict['ordered_profile_distance']:.4f}")
        else:
            print(f" -> Sample {idx}: Stable Geometry.")

    # Assertion Layer 1: Catch False Negatives
    if not audit_log["sample_3"]["is_mutation"]:
        raise AssertionError(
            f"FAIL-CLOSED (False Negative): The engine failed to isolate the known "
            f"structural mutation at index 3. Distance: {audit_log['sample_3']['ordered_profile_distance']:.4f}"
        )
        
    # Assertion Layer 2: Catch False Positives / Miscalibrations
    stable_indices = [i for i in range(len(test_pool)) if i != 3]
    false_positives = [i for i in stable_indices if audit_log[f"sample_{i}"]["is_mutation"]]
    if len(false_positives) > 0:
        raise AssertionError(
            f"FAIL-CLOSED (False Positive): Engine miscalibration detected. Stable samples "
            f"{false_positives} were incorrectly flagged as structural mutations."
        )
    
    print("\nRegression Check Passed: 100% detection accuracy. Zero false positives.")

    with open("audit_report.json", "w") as f:
        json.dump(audit_log, f, indent=4)
    print("Audit report exported to 'audit_report.json'.")

if __name__ == "__main__":
    main()
