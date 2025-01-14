import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm

def calculate_adaptive_threshold(vesselness, sigma_max, voxel_size_mm=0.6, Tmin=0.07, Tmax=0.17):
    """Calculate adaptive threshold based on vessel size
    
    Args:
        vesselness: Maximum vesselness response (Vmax)
        sigma_max: Scale of maximum response (σmax)
        voxel_size_mm: Voxel size in mm
        Tmin: Minimum threshold for smallest vessels (default: 0.07)
        Tmax: Maximum threshold for largest vessels (default: 0.17)
        
    Returns:
        binary_vessels: Binary vessel mask
        threshold_map: Map of thresholds used
    """
    print("Starting adaptive threshold calculation...")
    
    # Width parameter for scale transition (w = 2 × voxel_size_mm)
    w = 2.0 * voxel_size_mm
    
    print(f"Image shape: {vesselness.shape}")
    print("Counting scales...")
    
    # Count number of scales smaller than w
    n = np.sum(sigma_max < w)
    if n == 0:  # Handle case where all scales are larger than w
        n = 1
    
    print(f"Number of scales smaller than {w:.2f}mm: {n}")
    print("Calculating threshold values...")
    
    # Pre-calculate all thresholds at once
    scale_indices = np.arange(n)
    thresholds = Tmin * np.exp(np.log(Tmax/Tmin) * scale_indices / n)
    
    print("Creating scale index array...")
    # Create scale index array
    scale_indices = np.floor(sigma_max / (w/n)).astype(int)
    scale_indices = np.clip(scale_indices, 0, n-1)
    
    print("Applying thresholds...")
    # Apply thresholds using vectorized indexing
    threshold_map = thresholds[scale_indices]
    
    # Set maximum threshold for all scales >= w
    threshold_map[sigma_max >= w] = Tmax
    
    print("Generating binary vessel mask...")
    # Apply thresholds to get binary vessel mask
    binary_vessels = (vesselness >= threshold_map).astype(np.uint8)
    
    print("Adaptive threshold calculation complete.")
    return binary_vessels, threshold_map

def save_threshold_results(binary_vessels, threshold_map, output_dir):
    """Save thresholding results"""
    os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(
        sitk.GetImageFromArray(binary_vessels),
        os.path.join(output_dir, 'binary_vessels.nrrd')
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(threshold_map),
        os.path.join(output_dir, 'threshold_map.nrrd')
    )
