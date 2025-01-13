import numpy as np
import SimpleITK as sitk
import os

def calculate_adaptive_threshold(vesselness, sigma_max, voxel_size_mm=0.5, Tmin=0.07, Tmax=0.17):
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
    # Width parameter for scale transition (w = 2 × voxel_size_mm)
    w = 2.0 * voxel_size_mm
    
    # Count number of scales smaller than w
    n = np.sum(sigma_max < w)
    if n == 0:  # Handle case where all scales are larger than w
        n = 1
    
    # Initialize threshold map
    threshold_map = np.zeros_like(sigma_max)
    
    # Calculate thresholds for each scale index
    for i in range(n):
        # Find voxels at this scale index
        scale_mask = (sigma_max >= (i * w / n)) & (sigma_max < ((i + 1) * w / n))
        # Apply exponential interpolation formula
        threshold_map[scale_mask] = Tmin * np.exp(np.log(Tmax/Tmin) * i / n)
    
    # Set maximum threshold for all scales >= w
    threshold_map[sigma_max >= w] = Tmax
    
    # Apply thresholds to get binary vessel mask
    binary_vessels = (vesselness >= threshold_map).astype(np.uint8)
    
    return binary_vessels, threshold_map

def save_threshold_results(binary_vessels, threshold_map, output_dir):
    """Save thresholding results
    
    Args:
        binary_vessels: Binary vessel mask
        threshold_map: Map of thresholds used
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary vessel mask
    sitk.WriteImage(
        sitk.GetImageFromArray(binary_vessels),
        os.path.join(output_dir, 'binary_vessels.nrrd')
    )
    
    # Save threshold map
    sitk.WriteImage(
        sitk.GetImageFromArray(threshold_map),
        os.path.join(output_dir, 'threshold_map.nrrd')
    )
