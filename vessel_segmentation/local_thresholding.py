import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import logging
from tqdm import tqdm
from scipy.ndimage import label
import gc

logger = logging.getLogger(__name__)

@dataclass
class ROIParameters:
    """Parameters for ROI generation and local thresholding
    
    Parameters:
        min_radius_cyl: float = 3.0 (mm)
            Minimum radius for cylindrical ROIs around vessel segments.
            - Smaller values: Can capture thinner vessels but may introduce noise
            - Larger values: More stable but might miss small vessels
            
        min_radius_sphere: float = 4.0 (mm)
            Minimum radius for spherical ROIs at bifurcations/endpoints.
            - Smaller values: Better detail at junctions but may fragment
            - Larger values: More stable junctions but may blur details
            
        roi_multiplier: float = 1.5
            Multiplier for ROI size relative to vessel scale (sigma_max).
            - Smaller values (<1.5): Tighter ROIs, less leakage but may miss vessel boundaries
            - Larger values (>1.5): Larger ROIs, better vessel recovery but potential leakage
            
        max_segment_length: int = 20 (voxels)
            Maximum length of vessel segments before splitting.
            - Smaller values: Better local adaptation but more computation
            - Larger values: Faster but may miss local variations
            
        overlap: int = 5 (voxels)
            Overlap between split segments to ensure continuity.
            - Smaller values: Less computation but potential discontinuities
            - Larger values: Better continuity but more computation
            
        min_vesselness: float = 0.05
            Minimum vesselness threshold to prevent leakage.
            - Smaller values (<0.05): More vessel recovery but potential leakage
            - Larger values (>0.05): Less leakage but may miss weak vessels
    """
    min_radius_cyl: float = 3.0  # mm
    min_radius_sphere: float = 4.0  # mm
    roi_multiplier: float = 1.5
    max_segment_length: int = 20  # voxels
    overlap: int = 5  # voxels
    min_vesselness: float = 0.05
    
    @classmethod
    def get_parameter_sets(cls):
        """Get different parameter sets to try"""
        return {
            'default': cls(),
            'aggressive': cls(
                min_vesselness=0.03,  # Lower threshold to include more vessels
                roi_multiplier=1.8,   # Larger ROI to capture more surrounding area
                min_radius_cyl=2.5,   # Smaller minimum radius to catch thin vessels
                min_radius_sphere=3.5  # Smaller sphere radius for detailed bifurcations
            ),
            'very_aggressive': cls(
                min_vesselness=0.02,  # Very low threshold for maximum vessel recovery
                roi_multiplier=2.0,   # Much larger ROI for maximum capture
                min_radius_cyl=2.0,   # Very small radius to catch smallest vessels
                min_radius_sphere=3.0  # Small sphere radius for detailed junctions
            ),
            'conservative': cls(
                min_vesselness=0.06,  # Higher threshold to prevent leakage
                roi_multiplier=1.3,   # Smaller ROI for tight vessel boundaries
                min_radius_cyl=3.5,   # Larger minimum radius for stability
                min_radius_sphere=4.5  # Larger sphere radius for stable junctions
            )
        }
    
    @classmethod
    def from_dict(cls, params_dict):
        """Create parameters from dictionary of overrides"""
        base_params = cls()
        for key, value in params_dict.items():
            if hasattr(base_params, key):
                setattr(base_params, key, value)
        return base_params

@dataclass
class QualityReport:
    """Quality metrics for vessel segmentation"""
    total_volume: int
    continuity_breaks: List[Tuple[int, int, int]]
    size_inconsistencies: List[Tuple[int, int, int]]
    airway_overlap: float

@dataclass
class VesselSegmentation:
    """Final vessel segmentation results"""
    binary_mask: np.ndarray
    centerlines: np.ndarray
    vessel_radii: np.ndarray
    quality_metrics: QualityReport
    processing_parameters: ROIParameters

def extract_roi_region(center, radius, image, image_shape):
    """Extract a local region around a point"""
    pad = int(np.ceil(radius))
    min_bounds = np.maximum(np.array(center) - pad, 0)
    max_bounds = np.minimum(np.array(center) + pad, np.array(image_shape) - 1)
    
    return image[min_bounds[0]:max_bounds[0]+1,
                min_bounds[1]:max_bounds[1]+1,
                min_bounds[2]:max_bounds[2]+1], (min_bounds, max_bounds)

def group_segments(centerlines, point_types):
    """Group connected segment points together"""
    from scipy.ndimage import label
    
    # Get segment points
    segment_mask = (point_types == 2)
    
    # Label connected components
    labeled, num = label(segment_mask)
    
    # Extract groups
    groups = []
    for i in range(1, num+1):
        points = np.array(np.where(labeled == i)).T
        if len(points) > 0:
            groups.append(points)
    
    return groups

def calculate_average_direction(points, vessel_directions):
    """Calculate average direction vector for a group of points"""
    directions = []
    for point in points:
        z,y,x = point
        direction = vessel_directions[z,y,x]
        # Ensure consistent direction (avoid sign flips)
        if len(directions) > 0:
            if np.dot(direction, directions[0]) < 0:
                direction = -direction
        directions.append(direction)
    
    # Average directions
    avg_direction = np.mean(directions, axis=0)
    norm = np.linalg.norm(avg_direction)
    if norm > 0:
        avg_direction = avg_direction / norm
    
    return avg_direction

def create_cylindrical_roi(points, centerline_points, direction, radius):
    """Create precise cylindrical ROI following vessel direction using vectorized operations
    
    Args:
        points: (N,3) array of points to test
        centerline_points: (M,3) array of centerline points
        direction: (3,) normalized direction vector
        radius: cylinder radius
        
    Returns:
        Boolean mask indicating which points are inside the cylinder
    """
    # Convert centerline points to array
    centerline = np.array(centerline_points)
    
    # Calculate all segments at once
    segments = centerline[1:] - centerline[:-1]  # Shape: (M-1, 3)
    segment_lengths_sq = np.sum(segments**2, axis=1)  # Shape: (M-1,)
    
    # Expand points for broadcasting
    points_expanded = points[:, np.newaxis, :]  # Shape: (N, 1, 3)
    starts = centerline[:-1]  # Shape: (M-1, 3)
    
    # Calculate relative positions for all points and segments at once
    points_relative = points_expanded - starts  # Shape: (N, M-1, 3)
    
    # Vectorized projection calculation
    # Dot product of relative positions with segments
    dots = np.sum(points_relative * segments, axis=2)  # Shape: (N, M-1)
    
    # Calculate projection parameters
    t = dots / segment_lengths_sq  # Shape: (N, M-1)
    
    # Find valid projections (0 ≤ t ≤ 1)
    valid_mask = (t >= 0) & (t <= 1)  # Shape: (N, M-1)
    
    if not np.any(valid_mask):
        return np.zeros(len(points), dtype=bool)
    
    # Calculate projection points for valid projections
    t_valid = np.where(valid_mask, t, 0)
    proj_points = starts + t_valid[..., np.newaxis] * segments  # Shape: (N, M-1, 3)
    
    # Calculate distances from points to their projections
    diff = points_expanded - proj_points  # Shape: (N, M-1, 3)
    distances = np.sqrt(np.sum(diff * diff, axis=2))  # Shape: (N, M-1)
    
    # Set invalid distances to infinity
    distances = np.where(valid_mask, distances, np.inf)
    
    # Find minimum distance for each point
    min_distances = np.min(distances, axis=1)  # Shape: (N,)
    
    # Points are inside if minimum distance is less than radius
    return min_distances <= radius

def local_optimal_thresholding(binary_vessels, vesselness, centerlines, point_types, sigma_max, vessel_directions, parameter_set='default'):
    """Perform local optimal thresholding around centerlines using vectorized operations
    
    Args:
        binary_vessels: Initial binary vessel mask
        vesselness: Vesselness measure
        centerlines: Centerline points
        point_types: Point type labels
        sigma_max: Maximum scale at each point
        vessel_directions: Vessel directions
        parameter_set: Which parameter set to use ('default', 'aggressive', 'very_aggressive', 'conservative')
    """
    # Initialize parameters
    params = ROIParameters.get_parameter_sets()[parameter_set]
    image_shape = vesselness.shape
    
    # Initialize output arrays
    final_vessels = np.zeros_like(binary_vessels)
    local_thresholds = np.zeros_like(vesselness)
    
    # Process segment points first - use vectorized operations
    labeled_segments, num_segments = label(point_types == 2)
    print(f"Found {num_segments} connected segments")
    
    # Pre-compute meshgrid for ROI creation
    z_base, y_base, x_base = np.meshgrid(np.arange(image_shape[0]),
                                        np.arange(image_shape[1]),
                                        np.arange(image_shape[2]),
                                        indexing='ij')
    
    # Process segments in parallel where possible
    for segment_id in tqdm(range(1, num_segments + 1), desc="Processing segments", leave=True):
        segment_mask = labeled_segments == segment_id
        z, y, x = np.where(segment_mask)
        if len(z) == 0:
            continue
            
        # Split into sub-segments efficiently
        segment_points = np.column_stack([z, y, x])
        sub_segments = split_segments_vectorized(segment_points, 
                                               max_length=params.max_segment_length,
                                               overlap=params.overlap)
        
        # Process each sub-segment
        for sub_segment in sub_segments:
            # Calculate ROI parameters efficiently
            sub_mask = np.zeros_like(binary_vessels, dtype=bool)
            sub_mask[sub_segment[:,0], sub_segment[:,1], sub_segment[:,2]] = True
            max_sigma = np.max(sigma_max[sub_mask])
            radius = params.roi_multiplier * max(max_sigma, params.min_radius_cyl)
            
            # Get bounds with padding
            pad = int(np.ceil(radius))
            min_bounds = np.maximum([np.min(sub_segment[:,0]) - pad,
                                   np.min(sub_segment[:,1]) - pad,
                                   np.min(sub_segment[:,2]) - pad], 0)
            max_bounds = np.minimum([np.max(sub_segment[:,0]) + pad,
                                   np.max(sub_segment[:,1]) + pad,
                                   np.max(sub_segment[:,2]) + pad], 
                                  np.array(image_shape) - 1)
            
            # Extract local region efficiently
            local_vesselness = vesselness[min_bounds[0]:max_bounds[0]+1,
                                        min_bounds[1]:max_bounds[1]+1,
                                        min_bounds[2]:max_bounds[2]+1]
            
            # Calculate average direction vectorized
            sub_directions = vessel_directions[sub_segment[:,0], sub_segment[:,1], sub_segment[:,2]]
            ref_direction = sub_directions[0]
            # Fix direction signs efficiently
            sign_flips = np.sign(np.sum(sub_directions * ref_direction, axis=1))
            sub_directions *= sign_flips[:,np.newaxis]
            direction = np.mean(sub_directions, axis=0)
            direction /= np.linalg.norm(direction)
            
            # Create cylindrical ROI efficiently
            local_shape = tuple(max_bounds - min_bounds + 1)
            points = np.column_stack([
                z_base[min_bounds[0]:max_bounds[0]+1,
                      min_bounds[1]:max_bounds[1]+1,
                      min_bounds[2]:max_bounds[2]+1].ravel(),
                y_base[min_bounds[0]:max_bounds[0]+1,
                      min_bounds[1]:max_bounds[1]+1,
                      min_bounds[2]:max_bounds[2]+1].ravel(),
                x_base[min_bounds[0]:max_bounds[0]+1,
                      min_bounds[1]:max_bounds[1]+1,
                      min_bounds[2]:max_bounds[2]+1].ravel()
            ])
            local_centerline = sub_segment - min_bounds.reshape(1, 3)
            local_mask = create_cylindrical_roi_vectorized(points, local_centerline, direction, radius)
            local_mask = local_mask.reshape(local_shape)
            
            # Calculate and apply threshold efficiently
            if np.any(local_mask):
                vesselness_roi = local_vesselness[local_mask]
                threshold = optimal_threshold_vectorized(vesselness_roi)
                
                # Apply threshold with vectorized operations
                grown_mask = (local_vesselness >= threshold) & local_mask & (local_vesselness >= params.min_vesselness)
                
                # Update results efficiently
                final_vessels[min_bounds[0]:max_bounds[0]+1,
                            min_bounds[1]:max_bounds[1]+1,
                            min_bounds[2]:max_bounds[2]+1] |= grown_mask
                local_thresholds[min_bounds[0]:max_bounds[0]+1,
                               min_bounds[1]:max_bounds[1]+1,
                               min_bounds[2]:max_bounds[2]+1][grown_mask] = threshold
    
    # Process special points (bifurcations and endpoints) with vectorized operations
    special_mask = (point_types == 1) | (point_types == 3)
    special_points = np.argwhere(special_mask)
    print(f"Processing {len(special_points)} special points...")
    
    # Process special points in batches
    batch_size = 100
    for i in tqdm(range(0, len(special_points), batch_size), desc="Processing special points", leave=True):
        batch = special_points[i:i+batch_size]
        
        # Skip endpoints connected to bifurcations efficiently
        endpoint_mask = point_types[batch[:,0], batch[:,1], batch[:,2]] == 1
        if np.any(endpoint_mask):
            endpoint_batch = batch[endpoint_mask]
            for z, y, x in endpoint_batch:
                z_slice = slice(max(0, z-1), min(z+2, image_shape[0]))
                y_slice = slice(max(0, y-1), min(y+2, image_shape[1]))
                x_slice = slice(max(0, x-1), min(x+2, image_shape[2]))
                if np.any(point_types[z_slice, y_slice, x_slice] == 3):
                    continue
                
                # Process valid endpoint
                radius = params.roi_multiplier * max(sigma_max[z,y,x], params.min_radius_sphere)
                process_special_point(z, y, x, radius, vesselness, final_vessels, 
                                   local_thresholds, params.min_vesselness, image_shape)
        
        # Process remaining special points
        non_endpoint_mask = ~endpoint_mask
        if np.any(non_endpoint_mask):
            non_endpoint_batch = batch[non_endpoint_mask]
            for z, y, x in non_endpoint_batch:
                radius = params.roi_multiplier * max(sigma_max[z,y,x], params.min_radius_sphere)
                process_special_point(z, y, x, radius, vesselness, final_vessels, 
                                   local_thresholds, params.min_vesselness, image_shape)
    
    return final_vessels, local_thresholds

def process_special_point(z, y, x, radius, vesselness, final_vessels, local_thresholds, min_vesselness, image_shape):
    """Process a single special point efficiently"""
    # Get ROI bounds with padding
    pad = int(np.ceil(radius))
    min_bounds = np.maximum([z-pad, y-pad, x-pad], 0)
    max_bounds = np.minimum([z+pad, y+pad, x+pad], np.array(image_shape) - 1)
    
    # Extract local region
    local_vesselness = vesselness[min_bounds[0]:max_bounds[0]+1,
                                min_bounds[1]:max_bounds[1]+1,
                                min_bounds[2]:max_bounds[2]+1]
    
    # Create spherical mask efficiently
    local_shape = tuple(max_bounds - min_bounds + 1)
    local_center = np.array([z,y,x]) - min_bounds
    zz, yy, xx = np.meshgrid(np.arange(local_shape[0]),
                            np.arange(local_shape[1]),
                            np.arange(local_shape[2]),
                            indexing='ij')
    dist = np.sqrt((zz - local_center[0])**2 + 
                   (yy - local_center[1])**2 + 
                   (xx - local_center[2])**2)
    local_mask = dist <= radius
    
    # Calculate and apply threshold
    if np.any(local_mask):
        vesselness_roi = local_vesselness[local_mask]
        threshold = optimal_threshold_vectorized(vesselness_roi)
        
        # Apply threshold with vectorized operations
        grown_mask = (local_vesselness >= threshold) & local_mask & (local_vesselness >= min_vesselness)
        
        # Update results efficiently
        final_vessels[min_bounds[0]:max_bounds[0]+1,
                    min_bounds[1]:max_bounds[1]+1,
                    min_bounds[2]:max_bounds[2]+1] |= grown_mask
        local_thresholds[min_bounds[0]:max_bounds[0]+1,
                       min_bounds[1]:max_bounds[1]+1,
                       min_bounds[2]:max_bounds[2]+1][grown_mask] = threshold

def split_segments_vectorized(points, max_length, overlap):
    """Split segments into overlapping subsegments using vectorized operations"""
    if len(points) <= max_length:
        return [points]
        
    segments = []
    start_idx = 0
    
    while start_idx < len(points):
        end_idx = min(start_idx + max_length, len(points))
        segments.append(points[start_idx:end_idx])
        start_idx = end_idx - overlap
        
    return segments

def optimal_threshold_vectorized(intensities):
    """Vectorized implementation of Ridler's optimal threshold method"""
    if len(intensities) == 0:
        return 0
    
    # Initial threshold
    t = (np.min(intensities) + np.max(intensities)) / 2
    
    # Iterate until convergence
    epsilon = 1e-6
    max_iter = 100
    for _ in range(max_iter):
        # Split intensities efficiently
        below = intensities[intensities < t]
        above = intensities[intensities >= t]
        
        # Calculate means vectorized
        mean_below = np.mean(below) if len(below) > 0 else np.min(intensities)
        mean_above = np.mean(above) if len(above) > 0 else np.max(intensities)
        
        # Update threshold
        new_t = (mean_below + mean_above) / 2
        
        if abs(new_t - t) < epsilon:
            break
        t = new_t
    
    return t

def create_cylindrical_roi_vectorized(points, centerline, direction, radius):
    """Create cylindrical ROI using vectorized operations"""
    # Project points onto centerline direction
    centerline_mean = np.mean(centerline, axis=0)
    projected_distances = np.abs(np.dot(points - centerline_mean, direction))
    
    # Find closest point on centerline for each point
    distances = np.zeros(len(points))
    for center in centerline:
        point_distances = np.linalg.norm(points - center, axis=1)
        distances = np.minimum(distances, point_distances)
    
    # Points within cylinder have:
    # 1. Distance to centerline <= radius
    # 2. Projected distance <= half cylinder length
    cylinder_length = np.linalg.norm(centerline[-1] - centerline[0])
    return (distances <= radius) & (projected_distances <= cylinder_length/2)

def save_local_threshold_results(final_vessels, local_thresholds, output_dir, parameter_set='default', custom_params=None):
    """Save local thresholding results and metadata
    
    Args:
        final_vessels: Binary mask of final vessel segmentation
        local_thresholds: Array of local threshold values used
        output_dir: Directory to save results
        parameter_set: Name of parameter set used ('default', 'aggressive', etc.)
        custom_params: Dictionary of custom parameter values if used
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename suffix based on parameters
    if custom_params:
        # Create a compact representation of custom parameters
        param_str = '_'.join(f"{k}{v}" for k, v in sorted(custom_params.items()))
        suffix = f"_custom_{param_str}"
    else:
        suffix = f"_{parameter_set}" if parameter_set != 'default' else ''
    
    # Save binary mask with parameter info
    sitk.WriteImage(
        sitk.GetImageFromArray(final_vessels.astype(np.uint8)),
        os.path.join(output_dir, f'final_vessels{suffix}.nrrd')
    )
    
    # Save local thresholds with parameter info
    sitk.WriteImage(
        sitk.GetImageFromArray(local_thresholds),
        os.path.join(output_dir, f'local_thresholds{suffix}.nrrd')
    )
    
    # Get parameters used
    if custom_params:
        params = ROIParameters.from_dict(custom_params)
    else:
        params = ROIParameters.get_parameter_sets()[parameter_set]
    
    # Save metadata
    metadata = {
        "parameters": {
            "min_vesselness": params.min_vesselness,
            "roi_multiplier": params.roi_multiplier,
            "min_radius_cyl": params.min_radius_cyl,
            "min_radius_sphere": params.min_radius_sphere,
            "max_segment_length": params.max_segment_length,
            "overlap": params.overlap,
            "parameter_set": parameter_set if not custom_params else "custom"
        }
    }
    
    with open(os.path.join(output_dir, f'segmentation_metadata{suffix}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
