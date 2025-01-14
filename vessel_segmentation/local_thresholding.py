import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import os
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class ROIParameters:
    """Parameters for ROI generation"""
    min_radius_cyl: float = 3.0  # mm
    min_radius_sphere: float = 4.0  # mm
    roi_multiplier: float = 1.5
    max_segment_length: int = 20  # voxels
    overlap: int = 5  # voxels
    min_vesselness: float = 0.05

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

def local_optimal_thresholding(binary_vessels, vesselness, centerlines, point_types, sigma_max, vessel_directions):
    """Perform local optimal thresholding around centerlines"""
    # Initialize parameters
    params = ROIParameters()
    image_shape = vesselness.shape
    
    # Initialize output
    final_vessels = np.zeros_like(binary_vessels)
    local_thresholds = np.zeros_like(vesselness)
    
    # Process segment points first
    segment_points = np.where(point_types == 2)
    print(f"Processing {len(segment_points[0])} segment points...")
    
    # Group connected segment points using vectorized operations
    from scipy.ndimage import label
    labeled_segments, num_segments = label(point_types == 2)
    print(f"Found {num_segments} connected segments")
    
    # Process each segment
    for segment_id in tqdm(range(1, num_segments + 1), desc="Processing segments", leave=True):
        segment_mask = labeled_segments == segment_id
        z, y, x = np.where(segment_mask)
        if len(z) == 0:
            continue
            
        # Convert points to list of tuples for split_segments
        segment_points = list(zip(z, y, x))
        
        # Split long segments into overlapping pieces
        sub_segments = split_segments(centerlines, segment_points, 
                                    max_length=params.max_segment_length, 
                                    overlap=params.overlap)
        
        # Process each sub-segment
        for sub_segment in sub_segments:
            sub_z, sub_y, sub_x = zip(*sub_segment)
            centerline_points = np.array(list(zip(sub_z, sub_y, sub_x)))
            
            # Calculate ROI parameters for this sub-segment
            sub_mask = np.zeros_like(binary_vessels, dtype=bool)
            sub_mask[sub_z, sub_y, sub_x] = True
            max_sigma = np.max(sigma_max[sub_mask])
            radius = params.roi_multiplier * max(max_sigma, params.min_radius_cyl)
            
            # Get sub-segment bounds with padding
            pad = int(np.ceil(radius))
            min_bounds = np.maximum([min(sub_z) - pad, min(sub_y) - pad, min(sub_x) - pad], 0)
            max_bounds = np.minimum([max(sub_z) + pad, max(sub_y) + pad, max(sub_x) + pad], np.array(image_shape) - 1)
            
            # Extract local region
            local_vesselness = vesselness[min_bounds[0]:max_bounds[0]+1,
                                        min_bounds[1]:max_bounds[1]+1,
                                        min_bounds[2]:max_bounds[2]+1]
            
            # Create grid of points
            local_shape = tuple(max_bounds - min_bounds + 1)
            zz, yy, xx = np.meshgrid(np.arange(local_shape[0]),
                                    np.arange(local_shape[1]),
                                    np.arange(local_shape[2]),
                                    indexing='ij')
            
            # Calculate average direction for this sub-segment
            sub_directions = vessel_directions[sub_z, sub_y, sub_x]
            # Ensure consistent direction (avoid sign flips)
            ref_direction = sub_directions[0]
            for i in range(1, len(sub_directions)):
                if np.dot(sub_directions[i], ref_direction) < 0:
                    sub_directions[i] = -sub_directions[i]
            direction = np.mean(sub_directions, axis=0)
            direction = direction / np.linalg.norm(direction)
            
            # Create precise cylindrical ROI
            points = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
            # Adjust centerline points to local coordinates
            local_centerline = centerline_points - min_bounds.reshape(1, 3)
            local_mask = create_cylindrical_roi(points, local_centerline, direction, radius)
            local_mask = local_mask.reshape(local_shape)
            
            # Calculate optimal threshold
            if np.any(local_mask):
                vesselness_roi = local_vesselness[local_mask]
                threshold = optimal_threshold(vesselness_roi)
                
                # Apply threshold
                grown_mask = (local_vesselness >= threshold) & local_mask & (local_vesselness >= params.min_vesselness)
                
                # Update results
                final_vessels[min_bounds[0]:max_bounds[0]+1,
                            min_bounds[1]:max_bounds[1]+1,
                            min_bounds[2]:max_bounds[2]+1] |= grown_mask
                local_thresholds[min_bounds[0]:max_bounds[0]+1,
                               min_bounds[1]:max_bounds[1]+1,
                               min_bounds[2]:max_bounds[2]+1][grown_mask] = threshold
    
    # Process special points (bifurcations and endpoints)
    special_points = np.where((point_types == 1) | (point_types == 3))
    print(f"Processing {len(special_points[0])} special points...")
    
    for i in tqdm(range(len(special_points[0])), desc="Processing special points", leave=True):
        z, y, x = special_points[0][i], special_points[1][i], special_points[2][i]
        
        # Skip endpoints connected to bifurcations
        if point_types[z,y,x] == 1:
            # Check 26-neighborhood for bifurcations using array operations
            z_slice = slice(max(0, z-1), min(z+2, image_shape[0]))
            y_slice = slice(max(0, y-1), min(y+2, image_shape[1]))
            x_slice = slice(max(0, x-1), min(x+2, image_shape[2]))
            if np.any(point_types[z_slice, y_slice, x_slice] == 3):
                continue
        
        # Calculate ROI parameters
        radius = params.roi_multiplier * max(sigma_max[z,y,x], params.min_radius_sphere)
        
        # Get ROI bounds with padding
        pad = int(np.ceil(radius))
        min_bounds = np.maximum([z-pad, y-pad, x-pad], 0)
        max_bounds = np.minimum([z+pad, y+pad, x+pad], np.array(image_shape) - 1)
        
        # Extract local region
        local_vesselness = vesselness[min_bounds[0]:max_bounds[0]+1,
                                    min_bounds[1]:max_bounds[1]+1,
                                    min_bounds[2]:max_bounds[2]+1]
        
        # Create spherical mask using vectorized operations
        local_shape = tuple(max_bounds - min_bounds + 1)
        zz, yy, xx = np.meshgrid(np.arange(local_shape[0]),
                                np.arange(local_shape[1]),
                                np.arange(local_shape[2]),
                                indexing='ij')
        
        local_center = np.array([z,y,x]) - min_bounds
        dist = np.sqrt((zz - local_center[0])**2 + 
                      (yy - local_center[1])**2 + 
                      (xx - local_center[2])**2)
        local_mask = dist <= radius
        
        # Calculate optimal threshold
        if np.any(local_mask):
            vesselness_roi = local_vesselness[local_mask]
            threshold = optimal_threshold(vesselness_roi)
            
            # Apply threshold
            grown_mask = (local_vesselness >= threshold) & local_mask & (local_vesselness >= params.min_vesselness)
            
            # Update results
            final_vessels[min_bounds[0]:max_bounds[0]+1,
                        min_bounds[1]:max_bounds[1]+1,
                        min_bounds[2]:max_bounds[2]+1] |= grown_mask
            local_thresholds[min_bounds[0]:max_bounds[0]+1,
                           min_bounds[1]:max_bounds[1]+1,
                           min_bounds[2]:max_bounds[2]+1][grown_mask] = threshold
    
    return final_vessels, local_thresholds

def split_segments(centerlines, point_types, max_length, overlap):
    """Split long segments into overlapping subsegments"""
    segments = []
    current_segment = []
    
    # Find segment points
    segment_points = zip(*np.where(point_types == 2))
    
    for point in segment_points:
        current_segment.append(point)
        
        if len(current_segment) >= max_length:
            # Add segment with overlap
            segments.append(current_segment)
            # Keep overlap points for next segment
            current_segment = current_segment[-overlap:]
    
    # Add remaining points
    if current_segment:
        segments.append(current_segment)
    
    return segments

def region_growing(seeds, vesselness, threshold, roi_mask, min_vesselness):
    """Perform region growing within ROI"""
    # Initialize segmentation
    segmented = np.zeros_like(roi_mask)
    
    # Initialize queue with seed points
    queue = list(seeds)
    visited = set()
    
    # Grow region
    while queue:
        z,y,x = queue.pop(0)
        if (z,y,x) in visited:
            continue
            
        visited.add((z,y,x))
        
        # Check if point is valid
        if (0 <= z < roi_mask.shape[0] and 
            0 <= y < roi_mask.shape[1] and 
            0 <= x < roi_mask.shape[2] and
            roi_mask[z,y,x] and
            vesselness[z,y,x] >= threshold and
            vesselness[z,y,x] >= min_vesselness):
            
            segmented[z,y,x] = True
            
            # Add neighbors to queue
            for dz in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        queue.append((z+dz, y+dy, x+dx))
    
    return segmented

def ensure_connectivity(mask):
    """Ensure vessel connectivity by removing small components"""
    from scipy.ndimage import label
    
    # Label connected components
    labeled, num = label(mask)
    
    if num == 0:
        return mask
        
    # Find largest component
    sizes = np.bincount(labeled.ravel())[1:]
    largest = np.argmax(sizes) + 1
    
    # Keep only the largest component
    mask = labeled == largest
    
    return mask

def optimal_threshold(intensities):
    """Calculate optimal threshold using Ridler's method"""
    if len(intensities) == 0:
        return 0
    
    # Initial threshold
    t = (np.min(intensities) + np.max(intensities)) / 2
    
    # Iterate until convergence
    epsilon = 1e-6
    max_iter = 100
    for _ in range(max_iter):
        # Split intensities
        below = intensities[intensities < t]
        above = intensities[intensities >= t]
        
        # Calculate means
        mean_below = np.mean(below) if len(below) > 0 else np.min(intensities)
        mean_above = np.mean(above) if len(above) > 0 else np.max(intensities)
        
        # Update threshold
        new_t = (mean_below + mean_above) / 2
        
        if abs(new_t - t) < epsilon:
            break
        t = new_t
    
    return t

def save_local_threshold_results(final_vessels, local_thresholds, output_dir):
    """Save local thresholding results and metadata
    
    Args:
        final_vessels: Binary mask of final vessel segmentation
        local_thresholds: Array of local threshold values used
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary mask
    sitk.WriteImage(
        sitk.GetImageFromArray(final_vessels.astype(np.uint8)),
        os.path.join(output_dir, 'final_vessels.nrrd')
    )
    
    # Save local thresholds
    sitk.WriteImage(
        sitk.GetImageFromArray(local_thresholds),
        os.path.join(output_dir, 'local_thresholds.nrrd')
    )
    
    # Save metadata
    metadata = {
        "parameters": {
            "min_vesselness": 0.05,
            "roi_multiplier": 1.5,
            "min_radius_cyl": 3.0,
            "min_radius_sphere": 4.0
        }
    }
    
    with open(os.path.join(output_dir, 'segmentation_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
