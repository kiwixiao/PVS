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
import vtk

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
    """Perform local optimal thresholding around centerlines using optimized vectorized operations
    
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
            max_sigma = np.max(sigma_max[sub_segment[:,0], sub_segment[:,1], sub_segment[:,2]])
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
            
            # Extract local region efficiently using views instead of copies where possible
            local_vesselness = vesselness[min_bounds[0]:max_bounds[0]+1,
                                        min_bounds[1]:max_bounds[1]+1,
                                        min_bounds[2]:max_bounds[2]+1]
            
            # Calculate average direction vectorized
            sub_directions = vessel_directions[sub_segment[:,0], sub_segment[:,1], sub_segment[:,2]]
            ref_direction = sub_directions[0]
            # Fix direction signs efficiently using broadcasting
            sign_flips = np.sign(np.sum(sub_directions * ref_direction, axis=1))
            sub_directions *= sign_flips[:,np.newaxis]
            direction = np.mean(sub_directions, axis=0)
            direction /= np.linalg.norm(direction)
            
            # Create cylindrical ROI efficiently
            local_shape = tuple(max_bounds - min_bounds + 1)
            
            # Generate grid points more efficiently
            z_grid, y_grid, x_grid = np.meshgrid(
                np.arange(min_bounds[0], max_bounds[0] + 1),
                np.arange(min_bounds[1], max_bounds[1] + 1),
                np.arange(min_bounds[2], max_bounds[2] + 1),
                indexing='ij'
            )
            points = np.column_stack([z_grid.ravel(), y_grid.ravel(), x_grid.ravel()])
            
            # Create ROI mask
            local_centerline = sub_segment
            local_mask = create_cylindrical_roi_vectorized(points, local_centerline, direction, radius)
            
            if len(local_mask) > 0:
                local_mask = local_mask.reshape(local_shape)
                
                # Calculate and apply threshold efficiently
                if np.any(local_mask):
                    vesselness_roi = local_vesselness[local_mask]
                    threshold = optimal_threshold_vectorized(vesselness_roi)
                    
                    # Apply threshold with vectorized operations
                    grown_mask = (local_vesselness >= threshold) & local_mask & (local_vesselness >= params.min_vesselness)
                    
                    # Update results efficiently using views
                    final_vessels[min_bounds[0]:max_bounds[0]+1,
                                min_bounds[1]:max_bounds[1]+1,
                                min_bounds[2]:max_bounds[2]+1] |= grown_mask
                    local_thresholds[min_bounds[0]:max_bounds[0]+1,
                                   min_bounds[1]:max_bounds[1]+1,
                                   min_bounds[2]:max_bounds[2]+1][grown_mask] = threshold
            
            # Clean up explicitly
            del local_vesselness, local_mask
            if segment_id % 10 == 0:
                gc.collect()
    
    # Process special points (bifurcations and endpoints) with vectorized operations
    special_mask = (point_types == 1) | (point_types == 3)
    special_points = np.argwhere(special_mask)
    print(f"Processing {len(special_points)} special points...")
    
    # Process special points in larger batches
    batch_size = 200  # Increased batch size
    for i in tqdm(range(0, len(special_points), batch_size), desc="Processing special points", leave=True):
        batch = special_points[i:i+batch_size]
        
        # Process endpoints and bifurcations separately for efficiency
        endpoint_mask = point_types[batch[:,0], batch[:,1], batch[:,2]] == 1
        
        # Process valid endpoints (not connected to bifurcations)
        if np.any(endpoint_mask):
            endpoint_batch = batch[endpoint_mask]
            valid_endpoints = []
            
            # Vectorized check for bifurcation connections
            for z, y, x in endpoint_batch:
                z_slice = slice(max(0, z-1), min(z+2, image_shape[0]))
                y_slice = slice(max(0, y-1), min(y+2, image_shape[1]))
                x_slice = slice(max(0, x-1), min(x+2, image_shape[2]))
                if not np.any(point_types[z_slice, y_slice, x_slice] == 3):
                    valid_endpoints.append((z,y,x))
            
            # Process valid endpoints in parallel
            if valid_endpoints:
                for z, y, x in valid_endpoints:
                    radius = params.roi_multiplier * max(sigma_max[z,y,x], params.min_radius_sphere)
                    process_special_point(z, y, x, radius, vesselness, final_vessels, 
                                       local_thresholds, params.min_vesselness, image_shape)
        
        # Process remaining special points (bifurcations and invalid endpoints)
        non_endpoint_mask = ~endpoint_mask
        if np.any(non_endpoint_mask):
            non_endpoint_batch = batch[non_endpoint_mask]
            for z, y, x in non_endpoint_batch:
                radius = params.roi_multiplier * max(sigma_max[z,y,x], params.min_radius_sphere)
                process_special_point(z, y, x, radius, vesselness, final_vessels, 
                                   local_thresholds, params.min_vesselness, image_shape)
        
        if i % (batch_size * 5) == 0:
            gc.collect()
    
    return final_vessels, local_thresholds

def process_special_point(z, y, x, radius, vesselness, final_vessels, local_thresholds, min_vesselness, image_shape):
    """Process a special point (endpoint or bifurcation) efficiently
    
    Args:
        z, y, x: Coordinates of special point
        radius: Sphere radius
        vesselness: Vesselness measure array
        final_vessels: Output binary vessel mask
        local_thresholds: Output threshold values
        min_vesselness: Minimum vesselness value
        image_shape: Shape of the image arrays
    """
    # Calculate bounds efficiently
    pad = int(np.ceil(radius))
    min_bounds = np.maximum([z - pad, y - pad, x - pad], 0)
    max_bounds = np.minimum([z + pad, y + pad, x + pad], np.array(image_shape) - 1)
    
    # Extract local region using views
    local_vesselness = vesselness[min_bounds[0]:max_bounds[0]+1,
                                min_bounds[1]:max_bounds[1]+1,
                                min_bounds[2]:max_bounds[2]+1]
    
    # Create spherical mask efficiently
    local_shape = tuple(max_bounds - min_bounds + 1)
    center = np.array([z, y, x]) - min_bounds
    
    z_grid, y_grid, x_grid = np.meshgrid(np.arange(local_shape[0]),
                                        np.arange(local_shape[1]),
                                        np.arange(local_shape[2]),
                                        indexing='ij',
                                        sparse=True)
    
    # Calculate distances efficiently using broadcasting
    distances = np.sqrt((z_grid - center[0])**2 +
                       (y_grid - center[1])**2 +
                       (x_grid - center[2])**2)
    local_mask = distances <= radius
    
    # Apply threshold efficiently
    if np.any(local_mask):
        vesselness_roi = local_vesselness[local_mask]
        threshold = optimal_threshold_vectorized(vesselness_roi)
        
        # Update results using vectorized operations
        grown_mask = (local_vesselness >= threshold) & local_mask & (local_vesselness >= min_vesselness)
        
        final_vessels[min_bounds[0]:max_bounds[0]+1,
                     min_bounds[1]:max_bounds[1]+1,
                     min_bounds[2]:max_bounds[2]+1] |= grown_mask
        local_thresholds[min_bounds[0]:max_bounds[0]+1,
                        min_bounds[1]:max_bounds[1]+1,
                        min_bounds[2]:max_bounds[2]+1][grown_mask] = threshold

def split_segments_vectorized(points, max_length=50, overlap=10):
    """Split segment points into overlapping sub-segments efficiently
    
    Args:
        points: Nx3 array of point coordinates
        max_length: Maximum length of each sub-segment
        overlap: Number of points to overlap between segments
    """
    if len(points) <= max_length:
        return [points]
        
    # Calculate cumulative distances efficiently
    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(distances)])
    
    # Split based on cumulative distance
    total_dist = cum_dist[-1]
    n_segments = max(1, int(np.ceil(total_dist / (max_length - overlap))))
    segment_length = total_dist / n_segments + overlap
    
    sub_segments = []
    for i in range(n_segments):
        start_dist = max(0, i * segment_length - overlap)
        end_dist = min(total_dist, (i + 1) * segment_length)
        
        # Find points within distance range using vectorized operations
        mask = (cum_dist >= start_dist) & (cum_dist <= end_dist)
        sub_segment = points[mask]
        
        if len(sub_segment) > 0:
            sub_segments.append(sub_segment)
            
    return sub_segments

def create_cylindrical_roi_vectorized(points, centerline, direction, radius):
    """Create cylindrical ROI around centerline efficiently using vectorized operations
    
    Args:
        points: Nx3 array of point coordinates to check
        centerline: Mx3 array of centerline points
        direction: 3D direction vector
        radius: Cylinder radius
    """
    if len(points) == 0 or len(centerline) == 0:
        return np.array([], dtype=bool)
        
    # Ensure arrays are float for numerical stability
    points = points.astype(np.float32)
    centerline = centerline.astype(np.float32)
    direction = direction.astype(np.float32)
    
    # Get cylinder center and length
    center = np.mean(centerline, axis=0)
    length = np.linalg.norm(centerline[-1] - centerline[0])
    
    # Calculate relative positions
    relative_points = points - center
    
    # Project points onto cylinder axis
    proj_lengths = np.dot(relative_points, direction)
    
    # Calculate perpendicular distances
    proj_points = np.outer(proj_lengths, direction)
    perp_vectors = relative_points - proj_points
    perp_distances = np.sqrt(np.sum(perp_vectors * perp_vectors, axis=1))
    
    # Calculate valid range along axis
    half_length = length / 2 + radius  # Add radius to account for endpoints
    
    # Points are inside if:
    # 1. Perpendicular distance <= radius
    # 2. Projection length within cylinder extent
    return (perp_distances <= radius) & (np.abs(proj_lengths) <= half_length)

def optimal_threshold_vectorized(vesselness_values):
    """Calculate optimal threshold efficiently using vectorized operations
    
    Args:
        vesselness_values: Array of vesselness values to threshold
    """
    if len(vesselness_values) == 0:
        return 0.0
        
    # Use vectorized operations for histogram
    hist, bin_edges = np.histogram(vesselness_values, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peak efficiently
    peak_idx = np.argmax(hist)
    peak_val = bin_centers[peak_idx]
    
    # Calculate threshold using vectorized operations
    above_peak = vesselness_values[vesselness_values > peak_val]
    if len(above_peak) > 0:
        threshold = peak_val + 0.5 * (np.mean(above_peak) - peak_val)
    else:
        threshold = peak_val
        
    return threshold

def save_local_threshold_results(final_vessels, local_thresholds, output_dir, point_types, parameter_set='default', custom_params=None):
    """Save local thresholding results and metadata
    
    Args:
        final_vessels: Binary mask of final vessel segmentation
        local_thresholds: Array of local threshold values used
        output_dir: Directory to save results
        point_types: Array indicating point types (1=endpoint, 2=segment, 3=bifurcation)
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
    
    # Save special points information
    special_points = {
        'endpoints': np.argwhere(point_types == 1).tolist(),
        'bifurcations': np.argwhere(point_types == 3).tolist(),
        'segments': np.argwhere(point_types == 2).tolist()
    }
    
    # Count special points
    point_counts = {
        'num_endpoints': len(special_points['endpoints']),
        'num_bifurcations': len(special_points['bifurcations']),
        'num_segment_points': len(special_points['segments'])
    }
    
    # Get parameters used
    if custom_params:
        params = ROIParameters.from_dict(custom_params)
    else:
        params = ROIParameters.get_parameter_sets()[parameter_set]
    
    # Save metadata with special points info
    metadata = {
        "parameters": {
            "min_vesselness": params.min_vesselness,
            "roi_multiplier": params.roi_multiplier,
            "min_radius_cyl": params.min_radius_cyl,
            "min_radius_sphere": params.min_radius_sphere,
            "max_segment_length": params.max_segment_length,
            "overlap": params.overlap,
            "parameter_set": parameter_set if not custom_params else "custom"
        },
        "special_points": special_points,
        "point_counts": point_counts
    }
    
    with open(os.path.join(output_dir, f'segmentation_metadata{suffix}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save point types as NRRD for visualization
    sitk.WriteImage(
        sitk.GetImageFromArray(point_types.astype(np.uint8)),
        os.path.join(output_dir, f'point_types{suffix}.nrrd')
    )

def save_centerlines_to_vtk(centerlines, point_types, sigma_max, vessel_directions, output_dir, suffix=''):
    """Save centerlines and vessel attributes as VTK polydata
    
    Args:
        centerlines: Binary centerline mask
        point_types: Point type labels (1=endpoint, 2=segment, 3=bifurcation)
        sigma_max: Maximum scale at each point (related to vessel radius)
        vessel_directions: Vessel direction vectors
        output_dir: Directory to save results
        suffix: Optional suffix for output filename
    """
    # Create output filename
    if suffix:
        vtk_filename = f'centerlines_{suffix}.vtp'
    else:
        vtk_filename = 'centerlines.vtp'
    output_path = os.path.join(output_dir, vtk_filename)
    
    # Get centerline point coordinates
    points = np.argwhere(centerlines > 0)
    
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point[2], point[1], point[0])  # Convert to x,y,z order
        
    # Create point data arrays
    point_type_array = vtk.vtkIntArray()
    point_type_array.SetName("PointType")
    
    radius_array = vtk.vtkFloatArray()
    radius_array.SetName("Radius")
    
    direction_array = vtk.vtkFloatArray()
    direction_array.SetName("Direction")
    direction_array.SetNumberOfComponents(3)
    
    # Fill point data
    for point in points:
        z, y, x = point
        point_type_array.InsertNextValue(point_types[z, y, x])
        radius_array.InsertNextValue(sigma_max[z, y, x] * 2 * np.sqrt(2))  # Convert scale to approximate diameter
        direction = vessel_directions[z, y, x]
        direction_array.InsertNextTuple3(direction[0], direction[1], direction[2])
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    
    # Add point data
    polydata.GetPointData().AddArray(point_type_array)
    polydata.GetPointData().AddArray(radius_array)
    polydata.GetPointData().AddArray(direction_array)
    
    # Create lines connecting points
    lines = vtk.vtkCellArray()
    
    # Label connected components to identify separate segments
    labeled_segments, num_segments = label(centerlines)
    
    # Process each segment
    for segment_id in range(1, num_segments + 1):
        segment_points = np.argwhere(labeled_segments == segment_id)
        
        # Skip single points
        if len(segment_points) < 2:
            continue
            
        # Find endpoints and bifurcations in this segment
        special_points = segment_points[
            np.where(
                (point_types[segment_points[:,0], segment_points[:,1], segment_points[:,2]] == 1) |
                (point_types[segment_points[:,0], segment_points[:,1], segment_points[:,2]] == 3)
            )[0]
        ]
        
        if len(special_points) >= 2:
            # Create a line from each special point to its closest neighbor
            for start_point in special_points:
                # Find closest point that's not the start point
                other_points = segment_points[~np.all(segment_points == start_point, axis=1)]
                distances = np.sqrt(np.sum((other_points - start_point)**2, axis=1))
                closest_idx = np.argmin(distances)
                end_point = other_points[closest_idx]
                
                # Find point indices in the full points array
                start_idx = np.where(np.all(points == start_point, axis=1))[0][0]
                end_idx = np.where(np.all(points == end_point, axis=1))[0][0]
                
                # Create line
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, start_idx)
                line.GetPointIds().SetId(1, end_idx)
                lines.InsertNextCell(line)
    
    polydata.SetLines(lines)
    
    # Write to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"Saved centerlines as VTK polydata to: {output_path}")
    print(f"- Number of points: {len(points)}")
    print(f"- Number of segments: {num_segments}")
    print(f"Point types in VTK:")
    print("  1: Endpoint")
    print("  2: Segment point")
    print("  3: Bifurcation point")
    print("Additional attributes:")
    print("- Radius: Local vessel radius in mm")
    print("- Direction: Vessel direction vector (x,y,z)")
