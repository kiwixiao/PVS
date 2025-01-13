import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import os
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import logging

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

def local_optimal_thresholding(binary_vessels, vesselness, centerlines, point_types, sigma_max):
    """Perform local optimal thresholding around centerlines
    
    Args:
        binary_vessels: Initial binary vessel mask
        vesselness: Vesselness measure
        centerlines: Centerline points
        point_types: Point classification (0=isolated, 1=endpoint, 2=segment, 3=bifurcation)
        sigma_max: Scale of maximum vesselness response
    """
    # Initialize parameters
    params = ROIParameters()
    
    # Split long segments and create ROIs
    segments = split_segments(centerlines, point_types, params.max_segment_length, params.overlap)
    
    # Process each segment
    final_vessels = np.zeros_like(binary_vessels)
    local_thresholds = np.zeros_like(vesselness)
    
    # Process cylindrical ROIs (segments)
    segment_points = np.where(point_types == 2)
    for segment in segments:
        # Calculate ROI parameters
        radius = params.roi_multiplier * max(sigma_max[segment[0]], params.min_radius_cyl)
        direction = calculate_segment_direction(segment, centerlines)
        
        # Create ROI mask
        roi_mask = create_cylindrical_roi(segment, radius, direction)
        
        # Calculate optimal threshold
        threshold = optimal_threshold(vesselness[roi_mask])
        
        # Apply region growing
        segment_mask = region_growing(
            segment,
            vesselness,
            threshold,
            roi_mask,
            params.min_vesselness
        )
        
        # Update results
        final_vessels |= segment_mask
        local_thresholds[roi_mask] = threshold
    
    # Process spherical ROIs (bifurcations and endpoints)
    special_points = np.where((point_types == 1) | (point_types == 3))
    for z,y,x in zip(*special_points):
        # Calculate ROI parameters
        radius = params.roi_multiplier * max(sigma_max[z,y,x], params.min_radius_sphere)
        
        # Create ROI mask
        roi_mask = create_spherical_roi((z,y,x), radius)
        
        # Calculate optimal threshold
        threshold = optimal_threshold(vesselness[roi_mask])
        
        # Apply region growing
        point_mask = region_growing(
            [(z,y,x)],
            vesselness,
            threshold,
            roi_mask,
            params.min_vesselness
        )
        
        # Update results
        final_vessels |= point_mask
        local_thresholds[roi_mask] = threshold
    
    # Final processing
    final_vessels = ensure_connectivity(final_vessels)
    
    # Generate quality report
    quality_report = generate_quality_report(final_vessels, centerlines, sigma_max)
    
    # Create final segmentation object
    segmentation = VesselSegmentation(
        binary_mask=final_vessels,
        centerlines=centerlines,
        vessel_radii=sigma_max,
        quality_metrics=quality_report,
        processing_parameters=params
    )
    
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

def calculate_segment_direction(segment, centerlines):
    """Calculate average direction vector for segment"""
    if len(segment) < 2:
        return np.array([0,0,1])
    
    # Calculate direction from first to last point
    start = np.array(segment[0])
    end = np.array(segment[-1])
    direction = end - start
    
    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return direction

def create_cylindrical_roi(segment, radius, direction):
    """Create cylindrical ROI mask"""
    # Get segment bounds
    points = np.array(segment)
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    
    # Create mask
    shape = tuple(max_bounds - min_bounds + 2*int(np.ceil(radius)))
    mask = np.zeros(shape, dtype=bool)
    
    # Create cylinder
    center = (shape[0]//2, shape[1]//2, shape[2]//2)
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                point = np.array([z,y,x]) - center
                # Project point onto direction vector
                proj = np.dot(point, direction) * direction
                # Calculate distance to line
                dist = np.linalg.norm(point - proj)
                if dist <= radius:
                    mask[z,y,x] = True
    
    return mask

def create_spherical_roi(center, radius):
    """Create spherical ROI mask"""
    # Create mask
    shape = tuple(2*int(np.ceil(radius)) + 1 for _ in range(3))
    mask = np.zeros(shape, dtype=bool)
    
    # Create sphere
    center_point = tuple(s//2 for s in shape)
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                dist = np.sqrt((z-center_point[0])**2 + 
                             (y-center_point[1])**2 + 
                             (x-center_point[2])**2)
                if dist <= radius:
                    mask[z,y,x] = True
    
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

def region_growing(seeds, vesselness, threshold, roi_mask, min_vesselness):
    """Perform region growing within ROI"""
    # Initialize segmentation
    segmented = np.zeros_like(roi_mask)
    
    # Initialize queue with seed points
    queue = list(seeds)
    
    # Grow region
    while queue:
        z,y,x = queue.pop(0)
        
        # Check 26-neighbors
        for dz in [-1,0,1]:
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    
                    nz,ny,nx = z+dz, y+dy, x+dx
                    
                    # Check bounds and constraints
                    if not (0 <= nz < roi_mask.shape[0] and
                           0 <= ny < roi_mask.shape[1] and
                           0 <= nx < roi_mask.shape[2]):
                        continue
                    
                    if (not segmented[nz,ny,nx] and
                        roi_mask[nz,ny,nx] and
                        vesselness[nz,ny,nx] >= threshold and
                        vesselness[nz,ny,nx] >= min_vesselness):
                        
                        segmented[nz,ny,nx] = True
                        queue.append((nz,ny,nx))
    
    return segmented

def ensure_connectivity(mask):
    """Ensure vessel connectivity by removing small components"""
    from scipy.ndimage import label
    
    # Label connected components
    labeled, num = label(mask)
    
    # Calculate minimum size
    min_size = 100  # Adjust based on vessel size
    
    # Remove small components
    for i in range(1, num+1):
        component = labeled == i
        if np.sum(component) < min_size:
            mask[component] = False
    
    return mask

def generate_quality_report(mask, centerlines, sigma_max):
    """Generate quality metrics report"""
    # Calculate metrics
    total_volume = np.sum(mask)
    
    # Find discontinuities in centerlines
    continuity_breaks = []  # TODO: Implement discontinuity detection
    
    # Check size consistency
    size_inconsistencies = []  # TODO: Implement size consistency check
    
    # Calculate airway overlap
    airway_overlap = 0.0  # TODO: Implement airway overlap calculation
    
    return QualityReport(
        total_volume=total_volume,
        continuity_breaks=continuity_breaks,
        size_inconsistencies=size_inconsistencies,
        airway_overlap=airway_overlap
    )

def save_local_threshold_results(segmentation, output_dir):
    """Save local thresholding results and metadata
    
    Args:
        segmentation: VesselSegmentation object
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary mask
    sitk.WriteImage(
        sitk.GetImageFromArray(segmentation.binary_mask.astype(np.uint8)),
        os.path.join(output_dir, 'final_vessels.nrrd')
    )
    
    # Save vessel radii
    sitk.WriteImage(
        sitk.GetImageFromArray(segmentation.vessel_radii),
        os.path.join(output_dir, 'vessel_radii.nrrd')
    )
    
    # Save metadata
    import json
    metadata = {
        "parameters": {
            "min_vesselness": segmentation.processing_parameters.min_vesselness,
            "roi_multiplier": segmentation.processing_parameters.roi_multiplier,
            "min_radius_cyl": segmentation.processing_parameters.min_radius_cyl,
            "min_radius_sphere": segmentation.processing_parameters.min_radius_sphere
        },
        "quality_metrics": {
            "total_volume": int(segmentation.quality_metrics.total_volume),
            "num_continuity_breaks": len(segmentation.quality_metrics.continuity_breaks),
            "num_size_inconsistencies": len(segmentation.quality_metrics.size_inconsistencies),
            "airway_overlap": float(segmentation.quality_metrics.airway_overlap)
        }
    }
    
    with open(os.path.join(output_dir, 'segmentation_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
