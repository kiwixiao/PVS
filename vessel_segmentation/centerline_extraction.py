import numpy as np
import SimpleITK as sitk
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import gc

@dataclass
class CenterlinePoint:
    """Data structure for centerline points"""
    x: int
    y: int
    z: int
    type: int  # 0=isolated, 1=endpoint, 2=segment, 3=bifurcation
    direction: np.ndarray  # Primary direction vector
    vessel_radius: float  # From σmax
    neighbors: Set[Tuple[int, int, int]]  # Connected points
    group_id: int  # For connected components

@dataclass
class VesselSegment:
    """Data structure for vessel segments"""
    points: List[CenterlinePoint]
    mean_direction: np.ndarray
    mean_radius: float
    start_point: CenterlinePoint
    end_point: CenterlinePoint

def extract_centerlines(binary_vessels, sigma_max=None):
    """Extract vessel centerlines using Palagyi's 6-subiteration thinning
    
    Args:
        binary_vessels: Binary vessel mask
        sigma_max: Scale of maximum vesselness response (for vessel radius)
        
    Returns:
        centerlines: Binary centerline mask
        point_types: Point classification (0=isolated, 1=endpoint, 2=segment, 3=bifurcation)
    """
    # Initialize arrays
    centerlines = binary_vessels.copy()
    point_types = np.zeros_like(binary_vessels, dtype=np.int8)
    
    # Create lookup tables for topology preservation
    euler_lut = create_euler_lut()
    simple_point_lut = create_simple_point_lut()
    
    # Main thinning loop
    changed = True
    while changed:
        changed = False
        # Six subiteration directions
        for direction in ['up', 'down', 'north', 'south', 'east', 'west']:
            # Identify simple points in current direction
            simple_points = identify_simple_points(centerlines, direction, simple_point_lut, euler_lut)
            
            # Remove simple points simultaneously
            if np.any(simple_points):
                centerlines[simple_points] = 0
                changed = True
            
            gc.collect()
    
    # Classify centerline points
    points = classify_centerline_points(centerlines, sigma_max)
    
    # Create vessel segments
    segments = create_vessel_segments(points)
    
    # Clean centerlines
    centerlines, point_types = clean_centerlines(points, segments)
    
    return centerlines, point_types

def create_euler_lut():
    """Create lookup table for Euler characteristic"""
    # Implementation of lookup table for 26-connectivity
    # This is a simplified version - full implementation would have 256 entries
    lut = np.zeros(256, dtype=np.int8)
    # Fill lookup table based on topology preservation rules
    # ... (detailed implementation omitted for brevity)
    return lut

def create_simple_point_lut():
    """Create lookup table for simple point detection"""
    # Implementation of lookup table for simple point criteria
    # This is a simplified version - full implementation would have 256 entries
    lut = np.zeros(256, dtype=bool)
    # Fill lookup table based on simple point criteria
    # ... (detailed implementation omitted for brevity)
    return lut

def identify_simple_points(image, direction, simple_point_lut, euler_lut):
    """Identify simple points in the given direction"""
    # Get 3x3x3 neighborhood configuration
    from scipy.ndimage import binary_erosion, binary_dilation
    
    # Create structuring element for current direction
    strel = np.ones((3,3,3), dtype=bool)
    if direction == 'up':
        strel[2,:,:] = False
    elif direction == 'down':
        strel[0,:,:] = False
    elif direction == 'north':
        strel[:,2,:] = False
    elif direction == 'south':
        strel[:,0,:] = False
    elif direction == 'east':
        strel[:,:,2] = False
    elif direction == 'west':
        strel[:,:,0] = False
    
    # Find border points in current direction
    border = binary_erosion(image, strel) ^ image
    
    # Check topology preservation for each point
    simple_points = np.zeros_like(image, dtype=bool)
    points = np.where(border)
    
    for z,y,x in zip(*points):
        # Get 3x3x3 neighborhood
        neighborhood = image[z-1:z+2, y-1:y+2, x-1:x+2].copy()
        neighborhood[1,1,1] = 0  # Remove center point
        
        # Check if point is simple using lookup tables
        if check_topology(neighborhood, simple_point_lut, euler_lut):
            simple_points[z,y,x] = True
    
    return simple_points

def check_topology(neighborhood, simple_point_lut, euler_lut):
    """Check topology preservation using lookup tables"""
    # Convert neighborhood to index
    index = 0
    for i, value in enumerate(neighborhood.ravel()):
        if value:
            index |= (1 << i)
    
    # Check if point is simple
    return simple_point_lut[index]

def classify_centerline_points(centerlines, sigma_max=None):
    """Classify centerline points and create data structures"""
    points = {}
    group_id = 0
    
    # Find all centerline points
    for z,y,x in zip(*np.where(centerlines)):
        # Count 26-neighbors
        neighbors = set()
        for dz in [-1,0,1]:
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    if centerlines[z+dz,y+dy,x+dx]:
                        neighbors.add((z+dz,y+dy,x+dx))
        
        # Create point object
        radius = sigma_max[z,y,x] if sigma_max is not None else 1.0
        point = CenterlinePoint(
            x=x, y=y, z=z,
            type=len(neighbors),  # Will be updated later
            direction=np.zeros(3),  # Will be updated for segment points
            vessel_radius=radius,
            neighbors=neighbors,
            group_id=-1
        )
        points[(z,y,x)] = point
    
    # Classify points
    for point in points.values():
        n = len(point.neighbors)
        if n == 0:
            point.type = 0  # Isolated
        elif n == 1:
            point.type = 1  # Endpoint
        elif n == 2:
            point.type = 2  # Segment
            # Calculate direction vector for segment points
            n1, n2 = list(point.neighbors)
            vec = np.array([n2[0]-n1[0], n2[1]-n1[1], n2[2]-n1[2]])
            point.direction = vec / np.linalg.norm(vec)
        else:
            point.type = 3  # Bifurcation
    
    return points

def create_vessel_segments(points):
    """Create vessel segments from classified points"""
    segments = []
    visited = set()
    
    # Find segment start points (endpoints or bifurcations)
    start_points = [p for p in points.values() if p.type in [1,3]]
    
    for start in start_points:
        if (start.z,start.y,start.x) in visited:
            continue
            
        # Follow segment
        if start.type == 2:
            continue  # Skip if not endpoint/bifurcation
            
        for neighbor in start.neighbors:
            if neighbor in visited:
                continue
                
            # Start new segment
            segment_points = [start]
            current = points[neighbor]
            visited.add((start.z,start.y,start.x))
            
            # Follow until next endpoint/bifurcation
            while current.type == 2 and (current.z,current.y,current.x) not in visited:
                segment_points.append(current)
                visited.add((current.z,current.y,current.x))
                # Get next unvisited neighbor
                next_point = None
                for n in current.neighbors:
                    if n not in visited:
                        next_point = points[n]
                        break
                if next_point is None:
                    break
                current = next_point
            
            # Add final point if endpoint/bifurcation
            if current.type in [1,3]:
                segment_points.append(current)
                visited.add((current.z,current.y,current.x))
            
            # Create segment if valid
            if len(segment_points) > 1:
                # Calculate mean direction and radius
                directions = [p.direction for p in segment_points if p.type == 2]
                mean_dir = np.mean(directions, axis=0)
                mean_dir = mean_dir / np.linalg.norm(mean_dir)
                
                mean_radius = np.mean([p.vessel_radius for p in segment_points])
                
                segment = VesselSegment(
                    points=segment_points,
                    mean_direction=mean_dir,
                    mean_radius=mean_radius,
                    start_point=segment_points[0],
                    end_point=segment_points[-1]
                )
                segments.append(segment)
    
    return segments

def clean_centerlines(points, segments):
    """Clean centerlines by removing unwanted points"""
    # Initialize output arrays
    shape = max(p.z for p in points.values()) + 1, \
            max(p.y for p in points.values()) + 1, \
            max(p.x for p in points.values()) + 1
    centerlines = np.zeros(shape, dtype=np.uint8)
    point_types = np.zeros(shape, dtype=np.int8)
    
    # Remove isolated points and short endpoints
    for point in points.values():
        if point.type == 0:  # Remove isolated points
            continue
        if point.type == 1:  # Check endpoints
            # Remove if connected directly to bifurcation
            neighbor = list(point.neighbors)[0]
            if points[neighbor].type == 3:
                continue
        
        # Keep point
        centerlines[point.z,point.y,point.x] = 1
        point_types[point.z,point.y,point.x] = point.type
    
    return centerlines, point_types

def save_centerline_results(centerlines, point_types, output_dir):
    """Save centerline extraction results
    
    Args:
        centerlines: Binary centerline mask
        point_types: Point classification (0=isolated, 1=endpoint, 2=segment, 3=bifurcation)
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save centerlines
    sitk.WriteImage(
        sitk.GetImageFromArray(centerlines.astype(np.uint8)),
        os.path.join(output_dir, 'centerlines.nrrd')
    )
    
    # Save point types
    sitk.WriteImage(
        sitk.GetImageFromArray(point_types),
        os.path.join(output_dir, 'centerline_types.nrrd')
    )
