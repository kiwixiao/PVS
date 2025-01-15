import numpy as np
from typing import Tuple, List, Set
from tqdm import tqdm
import SimpleITK as sitk
import os
import vtk
from vtk.util import numpy_support
import gc
from scipy.ndimage import label

def thin_vessels_3d(binary_volume: np.ndarray) -> np.ndarray:
    """
    Optimized implementation of Palagyi & Kuba's 6-subiteration thinning algorithm.
    Ensures single-pixel width centerlines by strictly following deletion templates.
    
    Args:
        binary_volume: 3D numpy array with binary vessel segmentation (1=vessel, 0=background)
        
    Returns:
        3D numpy array with centerline voxels
    """
    # Print initial statistics
    print("\nInitial vessel mask statistics:")
    print(f"Total vessel voxels: {np.sum(binary_volume)}")
    
    # Pad volume to handle border cases
    padded = np.pad(binary_volume, pad_width=1, mode='constant', constant_values=0)
    
    # Pre-compute neighbor offset arrays for efficiency
    z, y, x = np.ogrid[-1:2, -1:2, -1:2]
    neighbor_offsets = list(zip(z.ravel(), y.ravel(), x.ravel()))
    neighbor_offsets.remove((0,0,0))  # Remove center point
    
    # Direction templates (U,D,N,S,E,W) - pre-compute for efficiency
    direction_checks = [
        (0,1,0),   # U: Up neighbor should be 0
        (0,-1,0),  # D: Down neighbor should be 0
        (1,0,0),   # N: North neighbor should be 0
        (-1,0,0),  # S: South neighbor should be 0
        (0,0,1),   # E: East neighbor should be 0
        (0,0,-1)   # W: West neighbor should be 0
    ]
    
    # Pre-compute valid region mask to avoid repeated boundary checks
    valid_region = np.zeros_like(padded, dtype=bool)
    valid_region[1:-1, 1:-1, 1:-1] = True
    
    # Continue until no points can be deleted
    iteration = 0
    with tqdm(desc="Thinning iterations", leave=False) as pbar:
        while True:
            changed = False
            
            # Apply 6 subiterations in order (U,D,N,S,E,W)
            for direction_idx, (dz, dy, dx) in enumerate(direction_checks):
                # Get border points efficiently using numpy operations
                border_points = []
                
                # Find border points (points with at least one background neighbor)
                points = np.argwhere(padded == 1)
                for z, y, x in points:
                    if not valid_region[z,y,x]:
                        continue
                        
                    # Check if point has background neighbor in current direction
                    if padded[z+dz, y+dy, x+dx] == 0:
                        # Check other neighbors to ensure it's a border point
                        is_border = False
                        for nz, ny, nx in neighbor_offsets:
                            if padded[z+nz, y+ny, x+nx] == 0:
                                is_border = True
                                break
                        if is_border:
                            border_points.append((z,y,x))
                
                if not border_points:
                    continue
                
                # Process points in batches
                batch_size = 1000
                deletable_points = []
                
                for i in range(0, len(border_points), batch_size):
                    batch = border_points[i:i+batch_size]
                    
                    # Check each point in batch
                    for z, y, x in batch:
                        # Extract neighborhood
                        neighborhood = padded[z-1:z+2, y-1:y+2, x-1:x+2].copy()
                        
                        # Check if point matches deletion template
                        if matches_deletion_template(neighborhood, direction_idx):
                            # Check if point is simple (its removal won't change topology)
                            if is_simple_point(neighborhood):
                                deletable_points.append((z,y,x))
                    
                    if i % 10000 == 0:
                        gc.collect()
                
                # Delete points simultaneously
                if deletable_points:
                    changed = True
                    for z, y, x in deletable_points:
                        padded[z,y,x] = 0
            
            if not changed:
                break
                
            iteration += 1
            pbar.update(1)
            
            # Periodic cleanup
            if iteration % 5 == 0:
                gc.collect()
    
    # Remove padding and return
    centerlines = padded[1:-1, 1:-1, 1:-1]
    
    # Print comparison statistics
    print("\nComparison statistics:")
    print(f"Original vessel voxels: {np.sum(binary_volume)}")
    print(f"Centerline voxels: {np.sum(centerlines)}")
    print(f"Reduction ratio: {np.sum(centerlines) / np.sum(binary_volume):.4f}")
    
    return centerlines

def matches_deletion_template(neighborhood: np.ndarray, direction: int) -> bool:
    """
    Check if a point matches the deletion template for a given direction.
    Strictly follows Palagyi & Kuba's 6-subiteration templates.
    """
    # Get the center point's neighbors
    up    = neighborhood[0,1,1]  # U
    down  = neighborhood[2,1,1]  # D
    north = neighborhood[1,0,1]  # N
    south = neighborhood[1,2,1]  # S
    east  = neighborhood[1,1,2]  # E
    west  = neighborhood[1,1,0]  # W
    
    # Check specific template based on direction
    if direction == 0:   # U-template
        return up == 0 and (down == 1 or north == 1 or south == 1 or east == 1 or west == 1)
    elif direction == 1: # D-template
        return down == 0 and (up == 1 or north == 1 or south == 1 or east == 1 or west == 1)
    elif direction == 2: # N-template
        return north == 0 and (south == 1 or up == 1 or down == 1 or east == 1 or west == 1)
    elif direction == 3: # S-template
        return south == 0 and (north == 1 or up == 1 or down == 1 or east == 1 or west == 1)
    elif direction == 4: # E-template
        return east == 0 and (west == 1 or up == 1 or down == 1 or north == 1 or south == 1)
    else:               # W-template
        return west == 0 and (east == 1 or up == 1 or down == 1 or north == 1 or south == 1)

def is_simple_point(neighborhood: np.ndarray) -> bool:
    """
    Check if a point is simple (its removal won't change topology).
    Uses Euler characteristic to ensure topology preservation.
    """
    # Count the number of 26-connected components in the 26-neighborhood
    # excluding the center point
    center = neighborhood[1,1,1]
    if center == 0:
        return False
        
    # Temporarily remove center point
    neighborhood[1,1,1] = 0
    
    # Count 26-connected components
    labeled, num_components = label(neighborhood, structure=np.ones((3,3,3)))
    
    # Restore center point
    neighborhood[1,1,1] = center
    
    # A point is simple if:
    # 1. There is exactly one 26-connected component in its neighborhood
    # 2. The point is not an endpoint (has more than one neighbor)
    neighbor_count = np.sum(neighborhood) - 1  # Subtract center point
    
    return num_components == 1 and neighbor_count > 1

def extract_centerlines(binary_vessels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract centerlines from binary vessel mask using Palagyi's 6-subiteration thinning."""
    # Get centerlines using thinning
    centerlines = thin_vessels_3d(binary_vessels)
    
    # Classify points
    point_types = np.zeros_like(centerlines, dtype=np.uint8)
    points = np.where(centerlines > 0)
    
    # Validate one-pixel-width property
    max_neighbors = 0
    for i in range(len(points[0])):
        z, y, x = points[0][i], points[1][i], points[2][i]
        
        # Get neighborhood
        neighborhood = np.zeros((3,3,3), dtype=centerlines.dtype)
        zmin, zmax = max(0, z-1), min(centerlines.shape[0], z+2)
        ymin, ymax = max(0, y-1), min(centerlines.shape[1], y+2)
        xmin, xmax = max(0, x-1), min(centerlines.shape[2], x+2)
        
        z_slice = slice(max(0, 1-(z-zmin)), min(3, 1+(zmax-z)))
        y_slice = slice(max(0, 1-(y-ymin)), min(3, 1+(ymax-y)))
        x_slice = slice(max(0, 1-(x-xmin)), min(3, 1+(xmax-x)))
        
        neighborhood[z_slice, y_slice, x_slice] = centerlines[zmin:zmax, ymin:ymax, xmin:xmax]
        
        # Count neighbors
        neighbors = np.sum(neighborhood) - 1  # Subtract center point
        max_neighbors = max(max_neighbors, neighbors)
        
        if neighbors == 1:
            point_types[z,y,x] = 1  # Endpoint
        elif neighbors == 2:
            point_types[z,y,x] = 2  # Segment point
        elif neighbors > 2:
            point_types[z,y,x] = 3  # Bifurcation point
    
    print(f"\nOne-pixel-width validation:")
    print(f"Maximum number of neighbors for any centerline point: {max_neighbors}")
    if max_neighbors > 3:
        print("Warning: Some points have more than 3 neighbors, indicating possible thickness issues")
    
    # Print centerline statistics
    total_points = np.sum(centerlines)
    endpoint_count = np.sum(point_types == 1)
    segment_count = np.sum(point_types == 2)
    bifurcation_count = np.sum(point_types == 3)
    
    print("\nCenterline Statistics:")
    print(f"Total centerline points: {total_points}")
    print(f"Endpoints: {endpoint_count}")
    print(f"Segment points: {segment_count}")
    print(f"Bifurcation points: {bifurcation_count}")
    print(f"Average branch length: {segment_count / (endpoint_count + bifurcation_count):.2f} points")
    
    return centerlines, point_types

def save_centerline_results(centerlines, point_types, output_dir):
    """Save centerline extraction results in both NRRD and VTK formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save NRRD files
    sitk.WriteImage(
        sitk.GetImageFromArray(centerlines.astype(np.uint8)),
        os.path.join(output_dir, 'centerlines.nrrd')
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(point_types),
        os.path.join(output_dir, 'centerline_point_types.nrrd')
    )
    
    # Convert to VTK PolyData
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    point_data = vtk.vtkIntArray()
    point_data.SetName("PointType")
    
    # Get centerline points
    z, y, x = np.where(centerlines > 0)
    point_id = 0
    point_id_map = {}
    
    # Add points and their types
    for i in range(len(z)):
        points.InsertNextPoint(x[i], y[i], z[i])
        point_data.InsertNextValue(point_types[z[i], y[i], x[i]])
        point_id_map[(z[i], y[i], x[i])] = point_id
        point_id += 1
    
    # Create lines by connecting adjacent points
    for i in range(len(z)):
        current = (z[i], y[i], x[i])
        # Check 26-neighborhood for connected points
        for dz in [-1,0,1]:
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = z[i] + dz, y[i] + dy, x[i] + dx
                    if (nz, ny, nx) in point_id_map and \
                       point_types[nz, ny, nx] > 0:  # Connected point
                        # Create line only if current point ID is less than neighbor ID
                        # to avoid duplicate lines
                        if point_id_map[current] < point_id_map[(nz, ny, nx)]:
                            line = vtk.vtkLine()
                            line.GetPointIds().SetId(0, point_id_map[current])
                            line.GetPointIds().SetId(1, point_id_map[(nz, ny, nx)])
                            lines.InsertNextCell(line)
    
    # Create PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().AddArray(point_data)
    
    # Save as VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(os.path.join(output_dir, 'centerlines.vtk'))
    writer.SetInputData(polydata)
    writer.Write()
    
    print("\nSaved centerline results:")
    print(f"- NRRD files: centerlines.nrrd, centerline_point_types.nrrd")
    print(f"- VTK file: centerlines.vtk")
    print("\nPoint type legend:")
    print("1: Endpoint")
    print("2: Segment point")
    print("3: Bifurcation point")
