import numpy as np
from typing import Tuple, List, Set
from tqdm import tqdm
import SimpleITK as sitk
import os
import vtk
from vtk.util import numpy_support
import gc
from scipy.ndimage import label

def matches_deletion_template(neighborhood, direction_idx):
    """Check if point matches deletion template for current direction
    
    Args:
        neighborhood: 3x3x3 binary neighborhood
        direction_idx: Index of current direction (0=U, 1=D, 2=N, 3=S, 4=E, 5=W)
        
    Returns:
        bool: True if point can be deleted, False otherwise
    """
    # Check if point has background neighbor in current direction
    if direction_idx == 0:  # U
        return neighborhood[2,1,1] == 0
    elif direction_idx == 1:  # D
        return neighborhood[0,1,1] == 0
    elif direction_idx == 2:  # N
        return neighborhood[1,2,1] == 0
    elif direction_idx == 3:  # S
        return neighborhood[1,0,1] == 0
    elif direction_idx == 4:  # E
        return neighborhood[1,1,2] == 0
    else:  # W
        return neighborhood[1,1,0] == 0

def is_simple_point(neighborhood):
    """Check if point is simple (its removal won't change topology)
    
    Args:
        neighborhood: 3x3x3 binary neighborhood
        
    Returns:
        bool: True if point is simple, False otherwise
    """
    center = (1, 1, 1)
    
    # Count number of connected components in 26-neighborhood
    # First with center point
    labeled_with, num_with = label(neighborhood, structure=np.ones((3,3,3)))
    
    # Then without center point
    temp = neighborhood.copy()
    temp[center] = 0
    labeled_without, num_without = label(temp, structure=np.ones((3,3,3)))
    
    # Point is simple if removing it doesn't change number of connected components
    return num_with == num_without

def thin_vessels_3d(binary_volume: np.ndarray) -> np.ndarray:
    """
    Implementation of Palagyi & Kuba's 6-subiteration thinning algorithm.
    
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
    
    # Pre-compute valid region mask to avoid repeated boundary checks
    valid_region = np.zeros_like(padded, dtype=bool)
    valid_region[1:-1, 1:-1, 1:-1] = True
    
    # Continue until no points can be deleted
    iteration = 0
    with tqdm(desc="Thinning iterations", leave=False) as pbar:
        while True:
            changed = False
            
            # Apply 6 subiterations in order (U,D,N,S,E,W)
            for direction_idx in range(6):
                # Find border points
                border_points = []
                points = np.argwhere(padded == 1)
                
                for z, y, x in points:
                    if not valid_region[z,y,x]:
                        continue
                    
                    # Extract neighborhood
                    neighborhood = padded[z-1:z+2, y-1:y+2, x-1:x+2].copy()
                    
                    # Check if point matches deletion template
                    if matches_deletion_template(neighborhood, direction_idx):
                        # Check if point is simple
                        if is_simple_point(neighborhood):
                            border_points.append((z,y,x))
                
                # Delete points simultaneously
                if border_points:
                    changed = True
                    for z, y, x in border_points:
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
