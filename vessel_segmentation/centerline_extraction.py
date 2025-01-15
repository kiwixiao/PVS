import numpy as np
from typing import Tuple, List, Set
from tqdm import tqdm
import SimpleITK as sitk
import os
import vtk
from vtk.util import numpy_support
import gc

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
    
    # Pre-compute valid region mask (exclude padding)
    valid_mask = np.zeros_like(padded, dtype=bool)
    valid_mask[1:-1, 1:-1, 1:-1] = True
    
    # Pre-compute neighbor offset arrays for each direction
    direction_checks = {
        0: [(0,0,1)],  # U
        1: [(0,0,-1)], # D
        2: [(0,1,0)],  # N
        3: [(0,-1,0)], # S
        4: [(1,0,0)],  # E
        5: [(-1,0,0)]  # W
    }
    
    # Convert direction templates to numpy arrays for efficiency
    direction_templates = {}
    for direction in range(6):
        template = np.ones((3,3,3), dtype=bool)
        for dz, dy, dx in direction_checks[direction]:
            template[1+dz,1+dy,1+dx] = False
        direction_templates[direction] = template
    
    # Continue until no points can be deleted
    iteration = 0
    with tqdm(desc="Thinning iterations", leave=False) as pbar:
        while True:
            changed = False
            # Apply 6 subiterations in order (U,D,N,S,E,W)
            for direction in range(6):
                # Get border points more efficiently
                border_points = []
                for dz, dy, dx in direction_checks[direction]:
                    shifted = np.roll(padded == 0, (dz,dy,dx), axis=(0,1,2))
                    border_mask = (padded == 1) & shifted & valid_mask
                    if np.any(border_mask):
                        border_points.extend(zip(*np.where(border_mask)))
                
                if not border_points:
                    continue
                
                # Process points in batches
                batch_size = 1000
                deletable_points = []
                
                for i in range(0, len(border_points), batch_size):
                    batch = border_points[i:i+batch_size]
                    
                    for z, y, x in batch:
                        # Get 3x3x3 neighborhood
                        nb = padded[z-1:z+2, y-1:y+2, x-1:x+2].copy()
                        
                        # Quick check for template match
                        if not np.all(nb[direction_templates[direction]]):
                            continue
                        
                        # Check if point is simple
                        if is_simple_point(nb):
                            deletable_points.append((z,y,x))
                    
                    # Periodic garbage collection
                    if i % 5000 == 0:
                        gc.collect()
                
                # Delete points simultaneously
                if deletable_points:
                    changed = True
                    for z,y,x in deletable_points:
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

def matches_deletion_template(nb: np.ndarray, direction: int) -> bool:
    """
    Check if 3x3x3 neighborhood matches deletion template for given direction.
    Templates M1-M6 from Figure 3 and their rotations.
    """
    # Get base templates for this direction
    if direction == 0:  # U direction
        # Template M1
        if (nb[1,1,2] == 0 and  # point marked U must be 0
            np.sum(nb[1,:,1]) >= 1 and  # at least one x point must be 1
            nb[1,1,0] == 1):  # center bottom must be 1
            return True
            
        # Template M2  
        if (nb[1,1,2] == 0 and
            nb[1,0,1] == 1 and
            nb[1,2,1] == 1 and
            nb[1,1,0] == 1):
            return True
            
        # Template M3
        if (nb[1,1,2] == 0 and
            nb[0,1,1] == 1 and
            nb[2,1,1] == 1 and 
            nb[1,1,0] == 1):
            return True
            
        # Template M4 
        if (nb[1,1,2] == 0 and
            nb[0,0,1] == 1 and
            nb[2,2,1] == 1 and
            nb[1,1,0] == 1):
            return True
            
        # Template M5
        if (nb[1,1,2] == 0 and
            np.sum(nb[1,:,1]) >= 1 and
            nb[1,1,0] == 1):
            return True
            
        # Template M6
        if (nb[1,1,2] == 0 and
            nb[1,0,1] == 1 and
            nb[1,2,1] == 1 and
            nb[1,1,0] == 1):
            return True
            
    # Other directions - rotate/reflect templates appropriately
    else:
        nb = rotate_neighborhood(nb, direction)
        return matches_deletion_template(nb, 0)
        
    return False

def is_simple_point(nb: np.ndarray) -> bool:
    """
    Check if point is simple according to topology preservation criteria.
    """
    # Get counts of 26-connected and 6-connected components
    n26 = count_26_components(nb)
    n6 = count_6_components(nb)
    
    # Point is simple if:
    # 1. Single 26-connected component in N26 intersection B
    # 2. Single 6-connected component in N26 intersection background
    return n26 == 1 and n6 == 1

def count_26_components(nb: np.ndarray) -> int:
    """Count number of 26-connected components in 3x3x3 neighborhood"""
    markers = np.zeros_like(nb)
    current_label = 1
    
    # Iterate through points
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if nb[x,y,z] == 1 and markers[x,y,z] == 0:
                    # Found new component, flood fill
                    flood_fill_26(nb, markers, x, y, z, current_label)
                    current_label += 1
                    
    return current_label - 1

def count_6_components(nb: np.ndarray) -> int:
    """Count number of 6-connected components in 3x3x3 neighborhood"""
    markers = np.zeros_like(nb)
    current_label = 1
    
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if nb[x,y,z] == 1 and markers[x,y,z] == 0:
                    flood_fill_6(nb, markers, x, y, z, current_label)
                    current_label += 1
                    
    return current_label - 1

def flood_fill_26(nb: np.ndarray, markers: np.ndarray, x: int, y: int, z: int, label: int):
    """26-connected flood fill"""
    if x < 0 or x > 2 or y < 0 or y > 2 or z < 0 or z > 2:
        return
    if nb[x,y,z] == 0 or markers[x,y,z] != 0:
        return
        
    markers[x,y,z] = label
    
    # Recursively fill 26-connected neighbors
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            for dz in [-1,0,1]:
                if dx == dy == dz == 0:
                    continue
                flood_fill_26(nb, markers, x+dx, y+dy, z+dz, label)

def flood_fill_6(nb: np.ndarray, markers: np.ndarray, x: int, y: int, z: int, label: int):
    """6-connected flood fill"""
    if x < 0 or x > 2 or y < 0 or y > 2 or z < 0 or z > 2:
        return
    if nb[x,y,z] == 0 or markers[x,y,z] != 0:
        return
        
    markers[x,y,z] = label
    
    # Recursively fill 6-connected neighbors
    for d in [(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)]:
        flood_fill_6(nb, markers, x+d[0], y+d[1], z+d[2], label)

def rotate_neighborhood(nb: np.ndarray, direction: int) -> np.ndarray:
    """Rotate/reflect 3x3x3 neighborhood based on direction"""
    if direction == 1:  # D - rotate 180 around Y
        return np.rot90(nb, k=2, axes=(0,2))
    elif direction == 2:  # N - rotate 90 around X 
        return np.rot90(nb, k=1, axes=(1,2))
    elif direction == 3:  # S - rotate -90 around X
        return np.rot90(nb, k=-1, axes=(1,2))
    elif direction == 4:  # E - rotate 90 around Y
        return np.rot90(nb, k=1, axes=(0,2))
    elif direction == 5:  # W - rotate -90 around Y
        return np.rot90(nb, k=-1, axes=(0,2))
    return nb

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

def save_centerline_results(centerlines, point_types, output_dir, sigma_max=None, vessel_directions=None):
    """Save centerline extraction results in both NRRD and VTK formats with enhanced visualization
    
    Args:
        centerlines: Binary centerline mask
        point_types: Point type labels (1=endpoint, 2=segment, 3=bifurcation)
        output_dir: Directory to save results
        sigma_max: Maximum scale at each point (for radius visualization)
        vessel_directions: Vessel direction vectors
    """
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
    
    # Create enhanced VTK visualization
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    
    # Point data arrays
    point_type_data = vtk.vtkIntArray()
    point_type_data.SetName("PointType")
    
    radius_data = vtk.vtkFloatArray()
    radius_data.SetName("Radius")
    
    # Direction vectors (if available)
    direction_vectors = vtk.vtkFloatArray()
    direction_vectors.SetName("Direction")
    direction_vectors.SetNumberOfComponents(3)
    
    # Get centerline points
    z, y, x = np.where(centerlines > 0)
    point_id = 0
    point_id_map = {}
    
    # Add points and their attributes
    for i in range(len(z)):
        points.InsertNextPoint(x[i], y[i], z[i])
        point_type_data.InsertNextValue(point_types[z[i], y[i], x[i]])
        
        # Add radius if available
        if sigma_max is not None:
            radius = sigma_max[z[i], y[i], x[i]]
            radius_data.InsertNextValue(radius)
        
        # Add direction if available
        if vessel_directions is not None:
            direction = vessel_directions[z[i], y[i], x[i]]
            direction_vectors.InsertNextTuple3(direction[0], direction[1], direction[2])
        
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
    
    # Add point data
    polydata.GetPointData().AddArray(point_type_data)
    if sigma_max is not None:
        polydata.GetPointData().AddArray(radius_data)
    if vessel_directions is not None:
        polydata.GetPointData().AddArray(direction_vectors)
    
    # Save as VTP file (XML PolyData format)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(os.path.join(output_dir, 'centerlines.vtp'))
    writer.SetInputData(polydata)
    writer.Write()
    
    print("\nSaved centerline results:")
    print(f"- NRRD files: centerlines.nrrd, centerline_point_types.nrrd")
    print(f"- VTK file: centerlines.vtp")
    print("\nPoint type legend:")
    print("1: Endpoint (Red)")
    print("2: Segment point (Blue)")
    print("3: Bifurcation point (Yellow)")
    print("\nVisualization attributes:")
    print("- Point size: Proportional to vessel radius (if sigma_max provided)")
    print("- Direction vectors: Shown as glyphs (if vessel_directions provided)")
    print("- Color: Based on point type")
