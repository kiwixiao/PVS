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
    Optimized implementation of Palagyi & Kuba's 6-subiteration thinning algorithm.
    
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
    
    # Direction templates (U,D,N,S,E,W)
    direction_checks = [
        (0,1,0),   # U: Up neighbor should be 0
        (0,-1,0),  # D: Down neighbor should be 0
        (1,0,0),   # N: North neighbor should be 0
        (-1,0,0),  # S: South neighbor should be 0
        (0,0,1),   # E: East neighbor should be 0
        (0,0,-1)   # W: West neighbor should be 0
    ]
    
    # Continue until no points can be deleted
    iteration = 0
    with tqdm(desc="Thinning iterations", leave=False) as pbar:
        while True:
            changed = False
            
            # Apply 6 subiterations in order (U,D,N,S,E,W)
            for direction, (dz, dy, dx) in enumerate(direction_checks):
                # Get border points efficiently using numpy operations
                border_mask = padded == 1
                for offset_z, offset_y, offset_x in neighbor_offsets:
                    rolled = np.roll(np.roll(np.roll(padded == 0, 
                                                    offset_z, axis=0),
                                           offset_y, axis=1),
                                   offset_x, axis=2)
                    border_mask &= rolled
                
                border_points = np.argwhere(border_mask)
                
                # Process points in larger batches
                batch_size = 5000
                deletable_points = []
                
                for i in range(0, len(border_points), batch_size):
                    batch = border_points[i:i+batch_size]
                    
                    # Skip border points due to padding
                    valid_mask = ((batch[:,0] >= 1) & (batch[:,1] >= 1) & (batch[:,2] >= 1) &
                                (batch[:,0] < padded.shape[0]-1) & 
                                (batch[:,1] < padded.shape[1]-1) & 
                                (batch[:,2] < padded.shape[2]-1))
                    batch = batch[valid_mask]
                    
                    if len(batch) == 0:
                        continue
                    
                    # Extract neighborhoods efficiently
                    neighborhoods = np.stack([padded[z-1:z+2, y-1:y+2, x-1:x+2] 
                                           for z,y,x in batch])
                    
                    # Check template and simplicity in parallel
                    template_match = np.array([matches_deletion_template(nb, direction) 
                                            for nb in neighborhoods])
                    simple_points = np.array([is_simple_point(nb) 
                                           for nb in neighborhoods[template_match]])
                    
                    # Add deletable points
                    deletable_idx = np.where(template_match)[0][simple_points]
                    if len(deletable_idx) > 0:
                        deletable_points.extend(batch[deletable_idx])
                    
                    if i % 10000 == 0:
                        gc.collect()
                
                # Delete points simultaneously
                if len(deletable_points) > 0:
                    changed = True
                    deletable_points = np.array(deletable_points)
                    padded[deletable_points[:,0], deletable_points[:,1], deletable_points[:,2]] = 0
            
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
