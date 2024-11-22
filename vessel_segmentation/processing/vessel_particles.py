import vtk
from vtk.util import numpy_support
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

class VesselParticleSystem:
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else None
        self.logger = logging.getLogger(__name__)
        
    def create_particle_system(self,
                               image_data: np.ndarray,
                               voxel_spacing: Tuple[float, float, float],
                               scales: List[float]) -> vtk.vtkPolyData:
        """Initialize particle system for vessel analysis"""
        # Create points for each voxel center
        x, y,z = np.meshgrid(
            np.arange(image_data.shape[0]),
            np.arange(image_data.shape[1]),
            np.arange(image_data.shape[2]),
            indexing='ij'
        )
        
        # Convert to physical coordinates
        x = x * voxel_spacing[0]
        y = y * voxel_spacing[1]
        z = z * voxel_spacing[2]
        
        # Create point array
        points = vtk.vtkPoints()
        for i in range(image_data.size):
            points.InsertNextPoint(
                x.flat[i],
                y.flat[i],
                z.flat[i]
            )
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Initialize arrays for multiscale analysis
        self._initialize_arrays(polydata, len(scales))
        
        return polydata
    
    def _initialize_arrays(self, polydata: vtk.vtkPolyData, num_scales: int):
        """Initialize arrays for storing multiscale vessel properities"""
        num_points = polydata.GetNumberOfPoints()
        
        # Create arrays for each scale
        for i in range(num_scales):
            # Vesselness response
            vesselness = vtk.vtkDoubleArray()  # Changed to vtkDoubleArray
            vesselness.SetName(f"vesselness_scale_{i}")  # Removed 'l'
            vesselness.SetNumberOfComponents(1)
            vesselness.SetNumberOfValues(num_points)  # Changed to SetNumberOfValues
            vesselness.Fill(0.0)  # Initialize with zeros
            polydata.GetPointData().AddArray(vesselness)
            
            # Eigenvalues (3 components)
            eigenvalues = vtk.vtkDoubleArray()  # Changed to vtkDoubleArray
            eigenvalues.SetName(f"eigenvalues_scale_{i}")
            eigenvalues.SetNumberOfComponents(3)
            eigenvalues.SetNumberOfTuples(num_points)
            eigenvalues.Fill(0.0)  # Initialize with zeros
            polydata.GetPointData().AddArray(eigenvalues)
            
            # Principal direction (3 components)
            direction = vtk.vtkDoubleArray()  # Changed to vtkDoubleArray
            direction.SetName(f"direction_scale_{i}")
            direction.SetNumberOfComponents(3)
            direction.SetNumberOfTuples(num_points)
            direction.Fill(0.0)  # Initialize with zeros
            polydata.GetPointData().AddArray(direction)
            
            # Vessel diameter (in mm)
            diameter = vtk.vtkDoubleArray()  # Changed to vtkDoubleArray
            diameter.SetName(f"diameter_mm_scale_{i}")  # Changed name
            diameter.SetNumberOfComponents(1)
            diameter.SetNumberOfValues(num_points)  # Changed to SetNumberOfValues
            diameter.Fill(0.0)  # Initialize with zeros
            polydata.GetPointData().AddArray(diameter)
            
            # Cross-sectional area
            area = vtk.vtkDoubleArray()  # Added cross-sectional area array
            area.SetName(f"cross_section_area_mm2_scale_{i}")
            area.SetNumberOfComponents(1)
            area.SetNumberOfValues(num_points)
            area.Fill(0.0)
            polydata.GetPointData().AddArray(area)
       
    def update_scale_response(self, 
                            polydata: vtk.vtkPolyData,
                            scale_index: int,
                            vesselness: np.ndarray,
                            eigenvalues: Dict[str, np.ndarray],
                            sigma: float):
        """Update particle system with response at specific scale"""
        try:
            # Get array names for debugging
            array_names = [polydata.GetPointData().GetArrayName(i) 
                      for i in range(polydata.GetPointData().GetNumberOfArrays())]
            print(f"Available arrays: {array_names}")
            
            # Flatten arrays for VTK
            flat_vesselness = vesselness.ravel()
            flat_eig1 = eigenvalues['lambda1'].ravel()
            flat_eig2 = eigenvalues['lambda2'].ravel()
            flat_eig3 = eigenvalues['lambda3'].ravel()
            
            # Get arrays with proper error checking
            array_name = f"vesselness_scale_{scale_index}"
            
            # Update vesselness response
            vtk_vesselness = polydata.GetPointData().GetArray(array_name)
            if vtk_vesselness is None:
                raise ValueError(f"Array {array_name} not found in point data")
            
            # Update vesselness
            for i in range(len(flat_vesselness)):
                vtk_vesselness.SetValue(i, flat_vesselness[i])
            
            # Update eigenvalues
            array_name = f"eigenvalues_scale_{scale_index}"
            vtk_eigenvalues = polydata.GetPointData().GetArray(array_name)
            if vtk_eigenvalues is None:
                raise ValueError(f"Array {array_name} not found in point data")
            
            for i in range(len(flat_eig1)):
                vtk_eigenvalues.SetTuple3(i, flat_eig1[i], flat_eig2[i], flat_eig3[i])
            
            # Compute and update vessel direction (smallest eigenvector)
            directions = self._compute_vessel_directions(eigenvalues)
            array_name = f"direction_scale_{scale_index}"
            vtk_directions = polydata.GetPointData().GetArray(array_name)
            if vtk_directions is None:
                raise ValueError(f"Array {array_name} not found in point data")
            
            for i in range(len(directions)):
                vtk_directions.SetTuple3(i, * [float(x) for x in directions[i]])
            
            # Calculate vessel diameter (2 * radius)
            diameters = 2 * sigma * np.sqrt(np.abs(flat_eig2 / (flat_eig1 + 1e-10)))
            array_name = f"diameter_mm_scale_{scale_index}"
            vtk_diameter = polydata.GetPointData().GetArray(array_name)
            if vtk_diameter is None:
                raise ValueError(f"Array {array_name} not found in point data")
                
            for i in range(len(diameters)):
                vtk_diameter.SetValue(i, float(diameters[i]))
            
            # Calculate cross-sectional area
            areas = np.pi * (diameters/2)**2
            array_name = f"cross_section_area_mm2_scale_{scale_index}"
            vtk_area = polydata.GetPointData().GetArray(array_name)
            if vtk_area is None:
                raise ValueError(f"Array {array_name} not found in point data")
                
            for i in range(len(areas)):
                vtk_area.SetValue(i, float(areas[i]))
                
        except Exception as e:
            self.logger.error(f"Error updating scale response: {str(e)}")
            raise
    
    def _compute_vessel_directions(self, eigenvalues: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute vessel directions from eigenvalues"""
        # For each point, construct the Hessian matrix and compute eigenvectors
        shape = eigenvalues['lambda1'].shape
        directions = np.zeros((np.prod(shape), 3))
        
        flat_eig1 = eigenvalues['lambda1'].flatten()
        flat_eig2 = eigenvalues['lambda2'].flatten()
        flat_eig3 = eigenvalues['lambda3'].flatten()
        
        for i in range(len(flat_eig1)):
            # Create diagonal matrix of eigenvalues
            D = np.diag([flat_eig1[i], flat_eig2[i], flat_eig3[i]])
            # Get eigenvectors (last column is smallest eigenvalue direction)
            _, V = np.linalg.eigh(D)
            directions[i] = V[:, 0]  # Vessel direction is eigenvector of smallest eigenvalue
            
        return directions
    
    def save_particle_system(self, polydata: vtk.vtkPolyData, case_id: str) -> str:
        """Save particle system to VTK file"""
        try:
            output_path = self.output_dir / f"{case_id}_vessel_particles.vtp"
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(str(output_path))
            writer.SetInputData(polydata)
            writer.Write()
            self.logger.info(f"Saved particle system to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving particle system: {str(e)}")
            raise
            
    def extract_vessel_particles(self, 
                           polydata: vtk.vtkPolyData,
                           vesselness_threshold: float = 0.1) -> vtk.vtkPolyData:
        """Extract significant vessel particles based on maximum vesselness"""
        try:
            # Find maximum vesselness across scales for each point
            num_points = polydata.GetNumberOfPoints()
            pointData = polydata.GetPointData()
            # Count numnber of vesselness arrays
            num_scales = sum(1 for i in range(pointData.GetNumberOfArrays()) 
                        if pointData.GetArrayName(i).startswith('vesselness_scale_'))
            
            self.logger.info(f"Found {num_scales} scales in particle data")
            
            max_vesselness = np.zeros(num_points)
            best_scale = np.zeros(num_points, dtype=int)
            
            # Find maximum response scale for each point
            for i in range(num_scales):
                array_name = f"vesselness_scale_{i}"
                vesselness_array = pointData.GetArray(array_name)
                if vesselness_array is None:
                    raise ValueError(f"Cound not found array {array_name}")
                
                vesselness = numpy_support.vtk_to_numpy(vesselness_array)
                update_mask = vesselness > max_vesselness
                max_vesselness[update_mask] = vesselness[update_mask]
                best_scale[update_mask] = i
            
            # Create mask for significant vessels
            vessel_mask = max_vesselness > vesselness_threshold
            num_vessels = np.sum(vessel_mask)
            self.logger.info(f"Found {num_vessels} vessel points above threshold")
            
            # Create new polydata with only vessel points
            vessel_points = vtk.vtkPoints()
            vessel_polydata = vtk.vtkPolyData()
            
            # Arrays for final vessel properties
            best_vesselness = vtk.vtkDoubleArray()
            best_vesselness.SetName("vesselness")
            
            best_directions = vtk.vtkDoubleArray()
            best_directions.SetName("direction")
            best_directions.SetNumberOfComponents(3)
            
            best_diameter = vtk.vtkDoubleArray()
            best_diameter.SetName("diameter_mm")
            
            best_area = vtk.vtkDoubleArray()
            best_area.SetName("cross_section_area_mm2")
            
            # Copy vessel points and their properties
            for i in range(num_points):
                if vessel_mask[i]:
                    # Add point
                    vessel_points.InsertNextPoint(polydata.GetPoint(i))
                    
                    # Get properties from best scale
                    scale = best_scale[i]
                    best_vesselness.InsertNextValue(max_vesselness[i])
                    
                    direction = pointData.GetArray(f'direction_scale_{scale}').GetTuple3(i)
                    best_directions.InsertNextTuple3(*direction)
                    
                    diameter = pointData.GetArray(f'diameter_mm_scale_{scale}').GetValue(i)
                    best_diameter.InsertNextValue(diameter)
                    
                    area = pointData.GetArray(f'cross_section_area_mm2_scale_{scale}').GetValue(i)
                    best_area.InsertNextValue(area)
            
            vessel_polydata.SetPoints(vessel_points)
            vessel_polydata.GetPointData().AddArray(best_vesselness)
            vessel_polydata.GetPointData().AddArray(best_directions)
            vessel_polydata.GetPointData().AddArray(best_diameter)
            vessel_polydata.GetPointData().AddArray(best_area)
            
            self.logger.info("Successfully extracted vessel particles")
            
            return vessel_polydata

            
        except Exception as e:
            self.logger.error(f"Error extracting vessel particles: {str(e)}")
            raise