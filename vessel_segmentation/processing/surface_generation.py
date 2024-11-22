# processing/surface_generation.py
import vtk
from vtk.util import numpy_support
import numpy as np
import networkx as nx
from pathlib import Path
import logging
from skimage import measure
import nibabel as nib

class SurfaceGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create debug directory
        self.debug_dir = self.output_dir / 'debug'
        self.debug_dir.mkdir(exist_ok=True)

    def generate_surface(self, vessel_mask: np.ndarray, vessel_tree: nx.Graph,
                        voxel_size: float = 1.0) -> str:
        """Generate 3D surface model with vessel properties"""
        try:
            self.logger.info("Starting surface generation")
            self._save_debug_data(vessel_mask, 'input_mask')
            
            # First try marching cubes from skimage
            self.logger.info("Generating surface using marching cubes")
            verts, faces, normals, values = measure.marching_cubes(
                vessel_mask,
                level=0.5,
                spacing=(voxel_size, voxel_size, voxel_size)
            )
            # Save intermediate OBJ for debugging
            self._save_obj(verts, faces, self.debug_dir / 'mc_surface.obj')
            # Create VTK surface
            surface = self._create_vtk_surface(verts, faces, normals)
            # Smooth the surface
            self.logger.info("Smoothing surface")
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputData(surface)
            smoother.SetNumberOfIterations(20)
            smoother.SetPassBand(0.1)
            smoother.BoundarySmoothingOn()
            smoother.FeatureEdgeSmoothingOn()
            smoother.Update()
            # Save smoothed intermediate
            self._save_vtk(smoother.GetOutput(), self.debug_dir / 'smoothed_surface.vtk')
            # Add vessel properities
            surface_with_props = self._add_vessel_properties(
                smoother.Getput(),
                vessel_tree,
                verts
            )
            # Save final surface
            output_path = str(self.output_dir / 'vessel_surface.vtk')
            self._save_vtk(surface_with_props, output_path)
            
            # Save statistics
            stats = {
                'vertex_count': verts.shape[0],
                'face_count': faces.shape[0],
                'surface_area': self._compute_surface_area(surface_with_props),
                'volume': self._compute_volume(surface_with_props)
            }
            self._save_stats('surface_stats.json', stats)
            
            # # Create VTK image from mask
            # vtk_data = numpy_support.numpy_to_vtk(
            #     vessel_mask.ravel(),
            #     deep=True,
            #     array_type=vtk.VTK_UNSIGNED_CHAR
            # )
            
            # image = vtk.vtkImageData()
            # image.SetDimensions(vessel_mask.shape)
            # image.SetSpacing(voxel_size, voxel_size, voxel_size)
            # image.GetPointData().SetScalars(vtk_data)
            
            # # Generate surface using Marching Cubes
            # surface = vtk.vtkMarchingCubes()
            # surface.SetInputData(image)
            # surface.SetValue(0, 0.5)
            # surface.ComputeNormalsOn()
            
            # # Add vessel properties
            # self._add_vessel_properties(surface, vessel_tree)
            
            # # Save surface
            # output_path = str(self.output_dir / 'vessel_surface.vtk')
            # writer = vtk.vtkPolyDataWriter()
            # writer.SetInputConnection(surface.GetOutputPort())
            # writer.SetFileName(output_path)
            # writer.Write()
            
            self.logger.info(f"Saved surface model to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating surface: {str(e)}")
            raise
        
    def _create_vtk_surface(self, verts, faces, normals):
        """Create VTK PolyData from vertices and faces"""
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(verts))
        
        cells = vtk.vtkCellArray()
        for face in faces:
            cells.InsertNextCell(3)
            for point_id in face:
                cells.InsertCellPoint(point_id)
        
        surface = vtk.vtkPolyData()
        surface.SetPoints(points)
        surface.SetPolys(cells)
        
        # Add normals
        if normals is not None:
            vtk_normals = numpy_support.numpy_to_vtk(normals)
            surface.GetPointData().setNormals(vtk_normals)
            
        return surface
        
    def _save_obj(self, verts, faces, filepath):
        """Save mesh as OBJ file"""
        with open(filepath, 'w') as f:
            for v in verts:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for face in faces:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    
    def _save_debug_data(self, data, name):
        """Save numpy array as NIFTI for debugging"""
        nib.save(
            nib.Nifti1Image(data, np.eye(4)),
            self.debug_dir / f'{name}.nii.gz'
        )
        
    def _save_vtk(self, polydata, filepath):
        """Save VTK PolyData"""
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(filepath))
        writer.SetInputData(polydata)
        writer.Write()

    def _save_stats(self, filename, stats):
        """Save statistics to JSON"""
        with open(self.output_dir / filename, 'w') as f:
            json.dump(stats, f, indent=2)

    def _compute_surface_area(self, polydata):
        """Compute surface area of mesh"""
        mass = vtk.vtkMassProperties()
        mass.SetInputData(polydata)
        return mass.GetSurfaceArea()

    def _compute_volume(self, polydata):
        """Compute volume of mesh"""
        mass = vtk.vtkMassProperties()
        mass.SetInputData(polydata)
        return mass.GetVolume()
            
    def _add_vessel_properties(self, surface: vtk.vtkMarchingCubes, vessel_tree: nx.Graph):
        """Add vessel properties as point data"""
        # Create arrays for properties
        radius_array = vtk.vtkFloatArray()
        radius_array.SetName("Radius")
        
        generation_array = vtk.vtkIntArray()
        generation_array.SetName("Generation")
        
        vesselness_array = vtk.vtkFloatArray()
        vesselness_array.SetName("Vesselness")
        
        eigenvalues_array = vtk.vtkFloatArray()
        eigenvalues_array.SetName("HessianEigenvalues")
        eigenvalues_array.SetNumberOfComponents(3)
        
        # Map properties from centerline points to surface points
        for node in vessel_tree.nodes():
            props = vessel_tree.nodes[node]
            
            # Add properties to arrays
            radius_array.InsertNextValue(props['radius'])
            generation_array.InsertNextValue(props['generation'])
            vesselness_array.InsertNextValue(props['vesselness'])
            eigenvalues_array.InsertNextTuple3(*props['eigenvalues'])
        
        # Add arrays to surface
        surface.GetOutput().GetPointData().AddArray(radius_array)
        surface.GetOutput().GetPointData().AddArray(generation_array)
        surface.GetOutput().GetPointData().AddArray(vesselness_array)
        surface.GetOutput().GetPointData().AddArray(eigenvalues_array)

    def export_stl(self, vessel_mask: np.ndarray, voxel_size: float = 1.0) -> str:
        """Export vessel surface as STL file"""
        try:
            # Create VTK image
            vtk_data = numpy_support.numpy_to_vtk(
                vessel_mask.ravel(),
                deep=True,
                array_type=vtk.VTK_UNSIGNED_CHAR
            )
            
            image = vtk.vtkImageData()
            image.SetDimensions(vessel_mask.shape)
            image.SetSpacing(voxel_size, voxel_size, voxel_size)
            image.GetPointData().SetScalars(vtk_data)
            
            # Generate surface
            surface = vtk.vtkMarchingCubes()
            surface.SetInputData(image)
            surface.SetValue(0, 0.5)
            surface.ComputeNormalsOn()
            
            # Save as STL
            output_path = str(self.output_dir / 'vessel_surface.stl')
            writer = vtk.vtkSTLWriter()
            writer.SetInputConnection(surface.GetOutputPort())
            writer.SetFileName(output_path)
            writer.Write()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting STL: {str(e)}")
            raise