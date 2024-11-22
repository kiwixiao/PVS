# visualization/visualizer.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import nibabel as nib

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
    def generate_visualizations(self, ct_data: np.ndarray, vessel_mask: np.ndarray,
                              vesselness: np.ndarray, vessel_tree: nx.Graph):
        """Generate visualization overlays for quality check"""
        try:
            self._save_slice_visualizations(ct_data, vessel_mask, vesselness)
            self._save_vessel_tree_visualization(vessel_tree)
            self._save_3d_maximum_intensity_projection(ct_data, vessel_mask)
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise
            
    def _save_slice_visualizations(self, ct_data: np.ndarray, vessel_mask: np.ndarray,
                                 vesselness: np.ndarray):
        """Save middle slice visualizations in each direction"""
        for axis, axis_name in enumerate(['axial', 'coronal', 'sagittal']):
            mid_slice_idx = ct_data.shape[axis] // 2
            
            # Extract slices
            if axis == 0:
                ct_slice = ct_data[mid_slice_idx]
                mask_slice = vessel_mask[mid_slice_idx]
                vesselness_slice = vesselness[mid_slice_idx]
            elif axis == 1:
                ct_slice = ct_data[:, mid_slice_idx]
                mask_slice = vessel_mask[:, mid_slice_idx]
                vesselness_slice = vesselness[:, mid_slice_idx]
            else:
                ct_slice = ct_data[:, :, mid_slice_idx]
                mask_slice = vessel_mask[:, :, mid_slice_idx]
                vesselness_slice = vesselness[:, :, mid_slice_idx]
            
            # Create figure
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original CT
            ax1.imshow(ct_slice, cmap='gray')
            ax1.set_title('CT Image')
            ax1.axis('off')
            
            # Vessel mask
            ax2.imshow(mask_slice, cmap='hot')
            ax2.set_title('Vessel Mask')
            ax2.axis('off')
            
            # Vesselness map
            ax3.imshow(vesselness_slice, cmap='jet')
            ax3.set_title('Vesselness')
            ax3.axis('off')
            
            # Overlay
            overlay = np.zeros((*ct_slice.shape, 3))
            overlay[..., 0] = ct_slice / ct_slice.max()
            overlay[..., 1] = ct_slice / ct_slice.max()
            overlay[..., 2] = ct_slice / ct_slice.max()
            
            overlay[mask_slice > 0] = [1, 0, 0]  # Red for vessels
            
            ax4.imshow(overlay)
            ax4.set_title('Overlay')
            ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{axis_name}_visualization.png',
                       bbox_inches='tight', dpi=300)
            plt.close()
            
    def _save_vessel_tree_visualization(self, vessel_tree: nx.Graph):
        """Generate vessel tree visualization"""
        # Create 3D plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes colored by generation
        generations = nx.get_node_attributes(vessel_tree, 'generation')
        max_gen = max(generations.values())
        
        for node in vessel_tree.nodes():
            gen = generations[node]
            color = plt.cm.viridis(gen / max_gen)
            ax.scatter(*node, c=[color], s=20)
        
        # Plot edges
        for edge in vessel_tree.edges():
            x = [edge[0][0], edge[1][0]]
            y = [edge[0][1], edge[1][1]]
            z = [edge[0][2], edge[1][2]]
            ax.plot(x, y, z, 'gray', alpha=0.5)
        
        ax.set_title('Vessel Tree Structure')
        plt.savefig(self.output_dir / 'vessel_tree_3d.png', dpi=300)
        plt.close()
        
    def _save_3d_maximum_intensity_projection(self, ct_data: np.ndarray, 
                                            vessel_mask: np.ndarray):
        """Generate maximum intensity projections"""
        for axis, axis_name in enumerate(['axial', 'coronal', 'sagittal']):
            # CT MIP
            ct_mip = np.max(ct_data, axis=axis)
            
            # Vessel MIP
            vessel_mip = np.max(vessel_mask, axis=axis)
            
            # Create overlay
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(ct_mip, cmap='gray')
            ax1.set_title(f'{axis_name.capitalize()} CT MIP')
            ax1.axis('off')
            
            ax2.imshow(ct_mip, cmap='gray')
            ax2.imshow(vessel_mip, cmap='hot', alpha=0.5)
            ax2.set_title(f'{axis_name.capitalize()} Vessel Overlay')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{axis_name}_mip.png', dpi=300)
            plt.close()
            
    def save_nifti_outputs(self, data: np.ndarray, affine: np.ndarray, 
                          name: str):
        """Save visualization results as NIfTI"""
        nifti = nib.Nifti1Image(data, affine)
        nib.save(nifti, self.output_dir / f'{name}.nii.gz')