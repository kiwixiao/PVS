# core/vessel_extractor.py
import numpy as np
import networkx as nx
from scipy.ndimage import label, distance_transform_edt
from skimage import morphology, filters
from collections import deque
import logging
from pathlib import Path
from typing import Dict, Tuple
from skimage.filters import threshold_otsu
import nibabel as nib
import json

class VesselExtractor:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

    def extract_vessels(self, vesselness: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
        """Extract vessel mask from vesselness map"""
        try:
            self.logger.info("Starting vessel mask extration")
            self.logger.info(f"Vesselness shape: {vesselness.shape}")
            self.logger.info(f"Lung mask shape: {lung_mask.shape}")
            
            if vesselness.shape != lung_mask.shape:
                self.logger.error(f"Shape mismatch: vesselness {vesselness.shape} vs lung mask {lung_mask.shape}")
                raise ValueError("Input data must have the same isotropic dimensions")
            
            # Save vesselness for debugging
            self._save_debug_nifti(vesselness, 'vesselness_input')
            self._save_debug_nifti(lung_mask, 'lung_mask_input')
            
            # Normalize vesselness to [0,1]
            v_min, v_max = vesselness.min(), vesselness.max()
            self.logger.info(f"Vesselness range: [{v_min}, {v_max}]")
            v_norm = (vesselness - v_min()) / (v_max() - v_min() + 1e-10)
            
            self._save_debug_nifti(v_norm, 'vesselness_normalized')
            
            # Automatic thresholding, only take the non zero values
            non_zero_values = v_norm[v_norm > 0]
            if len(non_zero_values) > 0:
                threshold = threshold_otsu(non_zero_values)
                self.logger.info(f"Otsu threshold: {threshold}")
            else:
                threshold = 0.5
                self.logger.warning("No non-zero values found, using default threshold: 0.5")
            
            # Create binary mask
            vessel_mask = v_norm > threshold
            self._save_debug_nifti(vessel_mask.astype(np.float32), 'vessel_mask_initial')
            
            # Apply lung mask
            vessel_mask = vessel_mask & (lung_mask > 0)
            self._save_debug_nfiti(vessel_mask.astype(np.float32), 'vessel_mask_lung_masked')
            
            # Save statistics
            stats = {
                'vesselness_range': [float(v_min), float(v_max)],
                'threshold': float(threshold),
                'vessel_voxel_count': int(np.sum(vessel_mask)),
                'vessel_volume_fraction': float(np.sum(vessel_mask) / np.sum(lung_mask > 0))
            }
            self._save_stats('vessel_extration_stats.json', stats)
            
            # Remove small components, commnet out for now.
            # labeled, num = label(vessel_mask)
            # if num > 0:
            #     sizes = np.bincount(labeled.ravel())[1:]
            #     min_size = max(50, int(0.001 * vessel_mask.size))
                
            #     for i in range(1, num + 1):
            #         if sizes[i - 1] < min_size:
            #             vessel_mask[labeled == i] = False
            
            return vessel_mask.astype(bool)
            
        except Exception as e:
            self.logger.error(f"Error extracting vessels: {str(e)}")
            raise
    
    def _save_debug_nifti(self, data: np.ndarray, name: str):
        """
        Save intermediate results fro debugging
        """
        output_path = self.output_dir / f"debug_{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), output_path)
        self.logger.info(f"Saved debug output: {output_path}")
    
    def _save_stats(self, filename: str, stats: dict):
        """Save statistics to JASON"""
        with open(self.output_dir / filename, 'w') as f:
            json.dump(stats, f, ident=2)

    def build_vessel_tree(self, vessel_mask: np.ndarray, 
                         vesselness: np.ndarray,
                         eigenvalues: Dict[str, np.ndarray]) -> nx.Graph:
        """Build vessel tree graph representation"""
        try:
            # Check if vessel mask is empty
            if not np.any(vessel_mask):
                self.logger.warning("Empty vessel mask provide")
                return nx.Graph()
            
            # Extract centerlines
            self.logger.info("Extracting centerlines")
            skeleton = morphology.skeletonize(vessel_mask)
            
            # Calculate distance transform for radii
            distance = distance_transform_edt(vessel_mask)
            
            # Create graph
            G = nx.Graph()
            
            # Get centerline points
            points = np.argwhere(skeleton > 0)
            
            if len(points) == 0:
                self.logger.warning("No centerline points found")
                return G
            
            # Add nodes with properties
            for point in points:
                x, y, z = point
                radius = distance[x, y, z]
                if radius > 0: # Only add points with valid radius
                    node_props = {
                        'coordinates': point,
                        'radius': radius,
                        'vesselness': vesselness[x, y, z],
                        'eigenvalues': [
                            float(eigenvalues['lambda1'][x, y, z]),
                            float(eigenvalues['lambda2'][x, y, z]),
                            float(eigenvalues['lambda3'][x, y, z])
                        ]
                    }
                    G.add_node(tuple(point), **node_props)
            
            # Connect neighboring points
            if len(G.nodes) > 0:
                self._connect_vessel_points(G, skeleton)
            
                # Assign generation numbers
                self._assign_generations(G)
            
                # Save intermediate results
                self._save_tree_data(G, skeleton, distance)
            else:
                self.loger.warning("No valid nodes found for vessel tree")
                    
            return G
            
        except Exception as e:
            self.logger.error(f"Error building vessel tree: {str(e)}")
            raise
            
    def _connect_vessel_points(self, G: nx.Graph, skeleton: np.ndarray):
        """Connect neighboring points in the skeleton"""
        if not G.nodes:
            return
        
        for point in G.nodes():
            x, y, z = point
            # Check 26-neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        neighbor = (x + dx, y + dy, z + dz)
                        if neighbor in G:
                            G.add_edge(point, neighbor)

    def _assign_generations(self, G: nx.Graph):
        """Assign generation numbers to vessel segments"""
        if not G.nodes:
            return

        try:
            # Find root (largest vessel)
            root = max(G.nodes(), key=lambda n: G.nodes[n]['radius'])
        except ValueError:
            self.logger.warning("Cannot assign generations: no nodes in graph")
            return
        
        generations = {root: 0}
        queue = deque([(root, 0)])
        
        while queue:
            node, gen = queue.popleft() # O(1) operation with deque
            for neighbor in G.neighbors(node):
                if neighbor not in generations:
                    # Increment generation if radius decreases significantly
                    curr_radius = G.nodes[node]['radius']
                    next_radius = G.nodes[neighbor]['radius']
                    if next_radius < curr_radius * 0.8:
                        generations[neighbor] = gen + 1
                    else:
                        generations[neighbor] = gen
                    queue.append((neighbor, generations[neighbor]))
        
        nx.set_node_attributes(G, generations, 'generation')

    def _save_tree_data(self, G: nx.Graph, skeleton: np.ndarray, 
                       distance: np.ndarray):
        """Save vessel tree data for analysis"""
        tree_dir = self.output_dir / 'vessel_tree'
        tree_dir.mkdir(exist_ok=True)
        
        # Save graph
        nx.write_gpickle(G, tree_dir / 'vessel_tree.gpickle')
        
        # Save numpy arrays
        np.save(tree_dir / 'centerlines.npy', skeleton)
        np.save(tree_dir / 'radius_map.npy', distance)