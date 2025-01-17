import numpy as np
from typing import List, Set, Dict, Optional
import logging
from tqdm import tqdm
import os
import SimpleITK as sitk
from .data_structures import VesselNode, Edge, NodeType, VesselType, VesselGraph
from .separator import SubTree

logger = logging.getLogger(__name__)

class VesselClassifier:
    """
    Handles final artery-vein classification.
    Implements methods from Section II-F of the paper.
    """
    
    def __init__(self, graph: VesselGraph, output_dir: str):
        """
        Initialize vessel classifier
        
        Args:
            graph: Input vessel graph
            output_dir: Directory for debug outputs
        """
        self.graph = graph
        self.output_dir = output_dir
        
    def classify_vessels(self, subtrees: List[SubTree], groups: List[Set[int]]) -> None:
        """
        Classify vessels using normalized volume (Section II-F)
        """
        logger.info("Classifying vessels...")
        
        # Track classification statistics
        stats = {
            'arteries': 0,
            'veins': 0,
            'unclassified': 0
        }
        
        # Process each group
        pbar = tqdm(groups, desc="Classifying vessel groups", unit="groups")
        for group in pbar:
            # Calculate normalized volumes
            volumes = {}
            for subtree_id in group:
                subtree = subtrees[subtree_id]
                total_volume = 0.0
                total_length = 0.0
                
                # Calculate volume and length for each edge
                for edge in subtree.edges:
                    # Volume = π(d/2)²l where d is scale (diameter) and l is length
                    volume = np.pi * (edge.scale/2)**2 * edge.length
                    total_volume += volume
                    total_length += edge.length
                    
                # Normalize by total length
                volumes[subtree_id] = total_volume / total_length if total_length > 0 else 0
            
            # Classify based on normalized volume (higher = vein)
            median_volume = np.median(list(volumes.values()))
            for subtree_id in group:
                vessel_type = VesselType.VEIN if volumes[subtree_id] > median_volume else VesselType.ARTERY
                
                # Apply classification to subtree
                subtree = subtrees[subtree_id]
                for node in subtree.nodes:
                    node.vessel_type = vessel_type
                    for edge in node.edges:
                        if edge.start_node in subtree.nodes and edge.end_node in subtree.nodes:
                            edge.vessel_type = vessel_type
                            
                # Update statistics
                if vessel_type == VesselType.ARTERY:
                    stats['arteries'] += 1
                else:
                    stats['veins'] += 1
                    
            pbar.set_postfix(stats)
        
        # Count unclassified vessels
        for edge in self.graph.edges:
            if edge.vessel_type == VesselType.UNKNOWN:
                stats['unclassified'] += 1
                
        logger.info("Classification summary:")
        logger.info(f"- Arteries: {stats['arteries']} segments")
        logger.info(f"- Veins: {stats['veins']} segments")
        logger.info(f"- Unclassified: {stats['unclassified']} segments")
        
        # Save debug visualization
        self._save_debug_visualization()
        
    def save_results(self, vessel_mask: np.ndarray, output_prefix: str) -> None:
        """
        Save classification results as NRRD files
        
        Args:
            vessel_mask: Binary vessel mask from input
            output_prefix: Prefix for output files
        """
        logger.info("Saving classification results...")
        
        # Create classification array
        classification = np.zeros_like(vessel_mask, dtype=np.uint8)
        
        # Count classifications
        stats = {
            'arteries': 0,
            'veins': 0,
            'unclassified': 0,
            'vessel_voxels': np.sum(vessel_mask > 0),
            'classified_voxels': 0
        }
        
        # For each edge in the graph
        pbar = tqdm(self.graph.edges, desc="Classifying vessel voxels")
        for edge in pbar:
            value = 0
            if edge.vessel_type == VesselType.ARTERY:
                value = 1
                stats['arteries'] += 1
            elif edge.vessel_type == VesselType.VEIN:
                value = 2
                stats['veins'] += 1
            else:
                stats['unclassified'] += 1
                continue
            
            # Get all vessel voxels near this edge's path
            for z, y, x in edge.path_points:
                # Look in a small neighborhood around the centerline point
                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if (0 <= nz < vessel_mask.shape[0] and 
                                0 <= ny < vessel_mask.shape[1] and 
                                0 <= nx < vessel_mask.shape[2] and 
                                vessel_mask[nz, ny, nx] > 0):
                                classification[nz, ny, nx] = value
                                stats['classified_voxels'] += 1
            
            pbar.set_postfix(stats)
        
        logger.info("Classification summary:")
        logger.info(f"- Arteries: {stats['arteries']} segments")
        logger.info(f"- Veins: {stats['veins']} segments")
        logger.info(f"- Unclassified: {stats['unclassified']} segments")
        logger.info(f"- Total vessel voxels: {stats['vessel_voxels']}")
        logger.info(f"- Classified voxels: {stats['classified_voxels']}")
        
        # Verify we have some classifications
        if stats['arteries'] == 0 and stats['veins'] == 0:
            logger.warning("No vessels were classified! Output will be empty.")
            
        # Save classification result
        output_img = sitk.GetImageFromArray(classification)
        output_file = f"{output_prefix}_classification.nrrd"
        sitk.WriteImage(output_img, output_file)
        logger.info(f"Results saved to: {output_file}")
        
        # Save debug version showing all vessels
        debug = np.zeros_like(classification)
        debug[vessel_mask > 0] = 3  # Mark all vessel voxels
        debug[classification > 0] = classification[classification > 0]  # Overlay classifications
        
        debug_img = sitk.GetImageFromArray(debug)
        debug_file = f"{output_prefix}_classification_debug.nrrd"
        sitk.WriteImage(debug_img, debug_file)
        logger.info(f"Debug version saved to: {debug_file}")
        
    def _save_debug_visualization(self) -> None:
        """Save debug visualization of vessel classification"""
        # Create visualization array
        viz = np.zeros_like(self.graph.nodes[next(iter(self.graph.nodes))].position, dtype=np.uint8)
        
        # Color vessels by type
        for edge in self.graph.edges:
            for z, y, x in edge.path_points:
                if edge.vessel_type == VesselType.ARTERY:
                    viz[z, y, x] = 1
                elif edge.vessel_type == VesselType.VEIN:
                    viz[z, y, x] = 2
                else:
                    viz[z, y, x] = 3  # Unclassified
                    
        # Save visualization
        output_img = sitk.GetImageFromArray(viz)
        output_file = os.path.join(self.output_dir, 'debug_classification.nrrd')
        sitk.WriteImage(output_img, output_file)
        logger.info(f"Saved classification visualization to: {output_file}") 