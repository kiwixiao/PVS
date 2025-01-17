import numpy as np
from typing import List, Dict, Set, Tuple
import logging
from tqdm import tqdm
import os
import SimpleITK as sitk
from .data_structures import VesselNode, Edge, NodeType, VesselGraph
from .separator import SubTree
from collections import defaultdict
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

class PeripheryMatcher:
    """
    Analyzes relationships between peripheral vessels.
    Implements methods from Section II-D of the paper.
    """
    
    def __init__(self, 
                 graph: VesselGraph,
                 max_periphery_distance: float = 30.0,
                 min_matching_vessels: int = 2,
                 output_dir: str = None):
        """
        Initialize periphery matcher
        
        Args:
            graph: Input vessel graph
            max_periphery_distance: Maximum distance (Dmax) for periphery matching in mm
            min_matching_vessels: Minimum number of matching vessels required
            output_dir: Directory for debug outputs
        """
        self.graph = graph
        self.max_periphery_distance = max_periphery_distance
        self.min_matching_vessels = min_matching_vessels
        self.output_dir = output_dir
        
    def analyze_periphery(self, subtrees: List[SubTree]) -> Dict[int, Dict[int, float]]:
        """Analyze relationships between peripheral vessels"""
        logger.info("Analyzing peripheral relationships...")
        
        # Filter out tiny subtrees (likely noise)
        min_nodes = 5  # Minimum nodes for a valid subtree
        valid_subtrees = [(i, st) for i, st in enumerate(subtrees) if len(st.nodes) >= min_nodes]
        logger.info(f"Filtered {len(subtrees)} subtrees to {len(valid_subtrees)} valid subtrees (min size: {min_nodes})")
        
        # Build spatial index for leaf nodes
        leaf_positions = []  # [(x,y,z), ...]
        leaf_mapping = []    # [(subtree_idx, node), ...]
        
        for subtree_idx, subtree in valid_subtrees:
            leaf_nodes = [n for n in subtree.nodes if len(n.edges) == 1]  # Get leaf nodes
            for leaf in leaf_nodes:
                leaf_positions.append(leaf.position)
                leaf_mapping.append((subtree_idx, leaf))
                
        if not leaf_positions:
            logger.warning("No leaf nodes found in valid subtrees")
            return {}
            
        # Build KD-tree for efficient spatial queries
        tree = cKDTree(leaf_positions)
        logger.info(f"Built spatial index for {len(leaf_positions)} leaf nodes")
        
        # Initialize relationships dictionary for valid subtrees
        relationships = {i: {} for i, _ in valid_subtrees}
        
        # Find relationships using spatial queries
        pairs_analyzed = 0
        relationships_found = 0
        batch_size = 1000  # Process in batches to show progress
        
        with tqdm(total=len(leaf_positions), desc="Analyzing leaf nodes") as pbar:
            for i in range(0, len(leaf_positions), batch_size):
                batch_end = min(i + batch_size, len(leaf_positions))
                batch_positions = leaf_positions[i:batch_end]
                
                # Find all points within max_periphery_distance
                nearby_points = tree.query_ball_point(batch_positions, self.max_periphery_distance)
                
                for j, nearby in enumerate(nearby_points):
                    current_idx = i + j
                    current_subtree_idx = leaf_mapping[current_idx][0]
                    
                    # Group nearby points by subtree
                    subtree_matches = defaultdict(int)
                    for point_idx in nearby:
                        if point_idx <= current_idx:  # Skip duplicates and self
                            continue
                        neighbor_subtree_idx = leaf_mapping[point_idx][0]
                        if neighbor_subtree_idx != current_subtree_idx:
                            subtree_matches[neighbor_subtree_idx] += 1
                    
                    # Record relationships that meet minimum matching criteria
                    for neighbor_idx, match_count in subtree_matches.items():
                        if match_count >= self.min_matching_vessels:
                            relationships[current_subtree_idx][neighbor_idx] = match_count
                            relationships[neighbor_idx][current_subtree_idx] = match_count
                            relationships_found += 1
                            
                    pairs_analyzed += len(nearby)
                
                pbar.update(batch_end - i)
                pbar.set_postfix({
                    "relationships": relationships_found,
                    "pairs": pairs_analyzed
                })
        
        logger.info(f"Found {relationships_found} relationships between subtrees")
        return relationships
        
    def link_subtrees(self, subtrees: List[SubTree], 
                     relationships: Dict[int, Dict[int, float]]) -> List[Set[int]]:
        """
        Link subtrees using direct and indirect relationships (Section II-D.2)
        
        Returns:
            List of sets containing indices of linked subtrees
        """
        logger.info("Linking subtrees...")
        
        groups = []
        visited = set()
        
        def find_all_relationships(subtree_id: int, group: Set[int]):
            """Recursively find all related subtrees"""
            group.add(subtree_id)
            visited.add(subtree_id)
            
            # Direct relationships
            for related_id, strength in relationships[subtree_id].items():
                if related_id not in visited:
                    find_all_relationships(related_id, group)
            
            # Indirect relationships (A ≠ C ∧ B ≠ C ⟹ A = B)
            for i in range(len(subtrees)):
                for j in range(len(subtrees)):
                    if i in group and j in group:
                        for k in range(len(subtrees)):
                            if k not in group and k in relationships[i] and k in relationships[j]:
                                if k not in visited:
                                    find_all_relationships(k, group)
        
        # Build groups
        pbar = tqdm(range(len(subtrees)), desc="Finding linked groups", unit="subtrees")
        for i in pbar:
            if i not in visited and relationships[i]:
                group = set()
                find_all_relationships(i, group)
                if len(group) > 1:  # Only keep groups with multiple subtrees
                    groups.append(group)
                    logger.debug(f"Found linked group with {len(group)} subtrees")
            pbar.set_postfix({"groups": len(groups)})
                    
        logger.info(f"Found {len(groups)} linked subtree groups")
        
        # Save debug visualization
        if self.output_dir:
            self._save_group_visualization(subtrees, groups)
        
        return groups
        
    def _save_debug_visualization(self, subtrees: List[SubTree], 
                                relationships: Dict[int, Dict[int, float]]) -> None:
        """Save debug visualization of peripheral relationships"""
        # Create visualization array
        viz = np.zeros_like(self.graph.nodes[next(iter(self.graph.nodes))].position, dtype=np.uint8)
        
        # Mark all vessel points
        for node in self.graph.nodes.values():
            z, y, x = node.position
            viz[z, y, x] = 1
        
        # Mark leaf nodes that have relationships
        for i, relations in relationships.items():
            if relations:  # If this subtree has any relationships
                subtree = subtrees[i]
                for leaf in subtree.leaf_nodes:
                    z, y, x = leaf.position
                    viz[z, y, x] = 2
                    
        # Save visualization
        output_img = sitk.GetImageFromArray(viz)
        output_file = os.path.join(self.output_dir, 'debug_peripheral_relations.nrrd')
        sitk.WriteImage(output_img, output_file)
        logger.info(f"Saved peripheral relationship visualization to: {output_file}")
        
    def _save_group_visualization(self, subtrees: List[SubTree], groups: List[Set[int]]) -> None:
        """Save debug visualization of linked groups"""
        # Create visualization array
        viz = np.zeros_like(self.graph.nodes[next(iter(self.graph.nodes))].position, dtype=np.uint8)
        
        # Color each group differently
        for i, group in enumerate(groups):
            color = (i % 254) + 1  # Avoid 0, cycle through colors
            for subtree_id in group:
                subtree = subtrees[subtree_id]
                for node in subtree.nodes:
                    z, y, x = node.position
                    viz[z, y, x] = color
                    
        # Save visualization
        output_img = sitk.GetImageFromArray(viz)
        output_file = os.path.join(self.output_dir, 'debug_linked_groups.nrrd')
        sitk.WriteImage(output_img, output_file)
        logger.info(f"Saved linked groups visualization to: {output_file}") 