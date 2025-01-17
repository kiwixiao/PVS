import numpy as np
import SimpleITK as sitk
from typing import List, Set, Tuple, Dict, Optional
from collections import defaultdict
from .vessel_graph import VesselGraph, VesselNode, NodeType, Edge
import logging
from scipy.spatial import cKDTree
from queue import Queue
import os
import copy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArteryVeinSeparator:
    def __init__(self, 
                 centerline_file: str,
                 point_type_file: str,
                 vessel_mask_file: str,
                 lung_mask_file: str,
                 min_branch_length: int = 5,
                 max_periphery_distance: float = 30.0,
                 min_matching_vessels: int = 2):
        """
        Initialize the artery-vein separator following Charbonnier's methodology
        
        Args:
            centerline_file: Path to centerline NRRD file (X = {xk})
            point_type_file: Path to point type NRRD file
            vessel_mask_file: Path to vessel mask NRRD file (V)
            lung_mask_file: Path to lung mask NRRD file (L)
        """
        self.min_branch_length = min_branch_length
        self.max_periphery_distance = max_periphery_distance
        self.min_matching_vessels = min_matching_vessels
        
        # Load the data
        self.centerline = sitk.GetArrayFromImage(sitk.ReadImage(centerline_file))
        self.point_types = sitk.GetArrayFromImage(sitk.ReadImage(point_type_file))
        self.vessel_mask = sitk.GetArrayFromImage(sitk.ReadImage(vessel_mask_file))
        self.lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_file))
        
        # Initialize graph and nodes
        self.graph = VesselGraph()
        self.nodes = {}
        
    def build_geometric_graph(self) -> None:
        """
        Two-stage geometric graph construction (Section II-B)
        G^I = (N, E) -> G
        """
        logger.info("Stage 1: Building initial geometric graph G^I...")
        
        # Find vessel-lung border intersection (Vb ∩ Lb)
        lung_border = self._get_lung_border()
        
        # Create nodes for points where Ωv(xk) ≠ 2
        points = np.argwhere(self.centerline > 0)
        pbar = tqdm(points, desc="Creating nodes", unit="points")
        for z, y, x in pbar:
            neighbors = self._count_neighbors(self.centerline, z, y, x)
            pos = (z, y, x)
            
            if neighbors != 2:  # ni = {xk : Ωv(xk) ≠ 2, xk are connected}
                if neighbors == 1:
                    # Root node detection based on border position
                    if self._is_at_lung_border(pos, lung_border):
                        node_type = NodeType.ROOT
                    else:
                        node_type = NodeType.ENDPOINT
                else:
                    node_type = NodeType.BIFURCATION
                    
                # Create node
                node = VesselNode(pos, node_type)
                self.nodes[pos] = node
                self.graph.nodes[pos] = node
                pbar.set_postfix({"nodes": len(self.nodes)})
        
        # Create edges ei,j = {xk : Ωv(xk) = 2, xk are connected}
        visited = set()
        pbar = tqdm(self.nodes.keys(), desc="Creating edges", unit="nodes")
        for node_pos in pbar:
            if node_pos not in visited:
                self._trace_paths_from_node(self.centerline, node_pos, visited)
                pbar.set_postfix({"edges": len(self.graph.edges)})
        
        logger.info("Stage 2: Refining to geometric graph G...")
        self.graph.refine_graph()
        
        logger.info(f"Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        
    def _get_lung_border(self) -> np.ndarray:
        """Get lung border Lb"""
        from scipy.ndimage import binary_erosion
        border = self.lung_mask - binary_erosion(self.lung_mask)
        return border
        
    def detect_attachment_points(self) -> List[VesselNode]:
        """
        Enhanced attachment point detection using leaf node pruning and path analysis (Section II-C)
        
        1) Leaf Node Pruning: Iteratively remove leaf nodes until early termination
        2) Path Analysis: Find sharp transitions in paths between root nodes
        """
        logger.info("Starting leaf node pruning...")
        
        # 1. Leaf Node Pruning - creates pruned graph Gp
        root_nodes = self.graph.get_root_nodes()
        logger.info(f"Initial state: {len(root_nodes)} root nodes, {len(self.graph.get_leaf_nodes())} leaf nodes")
        
        # Keep original graph for later
        Gp = copy.deepcopy(self.graph)
        
        # Prune until early termination (when more than one root node is reachable)
        pbar = tqdm(desc="Pruning leaf nodes", unit="iterations")
        while True:
            leaf_nodes = Gp.get_leaf_nodes()
            if not leaf_nodes:
                break
                
            # Count reachable root nodes from each leaf
            reachable_roots = set()
            for leaf in tqdm(leaf_nodes, desc="Checking reachability", leave=False):
                for root in root_nodes:
                    path = self._find_path_between_nodes(Gp, leaf, root)
                    if path is not None:
                        reachable_roots.add(root)
                        if len(reachable_roots) > 1:
                            break
                if len(reachable_roots) > 1:
                    break
            
            # Early termination if more than one root is reachable
            if len(reachable_roots) > 1:
                logger.info(f"Early termination: {len(reachable_roots)} root nodes reachable")
                break
                
            # Remove leaf nodes
            for leaf in leaf_nodes:
                if len(leaf.edges) == 1:
                    edge = leaf.edges[0]
                    other_node = edge.end_node if edge.start_node == leaf else edge.start_node
                    
                    # Remove edge and node
                    leaf.edges.clear()
                    other_node.edges.remove(edge)
                    Gp.edges.remove(edge)
                    del Gp.nodes[leaf.position]
            
            pbar.update(1)
            pbar.set_postfix({"nodes": len(Gp.nodes)})
        pbar.close()
        
        logger.info(f"After pruning: {len(Gp.nodes)} nodes remain in pruned graph")
        
        # 2. Path Analysis between root nodes in pruned graph
        logger.info("Analyzing paths between root nodes...")
        root_nodes = Gp.get_root_nodes()
        logger.info(f"Analyzing paths between {len(root_nodes)} root nodes")
        
        # Collect all transition angles in normal branches
        angles = []
        paths = []
        pbar = tqdm(enumerate(root_nodes), desc="Analyzing root paths", total=len(root_nodes))
        for i, root1 in pbar:
            for root2 in root_nodes[i+1:]:
                path = self._find_path_between_nodes(Gp, root1, root2)
                if path is not None:
                    paths.append(path)
                    # Calculate transition angles along path
                    for idx in range(1, len(path)-1):
                        node = path[idx]
                        if len(node.edges) > 1:
                            # Get edges along the path
                            prev_edge = next(e for e in node.edges 
                                           if path[idx-1] in [e.start_node, e.end_node])
                            next_edge = next(e for e in node.edges 
                                           if path[idx+1] in [e.start_node, e.end_node])
                            angle = node.calculate_transition_angle(prev_edge, next_edge)
                            angles.append(angle)
            pbar.set_postfix({"paths": len(paths), "angles": len(angles)})
        
        # Calculate patient-specific threshold α (15th percentile as per paper)
        if angles:
            alpha = np.percentile(angles, 15)
            logger.info(f"Patient-specific angle threshold α: {alpha:.2f} degrees")
        else:
            alpha = 75.0  # fallback threshold
            logger.warning("No angles found, using fallback threshold α: 75.0 degrees")
        
        # Detect attachment points based on sharp transitions
        attachment_points = []
        for path in paths:
            for idx in range(1, len(path)-1):
                node = path[idx]
                if len(node.edges) > 1:
                    # Get edges along the path
                    prev_edge = next(e for e in node.edges 
                                   if path[idx-1] in [e.start_node, e.end_node])
                    next_edge = next(e for e in node.edges 
                                   if path[idx+1] in [e.start_node, e.end_node])
                    angle = node.calculate_transition_angle(prev_edge, next_edge)
                    
                    # Check if angle is below threshold (indicates sharp transition)
                    if angle < alpha:
                        node.node_type = NodeType.ATTACHMENT
                        if node not in attachment_points:
                            attachment_points.append(node)
        
        logger.info(f"Found {len(attachment_points)} attachment points")
        return attachment_points
        
    def _find_path_between_nodes(self, graph: VesselGraph, start: VesselNode, end: VesselNode) -> Optional[List[VesselNode]]:
        """
        Find path between two nodes using BFS with enhanced smoothness criteria
        Returns the path if found, None otherwise
        """
        if start == end:
            return [start]
            
        # Use BFS with a queue of (node, path, last_direction) tuples
        queue = Queue()
        # Initialize with None direction since we don't have a previous edge
        queue.put((start, [start], None))
        visited = {start}
        
        best_path = None
        best_path_score = float('inf')
        
        while not queue.empty():
            current, path, last_direction = queue.get()
            
            # Get all possible next edges
            for edge in current.edges:
                next_node = edge.end_node if edge.start_node == current else edge.start_node
                
                if next_node == end:
                    # Found a path to target, calculate its smoothness score
                    new_path = path + [next_node]
                    path_score = self._calculate_path_smoothness(new_path)
                    if path_score < best_path_score:
                        best_path = new_path
                        best_path_score = path_score
                    continue
                    
                if next_node not in visited:
                    # Calculate direction of current edge
                    current_direction = np.array(next_node.position) - np.array(current.position)
                    current_direction = current_direction / np.linalg.norm(current_direction)
                    
                    # Check angle with previous direction if it exists
                    if last_direction is not None:
                        angle = np.degrees(np.arccos(np.clip(np.dot(last_direction, current_direction), -1.0, 1.0)))
                        if angle > 90:  # Avoid sharp turns
                            continue
                    
                    # Add unvisited node to queue
                    visited.add(next_node)
                    queue.put((next_node, path + [next_node], current_direction))
        
        return best_path
        
    def _calculate_path_smoothness(self, path: List[VesselNode]) -> float:
        """
        Calculate smoothness score for a path based on angles between consecutive segments
        Lower score means smoother path
        """
        if len(path) < 3:
            return 0.0
            
        total_angle = 0.0
        for i in range(1, len(path)-1):
            # Get vectors to previous and next nodes
            prev_vec = np.array(path[i-1].position) - np.array(path[i].position)
            next_vec = np.array(path[i+1].position) - np.array(path[i].position)
            
            # Normalize vectors
            prev_vec = prev_vec / np.linalg.norm(prev_vec)
            next_vec = next_vec / np.linalg.norm(next_vec)
            
            # Calculate angle
            angle = np.degrees(np.arccos(np.clip(np.dot(prev_vec, next_vec), -1.0, 1.0)))
            total_angle += angle
            
        # Also consider path length as a tiebreaker
        length_penalty = len(path) * 0.1
        return total_angle + length_penalty
        
    def _prune_leaf_nodes(self) -> VesselGraph:
        """Iteratively remove leaf nodes until early termination or completion"""
        pruned = True
        while pruned:
            leaves = self.graph.get_leaf_nodes()
            if not leaves:
                break
                
            pruned = False
            for leaf in leaves:
                if len(leaf.edges) == 1:
                    # Remove leaf node
                    edge = leaf.edges[0]
                    other_node = edge.end_node if edge.start_node == leaf else edge.start_node
                    
                    # Remove edge
                    leaf.edges.remove(edge)
                    other_node.edges.remove(edge)
                    self.graph.edges.remove(edge)
                    
                    # Remove node
                    del self.graph.nodes[leaf.position]
                    pruned = True
                    
            # Check for early termination (more than one root node)
            if len(self.graph.get_root_nodes()) > 1:
                break
                
        return self.graph
        
    def create_subtrees(self, attachment_points: List[VesselNode]) -> List[Set[VesselNode]]:
        """
        Create set S of subtrees Ψj that don't contain artery-vein attachments (Section II-C.2)
        """
        logger.info("Creating subtrees...")
        
        # Remove edges at attachment points
        edges_to_remove = []
        for node in attachment_points:
            edges_to_remove.extend(node.edges)
            
        for edge in edges_to_remove:
            edge.start_node.edges.remove(edge)
            edge.end_node.edges.remove(edge)
            self.graph.edges.remove(edge)
            
        # Find connected components (subtrees)
        subtrees = []
        visited = set()
        
        def dfs(node: VesselNode, subtree: Set[VesselNode]):
            subtree.add(node)
            visited.add(node)
            for edge in node.edges:
                next_node = edge.end_node if edge.start_node == node else edge.start_node
                if next_node not in visited:
                    dfs(next_node, subtree)
                    
        for node in self.graph.nodes.values():
            if node not in visited and node.node_type != NodeType.ATTACHMENT:
                subtree = set()
                dfs(node, subtree)
                if subtree:
                    subtrees.append(subtree)
                    
        logger.info(f"Created {len(subtrees)} subtrees")
        return subtrees
        
    def analyze_periphery(self, subtrees: List[Set[VesselNode]]) -> Dict[int, Dict[int, float]]:
        """
        Analyze peripheral vessels using periphery matching (Section II-D.1)
        """
        logger.info("Analyzing peripheral relationships...")
        
        # Build spatial index
        self.graph.build_spatial_index()
        
        # Initialize relationship matrix
        relationships = defaultdict(dict)
        
        # For each pair of subtrees
        pbar = tqdm(enumerate(subtrees), desc="Analyzing subtree pairs", total=len(subtrees))
        for i, subtree1 in pbar:
            leaves1 = [node for node in subtree1 if len(node.edges) == 1]
            
            for j, subtree2 in enumerate(subtrees[i+1:], i+1):
                leaves2 = [node for node in subtree2 if len(node.edges) == 1]
                
                # Count matching vessels within Dmax
                matches = 0
                for leaf1 in leaves1:
                    nearby = self.graph.find_neighbors_radius(
                        leaf1.position,
                        self.max_periphery_distance  # Dmax = 30mm
                    )
                    matches += sum(1 for node in nearby if node in leaves2)
                    
                # Calculate IMS if enough matches found
                if matches >= self.min_matching_vessels:  # Minimum 2 matches required
                    ims = matches / min(len(leaves1), len(leaves2))
                    relationships[i][j] = ims
                    relationships[j][i] = ims
            
            pbar.set_postfix({"relationships": len(relationships)})
                    
        logger.info(f"Found {len(relationships)} subtree relationships")
        return relationships
        
    def link_subtrees(self, subtrees: List[Set[VesselNode]], 
                     relationships: Dict[int, Dict[int, float]]) -> List[Set[int]]:
        """
        Link subtrees using direct and indirect relationships (Section II-D.2)
        """
        logger.info("Linking subtrees...")
        
        groups = []
        visited = set()
        
        def find_all_relationships(subtree_id: int, group: Set[int]):
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
                                group.add(k)
        
        # Build groups
        for i in range(len(subtrees)):
            if i not in visited and relationships[i]:
                group = set()
                find_all_relationships(i, group)
                if len(group) > 1:
                    groups.append(group)
                    
        logger.info(f"Found {len(groups)} linked subtree groups")
        return groups
        
    def classify_vessels(self, subtrees: List[Set[VesselNode]], groups: List[Set[int]]) -> None:
        """
        Classify vessels using normalized volume (Section II-F)
        """
        logger.info("Classifying vessels...")
        
        for group in groups:
            # Calculate normalized volumes
            volumes = {}
            for subtree_id in group:
                total_volume = 0.0
                total_length = 0.0
                
                # Calculate volume and length for each edge
                for node in subtrees[subtree_id]:
                    for edge in node.edges:
                        if edge.start_node in subtrees[subtree_id] and edge.end_node in subtrees[subtree_id]:
                            # Volume = π(d/2)²l
                            volume = np.pi * (edge.mean_diameter/2)**2 * edge.length
                            total_volume += volume
                            total_length += edge.length
                            
                # Normalize by total length
                volumes[subtree_id] = total_volume / total_length if total_length > 0 else 0
            
            # Classify based on normalized volume (higher = vein)
            median_volume = np.median(list(volumes.values()))
            for subtree_id in group:
                vessel_type = 'vein' if volumes[subtree_id] > median_volume else 'artery'
                # Apply classification
                for node in subtrees[subtree_id]:
                    node.vessel_type = vessel_type
                    for edge in node.edges:
                        if edge.start_node in subtrees[subtree_id] and edge.end_node in subtrees[subtree_id]:
                            edge.vessel_type = vessel_type
                            
        logger.info("Vessel classification completed")
        
    def save_results(self, output_prefix: str) -> None:
        """Save the classification results"""
        logger.info("Saving results...")
        
        # Create classification array based on vessel mask size
        classification = np.zeros_like(self.vessel_mask, dtype=np.uint8)
        
        # Count classifications
        artery_count = 0
        vein_count = 0
        unclassified_count = 0
        
        # For each edge in the graph
        pbar = tqdm(self.graph.edges, desc="Classifying vessel voxels")
        for edge in pbar:
            value = 0
            if edge.vessel_type == 'artery':
                value = 1
                artery_count += 1
            elif edge.vessel_type == 'vein':
                value = 2
                vein_count += 1
            else:
                unclassified_count += 1
                continue
            
            # Get all vessel voxels near this edge's path
            for z, y, x in edge.path_points:
                # Look in a small neighborhood around the centerline point
                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if (0 <= nz < self.vessel_mask.shape[0] and 
                                0 <= ny < self.vessel_mask.shape[1] and 
                                0 <= nx < self.vessel_mask.shape[2] and 
                                self.vessel_mask[nz, ny, nx] > 0):
                                classification[nz, ny, nx] = value
            
            pbar.set_postfix({
                "arteries": artery_count,
                "veins": vein_count,
                "unclassified": unclassified_count
            })
        
        logger.info(f"Classification summary:")
        logger.info(f"- Arteries: {artery_count} segments")
        logger.info(f"- Veins: {vein_count} segments")
        logger.info(f"- Unclassified: {unclassified_count} segments")
        logger.info(f"- Total vessel voxels: {np.sum(self.vessel_mask > 0)}")
        logger.info(f"- Classified voxels: {np.sum(classification > 0)}")
        
        # Verify we have some classifications
        if artery_count == 0 and vein_count == 0:
            logger.warning("No vessels were classified! Output will be empty.")
            
        # Save as NRRD with same metadata as input
        input_img = sitk.ReadImage(self.vessel_mask_file)
        output_img = sitk.GetImageFromArray(classification)
        output_img.CopyInformation(input_img)
        
        output_file = f"{output_prefix}_classification.nrrd"
        sitk.WriteImage(output_img, output_file)
        logger.info(f"Results saved to: {output_file}")
        
        # Save a debug version showing all vessels
        debug = np.zeros_like(classification)
        debug[self.vessel_mask > 0] = 3  # Mark all vessel voxels
        debug[classification > 0] = classification[classification > 0]  # Overlay classifications
        
        debug_file = f"{output_prefix}_classification_debug.nrrd"
        debug_img = sitk.GetImageFromArray(debug)
        debug_img.CopyInformation(input_img)
        sitk.WriteImage(debug_img, debug_file)
        logger.info(f"Debug version saved to: {debug_file}")
        
    def _save_debug_image(self, array: np.ndarray, output_file: str) -> None:
        """Helper to save debug images with proper metadata"""
        debug_img = sitk.GetImageFromArray(array)
        debug_img.CopyInformation(sitk.ReadImage(self.vessel_mask_file))
        sitk.WriteImage(debug_img, output_file)
        logger.info(f"Saved debug output: {output_file}")

    def save_debug_outputs(self, output_dir: str) -> None:
        """Save intermediate results for debugging"""
        logger.info("Saving debug outputs...")
        
        # 1. Initial graph structure
        graph_viz = np.zeros_like(self.vessel_mask, dtype=np.uint8)
        # Mark different node types
        for node in self.graph.nodes.values():
            z, y, x = node.position
            if node.node_type == NodeType.ROOT:
                graph_viz[z, y, x] = 1  # Root nodes
            elif node.node_type == NodeType.BIFURCATION:
                graph_viz[z, y, x] = 2  # Bifurcation nodes
            elif node.node_type == NodeType.ENDPOINT:
                graph_viz[z, y, x] = 3  # Endpoint nodes
        # Mark edges
        for edge in self.graph.edges:
            for z, y, x in edge.path_points:
                if graph_viz[z, y, x] == 0:  # Don't overwrite nodes
                    graph_viz[z, y, x] = 4  # Edge points
        self._save_debug_image(graph_viz, os.path.join(output_dir, 'debug_graph_structure.nrrd'))
        
        # 2. Attachment points
        attachment_viz = np.zeros_like(self.vessel_mask, dtype=np.uint8)
        attachment_viz[self.centerline > 0] = 1  # All centerline points
        for node in self.graph.nodes.values():
            if node.node_type == NodeType.ATTACHMENT:
                z, y, x = node.position
                attachment_viz[z, y, x] = 2  # Attachment points
        self._save_debug_image(attachment_viz, os.path.join(output_dir, 'debug_attachment_points.nrrd'))
        
        # 3. Subtrees
        subtree_viz = np.zeros_like(self.vessel_mask, dtype=np.uint8)
        for i, subtree in enumerate(self._current_subtrees):
            color = (i % 254) + 1  # Avoid 0, cycle through colors
            for node in subtree:
                for edge in node.edges:
                    if edge.start_node in subtree and edge.end_node in subtree:
                        for z, y, x in edge.path_points:
                            subtree_viz[z, y, x] = color
        self._save_debug_image(subtree_viz, os.path.join(output_dir, 'debug_subtrees.nrrd'))
        
        # 4. Peripheral relationships
        if hasattr(self, '_current_relationships'):
            relation_viz = np.zeros_like(self.vessel_mask, dtype=np.uint8)
            # Mark all vessel points
            relation_viz[self.centerline > 0] = 1
            # Mark leaf nodes that have relationships
            for i, relations in self._current_relationships.items():
                if relations:  # If this subtree has any relationships
                    subtree = self._current_subtrees[i]
                    leaves = [node for node in subtree if len(node.edges) == 1]
                    for leaf in leaves:
                        z, y, x = leaf.position
                        relation_viz[z, y, x] = 2
            self._save_debug_image(relation_viz, os.path.join(output_dir, 'debug_peripheral_relations.nrrd'))
        
        # 5. Linked groups
        if hasattr(self, '_current_groups'):
            groups_viz = np.zeros_like(self.vessel_mask, dtype=np.uint8)
            for i, group in enumerate(self._current_groups):
                color = (i % 254) + 1  # Avoid 0, cycle through colors
                for subtree_id in group:
                    subtree = self._current_subtrees[subtree_id]
                    for node in subtree:
                        for edge in node.edges:
                            if edge.start_node in subtree and edge.end_node in subtree:
                                for z, y, x in edge.path_points:
                                    groups_viz[z, y, x] = color
            self._save_debug_image(groups_viz, os.path.join(output_dir, 'debug_linked_groups.nrrd'))
            
        logger.info("Debug outputs saved successfully")

    def process(self, output_prefix: str) -> None:
        """Run the complete artery-vein separation pipeline"""
        logger.info("Starting artery-vein separation...")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Build geometric graph (Section II-B)
        self.build_geometric_graph()
        
        # 2. Detect attachment points (Section II-C)
        attachment_points = self.detect_attachment_points()
        
        # 3. Create subtrees (Section II-C.2)
        self._current_subtrees = self.create_subtrees(attachment_points)
        
        # 4. Analyze peripheral vessels (Section II-D.1)
        self._current_relationships = self.analyze_periphery(self._current_subtrees)
        
        # 5. Link subtrees (Section II-D.2)
        self._current_groups = self.link_subtrees(self._current_subtrees, self._current_relationships)
        
        # 6. Classify vessels (Section II-F)
        self.classify_vessels(self._current_subtrees, self._current_groups)
        
        # Save debug outputs
        self.save_debug_outputs(output_dir)
        
        # Save final results
        self.save_results(output_prefix)
        
        logger.info("Artery-vein separation completed successfully")
        
    def _count_neighbors(self, image: np.ndarray, z: int, y: int, x: int) -> int:
        """
        Count the number of neighboring vessel points in a 26-connected neighborhood
        
        Args:
            image: Binary image (centerline)
            z, y, x: Coordinates of the point to check
            
        Returns:
            Number of neighboring vessel points
        """
        count = 0
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (0 <= nz < image.shape[0] and 
                        0 <= ny < image.shape[1] and 
                        0 <= nx < image.shape[2] and 
                        image[nz, ny, nx] > 0):
                        count += 1
                        
        return count
        
    def _is_at_lung_border(self, pos: Tuple[int, int, int], lung_border: np.ndarray) -> bool:
        """Check if a point is at the lung border"""
        z, y, x = pos
        return lung_border[z, y, x] > 0 
        
    def _trace_paths_from_node(self, image: np.ndarray, start_pos: Tuple[int, int, int], 
                             visited: Set[Tuple[int, int, int]]) -> None:
        """
        Trace paths from a node through points with exactly 2 neighbors until reaching another node
        """
        def get_unvisited_neighbors(pos):
            z, y, x = pos
            neighbors = []
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        
                        nz, ny, nx = z + dz, y + dy, x + dx
                        npos = (nz, ny, nx)
                        if (0 <= nz < image.shape[0] and 
                            0 <= ny < image.shape[1] and 
                            0 <= nx < image.shape[2] and 
                            image[nz, ny, nx] > 0 and 
                            npos not in visited):
                            neighbors.append(npos)
            return neighbors
        
        # Start from each unvisited neighbor of the start node
        start_neighbors = get_unvisited_neighbors(start_pos)
        visited.add(start_pos)
        start_node = self.nodes[start_pos]
        
        for neighbor_pos in start_neighbors:
            if neighbor_pos in visited:
                continue
                
            # Trace path until reaching another node or dead end
            path = [start_pos, neighbor_pos]
            current_pos = neighbor_pos
            visited.add(current_pos)
            
            while True:
                # Get unvisited neighbors of current position
                neighbors = get_unvisited_neighbors(current_pos)
                
                # If no unvisited neighbors or more than one, we've reached a node or dead end
                if len(neighbors) != 1:
                    break
                    
                # Continue along the path
                next_pos = neighbors[0]
                path.append(next_pos)
                visited.add(next_pos)
                current_pos = next_pos
            
            # If path ends at another node, create an edge
            if current_pos in self.nodes:
                end_node = self.nodes[current_pos]
                edge = Edge(start_node, end_node, path)
                self.graph.edges.append(edge)
                start_node.edges.append(edge)
                end_node.edges.append(edge) 