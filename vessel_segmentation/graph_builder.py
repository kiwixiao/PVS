import os
import logging
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import KDTree
from typing import List, Set, Dict, Tuple
import networkx as nx

from .data_structures import NodeType, VesselNode, Edge, VesselGraph

logger = logging.getLogger(__name__)

class VesselSegment:
    """Class representing a vessel segment during graph construction"""
    def __init__(self):
        self.points = []  # List of positions
        self.direction = None
        self.scale = None
        self.start_point = None
        self.end_point = None
        
    def add_point(self, pos: np.ndarray, scale: float):
        """Add a point to the segment"""
        self.points.append(pos)
        if len(self.points) == 1:
            self.start_point = pos
            self.scale = scale
        self.end_point = pos
        # Update scale as average
        self.scale = (self.scale * (len(self.points) - 1) + scale) / len(self.points)
        
    def compute_direction(self):
        """Compute average direction of the segment"""
        if len(self.points) < 2:
            return
            
        directions = []
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)
            
        avg_direction = np.mean(directions, axis=0)
        self.direction = avg_direction / np.linalg.norm(avg_direction)

class GraphBuilder:
    """Class for building a geometric graph from vessel centerlines"""
    
    def __init__(self, centerline_file: str, point_type_file: str, 
                 vessel_mask_file: str, lung_mask_file: str, output_dir: str,
                 scale_file: str = None, direction_file: str = None,
                 max_angle: float = 45.0):
        """Initialize the graph builder
        
        Args:
            centerline_file: Path to centerline NRRD file (equivalent to particle positions)
            point_type_file: Path to point type NRRD file
            vessel_mask_file: Path to vessel mask NRRD file
            lung_mask_file: Path to lung mask NRRD file
            output_dir: Path to save debug outputs
            scale_file: Path to sigma_max.nrrd for vessel scales
            direction_file: Path to vessel_direction.nrrd for vessel directions
            max_angle: Maximum angle between connected segments
        """
        self.max_angle = max_angle
        
        # Load input data
        logger.info("Loading input data...")
        self.centerline = sitk.GetArrayFromImage(sitk.ReadImage(centerline_file))
        self.point_types = sitk.GetArrayFromImage(sitk.ReadImage(point_type_file))
        self.vessel_mask = sitk.GetArrayFromImage(sitk.ReadImage(vessel_mask_file))
        self.lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_file))
        self.reference_image = sitk.ReadImage(vessel_mask_file)
        
        # Load scale information (required)
        if not scale_file:
            raise ValueError("scale_file is required for graph building")
        self.scales = sitk.GetArrayFromImage(sitk.ReadImage(scale_file))
        vessel_scales = self.scales[self.centerline > 0]
        self.scale_stats = {
            'median': np.median(vessel_scales),
            'p95': np.percentile(vessel_scales, 95),
            'p5': np.percentile(vessel_scales, 5)
        }
        logger.info(f"Scale statistics: median={self.scale_stats['median']:.2f}, "
                   f"95th={self.scale_stats['p95']:.2f}, 5th={self.scale_stats['p5']:.2f}")
                   
        # Load direction information (required)
        if not direction_file:
            raise ValueError("direction_file is required for graph building")
        logger.info(f"Loading direction vectors from {direction_file}")
        self.directions = sitk.GetArrayFromImage(sitk.ReadImage(direction_file))
        logger.info(f"Loaded direction vectors with shape {self.directions.shape}")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def build_graph(self) -> VesselGraph:
        """Build a geometric graph from the vessel centerlines
        
        Returns:
            VesselGraph: The constructed graph
        """
        logger.info("Building geometric graph...")
        
        # Get centerline points and create KD-tree
        points = []
        point_data = []  # Store scale, direction, type for each point
        for z, y, x in zip(*np.where(self.centerline > 0)):
            pos = np.array([x, y, z])
            scale = self.scales[z, y, x]
            direction = self.directions[z, y, x]
            point_type = NodeType(self.point_types[z, y, x])
            
            points.append(pos)
            point_data.append((scale, direction, point_type))
            
        points = np.array(points)
        logger.info(f"Found {len(points)} centerline points")
        
        # Build KD-tree for neighbor search
        logger.info("Building KD-tree for neighbor search...")
        tree = KDTree(points)
        
        # Find initial neighbors using scale-based radius
        logger.info("Finding initial neighbors...")
        neighbors = self._find_initial_neighbors(points, point_data, tree)
        
        # Build vessel segments starting from endpoints
        logger.info("Building vessel segments...")
        segments = self._build_segments(points, point_data, neighbors)
        logger.info(f"Created {len(segments)} vessel segments")
        
        # Create graph from segments
        logger.info("Creating final graph...")
        graph = VesselGraph()
        
        # Add nodes at segment endpoints and bifurcations
        for segment in segments:
            start_pos = segment.start_point
            end_pos = segment.end_point
            
            # Get data for endpoints
            start_idx = np.where((points == start_pos).all(axis=1))[0][0]
            end_idx = np.where((points == end_pos).all(axis=1))[0][0]
            
            start_scale, start_dir, start_type = point_data[start_idx]
            end_scale, end_dir, end_type = point_data[end_idx]
            
            # Create nodes
            start_node = VesselNode(position=start_pos,
                                  node_type=start_type,
                                  scale=start_scale,
                                  direction=start_dir)
            end_node = VesselNode(position=end_pos,
                                node_type=end_type,
                                scale=end_scale,
                                direction=end_dir)
                                
            graph.add_node(start_node)
            graph.add_node(end_node)
            
            # Create edge
            edge = Edge(start_node=start_node,
                       end_node=end_node,
                       path=segment.points)
            graph.add_edge(edge)
            
        # Save debug visualizations
        self._save_debug_outputs(points, point_data, neighbors, segments, graph)
        
        logger.info(f"Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
        
    def _find_initial_neighbors(self, points: np.ndarray, point_data: List[Tuple], 
                              tree: KDTree) -> Dict[int, Set[int]]:
        """Find potential neighbors based on scale and direction"""
        neighbors = {}
        
        for i in tqdm(range(len(points)), desc="Finding neighbors"):
            pos = points[i]
            scale, direction, _ = point_data[i]
            
            # Adaptive search radius based on vessel size
            radius = scale * 2.0  # Base radius
            if scale > self.scale_stats['p95']:
                radius *= 1.5  # Larger radius for big vessels
                
            # Find potential neighbors within radius
            potential = tree.query_ball_point(pos, radius)
            
            # Filter by direction and scale consistency
            valid = set()
            for j in potential:
                if i == j:
                    continue
                    
                j_scale, j_direction, _ = point_data[j]
                
                # Check angle between directions
                angle = self._compute_angle(direction, j_direction)
                
                # Check scale consistency
                scale_ratio = min(scale, j_scale) / max(scale, j_scale)
                
                # More lenient angle threshold for large vessels
                threshold = self.max_angle
                if scale > self.scale_stats['p95'] or j_scale > self.scale_stats['p95']:
                    threshold *= 1.2
                    
                if angle < threshold and scale_ratio > 0.5:
                    valid.add(j)
                    
            neighbors[i] = valid
            
        return neighbors
        
    def _build_segments(self, points: np.ndarray, point_data: List[Tuple],
                       neighbors: Dict[int, Set[int]]) -> List[VesselSegment]:
        """Build vessel segments by following connected points"""
        segments = []
        visited = set()
        
        # Start from endpoints (points with single neighbor)
        start_points = [i for i in range(len(points)) if len(neighbors[i]) == 1]
        logger.info(f"Found {len(start_points)} endpoint points")
        
        # Then process points with exactly two neighbors
        two_neighbor_points = [i for i in range(len(points)) 
                             if len(neighbors[i]) == 2 and i not in visited]
        logger.info(f"Found {len(two_neighbor_points)} two-neighbor points")
        
        # Process endpoints first, then two-neighbor points
        start_points.extend(two_neighbor_points)
        
        for start_idx in tqdm(start_points, desc="Building segments"):
            if start_idx in visited:
                continue
                
            # Start new segment
            segment = VesselSegment()
            curr_idx = start_idx
            curr_pos = points[curr_idx]
            curr_scale, curr_direction, _ = point_data[curr_idx]
            
            while True:
                # Add current point to segment
                segment.add_point(curr_pos, curr_scale)
                visited.add(curr_idx)
                
                # Find best next point based on direction and scale
                best_next = None
                best_angle = float('inf')
                
                for next_idx in neighbors[curr_idx]:
                    if next_idx in visited:
                        continue
                        
                    next_pos = points[next_idx]
                    next_scale, next_direction, _ = point_data[next_idx]
                    
                    # Check if we can extend to this point
                    angle = self._compute_angle(curr_direction, next_direction)
                    scale_ratio = min(curr_scale, next_scale) / max(curr_scale, next_scale)
                    
                    if scale_ratio > 0.5 and angle < best_angle:
                        best_angle = angle
                        best_next = (next_idx, next_pos, next_scale, next_direction)
                        
                if best_next is None:
                    break
                    
                curr_idx, curr_pos, curr_scale, curr_direction = best_next
                
            # Only keep segments with at least 2 points
            if len(segment.points) > 1:
                segment.compute_direction()
                segments.append(segment)
                
        return segments
        
    def _compute_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors"""
        # Handle zero vectors
        if np.all(v1 == 0) or np.all(v2 == 0):
            return 180.0
            
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Compute angle, handling numerical instability
        cos_angle = np.clip(np.abs(np.dot(v1, v2)), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
        
    def _save_debug_outputs(self, points: np.ndarray, point_data: List[Tuple],
                          neighbors: Dict[int, Set[int]], segments: List[VesselSegment],
                          graph: VesselGraph) -> None:
        """Save debug visualizations"""
        # Get reference array shape
        ref_array = sitk.GetArrayFromImage(self.reference_image)
        
        # 1. Save scale visualization
        scale_viz = np.zeros_like(ref_array, dtype=np.float32)
        for pos, (scale, _, _) in zip(points, point_data):
            x, y, z = pos.astype(int)
            scale_viz[z,y,x] = scale
        scale_img = sitk.GetImageFromArray(scale_viz)
        scale_img.CopyInformation(self.reference_image)
        sitk.WriteImage(scale_img, os.path.join(self.output_dir, "debug_vessel_scales.nrrd"))
        
        # 2. Save neighbor count visualization
        neighbor_viz = np.zeros_like(ref_array, dtype=np.uint8)
        for i, pos in enumerate(points):
            x, y, z = pos.astype(int)
            neighbor_viz[z,y,x] = len(neighbors[i])
        neighbor_img = sitk.GetImageFromArray(neighbor_viz)
        neighbor_img.CopyInformation(self.reference_image)
        sitk.WriteImage(neighbor_img, os.path.join(self.output_dir, "debug_neighbor_counts.nrrd"))
        
        # 3. Save segment visualization
        segment_viz = np.zeros_like(ref_array, dtype=np.uint8)
        for i, segment in enumerate(segments):
            color = (i % 254) + 1  # Avoid 0
            for point in segment.points:
                x, y, z = point.astype(int)
                segment_viz[z,y,x] = color
        segment_img = sitk.GetImageFromArray(segment_viz)
        segment_img.CopyInformation(self.reference_image)
        sitk.WriteImage(segment_img, os.path.join(self.output_dir, "debug_segments.nrrd"))
        
        # 4. Save final graph visualization
        graph_viz = np.zeros_like(ref_array, dtype=np.uint8)
        
        # Color nodes by type
        for node in graph.nodes.values():
            x, y, z = node.position.astype(int)
            if node.node_type == NodeType.BIFURCATION:
                graph_viz[z,y,x] = 1  # Red
            elif node.node_type == NodeType.ENDPOINT:
                graph_viz[z,y,x] = 2  # Green
                
        # Color edge points
        for edge in graph.edges:
            for point in edge.path[1:-1]:  # Skip endpoints
                x, y, z = point.astype(int)
                graph_viz[z,y,x] = 3  # Blue
                
        graph_img = sitk.GetImageFromArray(graph_viz)
        graph_img.CopyInformation(self.reference_image)
        sitk.WriteImage(graph_img, os.path.join(self.output_dir, "debug_graph_structure.nrrd")) 