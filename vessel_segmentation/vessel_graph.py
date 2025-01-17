import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import numpy.typing as npt
from scipy.spatial import cKDTree
from dataclasses import dataclass

class NodeType(Enum):
    """Enumeration for different types of vessel nodes"""
    NORMAL = 0      # Degree 2 nodes
    BIFURCATION = 1 # Degree > 2
    ENDPOINT = 2    # Degree 1
    ROOT = 3        # At lung border
    ATTACHMENT = 4  # Detected attachment point

@dataclass
class Edge:
    """Class representing a vessel segment between two complex nodes"""
    start_node: 'VesselNode'
    end_node: 'VesselNode'
    path_points: List[Tuple[int, int, int]]  # List of points along the centerline
    length: float = 0.0
    mean_diameter: float = 0.0
    vessel_type: Optional[str] = None  # 'artery' or 'vein'

class VesselNode:
    """Class representing a point in the vessel network"""
    def __init__(self, position: Tuple[int, int, int], node_type: NodeType):
        self.position = position  # (z, y, x)
        self.node_type = node_type
        self.edges: List[Edge] = []
        self.diameter: float = 0.0
        self.direction: np.ndarray = np.zeros(3)
        self.vessel_type: Optional[str] = None  # 'artery' or 'vein'
        
    def get_neighbors(self) -> List['VesselNode']:
        """Get all neighboring nodes through edges"""
        return [edge.end_node if edge.start_node == self else edge.start_node 
                for edge in self.edges]
        
    def calculate_transition_angle(self, edge1: Edge, edge2: Edge) -> float:
        """Calculate the transition angle θ between two edges"""
        # Get direction vectors
        if edge1.start_node == self:
            v1 = np.array(edge1.path_points[1]) - np.array(self.position)
        else:
            v1 = np.array(edge1.path_points[-2]) - np.array(self.position)
            
        if edge2.start_node == self:
            v2 = np.array(edge2.path_points[1]) - np.array(self.position)
        else:
            v2 = np.array(edge2.path_points[-2]) - np.array(self.position)
        
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Calculate angle in degrees
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return np.degrees(angle)

class VesselGraph:
    """Class representing the vessel network as a graph"""
    def __init__(self):
        self.nodes: Dict[Tuple[int, int, int], VesselNode] = {}
        self.edges: List[Edge] = []
        self.spatial_index: Optional[cKDTree] = None
        
    def build_initial_graph(self, centerline: np.ndarray, point_types: np.ndarray,
                          lung_mask: np.ndarray) -> None:
        """First stage: Build initial graph with complex nodes"""
        # Find all centerline points
        points = np.argwhere(centerline > 0)
        
        # Create nodes for complex points (not degree 2)
        for z, y, x in points:
            neighbors = self._count_neighbors(centerline, z, y, x)
            pos = (z, y, x)
            
            if neighbors != 2:
                # Determine node type
                if neighbors == 1:
                    # Check if at lung border
                    if self._is_at_lung_border((z, y, x), lung_mask):
                        node_type = NodeType.ROOT
                    else:
                        node_type = NodeType.ENDPOINT
                else:
                    node_type = NodeType.BIFURCATION
                    
                self.nodes[pos] = VesselNode(pos, node_type)
        
        # Create edges by tracing paths between complex nodes
        visited = set()
        for node_pos in self.nodes:
            if node_pos not in visited:
                self._trace_paths_from_node(centerline, node_pos, visited)
                
    def refine_graph(self) -> None:
        """Second stage: Refine graph by incorporating degree-2 nodes into edges"""
        # Calculate edge properties
        for edge in self.edges:
            # Calculate edge length
            points = np.array(edge.path_points)
            diffs = np.diff(points, axis=0)
            edge.length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
            
            # Calculate mean diameter (assuming diameter is stored in nodes)
            diameters = [self.nodes.get(pos, VesselNode(pos, NodeType.NORMAL)).diameter 
                        for pos in edge.path_points]
            edge.mean_diameter = np.mean(diameters)
            
    def _count_neighbors(self, centerline: np.ndarray, z: int, y: int, x: int) -> int:
        """Count number of neighbors in 26-neighborhood"""
        count = 0
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (0 <= nz < centerline.shape[0] and
                        0 <= ny < centerline.shape[1] and
                        0 <= nx < centerline.shape[2] and
                        centerline[nz, ny, nx] > 0):
                        count += 1
        return count
        
    def _is_at_lung_border(self, pos: Tuple[int, int, int], lung_mask: np.ndarray) -> bool:
        """Check if a point is at the lung border"""
        z, y, x = pos
        # Check 6-neighborhood
        for dz, dy, dx in [(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)]:
            nz, ny, nx = z + dz, y + dy, x + dx
            if (0 <= nz < lung_mask.shape[0] and
                0 <= ny < lung_mask.shape[1] and
                0 <= nx < lung_mask.shape[2] and
                lung_mask[nz, ny, nx] == 0):
                return True
        return False
        
    def _trace_paths_from_node(self, centerline: np.ndarray, start_pos: Tuple[int, int, int],
                             visited: Set[Tuple[int, int, int]]) -> None:
        """Trace paths from a node to other complex nodes"""
        def get_next_point(current: Tuple[int, int, int], 
                          prev: Optional[Tuple[int, int, int]] = None) -> Optional[Tuple[int, int, int]]:
            """Get next point in path, excluding the previous point"""
            z, y, x = current
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        nz, ny, nx = z + dz, y + dy, x + dx
                        next_pos = (nz, ny, nx)
                        if (next_pos != prev and
                            0 <= nz < centerline.shape[0] and
                            0 <= ny < centerline.shape[1] and
                            0 <= nx < centerline.shape[2] and
                            centerline[nz, ny, nx] > 0 and
                            next_pos not in visited):
                            return next_pos
            return None
            
        # Start tracing from each unvisited neighbor
        current = start_pos
        start_node = self.nodes[start_pos]
        visited.add(current)
        
        while True:
            next_pos = get_next_point(current)
            if next_pos is None:
                break
                
            # Start new path
            path = [start_pos, next_pos]
            current = next_pos
            visited.add(current)
            
            # Continue until we hit another complex node or dead end
            while True:
                neighbors = self._count_neighbors(centerline, *current)
                if neighbors != 2 or current in self.nodes:
                    # We've hit a complex node or dead end
                    if current in self.nodes:
                        # Create edge between start and end nodes
                        end_node = self.nodes[current]
                        edge = Edge(start_node, end_node, path)
                        self.edges.append(edge)
                        start_node.edges.append(edge)
                        end_node.edges.append(edge)
                    break
                    
                next_pos = get_next_point(current, path[-2])
                if next_pos is None:
                    break
                    
                path.append(next_pos)
                current = next_pos
                visited.add(current)
                
    def build_spatial_index(self) -> None:
        """Build KD-tree for spatial queries"""
        positions = np.array([pos for pos in self.nodes.keys()])
        self.spatial_index = cKDTree(positions)
        
    def find_neighbors_radius(self, position: Tuple[int, int, int], radius: float) -> List[VesselNode]:
        """Find all nodes within a given radius"""
        if self.spatial_index is None:
            self.build_spatial_index()
            
        distances, indices = self.spatial_index.query(position, k=10, distance_upper_bound=radius)
        valid_indices = indices[distances != np.inf]
        positions = np.array(list(self.nodes.keys()))[valid_indices]
        return [self.nodes[tuple(pos)] for pos in positions]
        
    def get_leaf_nodes(self) -> List[VesselNode]:
        """Get all leaf nodes (nodes with single edge)"""
        return [node for node in self.nodes.values() if len(node.edges) == 1]
        
    def get_root_nodes(self) -> List[VesselNode]:
        """Get all root nodes (at lung border)"""
        return [node for node in self.nodes.values() if node.node_type == NodeType.ROOT]
        
    def calculate_subtree_volume(self, root: VesselNode, visited: Optional[Set[VesselNode]] = None) -> float:
        """Calculate the volume of a subtree starting from a given root"""
        if visited is None:
            visited = set()
            
        if root in visited:
            return 0.0
            
        visited.add(root)
        volume = 0.0
        
        # Add volume of all edges from this node
        for edge in root.edges:
            if edge.start_node in visited and edge.end_node in visited:
                continue
            # Calculate edge volume as cylinder
            volume += np.pi * (edge.mean_diameter/2)**2 * edge.length
            # Recurse to other node
            other_node = edge.end_node if edge.start_node == root else edge.start_node
            volume += self.calculate_subtree_volume(other_node, visited)
            
        return volume 