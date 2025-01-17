from enum import Enum, auto
from typing import List, Set, Dict, Tuple, Optional
import numpy as np

class NodeType(Enum):
    """Types of vessel nodes"""
    NORMAL = auto()
    BIFURCATION = auto()
    ENDPOINT = auto()
    ATTACHMENT = auto()
    ROOT = auto()  # Added ROOT type for root nodes

class VesselType(Enum):
    """Types of vessels"""
    UNKNOWN = 0
    ARTERY = 1
    VEIN = 2

class VesselNode:
    """Class representing a node in the vessel network"""
    
    def __init__(self, position: np.ndarray, node_type: NodeType, is_border: bool = False, scale: float = 1.0, direction: np.ndarray = None):
        """Initialize a vessel node
        
        Args:
            position: 3D position of the node
            node_type: Type of the node
            is_border: Whether this node is at the lung border
            scale: Vessel scale/diameter at this point
            direction: Optional vessel direction vector at this point
        """
        self.position = position
        self.node_type = node_type
        self.is_border = is_border
        self.scale = scale
        self.direction = direction
        self.edges: Set[Edge] = set()
        
    def __hash__(self):
        return hash(tuple(self.position))
        
    def __eq__(self, other):
        if not isinstance(other, VesselNode):
            return False
        return np.array_equal(self.position, other.position)
        
    def calculate_transition_angle(self, edge1: 'Edge', edge2: 'Edge') -> float:
        """Calculate the angle between two edges connected to this node
        
        Args:
            edge1: First edge
            edge2: Second edge
            
        Returns:
            float: Angle between edges in degrees
        """
        # Get vectors from this node to neighbors
        if edge1.start_node == self:
            vec1 = edge1.end_node.position - self.position
        else:
            vec1 = edge1.start_node.position - self.position
            
        if edge2.start_node == self:
            vec2 = edge2.end_node.position - self.position
        else:
            vec2 = edge2.start_node.position - self.position
            
        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        # Calculate angle
        cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle

class Edge:
    """Class representing an edge between two vessel nodes"""
    
    def __init__(self, start_node: VesselNode, end_node: VesselNode, path: List[np.ndarray]):
        """Initialize an edge
        
        Args:
            start_node: First node
            end_node: Second node
            path: List of points along the edge
        """
        self.start_node = start_node
        self.end_node = end_node
        self.path = path
        
    def __hash__(self):
        return hash((self.start_node, self.end_node))
        
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.start_node == other.start_node and self.end_node == other.end_node) or \
               (self.start_node == other.end_node and self.end_node == other.start_node)
               
    def get_length(self) -> float:
        """Calculate the length of the edge
        
        Returns:
            float: Length of the edge
        """
        length = 0.0
        for i in range(len(self.path) - 1):
            length += np.linalg.norm(self.path[i+1] - self.path[i])
        return length

class VesselGraph:
    """Class representing the vessel network as a graph"""
    
    def __init__(self):
        """Initialize an empty graph"""
        self.nodes: Dict[Tuple[int, int, int], VesselNode] = {}
        self.edges: Set[Edge] = set()
        
    def add_node(self, node: VesselNode) -> None:
        """Add a node to the graph
        
        Args:
            node: Node to add
        """
        pos_tuple = tuple(node.position)
        self.nodes[pos_tuple] = node
        
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph
        
        Args:
            edge: Edge to add
        """
        self.edges.add(edge)
        edge.start_node.edges.add(edge)
        edge.end_node.edges.add(edge)
        
    def remove_edge(self, edge: Edge) -> None:
        """Remove an edge from the graph
        
        Args:
            edge: Edge to remove
        """
        self.edges.remove(edge)
        edge.start_node.edges.remove(edge)
        edge.end_node.edges.remove(edge) 

class VesselSegment:
    """Represents a vessel segment with connected points"""
    
    def __init__(self):
        self.points = []  # List of point positions
        self.scales = []  # List of scales at each point
        self.direction = None  # Overall direction vector
        self.mean_scale = None  # Mean scale of segment
        self.length = 0.0  # Length of segment
        self.start_point = None  # First point
        self.end_point = None  # Last point
        
    def add_point(self, pos: np.ndarray, scale: float):
        """Add a point to the segment"""
        if not self.points:
            self.start_point = pos
        self.points.append(pos)
        self.scales.append(scale)
        self.end_point = pos
        
        # Update length if we have more than one point
        if len(self.points) > 1:
            prev = self.points[-2]
            curr = self.points[-1]
            self.length += np.linalg.norm(curr - prev)
            
    def compute_direction(self):
        """Compute overall direction vector and mean scale"""
        if len(self.points) < 2:
            return
            
        # Convert to numpy arrays
        points = np.array(self.points)
        scales = np.array(self.scales)
        
        # Compute mean scale
        self.mean_scale = np.mean(scales)
        
        # For short segments, use end-to-end vector
        if len(points) <= 3:
            direction = self.end_point - self.start_point
            self.direction = direction / np.linalg.norm(direction)
            return
            
        # For longer segments, use PCA to get main direction
        # Center the points
        mean_pos = np.mean(points, axis=0)
        centered = points - mean_pos
        
        # Compute covariance matrix
        cov = np.dot(centered.T, centered)
        
        # Get eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Direction is eigenvector with largest eigenvalue
        self.direction = eigenvecs[:, -1]
        
        # Ensure direction points from start to end
        if np.dot(self.direction, self.end_point - self.start_point) < 0:
            self.direction = -self.direction 