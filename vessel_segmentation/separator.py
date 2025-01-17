import numpy as np
from typing import List, Set, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import os
import networkx as nx
from queue import Queue
from dataclasses import dataclass
from .data_structures import VesselNode, Edge, NodeType, VesselGraph, VesselType
import SimpleITK as sitk

logger = logging.getLogger(__name__)

@dataclass
class SubTree:
    """Represents a subtree of the vessel network"""
    nodes: Set[VesselNode]  # Nodes in this subtree
    edges: Set[Edge]  # Edges in this subtree
    root_nodes: Set[VesselNode]  # Root nodes
    leaf_nodes: Set[VesselNode]  # Leaf nodes
    total_volume: float = 0.0  # Total volume of vessels in subtree
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def has_node(self, node: VesselNode) -> bool:
        return node in self.nodes

class VesselSeparator:
    """
    Handles attachment point detection and subtree creation.
    Implements methods from Sections II-C of the paper.
    """
    
    def __init__(self, graph: VesselGraph, output_dir: str):
        """
        Initialize vessel separator
        
        Args:
            graph: Input vessel graph
            output_dir: Directory for debug outputs
        """
        self.graph = graph
        self.output_dir = output_dir
        # Get reference image for dimensions
        self.reference_image = sitk.ReadImage(os.path.join(os.path.dirname(output_dir), 
                                                          'intermediate_results/sigma_max.nrrd'))
        
    def detect_attachment_points(self) -> List[VesselNode]:
        """
        Detect attachment points using leaf node pruning and path analysis (Section II-C)
        """
        logger.info("Starting leaf node pruning...")
        
        # Calculate scale statistics
        scales = np.array([node.scale for node in self.graph.nodes.values()])
        max_scale = np.max(scales)
        median_scale = np.median(scales)
        scale_percentile_95 = np.percentile(scales, 95)
        scale_percentile_5 = np.percentile(scales, 5)
        
        # Enhanced root detection with border threshold
        border_threshold = max_scale * 0.35  # As per reference code
        root_nodes = []
        
        for node in self.graph.nodes.values():
            if len(node.edges) == 1:  # Endpoint
                neighbor = next(iter(node.edges)).end_node if next(iter(node.edges)).start_node == node else next(iter(node.edges)).start_node
                direction = np.array(neighbor.position) - np.array(node.position)
                direction = direction / np.linalg.norm(direction)
                
                # Calculate alignment with vessel direction
                alignment = np.abs(np.dot(direction, node.direction))
                
                # Enhanced root detection criteria matching reference code
                if ((node.is_border and node.scale > median_scale) or  # Border vessel above median size
                    node.scale > scale_percentile_95 or               # Very large vessel
                    node.scale > border_threshold or                  # Large enough to be root
                    (node.scale > neighbor.scale * 1.2 and           # Significantly larger than neighbor
                     alignment > 0.6)):                              # With good alignment
                    root_nodes.append(node)
                    node.node_type = NodeType.ROOT
        
        logger.info(f"Enhanced root detection found {len(root_nodes)} significant root nodes")
        
        # Create NetworkX graph for efficient path finding
        G = nx.Graph()
        for node in self.graph.nodes.values():
            G.add_node(node)
        for edge in self.graph.edges:
            G.add_edge(edge.start_node, edge.end_node)
        
        # Collect transition angles in normal branches
        angles = []
        paths = []
        attachment_points = set()
        
        pbar = tqdm(enumerate(root_nodes), desc="Analyzing root paths", total=len(root_nodes))
        for i, root1 in pbar:
            for root2 in root_nodes[i+1:]:
                try:
                    path = nx.shortest_path(G, root1, root2)
                    paths.append(path)
                    
                    # Calculate transition angles along path
                    for j in range(1, len(path)-1):
                        curr_pos = np.array(path[j].position)
                        prev_pos = np.array(path[j-1].position)
                        next_pos = np.array(path[j+1].position)
                        
                        # Calculate vectors
                        v1 = prev_pos - curr_pos
                        v2 = next_pos - curr_pos
                        
                        # Calculate angle between vectors
                        angle = np.arccos(np.clip(np.dot(v1/np.linalg.norm(v1), 
                                                       v2/np.linalg.norm(v2)), -1.0, 1.0))
                        angles.append(np.degrees(angle))
                        
                except nx.NetworkXNoPath:
                    continue
                    
            pbar.set_postfix({"paths": len(paths), "angles": len(angles)})
        
        # Calculate patient-specific threshold α (15th percentile)
        if angles:
            alpha = np.percentile(angles, 15)
            logger.info(f"Patient-specific angle threshold α: {alpha:.2f} degrees")
            
            # Find attachment points using angle threshold
            for path in paths:
                for j in range(1, len(path)-1):
                    curr_pos = np.array(path[j].position)
                    prev_pos = np.array(path[j-1].position)
                    next_pos = np.array(path[j+1].position)
                    
                    v1 = prev_pos - curr_pos
                    v2 = next_pos - curr_pos
                    angle = np.degrees(np.arccos(np.clip(np.dot(v1/np.linalg.norm(v1), 
                                                              v2/np.linalg.norm(v2)), -1.0, 1.0)))
                    
                    if angle < alpha:
                        attachment_points.add(path[j])
                        path[j].node_type = NodeType.ATTACHMENT
        else:
            logger.warning("No angles found for threshold calculation")
            
        logger.info(f"Found {len(attachment_points)} attachment points")
        
        # Save debug visualization
        self._save_debug_visualization(self.graph, paths, list(attachment_points))
        
        return list(attachment_points)
        
    def create_subtrees(self, attachment_points: List[VesselNode]) -> List[SubTree]:
        """
        Create set S of subtrees Ψj that don't contain artery-vein attachments (Section II-C.2)
        """
        logger.info("Creating subtrees...")
        
        # Create NetworkX graph
        G = nx.Graph()
        for node in self.graph.nodes.values():
            if node not in attachment_points:
                G.add_node(node)
        for edge in self.graph.edges:
            if (edge.start_node not in attachment_points and 
                edge.end_node not in attachment_points):
                G.add_edge(edge.start_node, edge.end_node)
        
        # Find connected components
        components = list(nx.connected_components(G))
        logger.info(f"Found {len(components)} connected components")
        
        # Convert components to subtrees
        subtrees = []
        for component in tqdm(components, desc="Processing components"):
            # Skip very small components (likely noise)
            if len(component) <= 5:
                continue
                
            subtree = SubTree(set(), set(), set(), set())
            
            # Add nodes and find root/leaf nodes
            for node in component:
                subtree.nodes.add(node)
                if node.node_type == NodeType.ROOT:
                    subtree.root_nodes.add(node)
                elif len(node.edges) == 1:
                    subtree.leaf_nodes.add(node)
            
            # Add edges
            for edge in self.graph.edges:
                if edge.start_node in component and edge.end_node in component:
                    subtree.edges.add(edge)
                    
            # Calculate volume
            subtree.total_volume = sum(edge.scale * np.linalg.norm(
                np.array(edge.end_node.position) - np.array(edge.start_node.position)
            ) for edge in subtree.edges)
            
            subtrees.append(subtree)
            
        logger.info(f"Created {len(subtrees)} subtrees")
        
        # Save debug visualization
        self._save_subtree_visualization(subtrees)
        
        return subtrees
        
    def _save_debug_visualization(self, pruned_graph: VesselGraph, paths: List[List[VesselNode]], 
                                attachment_points: List[VesselNode]) -> None:
        """Save debug visualization of pruned graph and attachment points in both NRRD and VTK formats"""
        # Get reference dimensions
        ref_array = sitk.GetArrayFromImage(self.reference_image)
        ref_shape = ref_array.shape
        logger.info(f"Using reference dimensions: {ref_shape}")
        
        # Create visualization array with reference dimensions
        viz = np.zeros(ref_shape, dtype=np.uint8)
        
        # Mark pruned graph nodes
        for node in pruned_graph.nodes.values():
            z, y, x = node.position
            if z < ref_shape[0] and y < ref_shape[1] and x < ref_shape[2]:  # Ensure within bounds
                if node.node_type == NodeType.ROOT:
                    viz[z, y, x] = 1  # Root nodes (Red)
                else:
                    viz[z, y, x] = 2  # Other nodes (Green)
                
        # Mark paths between root nodes
        for path in paths:
            for node in path:
                z, y, x = node.position
                if z < ref_shape[0] and y < ref_shape[1] and x < ref_shape[2]:  # Ensure within bounds
                    if viz[z, y, x] != 1:  # Don't overwrite root nodes
                        viz[z, y, x] = 3  # Path points (Blue)
                    
        # Mark attachment points
        for node in attachment_points:
            z, y, x = node.position
            if z < ref_shape[0] and y < ref_shape[1] and x < ref_shape[2]:  # Ensure within bounds
                viz[z, y, x] = 4  # Attachment points (Yellow)
            
        # Save NRRD visualization
        output_img = sitk.GetImageFromArray(viz)
        output_img.CopyInformation(self.reference_image)  # Copy spacing, origin, and direction
        output_img.SetMetaData("Label 1", "Root nodes (Red)")
        output_img.SetMetaData("Label 2", "Other nodes (Green)")
        output_img.SetMetaData("Label 3", "Path points (Blue)")
        output_img.SetMetaData("Label 4", "Attachment points (Yellow)")
        
        nrrd_file = os.path.join(self.output_dir, 'debug_attachment_points.nrrd')
        sitk.WriteImage(output_img, nrrd_file)
        logger.info(f"Saved NRRD visualization to: {nrrd_file}")
        
        # Save VTK version (only points within bounds)
        import vtk
        from vtk.util import numpy_support
        
        # Create points for each type
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # Add points with colors (only if within bounds)
        for node in pruned_graph.nodes.values():
            z, y, x = node.position
            if z < ref_shape[0] and y < ref_shape[1] and x < ref_shape[2]:
                points.InsertNextPoint(x, y, z)  # VTK uses x,y,z order
                if node.node_type == NodeType.ROOT:
                    colors.InsertNextTuple3(255, 0, 0)  # Red for root nodes
                else:
                    colors.InsertNextTuple3(0, 255, 0)  # Green for other nodes
                
        for path in paths:
            for node in path:
                z, y, x = node.position
                if z < ref_shape[0] and y < ref_shape[1] and x < ref_shape[2]:
                    if node.node_type != NodeType.ROOT:
                        points.InsertNextPoint(x, y, z)
                        colors.InsertNextTuple3(0, 0, 255)  # Blue for path points
                    
        for node in attachment_points:
            z, y, x = node.position
            if z < ref_shape[0] and y < ref_shape[1] and x < ref_shape[2]:
                points.InsertNextPoint(x, y, z)
                colors.InsertNextTuple3(255, 255, 0)  # Yellow for attachment points
            
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Add vertex cells (to make points visible)
        vertices = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            vertices.InsertNextCell(vertex)
        polydata.SetVerts(vertices)
        
        # Add colors
        polydata.GetPointData().SetScalars(colors)
        
        # Write to file
        writer = vtk.vtkPolyDataWriter()
        vtk_file = os.path.join(self.output_dir, 'debug_attachment_points.vtk')
        writer.SetFileName(vtk_file)
        writer.SetInputData(polydata)
        writer.Write()
        logger.info(f"Saved VTK visualization to: {vtk_file}")
        
    def _save_subtree_visualization(self, subtrees: List[SubTree]) -> None:
        """Save debug visualization of subtrees in both NRRD and VTK formats"""
        # Get reference dimensions
        ref_array = sitk.GetArrayFromImage(self.reference_image)
        ref_shape = ref_array.shape
        logger.info(f"Using reference dimensions: {ref_shape}")
        
        # Create visualization array with reference dimensions
        viz = np.zeros(ref_shape, dtype=np.uint8)
        
        # Color each subtree differently
        subtree_sizes = []
        for i, subtree in enumerate(subtrees):
            color = (i % 254) + 1
            size = len(subtree.nodes)
            subtree_sizes.append((color, size))
            for node in subtree.nodes:
                z, y, x = node.position
                if z < ref_shape[0] and y < ref_shape[1] and x < ref_shape[2]:  # Ensure within bounds
                    viz[z, y, x] = color
                
        # Save NRRD visualization
        output_img = sitk.GetImageFromArray(viz)
        output_img.CopyInformation(self.reference_image)  # Copy spacing, origin, and direction
        
        subtree_sizes.sort(key=lambda x: x[1], reverse=True)
        for i, (color, size) in enumerate(subtree_sizes[:10]):
            output_img.SetMetaData(f"Subtree {color}", f"Size: {size} nodes")
            
        nrrd_file = os.path.join(self.output_dir, 'debug_subtrees.nrrd')
        sitk.WriteImage(output_img, nrrd_file)
        logger.info(f"Saved NRRD visualization to: {nrrd_file}")
        
        # Save VTK version (only points within bounds)
        import vtk
        from vtk.util import numpy_support
        
        # Create points
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # Add points with colors (only if within bounds)
        for i, subtree in enumerate(subtrees):
            r,g,b = get_color(i)
            for node in subtree.nodes:
                z, y, x = node.position
                if z < ref_shape[0] and y < ref_shape[1] and x < ref_shape[2]:
                    points.InsertNextPoint(x, y, z)  # VTK uses x,y,z order
                    colors.InsertNextTuple3(int(r), int(g), int(b))
                
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Add vertex cells
        vertices = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            vertices.InsertNextCell(vertex)
        polydata.SetVerts(vertices)
        
        # Add colors
        polydata.GetPointData().SetScalars(colors)
        
        # Write to file
        writer = vtk.vtkPolyDataWriter()
        vtk_file = os.path.join(self.output_dir, 'debug_subtrees.vtk')
        writer.SetFileName(vtk_file)
        writer.SetInputData(polydata)
        writer.Write()
        logger.info(f"Saved VTK visualization to: {vtk_file}")
        
def get_color(idx):
    """Get HSV-based color for subtree visualization"""
    hue = (idx * 137.5) % 360  # Golden angle in degrees for good distribution
    saturation = 1.0
    value = 1.0
    # Convert HSV to RGB
    c = value * saturation
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = value - c
    
    if hue < 60:
        r,g,b = c,x,0
    elif hue < 120:
        r,g,b = x,c,0
    elif hue < 180:
        r,g,b = 0,c,x
    elif hue < 240:
        r,g,b = 0,x,c
    elif hue < 300:
        r,g,b = x,0,c
    else:
        r,g,b = c,0,x
        
    return ((r+m)*255, (g+m)*255, (b+m)*255) 