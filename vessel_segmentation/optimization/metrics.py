# optimization/metrics.py
import numpy as np
import networkx as nx
from typing import Dict
from scipy.spatial.distance import directed_hausdorff

class VesselMetricsCalculator:
    def __init__(self):
        pass

    def calculate_metrics(self, vessel_tree: nx.Graph, 
                         vessel_mask: np.ndarray,
                         surface_mesh: Dict) -> Dict[str, float]:
        """Calculate comprehensive vessel segmentation metrics"""
        metrics = {}
        
        # Vessel continuity metrics
        continuity_metrics = self._calculate_continuity_metrics(vessel_tree)
        metrics.update(continuity_metrics)
        
        # Surface quality metrics
        surface_metrics = self._calculate_surface_metrics(surface_mesh)
        metrics.update(surface_metrics)
        
        # Branch detection metrics
        branch_metrics = self._calculate_branch_metrics(vessel_tree)
        metrics.update(branch_metrics)
        
        # Normalize scores to [0,1]
        metrics = self._normalize_metrics(metrics)
        
        return metrics

    def _calculate_continuity_metrics(self, vessel_tree: nx.Graph) -> Dict[str, float]:
        """Calculate vessel continuity metrics"""
        # Get all paths between high-generation points
        paths = []
        high_gen_nodes = [n for n,d in vessel_tree.nodes(data=True) 
                         if d['generation'] >= 3]
        
        for i in range(min(len(high_gen_nodes), 100)):  # Sample paths for efficiency
            node = high_gen_nodes[i]
            try:
                path = nx.shortest_path(vessel_tree, 
                                      source=self._find_root(vessel_tree),
                                      target=node)
                paths.append(path)
            except nx.NetworkXNoPath:
                continue
        
        # Calculate metrics
        if not paths:
            return {
                'path_continuity': 0.0,
                'radius_consistency': 0.0,
                'path_smoothness': 0.0
            }
        
        # Path continuity - check for gaps
        gaps = []
        for path in paths:
            path_points = np.array([vessel_tree.nodes[n]['coordinates'] for n in path])
            distances = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
            gaps.append(np.max(distances))
        
        path_continuity = np.exp(-np.mean(gaps))
        
        # Radius consistency along paths
        radius_variations = []
        for path in paths:
            radii = [vessel_tree.nodes[n]['radius'] for n in path]
            variations = np.abs(np.diff(radii)) / np.mean(radii)
            radius_variations.append(np.mean(variations))
        
        radius_consistency = np.exp(-np.mean(radius_variations))
        
        # Path smoothness - analyze direction changes
        smoothness = []
        for path in paths:
            if len(path) < 3:
                continue
            points = np.array([vessel_tree.nodes[n]['coordinates'] for n in path])
            vectors = np.diff(points, axis=0)
            angles = np.arccos(np.sum(vectors[1:] * vectors[:-1], axis=1) /
                             (np.linalg.norm(vectors[1:], axis=1) * 
                              np.linalg.norm(vectors[:-1], axis=1)))
            smoothness.append(np.mean(np.abs(angles)))
        
        path_smoothness = np.exp(-np.mean(smoothness))
        
        return {
            'path_continuity': path_continuity,
            'radius_consistency': radius_consistency,
            'path_smoothness': path_smoothness
        }

    def _calculate_surface_metrics(self, surface_mesh: Dict) -> Dict[str, float]:
        """Calculate surface quality metrics"""
        vertices = surface_mesh['vertices']
        faces = surface_mesh['faces']
        
        # Triangle quality (aspect ratio)
        triangle_qualities = []
        for face in faces:
            v1, v2, v3 = vertices[face]
            edges = [np.linalg.norm(v2-v1), np.linalg.norm(v3-v2), np.linalg.norm(v1-v3)]
            s = sum(edges) / 2
            area = np.sqrt(s*(s-edges[0])*(s-edges[1])*(s-edges[2]))
            quality = 4 * np.sqrt(3) * area / (sum([e*e for e in edges]))
            triangle_qualities.append(quality)
        
        mean_quality = np.mean(triangle_qualities)
        
        # Surface smoothness
        vertex_normals = surface_mesh.get('vertex_normals', None)
        if vertex_normals is not None:
            smoothness = np.mean([np.abs(np.dot(vertex_normals[i], vertex_normals[j]))
                                for i,j in surface_mesh['edges']])
        else:
            smoothness = 0.5  # default if normals not available
        
        return {
            'surface_quality': mean_quality,
            'surface_smoothness': smoothness
        }

    def _calculate_branch_metrics(self, vessel_tree: nx.Graph) -> Dict[str, float]:
        """Calculate branching pattern metrics"""
        # Find branch points (nodes with degree > 2)
        branch_points = [n for n in vessel_tree.nodes() if vessel_tree.degree(n) > 2]
        
        if not branch_points:
            return {
                'branch_angle_score': 0.0,
                'radius_ratio_score': 0.0,
                'branch_distribution': 0.0
            }
        
        # Analyze branch angles
        branch_angles = []
        radius_ratios = []
        
        for bp in branch_points:
            neighbors = list(vessel_tree.neighbors(bp))
            if len(neighbors) < 3:
                continue
            
            # Get vectors to neighbors
            vectors = np.array([vessel_tree.nodes[n]['coordinates'] - 
                              vessel_tree.nodes[bp]['coordinates'] 
                              for n in neighbors])
            vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
            
            # Calculate angles between vectors
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    angle = np.arccos(np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0))
                    branch_angles.append(angle)
            
            # Calculate radius ratios
            radii = [vessel_tree.nodes[n]['radius'] for n in neighbors]
            parent_radius = max(radii)
            child_radii = sorted(radii[:-1], reverse=True)
            if len(child_radii) >= 2:
                radius_ratios.append((child_radii[0] + child_radii[1]) / parent_radius)
        
        # Score branch angles (ideal ~70 degrees)
        angle_score = np.mean([np.exp(-(angle - np.pi/2.5)**2 / 0.5) 
                             for angle in branch_angles])
        
        # Score radius ratios (Murray's law: r^3 = r1^3 + r2^3)
        ratio_score = np.mean([np.exp(-(ratio - 0.8)**2 / 0.2) 
                             for ratio in radius_ratios])
        
        # Analyze branch distribution across generations
        generations = nx.get_node_attributes(vessel_tree, 'generation')
        gen_counts = np.bincount([generations[bp] for bp in branch_points])
        distribution_score = 1 - np.std(gen_counts) / np.mean(gen_counts)
        
        return {
            'branch_angle_score': angle_score,
            'radius_ratio_score': ratio_score,
            'branch_distribution': distribution_score
        }

    @staticmethod
    def _normalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        """Normalize all metrics to [0,1] range and compute composite scores"""
        # Composite scores
        metrics['vessel_continuity'] = np.mean([
            metrics['path_continuity'],
            metrics['radius_consistency'],
            metrics['path_smoothness']
        ])
        
        metrics['surface_quality'] = np.mean([
            metrics['surface_quality'],
            metrics['surface_smoothness']
        ])
        
        metrics['branch_detection'] = np.mean([
            metrics['branch_angle_score'],
            metrics['radius_ratio_score'],
            metrics['branch_distribution']
        ])
        
        return metrics

    @staticmethod
    def _find_root(G: nx.Graph) -> tuple:
        """Find root node (largest vessel)"""
        return max(G.nodes(), key=lambda n: G.nodes[n]['radius'])
    
class OptimizationMetrics:
    """Additional metrics specific for optimization"""
    
    @staticmethod
    def calculate_vessel_coverage(vessel_mask: np.ndarray, lung_mask: np.ndarray) -> float:
        """Calculate vessel coverage within lung regions"""
        total_lung_volume = np.sum(lung_mask > 0)
        vessel_volume = np.sum(vessel_mask > 0)
        return vessel_volume / total_lung_volume if total_lung_volume > 0 else 0

    @staticmethod
    def analyze_radius_distribution(vessel_tree: nx.Graph) -> Dict[str, float]:
        """Analyze vessel radius distribution"""
        radii = [data['radius'] for _, data in vessel_tree.nodes(data=True)]
        if not radii:
            return {'radius_distribution_score': 0.0}
            
        # Expected distribution based on anatomical knowledge
        # Log-normal distribution parameters
        mu, sigma = np.log(2), 0.5
        
        # Calculate histogram
        hist, bins = np.histogram(radii, bins=20, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Expected distribution
        expected = np.exp(-(np.log(bin_centers) - mu)**2 / (2 * sigma**2)) / (bin_centers * sigma * np.sqrt(2*np.pi))
        
        # Normalize
        expected = expected / np.sum(expected)
        hist = hist / np.sum(hist)
        
        # Calculate KL divergence
        kl_div = np.sum(hist * np.log(hist / (expected + 1e-10) + 1e-10))
        
        # Convert to score (0-1)
        score = np.exp(-kl_div)
        
        return {'radius_distribution_score': score}

    @staticmethod
    def calculate_connectivity_metrics(vessel_tree: nx.Graph) -> Dict[str, float]:
        """Calculate connectivity and topology metrics"""
        # Connected components analysis
        components = list(nx.connected_components(vessel_tree))
        
        if not components:
            return {
                'connectivity_score': 0.0,
                'component_ratio': 0.0,
                'largest_component_ratio': 0.0
            }
        
        # Number of components relative to total nodes
        total_nodes = vessel_tree.number_of_nodes()
        component_ratio = len(components) / total_nodes
        
        # Size of largest component
        largest_component = max(components, key=len)
        largest_component_ratio = len(largest_component) / total_nodes
        
        # Overall connectivity score
        connectivity_score = largest_component_ratio * (1 - component_ratio)
        
        return {
            'connectivity_score': connectivity_score,
            'component_ratio': 1 - component_ratio,  # Invert for scoring
            'largest_component_ratio': largest_component_ratio
        }

    @staticmethod
    def calculate_bifurcation_metrics(vessel_tree: nx.Graph) -> Dict[str, float]:
        """Calculate bifurcation-specific metrics"""
        bifurcations = []
        
        for node in vessel_tree.nodes():
            neighbors = list(vessel_tree.neighbors(node))
            if len(neighbors) == 3:  # True bifurcation
                bifurcations.append({
                    'node': node,
                    'neighbors': neighbors,
                    'radius': vessel_tree.nodes[node]['radius']
                })
        
        if not bifurcations:
            return {
                'bifurcation_planarity': 0.0,
                'murray_law_score': 0.0,
                'bifurcation_density': 0.0
            }
        
        # Calculate planarity
        planarity_scores = []
        murray_scores = []
        
        for bif in bifurcations:
            # Get coordinates
            p0 = vessel_tree.nodes[bif['node']]['coordinates']
            p1, p2, p3 = [vessel_tree.nodes[n]['coordinates'] for n in bif['neighbors']]
            
            # Calculate planarity
            v1, v2, v3 = p1-p0, p2-p0, p3-p0
            normal = np.cross(v1, v2)
            planarity = np.abs(np.dot(normal, v3)) / (np.linalg.norm(normal) * np.linalg.norm(v3))
            planarity_scores.append(planarity)
            
            # Check Murray's law
            r0 = vessel_tree.nodes[bif['node']]['radius']
            r1, r2, r3 = [vessel_tree.nodes[n]['radius'] for n in bif['neighbors']]
            murray_score = abs(r0**3 - (r1**3 + r2**3 + r3**3)) / r0**3
            murray_scores.append(murray_score)
        
        # Calculate density
        volume = np.prod([
            max(vessel_tree.nodes[n]['coordinates'][i] for n in vessel_tree.nodes()) -
            min(vessel_tree.nodes[n]['coordinates'][i] for n in vessel_tree.nodes())
            for i in range(3)
        ])
        bifurcation_density = len(bifurcations) / volume
        
        return {
            'bifurcation_planarity': 1 - np.mean(planarity_scores),
            'murray_law_score': 1 - np.mean(murray_scores),
            'bifurcation_density': min(bifurcation_density / 0.001, 1.0)  # Normalize to expected density
        }

class MetricsAggregator:
    """Aggregate all metrics and compute final scores"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'vessel_continuity': 0.3,
            'surface_quality': 0.2,
            'branch_detection': 0.2,
            'radius_distribution': 0.1,
            'connectivity': 0.1,
            'bifurcation_quality': 0.1
        }

    def aggregate_metrics(self, all_metrics: Dict[str, float]) -> Dict[str, float]:
        """Combine all metrics into final scores"""
        # Ensure weights sum to 1
        total_weight = sum(self.weights.values())
        normalized_weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Calculate weighted scores
        final_score = sum(
            all_metrics.get(metric, 0) * weight
            for metric, weight in normalized_weights.items()
        )
        
        # Add subscores and final score
        all_metrics.update({
            'final_score': final_score,
            'weighted_scores': {
                metric: all_metrics.get(metric, 0) * weight
                for metric, weight in normalized_weights.items()
            }
        })
        
        return all_metrics