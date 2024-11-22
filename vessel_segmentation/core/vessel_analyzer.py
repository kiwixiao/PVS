# core/vessel_analyzer.py
import numpy as np
import networkx as nx
import json
from pathlib import Path
import logging
from typing import Dict
import matplotlib.pyplot as plt

class VesselAnalyzer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
    def analyze_vessels(self, vessel_tree: nx.Graph) -> Dict:
        """Analyze vessel tree and compute metrics"""
        try:
            stats_by_generation = self._compute_generation_stats(vessel_tree)
            summary = self._compute_summary_stats(stats_by_generation)
            
            # Save analysis results
            self._save_analysis(summary)
            
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                self._generate_analysis_plots(stats_by_generation)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing vessels: {str(e)}")
            raise
            
    def _compute_generation_stats(self, vessel_tree: nx.Graph) -> Dict:
        """Compute statistics by vessel generation"""
        stats = {}
        
        for node, data in vessel_tree.nodes(data=True):
            gen = data['generation']
            if gen not in stats:
                stats[gen] = {
                    'count': 0,
                    'radii': [],
                    'vesselness': [],
                    'eigenvalues': []
                }
            
            gen_stats = stats[gen]
            gen_stats['count'] += 1
            gen_stats['radii'].append(data['radius'])
            gen_stats['vesselness'].append(data['vesselness'])
            gen_stats['eigenvalues'].append(data['eigenvalues'])
            
        return stats
        
    def _compute_summary_stats(self, generation_stats: Dict) -> Dict:
        """Compute summary statistics from generation data"""
        summary = {}
        
        for gen, stats in generation_stats.items():
            radii = np.array(stats['radii'])
            vesselness = np.array(stats['vesselness'])
            eigenvalues = np.array(stats['eigenvalues'])
            
            summary[f'generation_{gen}'] = {
                'vessel_count': stats['count'],
                'mean_radius': float(np.mean(radii)),
                'std_radius': float(np.std(radii)),
                'mean_vesselness': float(np.mean(vesselness)),
                'mean_eigenvalues': [float(x) for x in np.mean(eigenvalues, axis=0)],
                'branching_ratio': self._compute_branching_ratio(gen, stats['count']),
                'total_length': float(stats['count'])  # in voxels
            }
            
        return summary
        
    def _compute_branching_ratio(self, generation: int, count: int) -> float:
        """Compute branching ratio for a generation"""
        if generation == 0:
            return 1.0
        
        # Get count of previous generation
        previous_gen_count = sum(1 for _, data in self.vessel_tree.nodes(data=True)
                                 if data['generation'] == generation -1)
        return count / max(1, previous_gen_count)
        
    def _save_analysis(self, analysis: Dict):
        """Save analysis results to JSON"""
        output_path = self.output_dir / 'vessel_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        self.logger.info(f"Saved vessel analysis to: {output_path}")
        
    def _generate_analysis_plots(self, generation_stats: Dict):
        """Generate analysis visualization plots"""
        plot_dir = self.output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # Plot radius distribution by generation
        plt.figure(figsize=(10, 6))
        for gen, stats in generation_stats.items():
            plt.boxplot(stats['radii'], positions=[gen])
        plt.title('Vessel Radius Distribution by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Radius (voxels)')
        plt.savefig(plot_dir / 'radius_distribution.png')
        plt.close()
        
        # Plot vessel count by generation
        generations = sorted(generation_stats.keys())
        counts = [generation_stats[gen]['count'] for gen in generations]
        
        plt.figure(figsize=(10, 6))
        plt.bar(generations, counts)
        plt.title('Vessel Count by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Count')
        plt.savefig(plot_dir / 'vessel_count.png')
        plt.close()
        
        # Plot mean vesselness by generation
        mean_vesselness = [np.mean(generation_stats[gen]['vesselness']) 
                          for gen in generations]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, mean_vesselness, 'o-')
        plt.title('Mean Vesselness by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Vesselness')
        plt.savefig(plot_dir / 'mean_vesselness.png')
        plt.close()