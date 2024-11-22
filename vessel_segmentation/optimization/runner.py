from pathlib import Path
import logging
from typing import Dict, List
import json
import yaml
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .grid_search import GridSearch
from .metrics import VesselMetricsCalculator, OptimizationMetrics, MetricsAggregator
from ..pipeline import VesselSegmentationPipeline

class OptimizationRunner:
    def __init__(self, 
                base_config_path: str,
                input_dir: str,
                output_dir: str,
                case_id: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.case_id = case_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.grid_search = GridSearch(base_config_path, str(output_dir))
        self.metrics_calculator = VesselMetricsCalculator()
        self.opt_metrics = OptimizationMetrics()
        self.metrics_aggregator = MetricsAggregator()

    def run_pipeline(self, config: Dict) -> Dict:
        """Run vessel segmentation pipeline with given config"""
        pipeline = VesselSegmentationPipeline(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir / 'optimization_runs' / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            case_id=self.case_id,
            config=config,
            debug=True
        )
        
        return pipeline.run()

    def calculate_metrics(self, pipeline_output: Dict) -> Dict:
        """Calculate all metrics for optimization"""
        # Extract required data
        vessel_tree = pipeline_output['vessel_tree']
        vessel_mask = pipeline_output['vessel_mask']
        surface_mesh = pipeline_output['surface_model']
        lung_mask = pipeline_output['lung_mask']
        
        # Calculate basic metrics
        metrics = self.metrics_calculator.calculate_metrics(
            vessel_tree, vessel_mask, surface_mesh
        )
        
        # Calculate optimization-specific metrics
        coverage = self.opt_metrics.calculate_vessel_coverage(vessel_mask, lung_mask)
        radius_metrics = self.opt_metrics.analyze_radius_distribution(vessel_tree)
        connectivity_metrics = self.opt_metrics.calculate_connectivity_metrics(vessel_tree)
        bifurcation_metrics = self.opt_metrics.calculate_bifurcation_metrics(vessel_tree)
        
        # Combine all metrics
        all_metrics = {
            **metrics,
            'vessel_coverage': coverage,
            **radius_metrics,
            **connectivity_metrics,
            **bifurcation_metrics
        }
        
        # Aggregate metrics
        return self.metrics_aggregator.aggregate_metrics(all_metrics)

    def run_optimization(self) -> Dict:
        """Run complete optimization process"""
        try:
            self.logger.info("Starting vessel segmentation optimization")
            
            # Run grid search
            best_config, all_results = self.grid_search.run_grid_search(
                pipeline_runner=self.run_pipeline,
                metric_calculator=self.calculate_metrics
            )
            
            # Save optimization results
            self._save_optimization_results(best_config, all_results)
            
            return best_config
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise

    def _save_optimization_results(self, best_config: Dict, all_results: List[Dict]):
        """Save optimization results and analysis"""
        output_dir = self.output_dir / 'optimization_results'
        output_dir.mkdir(exist_ok=True)
        
        # Save best configuration
        with open(output_dir / 'best_config.yaml', 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        # Save all results
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate and save analysis plots
        self._generate_optimization_plots(all_results, output_dir)

    def _generate_optimization_plots(self, results: List[Dict], output_dir: Path):
        """Generate analysis plots for optimization results"""
        
        
        # Extract scores
        scores = [r['metrics']['final_score'] for r in results]
        param_names = list(results[0]['parameters'].keys())
        
        # Parameter importance plot
        plt.figure(figsize=(12, 6))
        for param in param_names:
            param_values = [r['parameters'][param] for r in results]
            plt.scatter(param_values, scores, alpha=0.5, label=param)
        plt.xlabel('Parameter Value')
        plt.ylabel('Score')
        plt.title('Parameter Impact on Final Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_impact.png')
        plt.close()
        
        # Score distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(scores, bins=20)
        plt.xlabel('Final Score')
        plt.ylabel('Count')
        plt.title('Distribution of Optimization Scores')
        plt.savefig(output_dir / 'score_distribution.png')
        plt.close()
        
        # Metric correlation heatmap
        metric_names = list(results[0]['metrics'].keys())
        correlation_data = np.zeros((len(metric_names), len(metric_names)))
        
        for i, m1 in enumerate(metric_names):
            for j, m2 in enumerate(metric_names):
                values1 = [r['metrics'][m1] for r in results]
                values2 = [r['metrics'][m2] for r in results]
                correlation_data[i,j] = np.corrcoef(values1, values2)[0,1]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_data, xticklabels=metric_names, 
                   yticklabels=metric_names, annot=True, cmap='coolwarm')
        plt.title('Metric Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'metric_correlations.png')
        plt.close()