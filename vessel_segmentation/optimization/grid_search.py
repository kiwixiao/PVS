# optimization/grid_search.py
from itertools import product
import numpy as np
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Any
from datetime import datetime
import json

class GridSearch:
    def __init__(self, base_config_path: str, output_dir: str):
        self.base_config = self._load_yaml(base_config_path)
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create optimization output directory
        self.opt_dir = self.output_dir / 'optimization'
        self.opt_dir.mkdir(parents=True, exist_ok=True)

    def define_parameter_grid(self) -> Dict[str, List[Any]]:
        """Define parameter ranges for grid search"""
        return {
            # Vessel detection parameters
            'vessel_detection.scale_range.min_vessel_size': [0.3, 0.5, 0.7],
            'vessel_detection.scale_range.max_vessel_size': [12.0, 14.0, 16.0],
            'vessel_detection.scale_range.num_scales': [10, 15, 20],
            
            # Frangi filter parameters
            'vessel_detection.frangi_filter.alpha': [0.3, 0.5, 0.7],
            'vessel_detection.frangi_filter.beta': [0.3, 0.5, 0.7],
            'vessel_detection.frangi_filter.c': [300, 500, 700],
            
            # Scale integration
            'vessel_detection.scale_integration.method': ['max', 'weighted', 'adaptive'],
            
            # Vessel connectivity parameters
            'vessel_refinement.vessel_connectivity.gap_size_threshold': [2, 3, 4],
            'vessel_refinement.vessel_connectivity.angle_threshold': [45, 60, 75],
            'vessel_refinement.vessel_connectivity.radius_ratio_max': [1.5, 2.0, 2.5],
            
            # Surface generation parameters
            'surface_generation.smoothing.iterations': [15, 20, 25],
            'surface_generation.smoothing.relaxation_factor': [0.05, 0.1, 0.15]
        }

    def generate_configs(self) -> List[Dict]:
        """Generate all possible parameter combinations"""
        param_grid = self.define_parameter_grid()
        keys, values = zip(*param_grid.items())
        configs = []
        
        for v in product(*values):
            # Create new config from base
            config = self._deep_copy_dict(self.base_config)
            
            # Update with new parameters
            for k, val in zip(keys, v):
                self._set_nested_dict(config, k.split('.'), val)
            
            configs.append(config)
        
        return configs

    def run_grid_search(self, pipeline_runner, metric_calculator):
        """Run grid search optimization"""
        configs = self.generate_configs()
        results = []
        
        for idx, config in enumerate(configs):
            try:
                self.logger.info(f"Running configuration {idx+1}/{len(configs)}")
                
                # Save current config
                config_path = self.opt_dir / f'config_{idx:03d}.yaml'
                self._save_yaml(config, config_path)
                
                # Run pipeline with current config
                pipeline_output = pipeline_runner(config)
                
                # Calculate metrics
                metrics = metric_calculator(pipeline_output)
                
                # Store results
                results.append({
                    'config_id': idx,
                    'config_path': str(config_path),
                    'metrics': metrics,
                    'parameters': config
                })
                
                # Save intermediate results
                self._save_results(results)
                
            except Exception as e:
                self.logger.error(f"Error in configuration {idx}: {str(e)}")
                continue
        
        # Find best configuration
        best_config = self._find_best_config(results)
        return best_config, results

    def _find_best_config(self, results: List[Dict]) -> Dict:
        """Find best configuration based on metrics"""
        # Example metric scoring - modify based on specific requirements
        def score_metrics(metrics):
            return (
                metrics['vessel_continuity'] * 0.4 +
                metrics['surface_quality'] * 0.3 +
                metrics['branch_detection'] * 0.3
            )
        
        scored_results = [
            (score_metrics(r['metrics']), r) 
            for r in results
        ]
        
        best_score, best_result = max(scored_results, key=lambda x: x[0])
        return best_result

    @staticmethod
    def _load_yaml(path: str) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _save_yaml(data: Dict, path: str):
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    @staticmethod
    def _deep_copy_dict(d: Dict) -> Dict:
        return json.loads(json.dumps(d))

    @staticmethod
    def _set_nested_dict(d: Dict, keys: List[str], value: Any):
        """Set value in nested dictionary using dot notation keys"""
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value

    def _save_results(self, results: List[Dict]):
        """Save intermediate optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.opt_dir / f'optimization_results_{timestamp}.json'
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)