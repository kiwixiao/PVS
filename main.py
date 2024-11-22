#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import logging
from datetime import datetime

from vessel_segmentation.utils.config_loader import ConfigLoader
from vessel_segmentation.utils.file_utils import ensure_input_files
from vessel_segmentation.pipeline import VesselSegmentationPipeline
from vessel_segmentation.optimization.runner import OptimizationRunner

def parse_args():
    parser = argparse.ArgumentParser(description='Vessel Segmentation Pipeline')
    
    # Required arguments
    parser.add_argument('--case-id', type=str, required=True,
                      help='Case identifier')
    
    # Directory options
    parser.add_argument('--input-dir', type=str, default="input",
                      help='Input directory containing CT and mask files')
    parser.add_argument('--output-dir', type=str, default="vessel_segmentation_output",
                      help='Output directory')
    
    # Preprocessing options
    preprocess_group = parser.add_argument_group('Preprocessing options')
    preprocess_group.add_argument('--preprocess', action='store_true',
                               help='Run preprocessing step')
    preprocess_group.add_argument('--force-preprocess', action='store_true',
                               help='Force rerun preprocessing even if exists')
    preprocess_group.add_argument('--auto-preprocess', action='store_true',
                               help='Automatically run preprocessing if needed (default: True)',
                               default=True)
    
    # Processing options
    parser.add_argument('--force-recompute', action='store_true',
                      help='Force recompute vessel segmentation')
    parser.add_argument('--enable-caching', action='store_true',
                      help='Enable Hessian caching (uses more disk space)')
    
    # Optimization options
    parser.add_argument('--optimize', action='store_true',
                      help='Run parameter optimization')
    parser.add_argument('--optimization-trials', type=int, default=20,
                      help='Number of optimization trials')
    
    # Configuration and debug options
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Configuration file path')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    
    return parser.parse_args()

def setup_logging(output_dir: Path, debug: bool):
    """Setup logging configuration"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'vessel_segmentation_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('main')

def main():
    args = parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # pareents parameter will created missing intermedia folders if not exists.
        
        # Setup logging
        logger = setup_logging(output_dir, args.debug)
        logger.info(f"Starting vessel segmentation for case: {args.case_id}")
        
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
        
        # Update config with command line arguments, in case the config file does not have any value.
        config['processing'] = config.get('processing', {})
        config['processing']['force_recompute'] = args.force_recompute
        config['processing']['enable_caching'] = args.enable_caching
        
        config['preprocessing'] = config.get('preprocessing', {})
        config['preprocessing']['auto_preprocess'] = args.auto_preprocess
        
        # Validate input directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Check required input files
        ensure_input_files(args.input_dir, args.case_id)
        
        if args.optimize:
            logger.info("Running optimization")
            optimizer = OptimizationRunner(
                base_config=config,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                case_id=args.case_id,
                n_trials=args.optimization_trials
            )
            
            best_config = optimizer.run_optimization()
            best_config_path = output_dir / 'optimization_results' / 'best_config.yaml'
            config_loader.save_config(best_config, best_config_path)
            
            logger.info(f"Optimization completed. Best configuration saved to: {best_config_path}")
            print(f"\nBest configuration saved to: {best_config_path}")
            
        else:
            logger.info("Running standard pipeline")
            pipeline = VesselSegmentationPipeline(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                case_id=args.case_id,
                config=config,
                preprocess=args.preprocess,
                force_preprocess=args.force_preprocess,
                debug=args.debug
            )
            
            results = pipeline.run()
            
            logger.info("Pipeline completed successfully")
            print("\nVessel segmentation completed successfully!")
            print("\nOutput files:")
            for key, path in results['output_files'].items():
                print(f"{key}: {path}")
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()