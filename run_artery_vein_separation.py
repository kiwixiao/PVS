#!/usr/bin/env python3

import os
import argparse
import logging
from vessel_segmentation.pipeline import run_av_separation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for artery-vein separation"""
    parser = argparse.ArgumentParser(
        description="Separate arteries and veins using Charbonnier's methodology"
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Directory containing vessel segmentation results"
    )
    parser.add_argument(
        "--output-folder",
        help="Directory for output files (default: input_folder/av_separation)"
    )
    parser.add_argument(
        "--max-periphery-distance",
        type=float,
        default=30.0,
        help="Maximum distance (Dmax) for periphery matching in mm (default: 30.0)"
    )
    parser.add_argument(
        "--min-matching-vessels",
        type=int,
        default=2,
        help="Minimum number of matching vessels required (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Set default output folder if not specified
    if args.output_folder is None:
        args.output_folder = os.path.join(args.input_folder, "av_separation")
    
    logger.info("Step 1: Verifying input folder...")
    if not os.path.isdir(args.input_folder):
        raise NotADirectoryError(f"Input folder does not exist: {args.input_folder}")
        
    logger.info("Step 2: Starting artery-vein separation using input from: %s", args.input_folder)
    run_av_separation(
        input_dir=args.input_folder,
        output_dir=args.output_folder,
        max_periphery_distance=args.max_periphery_distance,
        min_matching_vessels=args.min_matching_vessels
    )
    
    logger.info("Step 3: Processing completed. Results saved to: %s", args.output_folder)

if __name__ == "__main__":
    main() 