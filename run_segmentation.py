import SimpleITK as sitk
import numpy as np
import os
import argparse
from vessel_segmentation.preprocessing import resample_to_isotropic, segment_lungs, process_lung_mask, save_intermediate_results
from vessel_segmentation.vessel_enhancement import calculate_vesselness, save_vesselness_results
from vessel_segmentation.thresholding import calculate_adaptive_threshold, save_threshold_results
from vessel_segmentation.centerline_extraction import extract_centerlines, save_centerline_results
from vessel_segmentation.local_thresholding import local_optimal_thresholding, save_local_threshold_results

def parse_args():
    parser = argparse.ArgumentParser(description='Vessel segmentation pipeline with customizable parameters')
    parser.add_argument('--input', type=str, default='image_data/ct.nrrd',
                      help='Input image file')
    parser.add_argument('--project-name', type=str, default='default',
                      help='Project name for organizing output files')
    parser.add_argument('--skip-to-threshold', action='store_true',
                      help='Skip to thresholding step')
    
    # Parameter selection options
    param_group = parser.add_argument_group('Parameter Selection')
    param_group.add_argument('--parameter-set', type=str, default='default',
                          choices=['default', 'aggressive', 'very_aggressive', 'conservative'],
                          help='Predefined parameter set for local optimal thresholding')
    
    # Individual parameter overrides
    override_group = parser.add_argument_group('Parameter Overrides')
    override_group.add_argument('--min-vesselness', type=float,
                             help='Minimum vesselness threshold (default: 0.05)')
    override_group.add_argument('--roi-multiplier', type=float,
                             help='ROI size multiplier (default: 1.5)')
    override_group.add_argument('--min-radius-cyl', type=float,
                             help='Minimum cylinder radius in mm (default: 3.0)')
    override_group.add_argument('--min-radius-sphere', type=float,
                             help='Minimum sphere radius in mm (default: 4.0)')
    override_group.add_argument('--max-segment-length', type=int,
                             help='Maximum segment length in voxels (default: 20)')
    override_group.add_argument('--overlap', type=int,
                             help='Segment overlap in voxels (default: 5)')
    
    args = parser.parse_args()
    
    # Collect custom parameters if any are specified
    custom_params = {}
    if args.min_vesselness is not None:
        custom_params['min_vesselness'] = args.min_vesselness
    if args.roi_multiplier is not None:
        custom_params['roi_multiplier'] = args.roi_multiplier
    if args.min_radius_cyl is not None:
        custom_params['min_radius_cyl'] = args.min_radius_cyl
    if args.min_radius_sphere is not None:
        custom_params['min_radius_sphere'] = args.min_radius_sphere
    if args.max_segment_length is not None:
        custom_params['max_segment_length'] = args.max_segment_length
    if args.overlap is not None:
        custom_params['overlap'] = args.overlap
    
    args.custom_params = custom_params if custom_params else None
    return args

def setup_output_dirs(project_name):
    """Setup output directories for the project."""
    base_dir = f'output_{project_name}'
    dirs = {
        'base': base_dir,
        'intermediate': os.path.join(base_dir, 'intermediate_results'),
        'final': os.path.join(base_dir, 'final_results')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup output directories
    output_dirs = setup_output_dirs(args.project_name)
    
    # Check if isotropic image and eroded mask exist
    iso_image_path = os.path.join(output_dirs['intermediate'], 'isotropic_image.nrrd')
    eroded_mask_path = os.path.join(output_dirs['intermediate'], 'eroded_mask.nrrd')
    
    if os.path.exists(iso_image_path) and os.path.exists(eroded_mask_path):
        print("Found isotropic image and eroded mask, skipping preprocessing steps...")
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(iso_image_path))
        eroded_mask = sitk.GetArrayFromImage(sitk.ReadImage(eroded_mask_path))
        voxel_spacing = (0.6, 0.6, 0.6)  # Since we always resample to 0.6mm isotropic
    else:
        missing_files = []
        if not os.path.exists(iso_image_path):
            missing_files.append('isotropic_image.nrrd')
        if not os.path.exists(eroded_mask_path):
            missing_files.append('eroded_mask.nrrd')
        print(f"Missing files: {', '.join(missing_files)}, running preprocessing steps...")
        
        # Load input image
        print("Loading input image...")
        input_image = sitk.ReadImage(args.input)
        original_spacing = input_image.GetSpacing()
        print(f"Original image spacing: {original_spacing}")
        
        # Resample to isotropic resolution
        print("Resampling to isotropic resolution (0.6mm)...")
        iso_image = resample_to_isotropic(input_image, target_spacing=0.6)
        iso_spacing = iso_image.GetSpacing()
        print(f"Isotropic image spacing: {iso_spacing}")
        
        # Save isotropic image
        sitk.WriteImage(iso_image, iso_image_path)
        
        # Convert to numpy array for processing
        image_array = sitk.GetArrayFromImage(iso_image)
        voxel_spacing = iso_image.GetSpacing()
        
        # Preprocessing
        print("Segmenting lungs...")
        lung_mask = segment_lungs(iso_image)  # Pass isotropic image
        preprocess_results = process_lung_mask(lung_mask, voxel_spacing)
        save_intermediate_results(preprocess_results, output_dirs['intermediate'])
        eroded_mask = preprocess_results['eroded_mask']
        
        # Save masked image for future use
        masked_image = image_array * eroded_mask
        sitk.WriteImage(sitk.GetImageFromArray(masked_image), 
                       os.path.join(output_dirs['intermediate'], 'masked_image_for_hessian.nrrd'))
    
    if not args.skip_to_threshold:
        # Vessel enhancement using eroded mask
        print("Enhancing vessels...")
        # Calculate scales using geometric progression from 0.6mm to 6.0mm (10 scales)
        min_scale = 0.6  # Start at isotropic voxel size
        max_scale = 6.0  # Maximum scale for detecting larger vessels
        num_scales = 10
        
        # Generate geometric sequence of scales
        scales = np.exp(np.linspace(np.log(min_scale), np.log(max_scale), num_scales))
        
        # Calculate approximate vessel diameters detectable at each scale
        vessel_diameters = scales * 2 * np.sqrt(2)
        print("\nScale progression and corresponding vessel diameters:")
        for i, (scale, diameter) in enumerate(zip(scales, vessel_diameters)):
            print(f"Scale {i+1}: {scale:.2f}mm -> Vessel diameter: {diameter:.2f}mm")
        print()
        
        vesselness, sigma_max, vessel_direction = calculate_vesselness(
            image_array,
            eroded_mask,
            scales,
            output_dirs['intermediate']
        )
        save_vesselness_results(vesselness, sigma_max, vessel_direction, output_dirs['intermediate'])
    else:
        # Load pre-computed vesselness results
        print("Loading pre-computed vesselness results...")
        vesselness = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(output_dirs['intermediate'], 'vesselness.nrrd')))
        sigma_max = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(output_dirs['intermediate'], 'sigma_max.nrrd')))
        vessel_direction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(output_dirs['intermediate'], 'vessel_direction.nrrd')))
    
    # Check if binary vessels exist
    binary_vessels_path = os.path.join(output_dirs['intermediate'], 'binary_vessels.nrrd')
    if os.path.exists(binary_vessels_path):
        print("Found pre-computed binary vessels, loading...")
        binary_vessels = sitk.GetArrayFromImage(sitk.ReadImage(binary_vessels_path))
        print("Successfully loaded binary vessels")
    else:
        # Thresholding
        print("Applying adaptive threshold...")
        binary_vessels, threshold_map = calculate_adaptive_threshold(vesselness, sigma_max, voxel_size_mm=0.6)
        save_threshold_results(binary_vessels, threshold_map, output_dirs['intermediate'])
    
    # Check if centerline files exist
    centerline_path = os.path.join(output_dirs['intermediate'], 'centerlines.nrrd')
    point_types_path = os.path.join(output_dirs['intermediate'], 'centerline_point_types.nrrd')
    
    if os.path.exists(centerline_path) and os.path.exists(point_types_path):
        print("Found pre-computed centerlines, loading...")
        centerlines = sitk.GetArrayFromImage(sitk.ReadImage(centerline_path))
        point_types = sitk.GetArrayFromImage(sitk.ReadImage(point_types_path))
        print("Successfully loaded centerline results")
    else:
        print("Extracting centerlines...")
        centerlines, point_types = extract_centerlines(binary_vessels)
        save_centerline_results(centerlines, point_types, output_dirs['intermediate'])
    
    # Local thresholding
    print("Applying local optimal thresholding...")
    final_vessels, local_thresholds = local_optimal_thresholding(
        binary_vessels,
        vesselness,
        centerlines,
        point_types,
        sigma_max,
        vessel_direction,
        parameter_set=args.parameter_set
    )
    save_local_threshold_results(
        final_vessels, 
        local_thresholds, 
        output_dirs['final'],
        parameter_set=args.parameter_set,
        custom_params=args.custom_params
    )
    
    print(f"Segmentation complete! Results saved in {output_dirs['base']}")

if __name__ == "__main__":
    main()
