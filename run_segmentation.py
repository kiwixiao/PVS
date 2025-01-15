import SimpleITK as sitk
import numpy as np
import os
import argparse
from vessel_segmentation.preprocessing import resample_to_isotropic, segment_lungs, process_lung_mask, save_intermediate_results
from vessel_segmentation.vessel_enhancement import calculate_vesselness, deconvolve_image
from vessel_segmentation.thresholding import calculate_adaptive_threshold, save_threshold_results
from vessel_segmentation.centerline_extraction import extract_centerlines, save_centerline_results
from vessel_segmentation.local_thresholding import local_optimal_thresholding, save_local_threshold_results

def parse_args():
    parser = argparse.ArgumentParser(description='Vessel segmentation pipeline with customizable parameters')
    parser.add_argument('--input', type=str, default='image_data/ct.nrrd',
                      help='Input image file')
    parser.add_argument('--project-name', type=str, required=True,
                      help='Project name for output organization')
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
    """Setup output directory structure
    
    Args:
        project_name: Name of the project/experiment
        
    Returns:
        Dictionary containing paths to different output directories
    """
    # Create main output directory with project name
    base_output_dir = f'output_{project_name}'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create intermediate results directory
    intermediate_dir = os.path.join(base_output_dir, 'intermediate_results')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Create final results directory
    final_dir = os.path.join(base_output_dir, 'final_results')
    os.makedirs(final_dir, exist_ok=True)
    
    return {
        'base': base_output_dir,
        'intermediate': intermediate_dir,
        'final': final_dir
    }

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup output directories
    output_dirs = setup_output_dirs(args.project_name)
    
    # Check if deconvolved image and eroded mask exist
    deconv_image_path = os.path.join(output_dirs['intermediate'], 'deconvolved_image.nrrd')
    eroded_mask_path = os.path.join(output_dirs['intermediate'], 'eroded_mask.nrrd')
    
    if os.path.exists(deconv_image_path) and os.path.exists(eroded_mask_path):
        print("Found deconvolved image and eroded mask, skipping preprocessing steps...")
        deconv_image = sitk.GetArrayFromImage(sitk.ReadImage(deconv_image_path))
        eroded_mask = sitk.GetArrayFromImage(sitk.ReadImage(eroded_mask_path))
        voxel_spacing = (0.6, 0.6, 0.6)  # Since we always resample to 0.6mm isotropic
        
        # Save masked deconvolved image for debugging
        masked_deconv = deconv_image * eroded_mask
        masked_deconv_sitk = sitk.GetImageFromArray(masked_deconv)
        masked_deconv_sitk.SetSpacing((0.6, 0.6, 0.6))
        sitk.WriteImage(
            masked_deconv_sitk,
            os.path.join(output_dirs['intermediate'], 'masked_deconvolved_image.nrrd')
        )
    else:
        missing_files = []
        if not os.path.exists(deconv_image_path):
            missing_files.append('deconvolved_image.nrrd')
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
        
        # Save isotropic image for debugging
        sitk.WriteImage(iso_image, os.path.join(output_dirs['intermediate'], 'isotropic_image.nrrd'))
        
        # Convert to numpy array
        image_array = sitk.GetArrayFromImage(iso_image)
        voxel_spacing = iso_image.GetSpacing()
        
        # Preprocessing - do this before deconvolution so we have eroded_mask
        print("Segmenting lungs...")
        lung_mask = segment_lungs(iso_image)  # Pass isotropic image
        preprocess_results = process_lung_mask(lung_mask, voxel_spacing)
        save_intermediate_results(preprocess_results, output_dirs['intermediate'])
        eroded_mask = preprocess_results['eroded_mask']
        
        # Apply deconvolution after we have the mask
        print("Applying deconvolution...")
        deconv_image = deconvolve_image(image_array)
        deconv_sitk = sitk.GetImageFromArray(deconv_image)
        deconv_sitk.SetSpacing((0.6, 0.6, 0.6))  # Set correct isotropic spacing
        sitk.WriteImage(
            deconv_sitk,
            os.path.join(output_dirs['intermediate'], 'deconvolved_image.nrrd')
        )
        
        # Save masked deconvolved image for debugging
        masked_deconv = deconv_image * eroded_mask
        masked_deconv_sitk = sitk.GetImageFromArray(masked_deconv)
        masked_deconv_sitk.SetSpacing((0.6, 0.6, 0.6))
        sitk.WriteImage(
            masked_deconv_sitk,
            os.path.join(output_dirs['intermediate'], 'masked_deconvolved_image.nrrd')
        )
    
    if not args.skip_to_threshold:
        # Check if Hessian results exist
        vesselness_path = os.path.join(output_dirs['intermediate'], 'vesselness.nrrd')
        sigma_max_path = os.path.join(output_dirs['intermediate'], 'sigma_max.nrrd')
        vessel_direction_path = os.path.join(output_dirs['intermediate'], 'vessel_direction.nrrd')
        
        if (os.path.exists(vesselness_path) and 
            os.path.exists(sigma_max_path) and 
            os.path.exists(vessel_direction_path)):
            print("Found pre-computed Hessian results, loading...")
            vesselness = sitk.GetArrayFromImage(sitk.ReadImage(vesselness_path))
            sigma_max = sitk.GetArrayFromImage(sitk.ReadImage(sigma_max_path))
            vessel_direction = sitk.GetArrayFromImage(sitk.ReadImage(vessel_direction_path))
            vesselness_results = {
                'vesselness': vesselness,
                'sigma_max': sigma_max,
                'vessel_direction': vessel_direction
            }
            print("Successfully loaded Hessian results")
        else:
            # Vessel enhancement using deconvolved image and eroded mask separately
            print("Computing Hessian and vesselness...")
            min_scale = 0.6  # Start at isotropic voxel size
            max_scale = 6.0  # End at 6.0mm as discussed
            num_scales = 10
            scales = np.array([min_scale * (2 ** (i/(num_scales-1) * np.log2(max_scale/min_scale))) 
                             for i in range(num_scales)])
            print(f"Using scales: {scales}")
            
            vesselness_results = calculate_vesselness(
                deconv_image,
                eroded_mask,
                scales,
                output_dir=output_dirs['intermediate'],
                project_name=args.project_name
            )
            
            # Save vesselness results
            print("Saving vesselness results...")
            sitk.WriteImage(
                sitk.GetImageFromArray(vesselness_results['vesselness']),
                os.path.join(output_dirs['intermediate'], 'vesselness.nrrd')
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(vesselness_results['sigma_max']),
                os.path.join(output_dirs['intermediate'], 'sigma_max.nrrd')
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(vesselness_results['vessel_direction']),
                os.path.join(output_dirs['intermediate'], 'vessel_direction.nrrd')
            )
    else:
        # Load pre-computed vesselness results
        print("Loading pre-computed vesselness results...")
        vesselness = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(output_dirs['intermediate'], 'vesselness.nrrd')))
        sigma_max = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(output_dirs['intermediate'], 'sigma_max.nrrd')))
        vessel_direction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(output_dirs['intermediate'], 'vessel_direction.nrrd')))
        vesselness_results = {'vesselness': vesselness, 'sigma_max': sigma_max, 'vessel_direction': vessel_direction}
    
    # Check if binary vessels exist
    binary_vessels_path = os.path.join(output_dirs['intermediate'], 'binary_vessels.nrrd')
    if os.path.exists(binary_vessels_path):
        print("Found pre-computed binary vessels, loading...")
        binary_vessels = sitk.GetArrayFromImage(sitk.ReadImage(binary_vessels_path))
        print("Successfully loaded binary vessels")
    else:
        # Thresholding
        print("Applying adaptive threshold...")
        binary_vessels, threshold_map = calculate_adaptive_threshold(
            vesselness_results['vesselness'], 
            vesselness_results['sigma_max'], 
            voxel_size_mm=0.6
        )
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
        vesselness_results['vesselness'],
        centerlines,
        point_types,
        vesselness_results['sigma_max'],
        vesselness_results['vessel_direction'],
        parameter_set=args.parameter_set
    )
    save_local_threshold_results(
        final_vessels, 
        local_thresholds, 
        output_dirs['final'],
        parameter_set=args.parameter_set,
        custom_params=args.custom_params
    )
    
    print(f"Segmentation complete! Results saved in: {output_dirs['base']}")

if __name__ == "__main__":
    main()
