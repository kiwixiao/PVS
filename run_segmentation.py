import SimpleITK as sitk
import numpy as np
import os
import argparse
from vessel_segmentation.preprocessing import resample_to_isotropic, segment_lungs, process_lung_mask, save_intermediate_results
from vessel_segmentation.vessel_enhancement import calculate_vesselness, save_vesselness_results
from vessel_segmentation.thresholding import calculate_adaptive_threshold, save_threshold_results
from vessel_segmentation.centerline_extraction import extract_centerlines, save_centerline_results
from vessel_segmentation.local_thresholding import local_optimal_thresholding, save_local_threshold_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vessel segmentation pipeline')
    parser.add_argument('--input', default='image_data/ct.nrrd', help='Input image path')
    parser.add_argument('--intermediate-dir', default='intermediate_results', help='Directory for intermediate results')
    parser.add_argument('--output-dir', default='output', help='Directory for final results')
    parser.add_argument('--load-hessian', action='store_true', help='Load pre-computed Hessian results if available')
    parser.add_argument('--skip-to-threshold', action='store_true', help='Skip to thresholding step (load vesselness results)')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.intermediate_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if isotropic image and eroded mask exist
    iso_image_path = os.path.join(args.intermediate_dir, 'isotropic_image.nrrd')
    eroded_mask_path = os.path.join(args.intermediate_dir, 'eroded_mask.nrrd')
    
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
        save_intermediate_results(preprocess_results, args.intermediate_dir)
        eroded_mask = preprocess_results['eroded_mask']
        
        # Save masked image for future use
        masked_image = image_array * eroded_mask
        sitk.WriteImage(sitk.GetImageFromArray(masked_image), os.path.join(args.intermediate_dir, 'masked_image_for_hessian.nrrd'))
    
    if not args.skip_to_threshold:
        # Vessel enhancement using eroded mask
        print("Enhancing vessels...")
        # Calculate scales from voxel size to 4.5mm (7 scales)
        min_scale = 0.6  # Start at isotropic voxel size
        max_scale = 4.5  # End at 4.5mm as per paper
        num_scales = 7
        scales = np.exp(np.linspace(np.log(min_scale), np.log(max_scale), num_scales))
        
        vesselness, sigma_max, vessel_direction = calculate_vesselness(
            image_array,
            eroded_mask,
            scales,
            args.intermediate_dir,
            load_hessian=args.load_hessian
        )
        save_vesselness_results(vesselness, sigma_max, vessel_direction, args.intermediate_dir)
    else:
        # Load pre-computed vesselness results
        print("Loading pre-computed vesselness results...")
        vesselness = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.intermediate_dir, 'vesselness.nrrd')))
        sigma_max = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.intermediate_dir, 'sigma_max.nrrd')))
        vessel_direction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.intermediate_dir, 'vessel_direction.nrrd')))
    
    # Check if binary vessels exist
    binary_vessels_path = os.path.join(args.intermediate_dir, 'binary_vessels.nrrd')
    if os.path.exists(binary_vessels_path):
        print("Found pre-computed binary vessels, loading...")
        binary_vessels = sitk.GetArrayFromImage(sitk.ReadImage(binary_vessels_path))
        print("Successfully loaded binary vessels")
    else:
        # Thresholding
        print("Applying adaptive threshold...")
        binary_vessels, threshold_map = calculate_adaptive_threshold(vesselness, sigma_max, voxel_size_mm=0.6)
        save_threshold_results(binary_vessels, threshold_map, args.intermediate_dir)
    
    # Check if centerline files exist
    centerline_path = os.path.join(args.intermediate_dir, 'centerlines.nrrd')
    point_types_path = os.path.join(args.intermediate_dir, 'centerline_point_types.nrrd')
    
    if os.path.exists(centerline_path) and os.path.exists(point_types_path):
        print("Found pre-computed centerlines, loading...")
        centerlines = sitk.GetArrayFromImage(sitk.ReadImage(centerline_path))
        point_types = sitk.GetArrayFromImage(sitk.ReadImage(point_types_path))
        print("Successfully loaded centerline results")
    else:
        print("Extracting centerlines...")
        centerlines, point_types = extract_centerlines(binary_vessels)
        save_centerline_results(centerlines, point_types, args.intermediate_dir)
    
    # Local thresholding
    print("Applying local optimal thresholding...")
    final_vessels, local_thresholds = local_optimal_thresholding(
        binary_vessels,
        vesselness,
        centerlines,
        point_types,
        sigma_max,
        vessel_direction
    )
    save_local_threshold_results(final_vessels, local_thresholds, args.output_dir)
    
    print("Segmentation complete!")

if __name__ == "__main__":
    main()
