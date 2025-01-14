import SimpleITK as sitk
import numpy as np
import os
from .preprocessing import resample_to_isotropic, segment_lungs, process_lung_mask, save_intermediate_results
from .vessel_enhancement import calculate_vesselness, save_vesselness_results
from .thresholding import calculate_adaptive_threshold, save_threshold_results

def main():
    # Input/output paths
    input_image_path = 'image_data/ct.nrrd'
    intermediate_dir = 'intermediate_results'
    output_dir = 'output'
    
    # Create output directories
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input image
    print("Loading input image...")
    input_image = sitk.ReadImage(input_image_path)
    original_spacing = input_image.GetSpacing()
    print(f"Original image spacing: {original_spacing}")
    
    # Resample to isotropic resolution
    print("Resampling to isotropic resolution (0.6mm)...")
    iso_image = resample_to_isotropic(input_image, target_spacing=0.6)
    iso_spacing = iso_image.GetSpacing()
    print(f"Isotropic image spacing: {iso_spacing}")
    
    # Save isotropic image for debugging
    sitk.WriteImage(iso_image, os.path.join(intermediate_dir, 'isotropic_image.nrrd'))
    
    # Convert to numpy array for processing
    image_array = sitk.GetArrayFromImage(iso_image)
    voxel_spacing = iso_image.GetSpacing()
    
    # Preprocessing
    print("Segmenting lungs...")
    lung_mask = segment_lungs(iso_image)  # Pass isotropic image
    preprocess_results = process_lung_mask(lung_mask, voxel_spacing)
    save_intermediate_results(preprocess_results, intermediate_dir)
    
    # Vessel enhancement using eroded mask
    print("Enhancing vessels...")
    # Calculate scales from voxel size to 4.5mm (7 scales)
    min_scale = 0.6  # Start at isotropic voxel size
    max_scale = 4.5  # End at 4.5mm as per paper
    num_scales = 7
    scales = np.exp(np.linspace(np.log(min_scale), np.log(max_scale), num_scales))
    
    vesselness, sigma_max, vessel_direction = calculate_vesselness(
        image_array, 
        preprocess_results['eroded_mask'],
        scales,
        intermediate_dir
    )
    save_vesselness_results(vesselness, sigma_max, vessel_direction, intermediate_dir)
    
    # Thresholding
    print("Applying adaptive threshold...")
    binary_vessels, threshold_map = calculate_adaptive_threshold(vesselness, sigma_max, voxel_size_mm=0.6)
    save_threshold_results(binary_vessels, threshold_map, intermediate_dir)
    
    # Centerline extraction
    print("Extracting centerlines...")
    centerlines, point_types = extract_centerlines(binary_vessels)
    save_centerline_results(centerlines, point_types, intermediate_dir)
    
    # Local thresholding
    print("Applying local optimal thresholding...")
    final_vessels, local_thresholds = local_optimal_thresholding(
        binary_vessels,
        vesselness,
        centerlines,
        point_types,
        sigma_max
    )
    save_local_threshold_results(final_vessels, local_thresholds, output_dir)
    
    print("Segmentation complete!") 

if __name__ == "__main__":
    main() 