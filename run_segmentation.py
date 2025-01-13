import SimpleITK as sitk
from vessel_segmentation.preprocessing import *
from vessel_segmentation.vessel_enhancement import *
from vessel_segmentation.thresholding import *
from vessel_segmentation.centerline_extraction import *
from vessel_segmentation.local_thresholding import *

def main():
    # Input/output paths
    input_image_path = 'image_data/ct.nrrd'
    intermediate_dir = 'intermediate_results'
    output_dir = 'output'
    
    # Load input image
    print("Loading input image...")
    input_image = sitk.ReadImage(input_image_path)
    voxel_spacing = input_image.GetSpacing()
    image_array = sitk.GetArrayFromImage(input_image)
    
    # Preprocessing
    print("Segmenting lungs...")
    lung_mask = segment_lungs(input_image)
    preprocess_results = process_lung_mask(lung_mask, voxel_spacing)
    save_intermediate_results(preprocess_results, intermediate_dir)
    
    # Vessel enhancement
    print("Enhancing vessels...")
    scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    vesselness, sigma_max = calculate_vesselness(
        image_array, 
        preprocess_results['eroded_mask'],
        scales
    )
    save_vesselness_results(vesselness, sigma_max, intermediate_dir)
    
    # Thresholding
    print("Applying adaptive threshold...")
    binary_vessels, threshold_map = calculate_adaptive_threshold(vesselness, sigma_max)
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

if __name__ == '__main__':
    main()
