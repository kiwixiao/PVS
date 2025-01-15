Vessel Segmentation Pipeline
========================

This pipeline implements vessel segmentation with scale optimization using gradient descent.

Input Requirements
-----------------
- Input CT image (default: image_data/ct.nrrd)
- Project name for output organization

Pipeline Steps and File Dependencies
----------------------------------
1. Preprocessing:
   - Resamples input to 0.6mm isotropic resolution
   - Segments lung mask and creates eroded mask
   - Applies deconvolution to isotropic image
   Required files: None
   Outputs: 
   - intermediate_results/isotropic_image.nrrd (debug)
   - intermediate_results/eroded_mask.nrrd
   - intermediate_results/deconvolved_image.nrrd
   - intermediate_results/masked_deconvolved_image.nrrd (debug)

2. Hessian Analysis & Scale Optimization:
   - Calculates Hessian using discrete Gaussian kernel (size=5)
   - Initial scales: 10 scales from 0.6mm to 6.0mm (geometric progression)
   - Optimizes scale using gradient descent for maximum vesselness
   Required files: 
   - deconvolved_image.nrrd
   - eroded_mask.nrrd
   Outputs:
   - intermediate_results/vesselness.nrrd
   - intermediate_results/sigma_max.nrrd
   - intermediate_results/vessel_direction.nrrd
   - intermediate_results/scale_optimization.json
   - intermediate_results/V-*-010.nrrd (vesselness at each initial scale)

3. Adaptive Thresholding:
   - Creates binary vessel mask using adaptive thresholding
   Required files:
   - vesselness.nrrd
   - sigma_max.nrrd
   Outputs:
   - intermediate_results/binary_vessels.nrrd
   - intermediate_results/threshold_map.nrrd

4. Centerline Extraction:
   - Extracts vessel centerlines and classifies point types
   Required files:
   - binary_vessels.nrrd
   Outputs:
   - intermediate_results/centerlines.nrrd
   - intermediate_results/centerline_point_types.nrrd

5. Local Optimal Thresholding:
   - Refines segmentation using local ROI analysis
   Required files:
   - binary_vessels.nrrd
   - vesselness.nrrd
   - centerlines.nrrd
   - centerline_point_types.nrrd
   - sigma_max.nrrd
   - vessel_direction.nrrd
   Outputs:
   - final_results/final_vessels_{parameter_set}.nrrd
   - final_results/local_thresholds_{parameter_set}.nrrd
   - final_results/segmentation_metadata_{parameter_set}.json

Command Line Arguments
--------------------
Required:
--project-name: Name for output organization

Optional:
--input: Input image path (default: image_data/ct.nrrd)
--parameter-set: Threshold parameters [default|aggressive|very_aggressive|conservative]
--skip-to-threshold: Skip to thresholding step (requires existing vesselness results)

Individual parameter overrides:
--min-vesselness: Minimum vesselness threshold
--roi-multiplier: ROI size multiplier
--min-radius-cyl: Minimum cylinder radius (mm)
--min-radius-sphere: Minimum sphere radius (mm)
--max-segment-length: Maximum segment length (voxels)
--overlap: Segment overlap (voxels)

Pipeline Skipping Logic
---------------------
The pipeline implements smart skipping of steps based on existing files:

1. Preprocessing Skip:
   If both exist:
   - deconvolved_image.nrrd
   - eroded_mask.nrrd
   Then: Skip preprocessing steps

2. Hessian Skip:
   If all exist:
   - vesselness.nrrd
   - sigma_max.nrrd
   - vessel_direction.nrrd
   Then: Skip Hessian calculation

3. Binary Vessels Skip:
   If exists:
   - binary_vessels.nrrd
   Then: Skip adaptive thresholding

4. Centerline Skip:
   If both exist:
   - centerlines.nrrd
   - centerline_point_types.nrrd
   Then: Skip centerline extraction

Scale Optimization Details
------------------------
The pipeline implements scale optimization using gradient descent:
- Initial scales: 10 scales from 0.6mm to 6.0mm
- Optimization method: Gradient descent with central difference
- Learning rate: 0.1
- Max iterations: 10
- Convergence threshold: 1e-4
- Bounds: [0.6mm, 6.0mm]

Results are saved in scale_optimization.json including:
- Initial and optimal scales
- Optimization trajectories
- Statistical analysis of scale distribution
- Improvement metrics

Debug Outputs
------------
Several intermediate files are saved for debugging:
1. isotropic_image.nrrd: Resampled input image
2. deconvolved_image.nrrd: After deconvolution
3. masked_deconvolved_image.nrrd: Masked version for verification
4. V-*-010.nrrd: Vesselness response at each initial scale
5. scale_optimization.json: Detailed optimization records 