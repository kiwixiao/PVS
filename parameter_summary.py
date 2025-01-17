# Vessel Segmentation Pipeline Parameter Summary
# This file documents the key parameters used in the vessel segmentation pipeline
# and explains their effects on the results.

#------------------------------------------------------------------------------
# Image Preprocessing Parameters
#------------------------------------------------------------------------------

# Target isotropic spacing for image resampling (in mm)
ISOTROPIC_SPACING = 0.6  
# Effect: Smaller values give higher resolution but increase memory usage and computation time
# Larger values reduce detail but speed up processing
# 0.6mm is chosen to balance detail preservation and computational efficiency

# Erosion distance for lung mask (in voxels)
EROSION_DISTANCE = 5
# Effect: Larger values create bigger margin from lung boundary, reducing false positives
# Smaller values preserve more vessels near lung boundary
# Current value helps avoid boundary artifacts while keeping most vessels

#------------------------------------------------------------------------------
# Vessel Enhancement Parameters
#------------------------------------------------------------------------------

# Minimum scale for Hessian analysis (in mm)
MIN_SCALE = 0.6
# Effect: Should match smallest vessel size of interest
# Smaller values detect finer vessels but increase noise sensitivity

# Maximum scale for Hessian analysis (in mm)
MAX_SCALE = 6.0
# Effect: Should match largest vessel size of interest
# Larger values detect bigger vessels but increase computation time

# Frangi vesselness parameters
ALPHA = 0.5  # Controls sensitivity to vessel-like structures vs plate-like structures
# Effect: Smaller values increase sensitivity to plate-like structures
# Larger values favor more tubular structures

BETA = 0.5   # Controls blob-like vs line-like structure differentiation
# Effect: Smaller values increase sensitivity to blob-like structures
# Larger values favor more line-like structures

C = 70.0     # Controls background noise suppression
# Effect: Smaller values increase sensitivity but also noise
# Larger values suppress more noise but might miss weak vessels

#------------------------------------------------------------------------------
# Thresholding Parameters
#------------------------------------------------------------------------------

# Minimum threshold for smallest vessels
TMIN = 0.07
# Effect: Lower values detect more small vessels but increase false positives
# Higher values reduce false positives but might miss small vessels

# Maximum threshold for largest vessels
TMAX = 0.17
# Effect: Lower values detect more large vessels but might merge nearby vessels
# Higher values better separate large vessels but might fragment them

# Width parameter for scale transition
SCALE_WIDTH = 2.0  # Multiplied by voxel_size_mm
# Effect: Controls how quickly thresholds transition between vessel sizes
# Larger values make smoother transitions but might blur size distinctions

#------------------------------------------------------------------------------
# Local Thresholding Parameters
#------------------------------------------------------------------------------

# ROI multiplier for local processing
ROI_MULTIPLIER = 2.0
# Effect: Larger values consider more context but increase computation time
# Smaller values are faster but might miss context

# Minimum radius for cylindrical ROI (in mm)
MIN_RADIUS_CYL = 1.0
# Effect: Sets minimum size of local processing region
# Too small might miss context, too large increases computation time

# Maximum segment length for processing
MAX_SEGMENT_LENGTH = 50
# Effect: Longer segments process more at once but use more memory
# Shorter segments are more memory efficient but might miss long-range context

# Segment overlap
SEGMENT_OVERLAP = 10
# Effect: More overlap ensures continuity but increases computation time
# Less overlap is faster but might cause discontinuities

# Minimum vesselness threshold
MIN_VESSELNESS = 0.01
# Effect: Lower values keep more weak responses but increase noise
# Higher values reduce noise but might miss weak vessels

#------------------------------------------------------------------------------
# Post-processing Parameters
#------------------------------------------------------------------------------

# Minimum component size (in voxels)
MIN_COMPONENT_SIZE = 50
# Effect: Smaller values keep more isolated vessels but might include noise
# Larger values remove more noise but might remove small vessel segments

# Connectivity type for component analysis
CONNECTIVITY = 26  # 6, 18, or 26
# Effect: Higher connectivity preserves more connections but might create spurious ones
# Lower connectivity might break some connections but is more conservative

#------------------------------------------------------------------------------
# Output Parameters
#------------------------------------------------------------------------------

# Scale to diameter conversion factor
SCALE_TO_DIAMETER = 2 * np.sqrt(2)
# Effect: Converts scale of maximum response to approximate vessel diameter
# Based on theoretical relationship between Gaussian scale and vessel size 