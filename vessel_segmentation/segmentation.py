import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
import logging

logger = logging.getLogger(__name__)

def local_threshold_segmentation(vesselness, lung_mask, window_size=32, k=0.5):
    """Perform adaptive local thresholding segmentation based on local statistics"""
    try:
        logger.info("Performing adaptive local thresholding...")
        from scipy.ndimage import uniform_filter
        
        # Initialize output
        segmented = np.zeros_like(vesselness, dtype=bool)
        
        # Compute local statistics
        local_mean = uniform_filter(vesselness.astype(float), size=window_size)
        local_sq_mean = uniform_filter(vesselness.astype(float)**2, size=window_size)
        local_std = np.sqrt(local_sq_mean - local_mean**2)
        
        # Calculate adaptive threshold: T = μ + kσ
        thresholds = local_mean + k * local_std
        
        # Apply threshold within lung mask
        segmented = (vesselness >= thresholds) & lung_mask
        
        # Post-processing
        from skimage.morphology import binary_closing, binary_opening
        from scipy.ndimage import binary_fill_holes
        
        # Remove small objects and fill holes
        segmented = binary_opening(segmented)
        segmented = binary_closing(segmented)
        segmented = binary_fill_holes(segmented)
        
        return segmented
    except Exception as e:
        logger.error(f"Local thresholding failed: {str(e)}")
        raise
