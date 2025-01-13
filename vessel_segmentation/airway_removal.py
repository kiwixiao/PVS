import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, binary_closing
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)

def segment_airways(ct_scan, lower_threshold=-1000, upper_threshold=-400):
    """Segment airways using connected thresholding"""
    try:
        logger.info("Segmenting airways...")
        sitk_image = sitk.GetImageFromArray(ct_scan)
        
        airway_mask = sitk.ConnectedThreshold(
            sitk_image,
            seedList=[(int(ct_scan.shape[0]/2), int(ct_scan.shape[1]/2), int(ct_scan.shape[2]/2))],
            lower=lower_threshold,
            upper=upper_threshold
        )
        
        airway_mask = sitk.GetArrayFromImage(airway_mask)
        airway_mask = binary_opening(airway_mask, structure=np.ones((3,3,3)))
        airway_mask = binary_closing(airway_mask, structure=np.ones((3,3,3)))
        
        return airway_mask.astype(bool)
    except Exception as e:
        logger.error(f"Airway segmentation failed: {str(e)}")
        raise

def remove_airway_walls(lung_mask, airway_mask):
    """Remove airway walls from lung mask"""
    try:
        logger.info("Removing airway walls...")
        dilated_airway = binary_dilation(airway_mask, structure=np.ones((3,3,3)))
        cleaned_mask = lung_mask & ~dilated_airway
        cleaned_mask = binary_closing(cleaned_mask, structure=np.ones((3,3,3)))
        return cleaned_mask
    except Exception as e:
        logger.error(f"Airway wall removal failed: {str(e)}")
        raise
