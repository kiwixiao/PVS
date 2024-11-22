import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull
from skimage.morphology import convex_hull_image

def create_convex_lung_mask(nifti_path, output_path=None):
    """
    Convert a binary lung mask from concave to convex shape using convex hull.
    
    Parameters:
    -----------
    nifti_path : str
        Path to input NIfTI file containing binary lung mask
    output_path : str, optional
        Path to save the output convex mask. If None, doesn't save.
        
    Returns:
    --------
    nibabel.Nifti1Image
        NIfTI image containing the convex hull mask
    """
    # Load the NIfTI file
    img = nib.load(nifti_path)
    mask_data = img.get_fdata()
    
    # Create empty array for convex hull mask
    convex_mask = np.zeros_like(mask_data)
    
    # Process each slice separately (assuming axial orientation)
    for z in range(mask_data.shape[2]):
        slice_mask = mask_data[:, :, z].astype(bool)
        if slice_mask.any():  # Only process non-empty slices
            convex_mask[:, :, z] = convex_hull_image(slice_mask)
    
    # Create new NIfTI image with same header as input
    convex_img = nib.Nifti1Image(convex_mask, img.affine, img.header)
    
    # Save if output path is provided
    if output_path:
        nib.save(convex_img, output_path)
        
    return convex_img

# Example usage:
if __name__ == "__main__":
    input_path = "lung_mask.nii.gz"
    output_path = "lung_mask_convex.nii.gz"
    convex_mask = create_convex_lung_mask(input_path, output_path)