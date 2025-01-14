import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes as fill_holes
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt
import os

def resample_to_isotropic(image: sitk.Image, target_spacing: float = 0.6) -> sitk.Image:
    """Resample image to isotropic resolution.
    
    Args:
        image: Input SimpleITK image
        target_spacing: Target isotropic spacing in mm (default: 0.6mm)
        
    Returns:
        Resampled image with isotropic spacing
    """
    # Get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size
    new_spacing = [target_spacing] * 3
    new_size = [
        int(round(osz * ospc / target_spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    
    # Create resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    
    # Set interpolator based on image type
    if image.GetPixelID() in [sitk.sitkUInt8, sitk.sitkInt8]:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    
    # Perform resampling
    resampled_image = resample.Execute(image)
    
    return resampled_image

def segment_lungs(input_image):
    """Segment lungs using lungmask"""
    import lungmask
    from lungmask import mask
    
    # Convert to numpy array
    image_array = sitk.GetArrayFromImage(input_image)
    
    # Get lung mask using lungmask
    segmentation = mask.apply(image_array)
    lung_mask = ((segmentation == 1) | (segmentation == 2)).astype(np.uint8)  # Combine left and right lung masks
    
    # Remove airways from lung mask
    print("Removing airways...")
    lung_mask = remove_airways(image_array, lung_mask)
    
    return sitk.GetImageFromArray(lung_mask)

def remove_airways(image_array, lung_mask, threshold=-950):
    """Remove airways from lung mask using region growing"""
    # Convert to SimpleITK images
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(lung_mask)
    
    # Find trachea seed point (usually in the middle top of the image)
    z, y, x = image_array.shape
    seed_z = z // 4  # Start from upper quarter of the image
    seed_y = y // 2
    seed_x = x // 2
    
    # Create connected threshold filter
    segmentation = sitk.ConnectedThreshold(
        image,
        seedList=[(seed_x, seed_y, seed_z)],
        lower=float(-1024),
        upper=float(threshold)
    )
    
    # Dilate airways to ensure complete removal
    dilated_airways = sitk.BinaryDilate(segmentation, [3,3,3])
    
    # Remove airways from lung mask
    cleaned_mask = mask - (mask * dilated_airways)
    
    return sitk.GetArrayFromImage(cleaned_mask)

def process_lung_mask(lung_mask, voxel_spacing):
    """Process lung mask to create eroded mask and distance map"""
    # Convert to numpy array and ensure uint8
    lung_mask = sitk.GetArrayFromImage(lung_mask).astype(np.uint8)
    
    # Calculate distance map from original lung mask
    print("Calculating distance map...")
    # Invert mask for distance calculation (we want distances inside the lungs)
    inverted_mask = (lung_mask == 0).astype(np.uint8)
    mask_sitk = sitk.GetImageFromArray(inverted_mask)
    
    # Calculate distance map - will be positive inside lungs
    distance_map_sitk = sitk.SignedMaurerDistanceMap(mask_sitk, 
                                                    squaredDistance=False,
                                                    useImageSpacing=True)
    distance_map = sitk.GetArrayFromImage(distance_map_sitk)
    
    # Make distances positive inside lungs, zero outside
    distance_map = distance_map * (lung_mask > 0)
    
    # Create eroded mask by thresholding the distance map
    # Points with distance > 2 voxels from the boundary are kept
    eroded_mask = (distance_map > 2).astype(np.uint8)
    
    return {
        'lung_mask': lung_mask,
        'eroded_mask': eroded_mask,
        'distance_map': distance_map
    }

def calculate_distance_map(mask, voxel_spacing):
    """Calculate distance map from mask"""
    # Convert to SimpleITK image and invert (we want distances inside the mask)
    inverted_mask = (mask == 0).astype(np.uint8)
    mask_sitk = sitk.GetImageFromArray(inverted_mask)
    
    # Calculate distance transform
    distance_map_sitk = sitk.SignedMaurerDistanceMap(mask_sitk, 
                                                    squaredDistance=False,
                                                    useImageSpacing=True)
    distance_map = sitk.GetArrayFromImage(distance_map_sitk)
    
    # Make distances positive inside mask, zero outside
    distance_map = distance_map * (mask > 0)
    
    return distance_map

def save_intermediate_results(results, output_dir):
    """Save preprocessing intermediate results"""
    os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(
        sitk.GetImageFromArray(results['lung_mask']),
        os.path.join(output_dir, 'lung_mask.nrrd')
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(results['eroded_mask']),
        os.path.join(output_dir, 'eroded_mask.nrrd')
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(results['distance_map']),
        os.path.join(output_dir, 'distance_map.nrrd')
    )

def chunked_distance_transform(mask, voxel_spacing, chunk_size=128):
    """Calculate distance transform in chunks to manage memory"""
    from tqdm import tqdm
    import gc
    
    shape = mask.shape
    distance_map = np.zeros_like(mask, dtype=np.float32)
    
    # Calculate padding size (should be at least max expected distance)
    pad_size = 32  # Adjust based on expected vessel sizes
    
    # Calculate number of chunks
    nx = int(np.ceil(shape[0] / chunk_size))
    ny = int(np.ceil(shape[1] / chunk_size))
    nz = int(np.ceil(shape[2] / chunk_size))
    
    # Process chunks with progress bar
    total_chunks = nx * ny * nz
    with tqdm(total=total_chunks, desc="Calculating distance map", leave=False) as pbar:
        for x in range(nx):
            x_start = x * chunk_size
            x_end = min((x + 1) * chunk_size, shape[0])
            
            for y in range(ny):
                y_start = y * chunk_size
                y_end = min((y + 1) * chunk_size, shape[1])
                
                for z in range(nz):
                    z_start = z * chunk_size
                    z_end = min((z + 1) * chunk_size, shape[2])
                    
                    # Extract chunk with padding
                    x_pad_start = max(0, x_start - pad_size)
                    x_pad_end = min(shape[0], x_end + pad_size)
                    y_pad_start = max(0, y_start - pad_size)
                    y_pad_end = min(shape[1], y_end + pad_size)
                    z_pad_start = max(0, z_start - pad_size)
                    z_pad_end = min(shape[2], z_end + pad_size)
                    
                    # Process padded chunk
                    chunk_mask = mask[x_pad_start:x_pad_end,
                                    y_pad_start:y_pad_end,
                                    z_pad_start:z_pad_end]
                    
                    # Calculate distance transform for chunk
                    chunk_dist = distance_transform_edt(chunk_mask, sampling=voxel_spacing)
                    
                    # Extract the region corresponding to the original chunk
                    x_offset = x_start - x_pad_start
                    y_offset = y_start - y_pad_start
                    z_offset = z_start - z_pad_start
                    chunk_result = chunk_dist[x_offset:x_offset + (x_end - x_start),
                                            y_offset:y_offset + (y_end - y_start),
                                            z_offset:z_offset + (z_end - z_start)]
                    
                    # Store result
                    distance_map[x_start:x_end,
                               y_start:y_end,
                               z_start:z_end] = chunk_result
                    
                    pbar.update(1)
                    gc.collect()
    
    # Normalize distances
    max_dist = np.max(distance_map)
    if max_dist > 0:
        distance_map = distance_map / max_dist
    
    return distance_map
