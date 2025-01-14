import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import os
import gc

def calculate_vesselness(image_array, mask, scales, output_dir=None, load_hessian=True):
    """Calculate vesselness measure using multi-scale Hessian analysis
    
    Args:
        image_array: Input image array
        mask: Binary mask (eroded mask)
        scales: List of scales for Hessian calculation
        output_dir: Directory to save intermediate results
        load_hessian: Whether to load pre-computed results if available
    """
    # Convert mask to uint8 (this should be the eroded mask)
    mask = mask.astype(np.uint8)
    
    # Save masked image for debugging (only once at the beginning)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        masked_image = image_array * (mask > 0)
        sitk.WriteImage(
            sitk.GetImageFromArray(masked_image),
            os.path.join(output_dir, 'masked_image_for_hessian.nrrd')
        )
    
    # Check if we can load pre-computed results
    if output_dir and load_hessian:
        required_files = [
            'vesselness.nrrd',
            'sigma_max.nrrd',
            'vessel_direction.nrrd'
        ]
        
        try:
            # Check if all required files exist
            missing_files = [f for f in required_files 
                           if not os.path.exists(os.path.join(output_dir, f))]
            
            if missing_files:
                print(f"Missing required files: {missing_files}")
                print("Computing vesselness from scratch...")
            else:
                print("Loading pre-computed vesselness results...")
                vesselness = sitk.GetArrayFromImage(sitk.ReadImage(
                    os.path.join(output_dir, 'vesselness.nrrd')))
                sigma_max = sitk.GetArrayFromImage(sitk.ReadImage(
                    os.path.join(output_dir, 'sigma_max.nrrd')))
                vessel_direction = sitk.GetArrayFromImage(sitk.ReadImage(
                    os.path.join(output_dir, 'vessel_direction.nrrd')))
                print("Successfully loaded all pre-computed results")
                return vesselness, sigma_max, vessel_direction
                
        except Exception as e:
            print(f"Error loading pre-computed results: {e}")
            print("Computing vesselness from scratch...")
    
    # Initialize arrays
    vesselness = np.zeros_like(image_array, dtype=np.float32)
    sigma_max = np.zeros_like(image_array, dtype=np.float32)
    vessel_direction = np.zeros(image_array.shape + (3,), dtype=np.float32)
    
    from tqdm import tqdm
    
    # Calculate vesselness for each scale with progress bar
    for scale in tqdm(scales, desc="Processing scales", leave=False):
        # Calculate Hessian
        hessian = calculate_hessian(image_array, scale)
        
        # Calculate eigenvalues and eigenvectors
        # Note: eigenvalues are sorted by magnitude |λ1| ≤ |λ2| ≤ |λ3|
        # The vessel direction is the eigenvector corresponding to λ1 (smallest magnitude)
        eigenvalues, eigenvectors = calculate_eigenvalues(hessian, mask)
        
        # Calculate vesselness for this scale
        current_vesselness = frangi_vesselness(eigenvalues, image_array)
        current_vesselness *= scale  # γ-normalization
        
        # Get vessel direction (eigenvector corresponding to smallest eigenvalue λ1)
        # This eigenvector points along the vessel direction because λ1 corresponds
        # to the direction of least intensity variation
        current_direction = eigenvectors[0]  # First eigenvector (corresponding to λ1)
        
        # Update vesselness, sigma_max, and direction only within the mask where response is higher
        update_mask = (current_vesselness > vesselness) & (mask > 0)
        vesselness[update_mask] = current_vesselness[update_mask]
        sigma_max[update_mask] = scale
        vessel_direction[update_mask] = current_direction[update_mask]
        
        # Clean up memory
        gc.collect()
    
    # Normalize vesselness within the mask
    mask_indices = mask > 0
    if np.any(mask_indices):
        vesselness_roi = vesselness[mask_indices]
        min_val = np.min(vesselness_roi)
        max_val = np.max(vesselness_roi)
        if max_val > min_val:
            vesselness[mask_indices] = (vesselness_roi - min_val) / (max_val - min_val)
    
    # Save all results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print("Saving vesselness results...")
        save_vesselness_results(vesselness, sigma_max, vessel_direction, output_dir)
    
    return vesselness, sigma_max, vessel_direction

def save_vesselness_results(vesselness, sigma_max, vessel_direction, output_dir):
    """Save vessel enhancement results
    
    Args:
        vesselness: Maximum vesselness response (Vmax)
            - Used for multi-scale thresholding
            - Used for local optimal thresholding
        sigma_max: Scale of maximum response (σmax)
            - Used to determine threshold in multi-scale thresholding
            - Used for ROI size in local optimal thresholding
        vessel_direction: Vessel direction vector (e1)
            - Used for orienting cylindrical ROIs
            - Used for centerline tracking
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save maximum vesselness response (Vmax)
    sitk.WriteImage(
        sitk.GetImageFromArray(vesselness),
        os.path.join(output_dir, 'vesselness.nrrd')
    )
    
    # Save scale of maximum response (σmax)
    sitk.WriteImage(
        sitk.GetImageFromArray(sigma_max),
        os.path.join(output_dir, 'sigma_max.nrrd')
    )
    
    # Save vessel direction vectors (e1)
    sitk.WriteImage(
        sitk.GetImageFromArray(vessel_direction),
        os.path.join(output_dir, 'vessel_direction.nrrd')
    )

def calculate_hessian(image_array, scale):
    """Calculate Hessian matrix components using SimpleITK's recursive Gaussian filter"""
    from tqdm import tqdm
    import gc
    
    # Convert numpy array to SimpleITK image
    image = sitk.GetImageFromArray(image_array)
    
    # Initialize Hessian components
    deriv_filters = []
    
    # Create progress bar for derivative calculations
    derivative_pairs = [(i,j) for i in range(3) for j in range(i, 3)]
    
    for i, j in tqdm(derivative_pairs, desc=f"Computing Hessian at scale {scale:.2f}", leave=False):
        if i == j:
            # Second derivative
            gaussian = sitk.RecursiveGaussianImageFilter()
            gaussian.SetSigma(scale)
            gaussian.SetOrder(2)
            gaussian.SetDirection(i)
            deriv = gaussian.Execute(image)
        else:
            # Mixed derivative
            gaussian1 = sitk.RecursiveGaussianImageFilter()
            gaussian1.SetSigma(scale)
            gaussian1.SetOrder(1)
            gaussian1.SetDirection(i)
            
            gaussian2 = sitk.RecursiveGaussianImageFilter()
            gaussian2.SetSigma(scale)
            gaussian2.SetOrder(1)
            gaussian2.SetDirection(j)
            
            deriv = gaussian2.Execute(gaussian1.Execute(image))
        
        deriv_filters.append(deriv)
        gc.collect()
    
    # Convert to numpy arrays at the end
    hessian = {
        'dxx': sitk.GetArrayFromImage(deriv_filters[0]),
        'dxy': sitk.GetArrayFromImage(deriv_filters[1]),
        'dxz': sitk.GetArrayFromImage(deriv_filters[2]),
        'dyy': sitk.GetArrayFromImage(deriv_filters[3]),
        'dyz': sitk.GetArrayFromImage(deriv_filters[4]),
        'dzz': sitk.GetArrayFromImage(deriv_filters[5])
    }
    
    return hessian

def calculate_eigenvalues(hessian, mask):
    """Calculate eigenvalues and eigenvectors of Hessian matrix within ROI"""
    from tqdm import tqdm
    import gc
    
    # Get dimensions
    shape = hessian['dxx'].shape
    eigenvalues = np.zeros((3,) + shape, dtype=np.float32)
    eigenvectors = np.zeros((3,) + shape + (3,), dtype=np.float32)
    
    # Get ROI indices
    roi_indices = np.where(mask > 0)
    
    # Process only ROI voxels
    total_voxels = len(roi_indices[0])
    with tqdm(total=total_voxels, desc="Computing eigenvalues", leave=False) as pbar:
        for i in range(total_voxels):
            z, y, x = roi_indices[0][i], roi_indices[1][i], roi_indices[2][i]
            
            # Construct Hessian matrix at this voxel
            H = np.array([
                [hessian['dxx'][z,y,x], hessian['dxy'][z,y,x], hessian['dxz'][z,y,x]],
                [hessian['dxy'][z,y,x], hessian['dyy'][z,y,x], hessian['dyz'][z,y,x]],
                [hessian['dxz'][z,y,x], hessian['dyz'][z,y,x], hessian['dzz'][z,y,x]]
            ])
            
            # Compute eigenvalues and eigenvectors
            w, v = np.linalg.eigh(H)
            
            # Sort by absolute eigenvalue
            idx = np.argsort(np.abs(w))
            eigenvalues[:,z,y,x] = w[idx]
            eigenvectors[:,z,y,x] = v[:,idx]
            
            pbar.update(1)
            
            # Garbage collection periodically
            if i % 1000 == 0:
                gc.collect()
    
    return eigenvalues, eigenvectors

def frangi_vesselness(eigenvalues, image_array):
    """Calculate Frangi vesselness measure"""
    # Extract sorted eigenvalues
    lambda1 = eigenvalues[0]  # Smallest magnitude
    lambda2 = eigenvalues[1]  # Medium magnitude
    lambda3 = eigenvalues[2]  # Largest magnitude
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Calculate vesselness measures
    Ra = np.abs(lambda2) / (np.abs(lambda3) + epsilon)  # plate vs line
    Rb = np.abs(lambda1) / (np.sqrt(np.abs(lambda2 * lambda3)) + epsilon)  # blob vs line
    S = np.sqrt(lambda2**2 + lambda3**2)  # structure strength (using only λ2 and λ3)
    
    # Frangi parameters
    alpha = 0.5  # Controls sensitivity to Ra
    beta = 0.5   # Controls sensitivity to Rb
    c = 70.0     # Controls sensitivity to S
    
    # Initialize vesselness
    vesselness = np.zeros_like(lambda1)
    
    # Apply vesselness conditions
    valid_intensity = image_array >= -750  # HU threshold
    valid_eigenvalues = (lambda2 < 0) & (lambda3 < 0)  # tube-like structure
    valid_voxels = valid_intensity & valid_eigenvalues
    
    # Calculate vesselness only for valid voxels
    vesselness[valid_voxels] = (
        (1 - np.exp(-(Ra[valid_voxels]**2)/(2*alpha**2))) *
        np.exp(-(Rb[valid_voxels]**2)/(2*beta**2)) *
        (1 - np.exp(-(S[valid_voxels]**2)/(2*c**2)))
    )
    
    return vesselness
