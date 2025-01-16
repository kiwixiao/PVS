import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import os
import gc
from scipy import linalg, fftpack

def deconvolve_image(image_array, kernel_size=5, sigma=0.6):
    """Deconvolve image using Wiener deconvolution
    
    Args:
        image_array: Input image array
        kernel_size: Size of Gaussian kernel (default: 5)
        sigma: Sigma of Gaussian kernel (default: 0.6)
        
    Returns:
        Deconvolved image array
    """
    # Create Gaussian PSF
    x = np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    y = np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    z = np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    X, Y, Z = np.meshgrid(x, y, z)
    psf = np.exp(-(X**2 + Y**2 + Z**2)/(2*sigma**2))
    psf = psf / psf.sum()  # Normalize
    
    # Compute FFT of image and PSF
    image_fft = fftpack.fftn(image_array)
    psf_fft = fftpack.fftn(psf, image_array.shape)
    
    # Wiener deconvolution with regularization
    reg_param = 0.1  # Regularization parameter
    deconv_fft = image_fft * np.conj(psf_fft) / (np.abs(psf_fft)**2 + reg_param)
    
    # Inverse FFT
    deconv = np.real(fftpack.ifftn(deconv_fft))
    
    return deconv

def calculate_vesselness(image_array, mask, scales, output_dir=None):
    """Calculate vesselness measure using multi-scale Hessian analysis
    
    Args:
        image_array: Input image array (should be deconvolved)
        mask: Binary mask (eroded mask)
        scales: List of scales for Hessian calculation
        output_dir: Directory to save intermediate results
        
    Returns:
        vesselness: Original vesselness measure
        sigma_max: Scale of maximum response
        vessel_direction: Vessel direction vectors
    """
    # Convert mask to uint8 (this should be the eroded mask)
    mask = mask.astype(np.uint8)
    
    # Always try to load pre-computed results first if output directory is provided
    if output_dir:
        required_files = [
            'vesselness.nrrd',
            'sigma_max.nrrd',
            'vessel_direction.nrrd'
        ]
        
        try:
            # Check if all required files exist
            missing_files = [f for f in required_files 
                           if not os.path.exists(os.path.join(output_dir, f))]
            
            if not missing_files:
                print("Loading pre-computed vesselness results...")
                vesselness = sitk.GetArrayFromImage(sitk.ReadImage(
                    os.path.join(output_dir, 'vesselness.nrrd')))
                sigma_max = sitk.GetArrayFromImage(sitk.ReadImage(
                    os.path.join(output_dir, 'sigma_max.nrrd')))
                vessel_direction = sitk.GetArrayFromImage(sitk.ReadImage(
                    os.path.join(output_dir, 'vessel_direction.nrrd')))
                print("Successfully loaded all pre-computed results")
                return vesselness, sigma_max, vessel_direction
            else:
                print(f"Missing required files: {missing_files}")
                print("Computing vesselness from scratch...")
                
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
        eigenvalues, eigenvectors = calculate_eigenvalues(hessian, mask)
        
        # Calculate vesselness for this scale
        current_vesselness = frangi_vesselness(eigenvalues, image_array)
        current_vesselness *= scale  # γ-normalization
        
        # Get vessel direction
        current_direction = eigenvectors[0]
        
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
    """Calculate Hessian matrix components using SimpleITK's recursive Gaussian filter with optimized memory usage"""
    from tqdm import tqdm
    import gc
    import SimpleITK as sitk
    import numpy as np
    
    # Convert numpy array to SimpleITK image
    image = sitk.GetImageFromArray(image_array)
    
    # Initialize Hessian components dictionary
    hessian = {}
    component_names = ['dxx', 'dxy', 'dxz', 'dyy', 'dyz', 'dzz']
    derivative_pairs = [(i,j) for i in range(3) for j in range(i, 3)]
    
    # Pre-create Gaussian filters to avoid recreation
    gaussian_filters = {}
    for order in [1, 2]:
        gaussian_filters[order] = sitk.RecursiveGaussianImageFilter()
        gaussian_filters[order].SetSigma(scale)
        gaussian_filters[order].SetOrder(order)
    
    for (i, j), component in tqdm(zip(derivative_pairs, component_names), 
                                desc=f"Computing Hessian at scale {scale:.2f}", 
                                total=len(derivative_pairs),
                                leave=False):
        if i == j:
            # Second derivative
            gaussian_filters[2].SetDirection(i)
            deriv = gaussian_filters[2].Execute(image)
        else:
            # Mixed derivative - compute sequentially to save memory
            gaussian_filters[1].SetDirection(i)
            temp = gaussian_filters[1].Execute(image)
            gaussian_filters[1].SetDirection(j)
            deriv = gaussian_filters[1].Execute(temp)
            del temp
            gc.collect()
        
        # Convert to numpy array and store
        hessian[component] = sitk.GetArrayFromImage(deriv)
        del deriv
        gc.collect()
    
    return hessian

def calculate_eigenvalues(hessian, mask):
    """Calculate eigenvalues and eigenvectors of Hessian matrix within ROI using vectorized operations"""
    from tqdm import tqdm
    import gc
    import numpy as np
    
    # Get dimensions
    shape = hessian['dxx'].shape
    eigenvalues = np.zeros((3,) + shape, dtype=np.float32)
    eigenvectors = np.zeros((3,) + shape + (3,), dtype=np.float32)
    
    # Get ROI indices
    roi_indices = np.where(mask > 0)
    total_voxels = len(roi_indices[0])
    print(f"Total voxels to process: {total_voxels}")
    
    # Process in batches to manage memory
    batch_size = 10000  # Adjust based on available memory
    for i in tqdm(range(0, total_voxels, batch_size), desc="Computing eigenvalues", leave=False):
        batch_end = min(i + batch_size, total_voxels)
        batch_size_current = batch_end - i
        
        # Get current batch indices
        batch_indices = (
            roi_indices[0][i:batch_end],
            roi_indices[1][i:batch_end],
            roi_indices[2][i:batch_end]
        )
        
        # Construct Hessian matrices for current batch
        batch_H = np.zeros((batch_size_current, 3, 3), dtype=np.float32)
        
        # Fill the symmetric matrix
        batch_H[:,0,0] = hessian['dxx'][batch_indices]
        batch_H[:,0,1] = batch_H[:,1,0] = hessian['dxy'][batch_indices]
        batch_H[:,0,2] = batch_H[:,2,0] = hessian['dxz'][batch_indices]
        batch_H[:,1,1] = hessian['dyy'][batch_indices]
        batch_H[:,1,2] = batch_H[:,2,1] = hessian['dyz'][batch_indices]
        batch_H[:,2,2] = hessian['dzz'][batch_indices]
        
        # Debug print
        if i == 0:
            print(f"Batch Hessian shape: {batch_H.shape}")
            print(f"First Hessian matrix:\n{batch_H[0]}")
            # Verify matrix is symmetric
            is_symmetric = np.allclose(batch_H[0], batch_H[0].T)
            print(f"Is symmetric: {is_symmetric}")
        
        # Verify each matrix is valid before eigendecomposition
        if not np.all(np.isfinite(batch_H)):
            raise ValueError("Found non-finite values in Hessian matrix")
        
        # Process each matrix individually to ensure proper shape
        w_batch = np.zeros((batch_size_current, 3), dtype=np.float32)
        v_batch = np.zeros((batch_size_current, 3, 3), dtype=np.float32)
        
        for j in range(batch_size_current):
            # Ensure the matrix is exactly (3,3) and symmetric
            H = np.array(batch_H[j], dtype=np.float32)
            H = (H + H.T) / 2  # Ensure perfect symmetry
            
            try:
                w, v = np.linalg.eigh(H)
                w_batch[j] = w
                v_batch[j] = v
            except np.linalg.LinAlgError as e:
                print(f"Error in matrix {j} of batch {i}:")
                print(f"Matrix:\n{H}")
                raise e
        
        # Sort by absolute eigenvalue
        abs_w_batch = np.abs(w_batch)
        sort_idx = np.argsort(abs_w_batch, axis=1)
        
        # Apply sorting to eigenvalues and eigenvectors
        for j in range(batch_size_current):
            idx = sort_idx[j]
            z, y, x = batch_indices[0][j], batch_indices[1][j], batch_indices[2][j]
            eigenvalues[:,z,y,x] = w_batch[j,idx]
            eigenvectors[:,z,y,x] = v_batch[j,:,idx]
        
        if i % (batch_size * 10) == 0:
            gc.collect()
    
    return eigenvalues, eigenvectors

def frangi_vesselness(eigenvalues, image_array):
    """Calculate Frangi vesselness measure using optimized vectorized operations"""
    # Extract sorted eigenvalues - use views instead of copies
    lambda1 = eigenvalues[0]  # Smallest magnitude
    lambda2 = eigenvalues[1]  # Medium magnitude
    lambda3 = eigenvalues[2]  # Largest magnitude
    
    # Frangi parameters (constants)
    ALPHA = 0.5  # Controls sensitivity to Ra
    BETA = 0.5   # Controls sensitivity to Rb
    C = 70.0     # Controls sensitivity to S
    EPSILON = 1e-10  # Small value to avoid division by zero
    HU_THRESHOLD = -750  # HU threshold for valid intensity
    
    # Pre-compute valid voxels mask to avoid redundant computations
    valid_voxels = (image_array >= HU_THRESHOLD) & (lambda2 < 0) & (lambda3 < 0)
    
    # Only allocate arrays for valid voxels to save memory
    if not np.any(valid_voxels):
        return np.zeros_like(lambda1)
    
    # Get valid indices to reduce computation
    valid_indices = np.where(valid_voxels)
    
    # Pre-compute terms for valid voxels only
    abs_lambda2 = np.abs(lambda2[valid_indices])
    abs_lambda3 = np.abs(lambda3[valid_indices])
    
    # Calculate Ra (plate vs line)
    Ra = abs_lambda2 / (abs_lambda3 + EPSILON)
    
    # Calculate Rb (blob vs line)
    abs_lambda1 = np.abs(lambda1[valid_indices])
    Rb = abs_lambda1 / (np.sqrt(abs_lambda2 * abs_lambda3) + EPSILON)
    
    # Calculate S (structure strength)
    S = np.sqrt(lambda2[valid_indices]**2 + lambda3[valid_indices]**2)
    
    # Calculate exponential terms
    exp_Ra = np.exp(-Ra**2 / (2 * ALPHA**2))
    exp_Rb = np.exp(-Rb**2 / (2 * BETA**2))
    exp_S = np.exp(-S**2 / (2 * C**2))
    
    # Initialize output array
    vesselness = np.zeros_like(lambda1)
    
    # Calculate vesselness only for valid voxels
    vesselness[valid_indices] = (1 - exp_Ra) * exp_Rb * (1 - exp_S)
    
    return vesselness

def filter_disconnected_components(binary_vessels: np.ndarray, min_component_size: int = 50, connectivity: int = 26) -> np.ndarray:
    """Filter out small disconnected components from binary vessel segmentation
    
    Args:
        binary_vessels: Binary vessel segmentation
        min_component_size: Minimum size (in voxels) for a component to be kept
        connectivity: Connectivity for labeling (6, 18, or 26)
        
    Returns:
        Filtered binary vessel segmentation
    """
    from scipy.ndimage import label
    
    # Label connected components
    labeled, num = label(binary_vessels, structure=np.ones((3,3,3)))
    print(f"Found {num} connected components")
    
    # Get component sizes
    unique, counts = np.unique(labeled, return_counts=True)
    sizes = dict(zip(unique[1:], counts[1:]))  # Skip background (label 0)
    
    # Print statistics
    print(f"Component size range: {min(sizes.values()):.1f} to {max(sizes.values()):.1f} voxels")
    small_components = sum(1 for size in sizes.values() if size < min_component_size)
    print(f"Number of components smaller than {min_component_size} voxels: {small_components}")
    
    # Create mask of components to keep
    keep_mask = np.zeros_like(labeled)
    for label_id, size in sizes.items():
        if size >= min_component_size:
            keep_mask[labeled == label_id] = 1
    
    # Calculate removed volume
    removed_voxels = np.sum(binary_vessels) - np.sum(keep_mask)
    removed_percentage = (removed_voxels / np.sum(binary_vessels)) * 100
    print(f"Removed {removed_voxels:.1f} voxels ({removed_percentage:.2f}% of original volume)")
    
    return keep_mask
