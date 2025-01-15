import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import os
import gc
from scipy import fftpack
from tqdm import tqdm

def calculate_vesselness(image_array, mask, scales, output_dir=None):
    """Calculate vesselness measure using Frangi's method with scale optimization.
    
    Args:
        image_array: Input image as numpy array
        mask: Binary mask of regions to process
        scales: Array of scales to use
        output_dir: Directory to save results
    """
    # Initialize results
    vesselness = np.zeros_like(image_array)
    sigma_max = np.zeros_like(image_array)
    vessel_direction = np.zeros((*image_array.shape, 3))
    
    # Initialize scale optimization records
    scale_records = {
        'metadata': {
            'initial_scales': scales.tolist(),
            'kernel_size': 5,
            'optimization_params': {
                'learning_rate': 0.1,
                'max_iterations': 10,
                'convergence_threshold': 1e-4
            }
        },
        'optimizations': [],
        'statistics': {
            'total_points_optimized': 0,
            'average_improvement': 0.0,
            'scale_distribution': {}
        }
    }
    
    # Process each scale
    for scale_idx, scale in enumerate(tqdm(scales, desc="Processing scales")):
        print(f"Processing scale {scale:.3f}mm")
        
        # Calculate Hessian and vesselness at this scale
        current_vesselness = np.zeros_like(image_array)
        
        # Process points within mask
        points = np.where(mask > 0)
        for i in tqdm(range(len(points[0])), desc=f"Scale {scale:.3f}mm", leave=False):
            z, y, x = points[0][i], points[1][i], points[2][i]
            
            # Calculate Hessian at this point
            hessian = calculate_hessian_at_point(image_array, scale, (z,y,x))
            eigenvalues = calculate_eigenvalues_at_point(hessian)
            response = frangi_vesselness_at_point(eigenvalues, image_array[z,y,x])
            response *= scale  # γ-normalization
            
            current_vesselness[z,y,x] = response
            
            # Update maximum response
            if response > vesselness[z,y,x]:
                vesselness[z,y,x] = response
                sigma_max[z,y,x] = scale
                
                # Calculate vessel direction (first eigenvector)
                H = np.array([
                    [hessian['dxx'], hessian['dxy'], hessian['dxz']],
                    [hessian['dxy'], hessian['dyy'], hessian['dyz']],
                    [hessian['dxz'], hessian['dyz'], hessian['dzz']]
                ])
                _, v = np.linalg.eigh(H)
                vessel_direction[z,y,x] = v[:,0]
        
        # Save intermediate scale response
        if output_dir:
            scale_response = sitk.GetImageFromArray(current_vesselness)
            scale_response.SetSpacing((0.6, 0.6, 0.6))
            sitk.WriteImage(
                scale_response,
                os.path.join(output_dir, f'V-{scale_idx+1:03d}-010.nrrd')
            )
        
        # For points with maximum response at this scale, optimize the scale
        update_mask = (sigma_max == scale) & (vesselness > 0)
        if np.any(update_mask):
            positions = np.where(update_mask)
            optimized_scales = []
            improvements = []
            
            for i in tqdm(range(len(positions[0])), desc="Optimizing scales", leave=False):
                z, y, x = positions[0][i], positions[1][i], positions[2][i]
                
                # Optimize scale at this point
                optimal_scale, trajectory = optimize_scale(image_array, scale, (z,y,x))
                optimized_scales.append(optimal_scale)
                improvements.append(optimal_scale - scale)
                
                # Record optimization details
                scale_records['optimizations'].append({
                    'position': [int(z), int(y), int(x)],
                    'initial_scale': float(scale),
                    'optimal_scale': float(optimal_scale),
                    'improvement': float(optimal_scale - scale),
                    'trajectory': trajectory,
                    'initial_response': float(vesselness[z,y,x]),
                    'final_response': None  # Will be updated after recalculation
                })
                
                # Recalculate vesselness at optimal scale
                hessian_opt = calculate_hessian_at_point(image_array, optimal_scale, (z,y,x))
                eigenvalues_opt = calculate_eigenvalues_at_point(hessian_opt)
                vesselness_opt = frangi_vesselness_at_point(eigenvalues_opt, image_array[z,y,x])
                vesselness_opt *= optimal_scale  # γ-normalization
                
                # Update the last optimization record with final response
                scale_records['optimizations'][-1]['final_response'] = float(vesselness_opt)
                
                # Update if optimized response is better
                if vesselness_opt > vesselness[z,y,x]:
                    vesselness[z,y,x] = vesselness_opt
                    sigma_max[z,y,x] = optimal_scale
                    
                    # Update vessel direction for optimal scale
                    H = np.array([
                        [hessian_opt['dxx'], hessian_opt['dxy'], hessian_opt['dxz']],
                        [hessian_opt['dxy'], hessian_opt['dyy'], hessian_opt['dyz']],
                        [hessian_opt['dxz'], hessian_opt['dyz'], hessian_opt['dzz']]
                    ])
                    _, v = np.linalg.eigh(H)
                    vessel_direction[z,y,x] = v[:,0]
            
            # Update statistics
            scale_records['statistics']['total_points_optimized'] += len(positions[0])
            scale_records['statistics']['average_improvement'] = np.mean(improvements)
            scale_records['statistics']['scale_distribution'][f'scale_{scale:.3f}'] = {
                'points_optimized': len(positions[0]),
                'average_improvement': float(np.mean(improvements)),
                'std_improvement': float(np.std(improvements)),
                'min_optimal_scale': float(np.min(optimized_scales)),
                'max_optimal_scale': float(np.max(optimized_scales)),
                'mean_optimal_scale': float(np.mean(optimized_scales))
            }
    
    # Save scale optimization records
    if output_dir:
        import json
        with open(os.path.join(output_dir, 'scale_optimization.json'), 'w') as f:
            json.dump(scale_records, f, indent=2)
    
    return {
        'vesselness': vesselness,
        'sigma_max': sigma_max,
        'vessel_direction': vessel_direction
    }

def calculate_hessian(image_array, scale):
    """Calculate Hessian matrix components using discrete Gaussian kernel
    
    Args:
        image_array: Input image array
        scale: Scale (sigma) for Gaussian kernel
    """
    from tqdm import tqdm
    import gc
    
    # Create discrete Gaussian kernel with size 5
    kernel_size = 5
    x = np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    y = np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    z = np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Create Gaussian kernel
    gaussian = np.exp(-(X**2 + Y**2 + Z**2)/(2*scale**2))
    gaussian = gaussian / gaussian.sum()  # Normalize
    
    # Create first and second derivative kernels
    dg_dx = -X/(scale**2) * gaussian
    dg_dy = -Y/(scale**2) * gaussian
    dg_dz = -Z/(scale**2) * gaussian
    
    dg_dxx = (X**2/(scale**4) - 1/(scale**2)) * gaussian
    dg_dyy = (Y**2/(scale**4) - 1/(scale**2)) * gaussian
    dg_dzz = (Z**2/(scale**4) - 1/(scale**2)) * gaussian
    
    dg_dxy = (X*Y/(scale**4)) * gaussian
    dg_dxz = (X*Z/(scale**4)) * gaussian
    dg_dyz = (Y*Z/(scale**4)) * gaussian
    
    # Initialize Hessian components
    hessian = {}
    
    # Create progress bar for derivative calculations
    derivative_pairs = [
        ('dxx', dg_dxx), ('dxy', dg_dxy), ('dxz', dg_dxz),
        ('dyy', dg_dyy), ('dyz', dg_dyz), ('dzz', dg_dzz)
    ]
    
    from scipy.ndimage import convolve
    
    for name, kernel in tqdm(derivative_pairs, desc=f"Computing Hessian at scale {scale:.2f}", leave=False):
        # Convolve with appropriate kernel
        hessian[name] = convolve(image_array, kernel, mode='reflect')
        gc.collect()
    
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

def optimize_scale(image_array, initial_scale, position, learning_rate=0.1, max_iter=10):
    """Optimize scale using gradient descent for maximum vesselness response
    
    Args:
        image_array: Input image array
        initial_scale: Initial scale value
        position: (z,y,x) position to optimize scale at
        learning_rate: Learning rate for gradient descent
        max_iter: Maximum iterations
    """
    scale = initial_scale
    z, y, x = position
    
    # Record optimization trajectory with more details
    trajectory = [{
        'iteration': 0,
        'scale': float(scale),
        'vesselness': None,
        'gradient': None
    }]
    
    for iter_num in range(max_iter):
        # Calculate vesselness at current scale and neighboring scales
        delta = 0.1 * scale  # Small delta for gradient estimation
        
        # Calculate Hessian and vesselness at three scales
        hessian_minus = calculate_hessian_at_point(image_array, scale - delta, position)
        hessian = calculate_hessian_at_point(image_array, scale, position)
        hessian_plus = calculate_hessian_at_point(image_array, scale + delta, position)
        
        # Calculate eigenvalues and vesselness
        eig_minus = calculate_eigenvalues_at_point(hessian_minus)
        eig = calculate_eigenvalues_at_point(hessian)
        eig_plus = calculate_eigenvalues_at_point(hessian_plus)
        
        v_minus = frangi_vesselness_at_point(eig_minus, image_array[z,y,x]) * (scale - delta)
        v = frangi_vesselness_at_point(eig, image_array[z,y,x]) * scale
        v_plus = frangi_vesselness_at_point(eig_plus, image_array[z,y,x]) * (scale + delta)
        
        # Estimate gradient using central difference
        gradient = (v_plus - v_minus) / (2 * delta)
        
        # Update scale using gradient ascent (since we want maximum response)
        new_scale = scale + learning_rate * gradient
        
        # Record this iteration with more details
        trajectory.append({
            'iteration': iter_num + 1,
            'scale': float(new_scale),
            'vesselness': float(v),
            'gradient': float(gradient)
        })
        
        # Check convergence
        if abs(new_scale - scale) < 1e-4:
            break
            
        scale = new_scale
        
        # Keep scale within reasonable bounds
        scale = np.clip(scale, 0.6, 6.0)
    
    return scale, trajectory

def calculate_hessian_at_point(image_array, scale, position):
    """Calculate Hessian matrix at a single point"""
    z, y, x = position
    
    # Create discrete Gaussian kernel with size 5
    kernel_size = 5
    kx = np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    ky = np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    kz = np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    X, Y, Z = np.meshgrid(kx, ky, kz)
    
    # Create Gaussian kernel
    gaussian = np.exp(-(X**2 + Y**2 + Z**2)/(2*scale**2))
    gaussian = gaussian / gaussian.sum()
    
    # Create derivative kernels
    dg_dxx = (X**2/(scale**4) - 1/(scale**2)) * gaussian
    dg_dyy = (Y**2/(scale**4) - 1/(scale**2)) * gaussian
    dg_dzz = (Z**2/(scale**4) - 1/(scale**2)) * gaussian
    dg_dxy = (X*Y/(scale**4)) * gaussian
    dg_dxz = (X*Z/(scale**4)) * gaussian
    dg_dyz = (Y*Z/(scale**4)) * gaussian
    
    # Extract local patch
    pad = kernel_size//2
    patch = image_array[z-pad:z+pad+1, y-pad:y+pad+1, x-pad:x+pad+1]
    
    # Calculate Hessian components
    hessian = {}
    hessian['dxx'] = np.sum(patch * dg_dxx)
    hessian['dyy'] = np.sum(patch * dg_dyy)
    hessian['dzz'] = np.sum(patch * dg_dzz)
    hessian['dxy'] = np.sum(patch * dg_dxy)
    hessian['dxz'] = np.sum(patch * dg_dxz)
    hessian['dyz'] = np.sum(patch * dg_dyz)
    
    return hessian

def calculate_eigenvalues_at_point(hessian):
    """Calculate eigenvalues of Hessian matrix at a point"""
    H = np.array([
        [hessian['dxx'], hessian['dxy'], hessian['dxz']],
        [hessian['dxy'], hessian['dyy'], hessian['dyz']],
        [hessian['dxz'], hessian['dyz'], hessian['dzz']]
    ])
    
    # Compute eigenvalues
    w = np.linalg.eigvalsh(H)
    
    # Sort by absolute value
    idx = np.argsort(np.abs(w))
    return w[idx]

def frangi_vesselness_at_point(eigenvalues, intensity):
    """Calculate Frangi vesselness at a point"""
    lambda1, lambda2, lambda3 = eigenvalues
    
    # Skip if intensity is too low (air)
    if intensity < -750:
        return 0
    
    # Skip if not tube-like
    if lambda2 >= 0 or lambda3 >= 0:
        return 0
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Calculate vesselness measures
    Ra = abs(lambda2) / (abs(lambda3) + epsilon)
    Rb = abs(lambda1) / (np.sqrt(abs(lambda2 * lambda3)) + epsilon)
    S = np.sqrt(lambda2**2 + lambda3**2)
    
    # Frangi parameters
    alpha = 0.5
    beta = 0.5
    c = 70.0
    
    # Calculate vesselness
    vesselness = (
        (1 - np.exp(-(Ra**2)/(2*alpha**2))) *
        np.exp(-(Rb**2)/(2*beta**2)) *
        (1 - np.exp(-(S**2)/(2*c**2)))
    )
    
    return vesselness

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
