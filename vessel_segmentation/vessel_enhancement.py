import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import os
import gc
from scipy import fftpack
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import linalg
import seaborn as sns

def calculate_vesselness(image_array, mask, scales, output_dir=None, project_name=None):
    """Calculate vesselness measure using Frangi's method with scale optimization.
    
    Args:
        image_array: Input image as numpy array
        mask: Binary mask of regions to process
        scales: Array of scales to use
        output_dir: Directory to save results
        project_name: Name of the project for file naming
    """
    # Initialize results
    vesselness = np.zeros_like(image_array)
    sigma_max = np.zeros_like(image_array)
    vessel_direction = np.zeros((*image_array.shape, 3))
    
    # Initialize scale optimization records with minimal data
    scale_records = {
        'metadata': {
            'initial_scales': scales.tolist(),
            'optimization_params': {
                'base_lr': 0.1,
                'momentum': 0.9,
                'max_iterations': 10,
                'convergence_threshold': 1e-4
            }
        },
        'scale_statistics': {  # Track statistics per scale
            'total_points': {},  # Number of points optimized at each scale
            'average_improvement': {},  # Average scale improvement at each scale
            'average_iterations': {},  # Average iterations needed at each scale
            'final_scale_distribution': {},  # Distribution of final scales
            'convergence_stats': {  # Per-scale convergence statistics
                'min_iterations': {},
                'max_iterations': {},
                'avg_iterations': {},
                'avg_improvement': {},
                'std_improvement': {}
            }
        },
        'example_points': {}  # Store 2-3 representative points per scale
    }
    
    # Process each scale
    for scale_idx, scale in enumerate(tqdm(scales, desc="Processing scales")):
        print(f"Processing scale {scale:.3f}mm")
        
        # Initialize per-scale statistics
        scale_key = f'scale_{scale:.3f}'
        scale_records['scale_statistics']['total_points'][scale_key] = 0
        scale_records['scale_statistics']['average_improvement'][scale_key] = 0.0
        scale_records['scale_statistics']['average_iterations'][scale_key] = 0.0
        scale_records['example_points'][scale_key] = []
        
        # Calculate Hessian and vesselness at this scale
        current_vesselness = np.zeros_like(image_array)
        
        # Process points within mask
        points = np.where(mask > 0)
        
        # Track statistics for this scale
        total_points = 0
        improvements = []
        iterations_counts = []
        
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
            iterations_counts = []
            
            for i in tqdm(range(len(positions[0])), desc="Optimizing scales", leave=False):
                z, y, x = positions[0][i], positions[1][i], positions[2][i]
                
                # Optimize scale at this point
                optimal_scale, trajectory = optimize_scale(image_array, scale, (z,y,x))
                optimized_scales.append(optimal_scale)
                improvements.append(optimal_scale - scale)
                iterations_counts.append(len(trajectory))
                
                # Store example points (only first 2-3 points per scale)
                if len(scale_records['example_points'][scale_key]) < 3:
                    example_point = {
                        'position': [int(z), int(y), int(x)],
                        'initial_scale': float(scale),
                        'optimal_scale': float(optimal_scale),
                        'improvement': float(optimal_scale - scale),
                        'iterations': len(trajectory),
                        'trajectory': trajectory
                    }
                    scale_records['example_points'][scale_key].append(example_point)
                
                # Update if optimized response is better
                hessian_opt = calculate_hessian_at_point(image_array, optimal_scale, (z,y,x))
                eigenvalues_opt = calculate_eigenvalues_at_point(hessian_opt)
                vesselness_opt = frangi_vesselness_at_point(eigenvalues_opt, image_array[z,y,x])
                vesselness_opt *= optimal_scale  # γ-normalization
                
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
            
            # Update scale statistics
            scale_records['scale_statistics']['total_points'][scale_key] = len(positions[0])
            scale_records['scale_statistics']['average_improvement'][scale_key] = float(np.mean(improvements))
            scale_records['scale_statistics']['average_iterations'][scale_key] = float(np.mean(iterations_counts))
            
            # Update convergence statistics
            scale_records['scale_statistics']['convergence_stats']['min_iterations'][scale_key] = int(np.min(iterations_counts))
            scale_records['scale_statistics']['convergence_stats']['max_iterations'][scale_key] = int(np.max(iterations_counts))
            scale_records['scale_statistics']['convergence_stats']['avg_iterations'][scale_key] = float(np.mean(iterations_counts))
            scale_records['scale_statistics']['convergence_stats']['avg_improvement'][scale_key] = float(np.mean(improvements))
            scale_records['scale_statistics']['convergence_stats']['std_improvement'][scale_key] = float(np.std(improvements))
            
            # Update final scale distribution
            unique_scales, counts = np.unique(optimized_scales, return_counts=True)
            scale_records['scale_statistics']['final_scale_distribution'][scale_key] = {
                f'{s:.3f}': int(c) for s, c in zip(unique_scales, counts)
            }
    
    # Save scale optimization records
    if output_dir and project_name:
        import json
        
        # Save optimization records
        optimization_file = os.path.join(output_dir, f'{project_name}_scale_optimization.json')
        with open(optimization_file, 'w') as f:
            json.dump(scale_records, f, indent=2)
        
        # Create and save plots
        plot_convergence_data(scale_records, output_dir, project_name)
    
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
    import SimpleITK as sitk
    import numpy as np
    from tqdm import tqdm
    import gc
    
    # Initialize Hessian components dictionary
    hessian = {}
    
    # Pre-create Gaussian filters for orders 1 and 2
    gaussian1 = sitk.RecursiveGaussianImageFilter()
    gaussian1.SetSigma(scale)
    gaussian1.SetOrder(1)
    
    gaussian2 = sitk.RecursiveGaussianImageFilter()
    gaussian2.SetSigma(scale)
    gaussian2.SetOrder(2)
    
    # Convert numpy array to SimpleITK image
    image = sitk.GetImageFromArray(image_array)
    
    # Process second derivatives first (diagonal elements)
    for i, name in enumerate([('dxx', 0), ('dyy', 1), ('dzz', 2)]):
        gaussian2.SetDirection(name[1])
        result = gaussian2.Execute(image)
        hessian[name[0]] = sitk.GetArrayFromImage(result)
        gc.collect()
    
    # Process mixed derivatives
    for i, j, name in [((0,1), 'dxy'), ((0,2), 'dxz'), ((1,2), 'dyz')]:
        gaussian1.SetDirection(i)
        temp = gaussian1.Execute(image)
        gaussian1.SetDirection(j)
        result = gaussian1.Execute(temp)
        hessian[name] = sitk.GetArrayFromImage(result)
        del temp
        gc.collect()
    
    return hessian

def calculate_eigenvalues(hessian, mask):
    """Calculate eigenvalues and eigenvectors of Hessian matrix within ROI"""
    import numpy as np
    from scipy import linalg
    from tqdm import tqdm
    import gc
    
    # Get dimensions
    shape = hessian['dxx'].shape
    eigenvalues = np.zeros((3,) + shape, dtype=np.float32)
    eigenvectors = np.zeros((3,) + shape + (3,), dtype=np.float32)
    
    # Get ROI indices
    roi_indices = np.where(mask > 0)
    total_voxels = len(roi_indices[0])
    
    # Process ROI voxels in batches
    batch_size = 10000
    for start_idx in tqdm(range(0, total_voxels, batch_size), desc="Computing eigenvalues", leave=False):
        end_idx = min(start_idx + batch_size, total_voxels)
        batch_size_actual = end_idx - start_idx
        
        # Get batch indices
        z = roi_indices[0][start_idx:end_idx]
        y = roi_indices[1][start_idx:end_idx]
        x = roi_indices[2][start_idx:end_idx]
        
        # Construct batch of Hessian matrices (batch_size x 3 x 3)
        batch_H = np.zeros((batch_size_actual, 3, 3))
        batch_H[:,0,0] = hessian['dxx'][z,y,x]
        batch_H[:,0,1] = batch_H[:,1,0] = hessian['dxy'][z,y,x]
        batch_H[:,0,2] = batch_H[:,2,0] = hessian['dxz'][z,y,x]
        batch_H[:,1,1] = hessian['dyy'][z,y,x]
        batch_H[:,1,2] = batch_H[:,2,1] = hessian['dyz'][z,y,x]
        batch_H[:,2,2] = hessian['dzz'][z,y,x]
        
        # Compute eigenvalues and eigenvectors for the batch
        w_batch = np.zeros((batch_size_actual, 3))
        v_batch = np.zeros((batch_size_actual, 3, 3))
        
        for i in range(batch_size_actual):
            try:
                w, v = linalg.eigh(batch_H[i])
                # Sort by absolute value
                idx = np.argsort(np.abs(w))
                w_batch[i] = w[idx]
                v_batch[i] = v[:,idx]
            except Exception as e:
                print(f"Error computing eigenvalues for matrix:")
                print(batch_H[i])
                raise e
        
        # Store results
        eigenvalues[:,z,y,x] = w_batch.T
        eigenvectors[:,z,y,x] = v_batch.transpose(1,0,2)
        
        # Clean up
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

def optimize_scale(image_array, initial_scale, position, base_lr=0.1, momentum=0.9, max_iter=10):
    """Optimize scale using gradient descent with momentum and adaptive learning rate
    
    Args:
        image_array: Input image array
        initial_scale: Initial scale value
        position: (z,y,x) position to optimize scale at
        base_lr: Base learning rate (will be adapted)
        momentum: Momentum coefficient
        max_iter: Maximum iterations
    """
    scale = initial_scale
    z, y, x = position
    
    # Initialize momentum and adaptive learning rate
    velocity = 0.0
    prev_gradient = None
    adaptive_lr = base_lr
    
    # Calculate initial vesselness
    hessian = calculate_hessian_at_point(image_array, scale, position)
    eig = calculate_eigenvalues_at_point(hessian)
    v = frangi_vesselness_at_point(eig, image_array[z,y,x]) * scale
    
    trajectory = [{
        'iteration': 0,
        'scale': float(scale),
        'vesselness': float(v),
        'gradient': None,
        'learning_rate': float(adaptive_lr)
    }]
    
    prev_vesselness = v
    
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
        
        # Adapt learning rate based on gradient behavior
        if prev_gradient is not None:
            # If gradient direction changed, reduce learning rate
            if gradient * prev_gradient < 0:
                adaptive_lr *= 0.5
            # If gradient direction same, slowly increase learning rate
            else:
                adaptive_lr *= 1.1
            
            # Bound learning rate
            adaptive_lr = np.clip(adaptive_lr, 0.01 * base_lr, 10 * base_lr)
        
        # Update velocity (momentum)
        velocity = momentum * velocity + adaptive_lr * gradient
        
        # Update scale using momentum
        new_scale = scale + velocity
        
        # Record this iteration
        trajectory.append({
            'iteration': iter_num + 1,
            'scale': float(new_scale),
            'vesselness': float(v),
            'gradient': float(gradient),
            'learning_rate': float(adaptive_lr),
            'velocity': float(velocity)
        })
        
        # Check convergence based on both scale and vesselness change
        scale_change = abs(new_scale - scale)
        vesselness_change = abs(v - prev_vesselness) if prev_vesselness is not None else float('inf')
        
        if scale_change < 1e-4 and vesselness_change < 1e-6:
            break
            
        # Update for next iteration
        scale = new_scale
        prev_gradient = gradient
        prev_vesselness = v
        
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

def plot_convergence_data(scale_records, output_dir, project_name):
    """Create visualization plots for scale optimization data.
    
    Args:
        scale_records: Dictionary containing scale optimization records
        output_dir: Directory to save plots
        project_name: Name of the project for file naming
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'optimization_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract scales and prepare data
    scales = [float(scale.split('_')[1]) for scale in scale_records['scale_statistics']['total_points'].keys()]
    scales.sort()
    
    # 1. Plot scale statistics overview
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    
    # Plot total points and average iterations
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    total_points = [scale_records['scale_statistics']['total_points'][f'scale_{s:.3f}'] for s in scales]
    avg_iterations = [scale_records['scale_statistics']['average_iterations'][f'scale_{s:.3f}'] for s in scales]
    
    ax1.plot(scales, total_points, 'b-', label='Total Points')
    ax2.plot(scales, avg_iterations, 'r--', label='Avg Iterations')
    
    ax1.set_xlabel('Initial Scale (mm)')
    ax1.set_ylabel('Number of Points', color='b')
    ax2.set_ylabel('Average Iterations', color='r')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Scale Optimization Overview')
    
    # 2. Plot improvements distribution
    plt.subplot(2, 1, 2)
    improvements = [scale_records['scale_statistics']['average_improvement'][f'scale_{s:.3f}'] for s in scales]
    std_improvements = [scale_records['scale_statistics']['convergence_stats']['std_improvement'][f'scale_{s:.3f}'] for s in scales]
    
    plt.errorbar(scales, improvements, yerr=std_improvements, fmt='o-', capsize=5)
    plt.xlabel('Initial Scale (mm)')
    plt.ylabel('Scale Improvement (mm)')
    plt.title('Average Scale Improvements with Standard Deviation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{project_name}_scale_statistics.png'))
    plt.close()
    
    # 3. Plot example trajectories for each scale
    for scale in scales:
        scale_key = f'scale_{scale:.3f}'
        example_points = scale_records['example_points'][scale_key]
        
        if not example_points:
            continue
        
        plt.figure(figsize=(15, 5))
        
        # Plot scale convergence
        plt.subplot(1, 2, 1)
        for point in example_points:
            trajectory = point['trajectory']
            scales_t = [t['scale'] for t in trajectory]
            plt.plot(range(len(scales_t)), scales_t, '-o', label=f'Point {point["position"]}')
        
        plt.xlabel('Iteration')
        plt.ylabel('Scale (mm)')
        plt.title(f'Scale Convergence (Initial Scale: {scale:.3f}mm)')
        plt.legend()
        
        # Plot vesselness improvement
        plt.subplot(1, 2, 2)
        for point in example_points:
            trajectory = point['trajectory']
            vesselness_t = [t['vesselness'] for t in trajectory]
            plt.plot(range(len(vesselness_t)), vesselness_t, '-o', label=f'Point {point["position"]}')
        
        plt.xlabel('Iteration')
        plt.ylabel('Vesselness')
        plt.title('Vesselness Improvement')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{project_name}_trajectories_scale_{scale:.3f}.png'))
        plt.close()
    
    # 4. Plot final scale distribution heatmap
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    final_scales = set()
    for scale_key in scale_records['scale_statistics']['final_scale_distribution']:
        final_scales.update(float(s) for s in scale_records['scale_statistics']['final_scale_distribution'][scale_key].keys())
    
    final_scales = sorted(list(final_scales))
    initial_scales = scales
    
    heatmap_data = np.zeros((len(initial_scales), len(final_scales)))
    
    for i, init_scale in enumerate(initial_scales):
        scale_key = f'scale_{init_scale:.3f}'
        dist = scale_records['scale_statistics']['final_scale_distribution'][scale_key]
        
        for final_scale_str, count in dist.items():
            j = final_scales.index(float(final_scale_str))
            heatmap_data[i, j] = count
    
    # Normalize rows to show distribution
    row_sums = heatmap_data.sum(axis=1, keepdims=True)
    heatmap_data = np.where(row_sums > 0, heatmap_data / row_sums, 0)
    
    sns.heatmap(heatmap_data, 
                xticklabels=[f'{s:.3f}' for s in final_scales],
                yticklabels=[f'{s:.3f}' for s in initial_scales],
                cmap='viridis',
                fmt='.2f')
    
    plt.xlabel('Final Scale (mm)')
    plt.ylabel('Initial Scale (mm)')
    plt.title('Scale Optimization Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{project_name}_scale_distribution.png'))
    plt.close()
