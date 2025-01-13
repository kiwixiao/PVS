"""
Vessel enhancement filter implementation.
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

class VesselEnhancementFilter:
    """Vessel enhancement filter using Frangi's method."""
    
    def __init__(
        self,
        scales: np.ndarray,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 5.0,
        hist_eq_alpha: float = 0.96,  # Histogram equalization parameter
        hist_eq_bins: int = 10000     # Number of bins for histogram equalization
    ):
        """Initialize filter.
        
        Args:
            scales: Scales for computing Hessian (sigma values)
            alpha: Frangi vesselness parameter
            beta: Frangi vesselness parameter
            gamma: Frangi vesselness parameter
            hist_eq_alpha: Alpha parameter for histogram equalization
            hist_eq_bins: Number of bins for histogram equalization
        """
        self.scales = scales
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hist_eq_alpha = hist_eq_alpha
        self.hist_eq_bins = hist_eq_bins
    
    def _preprocess_image(self, image: sitk.Image, mask: sitk.Image) -> sitk.Image:
        """Apply preprocessing steps to the input image.
        
        Args:
            image: Input CT image
            mask: ROI mask
            
        Returns:
            Preprocessed image
        """
        print("Preprocessing image...")
        # Convert both image and mask to float
        image_float = sitk.Cast(image, sitk.sitkFloat32)
        mask_float = sitk.Cast(mask, sitk.sitkFloat32)
        
        # Apply mask first
        masked_image = sitk.Multiply(image_float, mask_float)
        
        # Histogram equalization within mask
        histogram = sitk.GetArrayFromImage(masked_image)[sitk.GetArrayFromImage(mask) > 0]
        min_val = float(np.percentile(histogram, (1 - self.hist_eq_alpha) * 100))
        max_val = float(np.percentile(histogram, self.hist_eq_alpha * 100))
        
        # Rescale to [0, 1] within the ROI
        rescaler = sitk.IntensityWindowingImageFilter()
        rescaler.SetWindowMinimum(min_val)
        rescaler.SetWindowMaximum(max_val)
        rescaler.SetOutputMinimum(0.0)
        rescaler.SetOutputMaximum(1.0)
        normalized = rescaler.Execute(masked_image)
        
        return normalized
        
    def execute(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None,
        save_dir: Optional[str] = None,
        debug_mode: bool = False
    ) -> sitk.Image:
        """Apply vessel enhancement filter.
        
        Args:
            image: Input image
            mask: Optional mask
            save_dir: Optional directory to save results
            debug_mode: If True, save intermediate results
            
        Returns:
            Enhanced image
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save input image
            sitk.WriteImage(image, str(save_dir / "input.nii.gz"))
            if mask is not None:
                sitk.WriteImage(mask, str(save_dir / "mask.nii.gz"))
        
        # Preprocess image
        if mask is not None:
            preprocessed = self._preprocess_image(image, mask)
            if save_dir:
                sitk.WriteImage(preprocessed, str(save_dir / "preprocessed.nii.gz"))
        else:
            preprocessed = image
        
        # Initialize list to store responses at each scale
        scale_responses = []
        
        # Get mask array for ROI processing
        if mask is not None:
            mask_array = sitk.GetArrayFromImage(mask)
            roi_indices = np.where(mask_array > 0)
        else:
            roi_indices = None
        
        # Process each scale
        print("\nEnhancing vessels across scales...")
        max_response = None
        max_response_scale = np.zeros_like(sitk.GetArrayFromImage(image))  # Track scale giving max response
        
        # Create directories for intermediate results
        if save_dir:
            hessian_dir = save_dir / "hessian"
            eigenvalues_dir = save_dir / "eigenvalues"
            hessian_dir.mkdir(exist_ok=True)
            eigenvalues_dir.mkdir(exist_ok=True)
        
        for i, scale in enumerate(tqdm(self.scales, desc="Processing scales")):
            # Compute Hessian only within ROI
            hessian_images = self._compute_hessian(preprocessed, scale, roi_indices)
            
            # Save Hessian components
            if save_dir:
                scale_hessian_dir = hessian_dir / f"scale_{i:02d}_{scale:.2f}mm"
                scale_hessian_dir.mkdir(exist_ok=True)
                for j, hessian in enumerate(hessian_images):
                    sitk.WriteImage(
                        hessian,
                        str(scale_hessian_dir / f"hessian_{j}.nii.gz")
                    )
            
            # Compute eigenvalues
            eigenvalues = self._compute_eigenvalues(hessian_images, roi_indices)
            
            # Save eigenvalues
            if save_dir:
                scale_eigenvalues_dir = eigenvalues_dir / f"scale_{i:02d}_{scale:.2f}mm"
                scale_eigenvalues_dir.mkdir(exist_ok=True)
                for j, eig in enumerate(eigenvalues):
                    sitk.WriteImage(
                        eig,
                        str(scale_eigenvalues_dir / f"eigenvalue_{j}.nii.gz")
                    )
            
            # Compute vesselness
            response = self._compute_vesselness(eigenvalues)
            scale_responses.append(response)
            
            # Update max response and track scale
            response_array = sitk.GetArrayFromImage(response)
            if max_response is None:
                max_response = response_array
                max_response_scale[response_array > 0] = scale
            else:
                scale_mask = response_array > max_response
                max_response[scale_mask] = response_array[scale_mask]
                max_response_scale[scale_mask] = scale
            
            if save_dir:
                # Save scale response
                sitk.WriteImage(
                    response,
                    str(save_dir / f"scale_{i:02d}_{scale:.2f}mm_response.nii.gz")
                )
        
        # Save scale that gave maximum response
        if save_dir:
            max_scale_image = sitk.GetImageFromArray(max_response_scale)
            max_scale_image.CopyInformation(image)
            sitk.WriteImage(
                max_scale_image,
                str(save_dir / "max_response_scale.nii.gz")
            )
        
        # Convert max response back to SimpleITK image
        max_response_image = sitk.GetImageFromArray(max_response)
        max_response_image.CopyInformation(image)
        
        if save_dir:
            # Save max response across scales
            sitk.WriteImage(
                max_response_image,
                str(save_dir / "max_response.nii.gz")
            )
        
        # Apply histogram equalization to max response
        if mask is not None:
            # Only equalize within mask
            histogram = max_response[mask_array > 0]
            min_val = float(np.percentile(histogram, (1 - self.hist_eq_alpha) * 100))
            max_val = float(np.percentile(histogram, self.hist_eq_alpha * 100))
            
            # Rescale to [0, 1] within the ROI
            rescaler = sitk.IntensityWindowingImageFilter()
            rescaler.SetWindowMinimum(min_val)
            rescaler.SetWindowMaximum(max_val)
            rescaler.SetOutputMinimum(0.0)
            rescaler.SetOutputMaximum(1.0)
            enhanced = rescaler.Execute(max_response_image)
            
            # Create binary segmentation using Otsu thresholding
            otsu_filter = sitk.OtsuThresholdImageFilter()
            otsu_filter.SetInsideValue(1)
            otsu_filter.SetOutsideValue(0)
            binary_seg = otsu_filter.Execute(enhanced)
            
            if save_dir:
                # Save binary segmentation
                sitk.WriteImage(
                    binary_seg,
                    str(save_dir / "vessel_segmentation.nii.gz")
                )
                
                # Save final enhanced image
                sitk.WriteImage(
                    enhanced,
                    str(save_dir / "enhanced.nii.gz")
                )
            
            return enhanced
        else:
            return max_response_image
    
    def _compute_hessian(
        self,
        image: sitk.Image,
        scale: float,
        roi_indices: Optional[Tuple] = None
    ) -> List[sitk.Image]:
        """Compute Hessian matrix.
        
        Args:
            image: Input image
            scale: Scale for computing derivatives (sigma)
            roi_indices: Optional tuple of arrays with indices where mask > 0
            
        Returns:
            List of Hessian components [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        """
        # Initialize empty Hessian arrays
        if roi_indices is not None:
            shape = sitk.GetArrayFromImage(image).shape
            hessian_arrays = [np.zeros(shape) for _ in range(6)]
        
        # Create Gaussian derivative filters
        deriv_filters = []
        for i, j in tqdm([(i,j) for i in range(3) for j in range(i, 3)], 
                        desc=f"Computing Hessian at scale {scale:.2f}mm",
                        leave=False):
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
            
            if roi_indices is not None:
                # Only store values within ROI
                deriv_array = sitk.GetArrayFromImage(deriv)
                hessian_arrays[len(deriv_filters)][roi_indices] = deriv_array[roi_indices]
                # Create new image from array
                deriv = sitk.GetImageFromArray(hessian_arrays[len(deriv_filters)])
                deriv.CopyInformation(image)
            
            deriv_filters.append(deriv)
        
        return deriv_filters
    
    def _compute_eigenvalues(
        self,
        hessian_images: List[sitk.Image],
        roi_indices: Optional[Tuple] = None
    ) -> List[sitk.Image]:
        """Compute eigenvalues of the Hessian matrix.
        
        Args:
            hessian_images: List of Hessian matrix elements [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
            roi_indices: Optional tuple of arrays with indices where mask > 0
            
        Returns:
            List of eigenvalue images sorted by magnitude (|λ1| ≤ |λ2| ≤ |λ3|)
        """
        # Convert Hessian images to numpy arrays
        hessian_arrays = [sitk.GetArrayFromImage(img) for img in hessian_images]
        
        # Get image dimensions
        shape = hessian_arrays[0].shape
        eigenvalues = np.zeros((3,) + shape)
        
        # Only compute eigenvalues within ROI if specified
        if roi_indices is not None:
            z, y, x = roi_indices
            for i in tqdm(range(len(z)), desc="Computing eigenvalues", leave=False):
                # Construct Hessian matrix at this voxel
                H = np.array([
                    [hessian_arrays[0][z[i],y[i],x[i]], hessian_arrays[1][z[i],y[i],x[i]], hessian_arrays[2][z[i],y[i],x[i]]],
                    [hessian_arrays[1][z[i],y[i],x[i]], hessian_arrays[3][z[i],y[i],x[i]], hessian_arrays[4][z[i],y[i],x[i]]],
                    [hessian_arrays[2][z[i],y[i],x[i]], hessian_arrays[4][z[i],y[i],x[i]], hessian_arrays[5][z[i],y[i],x[i]]]
                ])
                
                # Compute eigenvalues
                w = np.linalg.eigvalsh(H)
                
                # Sort by absolute value
                idx = np.argsort(np.abs(w))
                eigenvalues[:,z[i],y[i],x[i]] = w[idx]
        else:
            total_voxels = shape[0] * shape[1] * shape[2]
            with tqdm(total=total_voxels, desc="Computing eigenvalues", leave=False) as pbar:
                for z in range(shape[0]):
                    for y in range(shape[1]):
                        for x in range(shape[2]):
                            H = np.array([
                                [hessian_arrays[0][z,y,x], hessian_arrays[1][z,y,x], hessian_arrays[2][z,y,x]],
                                [hessian_arrays[1][z,y,x], hessian_arrays[3][z,y,x], hessian_arrays[4][z,y,x]],
                                [hessian_arrays[2][z,y,x], hessian_arrays[4][z,y,x], hessian_arrays[5][z,y,x]]
                            ])
                            
                            w = np.linalg.eigvalsh(H)
                            idx = np.argsort(np.abs(w))
                            eigenvalues[:,z,y,x] = w[idx]
                            pbar.update(1)
        
        # Convert back to SimpleITK images
        result = []
        for i in range(3):
            img = sitk.GetImageFromArray(eigenvalues[i])
            img.CopyInformation(hessian_images[0])
            result.append(img)
        
        return result
    
    def _compute_vesselness(
        self,
        eigenvalues: List[sitk.Image]
    ) -> sitk.Image:
        """Compute vesselness from eigenvalues using Frangi's method.
        
        Args:
            eigenvalues: List of eigenvalue images sorted by magnitude (|λ1| ≤ |λ2| ≤ |λ3|)
            
        Returns:
            Vesselness image
        """
        # Convert to numpy arrays
        lambda1 = sitk.GetArrayFromImage(eigenvalues[0])
        lambda2 = sitk.GetArrayFromImage(eigenvalues[1])
        lambda3 = sitk.GetArrayFromImage(eigenvalues[2])
        
        # Initialize vesselness
        vesselness = np.zeros_like(lambda1)
        
        # Vessel criteria: λ1 ≈ 0, λ2 ≈ λ3 < 0
        vessel_mask = (lambda2 < 0) & (lambda3 < 0)
        
        # Only compute for valid voxels
        valid_mask = vessel_mask & (np.abs(lambda2 * lambda3) > 1e-10) & (np.abs(lambda3) > 1e-10)
        
        if np.any(valid_mask):
            # Compute ratios only for valid voxels
            RB = np.zeros_like(lambda1)
            RA = np.zeros_like(lambda1)
            
            # Safe computation of ratios
            RB[valid_mask] = np.abs(lambda1[valid_mask]) / np.sqrt(np.abs(lambda2[valid_mask] * lambda3[valid_mask]))
            RA[valid_mask] = np.abs(lambda2[valid_mask]) / np.abs(lambda3[valid_mask])
            
            # Compute structure strength
            S = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)
            
            # Compute vesselness only for valid voxels
            vesselness[valid_mask] = (
                (1 - np.exp(-RA[valid_mask]**2 / (2 * self.alpha**2))) *
                np.exp(-RB[valid_mask]**2 / (2 * self.beta**2)) *
                (1 - np.exp(-S[valid_mask]**2 / (2 * self.gamma**2)))
            )
        
        # Convert back to SimpleITK image
        result = sitk.GetImageFromArray(vesselness)
        result.CopyInformation(eigenvalues[0])
        
        return result 