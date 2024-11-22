# preprocessing/preprocessing.py
import numpy as np
import nibabel as nib
import os
from scipy import ndimage
from typing import Tuple, Dict
import logging
from pathlib import Path
import json
from skimage import morphology, filters
from skimage.morphology import convex_hull_image

class CTPreprocessor:
    """Preprocess CT images for vessel segmentation"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Default parameters (can be overridden by config)
        self.target_spacing = (0.6, 0.6, 0.6)  # Standard 0.6mm isotropic
        self.hu_window = {'min': -850, 'max': -150}
        self.edge_enhancement = {'sigma': 0.5, 'enhancement': 1.5}
        self.border_exclusion = {
            'distance_threshold': 5, # mm from border
            'smoothing_sigma': 1.0 # gaussian smoothing for distance map            
            }

    def set_parameters(self, config: Dict):
        """Set preprocessing parameters from config"""
        if config.get('target_spacing'):
            self.target_spacing = tuple(config['target_spacing'])
        if config.get('hu_window'):
            self.hu_window.update(config['hu_window'])
        if config.get('edge_enhancement'):
            self.edge_enhancement.update(config['edge_enhancement'])
        if config.get('border_exclusion'):
            self.border_exclusion.update(config['border_exclusion'])

    def create_border_exclusion_mask(self, mask_data: np.ndarray, voxel_size: float) -> np.ndarray:
        """Create mask exluding edges of lung lobes
        Args:
            mask_data: Binary lung mask
            voxel_size: Isotropic voxel size in mm
        Returns:
            Border exclusion mask (1 where vessels should be detected, 0 near borders)    
        """
        self.logger.info("Create border exclusion mask")
        # Create distance transform
        distance_map = ndimage.distance_transform_edt(mask_data, sampling=voxel_size)
        # Smooth distance map to avoid sharp transitions
        distance_map = filters.gaussian(
            distance_map,
            sigma = self.border_exclusion['smoothing_sigma'] / voxel_size
        )
        # Create exclusion mask
        threshold_voxels = self.border_exclusion['distance_threshold'] / voxel_size
        border_mask = distance_map > threshold_voxels
        # Ensure mask in binary
        border_mask = border_mask.astype(np.float32)
        
        return border_mask

    def preprocess_ct_for_vessels(self, ct_path: str, 
                                mask_path: str,
                                config: Dict = None) -> Tuple[np.ndarray, dict]:
        """
        Preprocess CT for vessel segmentation with preserved orientation
        """
        try:
            # Update parameters if config provided
            if config:
                self.set_parameters(config)
            
            self.logger.info("Starting CT preprocessing")
            
            # Get base filename
            basename = Path(ct_path).stem
            if basename.endswith('.nii'):
                basename = Path(basename).stem
            
            # 1. Load CT data and get original info
            ct_nifti = nib.load(ct_path)
            ct_data = ct_nifti.get_fdata()
            original_affine = ct_nifti.affine
            original_spacing = nib.affines.voxel_sizes(original_affine)
            
            # 2. Load mask
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
            convex_mask_output = self.output_dir / f"{basename}_convex_lung_mask.nii.gz"
            # this convex idea is open to explore
            convex_mask_nifti = create_convex_lung_mask(mask_path, convex_mask_output)
            
            mask_nifti = nib.load(mask_path)
            mask_data = mask_nifti.get_fdata()
            # Verify dimensions match
            if ct_data.shape != mask_data.shape:
                raise ValueError(f"CT and mask dimensions do not match: {ct_data.shape} vs {mask_data.shape}")
            
            # 3. Calculate spacing and prepare for resampling
            resize_factors = [o/t for o, t in zip(original_spacing, self.target_spacing)]
            
            # Create new affine with isotropic spacing but preserved orientation
            new_affine = self._create_isotropic_affine(original_affine)
            
            # 4. Resample CT and mask to isotropic spacing
            self.logger.info(f"Resampling to {self.target_spacing}mm isotropic")
            ct_resampled = ndimage.zoom(ct_data, resize_factors, order=3)  # cubic for CT
            mask_resampled = ndimage.zoom(mask_data, resize_factors, order=0)  # nearest for mask
            mask_resampled = (mask_resampled > 0.5).astype(np.float32)
            
            
            
            # 5. Apply preprocessing steps and save results
            preprocessed_data = self._process_and_save_results(
                ct_resampled, mask_resampled, new_affine, 
                ct_nifti.header, basename
            )
            
            # 6. Store metadata
            metadata = {
                'original_spacing': original_spacing.tolist(),
                'current_spacing': self.target_spacing,
                'hu_window': self.hu_window,
                'edge_enhancement': self.edge_enhancement,
                'original_shape': ct_data.shape,
                'preprocessed_shape': ct_resampled.shape,
                'original_path': ct_path,
                'original_affine': original_affine.tolist(),
                'output_files': preprocessed_data['output_files']
            }
            
            # Save metadata
            self._save_metadata(metadata, basename)
            
            return preprocessed_data['ct_masked'], metadata
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _create_isotropic_affine(self, original_affine: np.ndarray) -> np.ndarray:
        """Create new affine matrix for isotropic spacing"""
        direction_cosines = original_affine[:3, :3]
        norm_factors = np.sqrt(np.sum(direction_cosines ** 2, axis=0))
        normalized_cosines = direction_cosines / norm_factors
        
        new_affine = np.eye(4)
        for i in range(3):
            new_affine[:3, i] = normalized_cosines[:, i] * self.target_spacing[i]
        new_affine[:3, 3] = original_affine[:3, 3]
        
        return new_affine

    def _process_and_save_results(self, ct_data: np.ndarray, 
                                mask_data: np.ndarray,
                                affine: np.ndarray,
                                orig_header: nib.Nifti1Header,
                                basename: str) -> Dict:
        """Process CT data and save intermediate results"""
        outputs = {'output_files': {}}
        
        def save_nifti(data: np.ndarray, suffix: str) -> str:
            """Helper to save NIfTI files with proper orientation"""
            img = nib.Nifti1Image(data, affine)
            
            # Copy orientation information
            new_header = img.header
            new_header.set_qform(affine, code=int(orig_header['qform_code']))
            new_header.set_sform(affine, code=int(orig_header['sform_code']))
            
            # Copy relevant header fields
            for field in ['descrip', 'aux_file', 'intent_name', 'dim_info']:
                if field in orig_header:
                    new_header[field] = orig_header[field]
            
            output_path = self.output_dir / f"{basename}_{suffix}.nii.gz"
            nib.save(img, output_path)
            return str(output_path)
        
        # Save resampled isotropic CT
        outputs['output_files']['isotropic'] = save_nifti(ct_data, 'isotropic')
        
        # Apply edge enhancement
        self.logger.info("Applying edge enhancement")
        ct_blurred = ndimage.gaussian_filter(ct_data, self.edge_enhancement['sigma'])
        ct_enhanced = ct_data + self.edge_enhancement['enhancement'] * (ct_data - ct_blurred)
        outputs['output_files']['enhanced'] = save_nifti(ct_enhanced, 'enhanced')
        
        # Apply HU windowing
        self.logger.info("Applying HU windowing")
        ct_windowed = np.clip(ct_enhanced, self.hu_window['min'], self.hu_window['max'])
        outputs['output_files']['windowed'] = save_nifti(ct_windowed, 'windowed')
        
        # Create border exclusion mask
        border_mask = self.create_border_exclusion_mask(
            mask_data,
            self.target_spacing[0]  # Using isotropic spacing
        )
        outputs['output_files']['border_mask'] = save_nifti(border_mask, 'border_mask')
        
        # Normalize to [0,1] range and apply mask
        ct_normalized = (ct_windowed - self.hu_window['min']) / (self.hu_window['max'] - self.hu_window['min'])
        ct_masked = ct_normalized * mask_data * border_mask # Added border_mask here
        outputs['ct_masked'] = ct_masked
        outputs['output_files']['preprocessed'] = save_nifti(ct_masked, 'preproc')
        
        # Save resampled mask
        outputs['output_files']['mask'] = save_nifti(mask_data, 'mask_resampled')
        
        return outputs

    def _save_metadata(self, metadata: Dict, basename: str):
        """Save preprocessing metadata"""
        metadata_path = self.output_dir / f"{basename}_preproc_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def verify_orientation(self, input_path: str, output_path: str) -> bool:
        """Verify image orientation is preserved"""
        input_img = nib.load(input_path)
        output_img = nib.load(output_path)
        
        input_orientation = nib.aff2axcodes(input_img.affine)
        output_orientation = nib.aff2axcodes(output_img.affine)
        
        self.logger.info(f"Input orientation: {input_orientation}")
        self.logger.info(f"Output orientation: {output_orientation}")
        
        return input_orientation == output_orientation