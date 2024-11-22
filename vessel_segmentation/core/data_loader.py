# core/data_loader.py
from pathlib import Path
import logging
import nibabel as nib
import numpy as np
from typing import Tuple, Dict
import json

class DataLoader:
    def __init__(self, input_dir: str, case_id: str, is_preprocessed: bool = False):
        self.input_dir = Path(input_dir)
        self.case_id = case_id
        self.is_preprocessed = is_preprocessed
        self.logger = logging.getLogger(f'vessel_seg_{case_id}')
        
        # Store image metadata
        self.affine = None
        self.header = None
        self.voxel_size = None
        self.metadata = None
        #self.target_spacing = (0.6, 0.6, 0.6) # Default isotropic spacing
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load CT data and mask"""
        try:
            if self.is_preprocessed:
                return self._load_preprocessed_data()
            else:
                return self._load_raw_data()
                
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _load_preprocessed_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load preprocessed CT and mask"""
        try:
            # Load preprocessed CT
            preproc_path = self.input_dir / f"{self.case_id}_preproc.nii.gz"
            if not preproc_path.exists():
                raise FileNotFoundError(f"Preprocessed CT not found: {preproc_path}")
            
            ct_nifti = nib.load(preproc_path)
            ct_data = ct_nifti.get_fdata()
            
            # Load preprocessed mask
            mask_path = self.input_dir / f"{self.case_id}_mask_resampled.nii.gz"
            if not mask_path.exists():
                raise FileNotFoundError(f"Preprocessed mask not found: {mask_path}")
            
            mask_data = nib.load(mask_path).get_fdata()
            
            # Store metadata
            self.affine = ct_nifti.affine
            self.header = ct_nifti.header
            current_spacing = np.array([np.abs(self.affine[i,i]) for i in range(3)])
            
            if not np.allclose(current_spacing, current_spacing[0], rtol=1e-3):
                self.logger.warning(f"Data not perfectly isotropic: {current_spacing}")
            
            self.voxel_size = float(np.abs(self.affine[0,0]))
            
            
            # Load preprocessing metadata if available
            metadata_path = self.input_dir / f"{self.case_id}_preproc_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
                    # Update voxel size from preprocessing metadata if available
                    if 'current_spacing' in self.metadata:
                        self.voxel_size = self.metadata['current_spacing'][0]
                        
            self.logger.info(f"Loaded preprocessed data: shape={ct_data.shape}, " 
                           f"voxel_size={self.voxel_size}mm")
            
            return ct_data, mask_data
            
        except Exception as e:
            self.logger.error(f"Error loading preprocessed data: {str(e)}")
            raise
        
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw CT and mask"""
        # Load original CT
        ct_path = self.input_dir / f"{self.case_id}.nii.gz"
        ct_nifti = nib.load(ct_path)
        ct_data = ct_nifti.get_fdata()
        
        # Load original mask
        mask_path = self.input_dir / f"{self.case_id}_lung_mask.nii.gz"
        mask_data = nib.load(mask_path).get_fdata()
        
        # Store metadata
        self.affine = ct_nifti.affine
        self.header = ct_nifti.header
        self.voxel_size = float(np.abs(self.affine[0,0]))
        
        return ct_data, mask_data
    
    def get_metadata(self) -> Dict:
        """Return image metadata"""
        return {
            'affine': self.affine,
            'header': self.header,
            'voxel_size': self.voxel_size,
            'preprocessing': self.metadata
        }