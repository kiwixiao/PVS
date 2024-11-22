from pathlib import Path
from typing import Dict
import os
import numpy as np

def create_folder_structure(output_dir: str, case_id: str) -> Dict[str, str]:
    """Create organized folder structure for vessel segmentation"""
    
    folders = {
        'root': output_dir,
        'cases': os.path.join(output_dir, 'cases'),
        'cache': os.path.join(output_dir, 'cache'),
        'logs': os.path.join(output_dir, 'logs')
    }
    
    # Case-specific folders
    case_root = os.path.join(folders['cases'], case_id)
    folders.update({
        'case_root': case_root,
        'preprocessed': os.path.join(case_root, 'preprocessed'),
        'hessian': os.path.join(case_root, 'hessian'),      # Optional caching directory
        'eigenvalues': os.path.join(case_root, 'eigenvalues'),
        'vesselness': os.path.join(case_root, 'vesselness'),
        'surfaces': os.path.join(case_root, 'surfaces'),
        'analysis': os.path.join(case_root, 'analysis'),
        'visualization': os.path.join(case_root, 'visualization')
    })
    
    # Create required folders (excluding optional caching directory)
    required_folders = {k: v for k, v in folders.items() if k != 'hessian'}
    for folder in required_folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Create hessian directory only if caching is enabled
    # This will be handled by HessianProcessor when initialized with enable_caching=True
    
    return folders

def ensure_input_files(input_dir: str, case_id: str):
    """Check required input files exist"""
    required_files = [
        f"{case_id}.nii.gz",
        f"{case_id}_lung_mask.nii.gz"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(input_dir, file)):
            raise ValueError(f"Required input file not found: {file}")

def save_nifti(data: np.ndarray, affine: np.ndarray, output_path: str):
    """Save data as NIfTI file"""
    import nibabel as nib
    nifti = nib.Nifti1Image(data, affine)
    nib.save(nifti, output_path)

def get_case_directory(output_dir: str, case_id: str) -> Path:
    """Get case-specific directory"""
    return Path(output_dir) / 'cases' / case_id

def clean_cache_directories(output_dir: str, case_id: str):
    """Clean up cache directories if they exist"""
    case_dir = get_case_directory(output_dir, case_id)
    hessian_cache = case_dir / 'hessian'
    if hessian_cache.exists():
        import shutil
        shutil.rmtree(hessian_cache)
        os.makedirs(hessian_cache)