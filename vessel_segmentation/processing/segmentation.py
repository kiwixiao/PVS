# processing/segmentation.py
import numpy as np
from scipy.ndimage import distance_transform_edt
import logging
from typing import Tuple, List, Dict
from vessel_segmentation.core.hessian_processor import HessianProcessor
from vessel_segmentation.processing.vessel_particles import VesselParticleSystem

class VesselSegmenter:
    
    def __init__(self, config: Dict):
        """Initialze VesselSegmentater with configuration
        Args:
            config: Dictionary containing vessel detection parameters
        """
        self.logger = logging.getLogger(__name__)
        
        # Debug print incoming config
        self.logger.info(f"VesselSegmenter received config: {config}")
        
        # Extract vessel size range from config
        scale_range = config.get('scale_range', {})
        self.min_vessel_size = float(scale_range.get('min_vessel_size', 0.5))
        self.max_vessel_size = float(scale_range.get('max_vessel_size', 14.0))
        self.num_scales = int(scale_range.get('num_scales', 15)) # this get the number of scales, if empty, take default value 15
        
        # Store output folders dict
        self.folders = None
        
        # hessian computation config
        hessian_config = config.get('hessian', {})
        self.hessian_method = hessian_config.get('method','standard')
        self.use_interpolation = hessian_config.get('use_interpolation',True)
        # Frangi filter parameters
        frangi_params = config.get('frangi_filter', {})
        self.alpha = float(frangi_params.get('alpha', 0.5))
        self.beta = float(frangi_params.get('beta', 0.5))
        self.c = float(frangi_params.get('c', 500)) # c will be computed dynamically for each scale
        self.logger.info(f"Initialized VesselSegmenter with parameters: min_size={self.min_vessel_size}, " 
                        f"max_size={self.max_vessel_size}, num_scales={self.num_scales}")
        
    def set_folders(self, folders: Dict):
        self.folders = folders

    def compute_vesselness(self, image: np.ndarray, 
                          hessian_processor: HessianProcessor,
                          voxel_size: float = 1.0,
                          case_id: str = None,
                          **kwargs) -> Tuple[np.ndarray, Dict]:
        """Compute vesselness measure using Frangi filter
        Args:
            image: Input image data
            hessian_processor: HessianProcessor instance
            voxel_size: Voxel size in mm
            **kwargs: Additional keyword arguments
        
        Returns:
            Tuple of (vesselness map, eigenvalues dictionary)
        """
        try:
            if self.folders is None:
                raise ValueError("Output folders not set, Call set_folders() first.")
            # Get scales samples for vessel enhancement
            scales = hessian_processor.get_hessian_scales(
                self.min_vessel_size,
                self.max_vessel_size,
                voxel_size,
                num_scales=self.num_scales
            )
            
            # Initialize particle system
            particle_system = VesselParticleSystem(output_dir=str(self.folders['vesselness']))
            particles = particle_system.create_particle_system(
            image,
            (voxel_size, voxel_size, voxel_size),
            scales
        )
        
        # Pre-compute Hessians at sample scales using chosen method
            sample_hessians = {}
            for s in scales:
                sample_hessians[s] = hessian_processor.compute_hessian(image, s)
            
            # Process each scale
            scale_responses = []
            best_eigenvalues = None
            best_response = None
            
            # Process each scale, potentially interpolating between samples
            for i, sigma in enumerate(scales):
                # Compute Hessian - either direct or interpolated
                if sigma in sample_hessians:
                    hessian = sample_hessians[sigma]
                elif self.use_interpolation:
                    hessian = hessian_processor.spline_interpolate_scales(
                        image, scales, sigma
                    )
                
                # Compute eigenvalues
                lambda1, lambda2, lambda3 = hessian_processor.compute_eigenvalues(hessian)
                
                # Compute vesselness
                vesselness_scale = self._frangi_filter(lambda1, lambda2, lambda3, sigma)
                scale_responses.append(vesselness_scale)
                
                # Update best response
                if best_response is None or np.max(vesselness_scale) > np.max(best_response):
                    best_response = vesselness_scale
                    best_eigenvalues = {
                        'lambda1': lambda1,
                        'lambda2': lambda2,
                        'lambda3': lambda3
                    }
            # Update particle system
                particle_system.update_scale_response(
                    particles,
                    i,
                    vesselness_scale,
                    best_eigenvalues,
                    sigma
                )
            
            # Combine responses
            vesselness = self._combine_responses(scale_responses, method='max')
            
            # Extract significant vessels
            vessel_particles = particle_system.extract_vessel_particles(particles)
        
            # Save particle system with case_id
            if case_id:
                particle_system.save_particle_system(vessel_particles, case_id)
            else:
                particle_system.save_particle_system(vessel_particles, "vessel_particles")
            
            return vesselness, best_eigenvalues, vessel_particles
            
        except Exception as e:
            self.logger.error(f"Error computing vesselness: {str(e)}")
            raise

    def _frangi_filter(self, lambda1: np.ndarray, lambda2: np.ndarray, 
                      lambda3: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Frangi vesselness filter"""
        # Compute ratios
        Ra = np.abs(lambda2) / (np.abs(lambda3) + 1e-10)
        Rb = np.abs(lambda1) / np.sqrt(np.abs(lambda2 * lambda3) + 1e-10)
        S = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)
        
        # Initialize vesselness
        vesselness = np.zeros_like(lambda1)
        
        # Vessel condition (lambda2 and lambda3 should be negative)
        vessel_mask = (lambda2 < 0) & (lambda3 < 0)
        
        # Apply Frangi's vesselness formula
        vesselness[vessel_mask] = (
            (1 - np.exp(-Ra[vessel_mask]**2 / (2 * self.alpha**2))) *
            np.exp(-Rb[vessel_mask]**2 / (2 * self.beta**2)) *
            (1 - np.exp(-S[vessel_mask]**2 / (2 * self.c**2)))
        )
        
        # Scale normalization
        vesselness *= sigma**2
        
        return vesselness

    def _combine_responses(self, responses: List[np.ndarray], 
                         method: str = 'max') -> np.ndarray:
        """Combine vesselness responses across scales"""
        if method == 'max':
            return np.maximum.reduce(responses)
        else:
            raise ValueError(f"Unknown combination method: {method}")

    def create_boundary_suppression_mask(self, lung_mask: np.ndarray, 
                                       edge_suppression_mm: float,
                                       voxel_size: float) -> np.ndarray:
        """Create mask to suppress false detections near lung boundaries"""
        try:
            # Ensure mask is binary
            lung_mask_binary = (lung_mask > 0).astype(np.uint8)
            
            # Calculate distance transform
            distance_internal = distance_transform_edt(lung_mask_binary)
            distance_external = distance_transform_edt(np.logical_not(lung_mask_binary))
            
            # Convert distance threshold from mm to voxels
            edge_threshold_voxels = edge_suppression_mm / voxel_size
            
            # Create smooth transition at boundaries
            distance_map = distance_internal - distance_external
            suppression_weights = 1 / (1 + np.exp(-2 * (np.abs(distance_map) - edge_threshold_voxels)))
            
            return suppression_weights
            
        except Exception as e:
            self.logger.error(f"Error creating boundary suppression: {str(e)}")
            raise