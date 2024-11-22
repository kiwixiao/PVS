# core/hessian_processor.py
import numpy as np
from scipy import ndimage
import logging
from pathlib import Path
from typing import Tuple, Dict
import os
from scipy.special import iv # added import for modified bessel function

class HessianProcessor:
    def __init__(self, cache_dir: str, enable_caching: bool = False,
                 method: str = 'standard'):
        """
        Initialize HessianProcessor
        
        Args:
            cache_dir: Directory for caching Hessian calculations
            enable_caching: Whether to cache Hessian calculations (default: False)
        """
        self.cache_dir = Path(cache_dir)
        self.enable_caching = enable_caching
        self.method = method
        self.logger = logging.getLogger(__name__)
        
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)

    def compute_hessian(self, image: np.ndarray, 
                        sigma: float) -> Tuple[np.ndarray, ...]:
        """Compute Hessian with optional caching"""
        cache_file = self.cache_dir / f'hessian_scale_{sigma:.2f}.npz'
        
        # Check cache if enabled
        if self.enable_caching and cache_file.exists():
            self.logger.info(f"Loading cached Hessian for scale {sigma}")
            data = np.load(cache_file)
            return (data['Dxx'], data['Dyy'], data['Dzz'], 
                    data['Dxy'], data['Dxz'], data['Dyz'])
        if self.method == 'scale_time':
            hessian = self._compute_scale_time_hessian(image, sigma)
        else:
            hessian = self._compute_standard_hessian(image, sigma)
            
        if self.enable_caching:
            Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = hessian
            np.savez_compressed(cache_file, Dxx=Dxx, Dyy=Dyy, Dzz=Dzz,
                                Dxy=Dxy, Dxz=Dxz, Dyz=Dyz)
            
        return hessian
    
    def _compute_standard_hessian(self, image: np.ndarray, sigma: float):
        
        # Compute Hessian
        self.logger.info(f"Computing Hessian at scale {sigma}")
        Dxx = ndimage.gaussian_filter(image, sigma=sigma, order=[2,0,0])
        Dyy = ndimage.gaussian_filter(image, sigma=sigma, order=[0,2,0])
        Dzz = ndimage.gaussian_filter(image, sigma=sigma, order=[0,0,2])
        Dxy = ndimage.gaussian_filter(image, sigma=sigma, order=[1,1,0])
        Dxz = ndimage.gaussian_filter(image, sigma=sigma, order=[1,0,1])
        Dyz = ndimage.gaussian_filter(image, sigma=sigma, order=[0,1,1])
          
        return Dxx, Dyy, Dzz, Dxy, Dxz, Dyz

    def _compute_scale_time_hessian(self, image: np.array, sigma: float):
        t = sigma * sigma
        kernel_radius = int(4 * sigma + 0.5)
        x = np.arange(-kernel_radius, kernel_radius +1)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        
        K = self._discrete_gaussian_kernel(X, Y, Z, t, kernel_radius)
        
        Dxx = ndimage.convolve(image, np.diff(np.diff(K, axis=0), axis=0))
        Dyy = ndimage.convolve(image, np.diff(np.diff(K, axis=1), axis=1))
        Dzz = ndimage.convolve(image, np.diff(np.diff(K, axis=2), axis=2))
        Dxy = ndimage.convolve(image, np.diff(np.diff(K, axis=0), axis=1))
        Dxz = ndimage.convolve(image, np.diff(np.diff(K, axis=0), axis=2))
        Dyz = ndimage.convolve(image, np.diff(np.diff(K, axis=1), axis=2))
        
        # Scale normalization
        normalizer = t # gamma = 1.0
        hessian = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
        hessian = [comp * normalizer for comp in hessian]
        
        return tuple(hessian)
    
    def _discrete_gaussian_kernel(self, X, Y, Z, t, kernel_radius):
        kernel = np.zeros_like(X, dtype=float)
        for i in range(-kernel_radius, kernel_radius + 1):
            for j in range(-kernel_radius, kernel_radius + 1):
                for k in range(-kernel_radius, kernel_radius + 1):
                    kernel[i+kernel_radius, j+kernel_radius, k+kernel_radius] = (
                        np.exp(-t) * iv(i, t) * iv(j, t) * iv(k, t)
                    )
        return kernel / np.sum(kernel)

    def spline_interpolate_scales(self, image: np.ndarray, scales: np.ndarray, 
                            target_scale: float) -> Tuple[np.ndarray, ...]:
        """Interpolate between pre-computed scale-space samples using cubic Hermite splines"""
        # Find bracketing scales
        idx = np.searchsorted(scales, target_scale)
        if idx == 0 or idx == len(scales):
            raise ValueError(f"Target scale {target_scale} outside of computed scale range [{scales[0]}, {scales[-1]}]")
                
        s0, s1 = scales[idx-1], scales[idx]
        t0, t1 = s0*s0, s1*s1  # Convert to diffusion times
        
        # Get values at bracketing scales
        f0 = self.compute_hessian(image, s0)
        f1 = self.compute_hessian(image, s1)
        
        # Compute scale derivatives
        m0 = self._compute_scale_derivatives(f0, t0)
        m1 = self._compute_scale_derivatives(f1, t1)
        
        # Interpolation parameter
        h = (target_scale - s0) / (s1 - s0)
        
        # Cubic Hermite basis functions
        h00 = 2*h*h*h - 3*h*h + 1
        h10 = h*h*h - 2*h*h + h
        h01 = -2*h*h*h + 3*h*h
        h11 = h*h*h - h*h
        
        # Interpolate each component
        result = []
        for i in range(6):
            component = (h00 * f0[i] + h10 * m0[i] * (s1-s0) +
                        h01 * f1[i] + h11 * m1[i] * (s1-s0))
            result.append(component)
        
        return tuple(result)

    def _compute_scale_derivatives(self, hessian, t):
        M = np.array([[[0,1,0],[1,-2,1],[0,1,0]],
                     [[1,-2,1],[-2,4,-2],[1,-2,1]],
                     [[0,1,0],[1,-2,1],[0,1,0]]]) / (6*t)
        return [ndimage.convolve(f, M) for f in hessian]


    def compute_eigenvalues(self, hessian_components: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute eigenvalues from Hessian components"""
        try:
            Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = hessian_components
            eigenvalues = np.zeros((3,) + Dxx.shape)
            
            for i in range(Dxx.shape[0]):
                for j in range(Dxx.shape[1]):
                    for k in range(Dxx.shape[2]):
                        H = np.array([
                            [Dxx[i,j,k], Dxy[i,j,k], Dxz[i,j,k]],
                            [Dxy[i,j,k], Dyy[i,j,k], Dyz[i,j,k]],
                            [Dxz[i,j,k], Dyz[i,j,k], Dzz[i,j,k]]
                        ])
                        eigs = np.linalg.eigvalsh(H)
                        eigenvalues[:,i,j,k] = np.sort(eigs)
            
            return eigenvalues[0], eigenvalues[1], eigenvalues[2]
            
        except Exception as e:
            self.logger.error(f"Error computing eigenvalues: {str(e)}")
            raise

    def get_hessian_scales(self, min_size: float, max_size: float, 
                          voxel_size: float, num_scales: int = 15) -> np.ndarray:
        """Calculate optimal scales for vessel enhancement"""
        min_sigma = (min_size / voxel_size) / 2
        max_sigma = (max_size / voxel_size) / 2
        return np.logspace(np.log10(min_sigma), np.log10(max_sigma), num_scales)

    def clear_cache(self):
        """Clear Hessian cache"""
        if self.enable_caching and self.cache_dir.exists():
            self.logger.info("Clearing Hessian cache")
            for cache_file in self.cache_dir.glob('hessian_scale_*.npz'):
                cache_file.unlink()