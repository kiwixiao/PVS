# Vessel Segmentation Pipeline (log_scale_parallel Branch)

## Overview
This branch implements a vessel segmentation pipeline with parallel processing for scale-space computation and a non-vectorized centerline extraction method. The pipeline consists of several key components:

### 1. Vessel Enhancement
- Implements parallel processing for multi-scale Hessian analysis
- Uses logarithmic scale sampling between 0.6mm and 6.0mm
- Computes vesselness response using Frangi's method
- Maintains maximum vesselness response across scales

### 2. Vessel Segmentation
- Performs local optimal thresholding based on vesselness measures
- Filters small disconnected components (threshold: 50 voxels)
- Uses 26-connectivity for component analysis

### 3. Centerline Extraction
- Implements Palagyi & Kuba's 6-subiteration thinning algorithm
- Non-vectorized implementation for reliable topology preservation
- Features:
  - Template-based deletion criteria (6 templates per direction)
  - Proper topology preservation checks
  - Sequential processing in U,D,N,S,E,W directions
  - Memory-efficient batch processing
  - Point classification (endpoints, segments, bifurcations)

### 4. Output Files
- Intermediate Results:
  - Vesselness measures
  - Binary vessel mask
- Final Results:
  - Filtered vessel segmentation
  - Centerlines (NRRD and VTK formats)
  - Point type classification

## Usage
```bash
python run_segmentation.py --project-name <project_name>
```

## Key Features
- Parallel processing for scale-space computation
- Logarithmic scale sampling for efficient vessel detection
- Non-vectorized thinning for reliable centerline extraction
- Memory-efficient implementation
- Comprehensive output formats (NRRD, VTK)

## Implementation Notes
- The thinning algorithm is intentionally kept non-vectorized to ensure proper topology preservation
- Uses batch processing for memory efficiency
- Includes detailed statistics and validation checks
- Supports both NRRD and VTK output formats for visualization 