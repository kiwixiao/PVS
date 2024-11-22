# pipeline.py
from pathlib import Path
import logging
from typing import Dict
import json
import nibabel as nib
import numpy as np
import os
import vtk
from vtk.util import numpy_support


from .core.data_loader import DataLoader
from .preprocessing.preprocessing import CTPreprocessor
from .core.hessian_processor import HessianProcessor
from .core.vessel_extractor import VesselExtractor
from .core.vessel_analyzer import VesselAnalyzer
from .processing.segmentation import VesselSegmenter
from .processing.surface_generation import SurfaceGenerator
from .visualization.visualizer import Visualizer
from .utils.file_utils import create_folder_structure, ensure_input_files

class VesselSegmentationPipeline:
    def __init__(self, 
                input_dir: str,
                output_dir: str,
                case_id: str,
                config: Dict,
                preprocess: bool = False,
                force_preprocess: bool = False,
                debug: bool = True):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.case_id = case_id
        self.config = config
        self.preprocess = preprocess
        self.force_preprocess = force_preprocess
        self.debug = debug
        
        # Create folder structure
        self.folders = create_folder_structure(str(output_dir), case_id)
        
        # Setup logging
        self.logger = logging.getLogger(f'vessel_seg_{case_id}')
        
        # Initialize components
        if preprocess:
            self.preprocessor = CTPreprocessor(self.folders['preprocessed'])
            
        # Initialize pipeline components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize pipeline components"""
        # Determine which directory to use for data loading
        data_dir = self.folders['preprocessed'] if self.preprocess else self.input_dir
        
        self.data_loader = DataLoader(data_dir, self.case_id, 
                                    is_preprocessed=self.preprocess)
        self.hessian_processor = HessianProcessor(
        self.folders['hessian'],
        enable_caching=self.config.get('processing', {}).get('enable_caching', False)
    )
    
        # Get vessel detection config with proper defaults
        vessel_detection_config = self.config.get('vessel_detection', {})
        if not vessel_detection_config:
            self.logger.warning("No vessel_detection configuration found. Using defaults.")
        
        self.logger.info(f"Initializing segmenter with config: {vessel_detection_config}")
        self.segmenter = VesselSegmenter(vessel_detection_config)
        # Set folders for segmenter
        self.segmenter.set_folders(self.folders)
        
        self.vessel_extractor = VesselExtractor(self.folders['analysis'])
        self.surface_generator = SurfaceGenerator(self.folders['surfaces'])
        self.analyzer = VesselAnalyzer(self.folders['analysis'])
        self.visualizer = Visualizer(self.folders['visualization'])
        
        

    def _run_preprocessing(self) -> Dict:
        """Run preprocessing step"""
        self.logger.info("Starting preprocessing step")
        
        ct_path = self.input_dir / f"{self.case_id}.nii.gz"
        mask_path = self.input_dir / f"{self.case_id}_lung_mask.nii.gz"
        
        if not ct_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Input files not found: {ct_path} or {mask_path}")
            
        # Run preprocessing with config
        _, metadata = self.preprocessor.preprocess_ct_for_vessels(
            str(ct_path),
            str(mask_path),
            config=self.config.get('preprocessing', {})
        )
        
        return metadata

    def _check_preprocessing_exists(self) -> bool:
        """Check if preprocessing files exist"""
        required_files = [
            f"{self.case_id}_preproc.nii.gz",
            f"{self.case_id}_windowed.nii.gz",
            f"{self.case_id}_mask_resampled.nii.gz"
        ]
        
        condition = all(os.path.exists(os.path.join(self.folders['preprocessed'], file)) 
                  for file in required_files) # check if any of the requried file exists? if one of them exists, it will return true.
        return condition

    def run(self) -> Dict:
        """Run complete vessel segmentation pipeline"""
        try:
            self.logger.info(f"Starting vessel segmentation for case {self.case_id}")
            
            # Handle preprocessing
            if self.preprocess: # first, check what is the input of preprocess bool value, if True, move next line
                preproc_exists = self._check_preprocessing_exists() # now, check do we already have preprocessed data?
                
                if not preproc_exists or self.force_preprocess: # if we do not have preprocessed before, or we force repreprocessing, move to preprocessing
                    # Detailed logging why doing preprocessing again.
                    if preproc_exists:
                        self.logger.info(f"Previous preprocessed data found, but force redo preprocessing")
                    elif not preproc_exists:
                        self.logger.info(f"Previous preprocessed data not found, will do preprocessing")
                    else:
                        self.logger.info(f"Wrong return of checking preprocessed function")
                    
                    preproc_metadata = self._run_preprocessing()
                    self.logger.info("Preprocessing completed")
                else:
                    self.logger.info("Using existing preprocessed data")
                    metadata_file = os.path.join(self.folders['preprocessed'], 
                                                 f"{self.case_id}_preproc_metadata.json")
                    if os.path.exists(metadata_file):
                        with open(metadata_file) as f:
                            preproc_metadata = json.load(f)
            
            # Check for existing vesselness results
            vesselness_file = os.path.join(self.folders['vesselness'], 
                                           f'{self.case_id}_vesselness.nii.gz')
            eigenvalues_file = os.path.join(self.folders['eigenvalues'], 'eigenvalues.npz')
            force_recompute = self.config.get('processing', {}).get('force_recompute', False)
            
            if os.path.exists(vesselness_file) and os.path.join(eigenvalues_file) and not force_recompute:
                self.logger.info("Loading existing vesselness results")
                vesselness = nib.load(vesselness_file).get_fdata()
                eigenvalues = dict(np.load(eigenvalues_file))
            else:
                # Load data and compute vesselness
                ct_data, mask_data = self.data_loader.load_data()
                self.logger.info("Computing vesselness")
                vesselness, eigenvalues, vessel_particles = self.segmenter.compute_vesselness(
                    ct_data, 
                    self.hessian_processor,
                    voxel_size=self.data_loader.get_metadata()['voxel_size'],
                    case_id = self.case_id # Pass case_id
                )
                
                # Save results
                nib.save(nib.Nifti1Image(vesselness, self.data_loader.affine), 
                        vesselness_file)
                np.savez(eigenvalues_file, **eigenvalues)

                # # Save vessel particles
                # particles_path = self.folders['vesselness'] / f'{self.case_id}_vessel_particles.vtp'
                # writer = vtk.vtkXMLPolyDataWriter()
                # writer.SetFileName(str(particles_path))
                # writer.SetInputData(vessel_particles)
                # writer.Write()
            
            # Extract vessel tree
            # self.logger.info("Extracting vessel tree")
            # vessel_mask = self.vessel_extractor.extract_vessels(
            #     vesselness, 
            #     self.data_loader.load_data()[1]  # mask
            # )
            
            # vessel_tree = self.vessel_extractor.build_vessel_tree(
            #     vessel_mask, 
            #     vesselness,
            #     eigenvalues
            # )
            
            # # Generate surface
            # self.logger.info("Generating surface model")
            # surface_path = self.surface_generator.generate_surface(
            #     vessel_mask,
            #     vessel_tree,
            #     self.data_loader.get_metadata()['voxel_size']
            # )
            
            # Run analysis
            # self.logger.info("Running vessel analysis")
            # analysis_results = self.analyzer.analyze_vessels(vessel_tree)

            
            return {
                #'surface_model': surface_path,
                #'analysis': analysis_results,
                #'vessel_tree': vessel_tree,
                #'vessel_mask': vessel_mask,
                'preprocessing_metadata': preproc_metadata if self.preprocess else None,
                'output_files': {
                    'vesselness_map': vesselness_file,
                    'vessel_mask': os.path.join(self.folders['vesselness'], 
                                                f'{self.case_id}_mask.nii.gz'),
                 #   'surface_model': surface_path,
                    'analysis': self.folders['analysis'],
                    'visualization': self.folders['visualization']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise