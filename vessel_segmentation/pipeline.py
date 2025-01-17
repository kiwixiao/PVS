import os
import logging
from typing import Optional
import SimpleITK as sitk
from .graph_builder import GraphBuilder
from .separator import VesselSeparator
from .periphery_matcher import PeripheryMatcher
from .classifier import VesselClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArteryVeinSeparator:
    """
    Main pipeline for artery-vein separation following Charbonnier's methodology.
    """
    
    def __init__(self, 
                 centerline_file: str,
                 point_type_file: str,
                 vessel_mask_file: str,
                 lung_mask_file: str,
                 scale_file: str,  # sigma_max.nrrd
                 direction_file: str,  # vessel_direction.nrrd
                 output_dir: str,
                 max_periphery_distance: float = 30.0,
                 min_matching_vessels: int = 2):
        """
        Initialize the artery-vein separator
        
        Args:
            centerline_file: Path to centerline NRRD file (X = {xk})
            point_type_file: Path to point type NRRD file
            vessel_mask_file: Path to vessel mask NRRD file (V)
            lung_mask_file: Path to lung mask NRRD file (L)
            scale_file: Path to vessel scale file (sigma_max.nrrd)
            direction_file: Path to vessel direction file (vessel_direction.nrrd)
            output_dir: Directory for output files and debug visualizations
            max_periphery_distance: Maximum distance (Dmax) for periphery matching in mm
            min_matching_vessels: Minimum number of matching vessels required
        """
        self.centerline_file = centerline_file
        self.point_type_file = point_type_file
        self.vessel_mask_file = vessel_mask_file
        self.lung_mask_file = lung_mask_file
        self.scale_file = scale_file
        self.direction_file = direction_file
        self.output_dir = output_dir
        self.max_periphery_distance = max_periphery_distance
        self.min_matching_vessels = min_matching_vessels
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up file logging
        fh = logging.FileHandler(os.path.join(output_dir, 'av_separation.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    def process(self):
        """Run the complete artery-vein separation pipeline"""
        logger.info("Starting artery-vein separation...")
        
        # Step 1: Build geometric graph
        logger.info("Step 1: Building geometric graph...")
        graph_builder = GraphBuilder(
            centerline_file=self.centerline_file,
            point_type_file=self.point_type_file,
            vessel_mask_file=self.vessel_mask_file,
            lung_mask_file=self.lung_mask_file,
            output_dir=self.output_dir,
            scale_file=self.scale_file
        )
        graph = graph_builder.build_graph()
        
        # Step 2: Detect attachment points
        logger.info("Step 2: Detecting attachment points...")
        separator = VesselSeparator(graph, self.output_dir)
        attachment_points = separator.detect_attachment_points()
        
        # Step 3: Create subtrees
        logger.info("Step 3: Creating subtrees...")
        subtrees = separator.create_subtrees(attachment_points)
        
        # Step 4: Analyze peripheral relationships
        logger.info("Step 4: Analyzing peripheral relationships...")
        matcher = PeripheryMatcher(
            graph=graph,
            max_periphery_distance=self.max_periphery_distance,
            min_matching_vessels=self.min_matching_vessels,
            output_dir=self.output_dir
        )
        relationships = matcher.analyze_periphery(subtrees)
        
        # Step 5: Link subtrees into groups
        logger.info("Step 5: Linking subtrees into groups...")
        groups = matcher.link_subtrees(subtrees, relationships)
        
        # Step 6: Classify vessels
        logger.info("Step 6: Classifying vessels...")
        classifier = VesselClassifier(
            graph=graph,
            output_dir=self.output_dir
        )
        classifier.classify_vessels(subtrees, groups)
        
        # Step 7: Save results
        logger.info("Step 7: Saving results...")
        classifier.save_results()
        
        logger.info("Artery-vein separation completed successfully")
        
def run_av_separation(input_dir: str, output_dir: str = None,
                     max_periphery_distance: float = 30.0,
                     min_matching_vessels: int = 2) -> None:
    """Run the artery-vein separation pipeline
    
    Args:
        input_dir: Directory containing vessel segmentation results
        output_dir: Directory to save results (default: input_dir/av_separation)
        max_periphery_distance: Maximum distance for periphery matching
        min_matching_vessels: Minimum matching vessels required
    """
    logger.info("Starting artery-vein separation...")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, "av_separation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "av_separation.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    try:
        # Step 1: Build geometric graph
        logger.info("Step 1: Building geometric graph...")
        
        # Input files
        files = {
            'centerline': os.path.join(input_dir, "final_results/refined_centerlines.nrrd"),
            'point_type': os.path.join(input_dir, "final_results/refined_centerline_point_types.nrrd"),
            'vessel_mask': os.path.join(input_dir, "final_results/filtered_final_vessels.nrrd"),
            'lung_mask': os.path.join(input_dir, "intermediate_results/eroded_mask.nrrd"),
            'scale': os.path.join(input_dir, "intermediate_results/sigma_max.nrrd"),
            'direction': os.path.join(input_dir, "intermediate_results/vessel_direction.nrrd")
        }
        
        # Verify input files exist
        for name, path in files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required input file not found: {path}")
        
        # Build graph
        graph_builder = GraphBuilder(
            centerline_file=files['centerline'],
            point_type_file=files['point_type'],
            vessel_mask_file=files['vessel_mask'],
            lung_mask_file=files['lung_mask'],
            scale_file=files['scale'],
            direction_file=files['direction'],
            output_dir=output_dir
        )
        graph = graph_builder.build_graph()
        
        # Step 2: Detect attachment points
        logger.info("Step 2: Detecting attachment points...")
        separator = VesselSeparator(graph=graph, output_dir=output_dir)
        attachment_points = separator.detect_attachment_points()
        
        # Step 3: Create subtrees
        logger.info("Step 3: Creating subtrees...")
        subtrees = separator.create_subtrees(attachment_points)
        
        # Step 4: Analyze peripheral relationships
        logger.info("Step 4: Analyzing peripheral relationships...")
        matcher = PeripheryMatcher(graph=graph,
                                 max_periphery_distance=max_periphery_distance,
                                 min_matching_vessels=min_matching_vessels,
                                 output_dir=output_dir)
        relationships = matcher.analyze_periphery(subtrees)
        
        # Step 5: Classify vessels
        logger.info("Step 5: Classifying vessels...")
        classifier = VesselClassifier(graph=graph,
                                    vessel_mask_file=files['vessel_mask'],
                                    output_dir=output_dir)
        classifier.classify_vessels(subtrees, relationships)
        
        logger.info("Artery-vein separation completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during artery-vein separation: {str(e)}")
        raise
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close() 