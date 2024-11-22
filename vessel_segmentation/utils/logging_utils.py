# utils/logging_utils.py
import logging
from pathlib import Path
from datetime import datetime
import sys

def setup_logging(log_dir: str, case_id: str, debug: bool = False) -> logging.Logger:
    """Configure logging for vessel segmentation"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'vessel_seg_{case_id}_{timestamp}.log'
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger(f'vessel_seg_{case_id}')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Initial log messages
    logger.info(f"Log file created at: {log_file}")
    logger.info(f"Logging level: {'DEBUG' if debug else 'INFO'}")
    
    return logger

def log_parameters(logger: logging.Logger, params: dict):
    """Log processing parameters"""
    logger.info("Processing parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

def log_error_with_traceback(logger: logging.Logger, error: Exception):
    """Log error with full traceback"""
    import traceback
    logger.error(f"Error: {str(error)}")
    logger.debug("Traceback:", exc_info=True)
    return traceback.format_exc()