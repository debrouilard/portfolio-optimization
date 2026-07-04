import os
import logging

def setup_logging():
    """
    Configures a unified logging format for all orchestration scripts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("PortfolioOptimization")

def ensure_dir(file_path):
    """
    Ensures that a directory exists given a specific target file path.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)