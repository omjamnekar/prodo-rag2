
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity
    format='%(asctime)s - %(levelname)s - %(message)s'  # Simplified format
)

def get_logger(name: str):
    """Get a logger instance with the specified name."""
    logger = logging.getLogger(name)
    return logger