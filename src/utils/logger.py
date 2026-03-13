"""Logging utility"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs", level=logging.INFO):
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
