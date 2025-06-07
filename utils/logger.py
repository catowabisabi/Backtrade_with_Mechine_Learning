"""
Logging configuration for the project.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "500 MB",
    retention: str = "10 days"
) -> None:
    """
    Configure logger with custom settings.
    
    Args:
        log_file: Path to log file. If None, logs will only go to stderr
        level: Minimum log level to record
        rotation: When to rotate log file (size or time)
        retention: How long to keep log files
    """
    # Remove default handler
    logger.remove()
    
    # Add stderr handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            rotation=rotation,
            retention=retention,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level
        )

def get_logger():
    """Get configured logger instance."""
    return logger 