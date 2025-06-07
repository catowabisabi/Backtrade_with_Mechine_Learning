"""
Machine Learning model service.
"""

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from loguru import logger

class MLModel:
    """Base class for ML models."""
    
    def __init__(self, model_path: Union[str, Path], config: Dict[str, Any]):
        """
        Initialize ML model.
        
        Args:
            model_path: Path to saved model
            config: Model configuration
        """
        self.model_path = Path(model_path)
        self.config = config
        self.model = None
        logger.info(f"Initializing model from {self.model_path}")
    
    def load(self) -> None:
        """Load model from disk."""
        raise NotImplementedError("Subclasses must implement load()")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Model predictions
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_data: Test input data
            test_labels: True labels for test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        raise NotImplementedError("Subclasses must implement save()") 