"""
Machine Learning model service.
Provides a base class and implementations for ML model management.
"""

from pathlib import Path
from typing import Any, Dict, Union, Optional
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger

class MLModel:
    """Base class for ML models."""
    
    def __init__(self, model_path: Union[str, Path], config: Dict[str, Any]):
        """
        Initialize ML model.
        
        Args:
            model_path: Path to saved model
            config: Model configuration including:
                - model_type: Type of model (e.g. 'sklearn', 'pytorch')
                - feature_columns: List of feature column names
                - target_column: Name of target column
                - scaler_path: Optional path to fitted scaler
        """
        self.model_path = Path(model_path)
        self.config = config
        self.model = None
        self.scaler = None
        logger.info(f"Initializing model from {self.model_path}")
        
        # Load scaler if specified
        scaler_path = config.get('scaler_path')
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def load_scaler(self, scaler_path: Union[str, Path]) -> None:
        """
        Load fitted scaler from disk.
        
        Args:
            scaler_path: Path to saved scaler
        """
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            logger.error(f"Failed to load scaler: {str(e)}")
            raise
    
    def load(self) -> None:
        """Load model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data
        """
        if self.scaler is not None:
            return self.scaler.transform(data)
        return data
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
            
        # Preprocess input data
        processed_data = self.preprocess(input_data)
        
        # Make predictions
        try:
            predictions = self.model.predict(processed_data)
            logger.debug(f"Made predictions for {len(input_data)} samples")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_proba(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Get prediction probabilities if model supports it.
        
        Args:
            input_data: Input data
            
        Returns:
            Prediction probabilities or None if not supported
        """
        if not hasattr(self.model, 'predict_proba'):
            return None
            
        processed_data = self.preprocess(input_data)
        return self.model.predict_proba(processed_data)
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_data: Test input data
            test_labels: True labels for test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(test_data)
        
        metrics = {
            'accuracy': accuracy_score(test_labels, predictions),
            'precision': precision_score(test_labels, predictions, average='weighted'),
            'recall': recall_score(test_labels, predictions, average='weighted'),
            'f1': f1_score(test_labels, predictions, average='weighted')
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, save_path)
            logger.info(f"Saved model to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise 