"""
API endpoints for the ML service.
"""

from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.ml_model import MLModel
from utils.logger import get_logger

logger = get_logger()
app = FastAPI(title="ML Model API")

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: List[List[float]]

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    confidence: List[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> Dict:
    """
    Make predictions using the ML model.
    
    Args:
        request: Prediction request containing input data
        
    Returns:
        Dictionary containing predictions and confidence scores
    """
    try:
        # Convert input data to numpy array
        input_data = np.array(request.data)
        
        # Make predictions (implement your model prediction here)
        predictions = [0.0] * len(request.data)  # Placeholder
        confidence = [1.0] * len(request.data)   # Placeholder
        
        return {
            "predictions": predictions,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"} 