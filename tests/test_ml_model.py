"""
Tests for ML model functionality.
"""

import numpy as np
import pytest

from services.ml_model import MLModel

@pytest.fixture
def model_config():
    """Fixture for model configuration."""
    return {
        "model_type": "test",
        "input_dim": 10,
        "output_dim": 1
    }

@pytest.fixture
def test_model(tmp_path, model_config):
    """Fixture for test model instance."""
    model_path = tmp_path / "test_model"
    return MLModel(model_path=model_path, config=model_config)

def test_model_initialization(test_model, model_config):
    """Test model initialization."""
    assert test_model.config == model_config
    assert test_model.model is None

def test_model_abstract_methods(test_model):
    """Test that abstract methods raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        test_model.load()
    
    with pytest.raises(NotImplementedError):
        test_model.predict(np.array([[1.0] * 10]))
    
    with pytest.raises(NotImplementedError):
        test_model.evaluate(
            np.array([[1.0] * 10]),
            np.array([1.0])
        )
    
    with pytest.raises(NotImplementedError):
        test_model.save("test_path") 