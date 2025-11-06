"""
Base classes and interfaces for weather prediction models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    mape: float  # Mean Absolute Percentage Error
    last_updated: datetime


class ModelInfo(BaseModel):
    """Model information and metadata."""
    name: str
    model_type: str
    version: str
    training_window_days: int
    feature_columns: List[str]
    target_columns: List[str]
    hyperparameters: Dict[str, Any]
    created_at: datetime
    last_trained: Optional[datetime] = None


class WeatherModel(ABC):
    """Abstract base class for all weather prediction models."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_trained = False
        self.model_info: Optional[ModelInfo] = None
        self.metrics: Optional[ModelMetrics] = None
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the model on historical weather data.
        
        Args:
            data: DataFrame with weather features and targets
        """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions for given features.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty bounds.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        pass
    
    def get_model_info(self) -> ModelInfo:
        """Get model information and metadata."""
        if self.model_info is None:
            raise ValueError("Model info not initialized")
        return self.model_info
    
    def get_metrics(self) -> ModelMetrics:
        """Get current model performance metrics."""
        if self.metrics is None:
            raise ValueError("Model metrics not available")
        return self.metrics
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        raise NotImplementedError("Subclasses must implement save_model")
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def validate_input(self, features: np.ndarray) -> bool:
        """Validate input features."""
        if features is None or len(features) == 0:
            return False
        
        expected_features = len(self.config.get('feature_columns', []))
        if expected_features > 0 and features.shape[-1] != expected_features:
            return False
        
        return True


class ModelException(Exception):
    """Base exception for model-related errors."""
    pass


class ModelTrainingException(ModelException):
    """Exception raised during model training."""
    pass


class ModelPredictionException(ModelException):
    """Exception raised during model prediction."""
    pass


class ModelLoadException(ModelException):
    """Exception raised when loading model fails."""
    pass