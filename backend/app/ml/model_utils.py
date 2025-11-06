"""
Utilities for machine learning models.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from loguru import logger

from app.models.base import ModelMetrics
from app.core.config import settings


class ModelEvaluator:
    """Utility class for evaluating model performance."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """
        Calculate comprehensive model performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            ModelMetrics object with calculated metrics
        """
        # Ensure arrays are 1D
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            logger.warning("No valid predictions for metric calculation")
            return ModelMetrics(
                mae=float('inf'),
                rmse=float('inf'),
                mape=float('inf'),
                last_updated=datetime.utcnow()
            )
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Handle MAPE calculation (avoid division by zero)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        return ModelMetrics(
            mae=float(mae),
            rmse=float(rmse),
            mape=float(mape),
            last_updated=datetime.utcnow()
        )
    
    @staticmethod
    def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate detailed performance metrics including R² score.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with detailed metrics
        """
        # Ensure arrays are 1D
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'r2': -float('inf'),
                'mean_error': 0.0,
                'std_error': float('inf')
            }
        
        # Calculate basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Calculate error statistics
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'mean_error': float(mean_error),
            'std_error': float(std_error)
        }


class ModelSerializer:
    """Utility class for saving and loading models."""
    
    def __init__(self, model_storage_path: str = None):
        self.storage_path = model_storage_path or settings.model_storage_path
        os.makedirs(self.storage_path, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Save a model to disk with metadata.
        
        Args:
            model: Model object to save
            model_name: Name for the model file
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model file
        """
        # Create model-specific directory
        model_dir = os.path.join(self.storage_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_dir, "model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        if metadata:
            metadata_file = os.path.join(model_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_metadata = self._make_json_serializable(metadata)
                json.dump(serializable_metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_file}")
        return model_file
    
    def load_model(self, model_name: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Load a model from disk with metadata.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, metadata)
        """
        model_dir = os.path.join(self.storage_path, model_name)
        model_file = os.path.join(model_dir, "model.pkl")
        metadata_file = os.path.join(model_dir, "metadata.json")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata if exists
        metadata = None
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Model loaded from {model_file}")
        return model, metadata
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model file exists."""
        model_file = os.path.join(self.storage_path, model_name, "model.pkl")
        return os.path.exists(model_file)
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        if not os.path.exists(self.storage_path):
            return []
        
        models = []
        for item in os.listdir(self.storage_path):
            model_dir = os.path.join(self.storage_path, item)
            if os.path.isdir(model_dir):
                model_file = os.path.join(model_dir, "model.pkl")
                if os.path.exists(model_file):
                    models.append(item)
        
        return models
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        import shutil
        
        model_dir = os.path.join(self.storage_path, model_name)
        if os.path.exists(model_dir):
            try:
                shutil.rmtree(model_dir)
                logger.info(f"Model {model_name} deleted successfully")
                return True
            except Exception as e:
                logger.error(f"Error deleting model {model_name}: {e}")
                return False
        
        return False
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


class DataSplitter:
    """Utility class for splitting time series data."""
    
    @staticmethod
    def time_series_split(
        data: pd.DataFrame, 
        test_size: float = 0.2, 
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data into train, validation, and test sets.
        
        Args:
            data: DataFrame with time series data
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(data)
        
        # Calculate split indices
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - validation_size))
        
        # Split data
        train_df = data.iloc[:val_start].copy()
        val_df = data.iloc[val_start:test_start].copy()
        test_df = data.iloc[test_start:].copy()
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def create_sequences(
        data: np.ndarray, 
        sequence_length: int, 
        target_columns: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input data array
            sequence_length: Length of input sequences
            target_columns: Indices of target columns (default: all)
            
        Returns:
            Tuple of (X, y) arrays
        """
        if target_columns is None:
            target_columns = list(range(data.shape[1]))
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            # Input sequence
            X.append(data[i:(i + sequence_length)])
            
            # Target (next time step)
            y.append(data[i + sequence_length, target_columns])
        
        return np.array(X), np.array(y)


class FeatureSelector:
    """Utility class for feature selection."""
    
    @staticmethod
    def select_important_features(
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        method: str = "correlation",
        top_k: int = None
    ) -> Tuple[List[int], List[str]]:
        """
        Select important features based on various criteria.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            method: Selection method ('correlation', 'variance')
            top_k: Number of top features to select
            
        Returns:
            Tuple of (selected_indices, selected_names)
        """
        if method == "correlation":
            # Calculate correlation with target
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y.flatten())[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            # Sort by correlation
            indices = np.argsort(correlations)[::-1]
            
        elif method == "variance":
            # Calculate variance of each feature
            variances = np.var(X, axis=0)
            indices = np.argsort(variances)[::-1]
            
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Select top k features
        if top_k is not None:
            indices = indices[:top_k]
        
        selected_names = [feature_names[i] for i in indices]
        
        logger.info(f"Selected {len(indices)} features using {method} method")
        
        return indices.tolist(), selected_names


class ModelValidator:
    """Utility class for model validation."""
    
    @staticmethod
    def cross_validate_time_series(
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        **model_kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            model_class: Model class to validate
            X: Feature matrix
            y: Target vector
            n_splits: Number of validation splits
            **model_kwargs: Arguments for model initialization
            
        Returns:
            Dictionary with validation metrics
        """
        n_samples = len(X)
        split_size = n_samples // (n_splits + 1)
        
        metrics = {'mae': [], 'rmse': [], 'mape': []}
        
        for i in range(n_splits):
            # Define train and validation indices
            train_end = split_size * (i + 1)
            val_start = train_end
            val_end = min(val_start + split_size, n_samples)
            
            if val_end <= val_start:
                break
            
            # Split data
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            
            # Train model
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            fold_metrics = ModelEvaluator.calculate_detailed_metrics(y_val, y_pred)
            
            metrics['mae'].append(fold_metrics['mae'])
            metrics['rmse'].append(fold_metrics['rmse'])
            metrics['mape'].append(fold_metrics['mape'])
        
        # Calculate mean and std for each metric
        result = {}
        for metric, values in metrics.items():
            if values:
                result[f"{metric}_mean"] = np.mean(values)
                result[f"{metric}_std"] = np.std(values)
            else:
                result[f"{metric}_mean"] = float('inf')
                result[f"{metric}_std"] = 0.0
        
        return result