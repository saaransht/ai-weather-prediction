"""
Ensemble system for combining multiple weather prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from loguru import logger

from app.models.base import WeatherModel, ModelMetrics
from app.models.lstm_model import LSTMWeatherModel
from app.models.arima_model import ARIMAWeatherModel
from app.models.random_forest_model import RandomForestWeatherModel
from app.models.fuzzy_model import FuzzyTimeSeriesModel
from app.models.lube_model import LUBEWeatherModel
from app.ml.model_utils import ModelEvaluator


class EnsembleWeatherPredictor:
    """Ensemble system for combining multiple weather prediction models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'models': ['lstm', 'arima', 'random_forest', 'fuzzy', 'lube'],
            'weighting_method': 'performance',  # 'equal', 'performance', 'adaptive'
            'performance_window': 100,  # Number of recent predictions for adaptive weighting
            'min_weight': 0.05,  # Minimum weight for any model
            'fallback_method': 'simple_average',  # Fallback when models fail
            'uncertainty_method': 'lube_primary',  # 'lube_primary', 'ensemble_variance', 'combined'
            'target_columns': ['temperature', 'humidity', 'pressure', 'wind_speed']
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.models: Dict[str, WeatherModel] = {}
        self.model_weights: Dict[str, float] = {}
        self.model_performance_history: Dict[str, List[float]] = {}
        self.prediction_history: List[Dict[str, Any]] = []
        self.is_trained = False
        
        logger.info("Ensemble Weather Predictor initialized")
    
    def add_model(self, model_name: str, model: WeatherModel) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model_name: Name identifier for the model
            model: Trained WeatherModel instance
        """
        if not model.is_trained:
            logger.warning(f"Model {model_name} is not trained")
        
        self.models[model_name] = model
        self.model_weights[model_name] = 1.0 / len(self.config['models'])  # Initial equal weight
        self.model_performance_history[model_name] = []
        
        logger.info(f"Added model {model_name} to ensemble")
    
    def initialize_models(self) -> Dict[str, WeatherModel]:
        """
        Initialize all models specified in config.
        
        Returns:
            Dictionary of initialized models
        """
        model_instances = {}
        
        for model_type in self.config['models']:
            try:
                if model_type == 'lstm':
                    model_instances[model_type] = LSTMWeatherModel()
                elif model_type == 'arima':
                    model_instances[model_type] = ARIMAWeatherModel()
                elif model_type == 'random_forest':
                    model_instances[model_type] = RandomForestWeatherModel()
                elif model_type == 'fuzzy':
                    model_instances[model_type] = FuzzyTimeSeriesModel()
                elif model_type == 'lube':
                    model_instances[model_type] = LUBEWeatherModel()
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                self.add_model(model_type, model_instances[model_type])
                
            except Exception as e:
                logger.error(f"Failed to initialize {model_type} model: {e}")
        
        return model_instances
    
    def train_ensemble(self, data: pd.DataFrame) -> None:
        """
        Train all models in the ensemble.
        
        Args:
            data: Training data DataFrame
        """
        logger.info("Training ensemble models")
        
        successful_models = []
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name} model")
                model.train(data)
                successful_models.append(model_name)
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                # Remove failed model from ensemble
                if model_name in self.model_weights:
                    del self.model_weights[model_name]
        
        if not successful_models:
            raise Exception("No models trained successfully")
        
        # Normalize weights for successful models
        self._normalize_weights()
        
        self.is_trained = True
        logger.info(f"Ensemble training completed. {len(successful_models)} models trained successfully")
    
    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1.0."""
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight
    
    def _calculate_performance_weights(self) -> Dict[str, float]:
        """
        Calculate weights based on recent model performance.
        
        Returns:
            Dictionary of performance-based weights
        """
        weights = {}
        
        for model_name, model in self.models.items():
            if model.metrics and model.is_trained:
                # Use inverse of MAE as performance score (lower MAE = higher weight)
                mae = model.metrics.mae
                if mae > 0:
                    performance_score = 1.0 / mae
                else:
                    performance_score = 1.0
            else:
                performance_score = 0.1  # Low weight for untrained/poor models
            
            weights[model_name] = performance_score
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        # Apply minimum weight constraint
        min_weight = self.config['min_weight']
        for name in weights:
            weights[name] = max(weights[name], min_weight)
        
        # Renormalize after applying minimum weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return weights
    
    def _calculate_adaptive_weights(self) -> Dict[str, float]:
        """
        Calculate adaptive weights based on recent prediction performance.
        
        Returns:
            Dictionary of adaptive weights
        """
        if len(self.prediction_history) < 10:
            # Not enough history, use performance weights
            return self._calculate_performance_weights()
        
        # Use recent prediction history
        window_size = min(self.config['performance_window'], len(self.prediction_history))
        recent_history = self.prediction_history[-window_size:]
        
        model_errors = {name: [] for name in self.models.keys()}
        
        # Calculate recent errors for each model
        for record in recent_history:
            if 'actual' in record and 'predictions' in record:
                actual = record['actual']
                predictions = record['predictions']
                
                for model_name in self.models.keys():
                    if model_name in predictions:
                        pred = predictions[model_name]
                        if isinstance(actual, (list, np.ndarray)) and isinstance(pred, (list, np.ndarray)):
                            error = np.mean(np.abs(np.array(actual) - np.array(pred)))
                        else:
                            error = abs(actual - pred)
                        model_errors[model_name].append(error)
        
        # Calculate weights based on inverse of average error
        weights = {}
        for model_name in self.models.keys():
            if model_errors[model_name]:
                avg_error = np.mean(model_errors[model_name])
                weights[model_name] = 1.0 / (avg_error + 1e-6)  # Add small epsilon to avoid division by zero
            else:
                weights[model_name] = 1.0
        
        # Normalize and apply constraints
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        # Apply minimum weight constraint
        min_weight = self.config['min_weight']
        for name in weights:
            weights[name] = max(weights[name], min_weight)
        
        # Final normalization
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return weights
    
    def update_weights(self) -> None:
        """Update model weights based on the configured weighting method."""
        method = self.config['weighting_method']
        
        if method == 'equal':
            # Equal weights for all models
            num_models = len(self.models)
            self.model_weights = {name: 1.0 / num_models for name in self.models.keys()}
        
        elif method == 'performance':
            # Weights based on training performance
            self.model_weights = self._calculate_performance_weights()
        
        elif method == 'adaptive':
            # Weights based on recent prediction performance
            self.model_weights = self._calculate_adaptive_weights()
        
        else:
            logger.warning(f"Unknown weighting method: {method}. Using equal weights.")
            num_models = len(self.models)
            self.model_weights = {name: 1.0 / num_models for name in self.models.keys()}
        
        logger.debug(f"Updated model weights: {self.model_weights}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Array of ensemble predictions
        """
        if not self.is_trained:
            raise Exception("Ensemble not trained")
        
        # Update weights before prediction
        self.update_weights()
        
        # Collect predictions from all models
        model_predictions = {}
        successful_predictions = []
        
        for model_name, model in self.models.items():
            try:
                if model.is_trained:
                    pred = model.predict(features)
                    model_predictions[model_name] = pred
                    successful_predictions.append((model_name, pred, self.model_weights.get(model_name, 0.0)))
                
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
                continue
        
        if not successful_predictions:
            raise Exception("All model predictions failed")
        
        # Combine predictions using weighted average
        ensemble_prediction = self._combine_predictions(successful_predictions)
        
        # Store prediction for adaptive weighting
        prediction_record = {
            'timestamp': datetime.utcnow(),
            'predictions': model_predictions,
            'ensemble': ensemble_prediction,
            'weights': self.model_weights.copy()
        }
        self.prediction_history.append(prediction_record)
        
        # Keep history manageable
        if len(self.prediction_history) > self.config['performance_window'] * 2:
            self.prediction_history = self.prediction_history[-self.config['performance_window']:]
        
        return ensemble_prediction
    
    def _combine_predictions(self, predictions: List[Tuple[str, np.ndarray, float]]) -> np.ndarray:
        """
        Combine model predictions using weighted average.
        
        Args:
            predictions: List of (model_name, prediction, weight) tuples
            
        Returns:
            Combined prediction array
        """
        if not predictions:
            raise Exception("No predictions to combine")
        
        # Initialize ensemble prediction
        first_pred = predictions[0][1]
        ensemble_pred = np.zeros_like(first_pred, dtype=np.float64)
        total_weight = 0.0
        
        # Weighted combination
        for model_name, pred, weight in predictions:
            try:
                # Ensure prediction has the right shape
                pred_array = np.asarray(pred, dtype=np.float64)
                
                if pred_array.shape != ensemble_pred.shape:
                    logger.warning(f"Shape mismatch for {model_name}: {pred_array.shape} vs {ensemble_pred.shape}")
                    continue
                
                ensemble_pred += pred_array * weight
                total_weight += weight
                
            except Exception as e:
                logger.warning(f"Error combining prediction from {model_name}: {e}")
                continue
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions with uncertainty quantification.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        method = self.config['uncertainty_method']
        
        if method == 'lube_primary' and 'lube' in self.models:
            # Use LUBE model for uncertainty if available
            try:
                lube_model = self.models['lube']
                if lube_model.is_trained:
                    return lube_model.predict_with_uncertainty(features)
            except Exception as e:
                logger.warning(f"LUBE uncertainty estimation failed: {e}")
        
        # Fallback to ensemble variance method
        return self._ensemble_variance_uncertainty(features)
    
    def _ensemble_variance_uncertainty(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate uncertainty using ensemble variance.
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        # Collect all model predictions
        all_predictions = []
        
        for model_name, model in self.models.items():
            try:
                if model.is_trained:
                    pred = model.predict(features)
                    all_predictions.append(np.asarray(pred))
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
                continue
        
        if not all_predictions:
            raise Exception("No successful predictions for uncertainty estimation")
        
        # Stack predictions
        predictions_array = np.array(all_predictions)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        # Calculate confidence intervals (95%)
        lower_bounds = mean_pred - 1.96 * std_pred
        upper_bounds = mean_pred + 1.96 * std_pred
        
        return mean_pred, lower_bounds, upper_bounds
    
    def update_with_actual(self, actual_values: np.ndarray) -> None:
        """
        Update ensemble with actual observed values for adaptive learning.
        
        Args:
            actual_values: Actual observed values
        """
        if self.prediction_history:
            # Update the most recent prediction record with actual values
            self.prediction_history[-1]['actual'] = actual_values
            
            # Update individual model performance histories
            recent_record = self.prediction_history[-1]
            if 'predictions' in recent_record:
                for model_name, pred in recent_record['predictions'].items():
                    if model_name in self.model_performance_history:
                        # Calculate error
                        error = np.mean(np.abs(np.array(actual_values) - np.array(pred)))
                        self.model_performance_history[model_name].append(error)
                        
                        # Keep history manageable
                        max_history = self.config['performance_window']
                        if len(self.model_performance_history[model_name]) > max_history:
                            self.model_performance_history[model_name] = \
                                self.model_performance_history[model_name][-max_history:]
    
    def get_model_contributions(self) -> Dict[str, float]:
        """
        Get current model contribution weights.
        
        Returns:
            Dictionary of model names and their weights
        """
        return self.model_weights.copy()
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """
        Get ensemble performance metrics.
        
        Returns:
            Dictionary with ensemble metrics
        """
        if not self.prediction_history:
            return {}
        
        # Calculate metrics from prediction history
        actuals = []
        predictions = []
        
        for record in self.prediction_history:
            if 'actual' in record and 'ensemble' in record:
                actuals.append(record['actual'])
                predictions.append(record['ensemble'])
        
        if not actuals:
            return {}
        
        # Calculate ensemble metrics
        actuals_array = np.array(actuals)
        predictions_array = np.array(predictions)
        
        ensemble_metrics = ModelEvaluator.calculate_detailed_metrics(
            actuals_array.flatten(), 
            predictions_array.flatten()
        )
        
        # Add model-specific metrics
        model_metrics = {}
        for model_name in self.models.keys():
            model_actuals = []
            model_preds = []
            
            for record in self.prediction_history:
                if ('actual' in record and 'predictions' in record and 
                    model_name in record['predictions']):
                    model_actuals.append(record['actual'])
                    model_preds.append(record['predictions'][model_name])
            
            if model_actuals:
                model_metrics[model_name] = ModelEvaluator.calculate_detailed_metrics(
                    np.array(model_actuals).flatten(),
                    np.array(model_preds).flatten()
                )
        
        return {
            'ensemble_metrics': ensemble_metrics,
            'model_metrics': model_metrics,
            'model_weights': self.model_weights,
            'prediction_count': len(self.prediction_history)
        }
    
    def save_ensemble(self, filepath: str) -> None:
        """Save ensemble configuration and state."""
        try:
            import pickle
            
            ensemble_data = {
                'config': self.config,
                'model_weights': self.model_weights,
                'model_performance_history': self.model_performance_history,
                'prediction_history': self.prediction_history[-100:],  # Keep recent history
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            logger.info(f"Ensemble saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save ensemble: {e}")
            raise
    
    def load_ensemble(self, filepath: str) -> None:
        """Load ensemble configuration and state."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            self.config = ensemble_data['config']
            self.model_weights = ensemble_data['model_weights']
            self.model_performance_history = ensemble_data['model_performance_history']
            self.prediction_history = ensemble_data['prediction_history']
            self.is_trained = ensemble_data['is_trained']
            
            logger.info(f"Ensemble loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            raise