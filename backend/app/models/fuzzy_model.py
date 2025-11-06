"""
Fuzzy Time Series model for weather prediction using fuzzy logic and FLRGs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
from collections import defaultdict, Counter
from loguru import logger

from app.models.base import WeatherModel, ModelInfo, ModelException, ModelTrainingException, ModelPredictionException
from app.ml.model_utils import ModelEvaluator
from app.core.config import settings


class FuzzySet:
    """Represents a fuzzy set with membership function."""
    
    def __init__(self, name: str, center: float, width: float, universe_min: float, universe_max: float):
        self.name = name
        self.center = center
        self.width = width
        self.universe_min = universe_min
        self.universe_max = universe_max
    
    def membership(self, value: float) -> float:
        """
        Calculate membership degree using triangular membership function.
        
        Args:
            value: Input value
            
        Returns:
            Membership degree [0, 1]
        """
        if self.width == 0:
            return 1.0 if value == self.center else 0.0
        
        # Triangular membership function
        distance = abs(value - self.center)
        if distance <= self.width:
            return 1.0 - (distance / self.width)
        else:
            return 0.0
    
    def __str__(self):
        return f"FuzzySet({self.name}, center={self.center:.2f}, width={self.width:.2f})"


class FuzzyLogicalRelationshipGroup:
    """Represents a Fuzzy Logical Relationship Group (FLRG)."""
    
    def __init__(self, antecedent: str):
        self.antecedent = antecedent  # Left-hand side
        self.consequents: List[str] = []  # Right-hand side
        self.weights: Dict[str, float] = {}  # Weights for each consequent
    
    def add_consequent(self, consequent: str, weight: float = 1.0):
        """Add a consequent with weight."""
        if consequent not in self.consequents:
            self.consequents.append(consequent)
            self.weights[consequent] = weight
        else:
            self.weights[consequent] += weight
    
    def get_weighted_consequents(self) -> Dict[str, float]:
        """Get normalized weights for consequents."""
        if not self.consequents:
            return {}
        
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return {c: 1.0 / len(self.consequents) for c in self.consequents}
        
        return {c: self.weights[c] / total_weight for c in self.consequents}
    
    def __str__(self):
        return f"FLRG({self.antecedent} -> {self.consequents})"


class FuzzyTimeSeriesModel(WeatherModel):
    """Fuzzy Time Series model for weather prediction."""
    
    def __init__(self, name: str = "FuzzyTimeSeries", config: Dict[str, Any] = None):
        default_config = {
            'target_column': 'temperature',  # Single target for fuzzy model
            'num_fuzzy_sets': 7,  # Number of fuzzy sets to create
            'overlap_factor': 0.5,  # Overlap between adjacent fuzzy sets
            'order': 1,  # Order of the fuzzy time series (1 = first-order)
            'defuzzification_method': 'centroid',  # 'centroid', 'weighted_average'
            'min_support': 2,  # Minimum support for FLRG creation
            'smoothing_factor': 0.1,  # For exponential smoothing
            'linguistic_terms': ['VeryLow', 'Low', 'MediumLow', 'Medium', 'MediumHigh', 'High', 'VeryHigh']
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.fuzzy_sets: Dict[str, FuzzySet] = {}
        self.flrgs: Dict[str, FuzzyLogicalRelationshipGroup] = {}
        self.universe_min: float = 0.0
        self.universe_max: float = 100.0
        self.target_column: str = ""
        self.fuzzified_series: List[str] = []
        self.original_series: List[float] = []
        
        logger.info("Fuzzy Time Series model initialized")
    
    def _determine_universe(self, data: pd.Series) -> Tuple[float, float]:
        """
        Determine the universe of discourse for the fuzzy sets.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (min_value, max_value) with some padding
        """
        data_min = data.min()
        data_max = data.max()
        
        # Add padding (10% on each side)
        padding = (data_max - data_min) * 0.1
        universe_min = data_min - padding
        universe_max = data_max + padding
        
        return universe_min, universe_max
    
    def _create_fuzzy_sets(self, universe_min: float, universe_max: float) -> Dict[str, FuzzySet]:
        """
        Create fuzzy sets with triangular membership functions.
        
        Args:
            universe_min: Minimum value of universe
            universe_max: Maximum value of universe
            
        Returns:
            Dictionary of fuzzy sets
        """
        num_sets = self.config['num_fuzzy_sets']
        linguistic_terms = self.config['linguistic_terms']
        
        # Ensure we have enough linguistic terms
        if len(linguistic_terms) < num_sets:
            linguistic_terms = [f"A{i+1}" for i in range(num_sets)]
        
        fuzzy_sets = {}
        
        # Calculate centers and widths
        range_size = universe_max - universe_min
        step = range_size / (num_sets - 1) if num_sets > 1 else range_size
        
        for i in range(num_sets):
            center = universe_min + i * step
            width = step * (1 + self.config['overlap_factor'])
            
            fuzzy_set = FuzzySet(
                name=linguistic_terms[i],
                center=center,
                width=width,
                universe_min=universe_min,
                universe_max=universe_max
            )
            
            fuzzy_sets[linguistic_terms[i]] = fuzzy_set
        
        logger.info(f"Created {len(fuzzy_sets)} fuzzy sets")
        return fuzzy_sets
    
    def _fuzzify_value(self, value: float) -> str:
        """
        Fuzzify a crisp value to the fuzzy set with highest membership.
        
        Args:
            value: Crisp value to fuzzify
            
        Returns:
            Name of fuzzy set with highest membership
        """
        max_membership = 0.0
        best_set = list(self.fuzzy_sets.keys())[0]  # Default to first set
        
        for set_name, fuzzy_set in self.fuzzy_sets.items():
            membership = fuzzy_set.membership(value)
            if membership > max_membership:
                max_membership = membership
                best_set = set_name
        
        return best_set
    
    def _fuzzify_series(self, data: pd.Series) -> List[str]:
        """
        Fuzzify the entire time series.
        
        Args:
            data: Time series data
            
        Returns:
            List of fuzzy set names
        """
        fuzzified = []
        for value in data:
            if pd.isna(value):
                # Handle missing values by using the previous fuzzy set
                if fuzzified:
                    fuzzified.append(fuzzified[-1])
                else:
                    fuzzified.append(list(self.fuzzy_sets.keys())[0])
            else:
                fuzzified.append(self._fuzzify_value(value))
        
        return fuzzified
    
    def _create_flrgs(self, fuzzified_series: List[str]) -> Dict[str, FuzzyLogicalRelationshipGroup]:
        """
        Create Fuzzy Logical Relationship Groups from fuzzified series.
        
        Args:
            fuzzified_series: List of fuzzy set names
            
        Returns:
            Dictionary of FLRGs
        """
        flrgs = {}
        order = self.config['order']
        
        # Create relationships based on order
        for i in range(order, len(fuzzified_series)):
            # Create antecedent (left-hand side)
            if order == 1:
                antecedent = fuzzified_series[i - 1]
            else:
                # For higher-order models, combine multiple previous states
                antecedent = ",".join(fuzzified_series[i - order:i])
            
            # Consequent (right-hand side)
            consequent = fuzzified_series[i]
            
            # Create or update FLRG
            if antecedent not in flrgs:
                flrgs[antecedent] = FuzzyLogicalRelationshipGroup(antecedent)
            
            flrgs[antecedent].add_consequent(consequent)
        
        # Filter FLRGs by minimum support
        min_support = self.config['min_support']
        filtered_flrgs = {}
        
        for antecedent, flrg in flrgs.items():
            total_support = sum(flrg.weights.values())
            if total_support >= min_support:
                filtered_flrgs[antecedent] = flrg
        
        logger.info(f"Created {len(filtered_flrgs)} FLRGs (filtered from {len(flrgs)})")
        return filtered_flrgs
    
    def _defuzzify(self, fuzzy_output: Dict[str, float]) -> float:
        """
        Convert fuzzy output to crisp value.
        
        Args:
            fuzzy_output: Dictionary of fuzzy set names and their weights
            
        Returns:
            Crisp output value
        """
        method = self.config['defuzzification_method']
        
        if not fuzzy_output:
            # Return middle of universe if no output
            return (self.universe_min + self.universe_max) / 2
        
        if method == 'centroid':
            # Weighted average of fuzzy set centers
            numerator = 0.0
            denominator = 0.0
            
            for set_name, weight in fuzzy_output.items():
                if set_name in self.fuzzy_sets:
                    center = self.fuzzy_sets[set_name].center
                    numerator += center * weight
                    denominator += weight
            
            return numerator / denominator if denominator > 0 else (self.universe_min + self.universe_max) / 2
        
        elif method == 'weighted_average':
            # Simple weighted average
            total_weight = sum(fuzzy_output.values())
            if total_weight == 0:
                return (self.universe_min + self.universe_max) / 2
            
            weighted_sum = 0.0
            for set_name, weight in fuzzy_output.items():
                if set_name in self.fuzzy_sets:
                    center = self.fuzzy_sets[set_name].center
                    weighted_sum += center * (weight / total_weight)
            
            return weighted_sum
        
        else:
            raise ValueError(f"Unknown defuzzification method: {method}")
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the Fuzzy Time Series model.
        
        Args:
            data: DataFrame with time series weather data
        """
        try:
            logger.info("Starting Fuzzy Time Series model training")
            
            # Get target column
            target_column = self.config['target_column']
            self.target_column = target_column
            
            if target_column not in data.columns:
                raise ModelTrainingException(f"Target column '{target_column}' not found in data")
            
            # Extract time series
            series = data[target_column].copy()
            
            # Handle missing values
            series = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            if len(series) < 10:
                raise ModelTrainingException(f"Insufficient data: {len(series)} points")
            
            # Store original series for validation
            self.original_series = series.tolist()
            
            # Determine universe of discourse
            self.universe_min, self.universe_max = self._determine_universe(series)
            
            # Create fuzzy sets
            self.fuzzy_sets = self._create_fuzzy_sets(self.universe_min, self.universe_max)
            
            # Fuzzify the series
            self.fuzzified_series = self._fuzzify_series(series)
            
            # Create FLRGs
            self.flrgs = self._create_flrgs(self.fuzzified_series)
            
            # Calculate training metrics
            self._calculate_training_metrics()
            
            self.is_trained = True
            
            # Create model info
            self.model_info = ModelInfo(
                name=self.name,
                model_type="FuzzyTimeSeries",
                version="1.0",
                training_window_days=settings.training_data_days,
                feature_columns=[target_column],
                target_columns=[target_column],
                hyperparameters=self.config,
                created_at=datetime.utcnow(),
                last_trained=datetime.utcnow()
            )
            
            logger.info(f"Fuzzy Time Series training completed. Created {len(self.fuzzy_sets)} fuzzy sets and {len(self.flrgs)} FLRGs")
            
        except Exception as e:
            logger.error(f"Fuzzy Time Series training failed: {e}")
            raise ModelTrainingException(f"Fuzzy Time Series training failed: {str(e)}")
    
    def _calculate_training_metrics(self) -> None:
        """Calculate training metrics using one-step-ahead predictions."""
        predictions = []
        actuals = []
        
        order = self.config['order']
        
        for i in range(order, len(self.original_series)):
            # Create antecedent
            if order == 1:
                antecedent = self.fuzzified_series[i - 1]
            else:
                antecedent = ",".join(self.fuzzified_series[i - order:i])
            
            # Predict
            prediction = self._predict_from_antecedent(antecedent)
            predictions.append(prediction)
            actuals.append(self.original_series[i])
        
        if predictions and actuals:
            self.metrics = ModelEvaluator.calculate_metrics(
                np.array(actuals), 
                np.array(predictions)
            )
        else:
            from app.models.base import ModelMetrics
            self.metrics = ModelMetrics(
                mae=float('inf'),
                rmse=float('inf'),
                mape=float('inf'),
                last_updated=datetime.utcnow()
            )
    
    def _predict_from_antecedent(self, antecedent: str) -> float:
        """
        Make prediction given an antecedent.
        
        Args:
            antecedent: Fuzzy antecedent state
            
        Returns:
            Predicted crisp value
        """
        if antecedent in self.flrgs:
            # Use FLRG to get fuzzy output
            flrg = self.flrgs[antecedent]
            fuzzy_output = flrg.get_weighted_consequents()
            return self._defuzzify(fuzzy_output)
        else:
            # No matching FLRG, use default prediction
            # Return the center value of the antecedent fuzzy set
            if "," in antecedent:
                # Multi-order antecedent, use the last fuzzy set
                last_set = antecedent.split(",")[-1]
            else:
                last_set = antecedent
            
            if last_set in self.fuzzy_sets:
                return self.fuzzy_sets[last_set].center
            else:
                return (self.universe_min + self.universe_max) / 2
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate prediction using Fuzzy Time Series model.
        
        Args:
            features: Recent values for creating antecedent (not used directly)
            
        Returns:
            Array with single prediction
        """
        if not self.is_trained:
            raise ModelPredictionException("Model not trained")
        
        try:
            # For fuzzy time series, we need the recent fuzzy states
            # If we have recent data, use it; otherwise use the last known state
            
            if len(self.fuzzified_series) == 0:
                raise ModelPredictionException("No historical fuzzy states available")
            
            order = self.config['order']
            
            # Create antecedent from recent fuzzy states
            if len(self.fuzzified_series) >= order:
                if order == 1:
                    antecedent = self.fuzzified_series[-1]
                else:
                    antecedent = ",".join(self.fuzzified_series[-order:])
            else:
                # Not enough history, use what we have
                antecedent = ",".join(self.fuzzified_series)
            
            # Make prediction
            prediction = self._predict_from_antecedent(antecedent)
            
            return np.array([prediction])
            
        except Exception as e:
            logger.error(f"Fuzzy Time Series prediction failed: {e}")
            raise ModelPredictionException(f"Fuzzy Time Series prediction failed: {str(e)}")
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained:
            raise ModelPredictionException("Model not trained")
        
        try:
            # Get base prediction
            prediction = self.predict(features)
            
            # Estimate uncertainty based on FLRG diversity
            order = self.config['order']
            
            if len(self.fuzzified_series) >= order:
                if order == 1:
                    antecedent = self.fuzzified_series[-1]
                else:
                    antecedent = ",".join(self.fuzzified_series[-order:])
            else:
                antecedent = ",".join(self.fuzzified_series)
            
            if antecedent in self.flrgs:
                flrg = self.flrgs[antecedent]
                consequents = flrg.get_weighted_consequents()
                
                # Calculate uncertainty based on consequent diversity
                if len(consequents) > 1:
                    # Multiple possible outcomes - higher uncertainty
                    centers = [self.fuzzy_sets[c].center for c in consequents.keys() if c in self.fuzzy_sets]
                    if centers:
                        uncertainty = np.std(centers)
                    else:
                        uncertainty = (self.universe_max - self.universe_min) * 0.1
                else:
                    # Single outcome - lower uncertainty
                    uncertainty = (self.universe_max - self.universe_min) * 0.05
            else:
                # No matching FLRG - high uncertainty
                uncertainty = (self.universe_max - self.universe_min) * 0.2
            
            # Calculate bounds (95% confidence interval approximation)
            lower_bound = prediction - 1.96 * uncertainty
            upper_bound = prediction + 1.96 * uncertainty
            
            return prediction, lower_bound, upper_bound
            
        except Exception as e:
            logger.error(f"Fuzzy Time Series uncertainty prediction failed: {e}")
            raise ModelPredictionException(f"Fuzzy Time Series uncertainty prediction failed: {str(e)}")
    
    def update_with_new_value(self, new_value: float) -> None:
        """
        Update the model with a new observed value.
        
        Args:
            new_value: New observed value to add to the series
        """
        if not self.is_trained:
            return
        
        try:
            # Fuzzify the new value
            new_fuzzy = self._fuzzify_value(new_value)
            
            # Add to series
            self.original_series.append(new_value)
            self.fuzzified_series.append(new_fuzzy)
            
            # Update FLRGs with new relationship
            order = self.config['order']
            if len(self.fuzzified_series) > order:
                if order == 1:
                    antecedent = self.fuzzified_series[-2]
                else:
                    antecedent = ",".join(self.fuzzified_series[-order-1:-1])
                
                consequent = new_fuzzy
                
                if antecedent not in self.flrgs:
                    self.flrgs[antecedent] = FuzzyLogicalRelationshipGroup(antecedent)
                
                self.flrgs[antecedent].add_consequent(consequent)
            
            # Keep series length manageable
            max_length = 1000
            if len(self.original_series) > max_length:
                self.original_series = self.original_series[-max_length:]
                self.fuzzified_series = self.fuzzified_series[-max_length:]
            
        except Exception as e:
            logger.warning(f"Failed to update fuzzy model with new value: {e}")
    
    def save_model(self, filepath: str) -> None:
        """Save the Fuzzy Time Series model to disk."""
        if not self.fuzzy_sets:
            raise ModelException("No model to save")
        
        try:
            import pickle
            
            model_data = {
                'fuzzy_sets': self.fuzzy_sets,
                'flrgs': self.flrgs,
                'universe_min': self.universe_min,
                'universe_max': self.universe_max,
                'target_column': self.target_column,
                'fuzzified_series': self.fuzzified_series,
                'original_series': self.original_series,
                'config': self.config,
                'model_info': self.model_info.dict() if self.model_info else None,
                'metrics': self.metrics.__dict__ if self.metrics else None,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Fuzzy Time Series model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save Fuzzy Time Series model: {e}")
            raise ModelException(f"Failed to save model: {str(e)}")
    
    def load_model(self, filepath: str) -> None:
        """Load the Fuzzy Time Series model from disk."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.fuzzy_sets = model_data['fuzzy_sets']
            self.flrgs = model_data['flrgs']
            self.universe_min = model_data['universe_min']
            self.universe_max = model_data['universe_max']
            self.target_column = model_data['target_column']
            self.fuzzified_series = model_data['fuzzified_series']
            self.original_series = model_data['original_series']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            
            # Load metadata
            if model_data.get('model_info'):
                self.model_info = ModelInfo(**model_data['model_info'])
            
            if model_data.get('metrics'):
                from app.models.base import ModelMetrics
                self.metrics = ModelMetrics(**model_data['metrics'])
            
            logger.info(f"Fuzzy Time Series model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load Fuzzy Time Series model: {e}")
            raise ModelException(f"Failed to load model: {str(e)}")
    
    def get_fuzzy_rules(self) -> Dict[str, Any]:
        """Get human-readable fuzzy rules."""
        rules = {}
        
        for antecedent, flrg in self.flrgs.items():
            consequents = flrg.get_weighted_consequents()
            rules[antecedent] = {
                'rule': f"IF {antecedent} THEN {list(consequents.keys())}",
                'weights': consequents,
                'support': sum(flrg.weights.values())
            }
        
        return rules