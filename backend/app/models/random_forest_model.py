"""
Random Forest model for weather prediction with advanced feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from loguru import logger

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.inspection import permutation_importance
import joblib

from app.models.base import WeatherModel, ModelInfo, ModelException, ModelTrainingException, ModelPredictionException
from app.ml.model_utils import ModelEvaluator, FeatureSelector
from app.core.config import settings


class RandomForestWeatherModel(WeatherModel):
    """Random Forest-based weather prediction model with feature engineering."""
    
    def __init__(self, name: str = "RandomForest", config: Dict[str, Any] = None):
        default_config = {
            'n_estimators': settings.random_forest_n_estimators,
            'max_depth': settings.random_forest_max_depth,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,
            'target_columns': ['temperature', 'humidity', 'pressure', 'wind_speed'],
            'feature_selection': True,
            'max_features_select': 50,
            'hyperparameter_tuning': True,
            'cv_folds': 5,
            'feature_importance_threshold': 0.001
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.model: Optional[MultiOutputRegressor] = None
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.best_params: Dict[str, Any] = {}
        
        logger.info("Random Forest model initialized")
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced engineered features for Random Forest.
        
        Args:
            df: DataFrame with basic weather features
            
        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()
        
        # Interaction features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
        
        if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
            # Wind components
            df['wind_u'] = df['wind_speed'] * np.cos(np.radians(df['wind_direction']))
            df['wind_v'] = df['wind_speed'] * np.sin(np.radians(df['wind_direction']))
            
        if 'pressure' in df.columns and 'temperature' in df.columns:
            df['pressure_temp_ratio'] = df['pressure'] / (df['temperature'] + 273.15)  # Kelvin
        
        # Polynomial features for key variables
        for col in ['temperature', 'pressure', 'wind_speed']:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_cubed'] = df[col] ** 3
        
        # Binned categorical features
        if 'temperature' in df.columns:
            df['temp_category'] = pd.cut(
                df['temperature'], 
                bins=[-50, 0, 10, 20, 30, 50], 
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        if 'wind_speed' in df.columns:
            df['wind_category'] = pd.cut(
                df['wind_speed'],
                bins=[0, 2, 5, 10, 15, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        if 'pressure' in df.columns:
            df['pressure_category'] = pd.cut(
                df['pressure'],
                bins=[0, 1000, 1013, 1025, 1100],
                labels=[0, 1, 2, 3]
            ).astype(float)
        
        # Weather pattern indicators
        if 'cloud_cover' in df.columns and 'precipitation' in df.columns:
            df['rain_probability'] = (df['cloud_cover'] > 70) & (df['precipitation'] > 0)
            df['clear_sky'] = df['cloud_cover'] < 20
            df['overcast'] = df['cloud_cover'] > 80
        
        # Comfort indices
        if 'temperature' in df.columns and 'humidity' in df.columns:
            # Simplified comfort index
            df['comfort_index'] = (
                (df['temperature'] >= 18) & (df['temperature'] <= 24) &
                (df['humidity'] >= 40) & (df['humidity'] <= 60)
            ).astype(float)
        
        # Extreme weather indicators
        if 'temperature' in df.columns:
            df['extreme_cold'] = (df['temperature'] < -10).astype(float)
            df['extreme_hot'] = (df['temperature'] > 35).astype(float)
        
        if 'wind_speed' in df.columns:
            df['high_wind'] = (df['wind_speed'] > 10).astype(float)
        
        # Pressure tendency indicators (if we have historical data)
        if 'pressure_tendency' in df.columns:
            df['pressure_rising'] = (df['pressure_tendency'] > 0.5).astype(float)
            df['pressure_falling'] = (df['pressure_tendency'] < -0.5).astype(float)
        
        return df
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Select most important features using multiple methods.
        
        Args:
            X: Feature matrix
            y: Target matrix
            feature_names: List of feature names
            
        Returns:
            Tuple of (selected_X, selected_feature_names)
        """
        if not self.config['feature_selection']:
            return X, feature_names
        
        logger.info(f"Starting feature selection from {len(feature_names)} features")
        
        # Method 1: Random Forest feature importance
        rf_selector = RandomForestRegressor(
            n_estimators=50,
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        
        # For multi-output, use the first target for feature selection
        y_single = y[:, 0] if y.ndim > 1 else y
        rf_selector.fit(X, y_single)
        
        # Get feature importance
        importance_scores = rf_selector.feature_importances_
        
        # Select features above threshold
        important_mask = importance_scores > self.config['feature_importance_threshold']
        
        # Ensure we don't select too many features
        if np.sum(important_mask) > self.config['max_features_select']:
            # Select top N features by importance
            top_indices = np.argsort(importance_scores)[::-1][:self.config['max_features_select']]
            important_mask = np.zeros(len(feature_names), dtype=bool)
            important_mask[top_indices] = True
        
        # Ensure we have at least some features
        if np.sum(important_mask) < 5:
            top_indices = np.argsort(importance_scores)[::-1][:10]
            important_mask = np.zeros(len(feature_names), dtype=bool)
            important_mask[top_indices] = True
        
        selected_X = X[:, important_mask]
        selected_features = [feature_names[i] for i in range(len(feature_names)) if important_mask[i]]
        
        logger.info(f"Selected {len(selected_features)} features using Random Forest importance")
        
        return selected_X, selected_features
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search with time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target matrix
            
        Returns:
            Dictionary of best parameters
        """
        if not self.config['hyperparameter_tuning']:
            return {
                'n_estimators': self.config['n_estimators'],
                'max_depth': self.config['max_depth'],
                'min_samples_split': self.config['min_samples_split'],
                'min_samples_leaf': self.config['min_samples_leaf'],
                'max_features': self.config['max_features']
            }
        
        logger.info("Starting hyperparameter tuning")
        
        # Parameter grid
        param_grid = {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__max_depth': [10, 20, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__max_features': ['sqrt', 'log2', None]
        }
        
        # Base model
        base_rf = RandomForestRegressor(
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs'],
            bootstrap=self.config['bootstrap']
        )
        
        # Multi-output wrapper
        multi_rf = MultiOutputRegressor(base_rf)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        
        # Grid search
        grid_search = GridSearchCV(
            multi_rf,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,  # Avoid nested parallelism
            verbose=0
        )
        
        # For hyperparameter tuning, use a subset of data if it's too large
        if len(X) > 5000:
            indices = np.random.choice(len(X), 5000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        try:
            grid_search.fit(X_sample, y_sample)
            best_params = grid_search.best_params_
            
            # Remove 'estimator__' prefix
            clean_params = {k.replace('estimator__', ''): v for k, v in best_params.items()}
            
            logger.info(f"Best parameters found: {clean_params}")
            return clean_params
            
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed: {e}. Using default parameters.")
            return {
                'n_estimators': self.config['n_estimators'],
                'max_depth': self.config['max_depth'],
                'min_samples_split': self.config['min_samples_split'],
                'min_samples_leaf': self.config['min_samples_leaf'],
                'max_features': self.config['max_features']
            }
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the Random Forest model on weather data.
        
        Args:
            data: DataFrame with preprocessed weather features and targets
        """
        try:
            logger.info("Starting Random Forest model training")
            
            # Prepare target columns
            target_columns = self.config['target_columns']
            self.target_names = [col for col in target_columns if col in data.columns]
            
            if not self.target_names:
                raise ModelTrainingException("No valid target columns found in data")
            
            # Create advanced features
            df_engineered = self._create_advanced_features(data)
            
            # Prepare feature columns (exclude targets and timestamp)
            exclude_cols = self.target_names + ['timestamp']
            feature_columns = [col for col in df_engineered.columns if col not in exclude_cols]
            self.feature_names = feature_columns
            
            # Extract features and targets
            X = df_engineered[feature_columns].values.astype(np.float32)
            y = df_engineered[self.target_names].values.astype(np.float32)
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
            
            # Feature selection
            X_selected, selected_features = self._select_features(X, y, feature_columns)
            self.selected_features = selected_features
            
            # Hyperparameter tuning
            best_params = self._tune_hyperparameters(X_selected, y)
            self.best_params = best_params
            
            # Train final model with best parameters
            base_rf = RandomForestRegressor(
                **best_params,
                random_state=self.config['random_state'],
                bootstrap=self.config['bootstrap'],
                n_jobs=self.config['n_jobs']
            )
            
            self.model = MultiOutputRegressor(base_rf)
            self.model.fit(X_selected, y)
            
            # Calculate feature importance
            self._calculate_feature_importance(X_selected)
            
            # Calculate training metrics
            y_pred = self.model.predict(X_selected)
            self.metrics = ModelEvaluator.calculate_metrics(y.flatten(), y_pred.flatten())
            
            self.is_trained = True
            
            # Create model info
            self.model_info = ModelInfo(
                name=self.name,
                model_type="RandomForest",
                version="1.0",
                training_window_days=settings.training_data_days,
                feature_columns=self.selected_features,
                target_columns=self.target_names,
                hyperparameters={**self.config, **self.best_params},
                created_at=datetime.utcnow(),
                last_trained=datetime.utcnow()
            )
            
            logger.info(f"Random Forest training completed. MAE: {self.metrics.mae:.4f}, RMSE: {self.metrics.rmse:.4f}")
            logger.info(f"Used {len(self.selected_features)} features out of {len(self.feature_names)} original features")
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            raise ModelTrainingException(f"Random Forest training failed: {str(e)}")
    
    def _calculate_feature_importance(self, X: np.ndarray) -> None:
        """Calculate and store feature importance scores."""
        try:
            # Get feature importance from the trained model
            # For MultiOutputRegressor, average importance across all estimators
            importances = []
            for estimator in self.model.estimators_:
                importances.append(estimator.feature_importances_)
            
            avg_importance = np.mean(importances, axis=0)
            
            # Create feature importance dictionary
            self.feature_importance = {
                feature: float(importance) 
                for feature, importance in zip(self.selected_features, avg_importance)
            }
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            logger.info(f"Top 5 important features: {list(self.feature_importance.keys())[:5]}")
            
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            self.feature_importance = {}
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained Random Forest model.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ModelPredictionException("Model not trained")
        
        if not self.validate_input(features):
            raise ModelPredictionException("Invalid input features")
        
        try:
            # Handle missing values
            features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Select features used during training
            if len(self.selected_features) != features_clean.shape[1]:
                # If feature dimensions don't match, try to map by feature names
                logger.warning(f"Feature dimension mismatch: expected {len(self.selected_features)}, got {features_clean.shape[1]}")
                # Use available features or pad with zeros
                if features_clean.shape[1] < len(self.selected_features):
                    padding = np.zeros((features_clean.shape[0], len(self.selected_features) - features_clean.shape[1]))
                    features_clean = np.hstack([features_clean, padding])
                else:
                    features_clean = features_clean[:, :len(self.selected_features)]
            
            # Predict
            predictions = self.model.predict(features_clean)
            
            # Return single prediction if input is single sample
            if predictions.shape[0] == 1:
                return predictions[0]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Random Forest prediction failed: {e}")
            raise ModelPredictionException(f"Random Forest prediction failed: {str(e)}")
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates using ensemble variance.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained or self.model is None:
            raise ModelPredictionException("Model not trained")
        
        try:
            # Handle missing values
            features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Adjust feature dimensions if needed
            if features_clean.shape[1] != len(self.selected_features):
                if features_clean.shape[1] < len(self.selected_features):
                    padding = np.zeros((features_clean.shape[0], len(self.selected_features) - features_clean.shape[1]))
                    features_clean = np.hstack([features_clean, padding])
                else:
                    features_clean = features_clean[:, :len(self.selected_features)]
            
            # Get predictions from individual trees
            all_predictions = []
            
            for estimator in self.model.estimators_:
                tree_predictions = []
                for tree in estimator.estimators_:
                    pred = tree.predict(features_clean)
                    tree_predictions.append(pred)
                
                # Average predictions for this target
                avg_pred = np.mean(tree_predictions, axis=0)
                all_predictions.append(avg_pred)
            
            # Transpose to get shape (n_samples, n_targets, n_estimators)
            all_predictions = np.array(all_predictions).T
            
            # Calculate statistics
            mean_pred = np.mean(all_predictions, axis=-1)
            std_pred = np.std(all_predictions, axis=-1)
            
            # Calculate confidence intervals (95%)
            lower_bounds = mean_pred - 1.96 * std_pred
            upper_bounds = mean_pred + 1.96 * std_pred
            
            # Return single prediction if input is single sample
            if mean_pred.shape[0] == 1:
                return mean_pred[0], lower_bounds[0], upper_bounds[0]
            
            return mean_pred, lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Random Forest uncertainty prediction failed: {e}")
            # Fallback to regular prediction with zero uncertainty
            pred = self.predict(features)
            return pred, pred * 0.95, pred * 1.05
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance.copy()
    
    def save_model(self, filepath: str) -> None:
        """Save the Random Forest model to disk."""
        if self.model is None:
            raise ModelException("No model to save")
        
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance,
                'best_params': self.best_params,
                'config': self.config,
                'model_info': self.model_info.dict() if self.model_info else None,
                'metrics': self.metrics.__dict__ if self.metrics else None,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Random Forest model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save Random Forest model: {e}")
            raise ModelException(f"Failed to save model: {str(e)}")
    
    def load_model(self, filepath: str) -> None:
        """Load the Random Forest model from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.target_names = model_data['target_names']
            self.selected_features = model_data['selected_features']
            self.feature_importance = model_data['feature_importance']
            self.best_params = model_data['best_params']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            
            # Load metadata
            if model_data.get('model_info'):
                self.model_info = ModelInfo(**model_data['model_info'])
            
            if model_data.get('metrics'):
                from app.models.base import ModelMetrics
                self.metrics = ModelMetrics(**model_data['metrics'])
            
            logger.info(f"Random Forest model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load Random Forest model: {e}")
            raise ModelException(f"Failed to load model: {str(e)}")