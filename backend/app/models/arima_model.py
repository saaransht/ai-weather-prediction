"""
ARIMA/SARIMA model for weather time series prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
from loguru import logger

# Statistical modeling libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools

from app.models.base import WeatherModel, ModelInfo, ModelException, ModelTrainingException, ModelPredictionException
from app.ml.model_utils import ModelEvaluator
from app.core.config import settings

# Suppress statsmodels warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ARIMAWeatherModel(WeatherModel):
    """ARIMA/SARIMA-based weather prediction model."""
    
    def __init__(self, name: str = "ARIMA", config: Dict[str, Any] = None):
        default_config = {
            'target_column': 'temperature',  # Single target for ARIMA
            'seasonal': True,  # Use SARIMA if True
            'seasonal_periods': 24,  # Daily seasonality for hourly data
            'auto_arima': True,  # Automatic parameter selection
            'max_p': 5,  # Maximum AR order
            'max_d': 2,  # Maximum differencing order
            'max_q': 5,  # Maximum MA order
            'max_P': 2,  # Maximum seasonal AR order
            'max_D': 1,  # Maximum seasonal differencing order
            'max_Q': 2,  # Maximum seasonal MA order
            'information_criterion': 'aic',  # 'aic', 'bic', 'hqic'
            'stepwise': True,  # Stepwise search for auto ARIMA
            'suppress_warnings': True,
            'error_action': 'ignore',
            'seasonal_test': 'ocsb',  # Test for seasonality
            'test': 'adf'  # Stationarity test
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.models: Dict[str, Any] = {}  # Store fitted models for each target
        self.model_results: Dict[str, Any] = {}
        self.target_columns: List[str] = []
        self.preprocessing_info: Dict[str, Any] = {}
        
        logger.info("ARIMA model initialized")
    
    def _check_stationarity(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Check if time series is stationary using ADF and KPSS tests.
        
        Args:
            series: Time series data
            alpha: Significance level
            
        Returns:
            Dictionary with stationarity test results
        """
        # Remove NaN values
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            return {
                'is_stationary': False,
                'adf_statistic': None,
                'adf_pvalue': None,
                'kpss_statistic': None,
                'kpss_pvalue': None,
                'suggested_d': 1
            }
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(series_clean, autolag='AIC')
            adf_stationary = adf_result[1] <= alpha
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            adf_result = (None, 1.0, None, None, None, None)
            adf_stationary = False
        
        # KPSS test
        try:
            kpss_result = kpss(series_clean, regression='c', nlags='auto')
            kpss_stationary = kpss_result[1] >= alpha
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
            kpss_result = (None, 0.0, None, None)
            kpss_stationary = False
        
        # Determine stationarity
        is_stationary = adf_stationary and kpss_stationary
        
        # Suggest differencing order
        suggested_d = 0 if is_stationary else 1
        
        return {
            'is_stationary': is_stationary,
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'suggested_d': suggested_d
        }
    
    def _detect_seasonality(self, series: pd.Series, periods: List[int] = None) -> Dict[str, Any]:
        """
        Detect seasonality in time series.
        
        Args:
            series: Time series data
            periods: List of potential seasonal periods to test
            
        Returns:
            Dictionary with seasonality information
        """
        if periods is None:
            periods = [24, 168, 8760]  # Hourly: daily, weekly, yearly
        
        series_clean = series.dropna()
        
        if len(series_clean) < max(periods) * 2:
            return {
                'has_seasonality': False,
                'seasonal_period': None,
                'seasonal_strength': 0.0
            }
        
        best_period = None
        best_strength = 0.0
        
        for period in periods:
            if len(series_clean) >= period * 2:
                try:
                    # Seasonal decomposition
                    decomposition = seasonal_decompose(
                        series_clean, 
                        model='additive', 
                        period=period,
                        extrapolate_trend='freq'
                    )
                    
                    # Calculate seasonal strength
                    seasonal_var = np.var(decomposition.seasonal.dropna())
                    residual_var = np.var(decomposition.resid.dropna())
                    
                    if residual_var > 0:
                        strength = seasonal_var / (seasonal_var + residual_var)
                        
                        if strength > best_strength:
                            best_strength = strength
                            best_period = period
                
                except Exception as e:
                    logger.warning(f"Seasonality detection failed for period {period}: {e}")
                    continue
        
        has_seasonality = best_strength > 0.1  # Threshold for significant seasonality
        
        return {
            'has_seasonality': has_seasonality,
            'seasonal_period': best_period if has_seasonality else None,
            'seasonal_strength': best_strength
        }
    
    def _auto_arima_search(self, series: pd.Series, seasonal_info: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
        """
        Automatic ARIMA parameter search using grid search.
        
        Args:
            series: Time series data
            seasonal_info: Seasonality information
            
        Returns:
            Tuple of (p, d, q, P, D, Q) parameters
        """
        series_clean = series.dropna()
        
        # Parameter ranges
        p_range = range(0, self.config['max_p'] + 1)
        d_range = range(0, self.config['max_d'] + 1)
        q_range = range(0, self.config['max_q'] + 1)
        
        if seasonal_info['has_seasonality']:
            P_range = range(0, self.config['max_P'] + 1)
            D_range = range(0, self.config['max_D'] + 1)
            Q_range = range(0, self.config['max_Q'] + 1)
            seasonal_period = seasonal_info['seasonal_period']
        else:
            P_range = [0]
            D_range = [0]
            Q_range = [0]
            seasonal_period = None
        
        best_aic = float('inf')
        best_params = (1, 1, 1, 0, 0, 0)
        
        # Grid search
        param_combinations = itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range)
        
        for p, d, q, P, D, Q in param_combinations:
            try:
                if seasonal_period:
                    model = SARIMAX(
                        series_clean,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                else:
                    model = ARIMA(
                        series_clean,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                
                fitted_model = model.fit(disp=False)
                
                # Use specified information criterion
                if self.config['information_criterion'] == 'aic':
                    ic_value = fitted_model.aic
                elif self.config['information_criterion'] == 'bic':
                    ic_value = fitted_model.bic
                else:  # hqic
                    ic_value = fitted_model.hqic
                
                if ic_value < best_aic:
                    best_aic = ic_value
                    best_params = (p, d, q, P, D, Q)
            
            except Exception:
                continue
        
        logger.info(f"Best ARIMA parameters: {best_params} with {self.config['information_criterion'].upper()}={best_aic:.2f}")
        return best_params
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train ARIMA/SARIMA models on weather data.
        
        Args:
            data: DataFrame with time series weather data
        """
        try:
            logger.info("Starting ARIMA model training")
            
            # Determine target columns
            if isinstance(self.config['target_column'], str):
                self.target_columns = [self.config['target_column']]
            else:
                self.target_columns = self.config['target_column']
            
            # Ensure timestamp is datetime and set as index
            if 'timestamp' in data.columns:
                data = data.copy()
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp').sort_index()
            
            # Train model for each target column
            for target_col in self.target_columns:
                if target_col not in data.columns:
                    logger.warning(f"Target column {target_col} not found in data")
                    continue
                
                logger.info(f"Training ARIMA model for {target_col}")
                
                # Extract time series
                series = data[target_col].copy()
                
                # Handle missing values
                series = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                
                if series.isnull().sum() > 0:
                    logger.warning(f"Still have {series.isnull().sum()} missing values after interpolation")
                    series = series.dropna()
                
                if len(series) < 50:
                    raise ModelTrainingException(f"Insufficient data for {target_col}: {len(series)} points")
                
                # Check stationarity
                stationarity_info = self._check_stationarity(series)
                
                # Detect seasonality
                seasonality_info = self._detect_seasonality(series)
                
                # Store preprocessing info
                self.preprocessing_info[target_col] = {
                    'stationarity': stationarity_info,
                    'seasonality': seasonality_info,
                    'series_length': len(series),
                    'series_mean': float(series.mean()),
                    'series_std': float(series.std())
                }
                
                # Determine model parameters
                if self.config['auto_arima']:
                    p, d, q, P, D, Q = self._auto_arima_search(series, seasonality_info)
                else:
                    # Use default parameters
                    p, d, q = 1, 1, 1
                    P, D, Q = (1, 1, 1) if seasonality_info['has_seasonality'] else (0, 0, 0)
                
                # Fit model
                try:
                    if seasonality_info['has_seasonality'] and self.config['seasonal']:
                        seasonal_period = seasonality_info['seasonal_period']
                        model = SARIMAX(
                            series,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, seasonal_period),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        model_type = f"SARIMA({p},{d},{q})x({P},{D},{Q},{seasonal_period})"
                    else:
                        model = ARIMA(
                            series,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        model_type = f"ARIMA({p},{d},{q})"
                    
                    fitted_model = model.fit(disp=False)
                    
                    # Store fitted model
                    self.models[target_col] = fitted_model
                    self.model_results[target_col] = {
                        'model_type': model_type,
                        'parameters': (p, d, q, P, D, Q),
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'hqic': fitted_model.hqic,
                        'seasonal_period': seasonality_info.get('seasonal_period')
                    }
                    
                    logger.info(f"Fitted {model_type} for {target_col} with AIC={fitted_model.aic:.2f}")
                
                except Exception as e:
                    logger.error(f"Failed to fit ARIMA model for {target_col}: {e}")
                    raise ModelTrainingException(f"ARIMA fitting failed for {target_col}: {str(e)}")
            
            # Calculate overall metrics using in-sample predictions
            self._calculate_training_metrics(data)
            
            self.is_trained = True
            
            # Create model info
            self.model_info = ModelInfo(
                name=self.name,
                model_type="ARIMA/SARIMA",
                version="1.0",
                training_window_days=settings.training_data_days,
                feature_columns=['timestamp'],
                target_columns=self.target_columns,
                hyperparameters=self.config,
                created_at=datetime.utcnow(),
                last_trained=datetime.utcnow()
            )
            
            logger.info(f"ARIMA training completed for {len(self.models)} targets")
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            raise ModelTrainingException(f"ARIMA training failed: {str(e)}")
    
    def _calculate_training_metrics(self, data: pd.DataFrame) -> None:
        """Calculate training metrics using in-sample predictions."""
        all_true = []
        all_pred = []
        
        for target_col in self.target_columns:
            if target_col in self.models:
                model = self.models[target_col]
                series = data[target_col].dropna()
                
                # Get in-sample predictions
                try:
                    predictions = model.fittedvalues
                    
                    # Align predictions with actual values
                    aligned_true = series.loc[predictions.index]
                    
                    all_true.extend(aligned_true.values)
                    all_pred.extend(predictions.values)
                
                except Exception as e:
                    logger.warning(f"Could not calculate metrics for {target_col}: {e}")
        
        if all_true and all_pred:
            self.metrics = ModelEvaluator.calculate_metrics(
                np.array(all_true), 
                np.array(all_pred)
            )
        else:
            from app.models.base import ModelMetrics
            self.metrics = ModelMetrics(
                mae=float('inf'),
                rmse=float('inf'),
                mape=float('inf'),
                last_updated=datetime.utcnow()
            )
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions using ARIMA models.
        
        Args:
            features: Not used for ARIMA (uses internal state)
            
        Returns:
            Array of predictions for each target
        """
        if not self.is_trained or not self.models:
            raise ModelPredictionException("Model not trained")
        
        try:
            predictions = []
            
            for target_col in self.target_columns:
                if target_col in self.models:
                    model = self.models[target_col]
                    
                    # Forecast one step ahead
                    forecast = model.forecast(steps=1)
                    predictions.append(forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0])
                else:
                    predictions.append(0.0)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            raise ModelPredictionException(f"ARIMA prediction failed: {str(e)}")
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            features: Not used for ARIMA
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained or not self.models:
            raise ModelPredictionException("Model not trained")
        
        try:
            predictions = []
            lower_bounds = []
            upper_bounds = []
            
            for target_col in self.target_columns:
                if target_col in self.models:
                    model = self.models[target_col]
                    
                    # Get forecast with confidence intervals
                    forecast = model.get_forecast(steps=1)
                    pred_mean = forecast.predicted_mean.iloc[0]
                    conf_int = forecast.conf_int()
                    
                    predictions.append(pred_mean)
                    lower_bounds.append(conf_int.iloc[0, 0])
                    upper_bounds.append(conf_int.iloc[0, 1])
                else:
                    predictions.append(0.0)
                    lower_bounds.append(0.0)
                    upper_bounds.append(0.0)
            
            return np.array(predictions), np.array(lower_bounds), np.array(upper_bounds)
            
        except Exception as e:
            logger.error(f"ARIMA uncertainty prediction failed: {e}")
            raise ModelPredictionException(f"ARIMA uncertainty prediction failed: {str(e)}")
    
    def predict_sequence(self, steps: int = 24) -> np.ndarray:
        """
        Generate multi-step forecasts.
        
        Args:
            steps: Number of future steps to predict
            
        Returns:
            Array of predictions for each step and target
        """
        if not self.is_trained or not self.models:
            raise ModelPredictionException("Model not trained")
        
        try:
            all_forecasts = []
            
            for target_col in self.target_columns:
                if target_col in self.models:
                    model = self.models[target_col]
                    
                    # Multi-step forecast
                    forecast = model.forecast(steps=steps)
                    all_forecasts.append(forecast.values if hasattr(forecast, 'values') else forecast)
                else:
                    all_forecasts.append(np.zeros(steps))
            
            # Transpose to get shape (steps, n_targets)
            return np.array(all_forecasts).T
            
        except Exception as e:
            logger.error(f"ARIMA sequence prediction failed: {e}")
            raise ModelPredictionException(f"ARIMA sequence prediction failed: {str(e)}")
    
    def save_model(self, filepath: str) -> None:
        """Save ARIMA models to disk."""
        if not self.models:
            raise ModelException("No models to save")
        
        try:
            import pickle
            
            model_data = {
                'models': self.models,
                'model_results': self.model_results,
                'target_columns': self.target_columns,
                'preprocessing_info': self.preprocessing_info,
                'config': self.config,
                'model_info': self.model_info.dict() if self.model_info else None,
                'metrics': self.metrics.__dict__ if self.metrics else None,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"ARIMA models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save ARIMA models: {e}")
            raise ModelException(f"Failed to save models: {str(e)}")
    
    def load_model(self, filepath: str) -> None:
        """Load ARIMA models from disk."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.model_results = model_data['model_results']
            self.target_columns = model_data['target_columns']
            self.preprocessing_info = model_data['preprocessing_info']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            
            # Load metadata
            if model_data.get('model_info'):
                self.model_info = ModelInfo(**model_data['model_info'])
            
            if model_data.get('metrics'):
                from app.models.base import ModelMetrics
                self.metrics = ModelMetrics(**model_data['metrics'])
            
            logger.info(f"ARIMA models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load ARIMA models: {e}")
            raise ModelException(f"Failed to load models: {str(e)}")
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for fitted models."""
        diagnostics = {}
        
        for target_col, model in self.models.items():
            try:
                # Residual diagnostics
                residuals = model.resid
                
                # Ljung-Box test for residual autocorrelation
                lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
                
                diagnostics[target_col] = {
                    'model_summary': str(model.summary()),
                    'residual_mean': float(residuals.mean()),
                    'residual_std': float(residuals.std()),
                    'ljung_box_pvalue': float(lb_test['lb_pvalue'].iloc[-1]),
                    'preprocessing_info': self.preprocessing_info.get(target_col, {}),
                    'model_results': self.model_results.get(target_col, {})
                }
            
            except Exception as e:
                logger.warning(f"Could not generate diagnostics for {target_col}: {e}")
                diagnostics[target_col] = {'error': str(e)}
        
        return diagnostics