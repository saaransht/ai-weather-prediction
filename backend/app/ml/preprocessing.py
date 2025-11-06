"""
Data preprocessing utilities for weather prediction models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from loguru import logger

from app.schemas.weather import WeatherData


class WeatherDataPreprocessor:
    """Preprocessor for weather data with feature engineering."""
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_columns = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 
            'wind_direction', 'cloud_cover', 'precipitation'
        ]
        self.engineered_features = []
        self.is_fitted = False
    
    def _create_scaler(self) -> Any:
        """Create scaler based on type."""
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def weather_data_to_dataframe(self, weather_data: List[WeatherData]) -> pd.DataFrame:
        """
        Convert list of WeatherData objects to pandas DataFrame.
        
        Args:
            weather_data: List of WeatherData objects
            
        Returns:
            DataFrame with weather data
        """
        if not weather_data:
            return pd.DataFrame()
        
        data_dicts = []
        for wd in weather_data:
            data_dict = {
                'timestamp': wd.timestamp,
                'temperature': wd.temperature,
                'humidity': wd.humidity,
                'pressure': wd.pressure,
                'wind_speed': wd.wind_speed,
                'wind_direction': wd.wind_direction,
                'cloud_cover': wd.cloud_cover,
                'precipitation': wd.precipitation or 0.0,
                'latitude': wd.location.latitude,
                'longitude': wd.location.longitude
            }
            data_dicts.append(data_dict)
        
        df = pd.DataFrame(data_dicts)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['month'].apply(self._get_season)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Wind direction cyclical encoding
        df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360)
        df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360)
        
        time_features = [
            'hour', 'day_of_week', 'day_of_year', 'month', 'season',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'month_sin', 'month_cos', 'wind_dir_sin', 'wind_dir_cos'
        ]
        
        self.engineered_features.extend(time_features)
        
        return df
    
    def _get_season(self, month: int) -> int:
        """Get season from month (0=Winter, 1=Spring, 2=Summer, 3=Autumn)."""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-specific engineered features.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with additional weather features
        """
        df = df.copy()
        
        # Derived weather features
        df['feels_like_temp'] = self._calculate_feels_like_temperature(
            df['temperature'], df['humidity'], df['wind_speed']
        )
        
        df['dew_point'] = self._calculate_dew_point(df['temperature'], df['humidity'])
        
        df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])
        
        df['wind_chill'] = self._calculate_wind_chill(df['temperature'], df['wind_speed'])
        
        # Pressure tendency (requires historical data)
        if len(df) > 1:
            df['pressure_tendency'] = df['pressure'].diff().fillna(0)
        else:
            df['pressure_tendency'] = 0
        
        # Weather stability indicators
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
        df['pressure_normalized'] = (df['pressure'] - 1013.25) / 1013.25
        
        # Precipitation intensity categories
        df['precip_intensity'] = pd.cut(
            df['precipitation'], 
            bins=[-0.1, 0, 2.5, 10, 50, float('inf')],
            labels=[0, 1, 2, 3, 4]  # None, Light, Moderate, Heavy, Extreme
        ).astype(int)
        
        weather_features = [
            'feels_like_temp', 'dew_point', 'heat_index', 'wind_chill',
            'pressure_tendency', 'temp_humidity_ratio', 'pressure_normalized',
            'precip_intensity'
        ]
        
        self.engineered_features.extend(weather_features)
        
        return df
    
    def _calculate_feels_like_temperature(self, temp: pd.Series, humidity: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate feels-like temperature using heat index and wind chill."""
        feels_like = temp.copy()
        
        # Use heat index for hot conditions
        hot_mask = temp >= 27
        feels_like[hot_mask] = self._calculate_heat_index(temp[hot_mask], humidity[hot_mask])
        
        # Use wind chill for cold conditions
        cold_mask = temp <= 10
        feels_like[cold_mask] = self._calculate_wind_chill(temp[cold_mask], wind_speed[cold_mask])
        
        return feels_like
    
    def _calculate_dew_point(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate dew point temperature."""
        a = 17.27
        b = 237.7
        
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return dew_point
    
    def _calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index (feels-like temperature for hot weather)."""
        # Simplified heat index formula
        hi = temp + 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (humidity * 0.094))
        
        # More complex formula for higher temperatures
        high_temp_mask = hi >= 80
        if high_temp_mask.any():
            t = temp[high_temp_mask]
            h = humidity[high_temp_mask]
            
            hi_complex = (
                -42.379 + 2.04901523 * t + 10.14333127 * h
                - 0.22475541 * t * h - 6.83783e-3 * t**2
                - 5.481717e-2 * h**2 + 1.22874e-3 * t**2 * h
                + 8.5282e-4 * t * h**2 - 1.99e-6 * t**2 * h**2
            )
            
            hi[high_temp_mask] = hi_complex
        
        return hi
    
    def _calculate_wind_chill(self, temp: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind chill temperature."""
        # Convert wind speed from m/s to km/h
        wind_kmh = wind_speed * 3.6
        
        # Wind chill formula (valid for temp <= 10°C and wind >= 4.8 km/h)
        mask = (temp <= 10) & (wind_kmh >= 4.8)
        wind_chill = temp.copy()
        
        wind_chill[mask] = (
            13.12 + 0.6215 * temp[mask] 
            - 11.37 * (wind_kmh[mask] ** 0.16)
            + 0.3965 * temp[mask] * (wind_kmh[mask] ** 0.16)
        )
        
        return wind_chill
    
    def create_lag_features(self, df: pd.DataFrame, lag_hours: List[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.
        
        Args:
            df: DataFrame with weather data
            lag_hours: List of lag periods in hours
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for feature in self.feature_columns:
            for lag in lag_hours:
                lag_col = f"{feature}_lag_{lag}h"
                df[lag_col] = df[feature].shift(lag)
                self.engineered_features.append(lag_col)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create rolling window features (mean, std, min, max).
        
        Args:
            df: DataFrame with weather data
            windows: List of window sizes in hours
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for feature in self.feature_columns:
            for window in windows:
                # Rolling statistics
                df[f"{feature}_rolling_mean_{window}h"] = df[feature].rolling(window=window, min_periods=1).mean()
                df[f"{feature}_rolling_std_{window}h"] = df[feature].rolling(window=window, min_periods=1).std()
                df[f"{feature}_rolling_min_{window}h"] = df[feature].rolling(window=window, min_periods=1).min()
                df[f"{feature}_rolling_max_{window}h"] = df[feature].rolling(window=window, min_periods=1).max()
                
                # Add to engineered features list
                rolling_features = [
                    f"{feature}_rolling_mean_{window}h",
                    f"{feature}_rolling_std_{window}h", 
                    f"{feature}_rolling_min_{window}h",
                    f"{feature}_rolling_max_{window}h"
                ]
                self.engineered_features.extend(rolling_features)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in self.imputers:
                # Use median for weather data (robust to outliers)
                self.imputers[col] = SimpleImputer(strategy='median')
                
            if df[col].isnull().any():
                df[[col]] = self.imputers[col].fit_transform(df[[col]])
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers in weather data.
        
        Args:
            df: DataFrame with weather data
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        
        for col in self.feature_columns:
            if col in df.columns:
                if method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                elif method == "zscore":
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outlier_mask = z_scores > threshold
                    
                    # Replace outliers with median
                    if outlier_mask.any():
                        df.loc[outlier_mask, col] = df[col].median()
        
        return df
    
    def fit_scalers(self, df: pd.DataFrame) -> None:
        """
        Fit scalers on the training data.
        
        Args:
            df: Training DataFrame
        """
        # Get all numeric columns for scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]
        
        for col in numeric_cols:
            if col not in self.scalers:
                self.scalers[col] = self._create_scaler()
                self.scalers[col].fit(df[[col]])
        
        self.is_fitted = True
        logger.info(f"Fitted scalers for {len(self.scalers)} features")
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling transformation to features.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        
        df = df.copy()
        
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]]).flatten()
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scalers and transform data in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        self.fit_scalers(df)
        return self.transform_features(df)
    
    def preprocess_for_training(self, weather_data: List[WeatherData]) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for training data.
        
        Args:
            weather_data: List of WeatherData objects
            
        Returns:
            Preprocessed DataFrame ready for training
        """
        logger.info(f"Preprocessing {len(weather_data)} weather records for training")
        
        # Convert to DataFrame
        df = self.weather_data_to_dataframe(weather_data)
        
        if df.empty:
            return df
        
        # Feature engineering
        df = self.create_time_features(df)
        df = self.create_weather_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        
        # Data cleaning
        df = self.handle_missing_values(df)
        df = self.detect_outliers(df)
        
        # Scaling
        df = self.fit_transform(df)
        
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def preprocess_for_prediction(self, weather_data: List[WeatherData]) -> pd.DataFrame:
        """
        Preprocessing pipeline for prediction data.
        
        Args:
            weather_data: List of WeatherData objects
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Train a model first.")
        
        # Convert to DataFrame
        df = self.weather_data_to_dataframe(weather_data)
        
        if df.empty:
            return df
        
        # Feature engineering (same as training)
        df = self.create_time_features(df)
        df = self.create_weather_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        
        # Data cleaning
        df = self.handle_missing_values(df)
        df = self.detect_outliers(df)
        
        # Scaling (using fitted scalers)
        df = self.transform_features(df)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names after preprocessing."""
        base_features = self.feature_columns.copy()
        all_features = base_features + self.engineered_features
        return list(set(all_features))  # Remove duplicates