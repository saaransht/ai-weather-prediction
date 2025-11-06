"""
Historical weather data collection and storage service.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from loguru import logger

from app.db.database import get_db, SessionLocal
from app.db.models import WeatherRecord, ModelTrainingData
from app.schemas.weather import WeatherData, Location
from app.services.weather_manager import WeatherAPIManager
from app.ml.preprocessing import WeatherDataPreprocessor
from app.core.config import settings


class DataCollector:
    """Service for collecting and storing historical weather data."""
    
    def __init__(self):
        self.weather_manager = WeatherAPIManager()
        self.preprocessor = WeatherDataPreprocessor()
    
    async def collect_historical_data(
        self, 
        latitude: float, 
        longitude: float, 
        days: int = None,
        location_name: str = None
    ) -> List[WeatherData]:
        """
        Collect historical weather data for a location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            days: Number of days of historical data (default from settings)
            location_name: Optional location name
            
        Returns:
            List of WeatherData objects
        """
        if days is None:
            days = settings.training_data_days
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Collecting {days} days of historical data for ({latitude}, {longitude})")
        
        try:
            # Check if we already have recent data in database
            existing_data = self._get_existing_data(latitude, longitude, start_date, end_date)
            
            if len(existing_data) >= (days * 20):  # At least 20 records per day (hourly with some gaps)
                logger.info(f"Using {len(existing_data)} existing records from database")
                return existing_data
            
            # Fetch from APIs
            weather_data = await self.weather_manager.get_historical_weather(
                latitude, longitude, start_date, end_date
            )
            
            if weather_data:
                # Store in database
                await self._store_weather_data(weather_data)
                logger.info(f"Collected and stored {len(weather_data)} historical records")
                return weather_data
            else:
                logger.warning("No historical data available from APIs, using existing data")
                return existing_data
                
        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            # Return existing data as fallback
            return self._get_existing_data(latitude, longitude, start_date, end_date)
    
    def _get_existing_data(
        self, 
        latitude: float, 
        longitude: float, 
        start_date: datetime, 
        end_date: datetime,
        tolerance: float = 0.1
    ) -> List[WeatherData]:
        """
        Get existing weather data from database.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date for data
            end_date: End date for data
            tolerance: Coordinate tolerance for matching locations
            
        Returns:
            List of WeatherData objects
        """
        db = SessionLocal()
        try:
            records = db.query(WeatherRecord).filter(
                and_(
                    WeatherRecord.latitude.between(latitude - tolerance, latitude + tolerance),
                    WeatherRecord.longitude.between(longitude - tolerance, longitude + tolerance),
                    WeatherRecord.timestamp.between(start_date, end_date)
                )
            ).order_by(WeatherRecord.timestamp).all()
            
            weather_data = []
            for record in records:
                location = Location(
                    name=record.location_name,
                    latitude=record.latitude,
                    longitude=record.longitude,
                    country=record.country or "Unknown",
                    region=record.region
                )
                
                weather_data.append(WeatherData(
                    timestamp=record.timestamp,
                    temperature=record.temperature,
                    humidity=record.humidity,
                    pressure=record.pressure,
                    wind_speed=record.wind_speed,
                    wind_direction=record.wind_direction,
                    cloud_cover=record.cloud_cover,
                    precipitation=record.precipitation or 0.0,
                    location=location
                ))
            
            return weather_data
            
        finally:
            db.close()
    
    async def _store_weather_data(self, weather_data: List[WeatherData]) -> None:
        """
        Store weather data in database.
        
        Args:
            weather_data: List of WeatherData objects to store
        """
        if not weather_data:
            return
        
        db = SessionLocal()
        try:
            records_to_add = []
            
            for wd in weather_data:
                # Check if record already exists
                existing = db.query(WeatherRecord).filter(
                    and_(
                        WeatherRecord.latitude == wd.location.latitude,
                        WeatherRecord.longitude == wd.location.longitude,
                        WeatherRecord.timestamp == wd.timestamp
                    )
                ).first()
                
                if not existing:
                    record = WeatherRecord(
                        location_name=wd.location.name,
                        latitude=wd.location.latitude,
                        longitude=wd.location.longitude,
                        country=wd.location.country,
                        region=wd.location.region,
                        timestamp=wd.timestamp,
                        temperature=wd.temperature,
                        humidity=wd.humidity,
                        pressure=wd.pressure,
                        wind_speed=wd.wind_speed,
                        wind_direction=wd.wind_direction,
                        cloud_cover=wd.cloud_cover,
                        precipitation=wd.precipitation or 0.0,
                        data_source="api_collection"
                    )
                    records_to_add.append(record)
            
            if records_to_add:
                db.add_all(records_to_add)
                db.commit()
                logger.info(f"Stored {len(records_to_add)} new weather records")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error storing weather data: {e}")
            raise
        finally:
            db.close()
    
    def prepare_training_data(
        self, 
        latitude: float, 
        longitude: float, 
        model_type: str,
        target_columns: List[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepare and preprocess training data for ML models.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            model_type: Type of model being trained
            target_columns: List of target column names
            
        Returns:
            Tuple of (features_dict, targets_dict)
        """
        if target_columns is None:
            target_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']
        
        logger.info(f"Preparing training data for {model_type} at ({latitude}, {longitude})")
        
        # Get historical data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=settings.training_data_days)
        
        weather_data = self._get_existing_data(latitude, longitude, start_date, end_date)
        
        if len(weather_data) < 24:  # Need at least 24 hours of data
            raise ValueError(f"Insufficient training data: {len(weather_data)} records")
        
        # Preprocess data
        df = self.preprocessor.preprocess_for_training(weather_data)
        
        if df.empty:
            raise ValueError("Preprocessing resulted in empty dataset")
        
        # Prepare features and targets
        feature_columns = self.preprocessor.get_feature_names()
        available_features = [col for col in feature_columns if col in df.columns]
        available_targets = [col for col in target_columns if col in df.columns]
        
        if not available_features or not available_targets:
            raise ValueError("No valid features or targets found after preprocessing")
        
        # Create feature and target datasets
        features_data = {
            'data': df[available_features].values.tolist(),
            'columns': available_features,
            'shape': df[available_features].shape
        }
        
        targets_data = {
            'data': df[available_targets].values.tolist(),
            'columns': available_targets,
            'shape': df[available_targets].shape
        }
        
        # Store preprocessed data in database
        self._store_training_data(
            latitude, longitude, model_type, 
            start_date, end_date, len(weather_data),
            features_data, targets_data, available_features
        )
        
        logger.info(f"Prepared training data: {features_data['shape']} features, {targets_data['shape']} targets")
        
        return features_data, targets_data
    
    def _store_training_data(
        self,
        latitude: float,
        longitude: float,
        model_type: str,
        start_date: datetime,
        end_date: datetime,
        data_points_count: int,
        features_data: Dict[str, Any],
        targets_data: Dict[str, Any],
        feature_names: List[str]
    ) -> None:
        """Store preprocessed training data in database."""
        db = SessionLocal()
        try:
            # Deactivate old training data for this model and location
            db.query(ModelTrainingData).filter(
                and_(
                    ModelTrainingData.model_type == model_type,
                    ModelTrainingData.latitude == latitude,
                    ModelTrainingData.longitude == longitude
                )
            ).update({'is_active': False})
            
            # Create new training data record
            training_data = ModelTrainingData(
                latitude=latitude,
                longitude=longitude,
                model_type=model_type,
                training_start_date=start_date,
                training_end_date=end_date,
                data_points_count=data_points_count,
                features_data=json.dumps(features_data),
                targets_data=json.dumps(targets_data),
                feature_names=json.dumps(feature_names),
                preprocessing_config=json.dumps({
                    'scaler_type': self.preprocessor.scaler_type,
                    'feature_columns': self.preprocessor.feature_columns,
                    'engineered_features': self.preprocessor.engineered_features
                })
            )
            
            db.add(training_data)
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error storing training data: {e}")
            raise
        finally:
            db.close()
    
    def get_training_data(
        self, 
        latitude: float, 
        longitude: float, 
        model_type: str
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Retrieve stored training data for a model and location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            model_type: Type of model
            
        Returns:
            Tuple of (features_dict, targets_dict) or None if not found
        """
        db = SessionLocal()
        try:
            training_data = db.query(ModelTrainingData).filter(
                and_(
                    ModelTrainingData.model_type == model_type,
                    ModelTrainingData.latitude == latitude,
                    ModelTrainingData.longitude == longitude,
                    ModelTrainingData.is_active == True
                )
            ).order_by(desc(ModelTrainingData.created_at)).first()
            
            if training_data:
                features_data = json.loads(training_data.features_data)
                targets_data = json.loads(training_data.targets_data)
                return features_data, targets_data
            
            return None
            
        finally:
            db.close()
    
    def validate_data_quality(self, weather_data: List[WeatherData]) -> Dict[str, Any]:
        """
        Validate the quality of collected weather data.
        
        Args:
            weather_data: List of WeatherData objects
            
        Returns:
            Dictionary with quality metrics
        """
        if not weather_data:
            return {
                'is_valid': False,
                'total_records': 0,
                'missing_data_percentage': 100,
                'issues': ['No data available']
            }
        
        df = self.preprocessor.weather_data_to_dataframe(weather_data)
        
        quality_metrics = {
            'is_valid': True,
            'total_records': len(weather_data),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            },
            'missing_data_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            'issues': []
        }
        
        # Check for data quality issues
        if quality_metrics['missing_data_percentage'] > 20:
            quality_metrics['issues'].append(f"High missing data: {quality_metrics['missing_data_percentage']:.1f}%")
        
        if len(weather_data) < 24:
            quality_metrics['issues'].append(f"Insufficient data: {len(weather_data)} records")
            quality_metrics['is_valid'] = False
        
        # Check for reasonable value ranges
        for col in ['temperature', 'humidity', 'pressure', 'wind_speed']:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    if col == 'temperature' and (col_data.min() < -50 or col_data.max() > 60):
                        quality_metrics['issues'].append(f"Extreme {col} values: {col_data.min():.1f} to {col_data.max():.1f}")
                    elif col == 'humidity' and (col_data.min() < 0 or col_data.max() > 100):
                        quality_metrics['issues'].append(f"Invalid {col} values: {col_data.min():.1f} to {col_data.max():.1f}")
                    elif col == 'pressure' and (col_data.min() < 800 or col_data.max() > 1100):
                        quality_metrics['issues'].append(f"Extreme {col} values: {col_data.min():.1f} to {col_data.max():.1f}")
        
        if quality_metrics['issues']:
            quality_metrics['is_valid'] = len(quality_metrics['issues']) <= 2  # Allow minor issues
        
        return quality_metrics
    
    async def close(self):
        """Close connections."""
        await self.weather_manager.close()