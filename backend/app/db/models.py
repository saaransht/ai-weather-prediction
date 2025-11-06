"""
SQLAlchemy models for weather data storage.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, Index
from sqlalchemy.sql import func
from datetime import datetime

from app.db.database import Base


class WeatherRecord(Base):
    """Model for storing weather data records."""
    
    __tablename__ = "weather_records"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Location information
    location_name = Column(String(255), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    country = Column(String(100))
    region = Column(String(100))
    
    # Weather data
    timestamp = Column(DateTime, nullable=False)
    temperature = Column(Float, nullable=False)  # Celsius
    humidity = Column(Float, nullable=False)  # Percentage
    pressure = Column(Float, nullable=False)  # hPa
    wind_speed = Column(Float, nullable=False)  # m/s
    wind_direction = Column(Float, nullable=False)  # degrees
    cloud_cover = Column(Float, nullable=False)  # Percentage
    precipitation = Column(Float, default=0.0)  # mm
    
    # Metadata
    data_source = Column(String(50), nullable=False)  # API source
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_location_timestamp', 'latitude', 'longitude', 'timestamp'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_location', 'latitude', 'longitude'),
    )


class ModelTrainingData(Base):
    """Model for storing preprocessed training data."""
    
    __tablename__ = "model_training_data"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Location information
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Training metadata
    model_type = Column(String(50), nullable=False)
    training_start_date = Column(DateTime, nullable=False)
    training_end_date = Column(DateTime, nullable=False)
    data_points_count = Column(Integer, nullable=False)
    
    # Preprocessed data (stored as JSON)
    features_data = Column(Text)  # JSON string of feature data
    targets_data = Column(Text)  # JSON string of target data
    feature_names = Column(Text)  # JSON string of feature names
    
    # Training configuration
    preprocessing_config = Column(Text)  # JSON string of preprocessing parameters
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_model_location', 'model_type', 'latitude', 'longitude'),
        Index('idx_training_dates', 'training_start_date', 'training_end_date'),
    )


class ModelMetadata(Base):
    """Model for storing ML model metadata and performance metrics."""
    
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Location information
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Training information
    training_start_date = Column(DateTime, nullable=False)
    training_end_date = Column(DateTime, nullable=False)
    training_data_points = Column(Integer, nullable=False)
    
    # Model configuration
    hyperparameters = Column(Text)  # JSON string
    feature_columns = Column(Text)  # JSON string
    target_columns = Column(Text)  # JSON string
    
    # Performance metrics
    mae = Column(Float)  # Mean Absolute Error
    rmse = Column(Float)  # Root Mean Square Error
    mape = Column(Float)  # Mean Absolute Percentage Error
    r2_score = Column(Float)  # R-squared score
    
    # Model file information
    model_file_path = Column(String(500))
    model_file_size = Column(Integer)  # bytes
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    is_trained = Column(Boolean, default=False)
    last_prediction_time = Column(DateTime)
    prediction_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_model_location_active', 'model_type', 'latitude', 'longitude', 'is_active'),
        Index('idx_model_performance', 'model_type', 'mae', 'rmse'),
    )


class PredictionCache(Base):
    """Model for caching prediction results."""
    
    __tablename__ = "prediction_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Location and time information
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    prediction_timestamp = Column(DateTime, nullable=False)
    forecast_horizon_hours = Column(Integer, nullable=False)
    
    # Prediction data
    predictions_data = Column(Text, nullable=False)  # JSON string of predictions
    model_performance = Column(Text)  # JSON string of model metrics
    ensemble_weights = Column(Text)  # JSON string of ensemble weights
    
    # Cache metadata
    cache_key = Column(String(255), unique=True, index=True)
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=False)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_cache_location_time', 'latitude', 'longitude', 'prediction_timestamp'),
        Index('idx_cache_expiry', 'expires_at'),
    )