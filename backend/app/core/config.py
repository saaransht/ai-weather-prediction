"""
Application configuration and settings.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    app_name: str = "AI Weather Prediction API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Weather API Keys (optional for free tiers)
    openweathermap_api_key: Optional[str] = None
    weatherapi_key: Optional[str] = None
    
    # API URLs
    openmeteo_base_url: str = "https://api.open-meteo.com/v1"
    weatherapi_base_url: str = "https://api.weatherapi.com/v1"
    openweathermap_base_url: str = "https://api.openweathermap.org/data/2.5"
    
    # Cache Configuration
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 900  # 15 minutes
    prediction_cache_ttl: int = 3600  # 1 hour
    
    # Database Configuration
    database_url: str = "sqlite:///./weather_data.db"
    
    # Model Configuration
    model_storage_path: str = "./models"
    training_data_days: int = 30
    prediction_horizon_hours: int = 24
    
    # ML Model Settings
    lstm_sequence_length: int = 168  # 7 days of hourly data
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    random_forest_n_estimators: int = 100
    random_forest_max_depth: int = 20
    
    # CORS Configuration
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://nwp-gtib5xtii-saaransh-tiwaris-projects.vercel.app",
        "https://*.vercel.app",
        "https://*.render.com"
    ]
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()