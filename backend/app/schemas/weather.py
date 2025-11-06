"""
Pydantic schemas for weather data validation.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


class Location(BaseModel):
    """Location information."""
    name: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    country: str
    region: Optional[str] = None


class WeatherData(BaseModel):
    """Current weather data."""
    timestamp: datetime
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    pressure: float = Field(..., gt=0, description="Pressure in hPa")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    wind_direction: float = Field(..., ge=0, le=360, description="Wind direction in degrees")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover percentage")
    precipitation: Optional[float] = Field(None, ge=0, description="Precipitation in mm")
    location: Location


class UncertaintyBounds(BaseModel):
    """Uncertainty bounds for predictions."""
    lower_bound: float
    upper_bound: float
    confidence: float = Field(..., ge=0, le=1)


class ModelPredictions(BaseModel):
    """Individual model predictions."""
    lstm: float
    arima: float
    random_forest: float
    fuzzy_time_series: float
    ensemble: float


class PredictionResult(BaseModel):
    """Single prediction result."""
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    uncertainty: UncertaintyBounds
    model_contributions: ModelPredictions


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    mae: float = Field(..., ge=0, description="Mean Absolute Error")
    rmse: float = Field(..., ge=0, description="Root Mean Square Error")
    mape: float = Field(..., ge=0, description="Mean Absolute Percentage Error")
    last_updated: datetime


class PredictionRequest(BaseModel):
    """Request for weather prediction."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    hours: int = Field(24, ge=1, le=168, description="Forecast horizon in hours")


class PredictionResponse(BaseModel):
    """Response with weather predictions."""
    location: Location
    prediction_time: datetime
    forecast_horizon: int
    predictions: List[PredictionResult]
    model_performance: dict
    current_weather: WeatherData


class CurrentWeatherResponse(BaseModel):
    """Response with current weather."""
    location: Location
    current: WeatherData
    last_updated: datetime


class LocationSearchRequest(BaseModel):
    """Request for location search."""
    query: str = Field(..., min_length=2, max_length=100)
    limit: int = Field(10, ge=1, le=50)


class LocationSearchResponse(BaseModel):
    """Response with location search results."""
    locations: List[Location]


class APIError(BaseModel):
    """API error response."""
    message: str
    code: str
    details: Optional[dict] = None


class WeatherAPIError(APIError):
    """Weather API specific error."""
    source: str = Field(..., description="API source that failed")
    retryable: bool = Field(True, description="Whether the error is retryable")