"""
Weather API client implementations for multiple providers.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import httpx
from loguru import logger

from app.core.config import settings
from app.schemas.weather import WeatherData, Location, WeatherAPIError


class WeatherAPIClient(ABC):
    """Abstract base class for weather API clients."""
    
    def __init__(self, name: str, base_url: str, api_key: Optional[str] = None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
    
    @abstractmethod
    async def get_current_weather(self, latitude: float, longitude: float) -> WeatherData:
        """Get current weather for given coordinates."""
        pass
    
    @abstractmethod
    async def get_historical_weather(
        self, 
        latitude: float, 
        longitude: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WeatherData]:
        """Get historical weather data for training."""
        pass
    
    @abstractmethod
    async def search_locations(self, query: str, limit: int = 10) -> List[Location]:
        """Search for locations by name."""
        pass
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class OpenMeteoAPI(WeatherAPIClient):
    """Open-Meteo API client (free, no API key required)."""
    
    def __init__(self):
        super().__init__(
            name="Open-Meteo",
            base_url=settings.openmeteo_base_url
        )
    
    async def get_current_weather(self, latitude: float, longitude: float) -> WeatherData:
        """Get current weather from Open-Meteo API."""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m", 
                    "surface_pressure",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "cloud_cover",
                    "precipitation"
                ],
                "timezone": "auto"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            current = data["current"]
            
            # Create location from coordinates
            location = Location(
                name=f"Location ({latitude:.2f}, {longitude:.2f})",
                latitude=latitude,
                longitude=longitude,
                country="Unknown"
            )
            
            return WeatherData(
                timestamp=datetime.fromisoformat(current["time"].replace("Z", "+00:00")),
                temperature=current["temperature_2m"],
                humidity=current["relative_humidity_2m"],
                pressure=current["surface_pressure"],
                wind_speed=current["wind_speed_10m"],
                wind_direction=current["wind_direction_10m"],
                cloud_cover=current["cloud_cover"],
                precipitation=current.get("precipitation", 0.0),
                location=location
            )
            
        except Exception as e:
            logger.error(f"Open-Meteo API error: {e}")
            raise WeatherAPIError(
                message=f"Failed to fetch weather data: {str(e)}",
                code="OPENMETEO_ERROR",
                source="openmeteo",
                retryable=True
            )
    
    async def get_historical_weather(
        self, 
        latitude: float, 
        longitude: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WeatherData]:
        """Get historical weather data from Open-Meteo."""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "surface_pressure", 
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "cloud_cover",
                    "precipitation"
                ],
                "timezone": "UTC"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            hourly = data["hourly"]
            weather_data = []
            
            location = Location(
                name=f"Location ({latitude:.2f}, {longitude:.2f})",
                latitude=latitude,
                longitude=longitude,
                country="Unknown"
            )
            
            for i in range(len(hourly["time"])):
                weather_data.append(WeatherData(
                    timestamp=datetime.fromisoformat(hourly["time"][i]),
                    temperature=hourly["temperature_2m"][i] or 0.0,
                    humidity=hourly["relative_humidity_2m"][i] or 0.0,
                    pressure=hourly["surface_pressure"][i] or 1013.25,
                    wind_speed=hourly["wind_speed_10m"][i] or 0.0,
                    wind_direction=hourly["wind_direction_10m"][i] or 0.0,
                    cloud_cover=hourly["cloud_cover"][i] or 0.0,
                    precipitation=hourly["precipitation"][i] or 0.0,
                    location=location
                ))
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Open-Meteo historical data error: {e}")
            raise WeatherAPIError(
                message=f"Failed to fetch historical data: {str(e)}",
                code="OPENMETEO_HISTORICAL_ERROR",
                source="openmeteo",
                retryable=True
            )
    
    async def search_locations(self, query: str, limit: int = 10) -> List[Location]:
        """Search locations using Open-Meteo geocoding."""
        try:
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {
                "name": query,
                "count": limit,
                "language": "en",
                "format": "json"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            locations = []
            for result in data.get("results", []):
                locations.append(Location(
                    name=result["name"],
                    latitude=result["latitude"],
                    longitude=result["longitude"],
                    country=result.get("country", "Unknown"),
                    region=result.get("admin1", None)
                ))
            
            return locations
            
        except Exception as e:
            logger.error(f"Open-Meteo location search error: {e}")
            return []


class WeatherAPIClient_WeatherAPI(WeatherAPIClient):
    """WeatherAPI.com client (free tier available)."""
    
    def __init__(self):
        super().__init__(
            name="WeatherAPI",
            base_url=settings.weatherapi_base_url,
            api_key=settings.weatherapi_key
        )
    
    async def get_current_weather(self, latitude: float, longitude: float) -> WeatherData:
        """Get current weather from WeatherAPI."""
        if not self.api_key:
            raise WeatherAPIError(
                message="WeatherAPI key not configured",
                code="NO_API_KEY",
                source="weatherapi",
                retryable=False
            )
        
        try:
            url = f"{self.base_url}/current.json"
            params = {
                "key": self.api_key,
                "q": f"{latitude},{longitude}",
                "aqi": "no"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            current = data["current"]
            location_data = data["location"]
            
            location = Location(
                name=location_data["name"],
                latitude=location_data["lat"],
                longitude=location_data["lon"],
                country=location_data["country"],
                region=location_data.get("region")
            )
            
            return WeatherData(
                timestamp=datetime.fromisoformat(location_data["localtime"]),
                temperature=current["temp_c"],
                humidity=current["humidity"],
                pressure=current["pressure_mb"],
                wind_speed=current["wind_kph"] / 3.6,  # Convert to m/s
                wind_direction=current["wind_degree"],
                cloud_cover=current["cloud"],
                precipitation=current.get("precip_mm", 0.0),
                location=location
            )
            
        except Exception as e:
            logger.error(f"WeatherAPI error: {e}")
            raise WeatherAPIError(
                message=f"Failed to fetch weather data: {str(e)}",
                code="WEATHERAPI_ERROR",
                source="weatherapi",
                retryable=True
            )
    
    async def get_historical_weather(
        self, 
        latitude: float, 
        longitude: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WeatherData]:
        """Get historical weather data from WeatherAPI."""
        if not self.api_key:
            raise WeatherAPIError(
                message="WeatherAPI key not configured",
                code="NO_API_KEY", 
                source="weatherapi",
                retryable=False
            )
        
        # WeatherAPI free tier has limited historical data
        # This is a simplified implementation
        weather_data = []
        current_date = start_date.date()
        
        while current_date <= end_date.date():
            try:
                url = f"{self.base_url}/history.json"
                params = {
                    "key": self.api_key,
                    "q": f"{latitude},{longitude}",
                    "dt": current_date.strftime("%Y-%m-%d")
                }
                
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                location_data = data["location"]
                location = Location(
                    name=location_data["name"],
                    latitude=location_data["lat"],
                    longitude=location_data["lon"],
                    country=location_data["country"],
                    region=location_data.get("region")
                )
                
                for hour_data in data["forecast"]["forecastday"][0]["hour"]:
                    weather_data.append(WeatherData(
                        timestamp=datetime.fromisoformat(hour_data["time"]),
                        temperature=hour_data["temp_c"],
                        humidity=hour_data["humidity"],
                        pressure=hour_data["pressure_mb"],
                        wind_speed=hour_data["wind_kph"] / 3.6,
                        wind_direction=hour_data["wind_degree"],
                        cloud_cover=hour_data["cloud"],
                        precipitation=hour_data.get("precip_mm", 0.0),
                        location=location
                    ))
                
                current_date += timedelta(days=1)
                
            except Exception as e:
                logger.error(f"WeatherAPI historical error for {current_date}: {e}")
                current_date += timedelta(days=1)
                continue
        
        return weather_data
    
    async def search_locations(self, query: str, limit: int = 10) -> List[Location]:
        """Search locations using WeatherAPI."""
        if not self.api_key:
            return []
        
        try:
            url = f"{self.base_url}/search.json"
            params = {
                "key": self.api_key,
                "q": query
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            locations = []
            for result in data[:limit]:
                locations.append(Location(
                    name=result["name"],
                    latitude=result["lat"],
                    longitude=result["lon"],
                    country=result["country"],
                    region=result.get("region")
                ))
            
            return locations
            
        except Exception as e:
            logger.error(f"WeatherAPI location search error: {e}")
            return []


class OpenWeatherMapAPI(WeatherAPIClient):
    """OpenWeatherMap API client (free tier available)."""
    
    def __init__(self):
        super().__init__(
            name="OpenWeatherMap",
            base_url=settings.openweathermap_base_url,
            api_key=settings.openweathermap_api_key
        )
    
    async def get_current_weather(self, latitude: float, longitude: float) -> WeatherData:
        """Get current weather from OpenWeatherMap."""
        if not self.api_key:
            raise WeatherAPIError(
                message="OpenWeatherMap API key not configured",
                code="NO_API_KEY",
                source="openweathermap",
                retryable=False
            )
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": latitude,
                "lon": longitude,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            location = Location(
                name=data["name"],
                latitude=data["coord"]["lat"],
                longitude=data["coord"]["lon"],
                country=data["sys"]["country"]
            )
            
            return WeatherData(
                timestamp=datetime.fromtimestamp(data["dt"]),
                temperature=data["main"]["temp"],
                humidity=data["main"]["humidity"],
                pressure=data["main"]["pressure"],
                wind_speed=data["wind"]["speed"],
                wind_direction=data["wind"].get("deg", 0),
                cloud_cover=data["clouds"]["all"],
                precipitation=data.get("rain", {}).get("1h", 0.0),
                location=location
            )
            
        except Exception as e:
            logger.error(f"OpenWeatherMap API error: {e}")
            raise WeatherAPIError(
                message=f"Failed to fetch weather data: {str(e)}",
                code="OPENWEATHERMAP_ERROR",
                source="openweathermap",
                retryable=True
            )
    
    async def get_historical_weather(
        self, 
        latitude: float, 
        longitude: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WeatherData]:
        """Get historical weather data from OpenWeatherMap."""
        # OpenWeatherMap historical data requires paid subscription
        # This is a placeholder implementation
        logger.warning("OpenWeatherMap historical data requires paid subscription")
        return []
    
    async def search_locations(self, query: str, limit: int = 10) -> List[Location]:
        """Search locations using OpenWeatherMap geocoding."""
        if not self.api_key:
            return []
        
        try:
            url = "http://api.openweathermap.org/geo/1.0/direct"
            params = {
                "q": query,
                "limit": limit,
                "appid": self.api_key
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            locations = []
            for result in data:
                locations.append(Location(
                    name=result["name"],
                    latitude=result["lat"],
                    longitude=result["lon"],
                    country=result["country"],
                    region=result.get("state")
                ))
            
            return locations
            
        except Exception as e:
            logger.error(f"OpenWeatherMap location search error: {e}")
            return []