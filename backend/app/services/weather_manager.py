"""
Weather API manager with fallback mechanism and caching.
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
from loguru import logger

from app.services.weather_apis import (
    WeatherAPIClient, 
    OpenMeteoAPI, 
    WeatherAPIClient_WeatherAPI, 
    OpenWeatherMapAPI
)
from app.schemas.weather import WeatherData, Location, WeatherAPIError
from app.core.config import settings


class WeatherAPIManager:
    """Manages multiple weather APIs with fallback and caching."""
    
    def __init__(self):
        self.apis: List[WeatherAPIClient] = [
            OpenMeteoAPI(),  # Primary - free, no API key required
            WeatherAPIClient_WeatherAPI(),  # Secondary
            OpenWeatherMapAPI()  # Tertiary
        ]
        self.current_api_index = 0
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = settings.cache_ttl_seconds
        
        # API health tracking
        self.api_health: Dict[str, Dict[str, Any]] = {}
        for api in self.apis:
            self.api_health[api.name] = {
                "consecutive_failures": 0,
                "last_failure": None,
                "is_healthy": True,
                "total_requests": 0,
                "successful_requests": 0
            }
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for method and parameters."""
        key_parts = [method]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if not cache_entry:
            return False
        
        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        return (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl
    
    def _cache_result(self, key: str, result: Any) -> None:
        """Cache a result with timestamp."""
        self.cache[key] = {
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _update_api_health(self, api_name: str, success: bool) -> None:
        """Update API health metrics."""
        health = self.api_health[api_name]
        health["total_requests"] += 1
        
        if success:
            health["successful_requests"] += 1
            health["consecutive_failures"] = 0
            health["is_healthy"] = True
        else:
            health["consecutive_failures"] += 1
            health["last_failure"] = datetime.utcnow().isoformat()
            
            # Mark as unhealthy after 3 consecutive failures
            if health["consecutive_failures"] >= 3:
                health["is_healthy"] = False
                logger.warning(f"API {api_name} marked as unhealthy after {health['consecutive_failures']} failures")
    
    def _get_healthy_apis(self) -> List[WeatherAPIClient]:
        """Get list of currently healthy APIs."""
        healthy_apis = []
        for api in self.apis:
            health = self.api_health[api.name]
            
            # Reset health status if enough time has passed since last failure
            if not health["is_healthy"] and health["last_failure"]:
                last_failure = datetime.fromisoformat(health["last_failure"])
                if (datetime.utcnow() - last_failure).total_seconds() > 300:  # 5 minutes
                    health["is_healthy"] = True
                    health["consecutive_failures"] = 0
                    logger.info(f"API {api.name} health status reset")
            
            if health["is_healthy"]:
                healthy_apis.append(api)
        
        return healthy_apis
    
    async def get_current_weather(self, latitude: float, longitude: float) -> WeatherData:
        """
        Get current weather with fallback mechanism.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            WeatherData object
            
        Raises:
            WeatherAPIError: If all APIs fail
        """
        # Check cache first
        cache_key = self._get_cache_key("current_weather", lat=latitude, lon=longitude)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.info(f"Returning cached weather data for ({latitude}, {longitude})")
            cached_data = self.cache[cache_key]["data"]
            # Convert dict back to WeatherData if needed
            if isinstance(cached_data, dict):
                return WeatherData(**cached_data)
            return cached_data
        
        healthy_apis = self._get_healthy_apis()
        if not healthy_apis:
            # If no APIs are healthy, try all APIs anyway
            healthy_apis = self.apis
            logger.warning("No healthy APIs available, trying all APIs")
        
        last_error = None
        
        for api in healthy_apis:
            try:
                logger.info(f"Attempting to fetch weather data from {api.name}")
                weather_data = await api.get_current_weather(latitude, longitude)
                
                # Cache successful result
                self._cache_result(cache_key, weather_data.dict())
                self._update_api_health(api.name, True)
                
                logger.info(f"Successfully fetched weather data from {api.name}")
                return weather_data
                
            except WeatherAPIError as e:
                last_error = e
                self._update_api_health(api.name, False)
                logger.error(f"API {api.name} failed: {e.message}")
                
                if not e.retryable:
                    logger.info(f"API {api.name} error is not retryable, skipping")
                    continue
                
                # Add small delay before trying next API
                await asyncio.sleep(0.5)
                continue
            
            except Exception as e:
                last_error = WeatherAPIError(
                    message=f"Unexpected error from {api.name}: {str(e)}",
                    code="UNEXPECTED_ERROR",
                    source=api.name.lower(),
                    retryable=True
                )
                self._update_api_health(api.name, False)
                logger.error(f"Unexpected error from {api.name}: {e}")
                await asyncio.sleep(0.5)
                continue
        
        # All APIs failed
        error_msg = f"All weather APIs failed. Last error: {last_error.message if last_error else 'Unknown'}"
        logger.error(error_msg)
        raise WeatherAPIError(
            message=error_msg,
            code="ALL_APIS_FAILED",
            source="weather_manager",
            retryable=True
        )
    
    async def get_historical_weather(
        self, 
        latitude: float, 
        longitude: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WeatherData]:
        """
        Get historical weather data with fallback mechanism.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            List of WeatherData objects
        """
        # Check cache first
        cache_key = self._get_cache_key(
            "historical_weather", 
            lat=latitude, 
            lon=longitude,
            start=start_date.isoformat(),
            end=end_date.isoformat()
        )
        
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.info(f"Returning cached historical data for ({latitude}, {longitude})")
            cached_data = self.cache[cache_key]["data"]
            return [WeatherData(**item) if isinstance(item, dict) else item for item in cached_data]
        
        healthy_apis = self._get_healthy_apis()
        if not healthy_apis:
            healthy_apis = self.apis
        
        for api in healthy_apis:
            try:
                logger.info(f"Attempting to fetch historical data from {api.name}")
                weather_data = await api.get_historical_weather(latitude, longitude, start_date, end_date)
                
                if weather_data:  # Only cache if we got data
                    self._cache_result(cache_key, [item.dict() for item in weather_data])
                    self._update_api_health(api.name, True)
                    logger.info(f"Successfully fetched {len(weather_data)} historical records from {api.name}")
                    return weather_data
                
            except Exception as e:
                self._update_api_health(api.name, False)
                logger.error(f"Historical data fetch failed from {api.name}: {e}")
                continue
        
        logger.warning("No historical data available from any API")
        return []
    
    async def search_locations(self, query: str, limit: int = 10) -> List[Location]:
        """
        Search for locations with fallback mechanism.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of Location objects
        """
        # Check cache first
        cache_key = self._get_cache_key("search_locations", query=query, limit=limit)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.info(f"Returning cached location search for '{query}'")
            cached_data = self.cache[cache_key]["data"]
            return [Location(**item) if isinstance(item, dict) else item for item in cached_data]
        
        healthy_apis = self._get_healthy_apis()
        if not healthy_apis:
            healthy_apis = self.apis
        
        all_locations = []
        
        # Try to get results from multiple APIs and combine them
        for api in healthy_apis:
            try:
                locations = await api.search_locations(query, limit)
                if locations:
                    all_locations.extend(locations)
                    self._update_api_health(api.name, True)
                
            except Exception as e:
                self._update_api_health(api.name, False)
                logger.error(f"Location search failed from {api.name}: {e}")
                continue
        
        # Remove duplicates and limit results
        unique_locations = []
        seen_coords = set()
        
        for location in all_locations:
            coord_key = (round(location.latitude, 2), round(location.longitude, 2))
            if coord_key not in seen_coords:
                unique_locations.append(location)
                seen_coords.add(coord_key)
                
                if len(unique_locations) >= limit:
                    break
        
        # Prioritize Indian cities as per requirements
        indian_cities = [loc for loc in unique_locations if loc.country.lower() in ['india', 'in']]
        other_cities = [loc for loc in unique_locations if loc.country.lower() not in ['india', 'in']]
        
        final_locations = indian_cities + other_cities
        
        if final_locations:
            self._cache_result(cache_key, [loc.dict() for loc in final_locations])
        
        return final_locations[:limit]
    
    def get_api_health_status(self) -> Dict[str, Any]:
        """Get current health status of all APIs."""
        return {
            "apis": self.api_health,
            "healthy_count": len(self._get_healthy_apis()),
            "total_count": len(self.apis)
        }
    
    async def close(self):
        """Close all API clients."""
        for api in self.apis:
            await api.close()