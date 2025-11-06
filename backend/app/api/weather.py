"""
Weather API endpoints.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Depends, Query
from loguru import logger

from app.schemas.weather import (
    CurrentWeatherResponse,
    LocationSearchRequest,
    LocationSearchResponse,
    WeatherAPIError
)
from app.services.weather_manager import WeatherAPIManager
from app.services.location_service import LocationService


router = APIRouter(prefix="/api/weather", tags=["weather"])

# Dependency to get weather manager
async def get_weather_manager() -> WeatherAPIManager:
    return WeatherAPIManager()

# Dependency to get location service
async def get_location_service(
    weather_manager: WeatherAPIManager = Depends(get_weather_manager)
) -> LocationService:
    return LocationService(weather_manager)


@router.get("/current", response_model=CurrentWeatherResponse)
async def get_current_weather(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    weather_manager: WeatherAPIManager = Depends(get_weather_manager)
):
    """
    Get current weather for given coordinates.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        
    Returns:
        Current weather data
    """
    try:
        logger.info(f"Fetching current weather for ({lat}, {lon})")
        
        weather_data = await weather_manager.get_current_weather(lat, lon)
        
        return CurrentWeatherResponse(
            location=weather_data.location,
            current=weather_data,
            last_updated=weather_data.timestamp
        )
        
    except WeatherAPIError as e:
        logger.error(f"Weather API error: {e.message}")
        raise HTTPException(
            status_code=503,
            detail={
                "message": e.message,
                "code": e.code,
                "source": e.source,
                "retryable": e.retryable
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_current_weather: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while fetching weather data"
        )
    finally:
        await weather_manager.close()


@router.get("/locations/search", response_model=LocationSearchResponse)
async def search_locations(
    q: str = Query(..., min_length=2, max_length=100, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    location_service: LocationService = Depends(get_location_service)
):
    """
    Search for locations by name.
    
    Args:
        q: Search query
        limit: Maximum number of results
        
    Returns:
        List of matching locations
    """
    try:
        logger.info(f"Searching locations for query: '{q}'")
        
        locations = await location_service.search_locations(q, limit)
        
        return LocationSearchResponse(locations=locations)
        
    except Exception as e:
        logger.error(f"Error in location search: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while searching locations"
        )
    finally:
        await location_service.close()


@router.get("/locations/reverse-geocode")
async def reverse_geocode(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    location_service: LocationService = Depends(get_location_service)
):
    """
    Convert GPS coordinates to location name.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        
    Returns:
        Location information
    """
    try:
        logger.info(f"Reverse geocoding for ({lat}, {lon})")
        
        if not location_service.validate_coordinates(lat, lon):
            raise HTTPException(
                status_code=400,
                detail="Invalid coordinates provided"
            )
        
        location = await location_service.reverse_geocode(lat, lon)
        
        if not location:
            raise HTTPException(
                status_code=404,
                detail="Could not determine location for given coordinates"
            )
        
        return location
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in reverse geocoding: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while reverse geocoding"
        )
    finally:
        await location_service.close()


@router.get("/locations/popular-indian-cities")
async def get_popular_indian_cities(
    limit: int = Query(10, ge=1, le=20, description="Maximum number of cities"),
    location_service: LocationService = Depends(get_location_service)
):
    """
    Get list of popular Indian cities for quick selection.
    
    Args:
        limit: Maximum number of cities to return
        
    Returns:
        List of popular Indian cities
    """
    try:
        cities = location_service.get_popular_indian_cities(limit)
        return {"cities": cities}
        
    except Exception as e:
        logger.error(f"Error getting popular cities: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while fetching popular cities"
        )


@router.get("/health")
async def get_api_health(
    weather_manager: WeatherAPIManager = Depends(get_weather_manager)
):
    """
    Get health status of weather APIs.
    
    Returns:
        API health information
    """
    try:
        health_status = weather_manager.get_api_health_status()
        return health_status
        
    except Exception as e:
        logger.error(f"Error getting API health: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while checking API health"
        )
    finally:
        await weather_manager.close()