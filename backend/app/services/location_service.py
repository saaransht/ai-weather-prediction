"""
Location service for GPS functionality and location management.
"""

import asyncio
from typing import List, Optional, Tuple
import httpx
from loguru import logger

from app.schemas.weather import Location
from app.services.weather_manager import WeatherAPIManager


class LocationService:
    """Service for location-related operations."""
    
    def __init__(self, weather_manager: WeatherAPIManager):
        self.weather_manager = weather_manager
        self.client = httpx.AsyncClient(timeout=10.0)
        
        # Indian cities for prioritization
        self.indian_major_cities = [
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "region": "Maharashtra"},
            {"name": "Delhi", "lat": 28.7041, "lon": 77.1025, "region": "Delhi"},
            {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946, "region": "Karnataka"},
            {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "region": "Telangana"},
            {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "region": "Tamil Nadu"},
            {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "region": "West Bengal"},
            {"name": "Pune", "lat": 18.5204, "lon": 73.8567, "region": "Maharashtra"},
            {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "region": "Gujarat"},
            {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "region": "Rajasthan"},
            {"name": "Surat", "lat": 21.1702, "lon": 72.8311, "region": "Gujarat"},
            {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "region": "Uttar Pradesh"},
            {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319, "region": "Uttar Pradesh"},
            {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882, "region": "Maharashtra"},
            {"name": "Indore", "lat": 22.7196, "lon": 75.8577, "region": "Madhya Pradesh"},
            {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126, "region": "Madhya Pradesh"},
            {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "region": "Andhra Pradesh"},
            {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "region": "Bihar"},
            {"name": "Vadodara", "lat": 22.3072, "lon": 73.1812, "region": "Gujarat"},
            {"name": "Ghaziabad", "lat": 28.6692, "lon": 77.4538, "region": "Uttar Pradesh"},
            {"name": "Ludhiana", "lat": 30.9010, "lon": 75.8573, "region": "Punjab"},
        ]
    
    async def reverse_geocode(self, latitude: float, longitude: float) -> Optional[Location]:
        """
        Convert GPS coordinates to location name using reverse geocoding.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Location object or None if failed
        """
        try:
            # Try Open-Meteo geocoding API first (free)
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {
                "name": f"{latitude:.4f},{longitude:.4f}",
                "count": 1,
                "language": "en",
                "format": "json"
            }
            
            response = await self.client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    result = data["results"][0]
                    return Location(
                        name=result["name"],
                        latitude=result["latitude"],
                        longitude=result["longitude"],
                        country=result.get("country", "Unknown"),
                        region=result.get("admin1")
                    )
            
            # Fallback: try to find nearest major city
            nearest_city = self._find_nearest_major_city(latitude, longitude)
            if nearest_city:
                return nearest_city
            
            # Last resort: create generic location
            return Location(
                name=f"Location ({latitude:.2f}, {longitude:.2f})",
                latitude=latitude,
                longitude=longitude,
                country="Unknown"
            )
            
        except Exception as e:
            logger.error(f"Reverse geocoding failed: {e}")
            return Location(
                name=f"Location ({latitude:.2f}, {longitude:.2f})",
                latitude=latitude,
                longitude=longitude,
                country="Unknown"
            )
    
    def _find_nearest_major_city(self, latitude: float, longitude: float) -> Optional[Location]:
        """Find the nearest major Indian city to given coordinates."""
        min_distance = float('inf')
        nearest_city = None
        
        for city in self.indian_major_cities:
            distance = self._calculate_distance(
                latitude, longitude, 
                city["lat"], city["lon"]
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_city = city
        
        # Only return if within reasonable distance (100km)
        if nearest_city and min_distance < 100:
            return Location(
                name=nearest_city["name"],
                latitude=nearest_city["lat"],
                longitude=nearest_city["lon"],
                country="India",
                region=nearest_city["region"]
            )
        
        return None
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates using Haversine formula.
        Returns distance in kilometers.
        """
        import math
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    async def search_locations(self, query: str, limit: int = 10) -> List[Location]:
        """
        Search for locations with Indian cities prioritized.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of Location objects with Indian cities first
        """
        try:
            # First, check if query matches any major Indian cities
            indian_matches = []
            query_lower = query.lower()
            
            for city in self.indian_major_cities:
                if query_lower in city["name"].lower():
                    indian_matches.append(Location(
                        name=city["name"],
                        latitude=city["lat"],
                        longitude=city["lon"],
                        country="India",
                        region=city["region"]
                    ))
            
            # Get results from weather APIs
            api_results = await self.weather_manager.search_locations(query, limit)
            
            # Combine and prioritize Indian cities
            all_locations = indian_matches + api_results
            
            # Remove duplicates while preserving order
            unique_locations = []
            seen_coords = set()
            
            for location in all_locations:
                coord_key = (round(location.latitude, 2), round(location.longitude, 2))
                if coord_key not in seen_coords:
                    unique_locations.append(location)
                    seen_coords.add(coord_key)
                    
                    if len(unique_locations) >= limit:
                        break
            
            return unique_locations
            
        except Exception as e:
            logger.error(f"Location search failed: {e}")
            return []
    
    def validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """
        Validate GPS coordinates.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            True if coordinates are valid
        """
        return (
            -90 <= latitude <= 90 and
            -180 <= longitude <= 180
        )
    
    def get_popular_indian_cities(self, limit: int = 10) -> List[Location]:
        """
        Get list of popular Indian cities for quick selection.
        
        Args:
            limit: Maximum number of cities to return
            
        Returns:
            List of popular Indian cities
        """
        cities = []
        for city in self.indian_major_cities[:limit]:
            cities.append(Location(
                name=city["name"],
                latitude=city["lat"],
                longitude=city["lon"],
                country="India",
                region=city["region"]
            ))
        
        return cities
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()