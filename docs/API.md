# API Documentation

## Overview

The AI Weather Prediction API provides endpoints for weather data retrieval, forecasting, and model management. The API is built with FastAPI and provides automatic OpenAPI documentation.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-backend-url.railway.app`

## Authentication

Currently, the API does not require authentication. Rate limiting is applied based on IP address.

## Rate Limiting

- **Limit**: 100 requests per hour per IP
- **Headers**: Rate limit information is included in response headers
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset time (Unix timestamp)

## Endpoints

### Weather Data

#### Get Current Weather

```http
GET /api/weather/current
```

**Parameters:**
- `lat` (float, required): Latitude (-90 to 90)
- `lon` (float, required): Longitude (-180 to 180)

**Response:**
```json
{
  "location": {
    "name": "New York",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "country": "United States",
    "region": "New York"
  },
  "current": {
    "timestamp": "2024-01-15T12:00:00Z",
    "temperature": 15.5,
    "humidity": 65.0,
    "pressure": 1013.25,
    "wind_speed": 5.2,
    "wind_direction": 180.0,
    "cloud_cover": 75.0,
    "precipitation": 0.0,
    "location": { /* same as above */ }
  },
  "last_updated": "2024-01-15T12:00:00Z"
}
```

#### Search Locations

```http
GET /api/weather/locations/search
```

**Parameters:**
- `q` (string, required): Search query (min 2 characters)
- `limit` (int, optional): Maximum results (default: 10, max: 50)

**Response:**
```json
{
  "locations": [
    {
      "name": "Mumbai",
      "latitude": 19.0760,
      "longitude": 72.8777,
      "country": "India",
      "region": "Maharashtra"
    }
  ]
}
```

#### Reverse Geocoding

```http
GET /api/weather/locations/reverse-geocode
```

**Parameters:**
- `lat` (float, required): Latitude
- `lon` (float, required): Longitude

**Response:**
```json
{
  "name": "Central Park",
  "latitude": 40.7829,
  "longitude": -73.9654,
  "country": "United States",
  "region": "New York"
}
```

#### Popular Indian Cities

```http
GET /api/weather/locations/popular-indian-cities
```

**Parameters:**
- `limit` (int, optional): Maximum cities (default: 10, max: 20)

**Response:**
```json
{
  "cities": [
    {
      "name": "Mumbai",
      "latitude": 19.0760,
      "longitude": 72.8777,
      "country": "India",
      "region": "Maharashtra"
    }
  ]
}
```

### Predictions

#### Generate Forecast

```http
POST /api/predictions/forecast
```

**Request Body:**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "hours": 24
}
```

**Response:**
```json
{
  "location": {
    "name": "New York",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "country": "United States"
  },
  "prediction_time": "2024-01-15T12:00:00Z",
  "forecast_horizon": 24,
  "predictions": [
    {
      "timestamp": "2024-01-15T13:00:00Z",
      "temperature": 16.2,
      "humidity": 63.5,
      "pressure": 1013.8,
      "wind_speed": 5.8,
      "uncertainty": {
        "lower_bound": 14.1,
        "upper_bound": 18.3,
        "confidence": 0.95
      },
      "model_contributions": {
        "lstm": 16.1,
        "arima": 16.0,
        "random_forest": 16.4,
        "fuzzy_time_series": 16.2,
        "ensemble": 16.2
      }
    }
  ],
  "model_performance": {
    "lstm": {
      "mae": 1.2,
      "rmse": 1.8,
      "mape": 7.5,
      "last_updated": "2024-01-15T10:00:00Z"
    }
  },
  "current_weather": { /* current weather object */ }
}
```

#### Model Status

```http
GET /api/predictions/models/status
```

**Response:**
```json
{
  "models": [
    {
      "name": "LSTM Weather Model",
      "type": "lstm",
      "is_loaded": true,
      "last_trained": "2024-01-15T08:00:00Z",
      "performance": {
        "mae": 1.2,
        "rmse": 1.8,
        "mape": 7.5,
        "last_updated": "2024-01-15T10:00:00Z"
      },
      "status": "healthy"
    }
  ],
  "system_health": "healthy"
}
```

#### Train Models

```http
POST /api/predictions/models/train
```

**Request Body:**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "model_types": ["lstm", "random_forest"],
  "force_retrain": false
}
```

**Response:**
```json
{
  "message": "Model training initiated",
  "job_id": "train_job_123",
  "estimated_time": 300,
  "models_to_train": ["lstm", "random_forest"]
}
```

### Health and Status

#### API Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "AI Weather Prediction API",
  "version": "1.0.0"
}
```

#### Weather API Health

```http
GET /api/weather/health
```

**Response:**
```json
{
  "apis": {
    "Open-Meteo": {
      "consecutive_failures": 0,
      "last_failure": null,
      "is_healthy": true,
      "total_requests": 150,
      "successful_requests": 148
    }
  },
  "healthy_count": 3,
  "total_count": 3
}
```

## Error Handling

### Error Response Format

```json
{
  "detail": {
    "message": "Error description",
    "code": "ERROR_CODE",
    "details": {
      "field": "Additional error details"
    }
  }
}
```

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (location not found)
- `422`: Validation Error (invalid request body)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error
- `503`: Service Unavailable (external API failure)

### Common Error Codes

- `INVALID_COORDINATES`: Invalid latitude/longitude values
- `LOCATION_NOT_FOUND`: Location not found in geocoding
- `ALL_APIS_FAILED`: All weather APIs are unavailable
- `MODEL_NOT_TRAINED`: Requested model is not trained
- `INSUFFICIENT_DATA`: Not enough data for training/prediction
- `RATE_LIMIT_EXCEEDED`: Too many requests

## Data Models

### Location

```typescript
interface Location {
  name: string
  latitude: number      // -90 to 90
  longitude: number     // -180 to 180
  country: string
  region?: string
}
```

### Weather Data

```typescript
interface WeatherData {
  timestamp: string     // ISO 8601 format
  temperature: number   // Celsius
  humidity: number      // Percentage (0-100)
  pressure: number      // hPa
  wind_speed: number    // m/s
  wind_direction: number // Degrees (0-360)
  cloud_cover: number   // Percentage (0-100)
  precipitation?: number // mm
  location: Location
}
```

### Uncertainty Bounds

```typescript
interface UncertaintyBounds {
  lower_bound: number
  upper_bound: number
  confidence: number    // 0-1 (typically 0.95)
}
```

### Model Predictions

```typescript
interface ModelPredictions {
  lstm: number
  arima: number
  random_forest: number
  fuzzy_time_series: number
  ensemble: number
}
```

## Usage Examples

### JavaScript/TypeScript

```javascript
// Get current weather
const getCurrentWeather = async (lat, lon) => {
  const response = await fetch(
    `${API_BASE_URL}/api/weather/current?lat=${lat}&lon=${lon}`
  )
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`)
  }
  
  return await response.json()
}

// Get forecast
const getForecast = async (lat, lon, hours = 24) => {
  const response = await fetch(`${API_BASE_URL}/api/predictions/forecast`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      latitude: lat,
      longitude: lon,
      hours: hours
    })
  })
  
  return await response.json()
}
```

### Python

```python
import requests

API_BASE_URL = "http://localhost:8000"

def get_current_weather(lat, lon):
    response = requests.get(
        f"{API_BASE_URL}/api/weather/current",
        params={"lat": lat, "lon": lon}
    )
    response.raise_for_status()
    return response.json()

def get_forecast(lat, lon, hours=24):
    response = requests.post(
        f"{API_BASE_URL}/api/predictions/forecast",
        json={
            "latitude": lat,
            "longitude": lon,
            "hours": hours
        }
    )
    response.raise_for_status()
    return response.json()
```

### cURL

```bash
# Get current weather
curl -X GET "http://localhost:8000/api/weather/current?lat=40.7128&lon=-74.0060"

# Get forecast
curl -X POST "http://localhost:8000/api/predictions/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "hours": 24
  }'
```

## Rate Limiting and Best Practices

### Rate Limits

- **Weather Data**: 100 requests/hour per IP
- **Predictions**: 50 requests/hour per IP
- **Model Training**: 5 requests/hour per IP

### Best Practices

1. **Cache Results**: Cache weather data for at least 15 minutes
2. **Handle Errors**: Implement proper error handling and retries
3. **Batch Requests**: Use single requests for multiple data points when possible
4. **Monitor Usage**: Track your API usage to avoid rate limits
5. **Use Appropriate Timeouts**: Set reasonable request timeouts (30s recommended)

### Optimization Tips

1. **Location Caching**: Cache location search results
2. **Prediction Caching**: Cache predictions for 1 hour
3. **Fallback Strategy**: Implement fallback for API failures
4. **Progressive Enhancement**: Load critical data first, then enhancements

## Webhooks (Future Feature)

Planned webhook support for:
- Model training completion
- Severe weather alerts
- API health status changes

## SDK and Libraries

Official SDKs planned for:
- JavaScript/TypeScript
- Python
- React hooks library

## Support

- **Documentation**: Visit `/docs` endpoint for interactive API docs
- **Issues**: Report bugs on GitHub
- **Questions**: Use GitHub Discussions
- **Status**: Check API status at `/health` endpoint