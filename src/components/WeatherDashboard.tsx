'use client'

import { useState, useEffect } from 'react'
import { 
  Thermometer, 
  Droplets, 
  Gauge, 
  Wind, 
  Cloud, 
  CloudRain,
  Loader2,
  AlertCircle,
  RefreshCw
} from 'lucide-react'
import { CurrentWeatherResponse, PredictionResult } from '@/types/weather'

interface WeatherDashboardProps {
  currentWeather?: CurrentWeatherResponse | null
  predictions?: PredictionResult[]
  isLoading?: boolean
  error?: string
  onRefresh?: () => void
  location?: any
}

export default function WeatherDashboard({
  currentWeather,
  predictions,
  isLoading = false,
  error,
  onRefresh
}: WeatherDashboardProps) {
  const [selectedHour, setSelectedHour] = useState(0)

  // Auto-select first prediction when predictions change
  useEffect(() => {
    if (predictions && predictions.length > 0) {
      setSelectedHour(0)
    }
  }, [predictions])

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    })
  }

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    })
  }

  const getWeatherIcon = (temp: number, humidity: number, cloudCover?: number) => {
    if (cloudCover && cloudCover > 80) {
      return humidity > 80 ? <CloudRain className="w-8 h-8" /> : <Cloud className="w-8 h-8" />
    }
    return <Thermometer className="w-8 h-8" />
  }

  const getTemperatureColor = (temp: number) => {
    if (temp < 0) return 'text-blue-600'
    if (temp < 10) return 'text-blue-500'
    if (temp < 20) return 'text-green-500'
    if (temp < 30) return 'text-yellow-500'
    if (temp < 40) return 'text-orange-500'
    return 'text-red-500'
  }

  if (error) {
    return (
      <div className="weather-card">
        <div className="flex items-center justify-center p-8">
          <div className="text-center">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Weather Data Unavailable</h3>
            <p className="text-gray-600 mb-4">{error}</p>
            {onRefresh && (
              <button onClick={onRefresh} className="btn-primary">
                <RefreshCw className="w-4 h-4 mr-2" />
                Try Again
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="weather-card">
        <div className="flex items-center justify-center p-8">
          <div className="text-center">
            <Loader2 className="w-8 h-8 animate-spin text-primary-600 mx-auto mb-4" />
            <p className="text-gray-600">Loading weather data...</p>
          </div>
        </div>
      </div>
    )
  }

  if (!currentWeather) {
    return (
      <div className="weather-card">
        <div className="text-center p-8">
          <div className="text-4xl mb-4">🌤️</div>
          <p className="text-gray-600">Select a location to view weather data</p>
        </div>
      </div>
    )
  }

  const selectedPrediction = predictions?.[selectedHour]

  return (
    <div className="space-y-6">
      {/* Current Weather */}
      <div className="weather-card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold">Current Weather</h2>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Location and Temperature */}
          <div className="text-center md:text-left">
            <h3 className="text-lg font-medium text-gray-900 mb-1">
              {currentWeather.location.name}
            </h3>
            <p className="text-sm text-gray-500 mb-4">
              {currentWeather.location.region && `${currentWeather.location.region}, `}
              {currentWeather.location.country}
            </p>
            
            <div className="flex items-center justify-center md:justify-start gap-4">
              {getWeatherIcon(
                currentWeather.current.temperature,
                currentWeather.current.humidity,
                currentWeather.current.cloudCover
              )}
              <span className={`text-4xl font-bold ${getTemperatureColor(currentWeather.current.temperature)}`}>
                {Math.round(currentWeather.current.temperature)}°C
              </span>
            </div>
          </div>

          {/* Weather Details */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center gap-2">
              <Droplets className="w-5 h-5 text-blue-500" />
              <div>
                <p className="text-sm text-gray-500">Humidity</p>
                <p className="font-semibold">{Math.round(currentWeather.current.humidity)}%</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Gauge className="w-5 h-5 text-gray-500" />
              <div>
                <p className="text-sm text-gray-500">Pressure</p>
                <p className="font-semibold">{Math.round(currentWeather.current.pressure)} hPa</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Wind className="w-5 h-5 text-green-500" />
              <div>
                <p className="text-sm text-gray-500">Wind Speed</p>
                <p className="font-semibold">{Math.round(currentWeather.current.windSpeed * 3.6)} km/h</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Cloud className="w-5 h-5 text-gray-400" />
              <div>
                <p className="text-sm text-gray-500">Cloud Cover</p>
                <p className="font-semibold">{Math.round(currentWeather.current.cloudCover)}%</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500">
            Last updated: {formatTime(currentWeather.lastUpdated)} on {formatDate(currentWeather.lastUpdated)}
          </p>
        </div>
      </div>

      {/* Predictions */}
      {predictions && predictions.length > 0 && (
        <div className="weather-card">
          <h2 className="text-2xl font-semibold mb-4">24-Hour Forecast</h2>

          {/* Hour Selector */}
          <div className="mb-6">
            <div className="flex overflow-x-auto pb-2 gap-2">
              {predictions.slice(0, 24).map((prediction, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedHour(index)}
                  className={`flex-shrink-0 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedHour === index
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <div>{formatTime(prediction.timestamp)}</div>
                  <div className="text-xs opacity-75">
                    {index === 0 ? 'Now' : `+${index}h`}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Selected Prediction Details */}
          {selectedPrediction && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Predicted Values */}
              <div>
                <h3 className="text-lg font-medium mb-4">
                  Forecast for {formatTime(selectedPrediction.timestamp)}
                </h3>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Thermometer className="w-4 h-4 text-red-500" />
                      <span className="text-sm text-gray-600">Temperature</span>
                    </div>
                    <span className={`font-semibold ${getTemperatureColor(selectedPrediction.temperature)}`}>
                      {Math.round(selectedPrediction.temperature)}°C
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Droplets className="w-4 h-4 text-blue-500" />
                      <span className="text-sm text-gray-600">Humidity</span>
                    </div>
                    <span className="font-semibold">{Math.round(selectedPrediction.humidity)}%</span>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Gauge className="w-4 h-4 text-gray-500" />
                      <span className="text-sm text-gray-600">Pressure</span>
                    </div>
                    <span className="font-semibold">{Math.round(selectedPrediction.pressure)} hPa</span>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Wind className="w-4 h-4 text-green-500" />
                      <span className="text-sm text-gray-600">Wind Speed</span>
                    </div>
                    <span className="font-semibold">{Math.round(selectedPrediction.windSpeed * 3.6)} km/h</span>
                  </div>
                </div>
              </div>

              {/* Uncertainty Information */}
              <div>
                <h3 className="text-lg font-medium mb-4">Prediction Confidence</h3>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Temperature Range</span>
                      <span className="font-medium">
                        {Math.round(selectedPrediction.uncertainty.lowerBound)}° - {Math.round(selectedPrediction.uncertainty.upperBound)}°C
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-primary-600 h-2 rounded-full"
                        style={{ width: `${selectedPrediction.uncertainty.confidence * 100}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {Math.round(selectedPrediction.uncertainty.confidence * 100)}% confidence
                    </p>
                  </div>

                  {/* Model Contributions */}
                  <div>
                    <p className="text-sm text-gray-600 mb-2">Model Contributions</p>
                    <div className="space-y-1">
                      {Object.entries(selectedPrediction.modelContributions).map(([model, contribution]) => (
                        <div key={model} className="flex justify-between text-xs">
                          <span className="capitalize text-gray-600">{model}</span>
                          <span className="font-medium">{Math.round(contribution * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}