'use client'

import { useState, useEffect, useCallback } from 'react'
import { 
  Location, 
  CurrentWeatherResponse, 
  PredictionResponse, 
  ModelStatusResponse,
  WeatherAPIError 
} from '@/types/weather'

interface WeatherState {
  currentWeather: CurrentWeatherResponse | null
  predictions: PredictionResponse | null
  modelStatus: ModelStatusResponse | null
  isLoading: boolean
  error: string | null
}

interface UseWeatherDataReturn extends WeatherState {
  fetchWeatherData: (location: Location) => Promise<void>
  fetchPredictions: (location: Location, hours?: number) => Promise<void>
  fetchModelStatus: () => Promise<void>
  clearError: () => void
  refreshData: () => Promise<void>
}

export function useWeatherData(): UseWeatherDataReturn {
  const [state, setState] = useState<WeatherState>({
    currentWeather: null,
    predictions: null,
    modelStatus: null,
    isLoading: false,
    error: null
  })

  const [lastLocation, setLastLocation] = useState<Location | null>(null)

  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }))
  }, [])

  const handleError = useCallback((error: any) => {
    console.error('Weather API Error:', error)
    
    let errorMessage = 'An unexpected error occurred'
    
    if (error.response) {
      // HTTP error response
      const status = error.response.status
      const data = error.response.data
      
      if (status === 404) {
        errorMessage = 'Weather data not available for this location'
      } else if (status === 503) {
        errorMessage = 'Weather service is temporarily unavailable'
      } else if (data?.detail) {
        errorMessage = typeof data.detail === 'string' ? data.detail : data.detail.message || errorMessage
      }
    } else if (error.message) {
      errorMessage = error.message
    }

    setState(prev => ({ 
      ...prev, 
      error: errorMessage,
      isLoading: false 
    }))
  }, [])

  const fetchWeatherData = useCallback(async (location: Location) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }))
    
    try {
      const response = await fetch(
        `/api/weather/current?lat=${location.latitude}&lon=${location.longitude}`
      )
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const data: CurrentWeatherResponse = await response.json()
      
      setState(prev => ({ 
        ...prev, 
        currentWeather: data,
        isLoading: false 
      }))
      
      setLastLocation(location)
      
    } catch (error) {
      handleError(error)
    }
  }, [handleError])

  const fetchPredictions = useCallback(async (location: Location, hours: number = 24) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }))
    
    try {
      const response = await fetch('/api/predictions/forecast', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          latitude: location.latitude,
          longitude: location.longitude,
          hours
        })
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const data: PredictionResponse = await response.json()
      
      setState(prev => ({ 
        ...prev, 
        predictions: data,
        isLoading: false 
      }))
      
    } catch (error) {
      handleError(error)
    }
  }, [handleError])

  const fetchModelStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/predictions/models/status')
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const data: ModelStatusResponse = await response.json()
      
      setState(prev => ({ 
        ...prev, 
        modelStatus: data
      }))
      
    } catch (error) {
      console.warn('Failed to fetch model status:', error)
      // Don't set error state for model status failures
    }
  }, [])

  const refreshData = useCallback(async () => {
    if (lastLocation) {
      await Promise.all([
        fetchWeatherData(lastLocation),
        fetchPredictions(lastLocation),
        fetchModelStatus()
      ])
    }
  }, [lastLocation, fetchWeatherData, fetchPredictions, fetchModelStatus])

  // Auto-refresh model status periodically
  useEffect(() => {
    fetchModelStatus()
    
    const interval = setInterval(fetchModelStatus, 60000) // Every minute
    
    return () => clearInterval(interval)
  }, [fetchModelStatus])

  return {
    ...state,
    fetchWeatherData,
    fetchPredictions,
    fetchModelStatus,
    clearError,
    refreshData
  }}
