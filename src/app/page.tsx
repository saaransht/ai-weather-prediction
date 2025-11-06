'use client'

import { useState } from 'react'
import { Metadata } from 'next'
import LocationSelector from '@/components/LocationSelector'
import WeatherDashboard from '@/components/WeatherDashboard'
import ForecastChart from '@/components/ForecastChart'
import ModelComparison from '@/components/ModelComparison'
import ErrorBoundary from '@/components/ErrorBoundary'
import { useWeatherData } from '@/hooks/useWeatherData'
import { Location } from '@/types/weather'

export default function HomePage() {
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null)
  const {
    currentWeather,
    predictions,
    modelStatus,
    isLoading,
    error,
    fetchWeatherData,
    fetchPredictions,
    clearError
  } = useWeatherData()

  const handleLocationSelect = async (location: Location) => {
    setSelectedLocation(location)
    clearError()
    
    try {
      await Promise.all([
        fetchWeatherData(location),
        fetchPredictions(location)
      ])
    } catch (err) {
      console.error('Failed to fetch weather data:', err)
    }
  }

  return (
    <ErrorBoundary>
      <main className="container mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold text-gradient mb-4">
            AI Weather Prediction
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Advanced weather forecasting using multiple AI/ML models with uncertainty quantification
          </p>
        </div>

        {error && (
          <div className="error-message mb-6">
            <p>{error}</p>
            <button 
              onClick={clearError}
              className="mt-2 text-sm underline hover:no-underline"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          <LocationSelector 
            onLocationSelect={handleLocationSelect}
          />
          
          <WeatherDashboard 
            currentWeather={currentWeather}
            isLoading={isLoading}
            location={selectedLocation}
          />
          
          <ModelComparison 
            modelStatus={modelStatus}
            predictions={predictions}
            isLoading={isLoading}
          />
        </div>

        {predictions && (
          <div className="space-y-8">
            <ForecastChart 
              predictions={predictions}
              isLoading={isLoading}
            />
          </div>
        )}
      </main>
    </ErrorBoundary>
  )
}