'use client'

import { useState, useEffect } from 'react'
import { MapPin, Search, Loader2 } from 'lucide-react'
import { Location } from '@/types/weather'

interface LocationSelectorProps {
  onLocationSelect: (location: Location) => void
  enableGPS?: boolean
  className?: string
}

export default function LocationSelector({ 
  onLocationSelect, 
  enableGPS = true, 
  className = '' 
}: LocationSelectorProps) {
  const [query, setQuery] = useState('')
  const [locations, setLocations] = useState<Location[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [isGettingLocation, setIsGettingLocation] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)
  const [popularCities, setPopularCities] = useState<Location[]>([])

  // Load popular Indian cities on component mount
  useEffect(() => {
    fetchPopularCities()
  }, [])

  const fetchPopularCities = async () => {
    try {
      const response = await fetch('/api/weather/locations/popular-indian-cities?limit=10')
      if (response.ok) {
        const data = await response.json()
        setPopularCities(data.cities || [])
      }
    } catch (error) {
      console.error('Failed to fetch popular cities:', error)
    }
  }

  const searchLocations = async (searchQuery: string) => {
    if (searchQuery.length < 2) {
      setLocations([])
      return
    }

    setIsSearching(true)
    try {
      const response = await fetch(
        `/api/weather/locations/search?q=${encodeURIComponent(searchQuery)}&limit=10`
      )
      
      if (response.ok) {
        const data = await response.json()
        setLocations(data.locations || [])
      } else {
        setLocations([])
      }
    } catch (error) {
      console.error('Location search failed:', error)
      setLocations([])
    } finally {
      setIsSearching(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setQuery(value)
    setShowDropdown(true)
    
    // Debounce search
    const timeoutId = setTimeout(() => {
      searchLocations(value)
    }, 300)

    return () => clearTimeout(timeoutId)
  }

  const handleLocationSelect = (location: Location) => {
    setQuery(location.name)
    setShowDropdown(false)
    onLocationSelect(location)
  }

  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by this browser.')
      return
    }

    setIsGettingLocation(true)
    
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          const { latitude, longitude } = position.coords
          
          // Reverse geocode to get location name
          const response = await fetch(
            `/api/weather/locations/reverse-geocode?lat=${latitude}&lon=${longitude}`
          )
          
          if (response.ok) {
            const location: Location = await response.json()
            handleLocationSelect(location)
          } else {
            // Fallback location
            const fallbackLocation: Location = {
              name: `Location (${latitude.toFixed(2)}, ${longitude.toFixed(2)})`,
              latitude,
              longitude,
              country: 'Unknown'
            }
            handleLocationSelect(fallbackLocation)
          }
        } catch (error) {
          console.error('Reverse geocoding failed:', error)
          alert('Failed to get location name. Please try searching manually.')
        } finally {
          setIsGettingLocation(false)
        }
      },
      (error) => {
        console.error('Geolocation error:', error)
        setIsGettingLocation(false)
        
        switch (error.code) {
          case error.PERMISSION_DENIED:
            alert('Location access denied. Please enable location services.')
            break
          case error.POSITION_UNAVAILABLE:
            alert('Location information is unavailable.')
            break
          case error.TIMEOUT:
            alert('Location request timed out.')
            break
          default:
            alert('An unknown error occurred while getting location.')
            break
        }
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 300000 // 5 minutes
      }
    )
  }

  const displayLocations = query.length >= 2 ? locations : popularCities

  return (
    <div className={`relative ${className}`}>
      <div className="space-y-4">
        {/* GPS Button */}
        {enableGPS && (
          <button
            onClick={getCurrentLocation}
            disabled={isGettingLocation}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {isGettingLocation ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <MapPin className="w-4 h-4" />
            )}
            {isGettingLocation ? 'Getting Location...' : 'Use Current Location'}
          </button>
        )}

        {/* Search Input */}
        <div className="relative">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              value={query}
              onChange={handleInputChange}
              onFocus={() => setShowDropdown(true)}
              placeholder="Search for a city..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
            {isSearching && (
              <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4 animate-spin" />
            )}
          </div>

          {/* Dropdown */}
          {showDropdown && (
            <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
              {displayLocations.length > 0 ? (
                <>
                  {query.length < 2 && (
                    <div className="px-4 py-2 text-sm text-gray-500 border-b">
                      Popular Indian Cities
                    </div>
                  )}
                  {displayLocations.map((location, index) => (
                    <button
                      key={index}
                      onClick={() => handleLocationSelect(location)}
                      className="w-full px-4 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none"
                    >
                      <div className="font-medium">{location.name}</div>
                      <div className="text-sm text-gray-500">
                        {location.region && `${location.region}, `}{location.country}
                      </div>
                    </button>
                  ))}
                </>
              ) : query.length >= 2 ? (
                <div className="px-4 py-2 text-sm text-gray-500">
                  {isSearching ? 'Searching...' : 'No locations found'}
                </div>
              ) : (
                <div className="px-4 py-2 text-sm text-gray-500">
                  Type to search for locations
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Click outside to close dropdown */}
      {showDropdown && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowDropdown(false)}
        />
      )}
    </div>
  )
}