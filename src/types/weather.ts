// Core weather data types
export interface Location {
  name: string;
  latitude: number;
  longitude: number;
  country: string;
  region?: string;
}

export interface WeatherData {
  timestamp: string;
  temperature: number; // Celsius
  humidity: number; // Percentage
  pressure: number; // hPa
  windSpeed: number; // m/s
  windDirection: number; // degrees
  cloudCover: number; // Percentage
  precipitation?: number; // mm
  location: Location;
}

export interface UncertaintyBounds {
  lowerBound: number;
  upperBound: number;
  confidence: number; // 0-1
}

export interface ModelPredictions {
  lstm: number;
  arima: number;
  randomForest: number;
  fuzzyTimeSeries: number;
  ensemble: number;
}

export interface PredictionPoint {
  timestamp: string;
  parameter: string;
  value: number;
  uncertainty: UncertaintyBounds;
  modelContributions: ModelPredictions;
}

export interface PredictionResult {
  timestamp: string;
  temperature: number;
  humidity: number;
  pressure: number;
  windSpeed: number;
  uncertainty: UncertaintyBounds;
  modelContributions: ModelPredictions;
}

export interface ModelMetrics {
  mae: number; // Mean Absolute Error
  rmse: number; // Root Mean Square Error
  mape: number; // Mean Absolute Percentage Error
  lastUpdated: string;
}

export interface PredictionResponse {
  location: Location;
  predictionTime: string;
  forecastHorizon: number; // hours
  predictions: PredictionResult[];
  modelPerformance: Record<string, ModelMetrics>;
  currentWeather: WeatherData;
}

export interface CurrentWeatherResponse {
  location: Location;
  current: WeatherData;
  lastUpdated: string;
}

// API Request types
export interface PredictionRequest {
  latitude: number;
  longitude: number;
  hours?: number; // default 24
}

export interface LocationSearchRequest {
  query: string;
  limit?: number;
}

export interface LocationSearchResponse {
  locations: Location[];
}

// Weather parameter types
export type WeatherParameter = 
  | 'temperature' 
  | 'humidity' 
  | 'pressure' 
  | 'windSpeed' 
  | 'cloudCover' 
  | 'precipitation';

// Model types
export type ModelType = 
  | 'lstm' 
  | 'arima' 
  | 'randomForest' 
  | 'fuzzyTimeSeries' 
  | 'lube' 
  | 'ensemble';

export interface ModelStatus {
  name: string;
  type: ModelType;
  isLoaded: boolean;
  lastTrained: string;
  performance: ModelMetrics;
  status: 'healthy' | 'degraded' | 'failed';
}

export interface ModelStatusResponse {
  models: ModelStatus[];
  systemHealth: 'healthy' | 'degraded' | 'failed';
}

// Error types
export interface APIError {
  message: string;
  code: string;
  details?: Record<string, any>;
}

export interface WeatherAPIError extends APIError {
  source: 'openmeteo' | 'weatherapi' | 'openweathermap';
  retryable: boolean;
}