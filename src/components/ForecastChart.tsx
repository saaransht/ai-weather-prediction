'use client'

import { useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions,
  TooltipItem
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { PredictionResult, WeatherParameter } from '@/types/weather'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

interface ForecastChartProps {
  predictions?: any
  isLoading?: boolean
  data?: PredictionResult[]
  parameter?: WeatherParameter
  showUncertainty?: boolean
  height?: number
  className?: string
}

export default function ForecastChart({
  predictions,
  isLoading,
  data,
  parameter,
  showUncertainty = true,
  height = 300,
  className = ''
}: ForecastChartProps) {
  const chartRef = useRef<ChartJS<'line'>>(null)

  const getParameterConfig = (param: WeatherParameter) => {
    const configs = {
      temperature: {
        label: 'Temperature',
        unit: '°C',
        color: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        min: undefined,
        max: undefined
      },
      humidity: {
        label: 'Humidity',
        unit: '%',
        color: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        min: 0,
        max: 100
      },
      pressure: {
        label: 'Pressure',
        unit: 'hPa',
        color: 'rgb(107, 114, 128)',
        backgroundColor: 'rgba(107, 114, 128, 0.1)',
        min: undefined,
        max: undefined
      },
      windSpeed: {
        label: 'Wind Speed',
        unit: 'km/h',
        color: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        min: 0,
        max: undefined
      },
      cloudCover: {
        label: 'Cloud Cover',
        unit: '%',
        color: 'rgb(156, 163, 175)',
        backgroundColor: 'rgba(156, 163, 175, 0.1)',
        min: 0,
        max: 100
      },
      precipitation: {
        label: 'Precipitation',
        unit: 'mm',
        color: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        min: 0,
        max: undefined
      }
    }
    return configs[param]
  }

  const config = getParameterConfig(parameter || 'temperature')

  // Prepare chart data
  const labels = (data || []).map(item => {
    const date = new Date(item.timestamp)
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    })
  })

  const values = (data || []).map(item => {
    switch (parameter) {
      case 'temperature':
        return item.temperature
      case 'humidity':
        return item.humidity
      case 'pressure':
        return item.pressure
      case 'windSpeed':
        return item.windSpeed * 3.6 // Convert m/s to km/h
      default:
        return item.temperature
    }
  })

  const lowerBounds = showUncertainty ? (data || []).map(item => item.uncertainty.lowerBound) : []
  const upperBounds = showUncertainty ? (data || []).map(item => item.uncertainty.upperBound) : []

  const datasets = [
    {
      label: config.label,
      data: values,
      borderColor: config.color,
      backgroundColor: config.backgroundColor,
      borderWidth: 2,
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0.4,
      fill: false
    }
  ]

  // Add uncertainty bands if enabled
  if (showUncertainty && lowerBounds.length > 0 && upperBounds.length > 0) {
    datasets.push(
      {
        label: 'Upper Bound',
        data: upperBounds,
        borderColor: 'transparent',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 0,
        pointRadius: 0,
        pointHoverRadius: 0,
        tension: 0.4,
        fill: false
      },
      {
        label: 'Lower Bound',
        data: lowerBounds,
        borderColor: 'transparent',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 0,
        pointRadius: 0,
        pointHoverRadius: 0,
        tension: 0.4,
        fill: false
      }
    )
  }

  const chartData = {
    labels,
    datasets
  }

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: showUncertainty,
        position: 'top' as const,
        labels: {
          filter: (legendItem) => {
            return legendItem.text !== 'Upper Bound' && legendItem.text !== 'Lower Bound'
          }
        }
      },
      title: {
        display: true,
        text: `${config.label} Forecast`,
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: (context: TooltipItem<'line'>) => {
            const datasetLabel = context.dataset.label
            const value = context.parsed.y
            
            if (datasetLabel === config.label) {
              return `${datasetLabel}: ${value?.toFixed(1) || 'N/A'}${config.unit}`
            }
            
            return `${datasetLabel}: ${value?.toFixed(1) || 'N/A'}${config.unit}`
          },
          afterBody: (tooltipItems) => {
            const index = tooltipItems[0]?.dataIndex
            if (index !== undefined && showUncertainty) {
              const lower = lowerBounds[index]
              const upper = upperBounds[index]
              if (lower !== undefined && upper !== undefined) {
                return [
                  '',
                  `Confidence Range:`,
                  `${lower.toFixed(1)}${config.unit} - ${upper.toFixed(1)}${config.unit}`
                ]
              }
            }
            return []
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time'
        },
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: `${config.label} (${config.unit})`
        },
        min: config.min,
        max: config.max,
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    },
    elements: {
      point: {
        hoverBackgroundColor: config.color,
        hoverBorderColor: '#ffffff',
        hoverBorderWidth: 2
      }
    }
  }

  return (
    <div className={`bg-white rounded-lg p-4 ${className}`}>
      <div style={{ height: `${height}px` }}>
        <Line ref={chartRef} data={chartData} options={options} />
      </div>
    </div>
  )
}

// Multi-parameter chart component
interface MultiParameterChartProps {
  data: PredictionResult[]
  parameters: WeatherParameter[]
  showUncertainty?: boolean
  height?: number
  className?: string
}

export function MultiParameterChart({
  data,
  parameters,
  showUncertainty = false,
  height = 400,
  className = ''
}: MultiParameterChartProps) {
  const chartRef = useRef<ChartJS<'line'>>(null)

  const parameterConfigs = {
    temperature: { color: 'rgb(239, 68, 68)', yAxisID: 'y' },
    humidity: { color: 'rgb(59, 130, 246)', yAxisID: 'y1' },
    pressure: { color: 'rgb(107, 114, 128)', yAxisID: 'y2' },
    windSpeed: { color: 'rgb(34, 197, 94)', yAxisID: 'y3' },
    cloudCover: { color: 'rgb(156, 163, 175)', yAxisID: 'y4' },
    precipitation: { color: 'rgb(99, 102, 241)', yAxisID: 'y5' }
  }

  const getParameterConfig = (param: WeatherParameter) => {
    const configs = {
      temperature: {
        label: 'Temperature',
        unit: '°C',
        color: 'rgb(239, 68, 68)',
        yAxisID: 'y'
      },
      humidity: {
        label: 'Humidity',
        unit: '%',
        color: 'rgb(59, 130, 246)',
        yAxisID: 'y1'
      },
      pressure: {
        label: 'Pressure',
        unit: ' hPa',
        color: 'rgb(107, 114, 128)',
        yAxisID: 'y2'
      },
      windSpeed: {
        label: 'Wind Speed',
        unit: ' m/s',
        color: 'rgb(34, 197, 94)',
        yAxisID: 'y3'
      },
      cloudCover: {
        label: 'Cloud Cover',
        unit: '%',
        color: 'rgb(156, 163, 175)',
        yAxisID: 'y4'
      },
      precipitation: {
        label: 'Precipitation',
        unit: ' mm',
        color: 'rgb(99, 102, 241)',
        yAxisID: 'y5'
      }
    }
    return configs[param] || configs.temperature
  }

  const labels = (data || []).map(item => {
    const date = new Date(item.timestamp)
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    })
  })

  const datasets = parameters.map(param => {
    const config = parameterConfigs[param as keyof typeof parameterConfigs]
    const paramConfig = getParameterConfig(param)
    
    const values = (data || []).map(item => {
      switch (param) {
        case 'temperature':
          return item.temperature
        case 'humidity':
          return item.humidity
        case 'pressure':
          return item.pressure
        case 'windSpeed':
          return item.windSpeed * 3.6
        default:
          return item.temperature
      }
    })

    return {
      label: paramConfig.label,
      data: values,
      borderColor: config.color,
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.4,
      yAxisID: config.yAxisID
    }
  })

  const chartData = {
    labels,
    datasets
  }

  const scales: any = {
    x: {
      display: true,
      title: {
        display: true,
        text: 'Time'
      }
    }
  }

  // Add Y axes for each parameter
  parameters.forEach((param, index) => {
    const config = getParameterConfig(param)
    const yAxisId = `y${index === 0 ? '' : index}`
    
    scales[yAxisId] = {
      type: 'linear',
      display: true,
      position: index % 2 === 0 ? 'left' : 'right',
      title: {
        display: true,
        text: `${config.label} (${config.unit})`
      },
      grid: {
        drawOnChartArea: index === 0,
      }
    }
  })

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const
      },
      title: {
        display: true,
        text: 'Multi-Parameter Weather Forecast',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales,
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  }

  return (
    <div className={`bg-white rounded-lg p-4 ${className}`}>
      <div style={{ height: `${height}px` }}>
        <Line ref={chartRef} data={chartData} options={options} />
      </div>
    </div>
  )
}