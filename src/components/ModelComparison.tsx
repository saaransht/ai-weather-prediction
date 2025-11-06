'use client'

import { useState } from 'react'
import { Bar, Doughnut } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js'
import { TrendingUp, TrendingDown, Minus, Info } from 'lucide-react'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
)

interface ModelMetrics {
  mae: number
  rmse: number
  mape: number
  lastUpdated: string
}

interface ModelComparisonProps {
  modelStatus?: any
  predictions?: any
  isLoading?: boolean
  modelPerformance?: Record<string, ModelMetrics>
  modelContributions?: Record<string, number>
  className?: string
}

export default function ModelComparison({
  modelStatus,
  predictions,
  isLoading,
  modelPerformance,
  modelContributions,
  className = ''
}: ModelComparisonProps) {
  const [selectedMetric, setSelectedMetric] = useState<'mae' | 'rmse' | 'mape'>('mae')

  const modelNames = Object.keys(modelPerformance || {})
  const modelColors = {
    lstm: '#ef4444',
    arima: '#3b82f6',
    random_forest: '#10b981',
    fuzzy: '#f59e0b',
    lube: '#8b5cf6',
    ensemble: '#6b7280'
  }

  const getModelDisplayName = (modelName: string) => {
    const displayNames: Record<string, string> = {
      lstm: 'LSTM',
      arima: 'ARIMA',
      random_forest: 'Random Forest',
      fuzzy: 'Fuzzy Time Series',
      lube: 'LUBE',
      ensemble: 'Ensemble'
    }
    return displayNames[modelName] || modelName.toUpperCase()
  }

  const getMetricDescription = (metric: string) => {
    const descriptions = {
      mae: 'Mean Absolute Error - Average prediction error magnitude',
      rmse: 'Root Mean Square Error - Penalizes larger errors more heavily',
      mape: 'Mean Absolute Percentage Error - Error as percentage of actual values'
    }
    return descriptions[metric as keyof typeof descriptions] || ''
  }

  const getPerformanceTrend = (value: number, metric: string) => {
    // Lower values are better for all these metrics
    if (value < 2) return { icon: TrendingUp, color: 'text-green-500', label: 'Excellent' }
    if (value < 5) return { icon: Minus, color: 'text-yellow-500', label: 'Good' }
    return { icon: TrendingDown, color: 'text-red-500', label: 'Needs Improvement' }
  }

  // Prepare performance comparison chart data
  const performanceData = {
    labels: modelNames.map(getModelDisplayName),
    datasets: [
      {
        label: selectedMetric.toUpperCase(),
        data: modelNames.map(name => modelPerformance?.[name]?.[selectedMetric] || 0),
        backgroundColor: modelNames.map(name => modelColors[name as keyof typeof modelColors] || '#6b7280'),
        borderColor: modelNames.map(name => modelColors[name as keyof typeof modelColors] || '#6b7280'),
        borderWidth: 1
      }
    ]
  }

  const performanceOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: `Model Performance Comparison (${selectedMetric.toUpperCase()})`,
        font: {
          size: 14,
          weight: 'bold'
        }
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const value = context.parsed.y
            return `${selectedMetric.toUpperCase()}: ${value?.toFixed(3) || 'N/A'}`
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: selectedMetric.toUpperCase()
        }
      }
    }
  }

  // Prepare model contributions chart data
  const contributionsData = modelContributions ? {
    labels: Object.keys(modelContributions).map(getModelDisplayName),
    datasets: [
      {
        data: Object.values(modelContributions).map(val => val * 100),
        backgroundColor: Object.keys(modelContributions).map(
          name => modelColors[name as keyof typeof modelColors] || '#6b7280'
        ),
        borderWidth: 2,
        borderColor: '#ffffff'
      }
    ]
  } : null

  const contributionsOptions: ChartOptions<'doughnut'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: {
          padding: 20,
          usePointStyle: true
        }
      },
      title: {
        display: true,
        text: 'Model Contributions to Ensemble',
        font: {
          size: 14,
          weight: 'bold'
        }
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const value = context.parsed
            return `${context.label}: ${value.toFixed(1)}%`
          }
        }
      }
    }
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Performance Metrics Table */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Model Performance Metrics</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3 font-medium text-gray-900">Model</th>
                <th className="text-center py-2 px-3 font-medium text-gray-900">
                  MAE
                  <div className="text-xs text-gray-500 font-normal">Lower is better</div>
                </th>
                <th className="text-center py-2 px-3 font-medium text-gray-900">
                  RMSE
                  <div className="text-xs text-gray-500 font-normal">Lower is better</div>
                </th>
                <th className="text-center py-2 px-3 font-medium text-gray-900">
                  MAPE (%)
                  <div className="text-xs text-gray-500 font-normal">Lower is better</div>
                </th>
                <th className="text-center py-2 px-3 font-medium text-gray-900">Status</th>
              </tr>
            </thead>
            <tbody>
              {modelNames.map((modelName) => {
                const metrics = modelPerformance?.[modelName]
                const maeTrend = getPerformanceTrend(metrics?.mae || 0, 'mae')
                
                return (
                  <tr key={modelName} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: modelColors[modelName as keyof typeof modelColors] }}
                        />
                        <span className="font-medium">{getModelDisplayName(modelName)}</span>
                      </div>
                    </td>
                    <td className="text-center py-3 px-3 font-mono">
                      {metrics?.mae?.toFixed(3) || 'N/A'}
                    </td>
                    <td className="text-center py-3 px-3 font-mono">
                      {metrics?.rmse?.toFixed(3) || 'N/A'}
                    </td>
                    <td className="text-center py-3 px-3 font-mono">
                      {metrics?.mape?.toFixed(2) || 'N/A'}%
                    </td>
                    <td className="text-center py-3 px-3">
                      <div className="flex items-center justify-center gap-1">
                        <maeTrend.icon className={`w-4 h-4 ${maeTrend.color}`} />
                        <span className={`text-xs ${maeTrend.color}`}>
                          {maeTrend.label}
                        </span>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Performance Comparison Chart */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Performance Comparison</h3>
          
          <div className="flex items-center gap-2">
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value as 'mae' | 'rmse' | 'mape')}
              className="px-3 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="mae">MAE</option>
              <option value="rmse">RMSE</option>
              <option value="mape">MAPE</option>
            </select>
            
            <div className="group relative">
              <Info className="w-4 h-4 text-gray-400 cursor-help" />
              <div className="absolute right-0 top-6 w-64 p-2 bg-gray-900 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-10">
                {getMetricDescription(selectedMetric)}
              </div>
            </div>
          </div>
        </div>
        
        <div style={{ height: '300px' }}>
          <Bar data={performanceData} options={performanceOptions} />
        </div>
      </div>

      {/* Model Contributions Chart */}
      {contributionsData && (
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h3 className="text-lg font-semibold mb-4">Ensemble Model Contributions</h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div style={{ height: '300px' }}>
              <Doughnut data={contributionsData} options={contributionsOptions} />
            </div>
            
            <div className="space-y-3">
              <h4 className="font-medium text-gray-900">Contribution Details</h4>
              {Object.entries(modelContributions || {}).map(([modelName, contribution]) => (
                <div key={modelName} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: modelColors[modelName as keyof typeof modelColors] }}
                    />
                    <span className="font-medium">{getModelDisplayName(modelName)}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">{(contribution * 100).toFixed(1)}%</div>
                    <div className="text-xs text-gray-500">
                      {contribution > 0.3 ? 'High' : contribution > 0.15 ? 'Medium' : 'Low'} influence
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Last Updated Info */}
      <div className="text-xs text-gray-500 text-center">
        Metrics last updated: {new Date(Object.values(modelPerformance || {})[0]?.lastUpdated || '').toLocaleString()}
      </div>
    </div>
  )
}