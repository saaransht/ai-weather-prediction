# AI Weather Prediction System

A production-ready full-stack web application for numerical weather prediction using multiple AI/ML models with uncertainty quantification.

## 🌟 Features

- **Multi-Model Forecasting**: LSTM, ARIMA/SARIMA, Random Forest, Fuzzy Time Series
- **Uncertainty Quantification**: LUBE (Lower-Upper Bound Estimation) for confidence intervals
- **Real-time Data**: Integration with multiple weather APIs (Open-Meteo, WeatherAPI, OpenWeatherMap)
- **Modern UI**: Responsive Next.js frontend with interactive charts
- **Global Coverage**: Support for worldwide locations with Indian cities prioritized
- **Free Deployment**: Designed for Vercel (frontend) + Railway (backend)

## 🏗️ Architecture

- **Frontend**: Next.js 14 with TypeScript, TailwindCSS, Chart.js
- **Backend**: FastAPI with Python, async support
- **ML Stack**: PyTorch, Scikit-learn, Statsmodels
- **Database**: SQLite + Redis caching
- **APIs**: Open-Meteo (primary), WeatherAPI, OpenWeatherMap (fallbacks)

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- Redis (optional, for caching)

### Easy Setup (Recommended)

Use our deployment scripts for quick setup:

**Windows:**
```powershell
.\scripts\deploy.ps1 local
```

**Linux/macOS:**
```bash
./scripts/deploy.sh local
```

This will:
- Install all dependencies
- Set up environment files
- Start both frontend and backend
- Open the application in your browser

### Manual Setup

#### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Visit `http://localhost:3000`

#### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

### Environment Configuration

1. Copy `.env.example` to `.env`
2. Configure API keys (optional for free tiers):
   ```env
   OPENWEATHERMAP_API_KEY=your_key_here
   WEATHERAPI_KEY=your_key_here
   ```
3. Adjust other settings as needed

## 📊 ML Models

### 1. LSTM (Long Short-Term Memory)
- **Purpose**: Deep learning time-series prediction
- **Input**: 7-day historical window (168 hours)
- **Output**: 24-hour hourly predictions
- **Features**: Monte Carlo dropout for uncertainty

### 2. ARIMA/SARIMA
- **Purpose**: Statistical baseline model
- **Features**: Automatic parameter optimization, seasonality detection
- **Output**: Statistical confidence intervals
- **Best for**: Stable weather patterns

### 3. Random Forest
- **Purpose**: Feature-engineered ensemble model
- **Features**: 50+ engineered features, automatic selection
- **Output**: Tree-based uncertainty estimation
- **Best for**: Complex weather interactions

### 4. Fuzzy Time Series
- **Purpose**: Rule-based forecasting
- **Features**: Linguistic weather patterns, fuzzy logic
- **Output**: Rule-based predictions
- **Best for**: Interpretable forecasts

### 5. LUBE (Lower-Upper Bound Estimation)
- **Purpose**: Neural uncertainty quantification
- **Features**: Calibrated prediction intervals
- **Output**: Lower/upper bounds with confidence levels
- **Best for**: Reliable uncertainty estimates

### 6. Ensemble System
- **Purpose**: Combines all models intelligently
- **Features**: Adaptive weighting, performance tracking
- **Output**: Optimized predictions with uncertainty
- **Best for**: Maximum accuracy and reliability

## 🔄 API Endpoints

### Weather Data
```http
GET /api/weather/current?lat={lat}&lon={lon}
```
Get current weather conditions

```http
GET /api/weather/locations/search?q={query}
```
Search for locations

```http
POST /api/predictions/forecast
Content-Type: application/json

{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "hours": 24
}
```
Get 24-hour weather predictions

### Model Management
```http
GET /api/predictions/models/status
```
Get model health and performance status

```http
GET /api/weather/health
```
Check API health status

## 🧪 Testing

### Run All Tests
```bash
# Using deployment script
./scripts/deploy.sh test

# Or manually
npm test                    # Frontend tests
cd backend && pytest       # Backend tests
```

### Test Coverage
```bash
npm run test:coverage       # Frontend coverage
cd backend && pytest --cov # Backend coverage
```

### Model Testing
```bash
cd backend
pytest tests/test_models.py -v
```

## 🚀 Deployment

### Automated Deployment

**Full deployment:**
```bash
./scripts/deploy.sh deploy
```

**Frontend only:**
```bash
./scripts/deploy.sh frontend
```

**Backend only:**
```bash
./scripts/deploy.sh backend
```

### Manual Deployment

#### Frontend (Vercel)
1. Install Vercel CLI: `npm install -g vercel`
2. Login: `vercel login`
3. Deploy: `vercel --prod`

#### Backend (Railway)
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Link project: `railway link`
4. Deploy: `railway up`

### Environment Variables

**Frontend (Vercel):**
- `BACKEND_URL`: Backend API URL

**Backend (Railway):**
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string
- `OPENWEATHERMAP_API_KEY`: OpenWeatherMap API key (optional)
- `WEATHERAPI_KEY`: WeatherAPI key (optional)

## 📈 Performance

- **Prediction Speed**: <5 seconds end-to-end
- **API Response**: <2 seconds for weather data
- **Model Training**: Automated with performance monitoring
- **Caching**: 15-minute weather data, 1-hour predictions
- **Scalability**: Horizontal scaling support

## 🛠️ Development

### Project Structure
```
├── src/                    # Next.js frontend
│   ├── app/               # App router pages
│   ├── components/        # React components
│   ├── hooks/             # Custom React hooks
│   ├── lib/              # Utilities
│   └── types/            # TypeScript types
├── backend/               # FastAPI backend
│   ├── app/              # Application code
│   │   ├── api/          # API routes
│   │   ├── models/       # ML models
│   │   ├── services/     # Business logic
│   │   └── ml/           # ML utilities
│   └── tests/            # Test suite
├── scripts/               # Deployment scripts
├── .github/              # CI/CD workflows
└── docs/                 # Documentation
```

### Development Workflow

1. **Setup**: `./scripts/deploy.sh local`
2. **Code**: Make changes to frontend/backend
3. **Test**: `npm test` and `pytest`
4. **Lint**: `npm run lint` and `flake8`
5. **Commit**: Git hooks run automatically
6. **Deploy**: CI/CD handles deployment

### Adding New Models

1. Create model class inheriting from `WeatherModel`
2. Implement required methods: `train()`, `predict()`, `predict_with_uncertainty()`
3. Add model to ensemble configuration
4. Write tests in `tests/test_models.py`
5. Update documentation

### Code Quality

- **TypeScript**: Strict type checking
- **ESLint**: Code linting and formatting
- **Prettier**: Code formatting
- **Husky**: Git hooks for quality checks
- **Jest**: Frontend testing
- **Pytest**: Backend testing

## 🔧 Configuration

### Model Configuration
```python
# backend/app/core/config.py
LSTM_CONFIG = {
    'sequence_length': 168,  # 7 days
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2
}
```

### API Configuration
```python
# Rate limiting, caching, CORS settings
RATE_LIMIT_REQUESTS = 100
CACHE_TTL_SECONDS = 900
ALLOWED_ORIGINS = ["http://localhost:3000"]
```

## 🐛 Troubleshooting

### Common Issues

**Frontend won't start:**
- Check Node.js version (18+)
- Run `npm install` to install dependencies
- Check for port conflicts (3000)

**Backend won't start:**
- Check Python version (3.9+)
- Activate virtual environment
- Install requirements: `pip install -r requirements.txt`
- Check for port conflicts (8000)

**Models not training:**
- Check available memory (models need 2GB+)
- Verify data quality and quantity
- Check logs for specific errors

**API errors:**
- Verify API keys in `.env` file
- Check network connectivity
- Review API rate limits

### Debug Mode

Enable debug logging:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

### Performance Issues

- Enable Redis caching
- Increase model batch sizes
- Use GPU acceleration for PyTorch models
- Monitor memory usage

## 📚 Documentation

- **API Documentation**: Visit `/docs` endpoint when backend is running
- **Model Documentation**: See `backend/app/models/` for detailed model docs
- **Component Documentation**: See `src/components/` for React component docs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run quality checks: `npm run lint && npm test`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open Pull Request

### Contribution Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation
- Ensure CI/CD passes
- Keep commits atomic and well-described

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🤝 Support

**Getting Help:**
- 📖 Check this README and documentation
- 🐛 Create GitHub issue for bugs
- 💡 Create GitHub discussion for questions
- 📧 Contact maintainers for urgent issues

**Resources:**
- API Documentation: `http://localhost:8000/docs`
- Frontend Storybook: `npm run storybook`
- Backend Tests: `pytest --html=report.html`

## 🏆 Acknowledgments

- OpenMeteo for free weather API
- Vercel for frontend hosting
- Railway for backend hosting
- Open source ML libraries (PyTorch, Scikit-learn, etc.)

---

**Built with ❤️ for accurate weather prediction**