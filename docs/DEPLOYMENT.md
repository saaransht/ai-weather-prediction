# Deployment Guide

This guide covers deploying the AI Weather Prediction system to production environments.

## Overview

The system consists of two main components:
- **Frontend**: Next.js application deployed to Vercel
- **Backend**: FastAPI application deployed to Railway

## Prerequisites

- GitHub account
- Vercel account (free tier available)
- Railway account (free tier available)
- Domain name (optional)

## Quick Deployment

### Automated Deployment Script

The easiest way to deploy is using our deployment script:

```bash
# Full deployment (frontend + backend)
./scripts/deploy.sh deploy

# Frontend only
./scripts/deploy.sh frontend

# Backend only  
./scripts/deploy.sh backend
```

### Manual Deployment Steps

## Frontend Deployment (Vercel)

### 1. Prepare Repository

Ensure your code is pushed to GitHub:

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2. Connect to Vercel

1. Visit [vercel.com](https://vercel.com)
2. Sign up/login with GitHub
3. Click "New Project"
4. Import your GitHub repository
5. Configure project settings:
   - **Framework Preset**: Next.js
   - **Root Directory**: `./` (leave empty)
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`

### 3. Environment Variables

Add these environment variables in Vercel dashboard:

```env
BACKEND_URL=https://your-backend-url.railway.app
NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
```

### 4. Deploy

Click "Deploy" and wait for the build to complete.

### 5. Custom Domain (Optional)

1. Go to Project Settings → Domains
2. Add your custom domain
3. Configure DNS records as instructed

## Backend Deployment (Railway)

### 1. Prepare Backend

Ensure your backend code is ready:

```bash
cd backend
# Test locally first
uvicorn app.main:app --reload
```

### 2. Connect to Railway

1. Visit [railway.app](https://railway.app)
2. Sign up/login with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Select the backend directory

### 3. Configure Build Settings

Railway should auto-detect the Python app. If not, configure:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Root Directory**: `backend`

### 4. Environment Variables

Add these environment variables in Railway dashboard:

```env
# Required
DATABASE_URL=sqlite:///./weather_data.db
REDIS_URL=redis://redis:6379

# Optional API Keys
OPENWEATHERMAP_API_KEY=your_key_here
WEATHERAPI_KEY=your_key_here

# Configuration
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=["https://your-frontend-url.vercel.app"]
```

### 5. Add Redis Service

1. In Railway dashboard, click "New Service"
2. Select "Redis"
3. Deploy Redis instance
4. Update `REDIS_URL` environment variable with the provided URL

### 6. Deploy

Railway will automatically deploy when you push to the main branch.

## Alternative Deployment Options

### Docker Deployment

#### Using Docker Compose (Local/VPS)

```bash
# Clone repository
git clone <your-repo-url>
cd ai-weather-prediction

# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

#### Individual Docker Containers

**Backend:**
```bash
cd backend
docker build -t ai-weather-backend .
docker run -p 8000:8000 \
  -e DATABASE_URL=sqlite:///./weather_data.db \
  ai-weather-backend
```

**Frontend:**
```bash
docker build -f Dockerfile.frontend -t ai-weather-frontend .
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  ai-weather-frontend
```

### Heroku Deployment

#### Backend (Heroku)

1. Install Heroku CLI
2. Create Heroku app:
   ```bash
   cd backend
   heroku create your-app-name
   ```

3. Add buildpack:
   ```bash
   heroku buildpacks:set heroku/python
   ```

4. Set environment variables:
   ```bash
   heroku config:set DATABASE_URL=sqlite:///./weather_data.db
   heroku config:set DEBUG=false
   ```

5. Deploy:
   ```bash
   git push heroku main
   ```

### AWS Deployment

#### Frontend (AWS Amplify)

1. Connect GitHub repository to AWS Amplify
2. Configure build settings:
   ```yaml
   version: 1
   frontend:
     phases:
       preBuild:
         commands:
           - npm ci
       build:
         commands:
           - npm run build
     artifacts:
       baseDirectory: .next
       files:
         - '**/*'
     cache:
       paths:
         - node_modules/**/*
   ```

#### Backend (AWS Lambda + API Gateway)

Use AWS SAM or Serverless Framework for Lambda deployment.

## Environment Configuration

### Production Environment Variables

#### Frontend (Vercel)
```env
BACKEND_URL=https://your-backend.railway.app
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
NODE_ENV=production
```

#### Backend (Railway)
```env
# Database
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://user:pass@host:port

# API Keys (Optional)
OPENWEATHERMAP_API_KEY=your_key
WEATHERAPI_KEY=your_key

# Security
ALLOWED_ORIGINS=["https://your-frontend.vercel.app"]
CORS_ALLOW_CREDENTIALS=true

# Performance
DEBUG=false
LOG_LEVEL=INFO
WORKERS=4

# Model Configuration
MODEL_STORAGE_PATH=/app/models
TRAINING_DATA_DAYS=30
```

### Security Configuration

#### CORS Settings
```python
# backend/app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

#### Rate Limiting
```python
# Configure in backend/app/core/config.py
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour
```

## Database Setup

### SQLite (Default)
- Suitable for small to medium deployments
- No additional setup required
- Data persists in container volumes

### PostgreSQL (Recommended for Production)

1. **Railway PostgreSQL:**
   ```bash
   # In Railway dashboard
   # Add PostgreSQL service
   # Copy connection URL to DATABASE_URL
   ```

2. **External PostgreSQL:**
   ```env
   DATABASE_URL=postgresql://user:password@host:port/database
   ```

3. **Migration:**
   ```bash
   cd backend
   # Run database migrations
   python -c "from app.db.database import create_tables; create_tables()"
   ```

### Redis Setup

#### Railway Redis
1. Add Redis service in Railway dashboard
2. Copy connection URL
3. Update `REDIS_URL` environment variable

#### External Redis
```env
REDIS_URL=redis://user:password@host:port/db
```

## Monitoring and Logging

### Application Monitoring

#### Vercel Analytics
- Enable in Vercel dashboard
- Monitor Core Web Vitals
- Track user interactions

#### Railway Metrics
- CPU and memory usage
- Request metrics
- Error rates

### Error Tracking

#### Sentry Integration
```bash
npm install @sentry/nextjs @sentry/python
```

**Frontend:**
```javascript
// next.config.js
const { withSentryConfig } = require('@sentry/nextjs')

module.exports = withSentryConfig({
  // Your Next.js config
}, {
  // Sentry config
})
```

**Backend:**
```python
# backend/app/main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()],
)
```

### Health Checks

#### Uptime Monitoring
Set up monitoring for:
- `https://your-frontend.vercel.app`
- `https://your-backend.railway.app/health`

#### Custom Health Checks
```python
# backend/app/api/health.py
@router.get("/health/detailed")
async def detailed_health():
    return {
        "status": "healthy",
        "database": check_database(),
        "redis": check_redis(),
        "models": check_models(),
        "apis": check_external_apis()
    }
```

## Performance Optimization

### Frontend Optimization

#### Next.js Configuration
```javascript
// next.config.js
module.exports = {
  experimental: {
    optimizeCss: true,
    optimizeImages: true,
  },
  images: {
    domains: ['your-api-domain.com'],
  },
  compress: true,
}
```

#### CDN Configuration
- Vercel automatically provides CDN
- Configure custom CDN if needed

### Backend Optimization

#### Gunicorn Configuration
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
max_requests = 1000
max_requests_jitter = 100
timeout = 30
```

#### Caching Strategy
```python
# Redis caching configuration
CACHE_TTL = {
    "weather_data": 900,      # 15 minutes
    "predictions": 3600,      # 1 hour
    "model_status": 300,      # 5 minutes
}
```

## Scaling

### Horizontal Scaling

#### Frontend (Vercel)
- Automatic scaling
- Global CDN distribution
- Edge functions for API routes

#### Backend (Railway)
- Increase instance count
- Configure load balancing
- Use Redis for session storage

### Vertical Scaling

#### Resource Allocation
```yaml
# railway.json
{
  "deploy": {
    "restartPolicyType": "ON_FAILURE",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100
  }
}
```

### Database Scaling

#### Read Replicas
```python
# Configure read/write splitting
DATABASE_WRITE_URL = "postgresql://..."
DATABASE_READ_URL = "postgresql://..."
```

#### Connection Pooling
```python
# SQLAlchemy configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)
```

## Backup and Recovery

### Database Backups

#### Automated Backups
```bash
# PostgreSQL backup script
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Upload to cloud storage
aws s3 cp backup_*.sql s3://your-backup-bucket/
```

#### Model Backups
```python
# Backup trained models
import boto3

def backup_models():
    s3 = boto3.client('s3')
    for model_file in os.listdir('/app/models'):
        s3.upload_file(
            f'/app/models/{model_file}',
            'your-model-bucket',
            f'models/{model_file}'
        )
```

### Disaster Recovery

#### Recovery Procedures
1. **Database Recovery:**
   ```bash
   psql $DATABASE_URL < backup_file.sql
   ```

2. **Model Recovery:**
   ```bash
   aws s3 sync s3://your-model-bucket/models/ /app/models/
   ```

3. **Configuration Recovery:**
   - Restore environment variables
   - Verify API keys
   - Test all endpoints

## Troubleshooting

### Common Deployment Issues

#### Build Failures
```bash
# Check build logs
vercel logs your-deployment-url

# Railway logs
railway logs
```

#### Environment Variables
```bash
# Verify environment variables
vercel env ls
railway variables
```

#### Database Connection
```python
# Test database connection
python -c "
from app.db.database import engine
try:
    engine.connect()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
"
```

### Performance Issues

#### Memory Usage
```bash
# Monitor memory usage
railway metrics

# Optimize model loading
# Load models on-demand instead of startup
```

#### API Response Times
```python
# Add request timing middleware
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## Security Considerations

### API Security
- Use HTTPS only
- Implement rate limiting
- Validate all inputs
- Use CORS properly

### Environment Security
- Never commit API keys
- Use environment variables
- Rotate keys regularly
- Monitor for security vulnerabilities

### Data Security
- Encrypt sensitive data
- Use secure database connections
- Implement proper access controls
- Regular security audits

## Maintenance

### Regular Tasks
- Monitor application health
- Update dependencies
- Backup data regularly
- Review logs for errors
- Update API keys as needed

### Updates and Rollbacks
```bash
# Deploy new version
git push origin main

# Rollback if needed (Vercel)
vercel rollback

# Rollback (Railway)
railway rollback
```

This deployment guide should help you successfully deploy the AI Weather Prediction system to production. For additional support, refer to the platform-specific documentation or create an issue in the repository.