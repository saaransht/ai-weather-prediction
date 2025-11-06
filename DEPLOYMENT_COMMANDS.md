# Deployment Commands

## After creating GitHub repository, run these commands:

```bash
# Add your GitHub repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ai-weather-prediction.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

## Render Deployment Configuration

The following files have been created for Render deployment:

- `backend/render.yaml` - Render service configuration
- `backend/Procfile` - Process definition
- `backend/runtime.txt` - Python version specification
- `backend/start.sh` - Startup script

## Render Deployment Steps:

1. Go to https://render.com
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Select your `ai-weather-prediction` repository
5. Configure:
   - **Name**: `nwp-backend`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
6. Click "Create Web Service"

## After Backend Deployment:

1. Copy the backend URL (e.g., `https://nwp-backend-xxxx.onrender.com`)
2. Update `next.config.js` with the new backend URL
3. Redeploy frontend to Vercel

## Frontend URLs:
- **Current**: https://nwp-gtib5xtii-saaransh-tiwaris-projects.vercel.app
- **Backend**: Will be provided after Render deployment