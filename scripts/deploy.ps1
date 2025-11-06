# AI Weather Prediction Deployment Script for Windows
param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# Colors for output
$ErrorColor = "Red"
$SuccessColor = "Green"
$WarningColor = "Yellow"
$InfoColor = "Cyan"

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $InfoColor
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $SuccessColor
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $WarningColor
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $ErrorColor
}

function Test-Dependencies {
    Write-Status "Checking dependencies..."
    
    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Success "Node.js found: $nodeVersion"
    }
    catch {
        Write-Error "Node.js is not installed. Please install Node.js 18 or later."
        exit 1
    }
    
    # Check npm
    try {
        $npmVersion = npm --version
        Write-Success "npm found: $npmVersion"
    }
    catch {
        Write-Error "npm is not installed. Please install npm."
        exit 1
    }
    
    # Check Python
    try {
        $pythonVersion = python --version
        Write-Success "Python found: $pythonVersion"
    }
    catch {
        Write-Error "Python is not installed. Please install Python 3.9 or later."
        exit 1
    }
    
    Write-Success "All dependencies are installed ✓"
}

function Install-Frontend {
    Write-Status "Installing frontend dependencies..."
    npm ci
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Frontend dependencies installed ✓"
    } else {
        Write-Error "Failed to install frontend dependencies"
        exit 1
    }
}

function Install-Backend {
    Write-Status "Installing backend dependencies..."
    Set-Location backend
    
    if (!(Test-Path "venv")) {
        Write-Status "Creating Python virtual environment..."
        python -m venv venv
    }
    
    # Activate virtual environment
    & "venv\Scripts\Activate.ps1"
    
    # Upgrade pip and install requirements
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Backend dependencies installed ✓"
    } else {
        Write-Error "Failed to install backend dependencies"
        Set-Location ..
        exit 1
    }
    
    Set-Location ..
}

function Build-Frontend {
    Write-Status "Building frontend..."
    npm run build
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Frontend built successfully ✓"
    } else {
        Write-Error "Frontend build failed"
        exit 1
    }
}

function Run-Tests {
    Write-Status "Running tests..."
    
    # Frontend tests
    if (Test-Path "package.json") {
        $packageJson = Get-Content "package.json" | ConvertFrom-Json
        if ($packageJson.scripts.test) {
            Write-Status "Running frontend tests..."
            npm test -- --passWithNoTests
        }
    }
    
    # Backend tests
    Set-Location backend
    if (Test-Path "requirements.txt") {
        $requirements = Get-Content "requirements.txt"
        if ($requirements -match "pytest") {
            Write-Status "Running backend tests..."
            & "venv\Scripts\Activate.ps1"
            python -m pytest tests/ -v --tb=short
        }
    }
    Set-Location ..
    
    Write-Success "All tests completed ✓"
}

function Deploy-Frontend {
    Write-Status "Deploying frontend to Vercel..."
    
    # Check if Vercel CLI is installed
    try {
        vercel --version | Out-Null
    }
    catch {
        Write-Warning "Vercel CLI not found. Installing..."
        npm install -g vercel
    }
    
    # Deploy to Vercel
    vercel --prod
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Frontend deployed to Vercel ✓"
    } else {
        Write-Error "Frontend deployment failed"
        exit 1
    }
}

function Deploy-Backend {
    Write-Status "Deploying backend to Railway..."
    
    Set-Location backend
    
    # Check if Railway CLI is installed
    try {
        railway version | Out-Null
    }
    catch {
        Write-Warning "Railway CLI not found. Please install it manually:"
        Write-Warning "npm install -g @railway/cli"
        Write-Warning "Then run: railway login && railway link"
        Set-Location ..
        return
    }
    
    # Deploy to Railway
    railway up
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Backend deployed to Railway ✓"
    } else {
        Write-Error "Backend deployment failed"
    }
    
    Set-Location ..
}

function Setup-Local {
    Write-Status "Setting up local development environment..."
    
    # Copy environment file
    if (!(Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Success "Created .env file from .env.example"
            Write-Warning "Please update .env file with your configuration"
        }
    }
    
    # Check for Docker
    try {
        docker-compose --version | Out-Null
        Write-Status "Starting services with Docker Compose..."
        docker-compose up -d
        Write-Success "Local development environment is ready ✓"
        Write-Success "Frontend: http://localhost:3000"
        Write-Success "Backend: http://localhost:8000"
        Write-Success "API Docs: http://localhost:8000/docs"
    }
    catch {
        Write-Warning "Docker Compose not found. Starting services manually..."
        
        # Start backend
        Write-Status "Starting backend..."
        Set-Location backend
        & "venv\Scripts\Activate.ps1"
        Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000" -WindowStyle Hidden
        Set-Location ..
        
        # Start frontend
        Write-Status "Starting frontend..."
        Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WindowStyle Hidden
        
        Write-Success "Services started manually"
        Write-Success "Frontend: http://localhost:3000"
        Write-Success "Backend: http://localhost:8000"
    }
}

function Show-Help {
    Write-Host "AI Weather Prediction Deployment Script" -ForegroundColor $InfoColor
    Write-Host ""
    Write-Host "Usage: .\scripts\deploy.ps1 [command]" -ForegroundColor $InfoColor
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor $InfoColor
    Write-Host "  local     - Set up local development environment"
    Write-Host "  build     - Build the application"
    Write-Host "  test      - Run all tests"
    Write-Host "  deploy    - Deploy both frontend and backend"
    Write-Host "  frontend  - Deploy only frontend to Vercel"
    Write-Host "  backend   - Deploy only backend to Railway"
    Write-Host "  help      - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor $InfoColor
    Write-Host "  .\scripts\deploy.ps1 local    # Start local development"
    Write-Host "  .\scripts\deploy.ps1 deploy   # Full deployment"
    Write-Host "  .\scripts\deploy.ps1 test     # Run tests"
}

# Main execution
switch ($Command.ToLower()) {
    "local" {
        Test-Dependencies
        Install-Frontend
        Install-Backend
        Setup-Local
    }
    "build" {
        Test-Dependencies
        Install-Frontend
        Install-Backend
        Build-Frontend
    }
    "test" {
        Test-Dependencies
        Install-Frontend
        Install-Backend
        Run-Tests
    }
    "deploy" {
        Test-Dependencies
        Install-Frontend
        Install-Backend
        Build-Frontend
        Run-Tests
        Deploy-Frontend
        Deploy-Backend
    }
    "frontend" {
        Test-Dependencies
        Install-Frontend
        Build-Frontend
        Deploy-Frontend
    }
    "backend" {
        Test-Dependencies
        Install-Backend
        Deploy-Backend
    }
    default {
        Show-Help
    }
}