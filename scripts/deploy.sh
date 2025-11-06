#!/bin/bash

# AI Weather Prediction Deployment Script

set -e

echo "🚀 Starting AI Weather Prediction deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18 or later."
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm."
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9 or later."
        exit 1
    fi
    
    print_status "All dependencies are installed ✓"
}

# Install frontend dependencies
install_frontend() {
    print_status "Installing frontend dependencies..."
    npm ci
    print_status "Frontend dependencies installed ✓"
}

# Install backend dependencies
install_backend() {
    print_status "Installing backend dependencies..."
    cd backend
    
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    cd ..
    print_status "Backend dependencies installed ✓"
}

# Build frontend
build_frontend() {
    print_status "Building frontend..."
    npm run build
    print_status "Frontend built successfully ✓"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    # Frontend tests
    if [ -f "package.json" ] && grep -q "test" package.json; then
        print_status "Running frontend tests..."
        npm test -- --passWithNoTests
    fi
    
    # Backend tests
    cd backend
    if [ -f "requirements.txt" ] && grep -q "pytest" requirements.txt; then
        print_status "Running backend tests..."
        source venv/bin/activate
        python -m pytest tests/ -v --tb=short
    fi
    cd ..
    
    print_status "All tests passed ✓"
}

# Deploy to Vercel (Frontend)
deploy_frontend() {
    print_status "Deploying frontend to Vercel..."
    
    if ! command -v vercel &> /dev/null; then
        print_warning "Vercel CLI not found. Installing..."
        npm install -g vercel
    fi
    
    # Deploy to Vercel
    vercel --prod
    
    print_status "Frontend deployed to Vercel ✓"
}

# Deploy to Railway (Backend)
deploy_backend() {
    print_status "Deploying backend to Railway..."
    
    cd backend
    
    if ! command -v railway &> /dev/null; then
        print_warning "Railway CLI not found. Please install it manually:"
        print_warning "npm install -g @railway/cli"
        print_warning "Then run: railway login && railway link"
        cd ..
        return 1
    fi
    
    # Deploy to Railway
    railway up
    
    cd ..
    print_status "Backend deployed to Railway ✓"
}

# Local development setup
setup_local() {
    print_status "Setting up local development environment..."
    
    # Copy environment file
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_status "Created .env file from .env.example"
            print_warning "Please update .env file with your configuration"
        fi
    fi
    
    # Start services with Docker Compose
    if command -v docker-compose &> /dev/null; then
        print_status "Starting services with Docker Compose..."
        docker-compose up -d
        print_status "Local development environment is ready ✓"
        print_status "Frontend: http://localhost:3000"
        print_status "Backend: http://localhost:8000"
        print_status "API Docs: http://localhost:8000/docs"
    else
        print_warning "Docker Compose not found. Starting services manually..."
        
        # Start backend
        cd backend
        source venv/bin/activate
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
        BACKEND_PID=$!
        cd ..
        
        # Start frontend
        npm run dev &
        FRONTEND_PID=$!
        
        print_status "Services started manually"
        print_status "Frontend PID: $FRONTEND_PID"
        print_status "Backend PID: $BACKEND_PID"
        
        # Save PIDs for cleanup
        echo $BACKEND_PID > .backend.pid
        echo $FRONTEND_PID > .frontend.pid
    fi
}

# Main deployment function
main() {
    case "${1:-help}" in
        "local")
            check_dependencies
            install_frontend
            install_backend
            setup_local
            ;;
        "build")
            check_dependencies
            install_frontend
            install_backend
            build_frontend
            ;;
        "test")
            check_dependencies
            install_frontend
            install_backend
            run_tests
            ;;
        "deploy")
            check_dependencies
            install_frontend
            install_backend
            build_frontend
            run_tests
            deploy_frontend
            deploy_backend
            ;;
        "frontend")
            check_dependencies
            install_frontend
            build_frontend
            deploy_frontend
            ;;
        "backend")
            check_dependencies
            install_backend
            deploy_backend
            ;;
        "help"|*)
            echo "AI Weather Prediction Deployment Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  local     - Set up local development environment"
            echo "  build     - Build the application"
            echo "  test      - Run all tests"
            echo "  deploy    - Deploy both frontend and backend"
            echo "  frontend  - Deploy only frontend to Vercel"
            echo "  backend   - Deploy only backend to Railway"
            echo "  help      - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 local    # Start local development"
            echo "  $0 deploy   # Full deployment"
            echo "  $0 test     # Run tests"
            ;;
    esac
}

# Run main function with all arguments
main "$@"