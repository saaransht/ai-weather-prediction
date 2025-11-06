"""
ML prediction API endpoints.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from loguru import logger

from app.schemas.weather import (
    PredictionRequest, 
    PredictionResponse, 
    ModelStatusResponse,
    WeatherAPIError
)
from app.services.model_manager import ModelManager


router = APIRouter(prefix="/api/predictions", tags=["predictions"])

# Global model manager instance
model_manager = ModelManager()


async def get_model_manager() -> ModelManager:
    """Dependency to get model manager."""
    return model_manager


@router.post("/forecast", response_model=PredictionResponse)
async def predict_weather(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Generate weather predictions using AI/ML models.
    
    Args:
        request: Prediction request with location and parameters
        background_tasks: Background tasks for async operations
        manager: Model manager instance
        
    Returns:
        Weather predictions with uncertainty bounds
    """
    try:
        logger.info(f"Generating predictions for ({request.latitude}, {request.longitude})")
        
        # Generate predictions
        prediction_response = await manager.predict_weather(request)
        
        # Schedule background model performance update
        background_tasks.add_task(
            _update_model_performance, 
            manager, 
            request.latitude, 
            request.longitude
        )
        
        return prediction_response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Prediction failed: {str(e)}",
                "code": "PREDICTION_ERROR",
                "retryable": True
            }
        )


@router.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status(
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Get status of all ML models.
    
    Returns:
        Model status information including health and performance metrics
    """
    try:
        status = manager.get_model_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model status"
        )


@router.post("/models/retrain")
async def retrain_models(
    latitude: float,
    longitude: float,
    model_type: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Trigger model retraining for a specific location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        model_type: Specific model type to retrain (optional)
        background_tasks: Background tasks for async operations
        manager: Model manager instance
        
    Returns:
        Retraining status
    """
    try:
        logger.info(f"Triggering model retraining for ({latitude}, {longitude})")
        
        # Start retraining in background
        if background_tasks:
            background_tasks.add_task(
                _retrain_models_background,
                manager,
                latitude,
                longitude,
                model_type
            )
            
            return {
                "message": "Model retraining started in background",
                "location": {"latitude": latitude, "longitude": longitude},
                "model_type": model_type or "all"
            }
        else:
            # Synchronous retraining (not recommended for production)
            success = await manager.retrain_models(latitude, longitude, model_type)
            
            return {
                "message": "Model retraining completed" if success else "Model retraining failed",
                "success": success,
                "location": {"latitude": latitude, "longitude": longitude},
                "model_type": model_type or "all"
            }
        
    except Exception as e:
        logger.error(f"Error triggering model retraining: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger model retraining: {str(e)}"
        )


@router.get("/models/performance/{model_type}")
async def get_model_performance(
    model_type: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Get detailed performance metrics for a specific model type.
    
    Args:
        model_type: Type of model (lstm, arima, random_forest, fuzzy, lube)
        latitude: Optional location latitude for location-specific metrics
        longitude: Optional location longitude for location-specific metrics
        manager: Model manager instance
        
    Returns:
        Detailed model performance metrics
    """
    try:
        if model_type not in ['lstm', 'arima', 'random_forest', 'fuzzy', 'lube', 'ensemble']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type: {model_type}"
            )
        
        # Get ensemble metrics if available
        if hasattr(manager.ensemble, 'get_ensemble_metrics'):
            ensemble_metrics = manager.ensemble.get_ensemble_metrics()
            
            if model_type == 'ensemble':
                return {
                    "model_type": model_type,
                    "metrics": ensemble_metrics.get('ensemble_metrics', {}),
                    "model_weights": ensemble_metrics.get('model_weights', {}),
                    "prediction_count": ensemble_metrics.get('prediction_count', 0)
                }
            elif model_type in ensemble_metrics.get('model_metrics', {}):
                return {
                    "model_type": model_type,
                    "metrics": ensemble_metrics['model_metrics'][model_type],
                    "weight": ensemble_metrics.get('model_weights', {}).get(model_type, 0.0),
                    "prediction_count": ensemble_metrics.get('prediction_count', 0)
                }
        
        # Fallback to individual model metrics
        if model_type in manager.ensemble.models:
            model = manager.ensemble.models[model_type]
            if model.is_trained and model.metrics:
                return {
                    "model_type": model_type,
                    "metrics": {
                        "mae": model.metrics.mae,
                        "rmse": model.metrics.rmse,
                        "mape": model.metrics.mape,
                        "last_updated": model.metrics.last_updated.isoformat()
                    },
                    "is_trained": model.is_trained,
                    "model_info": model.model_info.dict() if model.model_info else None
                }
        
        return {
            "model_type": model_type,
            "message": "Model not available or not trained",
            "metrics": {}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model performance: {str(e)}"
        )


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get prediction cache statistics.
    
    Returns:
        Cache usage statistics
    """
    try:
        from app.db.database import SessionLocal
        from app.db.models import PredictionCache
        from sqlalchemy import func
        
        db = SessionLocal()
        try:
            # Get cache statistics
            total_entries = db.query(func.count(PredictionCache.id)).scalar()
            
            # Get cache hit statistics (entries accessed recently)
            recent_hits = db.query(func.count(PredictionCache.id)).filter(
                PredictionCache.last_accessed > (func.now() - func.interval('1 hour'))
            ).scalar()
            
            # Get expired entries
            expired_entries = db.query(func.count(PredictionCache.id)).filter(
                PredictionCache.expires_at < func.now()
            ).scalar()
            
            return {
                "total_entries": total_entries or 0,
                "recent_hits": recent_hits or 0,
                "expired_entries": expired_entries or 0,
                "cache_hit_rate": (recent_hits / total_entries * 100) if total_entries > 0 else 0
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get cache statistics"
        )


@router.delete("/cache/cleanup")
async def cleanup_cache(
    background_tasks: BackgroundTasks,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Clean up expired cache entries and old models.
    
    Returns:
        Cleanup status
    """
    try:
        # Schedule cleanup in background
        background_tasks.add_task(_cleanup_cache_background, manager)
        
        return {
            "message": "Cache cleanup started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting cache cleanup: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start cache cleanup"
        )


# Background task functions

async def _update_model_performance(
    manager: ModelManager, 
    latitude: float, 
    longitude: float
) -> None:
    """Background task to update model performance metrics."""
    try:
        # This could include validation against actual weather data
        # For now, just log the update
        logger.info(f"Updating model performance for location ({latitude}, {longitude})")
        
        # In a production system, you would:
        # 1. Fetch actual weather data for recent predictions
        # 2. Compare with model predictions
        # 3. Update model performance metrics
        # 4. Trigger retraining if performance degrades
        
    except Exception as e:
        logger.error(f"Error updating model performance: {e}")


async def _retrain_models_background(
    manager: ModelManager,
    latitude: float,
    longitude: float,
    model_type: Optional[str] = None
) -> None:
    """Background task for model retraining."""
    try:
        success = await manager.retrain_models(latitude, longitude, model_type)
        if success:
            logger.info(f"Background retraining completed for ({latitude}, {longitude})")
        else:
            logger.error(f"Background retraining failed for ({latitude}, {longitude})")
            
    except Exception as e:
        logger.error(f"Error in background retraining: {e}")


async def _cleanup_cache_background(manager: ModelManager) -> None:
    """Background task for cache cleanup."""
    try:
        from app.db.database import SessionLocal
        from app.db.models import PredictionCache
        from sqlalchemy import func
        
        # Clean up expired cache entries
        db = SessionLocal()
        try:
            deleted_count = db.query(PredictionCache).filter(
                PredictionCache.expires_at < func.now()
            ).delete()
            
            db.commit()
            logger.info(f"Cleaned up {deleted_count} expired cache entries")
            
        finally:
            db.close()
        
        # Clean up old models
        await manager.cleanup_old_models(days_old=30)
        
    except Exception as e:
        logger.error(f"Error in background cache cleanup: {e}")