"""
Model management and performance tracking service.
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
from loguru import logger

from app.db.database import SessionLocal
from app.db.models import ModelMetadata, PredictionCache
from app.schemas.weather import PredictionRequest, PredictionResponse, ModelStatus, ModelStatusResponse
from app.services.data_collector import DataCollector
from app.ml.ensemble import EnsembleWeatherPredictor
from app.ml.preprocessing import WeatherDataPreprocessor
from app.models.base import ModelMetrics
from app.core.config import settings


class ModelManager:
    """Service for managing ML models and tracking their performance."""
    
    def __init__(self):
        self.ensemble = EnsembleWeatherPredictor()
        self.data_collector = DataCollector()
        self.preprocessor = WeatherDataPreprocessor()
        self.models_loaded = False
        self.model_cache: Dict[str, Any] = {}
        
    async def initialize_models(self, latitude: float, longitude: float) -> bool:
        """
        Initialize and load models for a specific location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            
        Returns:
            True if models loaded successfully
        """
        try:
            logger.info(f"Initializing models for location ({latitude}, {longitude})")
            
            # Check if models exist for this location
            existing_models = self._get_existing_models(latitude, longitude)
            
            if existing_models:
                # Load existing models
                success = await self._load_existing_models(existing_models, latitude, longitude)
                if success:
                    self.models_loaded = True
                    return True
            
            # Train new models if none exist or loading failed
            logger.info("Training new models for location")
            success = await self._train_new_models(latitude, longitude)
            
            if success:
                self.models_loaded = True
                logger.info("Models initialized successfully")
                return True
            else:
                logger.error("Failed to initialize models")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
    
    def _get_existing_models(self, latitude: float, longitude: float, tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """Get existing model metadata for a location."""
        db = SessionLocal()
        try:
            models = db.query(ModelMetadata).filter(
                and_(
                    ModelMetadata.latitude.between(latitude - tolerance, latitude + tolerance),
                    ModelMetadata.longitude.between(longitude - tolerance, longitude + tolerance),
                    ModelMetadata.is_active == True,
                    ModelMetadata.is_trained == True
                )
            ).order_by(desc(ModelMetadata.created_at)).all()
            
            model_data = []
            for model in models:
                model_data.append({
                    'id': model.id,
                    'model_type': model.model_type,
                    'model_name': model.model_name,
                    'model_file_path': model.model_file_path,
                    'mae': model.mae,
                    'rmse': model.rmse,
                    'mape': model.mape,
                    'last_trained': model.updated_at
                })
            
            return model_data
            
        finally:
            db.close()
    
    async def _load_existing_models(self, model_metadata: List[Dict[str, Any]], latitude: float, longitude: float) -> bool:
        """Load existing trained models."""
        try:
            # Initialize ensemble models
            model_instances = self.ensemble.initialize_models()
            
            loaded_count = 0
            
            # Group models by type
            models_by_type = {}
            for model_data in model_metadata:
                model_type = model_data['model_type'].lower()
                if model_type not in models_by_type:
                    models_by_type[model_type] = []
                models_by_type[model_type].append(model_data)
            
            # Load the best model for each type
            for model_type, models in models_by_type.items():
                if model_type in model_instances:
                    # Sort by performance (lowest MAE first)
                    best_model = min(models, key=lambda x: x.get('mae', float('inf')))
                    
                    if best_model['model_file_path'] and os.path.exists(best_model['model_file_path']):
                        try:
                            model_instances[model_type].load_model(best_model['model_file_path'])
                            loaded_count += 1
                            logger.info(f"Loaded {model_type} model with MAE: {best_model['mae']:.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to load {model_type} model: {e}")
            
            if loaded_count > 0:
                self.ensemble.is_trained = True
                logger.info(f"Successfully loaded {loaded_count} models")
                return True
            else:
                logger.warning("No models could be loaded")
                return False
                
        except Exception as e:
            logger.error(f"Error loading existing models: {e}")
            return False
    
    async def _train_new_models(self, latitude: float, longitude: float) -> bool:
        """Train new models for the location."""
        try:
            # Collect historical data
            historical_data = await self.data_collector.collect_historical_data(
                latitude, longitude, settings.training_data_days
            )
            
            if len(historical_data) < 100:
                logger.error(f"Insufficient training data: {len(historical_data)} records")
                return False
            
            # Validate data quality
            quality_metrics = self.data_collector.validate_data_quality(historical_data)
            if not quality_metrics['is_valid']:
                logger.error(f"Poor data quality: {quality_metrics['issues']}")
                return False
            
            # Preprocess data
            df = self.preprocessor.preprocess_for_training(historical_data)
            
            # Initialize and train ensemble
            self.ensemble.initialize_models()
            self.ensemble.train_ensemble(df)
            
            # Save trained models
            await self._save_trained_models(latitude, longitude)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training new models: {e}")
            return False
    
    async def _save_trained_models(self, latitude: float, longitude: float) -> None:
        """Save trained models to disk and database."""
        try:
            model_dir = os.path.join(settings.model_storage_path, f"location_{latitude:.2f}_{longitude:.2f}")
            os.makedirs(model_dir, exist_ok=True)
            
            db = SessionLocal()
            
            try:
                for model_name, model in self.ensemble.models.items():
                    if model.is_trained:
                        # Save model file
                        model_file = os.path.join(model_dir, f"{model_name}_model.pkl")
                        model.save_model(model_file)
                        
                        # Save metadata to database
                        model_metadata = ModelMetadata(
                            model_name=f"{model_name}_{latitude:.2f}_{longitude:.2f}",
                            model_type=model_name,
                            model_version="1.0",
                            latitude=latitude,
                            longitude=longitude,
                            training_start_date=datetime.utcnow() - timedelta(days=settings.training_data_days),
                            training_end_date=datetime.utcnow(),
                            training_data_points=1000,  # Approximate
                            hyperparameters=json.dumps(model.config),
                            feature_columns=json.dumps(getattr(model, 'feature_names', [])),
                            target_columns=json.dumps(getattr(model, 'target_names', [])),
                            mae=model.metrics.mae if model.metrics else None,
                            rmse=model.metrics.rmse if model.metrics else None,
                            mape=model.metrics.mape if model.metrics else None,
                            model_file_path=model_file,
                            model_file_size=os.path.getsize(model_file) if os.path.exists(model_file) else 0,
                            is_active=True,
                            is_trained=True
                        )
                        
                        db.add(model_metadata)
                
                db.commit()
                logger.info("Model metadata saved to database")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error saving trained models: {e}")
            raise
    
    async def predict_weather(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate weather predictions using the ensemble.
        
        Args:
            request: Prediction request with location and parameters
            
        Returns:
            Prediction response with forecasts and uncertainty
        """
        if not self.models_loaded:
            # Try to initialize models for this location
            success = await self.initialize_models(request.latitude, request.longitude)
            if not success:
                raise Exception("Models not available for this location")
        
        try:
            # Get current weather data
            current_weather = await self.data_collector.weather_manager.get_current_weather(
                request.latitude, request.longitude
            )
            
            # Get recent historical data for context
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)  # Last 7 days for context
            
            recent_data = self.data_collector._get_existing_data(
                request.latitude, request.longitude, start_date, end_date
            )
            
            if len(recent_data) < 24:  # Need at least 24 hours of recent data
                logger.warning("Limited recent data available, using current weather only")
                recent_data = [current_weather] * 24  # Repeat current weather
            
            # Preprocess recent data
            df = self.preprocessor.preprocess_for_prediction(recent_data)
            
            if df.empty:
                raise Exception("Failed to preprocess recent weather data")
            
            # Generate predictions
            features = df.iloc[-1:].values  # Use most recent features
            
            # Get ensemble predictions with uncertainty
            predictions, lower_bounds, upper_bounds = self.ensemble.predict_with_uncertainty(features)
            
            # Generate hourly predictions for next 24 hours
            prediction_results = []
            base_time = datetime.utcnow()
            
            for hour in range(request.hours):
                timestamp = base_time + timedelta(hours=hour + 1)
                
                # For simplicity, use the same prediction for all hours
                # In a more sophisticated system, you'd generate sequence predictions
                pred_result = {
                    'timestamp': timestamp,
                    'temperature': float(predictions[0]) if len(predictions) > 0 else 20.0,
                    'humidity': float(predictions[1]) if len(predictions) > 1 else 50.0,
                    'pressure': float(predictions[2]) if len(predictions) > 2 else 1013.25,
                    'wind_speed': float(predictions[3]) if len(predictions) > 3 else 5.0,
                    'uncertainty': {
                        'lower_bound': float(lower_bounds[0]) if len(lower_bounds) > 0 else float(predictions[0]) * 0.95,
                        'upper_bound': float(upper_bounds[0]) if len(upper_bounds) > 0 else float(predictions[0]) * 1.05,
                        'confidence': 0.95
                    },
                    'model_contributions': self.ensemble.get_model_contributions()
                }
                
                prediction_results.append(pred_result)
            
            # Get model performance metrics
            model_performance = {}
            for model_name, model in self.ensemble.models.items():
                if model.is_trained and model.metrics:
                    model_performance[model_name] = {
                        'mae': model.metrics.mae,
                        'rmse': model.metrics.rmse,
                        'mape': model.metrics.mape,
                        'last_updated': model.metrics.last_updated.isoformat()
                    }
            
            # Cache the prediction
            await self._cache_prediction(request, prediction_results, model_performance)
            
            return PredictionResponse(
                location=current_weather.location,
                prediction_time=datetime.utcnow(),
                forecast_horizon=request.hours,
                predictions=prediction_results,
                model_performance=model_performance,
                current_weather=current_weather
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def _cache_prediction(self, request: PredictionRequest, predictions: List[Dict], model_performance: Dict) -> None:
        """Cache prediction results."""
        try:
            cache_key = f"pred_{request.latitude:.2f}_{request.longitude:.2f}_{request.hours}"
            expires_at = datetime.utcnow() + timedelta(seconds=settings.prediction_cache_ttl)
            
            db = SessionLocal()
            try:
                # Remove old cache entries for this location
                db.query(PredictionCache).filter(
                    and_(
                        PredictionCache.latitude == request.latitude,
                        PredictionCache.longitude == request.longitude,
                        PredictionCache.forecast_horizon_hours == request.hours
                    )
                ).delete()
                
                # Add new cache entry
                cache_entry = PredictionCache(
                    latitude=request.latitude,
                    longitude=request.longitude,
                    prediction_timestamp=datetime.utcnow(),
                    forecast_horizon_hours=request.hours,
                    predictions_data=json.dumps(predictions, default=str),
                    model_performance=json.dumps(model_performance, default=str),
                    ensemble_weights=json.dumps(self.ensemble.get_model_contributions()),
                    cache_key=cache_key,
                    expires_at=expires_at
                )
                
                db.add(cache_entry)
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.warning(f"Failed to cache prediction: {e}")
    
    def get_model_status(self) -> ModelStatusResponse:
        """Get current status of all models."""
        try:
            model_statuses = []
            
            for model_name, model in self.ensemble.models.items():
                status = ModelStatus(
                    name=model_name,
                    type=model_name,
                    is_loaded=model.is_trained,
                    last_trained=model.model_info.last_trained.isoformat() if model.model_info and model.model_info.last_trained else "",
                    performance={
                        'mae': model.metrics.mae if model.metrics else 0.0,
                        'rmse': model.metrics.rmse if model.metrics else 0.0,
                        'mape': model.metrics.mape if model.metrics else 0.0,
                        'last_updated': model.metrics.last_updated.isoformat() if model.metrics else ""
                    },
                    status='healthy' if model.is_trained else 'failed'
                )
                model_statuses.append(status)
            
            # Determine overall system health
            healthy_models = sum(1 for status in model_statuses if status.status == 'healthy')
            total_models = len(model_statuses)
            
            if healthy_models == 0:
                system_health = 'failed'
            elif healthy_models < total_models * 0.5:
                system_health = 'degraded'
            else:
                system_health = 'healthy'
            
            return ModelStatusResponse(
                models=model_statuses,
                system_health=system_health
            )
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return ModelStatusResponse(
                models=[],
                system_health='failed'
            )
    
    async def retrain_models(self, latitude: float, longitude: float, model_type: Optional[str] = None) -> bool:
        """
        Retrain models for a location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            model_type: Specific model type to retrain (optional)
            
        Returns:
            True if retraining successful
        """
        try:
            logger.info(f"Retraining models for location ({latitude}, {longitude})")
            
            if model_type:
                logger.info(f"Retraining specific model: {model_type}")
            
            # Mark old models as inactive
            db = SessionLocal()
            try:
                db.query(ModelMetadata).filter(
                    and_(
                        ModelMetadata.latitude.between(latitude - 0.1, latitude + 0.1),
                        ModelMetadata.longitude.between(longitude - 0.1, longitude + 0.1),
                        ModelMetadata.model_type == model_type if model_type else True
                    )
                ).update({'is_active': False})
                db.commit()
            finally:
                db.close()
            
            # Retrain models
            success = await self._train_new_models(latitude, longitude)
            
            if success:
                logger.info("Model retraining completed successfully")
                return True
            else:
                logger.error("Model retraining failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            return False
    
    async def cleanup_old_models(self, days_old: int = 30) -> None:
        """Clean up old model files and database entries."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            db = SessionLocal()
            try:
                # Get old model entries
                old_models = db.query(ModelMetadata).filter(
                    and_(
                        ModelMetadata.created_at < cutoff_date,
                        ModelMetadata.is_active == False
                    )
                ).all()
                
                # Delete model files and database entries
                for model in old_models:
                    if model.model_file_path and os.path.exists(model.model_file_path):
                        try:
                            os.remove(model.model_file_path)
                            logger.info(f"Deleted old model file: {model.model_file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete model file {model.model_file_path}: {e}")
                    
                    db.delete(model)
                
                db.commit()
                logger.info(f"Cleaned up {len(old_models)} old model entries")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
    
    async def close(self):
        """Close connections and cleanup resources."""
        await self.data_collector.close()