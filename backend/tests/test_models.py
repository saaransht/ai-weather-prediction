"""
Unit tests for ML models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.models.lstm_model import LSTMWeatherModel
from app.models.arima_model import ARIMAWeatherModel
from app.models.random_forest_model import RandomForestWeatherModel
from app.models.fuzzy_model import FuzzyTimeSeriesModel
from app.models.lube_model import LUBEWeatherModel
from app.ml.ensemble import EnsembleWeatherPredictor


class TestDataGenerator:
    """Generate synthetic weather data for testing."""
    
    @staticmethod
    def generate_weather_data(n_samples: int = 100, n_features: int = 10) -> pd.DataFrame:
        """Generate synthetic weather data."""
        np.random.seed(42)
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=n_samples//24)
        timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
        
        # Generate weather features with realistic patterns
        data = {
            'timestamp': timestamps,
            'temperature': 20 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 2, n_samples),
            'humidity': 50 + 20 * np.sin(np.arange(n_samples) * 2 * np.pi / 24 + np.pi/4) + np.random.normal(0, 5, n_samples),
            'pressure': 1013 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 48) + np.random.normal(0, 3, n_samples),
            'wind_speed': 5 + 3 * np.abs(np.sin(np.arange(n_samples) * 2 * np.pi / 12)) + np.random.normal(0, 1, n_samples),
            'wind_direction': (np.arange(n_samples) * 15) % 360 + np.random.normal(0, 20, n_samples),
            'cloud_cover': 30 + 40 * np.random.random(n_samples),
            'precipitation': np.random.exponential(0.5, n_samples)
        }
        
        # Add engineered features
        for i in range(n_features - 7):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df


class TestLSTMModel:
    """Test cases for LSTM model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = LSTMWeatherModel(config={
            'sequence_length': 24,
            'hidden_size': 16,
            'num_layers': 1,
            'epochs': 5,
            'batch_size': 8
        })
        self.test_data = TestDataGenerator.generate_weather_data(200)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.name == "LSTM"
        assert not self.model.is_trained
        assert self.model.network is None
    
    def test_model_training(self):
        """Test model training."""
        self.model.train(self.test_data)
        
        assert self.model.is_trained
        assert self.model.network is not None
        assert self.model.metrics is not None
        assert len(self.model.feature_names) > 0
        assert len(self.model.target_names) > 0
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.train(self.test_data)
        
        # Test single prediction
        features = self.test_data[self.model.feature_names].values[-50:]
        prediction = self.model.predict(features)
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == len(self.model.target_names)
    
    def test_uncertainty_prediction(self):
        """Test uncertainty prediction."""
        self.model.train(self.test_data)
        
        features = self.test_data[self.model.feature_names].values[-50:]
        pred, lower, upper = self.model.predict_with_uncertainty(features)
        
        assert isinstance(pred, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)
    
    def test_sequence_prediction(self):
        """Test sequence prediction."""
        self.model.train(self.test_data)
        
        features = self.test_data[self.model.feature_names].values[-50:]
        sequence_pred = self.model.predict_sequence(features, steps=12)
        
        assert isinstance(sequence_pred, np.ndarray)
        assert sequence_pred.shape[0] == 12
        assert sequence_pred.shape[1] == len(self.model.target_names)
    
    def test_model_serialization(self):
        """Test model save/load."""
        self.model.train(self.test_data)
        
        # Save model
        filepath = "/tmp/test_lstm_model.pth"
        self.model.save_model(filepath)
        
        # Create new model and load
        new_model = LSTMWeatherModel()
        new_model.load_model(filepath)
        
        assert new_model.is_trained
        assert new_model.feature_names == self.model.feature_names
        assert new_model.target_names == self.model.target_names


class TestARIMAModel:
    """Test cases for ARIMA model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = ARIMAWeatherModel(config={
            'target_column': 'temperature',
            'auto_arima': True,
            'max_p': 2,
            'max_d': 1,
            'max_q': 2
        })
        self.test_data = TestDataGenerator.generate_weather_data(100)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.name == "ARIMA"
        assert not self.model.is_trained
        assert len(self.model.models) == 0
    
    def test_model_training(self):
        """Test model training."""
        self.model.train(self.test_data)
        
        assert self.model.is_trained
        assert len(self.model.models) > 0
        assert self.model.metrics is not None
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.train(self.test_data)
        
        # ARIMA doesn't use features directly
        prediction = self.model.predict(np.array([]))
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == len(self.model.target_columns)
    
    def test_uncertainty_prediction(self):
        """Test uncertainty prediction."""
        self.model.train(self.test_data)
        
        pred, lower, upper = self.model.predict_with_uncertainty(np.array([]))
        
        assert isinstance(pred, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)


class TestRandomForestModel:
    """Test cases for Random Forest model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = RandomForestWeatherModel(config={
            'n_estimators': 10,
            'max_depth': 5,
            'hyperparameter_tuning': False,
            'feature_selection': True
        })
        self.test_data = TestDataGenerator.generate_weather_data(150)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.name == "RandomForest"
        assert not self.model.is_trained
        assert self.model.model is None
    
    def test_model_training(self):
        """Test model training."""
        self.model.train(self.test_data)
        
        assert self.model.is_trained
        assert self.model.model is not None
        assert len(self.model.selected_features) > 0
        assert len(self.model.feature_importance) > 0
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        engineered_data = self.model._create_advanced_features(self.test_data)
        
        # Should have more features than original
        assert engineered_data.shape[1] > self.test_data.shape[1]
        
        # Check for specific engineered features
        assert 'temp_humidity_interaction' in engineered_data.columns
        assert 'wind_u' in engineered_data.columns
        assert 'wind_v' in engineered_data.columns
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.train(self.test_data)
        
        # Create test features
        test_features = np.random.random((5, len(self.model.selected_features)))
        prediction = self.model.predict(test_features)
        
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape[0] == 5
        assert prediction.shape[1] == len(self.model.target_names)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        self.model.train(self.test_data)
        
        importance = self.model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(isinstance(v, float) for v in importance.values())


class TestFuzzyModel:
    """Test cases for Fuzzy Time Series model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = FuzzyTimeSeriesModel(config={
            'target_column': 'temperature',
            'num_fuzzy_sets': 5,
            'order': 1
        })
        self.test_data = TestDataGenerator.generate_weather_data(100)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.name == "FuzzyTimeSeries"
        assert not self.model.is_trained
        assert len(self.model.fuzzy_sets) == 0
    
    def test_fuzzy_set_creation(self):
        """Test fuzzy set creation."""
        series = self.test_data['temperature']
        universe_min, universe_max = self.model._determine_universe(series)
        fuzzy_sets = self.model._create_fuzzy_sets(universe_min, universe_max)
        
        assert len(fuzzy_sets) == 5
        assert all(isinstance(fs.center, float) for fs in fuzzy_sets.values())
    
    def test_fuzzification(self):
        """Test fuzzification process."""
        self.model.train(self.test_data)
        
        # Test single value fuzzification
        test_value = 25.0
        fuzzy_label = self.model._fuzzify_value(test_value)
        
        assert isinstance(fuzzy_label, str)
        assert fuzzy_label in self.model.fuzzy_sets
    
    def test_model_training(self):
        """Test model training."""
        self.model.train(self.test_data)
        
        assert self.model.is_trained
        assert len(self.model.fuzzy_sets) > 0
        assert len(self.model.flrgs) > 0
        assert len(self.model.fuzzified_series) > 0
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.train(self.test_data)
        
        prediction = self.model.predict(np.array([]))
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == 1


class TestLUBEModel:
    """Test cases for LUBE model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = LUBEWeatherModel(config={
            'hidden_sizes': [16, 8],
            'epochs': 5,
            'batch_size': 8,
            'confidence_level': 0.95
        })
        self.test_data = TestDataGenerator.generate_weather_data(150)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.name == "LUBE"
        assert not self.model.is_trained
        assert len(self.model.networks) == 0
    
    def test_model_training(self):
        """Test model training."""
        self.model.train(self.test_data)
        
        assert self.model.is_trained
        assert len(self.model.networks) > 0
        assert self.model.metrics is not None
    
    def test_uncertainty_prediction(self):
        """Test uncertainty prediction."""
        self.model.train(self.test_data)
        
        # Create test features
        test_features = np.random.random((3, len(self.model.feature_names)))
        pred, lower, upper = self.model.predict_with_uncertainty(test_features)
        
        assert isinstance(pred, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert pred.shape == (3, len(self.model.target_names))
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)
    
    def test_coverage_metrics(self):
        """Test coverage metrics calculation."""
        self.model.train(self.test_data)
        
        # Create test data
        X_test = np.random.random((20, len(self.model.feature_names)))
        y_test = np.random.random((20, len(self.model.target_names)))
        
        metrics = self.model.calculate_coverage_metrics(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0


class TestEnsemblePredictor:
    """Test cases for Ensemble predictor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ensemble = EnsembleWeatherPredictor(config={
            'models': ['lstm', 'random_forest'],
            'weighting_method': 'equal'
        })
        self.test_data = TestDataGenerator.generate_weather_data(100)
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        assert not self.ensemble.is_trained
        assert len(self.ensemble.models) == 0
    
    def test_model_addition(self):
        """Test adding models to ensemble."""
        # Create and train a simple model
        rf_model = RandomForestWeatherModel(config={'n_estimators': 5})
        rf_model.train(self.test_data)
        
        self.ensemble.add_model('test_rf', rf_model)
        
        assert 'test_rf' in self.ensemble.models
        assert 'test_rf' in self.ensemble.model_weights
    
    def test_weight_calculation(self):
        """Test weight calculation methods."""
        # Add trained models
        rf_model = RandomForestWeatherModel(config={'n_estimators': 5})
        rf_model.train(self.test_data)
        self.ensemble.add_model('rf', rf_model)
        
        # Test equal weights
        self.ensemble.config['weighting_method'] = 'equal'
        self.ensemble.update_weights()
        
        assert abs(self.ensemble.model_weights['rf'] - 1.0) < 1e-6
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        # Add and train models
        rf_model = RandomForestWeatherModel(config={'n_estimators': 5})
        rf_model.train(self.test_data)
        self.ensemble.add_model('rf', rf_model)
        self.ensemble.is_trained = True
        
        # Test prediction
        test_features = np.random.random((1, 10))
        prediction = self.ensemble.predict(test_features)
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) > 0


# Pytest fixtures
@pytest.fixture
def sample_weather_data():
    """Fixture for sample weather data."""
    return TestDataGenerator.generate_weather_data(100)


@pytest.fixture
def trained_lstm_model(sample_weather_data):
    """Fixture for trained LSTM model."""
    model = LSTMWeatherModel(config={'epochs': 2, 'batch_size': 8})
    model.train(sample_weather_data)
    return model


@pytest.fixture
def trained_rf_model(sample_weather_data):
    """Fixture for trained Random Forest model."""
    model = RandomForestWeatherModel(config={'n_estimators': 5})
    model.train(sample_weather_data)
    return model


# Integration tests
class TestModelIntegration:
    """Integration tests for model interactions."""
    
    def test_model_compatibility(self, sample_weather_data):
        """Test that all models can work with the same data format."""
        models = [
            LSTMWeatherModel(config={'epochs': 2}),
            RandomForestWeatherModel(config={'n_estimators': 5}),
            ARIMAWeatherModel(),
            FuzzyTimeSeriesModel(),
            LUBEWeatherModel(config={'epochs': 2})
        ]
        
        for model in models:
            try:
                model.train(sample_weather_data)
                assert model.is_trained
            except Exception as e:
                pytest.fail(f"Model {model.name} failed to train: {e}")
    
    def test_ensemble_with_multiple_models(self, sample_weather_data):
        """Test ensemble with multiple trained models."""
        ensemble = EnsembleWeatherPredictor()
        
        # Train and add models
        rf_model = RandomForestWeatherModel(config={'n_estimators': 5})
        rf_model.train(sample_weather_data)
        ensemble.add_model('rf', rf_model)
        
        lstm_model = LSTMWeatherModel(config={'epochs': 2})
        lstm_model.train(sample_weather_data)
        ensemble.add_model('lstm', lstm_model)
        
        ensemble.is_trained = True
        
        # Test ensemble prediction
        test_features = np.random.random((1, 10))
        prediction = ensemble.predict(test_features)
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) > 0


if __name__ == "__main__":
    pytest.main([__file__])