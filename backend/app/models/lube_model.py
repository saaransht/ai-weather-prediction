"""
LUBE (Lower-Upper Bound Estimation) model for uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from loguru import logger

from app.models.base import WeatherModel, ModelInfo, ModelException, ModelTrainingException, ModelPredictionException
from app.ml.model_utils import ModelEvaluator
from app.core.config import settings


class LUBENetwork(nn.Module):
    """Neural network for LUBE uncertainty estimation."""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: List[int] = [64, 32], 
        dropout: float = 0.2,
        confidence_level: float = 0.95
    ):
        super(LUBENetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level  # Significance level
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer: 3 outputs (lower_bound, prediction, upper_bound)
        layers.append(nn.Linear(prev_size, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        outputs = self.network(x)
        
        # Split outputs: [lower_bound, prediction, upper_bound]
        lower_bound = outputs[:, 0:1]
        prediction = outputs[:, 1:2]
        upper_bound = outputs[:, 2:3]
        
        # Ensure lower_bound <= prediction <= upper_bound
        lower_bound = torch.min(lower_bound, prediction)
        upper_bound = torch.max(upper_bound, prediction)
        
        return torch.cat([lower_bound, prediction, upper_bound], dim=1)


class LUBELoss(nn.Module):
    """Custom loss function for LUBE training."""
    
    def __init__(self, confidence_level: float = 0.95, lambda_coverage: float = 50.0, lambda_width: float = 1.0):
        super(LUBELoss, self).__init__()
        self.confidence_level = confidence_level
        self.lambda_coverage = lambda_coverage  # Coverage penalty weight
        self.lambda_width = lambda_width  # Width penalty weight
        self.alpha = 1.0 - confidence_level
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate LUBE loss.
        
        Args:
            predictions: Tensor of shape (batch_size, 3) [lower, pred, upper]
            targets: Tensor of shape (batch_size, 1) [true_values]
            
        Returns:
            Combined loss value
        """
        lower_bounds = predictions[:, 0]
        point_predictions = predictions[:, 1]
        upper_bounds = predictions[:, 2]
        true_values = targets.squeeze()
        
        # 1. Point prediction loss (MSE)
        mse_loss = torch.mean((point_predictions - true_values) ** 2)
        
        # 2. Coverage loss (PICP - Prediction Interval Coverage Probability)
        coverage_indicators = ((true_values >= lower_bounds) & (true_values <= upper_bounds)).float()
        picp = torch.mean(coverage_indicators)
        coverage_loss = torch.max(torch.tensor(0.0, device=predictions.device), 
                                 torch.tensor(self.alpha, device=predictions.device) - picp) ** 2
        
        # 3. Width loss (PINAW - Prediction Interval Normalized Average Width)
        interval_widths = upper_bounds - lower_bounds
        # Normalize by target range to make it scale-invariant
        target_range = torch.max(true_values) - torch.min(true_values)
        if target_range > 0:
            normalized_width = torch.mean(interval_widths) / target_range
        else:
            normalized_width = torch.mean(interval_widths)
        
        # 4. Combine losses
        total_loss = (mse_loss + 
                     self.lambda_coverage * coverage_loss + 
                     self.lambda_width * normalized_width)
        
        return total_loss


class LUBEWeatherModel(WeatherModel):
    """LUBE-based uncertainty estimation model for weather prediction."""
    
    def __init__(self, name: str = "LUBE", config: Dict[str, Any] = None):
        default_config = {
            'hidden_sizes': [64, 32],
            'dropout': 0.2,
            'confidence_level': 0.95,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 200,
            'patience': 15,
            'lambda_coverage': 50.0,
            'lambda_width': 1.0,
            'target_columns': ['temperature', 'humidity', 'pressure', 'wind_speed'],
            'validation_split': 0.2
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.networks: Dict[str, LUBENetwork] = {}  # One network per target
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        self.scalers: Dict[str, Any] = {}
        
        logger.info(f"LUBE model initialized with device: {self.device}")
    
    def _prepare_data_for_target(self, X: np.ndarray, y: np.ndarray, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training a specific target.
        
        Args:
            X: Feature matrix
            y: Target matrix
            target_idx: Index of target column
            
        Returns:
            Tuple of (features, single_target)
        """
        # Extract single target
        single_target = y[:, target_idx] if y.ndim > 1 else y
        
        # Remove any rows with NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(single_target))
        X_clean = X[valid_mask]
        y_clean = single_target[valid_mask]
        
        return X_clean, y_clean
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train LUBE models for uncertainty estimation.
        
        Args:
            data: DataFrame with preprocessed weather features and targets
        """
        try:
            logger.info("Starting LUBE model training")
            
            # Prepare target columns
            target_columns = self.config['target_columns']
            self.target_names = [col for col in target_columns if col in data.columns]
            
            if not self.target_names:
                raise ModelTrainingException("No valid target columns found in data")
            
            # Prepare feature columns
            exclude_cols = self.target_names + ['timestamp']
            feature_columns = [col for col in data.columns if col not in exclude_cols]
            self.feature_names = feature_columns
            
            # Extract features and targets
            X = data[feature_columns].values.astype(np.float32)
            y = data[self.target_names].values.astype(np.float32)
            
            logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
            
            # Train separate LUBE network for each target
            all_metrics = []
            
            for target_idx, target_name in enumerate(self.target_names):
                logger.info(f"Training LUBE network for {target_name}")
                
                # Prepare data for this target
                X_target, y_target = self._prepare_data_for_target(X, y, target_idx)
                
                if len(X_target) < 50:
                    logger.warning(f"Insufficient data for {target_name}: {len(X_target)} samples")
                    continue
                
                # Split data
                val_size = int(len(X_target) * self.config['validation_split'])
                train_size = len(X_target) - val_size
                
                X_train = X_target[:train_size]
                y_train = y_target[:train_size]
                X_val = X_target[train_size:]
                y_val = y_target[train_size:]
                
                # Convert to PyTorch tensors
                X_train_tensor = torch.FloatTensor(X_train).to(self.device)
                y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
                
                # Create data loader
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
                
                # Initialize network
                network = LUBENetwork(
                    input_size=X_train.shape[1],
                    hidden_sizes=self.config['hidden_sizes'],
                    dropout=self.config['dropout'],
                    confidence_level=self.config['confidence_level']
                ).to(self.device)
                
                # Loss function and optimizer
                criterion = LUBELoss(
                    confidence_level=self.config['confidence_level'],
                    lambda_coverage=self.config['lambda_coverage'],
                    lambda_width=self.config['lambda_width']
                )
                optimizer = optim.Adam(network.parameters(), lr=self.config['learning_rate'])
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
                
                # Training loop
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(self.config['epochs']):
                    # Training phase
                    network.train()
                    train_loss = 0.0
                    
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = network(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        train_loss += loss.item()
                    
                    train_loss /= len(train_loader)
                    
                    # Validation phase
                    network.eval()
                    with torch.no_grad():
                        val_outputs = network(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    # Learning rate scheduling
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = network.state_dict().copy()
                    else:
                        patience_counter += 1
                    
                    if epoch % 20 == 0:
                        logger.info(f"Epoch {epoch} ({target_name}): Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                    
                    if patience_counter >= self.config['patience']:
                        logger.info(f"Early stopping at epoch {epoch} for {target_name}")
                        break
                
                # Load best model state
                network.load_state_dict(best_model_state)
                self.networks[target_name] = network
                
                # Calculate metrics
                network.eval()
                with torch.no_grad():
                    val_predictions = network(X_val_tensor)
                    point_predictions = val_predictions[:, 1].cpu().numpy()
                    
                target_metrics = ModelEvaluator.calculate_metrics(y_val, point_predictions)
                all_metrics.append([target_metrics.mae, target_metrics.rmse, target_metrics.mape])
                
                logger.info(f"LUBE training completed for {target_name}. Val MAE: {target_metrics.mae:.4f}")
            
            # Calculate overall metrics
            if all_metrics:
                avg_metrics = np.mean(all_metrics, axis=0)
                from app.models.base import ModelMetrics
                self.metrics = ModelMetrics(
                    mae=float(avg_metrics[0]),
                    rmse=float(avg_metrics[1]),
                    mape=float(avg_metrics[2]),
                    last_updated=datetime.utcnow()
                )
            
            self.is_trained = True
            
            # Create model info
            self.model_info = ModelInfo(
                name=self.name,
                model_type="LUBE",
                version="1.0",
                training_window_days=settings.training_data_days,
                feature_columns=self.feature_names,
                target_columns=self.target_names,
                hyperparameters=self.config,
                created_at=datetime.utcnow(),
                last_trained=datetime.utcnow()
            )
            
            logger.info(f"LUBE training completed for {len(self.networks)} targets")
            
        except Exception as e:
            logger.error(f"LUBE training failed: {e}")
            raise ModelTrainingException(f"LUBE training failed: {str(e)}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate point predictions using LUBE models.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Array of point predictions
        """
        if not self.is_trained or not self.networks:
            raise ModelPredictionException("Model not trained")
        
        if not self.validate_input(features):
            raise ModelPredictionException("Invalid input features")
        
        try:
            # Handle missing values
            features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(features_clean).to(self.device)
            
            predictions = []
            
            for target_name in self.target_names:
                if target_name in self.networks:
                    network = self.networks[target_name]
                    network.eval()
                    
                    with torch.no_grad():
                        outputs = network(X_tensor)
                        point_pred = outputs[:, 1].cpu().numpy()  # Middle output is point prediction
                        
                    predictions.append(point_pred)
                else:
                    predictions.append(np.zeros(len(features_clean)))
            
            # Transpose to get shape (n_samples, n_targets)
            predictions = np.array(predictions).T
            
            # Return single prediction if input is single sample
            if predictions.shape[0] == 1:
                return predictions[0]
            
            return predictions
            
        except Exception as e:
            logger.error(f"LUBE prediction failed: {e}")
            raise ModelPredictionException(f"LUBE prediction failed: {str(e)}")
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty bounds using LUBE.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained or not self.networks:
            raise ModelPredictionException("Model not trained")
        
        try:
            # Handle missing values
            features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(features_clean).to(self.device)
            
            predictions = []
            lower_bounds = []
            upper_bounds = []
            
            for target_name in self.target_names:
                if target_name in self.networks:
                    network = self.networks[target_name]
                    network.eval()
                    
                    with torch.no_grad():
                        outputs = network(X_tensor)
                        lower_pred = outputs[:, 0].cpu().numpy()
                        point_pred = outputs[:, 1].cpu().numpy()
                        upper_pred = outputs[:, 2].cpu().numpy()
                    
                    predictions.append(point_pred)
                    lower_bounds.append(lower_pred)
                    upper_bounds.append(upper_pred)
                else:
                    zero_pred = np.zeros(len(features_clean))
                    predictions.append(zero_pred)
                    lower_bounds.append(zero_pred)
                    upper_bounds.append(zero_pred)
            
            # Transpose to get shape (n_samples, n_targets)
            predictions = np.array(predictions).T
            lower_bounds = np.array(lower_bounds).T
            upper_bounds = np.array(upper_bounds).T
            
            # Return single prediction if input is single sample
            if predictions.shape[0] == 1:
                return predictions[0], lower_bounds[0], upper_bounds[0]
            
            return predictions, lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"LUBE uncertainty prediction failed: {e}")
            raise ModelPredictionException(f"LUBE uncertainty prediction failed: {str(e)}")
    
    def calculate_coverage_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculate coverage metrics for the prediction intervals.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with coverage metrics
        """
        try:
            predictions, lower_bounds, upper_bounds = self.predict_with_uncertainty(X_test)
            
            metrics = {}
            
            for i, target_name in enumerate(self.target_names):
                if i < y_test.shape[1]:
                    true_values = y_test[:, i]
                    pred_values = predictions[:, i] if predictions.ndim > 1 else predictions
                    lower_values = lower_bounds[:, i] if lower_bounds.ndim > 1 else lower_bounds
                    upper_values = upper_bounds[:, i] if upper_bounds.ndim > 1 else upper_bounds
                    
                    # Coverage probability
                    coverage = np.mean((true_values >= lower_values) & (true_values <= upper_values))
                    
                    # Average interval width
                    avg_width = np.mean(upper_values - lower_values)
                    
                    # Normalized average width
                    target_range = np.max(true_values) - np.min(true_values)
                    normalized_width = avg_width / target_range if target_range > 0 else avg_width
                    
                    metrics[f"{target_name}_coverage"] = float(coverage)
                    metrics[f"{target_name}_avg_width"] = float(avg_width)
                    metrics[f"{target_name}_normalized_width"] = float(normalized_width)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Coverage metrics calculation failed: {e}")
            return {}
    
    def save_model(self, filepath: str) -> None:
        """Save the LUBE models to disk."""
        if not self.networks:
            raise ModelException("No models to save")
        
        try:
            model_data = {
                'networks_state_dict': {name: net.state_dict() for name, net in self.networks.items()},
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'config': self.config,
                'model_info': self.model_info.dict() if self.model_info else None,
                'metrics': self.metrics.__dict__ if self.metrics else None,
                'is_trained': self.is_trained
            }
            
            torch.save(model_data, filepath)
            logger.info(f"LUBE models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save LUBE models: {e}")
            raise ModelException(f"Failed to save models: {str(e)}")
    
    def load_model(self, filepath: str) -> None:
        """Load the LUBE models from disk."""
        try:
            model_data = torch.load(filepath, map_location=self.device)
            
            self.feature_names = model_data['feature_names']
            self.target_names = model_data['target_names']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            
            # Recreate networks
            self.networks = {}
            for target_name in self.target_names:
                network = LUBENetwork(
                    input_size=len(self.feature_names),
                    hidden_sizes=self.config['hidden_sizes'],
                    dropout=self.config['dropout'],
                    confidence_level=self.config['confidence_level']
                ).to(self.device)
                
                if target_name in model_data['networks_state_dict']:
                    network.load_state_dict(model_data['networks_state_dict'][target_name])
                
                self.networks[target_name] = network
            
            # Load metadata
            if model_data.get('model_info'):
                self.model_info = ModelInfo(**model_data['model_info'])
            
            if model_data.get('metrics'):
                from app.models.base import ModelMetrics
                self.metrics = ModelMetrics(**model_data['metrics'])
            
            logger.info(f"LUBE models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load LUBE models: {e}")
            raise ModelException(f"Failed to load models: {str(e)}")