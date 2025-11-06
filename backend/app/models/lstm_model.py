"""
LSTM model for weather time series prediction.
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
from app.ml.model_utils import ModelEvaluator, ModelSerializer, DataSplitter
from app.core.config import settings


class LSTMNetwork(nn.Module):
    """PyTorch LSTM network for weather prediction."""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        output_size: int = 4,
        dropout: float = 0.2
    ):
        super(LSTMNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        dropped = self.dropout(last_output)
        
        # Final prediction
        output = self.fc(dropped)  # (batch_size, output_size)
        
        return output
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden states."""
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class LSTMWeatherModel(WeatherModel):
    """LSTM-based weather prediction model."""
    
    def __init__(self, name: str = "LSTM", config: Dict[str, Any] = None):
        default_config = {
            'sequence_length': settings.lstm_sequence_length,
            'hidden_size': settings.lstm_hidden_size,
            'num_layers': settings.lstm_num_layers,
            'dropout': settings.lstm_dropout,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'target_columns': ['temperature', 'humidity', 'pressure', 'wind_speed']
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network: Optional[LSTMNetwork] = None
        self.scaler_fitted = False
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        
        logger.info(f"LSTM model initialized with device: {self.device}")
    
    def _prepare_sequences(self, data: np.ndarray, targets: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for LSTM training/prediction.
        
        Args:
            data: Input feature data
            targets: Target data (optional for prediction)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        sequence_length = self.config['sequence_length']
        
        if len(data) < sequence_length:
            raise ValueError(f"Data length {len(data)} is less than sequence length {sequence_length}")
        
        X_sequences = []
        y_sequences = [] if targets is not None else None
        
        for i in range(len(data) - sequence_length + 1):
            # Input sequence
            X_sequences.append(data[i:i + sequence_length])
            
            # Target (next time step)
            if targets is not None:
                if i + sequence_length < len(targets):
                    y_sequences.append(targets[i + sequence_length])
        
        X_sequences = np.array(X_sequences)
        
        if y_sequences is not None:
            y_sequences = np.array(y_sequences)
        
        return X_sequences, y_sequences
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the LSTM model on weather data.
        
        Args:
            data: DataFrame with preprocessed weather features and targets
        """
        try:
            logger.info("Starting LSTM model training")
            
            # Prepare feature and target columns
            target_columns = self.config['target_columns']
            feature_columns = [col for col in data.columns if col not in target_columns + ['timestamp']]
            
            self.feature_names = feature_columns
            self.target_names = target_columns
            
            # Extract features and targets
            X = data[feature_columns].values.astype(np.float32)
            y = data[target_columns].values.astype(np.float32)
            
            # Prepare sequences
            X_seq, y_seq = self._prepare_sequences(X, y)
            
            if len(X_seq) == 0:
                raise ModelTrainingException("No sequences could be created from the data")
            
            logger.info(f"Created {len(X_seq)} sequences for training")
            
            # Split data
            train_size = int(0.8 * len(X_seq))
            val_size = int(0.1 * len(X_seq))
            
            X_train = X_seq[:train_size]
            y_train = y_seq[:train_size]
            X_val = X_seq[train_size:train_size + val_size]
            y_val = y_seq[train_size:train_size + val_size]
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
            
            # Initialize network
            input_size = X_train.shape[2]  # Number of features
            output_size = len(target_columns)
            
            self.network = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=output_size,
                dropout=self.config['dropout']
            ).to(self.device)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.network.parameters(), lr=self.config['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.config['epochs']):
                # Training phase
                self.network.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.network(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation phase
                self.network.eval()
                with torch.no_grad():
                    val_outputs = self.network(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = self.network.state_dict().copy()
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                
                if patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Load best model state
            self.network.load_state_dict(best_model_state)
            
            # Calculate final metrics
            self.network.eval()
            with torch.no_grad():
                train_pred = self.network(X_train_tensor).cpu().numpy()
                val_pred = self.network(X_val_tensor).cpu().numpy()
            
            # Calculate metrics for each target
            train_metrics = ModelEvaluator.calculate_metrics(y_train.flatten(), train_pred.flatten())
            val_metrics = ModelEvaluator.calculate_metrics(y_val.flatten(), val_pred.flatten())
            
            self.metrics = val_metrics
            self.is_trained = True
            
            # Create model info
            self.model_info = ModelInfo(
                name=self.name,
                model_type="LSTM",
                version="1.0",
                training_window_days=settings.training_data_days,
                feature_columns=self.feature_names,
                target_columns=self.target_names,
                hyperparameters=self.config,
                created_at=datetime.utcnow(),
                last_trained=datetime.utcnow()
            )
            
            logger.info(f"LSTM training completed. Validation MAE: {val_metrics.mae:.4f}, RMSE: {val_metrics.rmse:.4f}")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise ModelTrainingException(f"LSTM training failed: {str(e)}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained LSTM model.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_trained or self.network is None:
            raise ModelPredictionException("Model not trained")
        
        if not self.validate_input(features):
            raise ModelPredictionException("Invalid input features")
        
        try:
            # Prepare sequences
            X_seq, _ = self._prepare_sequences(features.astype(np.float32))
            
            if len(X_seq) == 0:
                raise ModelPredictionException("Cannot create sequences from input data")
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            # Predict
            self.network.eval()
            with torch.no_grad():
                predictions = self.network(X_tensor)
                predictions = predictions.cpu().numpy()
            
            # Return the last prediction (most recent)
            return predictions[-1] if len(predictions) > 0 else np.zeros(len(self.target_names))
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            raise ModelPredictionException(f"LSTM prediction failed: {str(e)}")
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates using Monte Carlo dropout.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained or self.network is None:
            raise ModelPredictionException("Model not trained")
        
        try:
            # Prepare sequences
            X_seq, _ = self._prepare_sequences(features.astype(np.float32))
            
            if len(X_seq) == 0:
                raise ModelPredictionException("Cannot create sequences from input data")
            
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            # Monte Carlo dropout for uncertainty estimation
            n_samples = 50
            predictions = []
            
            self.network.train()  # Enable dropout
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.network(X_tensor)
                    predictions.append(pred.cpu().numpy())
            
            predictions = np.array(predictions)  # Shape: (n_samples, n_sequences, n_targets)
            
            # Calculate statistics
            mean_pred = np.mean(predictions, axis=0)[-1]  # Last sequence prediction
            std_pred = np.std(predictions, axis=0)[-1]
            
            # Calculate confidence intervals (95%)
            lower_bounds = mean_pred - 1.96 * std_pred
            upper_bounds = mean_pred + 1.96 * std_pred
            
            return mean_pred, lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"LSTM uncertainty prediction failed: {e}")
            raise ModelPredictionException(f"LSTM uncertainty prediction failed: {str(e)}")
    
    def predict_sequence(self, features: np.ndarray, steps: int = 24) -> np.ndarray:
        """
        Generate multi-step predictions for future time steps.
        
        Args:
            features: Input features for prediction
            steps: Number of future steps to predict
            
        Returns:
            Array of predictions for each step
        """
        if not self.is_trained or self.network is None:
            raise ModelPredictionException("Model not trained")
        
        try:
            sequence_length = self.config['sequence_length']
            
            # Use the last sequence_length points as initial input
            if len(features) < sequence_length:
                raise ModelPredictionException(f"Need at least {sequence_length} input points")
            
            current_sequence = features[-sequence_length:].copy().astype(np.float32)
            predictions = []
            
            self.network.eval()
            
            for step in range(steps):
                # Prepare current sequence
                X_seq = current_sequence.reshape(1, sequence_length, -1)
                X_tensor = torch.FloatTensor(X_seq).to(self.device)
                
                # Predict next step
                with torch.no_grad():
                    pred = self.network(X_tensor)
                    pred_np = pred.cpu().numpy()[0]  # Remove batch dimension
                
                predictions.append(pred_np)
                
                # Update sequence for next prediction
                # Create new feature vector with predicted targets
                new_features = current_sequence[-1].copy()
                
                # Update target columns with predictions
                target_indices = [self.feature_names.index(col) for col in self.target_names if col in self.feature_names]
                for i, target_idx in enumerate(target_indices):
                    if i < len(pred_np):
                        new_features[target_idx] = pred_np[i]
                
                # Shift sequence and add new features
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = new_features
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"LSTM sequence prediction failed: {e}")
            raise ModelPredictionException(f"LSTM sequence prediction failed: {str(e)}")
    
    def save_model(self, filepath: str) -> None:
        """Save the LSTM model to disk."""
        if self.network is None:
            raise ModelException("No model to save")
        
        try:
            model_data = {
                'network_state_dict': self.network.state_dict(),
                'config': self.config,
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'model_info': self.model_info.dict() if self.model_info else None,
                'metrics': self.metrics.__dict__ if self.metrics else None,
                'is_trained': self.is_trained
            }
            
            torch.save(model_data, filepath)
            logger.info(f"LSTM model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save LSTM model: {e}")
            raise ModelException(f"Failed to save model: {str(e)}")
    
    def load_model(self, filepath: str) -> None:
        """Load the LSTM model from disk."""
        try:
            model_data = torch.load(filepath, map_location=self.device)
            
            self.config = model_data['config']
            self.feature_names = model_data['feature_names']
            self.target_names = model_data['target_names']
            self.is_trained = model_data['is_trained']
            
            # Recreate network
            input_size = len(self.feature_names)
            output_size = len(self.target_names)
            
            self.network = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=output_size,
                dropout=self.config['dropout']
            ).to(self.device)
            
            # Load state dict
            self.network.load_state_dict(model_data['network_state_dict'])
            
            # Load metadata
            if model_data.get('model_info'):
                self.model_info = ModelInfo(**model_data['model_info'])
            
            if model_data.get('metrics'):
                from app.models.base import ModelMetrics
                self.metrics = ModelMetrics(**model_data['metrics'])
            
            logger.info(f"LSTM model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            raise ModelException(f"Failed to load model: {str(e)}")