"""
Model interface and utilities for PINN training and inference.

This module provides a unified interface for PINN model operations including
training, inference, checkpointing, and parameter management.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

from .pinn_architecture import PINNArchitecture
from .tensor_manager import TensorManager
from ..core.interfaces import ModelInterface
from ..core.data_models import ModelConfig, TrainingConfig, ValidationMetrics


class PINNModelInterface:
    """
    Unified interface for PINN model operations.
    
    Provides high-level interface for:
    - Model creation and configuration
    - Training and inference operations
    - Model checkpointing and loading
    - Parameter initialization and management
    - Model evaluation and metrics
    """
    
    def __init__(self, 
                 model: Optional[PINNArchitecture] = None,
                 config: Optional[ModelConfig] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize model interface.
        
        Args:
            model: PINN architecture instance
            config: Model configuration
            device: Target device for computations
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensor_manager = TensorManager(device=self.device)
        
        if model is not None:
            self.model = model.to(self.device)
        elif config is not None:
            self.model = PINNArchitecture.from_config(config).to(self.device)
        else:
            self.model = None
        
        self.config = config
        self.training_history = {}
        self.is_trained = False
    
    def create_model(self, 
                    input_dim: int,
                    hidden_dims: List[int],
                    output_dim: int,
                    **kwargs) -> PINNArchitecture:
        """
        Create PINN model with specified architecture.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output features
            **kwargs: Additional model parameters
            
        Returns:
            Created PINN model
        """
        self.model = PINNArchitecture(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            **kwargs
        ).to(self.device)
        
        return self.model
    
    def forward(self, inputs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Input data
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Convert to tensor if needed
        if isinstance(inputs, np.ndarray):
            inputs = self.tensor_manager.to_tensor(inputs)
        
        inputs = inputs.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        
        return outputs
    
    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Make predictions and return as numpy array.
        
        Args:
            inputs: Input data
            
        Returns:
            Predictions as numpy array
        """
        outputs = self.forward(inputs)
        return outputs.cpu().numpy()
    
    def compute_derivatives(self, inputs: Union[torch.Tensor, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Compute derivatives for physics-informed loss.
        
        Args:
            inputs: Input data with spatial coordinates
            
        Returns:
            Dictionary of computed derivatives
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Convert to tensor with gradients enabled
        if isinstance(inputs, np.ndarray):
            inputs = self.tensor_manager.prepare_input_tensors(inputs, requires_grad=True)
        else:
            inputs = inputs.requires_grad_(True)
        
        inputs = inputs.to(self.device)
        
        self.model.eval()
        outputs = self.model(inputs)
        
        # Compute derivatives using tensor manager
        derivatives = self.tensor_manager.compute_spatial_derivatives(outputs, inputs)
        
        return derivatives
    
    def set_normalization_parameters(self, 
                                   input_stats: Dict[str, Tuple[float, float]],
                                   output_stats: Dict[str, Tuple[float, float]]) -> None:
        """
        Set input normalization and output scaling parameters.
        
        Args:
            input_stats: Dictionary of input feature statistics (mean, std)
            output_stats: Dictionary of output feature statistics (mean, std)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Convert to tensors
        input_means = []
        input_stds = []
        for feature in sorted(input_stats.keys()):
            mean, std = input_stats[feature]
            input_means.append(mean)
            input_stds.append(std)
        
        output_means = []
        output_stds = []
        for feature in sorted(output_stats.keys()):
            mean, std = output_stats[feature]
            output_means.append(mean)
            output_stds.append(std)
        
        input_mean_tensor = torch.tensor(input_means, device=self.device, dtype=torch.float32)
        input_std_tensor = torch.tensor(input_stds, device=self.device, dtype=torch.float32)
        output_mean_tensor = torch.tensor(output_means, device=self.device, dtype=torch.float32)
        output_std_tensor = torch.tensor(output_stds, device=self.device, dtype=torch.float32)
        
        self.model.set_input_normalization(input_mean_tensor, input_std_tensor)
        self.model.set_output_scaling(output_mean_tensor, output_std_tensor)
    
    def initialize_parameters(self, strategy: str = "xavier_uniform") -> None:
        """
        Initialize model parameters using specified strategy.
        
        Args:
            strategy: Initialization strategy
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model._initialize_weights(strategy)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get model parameter counts.
        
        Returns:
            Dictionary with parameter counts
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    def save_checkpoint(self, 
                       filepath: str,
                       epoch: int,
                       optimizer_state: Optional[Dict] = None,
                       loss_history: Optional[Dict] = None,
                       metadata: Optional[Dict] = None) -> None:
        """
        Save model checkpoint with training state.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current training epoch
            optimizer_state: Optimizer state dictionary
            loss_history: Training loss history
            metadata: Additional metadata
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'output_dim': self.model.output_dim,
                'activation': self.model.activation_name,
                'dropout_rate': self.model.dropout_rate,
                'batch_norm': self.model.batch_norm
            },
            'normalization_params': {
                'input_mean': self.model.input_mean.cpu(),
                'input_std': self.model.input_std.cpu(),
                'output_mean': self.model.output_mean.cpu(),
                'output_std': self.model.output_std.cpu()
            },
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if loss_history is not None:
            checkpoint['loss_history'] = loss_history
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = False) -> Dict[str, Any]:
        """
        Load model checkpoint and restore state.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to return optimizer state
            
        Returns:
            Checkpoint information
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Create model if not exists
        if self.model is None:
            config = checkpoint['model_config']
            self.model = PINNArchitecture(
                input_dim=config['input_dim'],
                hidden_dims=config['hidden_dims'],
                output_dim=config['output_dim'],
                activation=config['activation'],
                dropout_rate=config['dropout_rate'],
                batch_norm=config['batch_norm']
            ).to(self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load normalization parameters
        norm_params = checkpoint['normalization_params']
        self.model.input_mean.copy_(norm_params['input_mean'])
        self.model.input_std.copy_(norm_params['input_std'])
        self.model.output_mean.copy_(norm_params['output_mean'])
        self.model.output_std.copy_(norm_params['output_std'])
        
        # Update training history if available
        if 'loss_history' in checkpoint:
            self.training_history = checkpoint['loss_history']
        
        result = {
            'epoch': checkpoint['epoch'],
            'timestamp': checkpoint.get('timestamp'),
            'metadata': checkpoint.get('metadata', {})
        }
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        
        self.is_trained = True
        return result
    
    def save_model_only(self, filepath: str) -> None:
        """
        Save only the model state (for deployment).
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.save_model(filepath)
    
    def load_model_only(self, filepath: str) -> None:
        """
        Load only the model state.
        
        Args:
            filepath: Path to model file
        """
        if self.model is None:
            raise ValueError("Model not initialized. Create model first.")
        
        self.model.load_model(filepath)
        self.is_trained = True
    
    def evaluate_model(self, 
                      inputs: Union[torch.Tensor, np.ndarray],
                      targets: Union[torch.Tensor, np.ndarray],
                      metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate model performance on given data.
        
        Args:
            inputs: Input data
            targets: Target values
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of computed metrics
        """
        if metrics is None:
            metrics = ['mse', 'mae', 'r2']
        
        predictions = self.predict(inputs)
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        results = {}
        
        for metric in metrics:
            if metric == 'mse':
                results['mse'] = np.mean((predictions - targets) ** 2)
            elif metric == 'mae':
                results['mae'] = np.mean(np.abs(predictions - targets))
            elif metric == 'rmse':
                results['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
            elif metric == 'r2':
                ss_res = np.sum((targets - predictions) ** 2)
                ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                results['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
            elif metric == 'relative_error':
                results['relative_error'] = np.mean(np.abs((predictions - targets) / (targets + 1e-8)))
        
        return results
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Model summary information
        """
        if self.model is None:
            return {'status': 'Model not initialized'}
        
        summary = self.model.get_model_info()
        summary.update(self.get_parameter_count())
        summary['device'] = str(self.device)
        summary['is_trained'] = self.is_trained
        
        if self.training_history:
            summary['training_epochs'] = len(self.training_history.get('total_loss', []))
        
        return summary
    
    def set_training_mode(self, training: bool = True) -> None:
        """Set model to training or evaluation mode."""
        if self.model is not None:
            self.model.train(training)
    
    def get_model(self) -> Optional[PINNArchitecture]:
        """Get the underlying model."""
        return self.model
    
    def to_device(self, device: torch.device) -> None:
        """Move model to specified device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        self.tensor_manager.set_device(device)