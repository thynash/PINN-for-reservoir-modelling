"""
Physics-Informed Neural Network (PINN) architecture implementation.

This module implements the core neural network architecture for PINNs with
configurable hidden layers, activation functions, and proper input/output scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.interfaces import ModelInterface
from ..core.data_models import ModelConfig


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class PINNArchitecture(nn.Module, ModelInterface):
    """
    Physics-Informed Neural Network architecture with configurable layers.
    
    Features:
    - Configurable hidden layers (3-5 layers, 50-200 neurons each)
    - Multiple activation functions (Tanh, Swish, ReLU)
    - Input normalization and output scaling
    - Proper weight initialization
    - Gradient computation capabilities
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: str = "tanh",
                 dropout_rate: float = 0.0,
                 batch_norm: bool = False,
                 weight_init: str = "xavier_uniform"):
        """
        Initialize PINN architecture.
        
        Args:
            input_dim: Number of input features (depth, porosity, permeability, etc.)
            hidden_dims: List of hidden layer dimensions [50-200 neurons each]
            output_dim: Number of outputs (typically 2: pressure, saturation)
            activation: Activation function ("tanh", "swish", "relu")
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            weight_init: Weight initialization strategy
        """
        super(PINNArchitecture, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Validate architecture parameters
        self._validate_architecture()
        
        # Set up activation function
        self.activation = self._get_activation_function(activation)
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
        if dropout_rate > 0:
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Input/output scaling parameters
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))
        
        # Initialize weights
        self._initialize_weights(weight_init)
    
    def _validate_architecture(self):
        """Validate architecture parameters."""
        if len(self.hidden_dims) < 3 or len(self.hidden_dims) > 5:
            raise ValueError("Number of hidden layers should be between 3 and 5")
        
        for dim in self.hidden_dims:
            if dim < 50 or dim > 200:
                raise ValueError("Hidden layer dimensions should be between 50 and 200")
        
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ValueError("Dropout rate should be between 0 and 1")
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation = activation.lower()
        if activation == "tanh":
            return nn.Tanh()
        elif activation == "swish":
            return Swish()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _initialize_weights(self, init_method: str):
        """Initialize network weights."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if init_method == "xavier_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif init_method == "xavier_normal":
                    nn.init.xavier_normal_(layer.weight)
                elif init_method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif init_method == "kaiming_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                else:
                    raise ValueError(f"Unsupported initialization method: {init_method}")
                
                # Initialize biases to zero
                nn.init.zeros_(layer.bias)
    
    def set_input_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Set input normalization parameters."""
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)
    
    def set_output_scaling(self, mean: torch.Tensor, std: torch.Tensor):
        """Set output scaling parameters."""
        self.output_mean.copy_(mean)
        self.output_std.copy_(std)
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        return (x - self.input_mean) / (self.input_std + 1e-8)
    
    def scale_output(self, y: torch.Tensor) -> torch.Tensor:
        """Scale output tensor to physical ranges."""
        return y * self.output_std + self.output_mean
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Normalize input
        x = self.normalize_input(x)
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            
            # Apply batch normalization
            if self.batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout
            if self.dropout_rate > 0 and i < len(self.dropouts):
                x = self.dropouts[i](x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        # Scale output to physical ranges
        x = self.scale_output(x)
        
        return x
    
    def compute_derivatives(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute spatial derivatives for physics-informed loss.
        
        Args:
            x: Input tensor with gradients enabled
            
        Returns:
            Dictionary of derivatives
        """
        if not x.requires_grad:
            raise ValueError("Input tensor must have requires_grad=True for derivative computation")
        
        # Forward pass
        outputs = self.forward(x)
        
        derivatives = {}
        
        # Compute first-order derivatives for each output
        for i in range(outputs.shape[1]):
            output_i = outputs[:, i]
            
            # Compute gradients with respect to each input dimension
            grads = torch.autograd.grad(
                outputs=output_i.sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grads is not None:
                for j in range(x.shape[1]):
                    key = f"d_output_{i}_d_input_{j}"
                    derivatives[key] = grads[:, j]
        
        return derivatives
    
    def compute_second_derivatives(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute second-order derivatives for PDE residuals.
        
        Args:
            x: Input tensor with gradients enabled
            
        Returns:
            Dictionary of second derivatives
        """
        # First compute first derivatives
        first_derivatives = self.compute_derivatives(x)
        
        second_derivatives = {}
        
        # Compute second derivatives
        for key, grad in first_derivatives.items():
            if grad is not None:
                # Extract indices from key
                parts = key.split('_')
                output_idx = int(parts[2])
                input_idx = int(parts[5])
                
                # Compute second derivative
                second_grad = torch.autograd.grad(
                    outputs=grad.sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if second_grad is not None:
                    for k in range(x.shape[1]):
                        second_key = f"d2_output_{output_idx}_d_input_{input_idx}_d_input_{k}"
                        second_derivatives[second_key] = second_grad[:, k]
        
        return second_derivatives
    
    def save_model(self, filepath: str) -> None:
        """Save model state to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activation': self.activation_name,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model state from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load normalization parameters
        if 'input_mean' in checkpoint:
            self.input_mean.copy_(checkpoint['input_mean'])
        if 'input_std' in checkpoint:
            self.input_std.copy_(checkpoint['input_std'])
        if 'output_mean' in checkpoint:
            self.output_mean.copy_(checkpoint['output_mean'])
        if 'output_std' in checkpoint:
            self.output_std.copy_(checkpoint['output_std'])
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> 'PINNArchitecture':
        """Create PINN architecture from configuration."""
        return cls(
            input_dim=len(config.input_features),
            hidden_dims=config.hidden_layers,
            output_dim=len(config.output_features),
            activation=config.activation_function,
            dropout_rate=config.dropout_rate,
            batch_norm=config.batch_normalization,
            weight_init=config.weight_initialization
        )
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activation': self.activation_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate,
            'batch_normalization': self.batch_norm
        }