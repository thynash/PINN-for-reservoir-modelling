"""
Tensor management and gradient computation utilities for PINN training.

This module provides utilities for proper tensor handling, gradient computation,
and numerical stability during PINN training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import warnings


class TensorManager:
    """
    Manages tensor operations and gradient computation for PINN training.
    
    Features:
    - Proper tensor creation and device management
    - Gradient enabling and computation
    - Automatic differentiation for spatial derivatives
    - Gradient clipping and NaN detection
    - Numerical stability utilities
    """
    
    def __init__(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        """
        Initialize tensor manager.
        
        Args:
            device: Target device (CPU/GPU)
            dtype: Default tensor data type
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        
        # Gradient computation settings
        self.gradient_clip_value = 1.0
        self.nan_detection_enabled = True
        self.eps = 1e-8  # Small value for numerical stability
    
    def to_tensor(self, data: Union[np.ndarray, List, float], 
                  requires_grad: bool = False,
                  device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert data to tensor with proper device and gradient settings.
        
        Args:
            data: Input data (numpy array, list, or scalar)
            requires_grad: Whether tensor should track gradients
            device: Target device (uses default if None)
            
        Returns:
            Tensor on specified device with gradient tracking
        """
        if device is None:
            device = self.device
        
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=device, dtype=self.dtype)
        else:
            tensor = torch.tensor(data, dtype=self.dtype, device=device)
        
        if requires_grad:
            tensor = tensor.requires_grad_(True)
        
        return tensor
    
    def prepare_input_tensors(self, 
                            inputs: np.ndarray,
                            requires_grad: bool = True) -> torch.Tensor:
        """
        Prepare input tensors for PINN forward pass and gradient computation.
        
        Args:
            inputs: Input data array [batch_size, input_dim]
            requires_grad: Whether to enable gradient computation
            
        Returns:
            Prepared tensor with gradients enabled
        """
        tensor = self.to_tensor(inputs, requires_grad=requires_grad)
        
        # Ensure tensor is 2D
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def enable_gradients(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Enable gradient computation for tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor with gradients enabled
        """
        return tensor.requires_grad_(True)
    
    def compute_spatial_derivatives(self, 
                                  outputs: torch.Tensor,
                                  inputs: torch.Tensor,
                                  order: int = 1) -> Dict[str, torch.Tensor]:
        """
        Compute spatial derivatives using automatic differentiation.
        
        Args:
            outputs: Model outputs [batch_size, output_dim]
            inputs: Model inputs [batch_size, input_dim]
            order: Derivative order (1 or 2)
            
        Returns:
            Dictionary of computed derivatives
        """
        if not inputs.requires_grad:
            raise ValueError("Input tensor must have requires_grad=True")
        
        derivatives = {}
        
        # Compute first-order derivatives
        for i in range(outputs.shape[1]):
            for j in range(inputs.shape[1]):
                # Compute gradient of output i with respect to input j
                grad = torch.autograd.grad(
                    outputs=outputs[:, i],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(outputs[:, i]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if grad is not None:
                    derivatives[f"du{i}_dx{j}"] = grad[:, j]
        
        # Compute second-order derivatives if requested
        if order >= 2:
            second_derivatives = self._compute_second_derivatives(derivatives, inputs)
            derivatives.update(second_derivatives)
        
        return derivatives
    
    def _compute_second_derivatives(self, 
                                  first_derivatives: Dict[str, torch.Tensor],
                                  inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute second-order derivatives from first derivatives.
        
        Args:
            first_derivatives: Dictionary of first derivatives
            inputs: Input tensor
            
        Returns:
            Dictionary of second derivatives
        """
        second_derivatives = {}
        
        for key, grad in first_derivatives.items():
            if grad is not None:
                # Parse the key to get indices
                parts = key.split('_')
                output_idx = parts[0][-1]  # Extract from 'du0'
                input_idx = parts[1][-1]   # Extract from 'dx0'
                
                # Compute second derivative
                for k in range(inputs.shape[1]):
                    second_grad = torch.autograd.grad(
                        outputs=grad,
                        inputs=inputs,
                        grad_outputs=torch.ones_like(grad),
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    
                    if second_grad is not None:
                        key_name = f"d2u{output_idx}_dx{input_idx}dx{k}"
                        second_derivatives[key_name] = second_grad[:, k]
        
        return second_derivatives
    
    def compute_laplacian(self, 
                         outputs: torch.Tensor,
                         inputs: torch.Tensor,
                         output_idx: int = 0) -> torch.Tensor:
        """
        Compute Laplacian (sum of second derivatives) for a specific output.
        
        Args:
            outputs: Model outputs
            inputs: Model inputs
            output_idx: Index of output to compute Laplacian for
            
        Returns:
            Laplacian tensor
        """
        laplacian = torch.zeros_like(outputs[:, output_idx])
        
        # Compute second derivatives and sum them
        for i in range(inputs.shape[1]):
            # First derivative
            first_grad = torch.autograd.grad(
                outputs=outputs[:, output_idx],
                inputs=inputs,
                grad_outputs=torch.ones_like(outputs[:, output_idx]),
                create_graph=True,
                retain_graph=True
            )[0][:, i]
            
            # Second derivative
            second_grad = torch.autograd.grad(
                outputs=first_grad,
                inputs=inputs,
                grad_outputs=torch.ones_like(first_grad),
                create_graph=True,
                retain_graph=True
            )[0][:, i]
            
            laplacian += second_grad
        
        return laplacian
    
    def clip_gradients(self, model: nn.Module, max_norm: Optional[float] = None) -> float:
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            model: Neural network model
            max_norm: Maximum gradient norm (uses default if None)
            
        Returns:
            Total gradient norm before clipping
        """
        if max_norm is None:
            max_norm = self.gradient_clip_value
        
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return total_norm.item()
    
    def detect_nan_gradients(self, model: nn.Module) -> bool:
        """
        Detect NaN values in model gradients.
        
        Args:
            model: Neural network model
            
        Returns:
            True if NaN gradients detected
        """
        if not self.nan_detection_enabled:
            return False
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    warnings.warn(f"NaN gradient detected in parameter: {name}")
                    return True
        
        return False
    
    def handle_nan_gradients(self, model: nn.Module, strategy: str = "zero") -> None:
        """
        Handle NaN gradients using specified strategy.
        
        Args:
            model: Neural network model
            strategy: Strategy for handling NaNs ("zero", "skip")
        """
        if strategy == "zero":
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = torch.where(torch.isnan(param.grad), 
                                           torch.zeros_like(param.grad), 
                                           param.grad)
        elif strategy == "skip":
            # Skip this optimization step - gradients will be None
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    param.grad = None
    
    def check_tensor_health(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """
        Check tensor for NaN, Inf, or other problematic values.
        
        Args:
            tensor: Tensor to check
            name: Name for logging purposes
            
        Returns:
            True if tensor is healthy
        """
        # Handle empty tensors
        if tensor.numel() == 0:
            return True
        
        if torch.isnan(tensor).any():
            warnings.warn(f"NaN values detected in {name}")
            return False
        
        if torch.isinf(tensor).any():
            warnings.warn(f"Inf values detected in {name}")
            return False
        
        if tensor.abs().max() > 1e6:
            warnings.warn(f"Very large values detected in {name}: max={tensor.abs().max()}")
            return False
        
        return True
    
    def safe_divide(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """
        Perform safe division with numerical stability.
        
        Args:
            numerator: Numerator tensor
            denominator: Denominator tensor
            
        Returns:
            Result of safe division
        """
        return numerator / (denominator + self.eps)
    
    def normalize_tensor(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Normalize tensor along specified dimension.
        
        Args:
            tensor: Input tensor
            dim: Dimension to normalize along
            
        Returns:
            Normalized tensor
        """
        mean = tensor.mean(dim=dim, keepdim=True)
        std = tensor.std(dim=dim, keepdim=True)
        return (tensor - mean) / (std + self.eps)
    
    def create_coordinate_tensor(self, 
                               coordinates: np.ndarray,
                               requires_grad: bool = True) -> torch.Tensor:
        """
        Create coordinate tensor for spatial derivatives.
        
        Args:
            coordinates: Coordinate array [batch_size, spatial_dim]
            requires_grad: Whether to enable gradients
            
        Returns:
            Coordinate tensor with gradients enabled
        """
        coord_tensor = self.to_tensor(coordinates, requires_grad=requires_grad)
        
        # Ensure proper shape
        if coord_tensor.dim() == 1:
            coord_tensor = coord_tensor.unsqueeze(-1)
        
        return coord_tensor
    
    def batch_jacobian(self, 
                      outputs: torch.Tensor,
                      inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian matrix for batch of inputs.
        
        Args:
            outputs: Model outputs [batch_size, output_dim]
            inputs: Model inputs [batch_size, input_dim]
            
        Returns:
            Jacobian tensor [batch_size, output_dim, input_dim]
        """
        batch_size, output_dim = outputs.shape
        input_dim = inputs.shape[1]
        
        jacobian = torch.zeros(batch_size, output_dim, input_dim, 
                             device=outputs.device, dtype=outputs.dtype)
        
        for i in range(output_dim):
            grads = torch.autograd.grad(
                outputs=outputs[:, i],
                inputs=inputs,
                grad_outputs=torch.ones_like(outputs[:, i]),
                create_graph=True,
                retain_graph=True
            )[0]
            
            jacobian[:, i, :] = grads
        
        return jacobian
    
    def set_gradient_clip_value(self, value: float) -> None:
        """Set gradient clipping value."""
        self.gradient_clip_value = value
    
    def enable_nan_detection(self, enabled: bool = True) -> None:
        """Enable or disable NaN detection."""
        self.nan_detection_enabled = enabled
    
    def get_device(self) -> torch.device:
        """Get current device."""
        return self.device
    
    def set_device(self, device: torch.device) -> None:
        """Set target device."""
        self.device = device