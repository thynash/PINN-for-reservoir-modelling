"""
Physics Loss Calculator for Physics-Informed Neural Networks

This module implements the composite physics loss that combines data loss, PDE residuals,
boundary conditions, and initial conditions into a unified training objective.
It includes adaptive loss weighting strategies and comprehensive loss monitoring.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
from collections import defaultdict

from .pde_formulator import PDEFormulator, PDEResidualCalculator
from .boundary_conditions import (
    BoundaryConditionHandler, 
    InitialConditionHandler, 
    PhysicalConstraintHandler,
    BoundaryCondition
)


@dataclass
class LossWeights:
    """Configuration for loss term weights."""
    data_weight: float = 1.0
    pde_weight: float = 1.0
    boundary_weight: float = 1.0
    initial_weight: float = 1.0
    constraint_weight: float = 0.1
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'data': self.data_weight,
            'pde': self.pde_weight,
            'boundary': self.boundary_weight,
            'initial': self.initial_weight,
            'constraint': self.constraint_weight
        }


@dataclass
class LossComponents:
    """Container for individual loss components."""
    data_loss: torch.Tensor
    pde_losses: Dict[str, torch.Tensor]
    boundary_loss: torch.Tensor
    initial_loss: torch.Tensor
    constraint_loss: torch.Tensor
    total_loss: torch.Tensor
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        result = {
            'data_loss': self.data_loss.item(),
            'boundary_loss': self.boundary_loss.item(),
            'initial_loss': self.initial_loss.item(),
            'constraint_loss': self.constraint_loss.item(),
            'total_loss': self.total_loss.item()
        }
        
        # Add PDE losses
        for pde_name, pde_loss in self.pde_losses.items():
            result[f'pde_{pde_name}_loss'] = pde_loss.item()
            
        return result


class AdaptiveLossWeighting:
    """
    Implements adaptive loss weighting strategies for balanced PINN training.
    """
    
    def __init__(self, 
                 strategy: str = 'gradnorm',
                 update_frequency: int = 100,
                 alpha: float = 0.12):
        """
        Initialize adaptive loss weighting.
        
        Args:
            strategy: Weighting strategy ('gradnorm', 'uncertainty', 'fixed')
            update_frequency: How often to update weights (in training steps)
            alpha: Learning rate for weight updates
        """
        self.strategy = strategy
        self.update_frequency = update_frequency
        self.alpha = alpha
        self.step_count = 0
        self.loss_history = defaultdict(list)
        
    def update_weights(self,
                      loss_components: LossComponents,
                      model: nn.Module,
                      current_weights: LossWeights) -> LossWeights:
        """
        Update loss weights based on the chosen strategy.
        
        Args:
            loss_components: Current loss components
            model: Neural network model
            current_weights: Current loss weights
            
        Returns:
            Updated loss weights
        """
        self.step_count += 1
        
        # Store loss history
        loss_dict = loss_components.to_dict()
        for key, value in loss_dict.items():
            self.loss_history[key].append(value)
        
        # Update weights based on strategy
        if self.strategy == 'gradnorm' and self.step_count % self.update_frequency == 0:
            return self._gradnorm_weighting(loss_components, model, current_weights)
        elif self.strategy == 'uncertainty':
            return self._uncertainty_weighting(loss_components, current_weights)
        else:
            return current_weights  # Fixed weights
    
    def _gradnorm_weighting(self,
                           loss_components: LossComponents,
                           model: nn.Module,
                           current_weights: LossWeights) -> LossWeights:
        """
        Implement GradNorm adaptive weighting.
        
        This balances the gradient magnitudes of different loss terms.
        """
        # Get model parameters
        params = list(model.parameters())
        if not params:
            return current_weights
            
        # Compute gradients for each loss term
        gradients = {}
        
        # Data loss gradient
        if loss_components.data_loss.requires_grad:
            grad_data = torch.autograd.grad(
                loss_components.data_loss, params, retain_graph=True, allow_unused=True
            )
            gradients['data'] = self._compute_gradient_norm(grad_data)
        
        # PDE loss gradients
        for pde_name, pde_loss in loss_components.pde_losses.items():
            if pde_loss.requires_grad:
                grad_pde = torch.autograd.grad(
                    pde_loss, params, retain_graph=True, allow_unused=True
                )
                gradients[f'pde_{pde_name}'] = self._compute_gradient_norm(grad_pde)
        
        # Boundary loss gradient
        if loss_components.boundary_loss.requires_grad:
            grad_boundary = torch.autograd.grad(
                loss_components.boundary_loss, params, retain_graph=True, allow_unused=True
            )
            gradients['boundary'] = self._compute_gradient_norm(grad_boundary)
        
        # Compute target gradient norm (average)
        if gradients:
            target_norm = sum(gradients.values()) / len(gradients)
            
            # Update weights to balance gradients
            new_weights = LossWeights()
            
            if 'data' in gradients:
                ratio = target_norm / (gradients['data'] + 1e-8)
                new_weights.data_weight = current_weights.data_weight * (1 + self.alpha * (ratio - 1))
                
            if 'boundary' in gradients:
                ratio = target_norm / (gradients['boundary'] + 1e-8)
                new_weights.boundary_weight = current_weights.boundary_weight * (1 + self.alpha * (ratio - 1))
            
            # Keep other weights unchanged for now
            new_weights.pde_weight = current_weights.pde_weight
            new_weights.initial_weight = current_weights.initial_weight
            new_weights.constraint_weight = current_weights.constraint_weight
            
            # Clamp weights to reasonable range
            new_weights.data_weight = max(0.1, min(10.0, new_weights.data_weight))
            new_weights.boundary_weight = max(0.1, min(10.0, new_weights.boundary_weight))
            
            return new_weights
        
        return current_weights
    
    def _uncertainty_weighting(self,
                              loss_components: LossComponents,
                              current_weights: LossWeights) -> LossWeights:
        """
        Implement uncertainty-based weighting.
        
        This uses the variance of loss terms to adjust weights.
        """
        if len(self.loss_history['data_loss']) < 10:
            return current_weights
            
        # Compute recent variance for each loss term
        window_size = min(50, len(self.loss_history['data_loss']))
        
        data_var = np.var(self.loss_history['data_loss'][-window_size:])
        boundary_var = np.var(self.loss_history['boundary_loss'][-window_size:])
        
        # Adjust weights inversely proportional to variance
        new_weights = LossWeights()
        new_weights.data_weight = current_weights.data_weight / (1 + data_var)
        new_weights.boundary_weight = current_weights.boundary_weight / (1 + boundary_var)
        new_weights.pde_weight = current_weights.pde_weight
        new_weights.initial_weight = current_weights.initial_weight
        new_weights.constraint_weight = current_weights.constraint_weight
        
        return new_weights
    
    def _compute_gradient_norm(self, gradients: Tuple) -> float:
        """Compute L2 norm of gradients."""
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm += grad.norm().item() ** 2
        return total_norm ** 0.5


class PhysicsLossCalculator:
    """
    Main class for computing composite physics-informed loss.
    """
    
    def __init__(self,
                 pde_formulator: PDEFormulator,
                 boundary_handler: Optional[BoundaryConditionHandler] = None,
                 initial_handler: Optional[InitialConditionHandler] = None,
                 constraint_handler: Optional[PhysicalConstraintHandler] = None,
                 adaptive_weighting: Optional[AdaptiveLossWeighting] = None,
                 device: str = 'cpu'):
        """
        Initialize physics loss calculator.
        
        Args:
            pde_formulator: PDE formulation handler
            boundary_handler: Boundary condition handler
            initial_handler: Initial condition handler  
            constraint_handler: Physical constraint handler
            adaptive_weighting: Adaptive loss weighting strategy
            device: Device for computations
        """
        self.device = device
        self.pde_formulator = pde_formulator
        self.pde_calculator = PDEResidualCalculator(pde_formulator)
        
        self.boundary_handler = boundary_handler or BoundaryConditionHandler(device)
        self.initial_handler = initial_handler or InitialConditionHandler(device)
        self.constraint_handler = constraint_handler or PhysicalConstraintHandler(device)
        self.adaptive_weighting = adaptive_weighting
        
        # Loss weights
        self.loss_weights = LossWeights()
        
        # Loss monitoring
        self.loss_history = []
        self.logger = logging.getLogger(__name__)
        
    def compute_data_loss(self,
                         predictions: torch.Tensor,
                         targets: torch.Tensor,
                         loss_type: str = 'mse') -> torch.Tensor:
        """
        Compute data fitting loss.
        
        Args:
            predictions: Model predictions [batch_size, n_outputs]
            targets: Target values [batch_size, n_outputs]
            loss_type: Type of loss ('mse', 'mae', 'huber')
            
        Returns:
            Data loss tensor
        """
        if loss_type == 'mse':
            return torch.mean((predictions - targets) ** 2)
        elif loss_type == 'mae':
            return torch.mean(torch.abs(predictions - targets))
        elif loss_type == 'huber':
            delta = 1.0
            diff = torch.abs(predictions - targets)
            return torch.mean(torch.where(
                diff < delta,
                0.5 * diff ** 2,
                delta * (diff - 0.5 * delta)
            ))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def compute_physics_loss(self,
                           predictions: torch.Tensor,
                           coords: torch.Tensor,
                           material_properties: Dict[str, torch.Tensor],
                           time: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute PDE residual losses.
        
        Args:
            predictions: Model predictions [batch_size, n_outputs]
            coords: Coordinate points [batch_size, n_coords]
            material_properties: Material property tensors
            time: Time coordinates (optional)
            
        Returns:
            Dictionary of PDE loss terms
        """
        return self.pde_calculator.safe_residual_computation(
            predictions, coords, material_properties, time
        )
    
    def compute_total_loss(self,
                          model: nn.Module,
                          data_batch: Dict[str, torch.Tensor],
                          physics_batch: Dict[str, torch.Tensor],
                          boundary_batch: Optional[Dict[str, torch.Tensor]] = None,
                          initial_batch: Optional[Dict[str, torch.Tensor]] = None) -> LossComponents:
        """
        Compute total composite loss.
        
        Args:
            model: Neural network model
            data_batch: Data batch with 'inputs', 'targets'
            physics_batch: Physics batch with 'coords', 'material_properties'
            boundary_batch: Boundary batch (optional)
            initial_batch: Initial condition batch (optional)
            
        Returns:
            LossComponents object with all loss terms
        """
        # Data loss
        if 'inputs' in data_batch and 'targets' in data_batch:
            data_predictions = model(data_batch['inputs'])
            data_loss = self.compute_data_loss(data_predictions, data_batch['targets'])
        else:
            data_loss = torch.tensor(0.0, device=self.device)
        
        # Physics loss (PDE residuals)
        physics_coords = physics_batch['coords']
        physics_coords.requires_grad_(True)
        
        physics_predictions = model(physics_coords)
        material_props = physics_batch.get('material_properties', {})
        time_coords = physics_batch.get('time', None)
        
        pde_losses = self.compute_physics_loss(
            physics_predictions, physics_coords, material_props, time_coords
        )
        
        # Convert PDE residuals to loss terms (MSE of residuals)
        pde_loss_terms = {}
        for pde_name, residual in pde_losses.items():
            pde_loss_terms[pde_name] = torch.mean(residual ** 2)
        
        # Boundary loss
        boundary_loss = torch.tensor(0.0, device=self.device)
        if boundary_batch is not None and len(self.boundary_handler.boundary_conditions) > 0:
            boundary_coords = boundary_batch['coords']
            boundary_coords.requires_grad_(True)
            boundary_predictions = model(boundary_coords)
            boundary_conditions = boundary_batch.get('conditions', [])
            
            boundary_loss = self.boundary_handler.compute_boundary_loss(
                boundary_predictions, boundary_coords, boundary_conditions
            )
        
        # Initial condition loss
        initial_loss = torch.tensor(0.0, device=self.device)
        if initial_batch is not None and len(self.initial_handler.initial_conditions) > 0:
            initial_coords = initial_batch['coords']
            initial_coords.requires_grad_(True)
            initial_predictions = model(initial_coords)
            
            initial_loss = self.initial_handler.compute_initial_loss(
                initial_predictions, initial_coords
            )
        
        # Physical constraint loss
        constraint_loss = self.constraint_handler.compute_constraint_loss(physics_predictions)
        
        # Combine losses with weights
        total_pde_loss = sum(pde_loss_terms.values()) if pde_loss_terms else torch.tensor(0.0, device=self.device)
        
        total_loss = (
            self.loss_weights.data_weight * data_loss +
            self.loss_weights.pde_weight * total_pde_loss +
            self.loss_weights.boundary_weight * boundary_loss +
            self.loss_weights.initial_weight * initial_loss +
            self.loss_weights.constraint_weight * constraint_loss
        )
        
        # Create loss components object
        loss_components = LossComponents(
            data_loss=data_loss,
            pde_losses=pde_loss_terms,
            boundary_loss=boundary_loss,
            initial_loss=initial_loss,
            constraint_loss=constraint_loss,
            total_loss=total_loss
        )
        
        # Update adaptive weights if enabled
        if self.adaptive_weighting is not None:
            self.loss_weights = self.adaptive_weighting.update_weights(
                loss_components, model, self.loss_weights
            )
        
        # Log loss components
        self._log_losses(loss_components)
        
        return loss_components
    
    def _log_losses(self, loss_components: LossComponents):
        """Log loss components for monitoring."""
        loss_dict = loss_components.to_dict()
        self.loss_history.append(loss_dict)
        
        # Log every 100 steps
        if len(self.loss_history) % 100 == 0:
            self.logger.info(f"Loss components: {loss_dict}")
            self.logger.info(f"Current weights: {self.loss_weights.to_dict()}")
    
    def get_loss_statistics(self, window_size: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Get statistics of recent loss components.
        
        Args:
            window_size: Number of recent steps to analyze
            
        Returns:
            Dictionary with mean, std, min, max for each loss component
        """
        if not self.loss_history:
            return {}
            
        recent_history = self.loss_history[-window_size:]
        stats = {}
        
        # Get all loss component names
        if recent_history:
            loss_names = recent_history[0].keys()
            
            for loss_name in loss_names:
                values = [step[loss_name] for step in recent_history]
                stats[loss_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'current': values[-1] if values else 0.0
                }
        
        return stats
    
    def reset_loss_history(self):
        """Reset loss history for new training run."""
        self.loss_history.clear()
        if self.adaptive_weighting:
            self.adaptive_weighting.loss_history.clear()
            self.adaptive_weighting.step_count = 0
    
    def set_loss_weights(self, **kwargs):
        """
        Set loss weights manually.
        
        Args:
            **kwargs: Weight values (data_weight, pde_weight, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self.loss_weights, key):
                setattr(self.loss_weights, key, value)
            else:
                self.logger.warning(f"Unknown loss weight: {key}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.loss_weights.to_dict()


class LossMonitor:
    """
    Utility class for monitoring and visualizing loss evolution during training.
    """
    
    def __init__(self, save_frequency: int = 10):
        """
        Initialize loss monitor.
        
        Args:
            save_frequency: How often to save loss snapshots
        """
        self.save_frequency = save_frequency
        self.step_count = 0
        self.snapshots = []
        
    def update(self, loss_components: LossComponents, weights: LossWeights):
        """
        Update monitor with new loss values.
        
        Args:
            loss_components: Current loss components
            weights: Current loss weights
        """
        self.step_count += 1
        
        if self.step_count % self.save_frequency == 0:
            snapshot = {
                'step': self.step_count,
                'losses': loss_components.to_dict(),
                'weights': weights.to_dict(),
                'timestamp': torch.tensor(self.step_count).float()
            }
            self.snapshots.append(snapshot)
    
    def get_loss_curves(self) -> Dict[str, List[float]]:
        """
        Get loss evolution curves for plotting.
        
        Returns:
            Dictionary with loss curves for each component
        """
        if not self.snapshots:
            return {}
            
        curves = defaultdict(list)
        
        for snapshot in self.snapshots:
            for loss_name, loss_value in snapshot['losses'].items():
                curves[loss_name].append(loss_value)
                
        return dict(curves)
    
    def export_to_csv(self, filename: str):
        """Export loss history to CSV file."""
        import pandas as pd
        
        if not self.snapshots:
            return
            
        # Flatten snapshots into rows
        rows = []
        for snapshot in self.snapshots:
            row = {'step': snapshot['step']}
            row.update(snapshot['losses'])
            row.update({f'weight_{k}': v for k, v in snapshot['weights'].items()})
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Loss history exported to {filename}")


# Factory functions for common setups

def create_standard_physics_loss(device: str = 'cpu',
                               adaptive_weighting: bool = True) -> PhysicsLossCalculator:
    """
    Create a standard physics loss calculator with common settings.
    
    Args:
        device: Device for computations
        adaptive_weighting: Whether to use adaptive loss weighting
        
    Returns:
        Configured PhysicsLossCalculator
    """
    # Create components
    pde_formulator = PDEFormulator(device)
    boundary_handler = BoundaryConditionHandler(device)
    initial_handler = InitialConditionHandler(device)
    constraint_handler = PhysicalConstraintHandler(device)
    
    # Create adaptive weighting if requested
    adaptive_weighter = None
    if adaptive_weighting:
        adaptive_weighter = AdaptiveLossWeighting(
            strategy='gradnorm',
            update_frequency=100,
            alpha=0.12
        )
    
    # Create loss calculator
    loss_calculator = PhysicsLossCalculator(
        pde_formulator=pde_formulator,
        boundary_handler=boundary_handler,
        initial_handler=initial_handler,
        constraint_handler=constraint_handler,
        adaptive_weighting=adaptive_weighter,
        device=device
    )
    
    return loss_calculator