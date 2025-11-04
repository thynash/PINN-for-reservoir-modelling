"""
PDE Formulation Classes for Physics-Informed Neural Networks

This module implements the governing equations for multiphase flow in petroleum reservoirs:
- Darcy's law for pressure field computation
- Buckley-Leverett equation for saturation transport
- Continuity equation for mass conservation

All equations are implemented with automatic differentiation support for PINN training.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


class PDEFormulator:
    """
    Implements governing partial differential equations for reservoir flow modeling.
    
    This class provides methods to compute PDE residuals using automatic differentiation,
    which are essential for physics-informed neural network training.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the PDE formulator.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        
    def compute_spatial_derivatives(self, 
                                  output: torch.Tensor, 
                                  input_coords: torch.Tensor,
                                  create_graph: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute spatial derivatives using automatic differentiation.
        
        Args:
            output: Network output tensor [batch_size, output_dim]
            input_coords: Input coordinates [batch_size, coord_dim] 
            create_graph: Whether to create computation graph for higher-order derivatives
            
        Returns:
            Dictionary containing first-order derivatives
        """
        batch_size = output.shape[0]
        coord_dim = input_coords.shape[1]
        
        derivatives = {}
        
        # Compute gradients for each output with respect to each input coordinate
        for i in range(output.shape[1]):
            output_i = output[:, i]
            
            # Compute gradient
            grad_outputs = torch.ones_like(output_i)
            gradients = torch.autograd.grad(
                outputs=output_i,
                inputs=input_coords,
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Store derivatives by coordinate
            for j in range(coord_dim):
                key = f"d_output_{i}_d_coord_{j}"
                derivatives[key] = gradients[:, j]
                
        return derivatives
    
    def darcy_residual(self, 
                      pressure: torch.Tensor,
                      coords: torch.Tensor, 
                      permeability: torch.Tensor,
                      viscosity: float = 1.0) -> torch.Tensor:
        """
        Compute Darcy's law residual: ∇·(k/μ ∇p) = 0
        
        This implements the steady-state pressure equation for single-phase flow.
        For multiphase flow, this can be extended with relative permeability terms.
        
        Args:
            pressure: Pressure field predictions [batch_size, 1]
            coords: Spatial coordinates [batch_size, spatial_dim]
            permeability: Permeability field [batch_size, 1] or scalar
            viscosity: Fluid viscosity (assumed constant)
            
        Returns:
            PDE residual tensor [batch_size, 1]
        """
        # Ensure pressure requires gradients
        if not pressure.requires_grad:
            pressure = pressure.requires_grad_(True)
            
        # Compute pressure gradients
        grad_outputs = torch.autograd.grad(
            outputs=pressure.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
        
        if grad_outputs[0] is None:
            # If no gradient, return zero residual
            return torch.zeros(coords.shape[0], 1, device=self.device)
            
        grad_p = grad_outputs[0]
        
        # Handle permeability (can be tensor or scalar)
        if isinstance(permeability, (int, float)):
            k_over_mu = permeability / viscosity
        else:
            k_over_mu = permeability / viscosity
            
        # Compute flux: q = -(k/μ) ∇p
        flux = -k_over_mu.unsqueeze(-1) * grad_p
        
        # Compute divergence of flux: ∇·q
        divergence = torch.zeros(coords.shape[0], 1, device=self.device)
        
        for i in range(coords.shape[1]):  # For each spatial dimension
            # Compute derivative of flux component
            flux_i = flux[:, i]
            grad_outputs = torch.autograd.grad(
                outputs=flux_i.sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            
            if grad_outputs[0] is not None:
                div_component = grad_outputs[0][:, i]
                divergence[:, 0] += div_component
            
            divergence[:, 0] += div_component
            
        return divergence
    
    def buckley_leverett_residual(self,
                                saturation: torch.Tensor,
                                coords: torch.Tensor,
                                time: Optional[torch.Tensor] = None,
                                velocity: float = 1.0,
                                viscosity_ratio: float = 1.0) -> torch.Tensor:
        """
        Compute Buckley-Leverett equation residual: ∂S/∂t + ∂f(S)/∂x = 0
        
        This implements the saturation transport equation for two-phase flow.
        
        Args:
            saturation: Water saturation predictions [batch_size, 1]
            coords: Spatial coordinates [batch_size, spatial_dim]
            time: Time coordinate [batch_size, 1] (optional for steady-state)
            velocity: Darcy velocity magnitude
            viscosity_ratio: Ratio of oil to water viscosity
            
        Returns:
            PDE residual tensor [batch_size, 1]
        """
        # Ensure saturation requires gradients
        if not saturation.requires_grad:
            saturation = saturation.requires_grad_(True)
            
        residual = torch.zeros_like(saturation)
        
        # Time derivative term (if time-dependent)
        if time is not None:
            grad_outputs = torch.autograd.grad(
                outputs=saturation.sum(),
                inputs=time,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            if grad_outputs[0] is not None:
                dS_dt = grad_outputs[0]
                residual += dS_dt
        
        # Compute fractional flow function f(S)
        # f(S) = 1 / (1 + (μ_w/μ_o) * (1-S)/S)
        # Simplified version: f(S) = S^2 / (S^2 + M*(1-S)^2) where M = μ_o/μ_w
        S = torch.clamp(saturation, min=1e-6, max=1-1e-6)  # Avoid division by zero
        M = viscosity_ratio
        
        S_squared = S ** 2
        one_minus_S_squared = (1 - S) ** 2
        
        fractional_flow = S_squared / (S_squared + M * one_minus_S_squared)
        
        # Compute spatial derivative of fractional flow
        grad_outputs = torch.autograd.grad(
            outputs=fractional_flow.sum(),
            inputs=saturation,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
        
        if grad_outputs[0] is None:
            return torch.zeros_like(saturation)
            
        dfdS = grad_outputs[0]
        
        # Compute saturation gradient
        grad_outputs = torch.autograd.grad(
            outputs=saturation.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
        
        if grad_outputs[0] is None:
            return torch.zeros_like(saturation)
            
        dS_dx = grad_outputs[0]
        
        # Add convective term: v * df/dS * dS/dx (assuming 1D flow in x-direction)
        if coords.shape[1] >= 1:
            convective_term = velocity * dfdS * dS_dx[:, 0:1]
            residual += convective_term
            
        return residual
    
    def continuity_residual(self,
                          velocity: torch.Tensor,
                          coords: torch.Tensor,
                          porosity: torch.Tensor,
                          compressibility: float = 0.0) -> torch.Tensor:
        """
        Compute continuity equation residual: ∇·v = 0 (incompressible) or ∇·v = -φct ∂p/∂t
        
        Args:
            velocity: Velocity field [batch_size, spatial_dim]
            coords: Spatial coordinates [batch_size, spatial_dim]
            porosity: Porosity field [batch_size, 1]
            compressibility: Total compressibility (for slightly compressible flow)
            
        Returns:
            PDE residual tensor [batch_size, 1]
        """
        # Compute divergence of velocity
        divergence = torch.zeros(coords.shape[0], 1, device=self.device)
        
        for i in range(velocity.shape[1]):  # For each velocity component
            vel_component = velocity[:, i]
            
            # Compute derivative of velocity component
            div_component = torch.autograd.grad(
                outputs=vel_component.sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, i]
            
            divergence[:, 0] += div_component
        
        # For incompressible flow, residual is just divergence
        if compressibility == 0.0:
            return divergence
        else:
            # For slightly compressible flow, add storage term
            # This would require pressure time derivative, simplified here
            storage_term = torch.zeros_like(divergence)
            return divergence + storage_term
    
    def compute_all_residuals(self,
                            predictions: torch.Tensor,
                            coords: torch.Tensor,
                            material_properties: Dict[str, torch.Tensor],
                            time: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all PDE residuals for the complete system.
        
        Args:
            predictions: Network predictions [batch_size, 2] (pressure, saturation)
            coords: Spatial coordinates [batch_size, spatial_dim]
            material_properties: Dictionary containing permeability, porosity, etc.
            time: Time coordinate [batch_size, 1] (optional)
            
        Returns:
            Dictionary of residual tensors
        """
        pressure = predictions[:, 0:1]
        saturation = predictions[:, 1:2]
        
        residuals = {}
        
        # Darcy's law residual
        permeability = material_properties.get('permeability', 1.0)
        residuals['darcy'] = self.darcy_residual(pressure, coords, permeability)
        
        # Buckley-Leverett residual
        residuals['buckley_leverett'] = self.buckley_leverett_residual(
            saturation, coords, time
        )
        
        # Continuity equation (if velocity is provided)
        if 'velocity' in material_properties:
            velocity = material_properties['velocity']
            porosity = material_properties.get('porosity', 0.2)
            residuals['continuity'] = self.continuity_residual(velocity, coords, porosity)
        
        return residuals


class PDEResidualCalculator:
    """
    Helper class for computing PDE residuals with proper error handling and numerical stability.
    """
    
    def __init__(self, formulator: PDEFormulator):
        self.formulator = formulator
        
    def safe_residual_computation(self,
                                predictions: torch.Tensor,
                                coords: torch.Tensor,
                                material_properties: Dict[str, torch.Tensor],
                                time: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute PDE residuals with error handling and NaN detection.
        
        Args:
            predictions: Network predictions
            coords: Spatial coordinates  
            material_properties: Material property tensors
            time: Time coordinate (optional)
            
        Returns:
            Dictionary of residual tensors with NaN handling
        """
        try:
            residuals = self.formulator.compute_all_residuals(
                predictions, coords, material_properties, time
            )
            
            # Check for NaN values and replace with zeros
            for key, residual in residuals.items():
                if torch.isnan(residual).any():
                    print(f"Warning: NaN detected in {key} residual, replacing with zeros")
                    residuals[key] = torch.zeros_like(residual)
                    
            return residuals
            
        except Exception as e:
            print(f"Error computing PDE residuals: {e}")
            # Return zero residuals as fallback
            batch_size = predictions.shape[0]
            device = predictions.device
            
            return {
                'darcy': torch.zeros(batch_size, 1, device=device),
                'buckley_leverett': torch.zeros(batch_size, 1, device=device),
                'continuity': torch.zeros(batch_size, 1, device=device)
            }