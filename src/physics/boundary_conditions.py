"""
Boundary and Initial Condition Handlers for Physics-Informed Neural Networks

This module implements various boundary condition types and initial condition enforcement
for reservoir flow modeling problems. It supports Dirichlet, Neumann, and Robin boundary
conditions, as well as physical constraint enforcement.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import numpy as np


class BoundaryType(Enum):
    """Enumeration of supported boundary condition types."""
    DIRICHLET = "dirichlet"  # Fixed value: u = g
    NEUMANN = "neumann"      # Fixed flux: ∂u/∂n = g  
    ROBIN = "robin"          # Mixed: αu + β∂u/∂n = g
    PERIODIC = "periodic"    # Periodic: u(x1) = u(x2)
    NO_FLOW = "no_flow"      # Zero flux: ∂u/∂n = 0


class BoundaryCondition:
    """
    Represents a single boundary condition specification.
    """
    
    def __init__(self,
                 boundary_type: BoundaryType,
                 location: Dict[str, float],
                 value: float,
                 variable_index: int = 0,
                 normal_direction: Optional[List[float]] = None,
                 alpha: float = 1.0,
                 beta: float = 0.0):
        """
        Initialize boundary condition.
        
        Args:
            boundary_type: Type of boundary condition
            location: Dictionary specifying boundary location (e.g., {'x': 0.0})
            value: Boundary condition value
            variable_index: Index of variable this BC applies to (0=pressure, 1=saturation)
            normal_direction: Outward normal vector for Neumann/Robin BCs
            alpha: Coefficient for Robin BC (αu + β∂u/∂n = g)
            beta: Coefficient for Robin BC
        """
        self.boundary_type = boundary_type
        self.location = location
        self.value = value
        self.variable_index = variable_index
        self.normal_direction = normal_direction or [1.0, 0.0]
        self.alpha = alpha
        self.beta = beta


class BoundaryConditionHandler:
    """
    Handles enforcement of boundary conditions in PINN training.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize boundary condition handler.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        self.boundary_conditions: List[BoundaryCondition] = []
        
    def add_boundary_condition(self, bc: BoundaryCondition):
        """Add a boundary condition to the handler."""
        self.boundary_conditions.append(bc)
        
    def clear_boundary_conditions(self):
        """Clear all boundary conditions."""
        self.boundary_conditions.clear()
        
    def generate_boundary_points(self,
                                domain_bounds: Dict[str, Tuple[float, float]],
                                n_points_per_boundary: int = 100) -> Tuple[torch.Tensor, List[BoundaryCondition]]:
        """
        Generate boundary points for training.
        
        Args:
            domain_bounds: Dictionary of domain bounds {'x': (min, max), 'y': (min, max)}
            n_points_per_boundary: Number of points to generate per boundary
            
        Returns:
            Tuple of (boundary_points, corresponding_boundary_conditions)
        """
        boundary_points = []
        boundary_bcs = []
        
        # Generate points for each boundary
        for coord_name, (min_val, max_val) in domain_bounds.items():
            # Lower boundary
            if coord_name == 'x':
                # Left boundary (x = min_val)
                other_coords = [k for k in domain_bounds.keys() if k != coord_name]
                if other_coords:
                    other_coord = other_coords[0]
                    other_min, other_max = domain_bounds[other_coord]
                    other_values = torch.linspace(other_min, other_max, n_points_per_boundary)
                    x_values = torch.full_like(other_values, min_val)
                    
                    if coord_name == 'x' and other_coord == 'y':
                        points = torch.stack([x_values, other_values], dim=1)
                    else:
                        points = torch.stack([other_values, x_values], dim=1)
                        
                    boundary_points.append(points)
                    
                    # Find matching boundary conditions
                    for bc in self.boundary_conditions:
                        if coord_name in bc.location and abs(bc.location[coord_name] - min_val) < 1e-6:
                            boundary_bcs.extend([bc] * n_points_per_boundary)
                
                # Right boundary (x = max_val)
                if other_coords:
                    other_coord = other_coords[0]
                    other_min, other_max = domain_bounds[other_coord]
                    other_values = torch.linspace(other_min, other_max, n_points_per_boundary)
                    x_values = torch.full_like(other_values, max_val)
                    
                    if coord_name == 'x' and other_coord == 'y':
                        points = torch.stack([x_values, other_values], dim=1)
                    else:
                        points = torch.stack([other_values, x_values], dim=1)
                        
                    boundary_points.append(points)
                    
                    # Find matching boundary conditions
                    for bc in self.boundary_conditions:
                        if coord_name in bc.location and abs(bc.location[coord_name] - max_val) < 1e-6:
                            boundary_bcs.extend([bc] * n_points_per_boundary)
        
        if boundary_points:
            all_boundary_points = torch.cat(boundary_points, dim=0).to(self.device)
            return all_boundary_points, boundary_bcs
        else:
            return torch.empty(0, 2, device=self.device), []
    
    def compute_boundary_loss(self,
                            predictions: torch.Tensor,
                            boundary_coords: torch.Tensor,
                            boundary_conditions: List[BoundaryCondition]) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Args:
            predictions: Network predictions at boundary points [n_boundary, n_outputs]
            boundary_coords: Boundary coordinate points [n_boundary, n_coords]
            boundary_conditions: List of boundary conditions for each point
            
        Returns:
            Boundary loss tensor
        """
        if len(boundary_conditions) == 0:
            return torch.tensor(0.0, device=self.device)
            
        total_loss = torch.tensor(0.0, device=self.device)
        n_conditions = 0
        
        for i, bc in enumerate(boundary_conditions):
            if i >= predictions.shape[0]:
                break
                
            pred_value = predictions[i, bc.variable_index]
            
            if bc.boundary_type == BoundaryType.DIRICHLET:
                # Dirichlet: u = g
                loss = (pred_value - bc.value) ** 2
                total_loss += loss
                n_conditions += 1
                
            elif bc.boundary_type == BoundaryType.NEUMANN:
                # Neumann: ∂u/∂n = g
                # Compute normal derivative
                normal_derivative = self._compute_normal_derivative(
                    predictions[i:i+1, bc.variable_index:bc.variable_index+1],
                    boundary_coords[i:i+1, :],
                    bc.normal_direction
                )
                loss = (normal_derivative - bc.value) ** 2
                total_loss += loss
                n_conditions += 1
                
            elif bc.boundary_type == BoundaryType.ROBIN:
                # Robin: αu + β∂u/∂n = g
                normal_derivative = self._compute_normal_derivative(
                    predictions[i:i+1, bc.variable_index:bc.variable_index+1],
                    boundary_coords[i:i+1, :],
                    bc.normal_direction
                )
                robin_value = bc.alpha * pred_value + bc.beta * normal_derivative
                loss = (robin_value - bc.value) ** 2
                total_loss += loss
                n_conditions += 1
                
            elif bc.boundary_type == BoundaryType.NO_FLOW:
                # No-flow: ∂u/∂n = 0
                normal_derivative = self._compute_normal_derivative(
                    predictions[i:i+1, bc.variable_index:bc.variable_index+1],
                    boundary_coords[i:i+1, :],
                    bc.normal_direction
                )
                loss = normal_derivative ** 2
                total_loss += loss
                n_conditions += 1
        
        return total_loss / max(n_conditions, 1)
    
    def _compute_normal_derivative(self,
                                 output: torch.Tensor,
                                 coords: torch.Tensor,
                                 normal: List[float]) -> torch.Tensor:
        """
        Compute normal derivative ∂u/∂n.
        
        Args:
            output: Network output [1, 1]
            coords: Coordinates [1, n_coords]
            normal: Normal vector components
            
        Returns:
            Normal derivative value
        """
        # Compute gradient
        gradients = torch.autograd.grad(
            outputs=output.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        # Compute dot product with normal vector
        normal_tensor = torch.tensor(normal[:coords.shape[1]], 
                                   device=self.device, dtype=torch.float32)
        normal_derivative = torch.sum(gradients * normal_tensor, dim=1)
        
        return normal_derivative


class InitialConditionHandler:
    """
    Handles enforcement of initial conditions for time-dependent problems.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize initial condition handler.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
        self.initial_conditions: Dict[int, Callable] = {}
        
    def add_initial_condition(self, 
                            variable_index: int,
                            initial_function: Callable[[torch.Tensor], torch.Tensor]):
        """
        Add initial condition for a variable.
        
        Args:
            variable_index: Index of variable (0=pressure, 1=saturation)
            initial_function: Function that takes coordinates and returns initial values
        """
        self.initial_conditions[variable_index] = initial_function
        
    def generate_initial_points(self,
                              domain_bounds: Dict[str, Tuple[float, float]],
                              n_points: int = 1000) -> torch.Tensor:
        """
        Generate initial condition points (at t=0).
        
        Args:
            domain_bounds: Spatial domain bounds
            n_points: Number of initial points to generate
            
        Returns:
            Initial condition points [n_points, n_coords+1] (includes t=0)
        """
        # Generate spatial points
        spatial_coords = []
        for coord_name, (min_val, max_val) in domain_bounds.items():
            coords = torch.rand(n_points, device=self.device) * (max_val - min_val) + min_val
            spatial_coords.append(coords)
            
        spatial_points = torch.stack(spatial_coords, dim=1)
        
        # Add time coordinate (t=0)
        time_coords = torch.zeros(n_points, 1, device=self.device)
        initial_points = torch.cat([spatial_points, time_coords], dim=1)
        
        return initial_points
    
    def compute_initial_loss(self,
                           predictions: torch.Tensor,
                           initial_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss.
        
        Args:
            predictions: Network predictions at initial points [n_initial, n_outputs]
            initial_coords: Initial coordinate points [n_initial, n_coords]
            
        Returns:
            Initial condition loss tensor
        """
        if len(self.initial_conditions) == 0:
            return torch.tensor(0.0, device=self.device)
            
        total_loss = torch.tensor(0.0, device=self.device)
        n_conditions = 0
        
        # Extract spatial coordinates (exclude time)
        spatial_coords = initial_coords[:, :-1]
        
        for var_index, initial_func in self.initial_conditions.items():
            if var_index < predictions.shape[1]:
                # Compute expected initial values
                expected_values = initial_func(spatial_coords)
                predicted_values = predictions[:, var_index]
                
                # Compute loss
                loss = torch.mean((predicted_values - expected_values) ** 2)
                total_loss += loss
                n_conditions += 1
                
        return total_loss / max(n_conditions, 1)


class PhysicalConstraintHandler:
    """
    Handles enforcement of physical constraints (bounds, monotonicity, etc.).
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize physical constraint handler.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
        
    def compute_constraint_loss(self,
                              predictions: torch.Tensor,
                              constraint_type: str = 'bounds') -> torch.Tensor:
        """
        Compute physical constraint violation loss.
        
        Args:
            predictions: Network predictions [batch_size, n_outputs]
            constraint_type: Type of constraint ('bounds', 'monotonicity')
            
        Returns:
            Constraint violation loss
        """
        if constraint_type == 'bounds':
            return self._compute_bounds_loss(predictions)
        elif constraint_type == 'monotonicity':
            return self._compute_monotonicity_loss(predictions)
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_bounds_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Enforce physical bounds on variables.
        
        Pressure should be positive, saturation should be in [0, 1].
        """
        loss = torch.tensor(0.0, device=self.device)
        
        if predictions.shape[1] >= 1:
            # Pressure bounds (should be positive)
            pressure = predictions[:, 0]
            pressure_violation = torch.relu(-pressure)  # Penalty for negative pressure
            loss += torch.mean(pressure_violation ** 2)
            
        if predictions.shape[1] >= 2:
            # Saturation bounds (should be in [0, 1])
            saturation = predictions[:, 1]
            sat_lower_violation = torch.relu(-saturation)  # Penalty for S < 0
            sat_upper_violation = torch.relu(saturation - 1.0)  # Penalty for S > 1
            loss += torch.mean(sat_lower_violation ** 2 + sat_upper_violation ** 2)
            
        return loss
    
    def _compute_monotonicity_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Enforce monotonicity constraints (if applicable).
        
        This is a placeholder for domain-specific monotonicity requirements.
        """
        # Placeholder implementation
        return torch.tensor(0.0, device=self.device)


# Utility functions for common boundary condition setups

def create_reservoir_boundary_conditions(domain_bounds: Dict[str, Tuple[float, float]],
                                       injection_pressure: float = 100.0,
                                       production_pressure: float = 50.0) -> List[BoundaryCondition]:
    """
    Create typical reservoir boundary conditions.
    
    Args:
        domain_bounds: Domain boundaries
        injection_pressure: Injection well pressure
        production_pressure: Production well pressure
        
    Returns:
        List of boundary conditions
    """
    bcs = []
    
    # Left boundary: injection (high pressure)
    if 'x' in domain_bounds:
        x_min, x_max = domain_bounds['x']
        bcs.append(BoundaryCondition(
            BoundaryType.DIRICHLET,
            {'x': x_min},
            injection_pressure,
            variable_index=0  # Pressure
        ))
        
        # Right boundary: production (low pressure)
        bcs.append(BoundaryCondition(
            BoundaryType.DIRICHLET,
            {'x': x_max},
            production_pressure,
            variable_index=0  # Pressure
        ))
        
        # Top and bottom: no-flow
        if 'y' in domain_bounds:
            y_min, y_max = domain_bounds['y']
            bcs.extend([
                BoundaryCondition(
                    BoundaryType.NO_FLOW,
                    {'y': y_min},
                    0.0,
                    variable_index=0,
                    normal_direction=[0.0, -1.0]
                ),
                BoundaryCondition(
                    BoundaryType.NO_FLOW,
                    {'y': y_max},
                    0.0,
                    variable_index=0,
                    normal_direction=[0.0, 1.0]
                )
            ])
    
    return bcs


def create_initial_saturation_distribution(saturation_type: str = 'uniform') -> Callable:
    """
    Create initial saturation distribution function.
    
    Args:
        saturation_type: Type of initial distribution ('uniform', 'linear', 'step')
        
    Returns:
        Function that computes initial saturation values
    """
    def uniform_saturation(coords: torch.Tensor) -> torch.Tensor:
        """Uniform initial saturation."""
        return torch.full((coords.shape[0],), 0.2, device=coords.device)
    
    def linear_saturation(coords: torch.Tensor) -> torch.Tensor:
        """Linear initial saturation profile."""
        x_coords = coords[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()
        normalized_x = (x_coords - x_min) / (x_max - x_min + 1e-8)
        return 0.1 + 0.3 * normalized_x  # Saturation from 0.1 to 0.4
    
    def step_saturation(coords: torch.Tensor) -> torch.Tensor:
        """Step function initial saturation."""
        x_coords = coords[:, 0]
        x_mid = (x_coords.min() + x_coords.max()) / 2
        return torch.where(x_coords < x_mid, 0.1, 0.3)
    
    if saturation_type == 'uniform':
        return uniform_saturation
    elif saturation_type == 'linear':
        return linear_saturation
    elif saturation_type == 'step':
        return step_saturation
    else:
        return uniform_saturation