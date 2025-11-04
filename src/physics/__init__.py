"""
Physics module for Physics-Informed Neural Networks.

This module provides the core physics components for PINN training:
- PDE formulation and residual computation
- Boundary and initial condition handling
- Composite physics loss calculation with adaptive weighting
"""

from .pde_formulator import PDEFormulator, PDEResidualCalculator
from .boundary_conditions import (
    BoundaryConditionHandler,
    InitialConditionHandler, 
    PhysicalConstraintHandler,
    BoundaryCondition,
    BoundaryType,
    create_reservoir_boundary_conditions,
    create_initial_saturation_distribution
)
from .physics_loss import (
    PhysicsLossCalculator,
    LossWeights,
    LossComponents,
    AdaptiveLossWeighting,
    LossMonitor,
    create_standard_physics_loss
)

__all__ = [
    # PDE formulation
    'PDEFormulator',
    'PDEResidualCalculator',
    
    # Boundary conditions
    'BoundaryConditionHandler',
    'InitialConditionHandler',
    'PhysicalConstraintHandler',
    'BoundaryCondition',
    'BoundaryType',
    'create_reservoir_boundary_conditions',
    'create_initial_saturation_distribution',
    
    # Physics loss
    'PhysicsLossCalculator',
    'LossWeights',
    'LossComponents', 
    'AdaptiveLossWeighting',
    'LossMonitor',
    'create_standard_physics_loss'
]