"""
Core data models for the PINN tutorial system.

Defines data structures for well data, training configurations, and validation metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class WellMetadata:
    """Metadata for a well log."""
    well_id: str
    location: Tuple[float, float]  # (latitude, longitude)
    formation: str
    date_logged: Optional[datetime]
    curve_units: Dict[str, str]
    total_depth: float
    surface_elevation: Optional[float] = None
    operator: Optional[str] = None
    field_name: Optional[str] = None


@dataclass
class WellData:
    """Complete well log data structure."""
    well_id: str
    depth: np.ndarray
    curves: Dict[str, np.ndarray]  # curve_name -> values
    metadata: WellMetadata
    
    def __post_init__(self):
        """Validate data consistency."""
        if len(self.depth) == 0:
            raise ValueError("Depth array cannot be empty")
        
        for curve_name, values in self.curves.items():
            if len(values) != len(self.depth):
                raise ValueError(f"Curve {curve_name} length {len(values)} "
                               f"does not match depth length {len(self.depth)}")


@dataclass
class TrainingConfig:
    """Configuration for PINN training."""
    # Model architecture
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: str = "tanh"
    
    # Training parameters
    batch_size: int = 1024
    learning_rate: float = 1e-3
    num_epochs: int = 5000
    
    # Optimization strategy
    optimizer_switch_epoch: int = 3000
    lbfgs_iterations: int = 1000
    
    # Loss weighting
    loss_weights: Dict[str, float] = None
    
    # Physics parameters
    enable_physics_loss: bool = True
    pde_weight: float = 1.0
    boundary_weight: float = 1.0
    data_weight: float = 1.0
    
    # Validation
    validation_split: float = 0.2
    validation_frequency: int = 100
    
    def __post_init__(self):
        """Set default loss weights if not provided."""
        if self.loss_weights is None:
            self.loss_weights = {
                'data': self.data_weight,
                'pde': self.pde_weight,
                'boundary': self.boundary_weight
            }


@dataclass
class ValidationMetrics:
    """Metrics for model validation."""
    # Error metrics
    l2_error: float
    mean_absolute_error: float
    root_mean_square_error: float
    
    # Physics-informed metrics
    mean_pde_residual: float
    max_pde_residual: float
    pde_residual_std: float
    
    # Per-output metrics
    pressure_mae: float
    pressure_rmse: float
    saturation_mae: float
    saturation_rmse: float
    
    # Additional metrics
    r2_score: float
    relative_error: float
    
    # Training info
    epoch: int
    training_time: float


@dataclass
class ModelConfig:
    """Configuration for PINN model architecture."""
    input_features: List[str]
    output_features: List[str]
    hidden_layers: List[int]
    activation_function: str
    dropout_rate: float = 0.0
    batch_normalization: bool = False
    weight_initialization: str = "xavier_uniform"
    
    # Input/output scaling
    input_scaling: Dict[str, Tuple[float, float]] = None  # feature -> (min, max)
    output_scaling: Dict[str, Tuple[float, float]] = None  # feature -> (min, max)


@dataclass
class BatchData:
    """Data structure for training batches."""
    inputs: np.ndarray
    targets: Optional[np.ndarray] = None
    coordinates: Optional[np.ndarray] = None
    well_ids: Optional[List[str]] = None
    
    def __len__(self) -> int:
        return len(self.inputs)


@dataclass
class TrainingResults:
    """Results from PINN training."""
    # Loss history
    total_loss_history: List[float]
    data_loss_history: List[float]
    pde_loss_history: List[float]
    boundary_loss_history: List[float]
    
    # Validation history
    validation_metrics_history: List[ValidationMetrics]
    
    # Final metrics
    final_metrics: ValidationMetrics
    
    # Training info
    total_epochs: int
    training_time: float
    convergence_epoch: Optional[int] = None
    
    # Model state
    best_model_state: Optional[Dict[str, Any]] = None


@dataclass
class PhysicsParameters:
    """Physical parameters for reservoir modeling."""
    # Fluid properties
    oil_viscosity: float = 1.0  # cp
    water_viscosity: float = 1.0  # cp
    oil_density: float = 0.8  # g/cm³
    water_density: float = 1.0  # g/cm³
    
    # Rock properties
    porosity_range: Tuple[float, float] = (0.05, 0.35)
    permeability_range: Tuple[float, float] = (0.1, 1000.0)  # mD
    
    # Boundary conditions
    pressure_boundaries: Dict[str, float] = None
    saturation_boundaries: Dict[str, float] = None
    
    def __post_init__(self):
        """Set default boundary conditions."""
        if self.pressure_boundaries is None:
            self.pressure_boundaries = {
                'inlet': 3000.0,  # psi
                'outlet': 1000.0  # psi
            }
        
        if self.saturation_boundaries is None:
            self.saturation_boundaries = {
                'inlet': 0.0,  # water saturation
                'outlet': 1.0   # water saturation
            }