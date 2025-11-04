"""
Base interfaces for the PINN tutorial system components.

Defines abstract base classes that establish contracts for different system components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

from .data_models import (
    WellData, 
    TrainingConfig, 
    ValidationMetrics, 
    TrainingResults,
    BatchData,
    ModelConfig
)


class DataProcessorInterface(ABC):
    """Abstract interface for data processing components."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data and return processed output."""
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """Validate that data meets processing requirements."""
        pass


class LASReaderInterface(DataProcessorInterface):
    """Interface for LAS file reading and parsing."""
    
    @abstractmethod
    def read_las_file(self, filepath: str) -> WellData:
        """Read and parse a single LAS file."""
        pass
    
    @abstractmethod
    def extract_curves(self, well_data: WellData) -> Dict[str, np.ndarray]:
        """Extract specific curves from well data."""
        pass
    
    @abstractmethod
    def get_well_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from LAS file."""
        pass


class PreprocessorInterface(DataProcessorInterface):
    """Interface for data preprocessing operations."""
    
    @abstractmethod
    def clean_data(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clean data by removing invalid values and outliers."""
        pass
    
    @abstractmethod
    def normalize_curves(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize curve values to standard ranges."""
        pass
    
    @abstractmethod
    def handle_missing_values(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Handle missing values through interpolation or imputation."""
        pass


class ModelInterface(ABC):
    """Abstract interface for PINN model components."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def compute_derivatives(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute spatial and temporal derivatives."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save model state to file."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load model state from file."""
        pass


class PhysicsEngineInterface(ABC):
    """Interface for physics-informed loss computation."""
    
    @abstractmethod
    def compute_pde_residual(self, predictions: torch.Tensor, 
                           inputs: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual for physics loss."""
        pass
    
    @abstractmethod
    def compute_boundary_loss(self, predictions: torch.Tensor,
                            boundary_data: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss."""
        pass
    
    @abstractmethod
    def compute_physics_loss(self, predictions: torch.Tensor,
                           inputs: torch.Tensor,
                           boundary_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute complete physics-informed loss."""
        pass


class TrainerInterface(ABC):
    """Abstract interface for training components."""
    
    @abstractmethod
    def train(self, model: ModelInterface, 
              dataset: Any, 
              config: TrainingConfig) -> TrainingResults:
        """Train the model with given dataset and configuration."""
        pass
    
    @abstractmethod
    def validate(self, model: ModelInterface, 
                validation_data: Any) -> ValidationMetrics:
        """Validate model performance on validation dataset."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, filepath: str, epoch: int, 
                       model_state: Dict[str, Any]) -> None:
        """Save training checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, filepath: str) -> Tuple[int, Dict[str, Any]]:
        """Load training checkpoint."""
        pass


class OptimizerInterface(ABC):
    """Interface for optimizer management."""
    
    @abstractmethod
    def setup_optimizer(self, model: torch.nn.Module, 
                       optimizer_type: str, 
                       **kwargs) -> torch.optim.Optimizer:
        """Set up optimizer for training."""
        pass
    
    @abstractmethod
    def update_learning_rate(self, optimizer: torch.optim.Optimizer, 
                           new_lr: float) -> None:
        """Update optimizer learning rate."""
        pass
    
    @abstractmethod
    def switch_optimizer(self, model: torch.nn.Module,
                        current_optimizer: torch.optim.Optimizer,
                        new_optimizer_type: str) -> torch.optim.Optimizer:
        """Switch to different optimizer during training."""
        pass


class VisualizerInterface(ABC):
    """Abstract interface for visualization components."""
    
    @abstractmethod
    def create_plot(self, data: Any, plot_type: str, **kwargs) -> plt.Figure:
        """Create a plot of specified type with given data."""
        pass
    
    @abstractmethod
    def save_plot(self, figure: plt.Figure, filepath: str, **kwargs) -> None:
        """Save plot to file."""
        pass


class ScientificPlotterInterface(VisualizerInterface):
    """Interface for scientific plotting."""
    
    @abstractmethod
    def plot_data_distribution(self, data: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot data distributions (histograms, boxplots)."""
        pass
    
    @abstractmethod
    def plot_well_profiles(self, depth: np.ndarray, 
                          curves: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot well log profiles with depth."""
        pass
    
    @abstractmethod
    def plot_training_curves(self, losses: Dict[str, List[float]]) -> plt.Figure:
        """Plot training loss curves."""
        pass


class DiagramGeneratorInterface(VisualizerInterface):
    """Interface for generating conceptual diagrams."""
    
    @abstractmethod
    def create_architecture_diagram(self, model_config: ModelConfig) -> plt.Figure:
        """Create neural network architecture diagram."""
        pass
    
    @abstractmethod
    def create_physics_flowchart(self, equations: List[str]) -> plt.Figure:
        """Create flowchart showing physics equations."""
        pass
    
    @abstractmethod
    def create_data_flow_diagram(self, components: List[str]) -> plt.Figure:
        """Create data flow diagram for PINN training process."""
        pass


class ValidationInterface(ABC):
    """Interface for model validation and benchmarking."""
    
    @abstractmethod
    def cross_validate(self, model: ModelInterface,
                      dataset: Any,
                      k_folds: int = 5) -> List[ValidationMetrics]:
        """Perform k-fold cross validation."""
        pass
    
    @abstractmethod
    def holdout_validate(self, model: ModelInterface,
                        train_data: Any,
                        test_data: Any) -> ValidationMetrics:
        """Perform hold-out validation."""
        pass
    
    @abstractmethod
    def compute_pde_residuals(self, model: ModelInterface,
                            test_points: torch.Tensor) -> np.ndarray:
        """Compute PDE residuals on test points."""
        pass


class DatasetInterface(ABC):
    """Interface for dataset management."""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> BatchData:
        """Get item at index."""
        pass
    
    @abstractmethod
    def split(self, train_ratio: float) -> Tuple['DatasetInterface', 'DatasetInterface']:
        """Split dataset into train and validation sets."""
        pass
    
    @abstractmethod
    def get_batch(self, batch_size: int) -> BatchData:
        """Get random batch of specified size."""
        pass