# PINN Tutorial API Documentation

## Overview

This documentation provides a comprehensive reference for the Physics-Informed Neural Networks (PINNs) tutorial codebase. The API is designed to be modular, extensible, and production-ready for reservoir modeling applications.

## Table of Contents

1. [Core Data Models](#core-data-models)
2. [Data Processing](#data-processing)
3. [Model Architecture](#model-architecture)
4. [Physics Engine](#physics-engine)
5. [Training Components](#training-components)
6. [Validation Framework](#validation-framework)
7. [Visualization Tools](#visualization-tools)
8. [Configuration](#configuration)

---

## Core Data Models

### WellData

```python
@dataclass
class WellData:
    """Container for well log data and metadata"""
    
    well_id: str
    depth: np.ndarray
    curves: Dict[str, np.ndarray]
    metadata: Optional[WellMetadata] = None
```

**Attributes:**
- `well_id`: Unique identifier for the well
- `depth`: Depth measurements in feet
- `curves`: Dictionary mapping curve names to measurement arrays
- `metadata`: Optional well metadata (location, formation, etc.)

### WellMetadata

```python
@dataclass
class WellMetadata:
    """Metadata for well information"""
    
    location: Tuple[float, float]  # (latitude, longitude)
    formation: str
    date_logged: datetime
    curve_units: Dict[str, str]
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    """Configuration for PINN training"""
    
    batch_size: int = 256
    learning_rate: float = 1e-3
    num_epochs: int = 1000
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'data': 1.0, 'physics': 0.1, 'boundary': 0.5
    })
    optimizer_switch_epoch: int = 500
```

---

## Data Processing

### LASFileReader

```python
class LASFileReader:
    """Reader for LAS (Log ASCII Standard) files"""
    
    def read_las_file(self, filepath: str) -> WellData:
        """
        Read and parse a LAS file
        
        Args:
            filepath: Path to the LAS file
            
        Returns:
            WellData object containing parsed data
            
        Raises:
            LASParsingError: If file cannot be parsed
        """
    
    def extract_curves(self, well_data: WellData) -> Dict[str, np.ndarray]:
        """
        Extract specific curves from well data
        
        Args:
            well_data: WellData object
            
        Returns:
            Dictionary of curve name to data array mappings
        """
    
    def get_well_metadata(self, well_data: WellData) -> WellMetadata:
        """Extract metadata from well data"""
```

### DataPreprocessor

```python
class DataPreprocessor:
    """Preprocessing utilities for well log data"""
    
    def clean_data(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Clean raw curve data by removing outliers and invalid values
        
        Args:
            curves: Raw curve data
            
        Returns:
            Cleaned curve data
        """
    
    def normalize_curves(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize curves to zero mean and unit variance
        
        Args:
            curves: Input curve data
            
        Returns:
            Normalized curve data
        """
    
    def handle_missing_values(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Handle missing values using interpolation and domain knowledge
        
        Args:
            curves: Curve data with potential missing values
            
        Returns:
            Curve data with missing values handled
        """
```

### DatasetBuilder

```python
class DatasetBuilder:
    """Build training/validation datasets from processed well data"""
    
    def create_datasets(self, processed_wells: List[Dict], 
                       config: Dict) -> Dict[str, Dict]:
        """
        Create train/validation/test datasets from processed wells
        
        Args:
            processed_wells: List of processed well dictionaries
            config: Dataset configuration
            
        Returns:
            Dictionary containing train/val/test splits
        """
    
    def split_by_wells(self, wells: List[Dict], 
                      train_ratio: float = 0.7) -> Tuple[List, List, List]:
        """Split wells into train/validation/test sets"""
```

---

## Model Architecture

### PINNArchitecture

```python
class PINNArchitecture(nn.Module):
    """Physics-Informed Neural Network architecture"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, activation: str = 'tanh'):
        """
        Initialize PINN architecture
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output predictions
            activation: Activation function ('tanh', 'swish', 'relu')
        """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
    
    def compute_derivatives(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute spatial derivatives for physics constraints
        
        Args:
            x: Input tensor with requires_grad=True
            
        Returns:
            Dictionary of derivative tensors
        """
```

### TensorManager

```python
class TensorManager:
    """Utilities for tensor operations and gradient computation"""
    
    def __init__(self, device: torch.device):
        """Initialize tensor manager with device"""
    
    def prepare_input_tensors(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy arrays to PyTorch tensors on correct device"""
    
    def enable_gradients(self, tensor: torch.Tensor) -> torch.Tensor:
        """Enable gradient computation for tensor"""
    
    def compute_spatial_derivatives(self, output: torch.Tensor, 
                                  input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute spatial derivatives using automatic differentiation"""
```

### ModelInterface

```python
class ModelInterface:
    """High-level interface for model operations"""
    
    def __init__(self, model: nn.Module, tensor_manager: TensorManager):
        """Initialize model interface"""
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions on input data"""
    
    def save_model(self, filepath: str) -> None:
        """Save model state to file"""
    
    def load_model(self, filepath: str) -> None:
        """Load model state from file"""
```

---

## Physics Engine

### PDEFormulator

```python
class PDEFormulator:
    """Formulate partial differential equations for reservoir physics"""
    
    def darcy_residual(self, pressure: torch.Tensor, coords: torch.Tensor,
                      permeability: torch.Tensor) -> torch.Tensor:
        """
        Compute Darcy's law residual
        
        Args:
            pressure: Pressure field predictions
            coords: Spatial coordinates
            permeability: Permeability field
            
        Returns:
            PDE residual tensor
        """
    
    def buckley_leverett_residual(self, saturation: torch.Tensor,
                                 coords: torch.Tensor) -> torch.Tensor:
        """Compute Buckley-Leverett equation residual"""
    
    def continuity_residual(self, velocity: torch.Tensor,
                           coords: torch.Tensor) -> torch.Tensor:
        """Compute continuity equation residual"""
```

### BoundaryConditionHandler

```python
class BoundaryConditionHandler:
    """Handle various types of boundary conditions"""
    
    def dirichlet_bc(self, predictions: torch.Tensor,
                    boundary_values: torch.Tensor) -> torch.Tensor:
        """
        Compute Dirichlet boundary condition residual
        
        Args:
            predictions: Model predictions at boundary points
            boundary_values: Known boundary values
            
        Returns:
            Boundary condition residual
        """
    
    def neumann_bc(self, predictions: torch.Tensor, coords: torch.Tensor,
                  boundary_gradients: torch.Tensor) -> torch.Tensor:
        """Compute Neumann boundary condition residual"""
    
    def robin_bc(self, predictions: torch.Tensor, coords: torch.Tensor,
                alpha: float, beta: float, gamma: float) -> torch.Tensor:
        """Compute Robin boundary condition residual"""
```

### PhysicsLossCalculator

```python
class PhysicsLossCalculator:
    """Calculate physics-informed loss components"""
    
    def __init__(self, pde_formulator: PDEFormulator,
                 boundary_handler: BoundaryConditionHandler):
        """Initialize with PDE formulator and boundary handler"""
    
    def compute_pde_loss(self, predictions: torch.Tensor,
                        inputs: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual loss"""
    
    def compute_boundary_loss(self, predictions: torch.Tensor,
                             boundary_data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute boundary condition loss"""
    
    def compute_total_physics_loss(self, predictions: torch.Tensor,
                                  inputs: torch.Tensor,
                                  boundary_data: Optional[Tuple] = None) -> torch.Tensor:
        """Compute total physics loss combining PDE and boundary terms"""
```

---

## Training Components

### PINNTrainer

```python
class PINNTrainer:
    """Main training orchestrator for PINN models"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize trainer with model and device"""
    
    def train(self, dataset: Dict, config: TrainingConfig) -> Dict:
        """
        Train PINN model with given dataset and configuration
        
        Args:
            dataset: Training/validation data
            config: Training configuration
            
        Returns:
            Training history and metrics
        """
    
    def validate(self, model: nn.Module, validation_data: Dict) -> Dict:
        """Validate model performance on held-out data"""
```

### OptimizerManager

```python
class OptimizerManager:
    """Manage different optimization phases"""
    
    def setup_adam_phase(self, model: nn.Module, lr: float) -> optim.Adam:
        """Set up Adam optimizer for initial training phase"""
    
    def setup_lbfgs_phase(self, model: nn.Module) -> optim.LBFGS:
        """Set up L-BFGS optimizer for refinement phase"""
    
    def switch_optimizer(self, current_epoch: int, switch_epoch: int) -> bool:
        """Determine if optimizer should be switched"""
```

### BatchProcessor

```python
class BatchProcessor:
    """Handle batch processing for training data"""
    
    def __init__(self, device: torch.device):
        """Initialize batch processor"""
    
    def create_batches(self, data: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
        """Create batches from input data"""
    
    def sample_physics_points(self, data: torch.Tensor, n_points: int) -> torch.Tensor:
        """Sample points for physics loss computation"""
```

### ConvergenceMonitor

```python
class ConvergenceMonitor:
    """Monitor training convergence and implement early stopping"""
    
    def __init__(self, patience: int = 100, min_delta: float = 1e-6):
        """
        Initialize convergence monitor
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
    
    def should_stop(self, current_loss: float) -> bool:
        """Check if training should stop based on convergence criteria"""
    
    def reset(self) -> None:
        """Reset convergence monitor state"""
```

---

## Validation Framework

### ValidationFramework

```python
class ValidationFramework:
    """Comprehensive validation framework for PINN models"""
    
    def holdout_validation(self, model: nn.Module, test_wells: List[Dict]) -> Dict:
        """Perform hold-out validation on reserved wells"""
    
    def cross_validation(self, model_class: type, wells: List[Dict], 
                        k_folds: int = 5) -> Dict:
        """Perform k-fold cross-validation across wells"""
    
    def compute_metrics(self, predictions: np.ndarray, 
                       targets: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive performance metrics"""
```

### PDEResidualAnalyzer

```python
class PDEResidualAnalyzer:
    """Analyze PDE residuals for physics compliance"""
    
    def compute_residual_distribution(self, model: nn.Module,
                                    test_points: torch.Tensor) -> Dict:
        """Compute PDE residual statistics on test points"""
    
    def analyze_physics_violations(self, residuals: torch.Tensor) -> Dict:
        """Analyze physics constraint violations"""
```

### PredictionComparator

```python
class PredictionComparator:
    """Compare model predictions with ground truth"""
    
    def generate_comparison_plots(self, predictions: np.ndarray,
                                 targets: np.ndarray, features: np.ndarray) -> None:
        """Generate comprehensive comparison visualizations"""
    
    def compute_error_statistics(self, predictions: np.ndarray,
                                targets: np.ndarray) -> Dict:
        """Compute detailed error statistics"""
```

---

## Visualization Tools

### ScientificPlotter

```python
class ScientificPlotter:
    """Create publication-quality scientific plots"""
    
    def plot_data_distribution(self, data: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot data distribution histograms and statistics"""
    
    def plot_well_profiles(self, depth: np.ndarray, 
                          curves: Dict[str, np.ndarray]) -> plt.Figure:
        """Create well log profile plots"""
    
    def plot_training_curves(self, losses: Dict[str, List[float]]) -> plt.Figure:
        """Plot training loss evolution"""
```

### TrainingVisualizer

```python
class TrainingVisualizer:
    """Visualize training progress and metrics"""
    
    def plot_loss_evolution(self, history: Dict) -> None:
        """Plot training and validation loss evolution"""
    
    def plot_learning_rate_schedule(self, lr_history: List[float]) -> None:
        """Plot learning rate schedule"""
    
    def plot_gradient_norms(self, gradient_norms: List[float]) -> None:
        """Plot gradient norm evolution"""
```

### ResultsAnalyzer

```python
class ResultsAnalyzer:
    """Analyze and visualize final results"""
    
    def create_prediction_comparison(self, predictions: np.ndarray,
                                   targets: np.ndarray) -> plt.Figure:
        """Create prediction vs. target comparison plots"""
    
    def create_error_analysis(self, errors: np.ndarray, 
                             features: np.ndarray) -> plt.Figure:
        """Create error analysis visualizations"""
    
    def create_physics_compliance_plot(self, residuals: Dict) -> plt.Figure:
        """Create physics compliance visualization"""
```

### DiagramGenerator

```python
class DiagramGenerator:
    """Generate conceptual diagrams and flowcharts"""
    
    def create_architecture_diagram(self, model_config: Dict) -> plt.Figure:
        """Create neural network architecture diagram"""
    
    def create_physics_flowchart(self, equations: List[str]) -> plt.Figure:
        """Create physics equation flowchart"""
    
    def create_data_flow_diagram(self, pipeline_steps: List[str]) -> plt.Figure:
        """Create data processing pipeline diagram"""
```

---

## Configuration

### Default Configurations

```python
# Model configuration
MODEL_CONFIG = {
    'input_dim': 4,
    'hidden_dims': [64, 64, 64],
    'output_dim': 2,
    'activation': 'tanh',
    'initialization': 'xavier_normal'
}

# Training configuration
TRAINING_CONFIG = {
    'phase1': {
        'optimizer': 'adam',
        'lr': 1e-3,
        'epochs': 2000,
        'batch_size': 256
    },
    'phase2': {
        'optimizer': 'lbfgs',
        'lr': 1.0,
        'max_iter': 1000
    }
}

# Loss configuration
LOSS_CONFIG = {
    'weights': {
        'data': 1.0,
        'physics': 0.1,
        'boundary': 0.5
    },
    'adaptive_weighting': True
}
```

---

## Usage Examples

### Basic PINN Training

```python
from src.models.pinn_architecture import PINNArchitecture
from src.training.pinn_trainer import PINNTrainer
from src.data.dataset_builder import DatasetBuilder

# Load and prepare data
dataset_builder = DatasetBuilder()
datasets = dataset_builder.create_datasets(processed_wells, config)

# Create model
model = PINNArchitecture(
    input_dim=4,
    hidden_dims=[64, 64, 64],
    output_dim=2
)

# Train model
trainer = PINNTrainer(model, device)
results = trainer.train(datasets, training_config)
```

### Custom Physics Implementation

```python
from src.physics.pde_formulator import PDEFormulator

class CustomPDEFormulator(PDEFormulator):
    def custom_physics_residual(self, predictions, inputs):
        # Implement custom physics constraints
        return residual

# Use custom formulator
pde_formulator = CustomPDEFormulator()
physics_calculator = PhysicsLossCalculator(pde_formulator, boundary_handler)
```

---

## Error Handling

### Common Exceptions

```python
class LASParsingError(Exception):
    """Raised when LAS file cannot be parsed"""

class PhysicsViolationError(Exception):
    """Raised when physics constraints are severely violated"""

class ConvergenceError(Exception):
    """Raised when training fails to converge"""

class ValidationError(Exception):
    """Raised when validation fails"""
```

### Error Recovery

```python
try:
    well_data = las_reader.read_las_file(filepath)
except LASParsingError as e:
    logger.warning(f"Failed to parse {filepath}: {e}")
    # Implement fallback or skip file
```

---

## Performance Considerations

### Memory Management
- Use gradient checkpointing for large models
- Implement batch processing for large datasets
- Clear intermediate gradients when not needed

### Computational Efficiency
- Leverage GPU acceleration when available
- Use mixed precision training for speed
- Implement efficient physics loss computation

### Scalability
- Support distributed training for large problems
- Implement data streaming for massive datasets
- Use model parallelism for very large networks

---

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Include comprehensive docstrings
- Write unit tests for all components

### Testing
```python
pytest tests/  # Run all tests
pytest tests/test_models/  # Run model tests
pytest tests/test_physics/  # Run physics tests
```

### Documentation
- Update API documentation for new features
- Include usage examples
- Maintain changelog for version updates

---

This API documentation provides a comprehensive reference for the PINN tutorial codebase. For additional examples and tutorials, see the Jupyter notebooks in the `tutorials/` directory.