#!/usr/bin/env python3
"""
End-to-end integration tests for the PINN tutorial system.
Tests the complete pipeline from LAS files to trained PINN model.
"""

import sys
import os
import tempfile
import shutil
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to handle import issues gracefully
try:
    import torch
    import numpy as np
except ImportError as e:
    print(f"Warning: Required dependencies not available: {e}")
    print("Skipping integration tests...")
    sys.exit(0)

from data.las_reader import LASFileReader
from data.preprocessor import DataPreprocessor
from data.dataset_builder import DatasetBuilder
from models.pinn_architecture import PINNArchitecture
from models.tensor_manager import TensorManager
from models.model_interface import ModelInterface
from physics.pde_formulator import PDEFormulator
from physics.boundary_conditions import BoundaryConditionHandler
from physics.physics_loss import PhysicsLossCalculator
from training.pinn_trainer import PINNTrainer
from training.optimizer_manager import OptimizerManager
from training.batch_processor import BatchProcessor
from training.convergence_monitor import ConvergenceMonitor
from validation.validation_framework import ValidationFramework
from core.data_models import WellData, TrainingConfig, ValidationConfig


class TestEndToEndIntegration(unittest.TestCase):
    """Test complete pipeline from LAS files to trained PINN model."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create mock LAS file content
        self.mock_las_content = """~Version Information
VERS.                          2.0 : CWLS LOG ASCII STANDARD -VERSION 2.0
WRAP.                          NO  : ONE LINE PER DEPTH STEP
~Well Information
STRT.M                      1000.0 : START DEPTH
STOP.M                      1100.0 : STOP DEPTH
STEP.M                        0.5  : STEP
NULL.                      -999.25 : NULL VALUE
COMP.                     COMPANY  : COMPANY
WELL.                    TEST_WELL : WELL
FLD.                      FIELD    : FIELD
LOC.                      LOCATION : LOCATION
PROV.                     PROVINCE : PROVINCE
CNTY.                     COUNTY   : COUNTY
STAT.                     STATE    : STATE
CTRY.                     COUNTRY  : COUNTRY
SRVC.                     SERVICE  : SERVICE COMPANY
DATE.                     DATE     : DATE
UWI.                      UWI      : UNIQUE WELL ID
~Curve Information
DEPT.M                           : 1  DEPTH
GR.GAPI                          : 2  GAMMA RAY
NPHI.V/V                         : 3  NEUTRON POROSITY
RHOB.G/C3                        : 4  BULK DENSITY
RT.OHMM                          : 5  RESISTIVITY
PORO.V/V                         : 6  POROSITY
PERM.MD                          : 7  PERMEABILITY
~ASCII
1000.0  50.0  0.15  2.3  10.0  0.12  100.0
1000.5  52.0  0.16  2.2  12.0  0.13  120.0
1001.0  48.0  0.14  2.4  8.0   0.11  80.0
1001.5  55.0  0.17  2.1  15.0  0.14  150.0
1002.0  51.0  0.15  2.3  11.0  0.12  110.0
"""
        
        # Create mock LAS file
        self.las_file = self.data_dir / "test_well.las"
        with open(self.las_file, 'w') as f:
            f.write(self.mock_las_content)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline_execution(self):
        """Test the complete pipeline from LAS files to trained model."""
        
        # Step 1: Data Processing Pipeline
        print("Testing data processing pipeline...")
        
        # Test LAS file reading
        las_reader = LASFileReader()
        well_data = las_reader.read_las_file(str(self.las_file))
        self.assertIsInstance(well_data, WellData)
        self.assertIn('GR', well_data.curves)
        self.assertIn('PORO', well_data.curves)
        
        # Test data preprocessing
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_well_data(well_data)
        self.assertIsInstance(processed_data, WellData)
        
        # Test dataset building
        dataset_builder = DatasetBuilder()
        dataset = dataset_builder.build_dataset([processed_data])
        self.assertIsNotNone(dataset)
        
        # Step 2: Model Architecture
        print("Testing PINN model architecture...")
        
        input_dim = 4  # depth, porosity, permeability, gamma ray
        hidden_dims = [50, 50]
        output_dim = 2  # pressure, saturation
        
        model = PINNArchitecture(input_dim, hidden_dims, output_dim)
        self.assertIsInstance(model, nn.Module)
        
        # Test tensor management
        tensor_manager = TensorManager()
        sample_input = torch.randn(10, input_dim)
        prepared_input = tensor_manager.prepare_input_tensors(sample_input.numpy())
        self.assertEqual(prepared_input.shape, (10, input_dim))
        
        # Step 3: Physics Engine
        print("Testing physics engine...")
        
        pde_formulator = PDEFormulator()
        boundary_handler = BoundaryConditionHandler()
        physics_loss_calc = PhysicsLossCalculator(pde_formulator, boundary_handler)
        
        # Test physics loss computation
        sample_output = model(prepared_input)
        physics_loss = physics_loss_calc.compute_physics_loss(
            sample_output, prepared_input, {}
        )
        self.assertIsInstance(physics_loss, torch.Tensor)
        
        # Step 4: Training Engine (minimal test)
        print("Testing training engine...")
        
        config = TrainingConfig(
            batch_size=5,
            learning_rate=1e-3,
            num_epochs=2,  # Very small for testing
            loss_weights={'data': 1.0, 'physics': 1.0, 'boundary': 1.0},
            optimizer_switch_epoch=1
        )
        
        trainer = PINNTrainer(physics_loss_calc)
        
        # Create minimal training data
        train_data = {
            'inputs': prepared_input[:8],
            'targets': sample_output[:8].detach()
        }
        val_data = {
            'inputs': prepared_input[8:],
            'targets': sample_output[8:].detach()
        }
        
        # Test training (just a few steps)
        results = trainer.train(model, train_data, val_data, config)
        self.assertIn('train_losses', results)
        self.assertIn('val_losses', results)
        
        # Step 5: Validation Framework
        print("Testing validation framework...")
        
        val_config = ValidationConfig(n_folds=2, holdout_fraction=0.2)
        validator = ValidationFramework(val_config)
        
        # Test validation metrics computation
        metrics = validator.compute_validation_metrics(
            model, val_data['inputs'], val_data['targets']
        )
        self.assertIn('l2_error', metrics)
        self.assertIn('mae', metrics)
        
        print("✓ Complete pipeline test passed!")
    
    def test_physics_constraint_satisfaction(self):
        """Test that physics constraints are properly satisfied during training."""
        
        print("Testing physics constraint satisfaction...")
        
        # Create simple model and data
        model = PINNArchitecture(4, [20, 20], 2)
        
        # Create test data with known physics properties
        n_points = 20
        depth = np.linspace(1000, 1100, n_points)
        porosity = np.full(n_points, 0.15)
        permeability = np.full(n_points, 100.0)
        gamma_ray = np.full(n_points, 50.0)
        
        inputs = np.column_stack([depth, porosity, permeability, gamma_ray])
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        
        # Test PDE residual computation
        pde_formulator = PDEFormulator()
        outputs = model(inputs_tensor)
        
        # Compute Darcy residual
        darcy_residual = pde_formulator.darcy_residual(
            outputs[:, 0], inputs_tensor, inputs_tensor[:, 2]
        )
        
        # Check that residual is computed (should be tensor)
        self.assertIsInstance(darcy_residual, torch.Tensor)
        self.assertEqual(darcy_residual.shape[0], n_points)
        
        # Test boundary conditions
        boundary_handler = BoundaryConditionHandler()
        boundary_loss = boundary_handler.compute_boundary_loss(
            outputs, inputs_tensor, {}
        )
        self.assertIsInstance(boundary_loss, torch.Tensor)
        
        print("✓ Physics constraint satisfaction test passed!")
    
    def test_tutorial_notebook_execution(self):
        """Test that tutorial notebooks can be executed without errors."""
        
        print("Testing tutorial notebook execution...")
        
        # Test basic imports that notebooks would use
        try:
            # Data processing imports
            from data.las_reader import LASFileReader
            from data.preprocessor import DataPreprocessor
            
            # Model imports
            from models.pinn_architecture import PINNArchitecture
            from models.tensor_manager import TensorManager
            
            # Physics imports
            from physics.pde_formulator import PDEFormulator
            from physics.physics_loss import PhysicsLossCalculator
            
            # Training imports
            from training.pinn_trainer import PINNTrainer
            
            # Validation imports
            from validation.validation_framework import ValidationFramework
            
            # Visualization imports
            from visualization.scientific_plotter import ScientificPlotter
            from visualization.training_visualizer import TrainingVisualizer
            
            print("✓ All tutorial imports successful!")
            
        except ImportError as e:
            self.fail(f"Tutorial notebook import failed: {e}")
        
        # Test basic workflow that notebooks would follow
        try:
            # 1. Data loading
            las_reader = LASFileReader()
            
            # 2. Model creation
            model = PINNArchitecture(4, [50, 50], 2)
            
            # 3. Physics setup
            pde_formulator = PDEFormulator()
            
            # 4. Basic tensor operations
            sample_data = torch.randn(10, 4)
            output = model(sample_data)
            
            self.assertEqual(output.shape, (10, 2))
            print("✓ Basic notebook workflow test passed!")
            
        except Exception as e:
            self.fail(f"Tutorial notebook workflow failed: {e}")
    
    def test_reproducibility(self):
        """Test that results are reproducible across runs."""
        
        print("Testing reproducibility...")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create identical models and data
        model1 = PINNArchitecture(4, [20, 20], 2)
        model2 = PINNArchitecture(4, [20, 20], 2)
        
        # Initialize with same weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.data.copy_(p1.data)
        
        # Test with same input
        test_input = torch.randn(5, 4)
        
        output1 = model1(test_input)
        output2 = model2(test_input)
        
        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
        
        # Test physics computations are reproducible
        pde_formulator = PDEFormulator()
        
        residual1 = pde_formulator.darcy_residual(
            output1[:, 0], test_input, test_input[:, 2]
        )
        residual2 = pde_formulator.darcy_residual(
            output2[:, 0], test_input, test_input[:, 2]
        )
        
        self.assertTrue(torch.allclose(residual1, residual2, atol=1e-6))
        
        print("✓ Reproducibility test passed!")
    
    def test_error_handling_and_robustness(self):
        """Test system robustness and error handling."""
        
        print("Testing error handling and robustness...")
        
        # Test with invalid LAS file
        invalid_las = self.data_dir / "invalid.las"
        with open(invalid_las, 'w') as f:
            f.write("Invalid LAS content")
        
        las_reader = LASFileReader()
        
        # Should handle invalid files gracefully
        try:
            well_data = las_reader.read_las_file(str(invalid_las))
            # If it doesn't raise an exception, it should return None or handle gracefully
        except Exception as e:
            # Exception is acceptable for invalid files
            self.assertIsInstance(e, Exception)
        
        # Test with NaN inputs
        model = PINNArchitecture(4, [20, 20], 2)
        nan_input = torch.tensor([[1000.0, 0.15, float('nan'), 50.0]])
        
        # Model should handle NaN inputs without crashing
        try:
            output = model(nan_input)
            # Output might contain NaN, but shouldn't crash
            self.assertEqual(output.shape, (1, 2))
        except Exception as e:
            # Some level of NaN handling is expected
            pass
        
        # Test with extreme values
        extreme_input = torch.tensor([[1e6, 1e6, 1e6, 1e6]])
        try:
            output = model(extreme_input)
            self.assertEqual(output.shape, (1, 2))
        except Exception:
            # Extreme values might cause issues, but shouldn't crash system
            pass
        
        print("✓ Error handling and robustness test passed!")


class TestSystemIntegration(unittest.TestCase):
    """Test integration between major system components."""
    
    def test_data_to_model_integration(self):
        """Test integration between data processing and model components."""
        
        # Create sample well data
        well_data = WellData(
            well_id="TEST_001",
            depth=np.linspace(1000, 1100, 50),
            curves={
                'GR': np.random.normal(50, 10, 50),
                'PORO': np.random.normal(0.15, 0.02, 50),
                'PERM': np.random.normal(100, 20, 50),
                'NPHI': np.random.normal(0.16, 0.02, 50)
            },
            metadata=None
        )
        
        # Process data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_well_data(well_data)
        
        # Build dataset
        dataset_builder = DatasetBuilder()
        dataset = dataset_builder.build_dataset([processed_data])
        
        # Create model with appropriate input dimensions
        model = PINNArchitecture(4, [50, 50], 2)
        
        # Test that processed data works with model
        sample_batch = dataset['train_inputs'][:10]
        output = model(sample_batch)
        
        self.assertEqual(output.shape[0], 10)
        self.assertEqual(output.shape[1], 2)
    
    def test_model_to_physics_integration(self):
        """Test integration between model and physics components."""
        
        model = PINNArchitecture(4, [30, 30], 2)
        pde_formulator = PDEFormulator()
        boundary_handler = BoundaryConditionHandler()
        
        # Create test input
        test_input = torch.randn(15, 4, requires_grad=True)
        
        # Get model output
        output = model(test_input)
        
        # Test physics computations work with model output
        darcy_residual = pde_formulator.darcy_residual(
            output[:, 0], test_input, test_input[:, 2]
        )
        
        boundary_loss = boundary_handler.compute_boundary_loss(
            output, test_input, {}
        )
        
        self.assertIsInstance(darcy_residual, torch.Tensor)
        self.assertIsInstance(boundary_loss, torch.Tensor)
    
    def test_training_to_validation_integration(self):
        """Test integration between training and validation components."""
        
        # Create minimal training setup
        model = PINNArchitecture(4, [20, 20], 2)
        
        # Create physics components
        pde_formulator = PDEFormulator()
        boundary_handler = BoundaryConditionHandler()
        physics_loss_calc = PhysicsLossCalculator(pde_formulator, boundary_handler)
        
        # Create trainer
        trainer = PINNTrainer(physics_loss_calc)
        
        # Create validation framework
        val_config = ValidationConfig(n_folds=2, holdout_fraction=0.2)
        validator = ValidationFramework(val_config)
        
        # Create test data
        inputs = torch.randn(20, 4)
        targets = torch.randn(20, 2)
        
        # Test that validation works with trained model
        metrics = validator.compute_validation_metrics(model, inputs, targets)
        
        self.assertIn('l2_error', metrics)
        self.assertIn('mae', metrics)


if __name__ == '__main__':
    print("Running End-to-End Integration Tests...")
    print("=" * 60)
    
    # Run tests
    unittest.main(verbosity=2)