"""
Tests for core data models.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict

from src.core.data_models import (
    WellData,
    WellMetadata,
    TrainingConfig,
    ValidationMetrics,
    ModelConfig,
    BatchData,
    TrainingResults,
    PhysicsParameters
)


class TestWellMetadata:
    """Test WellMetadata data class."""
    
    def test_well_metadata_creation(self):
        """Test basic WellMetadata creation."""
        metadata = WellMetadata(
            well_id="TEST_001",
            location=(39.5, -98.5),
            formation="Test Formation",
            date_logged=datetime(2023, 1, 1),
            curve_units={"GR": "API", "RHOB": "g/cm3"},
            total_depth=1000.0
        )
        
        assert metadata.well_id == "TEST_001"
        assert metadata.location == (39.5, -98.5)
        assert metadata.formation == "Test Formation"
        assert metadata.total_depth == 1000.0


class TestWellData:
    """Test WellData data class."""
    
    def test_well_data_creation(self):
        """Test basic WellData creation."""
        metadata = WellMetadata(
            well_id="TEST_001",
            location=(39.5, -98.5),
            formation="Test Formation",
            date_logged=datetime(2023, 1, 1),
            curve_units={"GR": "API", "RHOB": "g/cm3"},
            total_depth=1000.0
        )
        
        depth = np.linspace(0, 1000, 100)
        curves = {
            "GR": np.random.normal(50, 10, 100),
            "RHOB": np.random.normal(2.5, 0.2, 100)
        }
        
        well_data = WellData(
            well_id="TEST_001",
            depth=depth,
            curves=curves,
            metadata=metadata
        )
        
        assert well_data.well_id == "TEST_001"
        assert len(well_data.depth) == 100
        assert len(well_data.curves["GR"]) == 100
        assert len(well_data.curves["RHOB"]) == 100
    
    def test_well_data_validation_empty_depth(self):
        """Test validation with empty depth array."""
        metadata = WellMetadata(
            well_id="TEST_001",
            location=(39.5, -98.5),
            formation="Test Formation",
            date_logged=datetime(2023, 1, 1),
            curve_units={},
            total_depth=0.0
        )
        
        with pytest.raises(ValueError, match="Depth array cannot be empty"):
            WellData(
                well_id="TEST_001",
                depth=np.array([]),
                curves={},
                metadata=metadata
            )
    
    def test_well_data_validation_mismatched_lengths(self):
        """Test validation with mismatched curve lengths."""
        metadata = WellMetadata(
            well_id="TEST_001",
            location=(39.5, -98.5),
            formation="Test Formation",
            date_logged=datetime(2023, 1, 1),
            curve_units={"GR": "API"},
            total_depth=1000.0
        )
        
        depth = np.linspace(0, 1000, 100)
        curves = {"GR": np.random.normal(50, 10, 50)}  # Wrong length
        
        with pytest.raises(ValueError, match="Curve GR length"):
            WellData(
                well_id="TEST_001",
                depth=depth,
                curves=curves,
                metadata=metadata
            )


class TestTrainingConfig:
    """Test TrainingConfig data class."""
    
    def test_training_config_creation(self):
        """Test basic TrainingConfig creation."""
        config = TrainingConfig(
            input_dim=4,
            hidden_dims=[100, 100, 100],
            output_dim=2,
            batch_size=512,
            learning_rate=1e-3,
            num_epochs=1000
        )
        
        assert config.input_dim == 4
        assert config.hidden_dims == [100, 100, 100]
        assert config.output_dim == 2
        assert config.batch_size == 512
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 1000
    
    def test_training_config_default_loss_weights(self):
        """Test default loss weights initialization."""
        config = TrainingConfig(
            input_dim=4,
            hidden_dims=[100, 100],
            output_dim=2
        )
        
        assert config.loss_weights is not None
        assert "data" in config.loss_weights
        assert "pde" in config.loss_weights
        assert "boundary" in config.loss_weights


class TestValidationMetrics:
    """Test ValidationMetrics data class."""
    
    def test_validation_metrics_creation(self):
        """Test basic ValidationMetrics creation."""
        metrics = ValidationMetrics(
            l2_error=0.1,
            mean_absolute_error=0.05,
            root_mean_square_error=0.08,
            mean_pde_residual=0.01,
            max_pde_residual=0.1,
            pde_residual_std=0.02,
            pressure_mae=0.03,
            pressure_rmse=0.04,
            saturation_mae=0.02,
            saturation_rmse=0.03,
            r2_score=0.95,
            relative_error=0.05,
            epoch=1000,
            training_time=300.0
        )
        
        assert metrics.l2_error == 0.1
        assert metrics.mean_absolute_error == 0.05
        assert metrics.r2_score == 0.95
        assert metrics.epoch == 1000


class TestModelConfig:
    """Test ModelConfig data class."""
    
    def test_model_config_creation(self):
        """Test basic ModelConfig creation."""
        config = ModelConfig(
            input_features=["depth", "porosity", "permeability", "gamma_ray"],
            output_features=["pressure", "saturation"],
            hidden_layers=[100, 100, 100],
            activation_function="tanh"
        )
        
        assert len(config.input_features) == 4
        assert len(config.output_features) == 2
        assert config.hidden_layers == [100, 100, 100]
        assert config.activation_function == "tanh"


class TestBatchData:
    """Test BatchData data class."""
    
    def test_batch_data_creation(self):
        """Test basic BatchData creation."""
        inputs = np.random.randn(32, 4)
        targets = np.random.randn(32, 2)
        
        batch = BatchData(
            inputs=inputs,
            targets=targets,
            well_ids=["well_1", "well_2"] * 16
        )
        
        assert len(batch) == 32
        assert batch.inputs.shape == (32, 4)
        assert batch.targets.shape == (32, 2)
        assert len(batch.well_ids) == 32


class TestPhysicsParameters:
    """Test PhysicsParameters data class."""
    
    def test_physics_parameters_creation(self):
        """Test basic PhysicsParameters creation."""
        params = PhysicsParameters(
            oil_viscosity=2.0,
            water_viscosity=1.0,
            porosity_range=(0.1, 0.3)
        )
        
        assert params.oil_viscosity == 2.0
        assert params.water_viscosity == 1.0
        assert params.porosity_range == (0.1, 0.3)
    
    def test_physics_parameters_default_boundaries(self):
        """Test default boundary conditions initialization."""
        params = PhysicsParameters()
        
        assert params.pressure_boundaries is not None
        assert "inlet" in params.pressure_boundaries
        assert "outlet" in params.pressure_boundaries
        assert params.saturation_boundaries is not None