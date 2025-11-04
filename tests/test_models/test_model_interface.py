"""
Tests for PINN model interface and utilities.

This module tests the unified interface for PINN model operations including
training, inference, checkpointing, and parameter management.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import json
from pathlib import Path

from src.models.model_interface import PINNModelInterface
from src.models.pinn_architecture import PINNArchitecture
from src.core.data_models import ModelConfig


class TestPINNModelInterface:
    """Test PINNModelInterface functionality."""
    
    @pytest.fixture
    def model_config(self):
        """Create model configuration for testing."""
        return ModelConfig(
            input_features=["depth", "porosity", "permeability", "gamma_ray"],
            output_features=["pressure", "saturation"],
            hidden_layers=[64, 64, 64],
            activation_function="tanh",
            dropout_rate=0.0,
            batch_normalization=False,
            weight_initialization="xavier_uniform"
        )
    
    @pytest.fixture
    def basic_model(self):
        """Create basic PINN model for testing."""
        return PINNArchitecture(
            input_dim=4,
            hidden_dims=[64, 64, 64],
            output_dim=2,
            activation="tanh"
        )
    
    @pytest.fixture
    def model_interface(self, basic_model):
        """Create model interface with basic model."""
        return PINNModelInterface(model=basic_model, device=torch.device('cpu'))
    
    @pytest.fixture
    def interface_from_config(self, model_config):
        """Create model interface from configuration."""
        return PINNModelInterface(config=model_config, device=torch.device('cpu'))
    
    def test_initialization_with_model(self, basic_model):
        """Test initialization with existing model."""
        interface = PINNModelInterface(model=basic_model)
        
        assert interface.model is not None
        assert interface.model == basic_model
        assert interface.device.type in ['cpu', 'cuda']
        assert not interface.is_trained
    
    def test_initialization_with_config(self, model_config):
        """Test initialization with model configuration."""
        interface = PINNModelInterface(config=model_config)
        
        assert interface.model is not None
        assert interface.config == model_config
        assert interface.model.input_dim == 4
        assert interface.model.output_dim == 2
    
    def test_initialization_empty(self):
        """Test initialization without model or config."""
        interface = PINNModelInterface()
        
        assert interface.model is None
        assert interface.config is None
        assert interface.device.type in ['cpu', 'cuda']
    
    def test_create_model(self, model_interface):
        """Test model creation through interface."""
        new_model = model_interface.create_model(
            input_dim=6,
            hidden_dims=[50, 100, 50],
            output_dim=3,
            activation="swish"
        )
        
        assert isinstance(new_model, PINNArchitecture)
        assert new_model.input_dim == 6
        assert new_model.output_dim == 3
        assert new_model.activation_name == "swish"
        assert model_interface.model == new_model
    
    def test_forward_numpy_input(self, model_interface):
        """Test forward pass with numpy input."""
        inputs = np.random.randn(5, 4)
        outputs = model_interface.forward(inputs)
        
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (5, 2)
        assert not torch.isnan(outputs).any()
    
    def test_forward_tensor_input(self, model_interface):
        """Test forward pass with tensor input."""
        inputs = torch.randn(3, 4)
        outputs = model_interface.forward(inputs)
        
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (3, 2)
        assert not torch.isnan(outputs).any()
    
    def test_forward_no_model(self):
        """Test forward pass without model raises error."""
        interface = PINNModelInterface()
        inputs = np.random.randn(5, 4)
        
        with pytest.raises(ValueError, match="Model not initialized"):
            interface.forward(inputs)
    
    def test_predict(self, model_interface):
        """Test prediction method."""
        inputs = np.random.randn(5, 4)
        predictions = model_interface.predict(inputs)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (5, 2)
        assert not np.isnan(predictions).any()
    
    def test_compute_derivatives(self, model_interface):
        """Test derivative computation."""
        inputs = np.random.randn(3, 4)
        derivatives = model_interface.compute_derivatives(inputs)
        
        assert isinstance(derivatives, dict)
        assert len(derivatives) > 0
        
        # Check that derivatives are computed for all output-input pairs
        for key, value in derivatives.items():
            assert isinstance(value, torch.Tensor)
            assert value.shape == (3,)
            assert not torch.isnan(value).any()
    
    def test_compute_derivatives_no_model(self):
        """Test derivative computation without model raises error."""
        interface = PINNModelInterface()
        inputs = np.random.randn(3, 4)
        
        with pytest.raises(ValueError, match="Model not initialized"):
            interface.compute_derivatives(inputs)
    
    def test_set_normalization_parameters(self, model_interface):
        """Test setting normalization parameters."""
        input_stats = {
            "depth": (1000.0, 500.0),
            "porosity": (0.2, 0.05),
            "permeability": (100.0, 50.0),
            "gamma_ray": (50.0, 20.0)
        }
        
        output_stats = {
            "pressure": (2000.0, 500.0),
            "saturation": (0.5, 0.2)
        }
        
        model_interface.set_normalization_parameters(input_stats, output_stats)
        
        # Check that parameters are set in the model
        assert model_interface.model.input_mean.shape == (4,)
        assert model_interface.model.input_std.shape == (4,)
        assert model_interface.model.output_mean.shape == (2,)
        assert model_interface.model.output_std.shape == (2,)
    
    def test_set_normalization_no_model(self):
        """Test setting normalization without model raises error."""
        interface = PINNModelInterface()
        input_stats = {"depth": (1000.0, 500.0)}
        output_stats = {"pressure": (2000.0, 500.0)}
        
        with pytest.raises(ValueError, match="Model not initialized"):
            interface.set_normalization_parameters(input_stats, output_stats)
    
    def test_initialize_parameters(self, model_interface):
        """Test parameter initialization."""
        # Get initial weights (skip bias parameters which are always zero)
        initial_weights = []
        for param in model_interface.model.parameters():
            if param.dim() > 1:  # Only check weight matrices, not biases
                initial_weights.append(param.clone())
        
        # Reinitialize
        model_interface.initialize_parameters("xavier_normal")
        
        # Check that weights changed (skip bias parameters)
        current_weights = []
        for param in model_interface.model.parameters():
            if param.dim() > 1:  # Only check weight matrices, not biases
                current_weights.append(param)
        
        for initial, current in zip(initial_weights, current_weights):
            assert not torch.allclose(initial, current)
    
    def test_get_parameter_count(self, model_interface):
        """Test parameter count retrieval."""
        counts = model_interface.get_parameter_count()
        
        assert 'total_parameters' in counts
        assert 'trainable_parameters' in counts
        assert 'non_trainable_parameters' in counts
        
        assert counts['total_parameters'] > 0
        assert counts['trainable_parameters'] > 0
        assert counts['total_parameters'] == counts['trainable_parameters'] + counts['non_trainable_parameters']
    
    def test_save_load_checkpoint(self, model_interface):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_checkpoint.pth")
            
            # Set some normalization parameters
            input_stats = {"depth": (1000.0, 500.0), "porosity": (0.2, 0.05), 
                          "permeability": (100.0, 50.0), "gamma_ray": (50.0, 20.0)}
            output_stats = {"pressure": (2000.0, 500.0), "saturation": (0.5, 0.2)}
            model_interface.set_normalization_parameters(input_stats, output_stats)
            
            # Save checkpoint
            optimizer_state = {'lr': 0.001, 'step': 100}
            loss_history = {'total_loss': [1.0, 0.8, 0.6]}
            metadata = {'experiment': 'test'}
            
            model_interface.save_checkpoint(
                filepath=filepath,
                epoch=10,
                optimizer_state=optimizer_state,
                loss_history=loss_history,
                metadata=metadata
            )
            
            assert os.path.exists(filepath)
            
            # Create new interface and load
            new_interface = PINNModelInterface(device=torch.device('cpu'))
            checkpoint_info = new_interface.load_checkpoint(filepath, load_optimizer=True)
            
            assert checkpoint_info['epoch'] == 10
            assert 'optimizer_state_dict' in checkpoint_info
            assert new_interface.is_trained
            
            # Test that models produce same output
            test_input = torch.randn(1, 4)
            output1 = model_interface.forward(test_input)
            output2 = new_interface.forward(test_input)
            assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_load_checkpoint_nonexistent(self):
        """Test loading nonexistent checkpoint raises error."""
        interface = PINNModelInterface()
        
        with pytest.raises(FileNotFoundError):
            interface.load_checkpoint("nonexistent_file.pth")
    
    def test_save_load_model_only(self, model_interface):
        """Test saving and loading model only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_model.pth")
            
            # Save model
            model_interface.save_model_only(filepath)
            assert os.path.exists(filepath)
            
            # Create new interface and load
            new_interface = PINNModelInterface()
            new_interface.create_model(input_dim=4, hidden_dims=[64, 64, 64], output_dim=2)
            new_interface.load_model_only(filepath)
            
            assert new_interface.is_trained
            
            # Test that models produce same output
            test_input = torch.randn(1, 4)
            output1 = model_interface.forward(test_input)
            output2 = new_interface.forward(test_input)
            assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_evaluate_model(self, model_interface):
        """Test model evaluation with metrics."""
        inputs = np.random.randn(10, 4)
        targets = np.random.randn(10, 2)
        
        metrics = model_interface.evaluate_model(inputs, targets, metrics=['mse', 'mae', 'r2'])
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert isinstance(metrics['r2'], float)
    
    def test_evaluate_model_tensor_targets(self, model_interface):
        """Test model evaluation with tensor targets."""
        inputs = np.random.randn(10, 4)
        targets = torch.randn(10, 2)
        
        metrics = model_interface.evaluate_model(inputs, targets)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
    
    def test_evaluate_model_all_metrics(self, model_interface):
        """Test model evaluation with all available metrics."""
        inputs = np.random.randn(10, 4)
        targets = np.random.randn(10, 2)
        
        all_metrics = ['mse', 'mae', 'rmse', 'r2', 'relative_error']
        metrics = model_interface.evaluate_model(inputs, targets, metrics=all_metrics)
        
        for metric in all_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
    
    def test_get_model_summary(self, model_interface):
        """Test model summary retrieval."""
        summary = model_interface.get_model_summary()
        
        assert 'input_dim' in summary
        assert 'output_dim' in summary
        assert 'total_parameters' in summary
        assert 'device' in summary
        assert 'is_trained' in summary
        
        assert summary['input_dim'] == 4
        assert summary['output_dim'] == 2
        assert not summary['is_trained']
    
    def test_get_model_summary_no_model(self):
        """Test model summary without model."""
        interface = PINNModelInterface()
        summary = interface.get_model_summary()
        
        assert summary['status'] == 'Model not initialized'
    
    def test_set_training_mode(self, model_interface):
        """Test setting training mode."""
        # Set to training mode
        model_interface.set_training_mode(True)
        assert model_interface.model.training
        
        # Set to evaluation mode
        model_interface.set_training_mode(False)
        assert not model_interface.model.training
    
    def test_get_model(self, model_interface):
        """Test getting underlying model."""
        model = model_interface.get_model()
        assert model == model_interface.model
        assert isinstance(model, PINNArchitecture)
    
    def test_to_device(self, model_interface):
        """Test moving model to device."""
        original_device = model_interface.device
        
        # Move to same device (should work)
        model_interface.to_device(torch.device('cpu'))
        assert model_interface.device == torch.device('cpu')
        
        # Check that model parameters are on the correct device
        for param in model_interface.model.parameters():
            assert param.device.type == 'cpu'
    
    def test_model_consistency_after_operations(self, model_interface):
        """Test model consistency after various operations."""
        # Set normalization
        input_stats = {"depth": (1000.0, 500.0), "porosity": (0.2, 0.05), 
                      "permeability": (100.0, 50.0), "gamma_ray": (50.0, 20.0)}
        output_stats = {"pressure": (2000.0, 500.0), "saturation": (0.5, 0.2)}
        model_interface.set_normalization_parameters(input_stats, output_stats)
        
        # Test prediction
        inputs = np.random.randn(5, 4)
        predictions1 = model_interface.predict(inputs)
        
        # Set training mode and back to eval
        model_interface.set_training_mode(True)
        model_interface.set_training_mode(False)
        
        # Test prediction again
        predictions2 = model_interface.predict(inputs)
        
        # Should be the same
        assert np.allclose(predictions1, predictions2)


class TestPINNModelInterfaceEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def basic_model(self):
        """Create basic PINN model for testing."""
        return PINNArchitecture(
            input_dim=4,
            hidden_dims=[64, 64, 64],
            output_dim=2,
            activation="tanh"
        )
    
    @pytest.fixture
    def model_interface(self, basic_model):
        """Create model interface with basic model."""
        return PINNModelInterface(model=basic_model, device=torch.device('cpu'))
    
    def test_operations_without_model(self):
        """Test various operations without initialized model."""
        interface = PINNModelInterface()
        
        # These should raise ValueError
        with pytest.raises(ValueError):
            interface.forward(np.random.randn(1, 4))
        
        with pytest.raises(ValueError):
            interface.compute_derivatives(np.random.randn(1, 4))
        
        with pytest.raises(ValueError):
            interface.set_normalization_parameters({}, {})
        
        with pytest.raises(ValueError):
            interface.initialize_parameters()
        
        with pytest.raises(ValueError):
            interface.get_parameter_count()
        
        with pytest.raises(ValueError):
            interface.save_checkpoint("test.pth", 0)
        
        with pytest.raises(ValueError):
            interface.save_model_only("test.pth")
    
    def test_load_model_only_without_model(self):
        """Test loading model only without initialized model."""
        interface = PINNModelInterface()
        
        with pytest.raises(ValueError, match="Model not initialized"):
            interface.load_model_only("test.pth")
    
    def test_empty_input_handling(self, model_interface):
        """Test handling of empty inputs."""
        empty_input = np.empty((0, 4))
        output = model_interface.forward(empty_input)
        
        assert output.shape == (0, 2)
    
    def test_single_sample_input(self, model_interface):
        """Test handling of single sample input."""
        single_input = np.random.randn(1, 4)
        output = model_interface.forward(single_input)
        
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()
    
    def test_large_batch_input(self, model_interface):
        """Test handling of large batch input."""
        large_input = np.random.randn(1000, 4)
        output = model_interface.forward(large_input)
        
        assert output.shape == (1000, 2)
        assert not torch.isnan(output).any()
    
    def test_checkpoint_directory_creation(self, model_interface):
        """Test checkpoint saving creates directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "dir", "checkpoint.pth")
            
            model_interface.save_checkpoint(nested_path, epoch=1)
            
            assert os.path.exists(nested_path)
            assert os.path.isdir(os.path.dirname(nested_path))
    
    def test_model_interface_with_different_devices(self):
        """Test model interface behavior with different device specifications."""
        # CPU device
        interface_cpu = PINNModelInterface(device=torch.device('cpu'))
        assert interface_cpu.device == torch.device('cpu')
        
        # Default device (should work)
        interface_default = PINNModelInterface()
        assert interface_default.device.type in ['cpu', 'cuda']