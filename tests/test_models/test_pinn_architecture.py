"""
Tests for PINN architecture implementation.

This module tests the core neural network architecture including forward pass,
gradient computation, and model persistence functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from pathlib import Path

from src.models.pinn_architecture import PINNArchitecture, Swish
from src.core.data_models import ModelConfig


class TestSwishActivation:
    """Test Swish activation function."""
    
    def test_swish_forward(self):
        """Test Swish activation forward pass."""
        swish = Swish()
        x = torch.tensor([0.0, 1.0, -1.0, 2.0])
        output = swish(x)
        
        # Swish(x) = x * sigmoid(x)
        expected = x * torch.sigmoid(x)
        assert torch.allclose(output, expected, atol=1e-6)
    
    def test_swish_gradient(self):
        """Test Swish activation gradient computation."""
        swish = Swish()
        x = torch.tensor([1.0], requires_grad=True)
        output = swish(x)
        output.backward()
        
        # Gradient should be computed correctly
        assert x.grad is not None
        assert not torch.isnan(x.grad)


class TestPINNArchitecture:
    """Test PINN architecture implementation."""
    
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
    def model_config(self):
        """Create model configuration for testing."""
        return ModelConfig(
            input_features=["depth", "porosity", "permeability", "gamma_ray"],
            output_features=["pressure", "saturation"],
            hidden_layers=[50, 100, 50],
            activation_function="swish",
            dropout_rate=0.1,
            batch_normalization=True,
            weight_initialization="xavier_uniform"
        )
    
    def test_model_initialization(self, basic_model):
        """Test basic model initialization."""
        assert basic_model.input_dim == 4
        assert basic_model.hidden_dims == [64, 64, 64]
        assert basic_model.output_dim == 2
        assert basic_model.activation_name == "tanh"
        assert isinstance(basic_model.activation, nn.Tanh)
    
    def test_model_from_config(self, model_config):
        """Test model creation from configuration."""
        model = PINNArchitecture.from_config(model_config)
        
        assert model.input_dim == 4
        assert model.output_dim == 2
        assert model.hidden_dims == [50, 100, 50]
        assert model.activation_name == "swish"
        assert model.dropout_rate == 0.1
        assert model.batch_norm == True
    
    def test_architecture_validation(self):
        """Test architecture parameter validation."""
        # Test invalid number of layers
        with pytest.raises(ValueError, match="Number of hidden layers should be between 3 and 5"):
            PINNArchitecture(input_dim=4, hidden_dims=[64, 64], output_dim=2)
        
        # Test invalid layer size
        with pytest.raises(ValueError, match="Hidden layer dimensions should be between 50 and 200"):
            PINNArchitecture(input_dim=4, hidden_dims=[30, 64, 64], output_dim=2)
        
        # Test invalid dropout rate
        with pytest.raises(ValueError, match="Dropout rate should be between 0 and 1"):
            PINNArchitecture(input_dim=4, hidden_dims=[64, 64, 64], output_dim=2, dropout_rate=1.5)
    
    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ["tanh", "swish", "relu", "elu", "gelu"]
        
        for activation in activations:
            model = PINNArchitecture(
                input_dim=4,
                hidden_dims=[64, 64, 64],
                output_dim=2,
                activation=activation
            )
            assert model.activation_name == activation
    
    def test_invalid_activation(self):
        """Test invalid activation function."""
        with pytest.raises(ValueError, match="Unsupported activation function"):
            PINNArchitecture(
                input_dim=4,
                hidden_dims=[64, 64, 64],
                output_dim=2,
                activation="invalid"
            )
    
    def test_forward_pass_basic(self, basic_model):
        """Test basic forward pass with single input."""
        x = torch.randn(1, 4)
        output = basic_model(x)
        
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_pass_batch(self, basic_model):
        """Test forward pass with batch input."""
        batch_sizes = [1, 5, 10, 32]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 4)
            output = basic_model(x)
            
            assert output.shape == (batch_size, 2)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_forward_pass_different_input_dims(self):
        """Test forward pass with various input dimensions."""
        input_dims = [2, 4, 6, 8]
        
        for input_dim in input_dims:
            model = PINNArchitecture(
                input_dim=input_dim,
                hidden_dims=[64, 64, 64],
                output_dim=2
            )
            
            x = torch.randn(5, input_dim)
            output = model(x)
            
            assert output.shape == (5, 2)
            assert not torch.isnan(output).any()
    
    def test_normalization_parameters(self, basic_model):
        """Test input normalization and output scaling."""
        # Set normalization parameters
        input_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
        input_std = torch.tensor([0.5, 1.0, 1.5, 2.0])
        output_mean = torch.tensor([100.0, 0.5])
        output_std = torch.tensor([50.0, 0.2])
        
        basic_model.set_input_normalization(input_mean, input_std)
        basic_model.set_output_scaling(output_mean, output_std)
        
        # Test that parameters are set correctly
        assert torch.allclose(basic_model.input_mean, input_mean)
        assert torch.allclose(basic_model.input_std, input_std)
        assert torch.allclose(basic_model.output_mean, output_mean)
        assert torch.allclose(basic_model.output_std, output_std)
        
        # Test normalization function
        x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
        normalized = basic_model.normalize_input(x)
        expected = (x - input_mean) / (input_std + 1e-8)
        assert torch.allclose(normalized, expected)
    
    def test_gradient_computation(self, basic_model):
        """Test gradient computation for automatic differentiation."""
        x = torch.randn(5, 4, requires_grad=True)
        output = basic_model(x)
        
        # Test that gradients can be computed
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.shape == x.shape
    
    def test_compute_derivatives(self, basic_model):
        """Test derivative computation method."""
        x = torch.randn(3, 4, requires_grad=True)
        derivatives = basic_model.compute_derivatives(x)
        
        # Check that derivatives are computed for all output-input pairs
        expected_keys = []
        for i in range(2):  # output_dim
            for j in range(4):  # input_dim
                expected_keys.append(f"d_output_{i}_d_input_{j}")
        
        for key in expected_keys:
            assert key in derivatives
            assert derivatives[key].shape == (3,)  # batch_size
            assert not torch.isnan(derivatives[key]).any()
    
    def test_compute_derivatives_no_grad(self, basic_model):
        """Test derivative computation fails without gradients."""
        x = torch.randn(3, 4, requires_grad=False)
        
        with pytest.raises(ValueError, match="Input tensor must have requires_grad=True"):
            basic_model.compute_derivatives(x)
    
    def test_compute_second_derivatives(self, basic_model):
        """Test second derivative computation."""
        x = torch.randn(2, 4, requires_grad=True)
        second_derivatives = basic_model.compute_second_derivatives(x)
        
        # Should have second derivatives for all combinations
        assert len(second_derivatives) > 0
        
        # Check some expected keys exist
        for key, value in second_derivatives.items():
            assert value.shape == (2,)  # batch_size
            assert not torch.isnan(value).any()
    
    def test_model_info(self, basic_model):
        """Test model information retrieval."""
        info = basic_model.get_model_info()
        
        assert info['input_dim'] == 4
        assert info['hidden_dims'] == [64, 64, 64]
        assert info['output_dim'] == 2
        assert info['activation'] == "tanh"
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
    
    def test_model_save_load(self, basic_model):
        """Test model saving and loading functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_model.pth")
            
            # Set some normalization parameters
            input_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
            input_std = torch.tensor([0.5, 1.0, 1.5, 2.0])
            basic_model.set_input_normalization(input_mean, input_std)
            
            # Save model
            basic_model.save_model(filepath)
            assert os.path.exists(filepath)
            
            # Create new model and load
            new_model = PINNArchitecture(
                input_dim=4,
                hidden_dims=[64, 64, 64],
                output_dim=2,
                activation="tanh"
            )
            new_model.load_model(filepath)
            
            # Test that parameters are the same
            for p1, p2 in zip(basic_model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
            
            # Test that normalization parameters are loaded
            assert torch.allclose(new_model.input_mean, input_mean)
    
    def test_model_consistency(self, basic_model):
        """Test model output consistency."""
        x = torch.randn(5, 4)
        
        # Multiple forward passes should give same result
        basic_model.eval()
        output1 = basic_model(x)
        output2 = basic_model(x)
        
        assert torch.allclose(output1, output2)
    
    def test_batch_normalization(self):
        """Test model with batch normalization."""
        model = PINNArchitecture(
            input_dim=4,
            hidden_dims=[64, 64, 64],
            output_dim=2,
            batch_norm=True
        )
        
        x = torch.randn(10, 4)
        
        # Test training mode
        model.train()
        output_train = model(x)
        
        # Test evaluation mode
        model.eval()
        output_eval = model(x)
        
        assert output_train.shape == (10, 2)
        assert output_eval.shape == (10, 2)
        # Outputs should be different due to batch norm behavior
        assert not torch.allclose(output_train, output_eval, atol=1e-5)
    
    def test_dropout(self):
        """Test model with dropout."""
        model = PINNArchitecture(
            input_dim=4,
            hidden_dims=[64, 64, 64],
            output_dim=2,
            dropout_rate=0.5
        )
        
        x = torch.randn(10, 4)
        
        # Test training mode (dropout active)
        model.train()
        output1 = model(x)
        output2 = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2, atol=1e-5)
        
        # Test evaluation mode (dropout inactive)
        model.eval()
        output3 = model(x)
        output4 = model(x)
        
        # Outputs should be the same in eval mode
        assert torch.allclose(output3, output4)
    
    def test_weight_initialization(self):
        """Test different weight initialization methods."""
        init_methods = ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]
        
        for method in init_methods:
            model = PINNArchitecture(
                input_dim=4,
                hidden_dims=[64, 64, 64],
                output_dim=2,
                weight_init=method
            )
            
            # Check that weights are initialized (not all zeros)
            for layer in model.layers:
                if isinstance(layer, nn.Linear):
                    assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
                    assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
    
    def test_invalid_weight_initialization(self):
        """Test invalid weight initialization method."""
        with pytest.raises(ValueError, match="Unsupported initialization method"):
            PINNArchitecture(
                input_dim=4,
                hidden_dims=[64, 64, 64],
                output_dim=2,
                weight_init="invalid_method"
            )


class TestPINNArchitectureEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input(self):
        """Test model behavior with empty input."""
        model = PINNArchitecture(
            input_dim=4,
            hidden_dims=[64, 64, 64],
            output_dim=2
        )
        
        # Empty batch should work
        x = torch.empty(0, 4)
        output = model(x)
        assert output.shape == (0, 2)
    
    def test_large_batch(self):
        """Test model with large batch size."""
        model = PINNArchitecture(
            input_dim=4,
            hidden_dims=[64, 64, 64],
            output_dim=2
        )
        
        # Large batch
        x = torch.randn(1000, 4)
        output = model(x)
        
        assert output.shape == (1000, 2)
        assert not torch.isnan(output).any()
    
    def test_extreme_values(self):
        """Test model with extreme input values."""
        model = PINNArchitecture(
            input_dim=4,
            hidden_dims=[64, 64, 64],
            output_dim=2
        )
        
        # Very large values
        x_large = torch.tensor([[1e6, 1e6, 1e6, 1e6]])
        output_large = model(x_large)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()
        
        # Very small values
        x_small = torch.tensor([[1e-6, 1e-6, 1e-6, 1e-6]])
        output_small = model(x_small)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()