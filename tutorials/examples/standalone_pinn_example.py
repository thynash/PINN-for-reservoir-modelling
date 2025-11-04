#!/usr/bin/env python3
"""
Standalone PINN Example for Reservoir Modeling
==============================================

This standalone script demonstrates key PINN concepts for reservoir modeling
without requiring the full tutorial framework. Perfect for quick experimentation
and understanding core principles.

Author: PINN Tutorial Team
Date: 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SimplePINN(nn.Module):
    """
    Simple Physics-Informed Neural Network for reservoir modeling
    
    Architecture:
    - Input: [depth, porosity, permeability] 
    - Output: [pressure, saturation]
    - Hidden layers with Tanh activation for smooth derivatives
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 50, output_dim: int = 2):
        super(SimplePINN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization for stable training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.layers(x)


def generate_synthetic_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic reservoir data for demonstration
    
    Returns:
        X: Input features [depth, porosity, permeability]
        y: Target outputs [pressure, saturation]
    """
    
    # Generate realistic reservoir properties
    depth = np.random.uniform(2000, 3000, n_samples)  # feet
    porosity = np.random.uniform(0.1, 0.3, n_samples)  # fraction
    permeability = 10 ** np.random.uniform(-1, 3, n_samples)  # mD
    
    # Physics-based relationships for targets
    # Pressure increases with depth (hydrostatic + formation pressure)
    pressure = 0.433 * depth + 1000 + 100 * np.random.randn(n_samples)  # psi
    
    # Saturation related to porosity and permeability (simplified)
    saturation = 0.3 + 0.5 * porosity + 0.1 * np.log10(permeability) + 0.05 * np.random.randn(n_samples)
    saturation = np.clip(saturation, 0.2, 0.8)  # Physical bounds
    
    # Combine features and targets
    X = np.column_stack([depth, porosity, permeability])
    y = np.column_stack([pressure, saturation])
    
    return X, y


def physics_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute physics-informed loss based on Darcy's law
    
    Simplified physics constraint:
    - Pressure should vary smoothly with depth
    - Saturation should be consistent with porosity
    """
    
    # Enable gradient computation
    x.requires_grad_(True)
    
    # Forward pass
    output = model(x)
    pressure = output[:, 0:1]
    saturation = output[:, 1:2]
    
    # Extract coordinates and properties
    depth = x[:, 0:1]
    porosity = x[:, 1:2]
    permeability = x[:, 2:3]
    
    # Compute gradients
    pressure_grad = torch.autograd.grad(
        outputs=pressure.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Physics constraints
    # 1. Pressure gradient should be related to depth (simplified Darcy's law)
    dp_ddepth = pressure_grad[:, 0:1]
    expected_gradient = 0.433 * torch.ones_like(dp_ddepth)  # Hydrostatic gradient
    pressure_physics = torch.mean((dp_ddepth - expected_gradient) ** 2)
    
    # 2. Saturation should be physically bounded and related to porosity
    saturation_bounds = torch.mean(
        torch.relu(saturation - 0.8) + torch.relu(0.2 - saturation)
    )
    
    # 3. Saturation-porosity relationship (simplified)
    expected_saturation = 0.3 + 0.5 * porosity
    saturation_physics = torch.mean((saturation - expected_saturation) ** 2)
    
    return pressure_physics + saturation_bounds + 0.1 * saturation_physics


def train_pinn(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
               X_val: torch.Tensor, y_val: torch.Tensor, 
               epochs: int = 1000, lr: float = 1e-3) -> Dict[str, List[float]]:
    """
    Train the PINN with combined data and physics losses
    """
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    # Loss weights
    data_weight = 1.0
    physics_weight = 0.1
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'data_loss': [],
        'physics_loss': []
    }
    
    print("Starting PINN training...")
    print(f"Data samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(X_train)
        
        # Data loss (MSE)
        data_loss = nn.functional.mse_loss(y_pred, y_train)
        
        # Physics loss
        phys_loss = physics_loss(model, X_train)
        
        # Combined loss
        total_loss = data_weight * data_loss + physics_weight * phys_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = nn.functional.mse_loss(y_val_pred, y_val)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(total_loss.item())
        history['val_loss'].append(val_loss.item())
        history['data_loss'].append(data_loss.item())
        history['physics_loss'].append(phys_loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}: "
                  f"Total={total_loss.item():.6f}, "
                  f"Data={data_loss.item():.6f}, "
                  f"Physics={phys_loss.item():.6f}, "
                  f"Val={val_loss.item():.6f}")
    
    print("Training completed!")
    return history


def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate trained model performance
    """
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    
    # Convert to numpy
    y_true = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Compute metrics
    metrics = {}
    
    # Pressure metrics
    pressure_mse = np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2)
    pressure_mae = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
    pressure_r2 = 1 - pressure_mse / np.var(y_true[:, 0])
    
    # Saturation metrics
    saturation_mse = np.mean((y_true[:, 1] - y_pred[:, 1]) ** 2)
    saturation_mae = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
    saturation_r2 = 1 - saturation_mse / np.var(y_true[:, 1])
    
    metrics.update({
        'pressure_mse': pressure_mse,
        'pressure_mae': pressure_mae,
        'pressure_r2': pressure_r2,
        'saturation_mse': saturation_mse,
        'saturation_mae': saturation_mae,
        'saturation_r2': saturation_r2
    })
    
    return metrics


def visualize_results(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, 
                     history: Dict[str, List[float]]):
    """
    Create comprehensive visualization of PINN results
    """
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    
    # Convert to numpy
    X_np = X_test.cpu().numpy()
    y_true = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Training curves
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Total Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Loss components
    axes[0, 1].plot(epochs, history['data_loss'], 'g-', label='Data Loss', linewidth=2)
    axes[0, 1].plot(epochs, history['physics_loss'], 'purple', label='Physics Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss Component')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Pressure predictions
    axes[0, 2].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6, s=20)
    min_p, max_p = np.min(y_true[:, 0]), np.max(y_true[:, 0])
    axes[0, 2].plot([min_p, max_p], [min_p, max_p], 'r--', linewidth=2)
    axes[0, 2].set_xlabel('True Pressure (psi)')
    axes[0, 2].set_ylabel('Predicted Pressure (psi)')
    axes[0, 2].set_title('Pressure Predictions')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Saturation predictions
    axes[1, 0].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6, s=20, color='orange')
    min_s, max_s = np.min(y_true[:, 1]), np.max(y_true[:, 1])
    axes[1, 0].plot([min_s, max_s], [min_s, max_s], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('True Saturation')
    axes[1, 0].set_ylabel('Predicted Saturation')
    axes[1, 0].set_title('Saturation Predictions')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Depth vs Pressure
    axes[1, 1].scatter(X_np[:, 0], y_pred[:, 0], alpha=0.6, s=20, label='Predicted', color='blue')
    axes[1, 1].scatter(X_np[:, 0], y_true[:, 0], alpha=0.6, s=20, label='True', color='red')
    axes[1, 1].set_xlabel('Depth (ft)')
    axes[1, 1].set_ylabel('Pressure (psi)')
    axes[1, 1].set_title('Pressure vs Depth')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Porosity vs Saturation
    axes[1, 2].scatter(X_np[:, 1], y_pred[:, 1], alpha=0.6, s=20, label='Predicted', color='orange')
    axes[1, 2].scatter(X_np[:, 1], y_true[:, 1], alpha=0.6, s=20, label='True', color='red')
    axes[1, 2].set_xlabel('Porosity')
    axes[1, 2].set_ylabel('Saturation')
    axes[1, 2].set_title('Saturation vs Porosity')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('PINN Results Analysis', fontsize=16, y=1.02)
    plt.show()


def main():
    """
    Main function demonstrating complete PINN workflow
    """
    
    print("=" * 60)
    print("Physics-Informed Neural Network - Standalone Example")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic reservoir data...")
    X, y = generate_synthetic_data(n_samples=2000)
    
    # Split data
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    
    # 2. Create and train PINN
    print("\n2. Creating PINN model...")
    model = SimplePINN(input_dim=3, hidden_dim=64, output_dim=2).to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n3. Training PINN...")
    history = train_pinn(model, X_train, y_train, X_val, y_val, epochs=500, lr=1e-3)
    
    # 3. Evaluate model
    print("\n4. Evaluating model performance...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("   Performance Metrics:")
    print(f"   Pressure - R²: {metrics['pressure_r2']:.4f}, MAE: {metrics['pressure_mae']:.2f} psi")
    print(f"   Saturation - R²: {metrics['saturation_r2']:.4f}, MAE: {metrics['saturation_mae']:.4f}")
    
    # 4. Visualize results
    print("\n5. Creating visualizations...")
    visualize_results(model, X_test, y_test, history)
    
    print("\n" + "=" * 60)
    print("PINN Example Complete!")
    print("Key Insights:")
    print("• PINNs successfully combine data fitting with physics constraints")
    print("• Automatic differentiation enables seamless physics integration")
    print("• Two-phase training (Adam + L-BFGS) often improves convergence")
    print("• Physics constraints help with generalization and interpretability")
    print("=" * 60)


if __name__ == "__main__":
    main()