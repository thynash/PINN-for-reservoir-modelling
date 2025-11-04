# Exercise 2: PINN Implementation and Physics Integration

## Objective
Implement a complete Physics-Informed Neural Network from scratch, including automatic differentiation for physics constraints and proper loss function design.

## Background
The key innovation of PINNs is the integration of physics constraints directly into the neural network training process. This exercise will guide you through implementing the core components that make PINNs work.

## Tasks

### Task 2.1: Basic PINN Architecture (25 minutes)

**Goal**: Implement a neural network architecture suitable for physics-informed learning.

**Instructions**:
1. Create a `ReservoirPINN` class that inherits from `nn.Module`
2. Implement proper weight initialization for stable training
3. Use smooth activation functions (Tanh or Swish)
4. Add methods for computing derivatives

**Code Template**:
```python
import torch
import torch.nn as nn
import numpy as np

class ReservoirPINN(nn.Module):
    """Physics-Informed Neural Network for reservoir modeling"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        super(ReservoirPINN, self).__init__()
        
        # TODO: Implement network architecture
        # Hint: Use nn.ModuleList for flexible hidden layers
        
    def forward(self, x):
        """Forward pass through the network"""
        # TODO: Implement forward pass
        pass
    
    def compute_derivatives(self, x, output):
        """Compute spatial derivatives for physics constraints"""
        # TODO: Implement automatic differentiation
        # Hint: Use torch.autograd.grad with create_graph=True
        pass

# Test your implementation
model = ReservoirPINN(input_dim=4, hidden_dims=[64, 64, 64], output_dim=2)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Expected Output**:
- Working neural network with proper architecture
- Successful forward pass
- Derivative computation capability

### Task 2.2: Physics Constraint Implementation (30 minutes)

**Goal**: Implement physics constraints based on Darcy's law and continuity equation.

**Instructions**:
1. Implement Darcy's law residual: $\nabla \cdot \left(\frac{k}{\mu} \nabla p\right) = 0$
2. Add boundary condition handling
3. Create a physics loss calculator
4. Test with synthetic data

**Code Template**:
```python
class PhysicsConstraints:
    """Physics constraints for reservoir modeling"""
    
    def __init__(self, viscosity=1e-3):
        self.viscosity = viscosity
    
    def darcy_residual(self, pressure, coords, permeability):
        """
        Compute Darcy's law residual
        
        Args:
            pressure: Pressure predictions from PINN
            coords: Spatial coordinates (x, y)
            permeability: Permeability field
        
        Returns:
            PDE residual tensor
        """
        # TODO: Implement Darcy's law residual
        # Steps:
        # 1. Compute pressure gradients: dp/dx, dp/dy
        # 2. Compute flux: q = -(k/μ) * ∇p
        # 3. Compute divergence: ∇·q
        pass
    
    def boundary_conditions(self, predictions, boundary_points, boundary_values):
        """Compute boundary condition residuals"""
        # TODO: Implement Dirichlet boundary conditions
        pass
    
    def compute_physics_loss(self, model, inputs, boundary_data=None):
        """Compute total physics loss"""
        # TODO: Combine PDE and boundary losses
        pass

# Test physics constraints
physics = PhysicsConstraints()
# Add test code here
```

**Expected Output**:
- Working PDE residual computation
- Boundary condition enforcement
- Combined physics loss function

### Task 2.3: Complete PINN Loss Function (25 minutes)

**Goal**: Implement the complete PINN loss function combining data fitting and physics constraints.

**Instructions**:
1. Create a `PINNLoss` class that combines multiple loss terms
2. Implement adaptive loss weighting
3. Add gradient monitoring for numerical stability
4. Test convergence behavior

**Code Template**:
```python
class PINNLoss(nn.Module):
    """Complete PINN loss function"""
    
    def __init__(self, physics_constraints, loss_weights=None):
        super(PINNLoss, self).__init__()
        self.physics = physics_constraints
        self.weights = loss_weights or {'data': 1.0, 'physics': 0.1, 'boundary': 0.5}
        
    def forward(self, model, inputs, targets, physics_points=None, boundary_data=None):
        """
        Compute complete PINN loss
        
        Returns:
            total_loss, loss_components
        """
        # TODO: Implement complete loss computation
        # 1. Data loss (MSE between predictions and targets)
        # 2. Physics loss (PDE residuals)
        # 3. Boundary loss (BC violations)
        # 4. Combine with weights
        pass
    
    def update_weights(self, loss_components, epoch):
        """Adaptive loss weight updating"""
        # TODO: Implement adaptive weighting strategy
        pass

# Test loss function
pinn_loss = PINNLoss(physics)
# Add test code here
```

**Expected Output**:
- Working combined loss function
- Proper loss component tracking
- Adaptive weight updating

### Task 2.4: Training Loop Implementation (30 minutes)

**Goal**: Implement a complete training loop with proper convergence monitoring.

**Instructions**:
1. Create a training function with both Adam and L-BFGS phases
2. Implement gradient clipping and NaN detection
3. Add convergence monitoring and early stopping
4. Create comprehensive logging

**Code Template**:
```python
def train_pinn(model, train_data, val_data, physics_points, config):
    """
    Complete PINN training with two-phase optimization
    
    Args:
        model: PINN model
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        physics_points: Points for physics loss computation
        config: Training configuration
    
    Returns:
        training_history
    """
    
    # Initialize components
    physics = PhysicsConstraints()
    pinn_loss = PINNLoss(physics, config['loss_weights'])
    
    # Phase 1: Adam optimizer
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=config['lr_adam'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, patience=50)
    
    history = {'train_loss': [], 'val_loss': [], 'data_loss': [], 'physics_loss': []}
    
    print("Phase 1: Adam Optimization")
    for epoch in range(config['adam_epochs']):
        # TODO: Implement training step
        # 1. Forward pass
        # 2. Compute losses
        # 3. Backward pass with gradient clipping
        # 4. Validation
        # 5. Update learning rate
        # 6. Check for early stopping
        pass
    
    # Phase 2: L-BFGS refinement
    print("Phase 2: L-BFGS Refinement")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=config['lbfgs_iter'],
        tolerance_grad=1e-7,
        tolerance_change=1e-9
    )
    
    def closure():
        # TODO: Implement L-BFGS closure function
        pass
    
    # TODO: Run L-BFGS optimization
    
    return history

# Training configuration
config = {
    'adam_epochs': 1000,
    'lbfgs_iter': 500,
    'lr_adam': 1e-3,
    'loss_weights': {'data': 1.0, 'physics': 0.1, 'boundary': 0.5},
    'gradient_clip': 1.0,
    'early_stopping_patience': 100
}

# Test training
# history = train_pinn(model, train_data, val_data, physics_points, config)
```

**Expected Output**:
- Successful two-phase training
- Convergence monitoring
- Training history with all loss components

## Solutions

### Solution 2.1: Basic PINN Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReservoirPINN(nn.Module):
    """Physics-Informed Neural Network for reservoir modeling"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        super(ReservoirPINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # Choose activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization for stable training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        
        # Input normalization (optional but recommended)
        # x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        return x
    
    def compute_derivatives(self, x, output=None):
        """Compute spatial derivatives for physics constraints"""
        
        if output is None:
            output = self.forward(x)
        
        # Ensure gradients are enabled
        if not x.requires_grad:
            x.requires_grad_(True)
        
        derivatives = {}
        
        # First derivatives
        for i in range(self.output_dim):
            for j in range(self.input_dim):
                grad_output = torch.zeros_like(output)
                grad_output[:, i] = 1.0
                
                grad = torch.autograd.grad(
                    outputs=output,
                    inputs=x,
                    grad_outputs=grad_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                
                derivatives[f'd{i}_d{j}'] = grad[:, j:j+1]
        
        return derivatives

# Test implementation
def test_pinn_architecture():
    """Test the PINN architecture"""
    
    # Create model
    model = ReservoirPINN(input_dim=4, hidden_dims=[64, 64, 64], output_dim=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 4, requires_grad=True)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test derivatives
    derivatives = model.compute_derivatives(x, output)
    print(f"Computed derivatives: {list(derivatives.keys())}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
    print(f"Gradient norm: {grad_norm.item():.6f}")
    
    return model

# Run test
model = test_pinn_architecture()
```

### Solution 2.2: Physics Constraint Implementation

```python
class PhysicsConstraints:
    """Physics constraints for reservoir modeling"""
    
    def __init__(self, viscosity=1e-3):
        self.viscosity = viscosity
    
    def darcy_residual(self, pressure, coords, permeability):
        """
        Compute Darcy's law residual: ∇·(k/μ ∇p) = 0
        
        Args:
            pressure: Pressure predictions [batch_size, 1]
            coords: Spatial coordinates [batch_size, 2] (x, y)
            permeability: Permeability field [batch_size, 1]
        
        Returns:
            PDE residual tensor [batch_size, 1]
        """
        
        # Ensure gradients are enabled
        coords.requires_grad_(True)
        
        # Compute pressure gradients
        dp_dx = torch.autograd.grad(
            outputs=pressure.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]  # ∂p/∂x
        
        dp_dy = torch.autograd.grad(
            outputs=pressure.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]  # ∂p/∂y
        
        # Compute flux components: q = -(k/μ) ∇p
        qx = -(permeability / self.viscosity) * dp_dx
        qy = -(permeability / self.viscosity) * dp_dy
        
        # Compute divergence: ∇·q = ∂qx/∂x + ∂qy/∂y
        dqx_dx = torch.autograd.grad(
            outputs=qx.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        dqy_dy = torch.autograd.grad(
            outputs=qy.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # Continuity equation residual
        residual = dqx_dx + dqy_dy
        
        return residual
    
    def saturation_constraint(self, saturation):
        """Ensure saturation is physically bounded [0, 1]"""
        
        # Penalty for values outside [0, 1]
        lower_violation = torch.relu(-saturation)  # Penalty for S < 0
        upper_violation = torch.relu(saturation - 1.0)  # Penalty for S > 1
        
        return lower_violation + upper_violation
    
    def boundary_conditions(self, predictions, boundary_points, boundary_values):
        """Compute Dirichlet boundary condition residuals"""
        
        # Simple Dirichlet BC: u(boundary) = u_boundary
        bc_residual = predictions - boundary_values
        
        return bc_residual
    
    def compute_physics_loss(self, model, inputs, boundary_data=None):
        """Compute total physics loss"""
        
        # Ensure inputs require gradients
        inputs.requires_grad_(True)
        
        # Forward pass
        predictions = model(inputs)
        pressure = predictions[:, 0:1]
        saturation = predictions[:, 1:2]
        
        # Extract coordinates and properties
        coords = inputs[:, :2]  # Assume first 2 inputs are coordinates
        permeability = inputs[:, 3:4]  # Assume 4th input is permeability
        
        # Darcy's law residual
        darcy_loss = torch.mean(self.darcy_residual(pressure, coords, permeability) ** 2)
        
        # Saturation constraints
        saturation_loss = torch.mean(self.saturation_constraint(saturation) ** 2)
        
        total_physics_loss = darcy_loss + saturation_loss
        
        # Boundary conditions
        boundary_loss = torch.tensor(0.0, device=inputs.device)
        if boundary_data is not None:
            boundary_points, boundary_values = boundary_data
            boundary_pred = model(boundary_points)
            boundary_loss = torch.mean(
                self.boundary_conditions(boundary_pred, boundary_points, boundary_values) ** 2
            )
        
        return {
            'darcy': darcy_loss,
            'saturation': saturation_loss,
            'boundary': boundary_loss,
            'total': total_physics_loss + boundary_loss
        }

# Test physics constraints
def test_physics_constraints():
    """Test physics constraint implementation"""
    
    physics = PhysicsConstraints()
    model = ReservoirPINN(input_dim=4, hidden_dims=[32, 32], output_dim=2)
    
    # Create test data
    batch_size = 100
    inputs = torch.randn(batch_size, 4, requires_grad=True)
    
    # Test physics loss computation
    physics_losses = physics.compute_physics_loss(model, inputs)
    
    print("Physics Loss Components:")
    for key, value in physics_losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # Test gradient flow
    total_loss = physics_losses['total']
    total_loss.backward()
    
    print(f"Physics loss gradient computed successfully")
    
    return physics

physics = test_physics_constraints()
```

### Solution 2.3: Complete PINN Loss Function

```python
class PINNLoss(nn.Module):
    """Complete PINN loss function combining data and physics terms"""
    
    def __init__(self, physics_constraints, loss_weights=None):
        super(PINNLoss, self).__init__()
        
        self.physics = physics_constraints
        self.weights = loss_weights or {
            'data': 1.0,
            'physics': 0.1,
            'boundary': 0.5
        }
        
        # For adaptive weighting
        self.loss_history = {key: [] for key in self.weights.keys()}
        self.adaptive_alpha = 0.9  # Exponential moving average factor
    
    def forward(self, model, inputs, targets, physics_points=None, boundary_data=None):
        """
        Compute complete PINN loss
        
        Args:
            model: PINN model
            inputs: Input features for data loss
            targets: Target values for data loss
            physics_points: Points for physics loss computation
            boundary_data: (boundary_points, boundary_values)
        
        Returns:
            total_loss, loss_components
        """
        
        loss_components = {}
        
        # 1. Data Loss (MSE)
        predictions = model(inputs)
        data_loss = F.mse_loss(predictions, targets)
        loss_components['data'] = data_loss
        
        # 2. Physics Loss
        physics_loss = torch.tensor(0.0, device=inputs.device)
        if physics_points is not None:
            physics_losses = self.physics.compute_physics_loss(model, physics_points, boundary_data)
            physics_loss = physics_losses['total']
            
            # Store individual physics components
            loss_components.update({f'physics_{k}': v for k, v in physics_losses.items()})
        
        loss_components['physics'] = physics_loss
        
        # 3. Boundary Loss (already included in physics loss)
        boundary_loss = torch.tensor(0.0, device=inputs.device)
        if boundary_data is not None:
            boundary_points, boundary_values = boundary_data
            boundary_pred = model(boundary_points)
            boundary_loss = F.mse_loss(boundary_pred, boundary_values)
        
        loss_components['boundary'] = boundary_loss
        
        # 4. Combine with weights
        total_loss = (self.weights['data'] * data_loss +
                     self.weights['physics'] * physics_loss +
                     self.weights['boundary'] * boundary_loss)
        
        loss_components['total'] = total_loss
        
        return total_loss, loss_components
    
    def update_weights(self, loss_components, epoch, update_frequency=100):
        """Adaptive loss weight updating based on loss magnitudes"""
        
        # Record loss history
        for key in self.weights.keys():
            if key in loss_components:
                self.loss_history[key].append(loss_components[key].item())
        
        # Update weights periodically
        if epoch > 0 and epoch % update_frequency == 0:
            self._rebalance_weights()
    
    def _rebalance_weights(self):
        """Rebalance weights based on recent loss history"""
        
        if len(self.loss_history['data']) < 10:
            return
        
        # Calculate recent average losses
        recent_losses = {}
        for key in self.weights.keys():
            if len(self.loss_history[key]) > 0:
                recent_losses[key] = np.mean(self.loss_history[key][-10:])
        
        # Normalize by data loss (reference)
        if 'data' in recent_losses and recent_losses['data'] > 0:
            reference_loss = recent_losses['data']
            
            for key in ['physics', 'boundary']:
                if key in recent_losses and recent_losses[key] > 0:
                    # Adjust weight inversely proportional to loss magnitude
                    ratio = recent_losses[key] / reference_loss
                    new_weight = self.weights[key] / max(ratio, 0.1)
                    
                    # Smooth update using exponential moving average
                    self.weights[key] = (self.adaptive_alpha * self.weights[key] +
                                       (1 - self.adaptive_alpha) * new_weight)
        
        print(f"Updated loss weights: {self.weights}")
    
    def get_weights(self):
        """Get current loss weights"""
        return self.weights.copy()

# Test complete loss function
def test_pinn_loss():
    """Test the complete PINN loss function"""
    
    # Initialize components
    physics = PhysicsConstraints()
    pinn_loss = PINNLoss(physics)
    model = ReservoirPINN(input_dim=4, hidden_dims=[32, 32], output_dim=2)
    
    # Create test data
    batch_size = 64
    inputs = torch.randn(batch_size, 4)
    targets = torch.randn(batch_size, 2)
    physics_points = torch.randn(32, 4, requires_grad=True)
    
    # Boundary data
    boundary_points = torch.randn(16, 4)
    boundary_values = torch.randn(16, 2)
    boundary_data = (boundary_points, boundary_values)
    
    # Compute loss
    total_loss, loss_components = pinn_loss(
        model, inputs, targets, physics_points, boundary_data
    )
    
    print("PINN Loss Components:")
    for key, value in loss_components.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
    
    # Test backward pass
    total_loss.backward()
    print("Backward pass successful")
    
    # Test adaptive weighting
    pinn_loss.update_weights(loss_components, epoch=100)
    
    return pinn_loss

pinn_loss = test_pinn_loss()
```

### Solution 2.4: Training Loop Implementation

```python
def train_pinn(model, train_data, val_data, physics_points, config):
    """
    Complete PINN training with two-phase optimization
    """
    
    # Unpack data
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Initialize components
    physics = PhysicsConstraints()
    pinn_loss = PINNLoss(physics, config['loss_weights'])
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'data_loss': [],
        'physics_loss': [], 'boundary_loss': [], 'lr': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Phase 1: Adam Optimization
    print("Phase 1: Adam Optimization")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=config['lr_adam'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, patience=50, factor=0.5, verbose=True
    )
    
    for epoch in range(config['adam_epochs']):
        # Training step
        model.train()
        optimizer_adam.zero_grad()
        
        # Compute PINN loss
        total_loss, loss_components = pinn_loss(
            model, X_train, y_train, physics_points
        )
        
        # Backward pass with gradient clipping
        total_loss.backward()
        
        # Gradient clipping for stability
        if config.get('gradient_clip'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        
        # NaN detection
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"NaN/Inf detected at epoch {epoch}. Stopping training.")
            break
        
        optimizer_adam.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = F.mse_loss(val_pred, y_val)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer_adam.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(total_loss.item())
        history['val_loss'].append(val_loss.item())
        history['data_loss'].append(loss_components['data'].item())
        history['physics_loss'].append(loss_components['physics'].item())
        history['boundary_loss'].append(loss_components.get('boundary', torch.tensor(0.0)).item())
        history['lr'].append(current_lr)
        
        # Update adaptive weights
        pinn_loss.update_weights(loss_components, epoch)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Progress reporting
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}: "
                  f"Loss={total_loss.item():.6f}, "
                  f"Val={val_loss.item():.6f}, "
                  f"LR={current_lr:.2e}")
    
    # Restore best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val_loss={best_val_loss:.6f})")
    
    # Phase 2: L-BFGS Refinement
    print("\\nPhase 2: L-BFGS Refinement")
    
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=config['lbfgs_iter'],
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss, _ = pinn_loss(model, X_train, y_train, physics_points)
        total_loss.backward()
        return total_loss
    
    # Run L-BFGS optimization
    try:
        optimizer_lbfgs.step(closure)
        
        # Final validation
        model.eval()
        with torch.no_grad():
            final_loss, final_components = pinn_loss(model, X_val, y_val, physics_points)
            print(f"Final validation loss: {final_loss.item():.6f}")
            
    except Exception as e:
        print(f"L-BFGS optimization failed: {e}")
    
    return history

# Training configuration
config = {
    'adam_epochs': 1000,
    'lbfgs_iter': 500,
    'lr_adam': 1e-3,
    'loss_weights': {'data': 1.0, 'physics': 0.1, 'boundary': 0.5},
    'gradient_clip': 1.0,
    'early_stopping_patience': 100
}

# Example usage (with synthetic data)
def create_synthetic_data():
    """Create synthetic data for testing"""
    
    n_train, n_val = 1000, 200
    
    # Features: [x, y, porosity, permeability]
    X_train = torch.randn(n_train, 4)
    X_val = torch.randn(n_val, 4)
    
    # Targets: [pressure, saturation] (synthetic)
    y_train = torch.randn(n_train, 2)
    y_val = torch.randn(n_val, 2)
    
    # Physics points
    physics_points = torch.randn(500, 4, requires_grad=True)
    
    return (X_train, y_train), (X_val, y_val), physics_points

# Test complete training
def test_complete_training():
    """Test the complete training pipeline"""
    
    # Create model and data
    model = ReservoirPINN(input_dim=4, hidden_dims=[64, 64], output_dim=2)
    train_data, val_data, physics_points = create_synthetic_data()
    
    # Run training
    history = train_pinn(model, train_data, val_data, physics_points, config)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss evolution
    epochs = range(len(history['train_loss']))
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Loss components
    axes[1].plot(epochs, history['data_loss'], label='Data Loss')
    axes[1].plot(epochs, history['physics_loss'], label='Physics Loss')
    axes[1].plot(epochs, history['boundary_loss'], label='Boundary Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Component')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return model, history

# Run complete test
# model, history = test_complete_training()
```

## Evaluation Criteria

### Task 2.1 (25 points)
- **Correct architecture implementation** (10 points)
- **Proper weight initialization** (5 points)
- **Working derivative computation** (10 points)

### Task 2.2 (30 points)
- **Correct PDE formulation** (15 points)
- **Proper automatic differentiation** (10 points)
- **Boundary condition handling** (5 points)

### Task 2.3 (25 points)
- **Complete loss function** (10 points)
- **Adaptive weighting implementation** (8 points)
- **Proper loss combination** (7 points)

### Task 2.4 (20 points)
- **Two-phase training implementation** (8 points)
- **Convergence monitoring** (6 points)
- **Proper gradient handling** (6 points)

## Common Mistakes to Avoid

1. **Gradient Issues**: Always use `create_graph=True` for higher-order derivatives
2. **Device Mismatch**: Ensure all tensors are on the same device
3. **NaN Gradients**: Implement proper gradient clipping and NaN detection
4. **Loss Scaling**: Balance loss weights properly to avoid one term dominating
5. **Memory Leaks**: Clear gradients and use `torch.no_grad()` for validation

## Extensions

1. **Advanced Physics**: Implement Buckley-Leverett equation for two-phase flow
2. **Uncertainty Quantification**: Add Bayesian layers for uncertainty estimation
3. **Multi-Scale**: Implement multi-scale physics constraints
4. **Adaptive Sampling**: Smart selection of physics points during training
5. **Transfer Learning**: Pre-train on synthetic data, fine-tune on real data

## Resources

- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [PINN Original Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [Automatic Differentiation Guide](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [L-BFGS Optimization](https://pytorch.org/docs/stable/optim.html#torch.optim.LBFGS)