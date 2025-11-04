# PINN Troubleshooting Guide

## Common Training Issues and Solutions

This guide addresses the most common problems encountered when training Physics-Informed Neural Networks for reservoir modeling, along with practical solutions and debugging strategies.

---

## 1. Training Convergence Issues

### Problem: Model Not Learning / Loss Not Decreasing

**Symptoms:**
- Loss remains constant or decreases very slowly
- Validation loss doesn't improve
- Model predictions are poor even after many epochs

**Possible Causes & Solutions:**

#### A. Poor Loss Weight Balance
```python
# Problem: Physics loss dominates data loss (or vice versa)
loss_weights = {'data': 1.0, 'physics': 100.0, 'boundary': 1.0}  # BAD

# Solution: Start with balanced weights
loss_weights = {'data': 1.0, 'physics': 0.1, 'boundary': 0.5}   # GOOD

# Implement adaptive weighting
def adaptive_loss_weighting(data_loss, physics_loss, epoch):
    if epoch < 500:
        return {'data': 1.0, 'physics': 0.01}  # Focus on data first
    else:
        return {'data': 1.0, 'physics': 0.1}   # Gradually increase physics
```

#### B. Learning Rate Issues
```python
# Problem: Learning rate too high or too low
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)  # Too high
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  # Too low

# Solution: Use appropriate learning rate with scheduling
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Good starting point
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=100, factor=0.5, min_lr=1e-6
)
```

#### C. Poor Data Normalization
```python
# Problem: Features have very different scales
depth = [2000, 2500]      # Large values
porosity = [0.1, 0.3]     # Small values

# Solution: Normalize all features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

### Problem: Loss Explodes or Becomes NaN

**Symptoms:**
- Loss suddenly jumps to very large values
- NaN or Inf values in loss
- Gradients become extremely large

**Solutions:**

#### A. Gradient Clipping
```python
# Add gradient clipping to prevent explosion
def train_step(model, optimizer, loss_fn, inputs, targets):
    optimizer.zero_grad()
    loss = loss_fn(model, inputs, targets)
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Check for NaN gradients
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print("NaN gradient detected!")
            return None
    
    optimizer.step()
    return loss
```

#### B. Better Weight Initialization
```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        # Xavier initialization works well for PINNs
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)

model.apply(init_weights)
```

#### C. Numerical Stability in Physics Loss
```python
def stable_darcy_residual(pressure, coords, permeability):
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    
    # Compute gradients
    dp_dx = torch.autograd.grad(pressure.sum(), coords, create_graph=True)[0][:, 0:1]
    
    # Stable flux computation
    flux = -(permeability + eps) * dp_dx
    
    # Check for NaN in intermediate results
    if torch.isnan(flux).any():
        print("NaN detected in flux computation")
        return torch.zeros_like(pressure)
    
    return flux
```

---

## 2. Physics Constraint Issues

### Problem: Physics Loss Not Decreasing

**Symptoms:**
- Data loss decreases but physics loss remains high
- Model violates physical laws
- PDE residuals are large

**Solutions:**

#### A. Check PDE Implementation
```python
def debug_pde_residual(model, inputs):
    """Debug PDE residual computation"""
    
    inputs.requires_grad_(True)
    output = model(inputs)
    pressure = output[:, 0:1]
    
    # Compute gradients step by step
    dp_dx = torch.autograd.grad(
        pressure.sum(), inputs, create_graph=True, retain_graph=True
    )[0][:, 0:1]
    
    print(f"Pressure range: [{pressure.min():.3f}, {pressure.max():.3f}]")
    print(f"Gradient range: [{dp_dx.min():.3f}, {dp_dx.max():.3f}]")
    
    # Check for reasonable values
    if torch.abs(dp_dx).max() > 1000:
        print("WARNING: Very large gradients detected")
    
    return dp_dx
```

#### B. Gradual Physics Introduction
```python
def progressive_physics_training(epoch, max_epochs):
    """Gradually increase physics weight during training"""
    
    if epoch < max_epochs * 0.3:
        physics_weight = 0.01  # Start very low
    elif epoch < max_epochs * 0.6:
        physics_weight = 0.05  # Gradually increase
    else:
        physics_weight = 0.1   # Full physics weight
    
    return physics_weight
```

#### C. Physics Point Selection
```python
def smart_physics_sampling(X_train, n_physics_points=1000):
    """Smart sampling of physics points"""
    
    # Sample from training distribution
    indices = torch.randperm(len(X_train))[:n_physics_points]
    physics_points = X_train[indices].clone()
    
    # Add some random perturbation for better coverage
    noise = 0.1 * torch.randn_like(physics_points)
    physics_points += noise
    
    physics_points.requires_grad_(True)
    return physics_points
```

### Problem: Boundary Conditions Not Satisfied

**Symptoms:**
- Model predictions don't match known boundary values
- Boundary loss remains high
- Unphysical behavior at domain boundaries

**Solutions:**

#### A. Proper Boundary Implementation
```python
def enforce_boundary_conditions(model, boundary_points, boundary_values):
    """Properly enforce boundary conditions"""
    
    # Hard constraint approach
    def modified_forward(x):
        # Get raw network output
        raw_output = model.network(x)
        
        # Identify boundary points (example: x=0 or x=1)
        is_boundary = (torch.abs(x[:, 0]) < 1e-6) | (torch.abs(x[:, 0] - 1.0) < 1e-6)
        
        # Apply boundary conditions
        output = raw_output.clone()
        output[is_boundary] = boundary_values[is_boundary]
        
        return output
    
    return modified_forward
```

#### B. Soft Boundary Enforcement
```python
def boundary_loss_with_weighting(predictions, boundary_values, distance_to_boundary):
    """Weight boundary loss by distance to boundary"""
    
    # Higher weight closer to boundary
    weights = torch.exp(-10 * distance_to_boundary)
    weighted_loss = weights * (predictions - boundary_values) ** 2
    
    return torch.mean(weighted_loss)
```

---

## 3. Data-Related Issues

### Problem: Poor Data Quality Affecting Training

**Symptoms:**
- Inconsistent training behavior
- Model performs well on some wells but poorly on others
- High validation loss despite low training loss

**Solutions:**

#### A. Data Quality Assessment
```python
def assess_data_quality(X, y, well_ids):
    """Comprehensive data quality assessment"""
    
    quality_report = {}
    
    for well_id in np.unique(well_ids):
        well_mask = well_ids == well_id
        well_X = X[well_mask]
        well_y = y[well_mask]
        
        # Check for outliers
        outlier_mask = np.abs(well_X - np.mean(well_X, axis=0)) > 3 * np.std(well_X, axis=0)
        outlier_percentage = np.mean(outlier_mask) * 100
        
        # Check for missing values
        missing_percentage = np.mean(np.isnan(well_X)) * 100
        
        # Check data range
        data_range = np.max(well_X, axis=0) - np.min(well_X, axis=0)
        
        quality_report[well_id] = {
            'outliers': outlier_percentage,
            'missing': missing_percentage,
            'range': data_range,
            'n_samples': len(well_X)
        }
        
        # Flag problematic wells
        if outlier_percentage > 10 or missing_percentage > 20:
            print(f"WARNING: Well {well_id} has quality issues")
    
    return quality_report
```

#### B. Robust Data Preprocessing
```python
def robust_preprocessing(X, method='quantile'):
    """Robust preprocessing for noisy data"""
    
    if method == 'quantile':
        # Use quantile-based normalization (robust to outliers)
        q25, q75 = np.percentile(X, [25, 75], axis=0)
        X_normalized = (X - q25) / (q75 - q25 + 1e-8)
    
    elif method == 'robust_scaler':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_normalized = scaler.fit_transform(X)
    
    return X_normalized
```

### Problem: Data Leakage Between Train/Validation

**Symptoms:**
- Very low validation loss that doesn't reflect real performance
- Model fails on truly unseen data
- Overly optimistic performance metrics

**Solutions:**

#### A. Proper Well-Based Splitting
```python
def well_based_split(X, y, well_ids, train_ratio=0.7, val_ratio=0.15):
    """Split data at well level to prevent leakage"""
    
    unique_wells = np.unique(well_ids)
    n_wells = len(unique_wells)
    
    # Split wells, not samples
    n_train = int(n_wells * train_ratio)
    n_val = int(n_wells * val_ratio)
    
    np.random.shuffle(unique_wells)
    
    train_wells = unique_wells[:n_train]
    val_wells = unique_wells[n_train:n_train + n_val]
    test_wells = unique_wells[n_train + n_val:]
    
    # Create masks
    train_mask = np.isin(well_ids, train_wells)
    val_mask = np.isin(well_ids, val_wells)
    test_mask = np.isin(well_ids, test_wells)
    
    return {
        'train': (X[train_mask], y[train_mask]),
        'val': (X[val_mask], y[val_mask]),
        'test': (X[test_mask], y[test_mask])
    }
```

---

## 4. Model Architecture Issues

### Problem: Model Too Simple or Too Complex

**Symptoms:**
- Underfitting: High bias, poor performance on both train and validation
- Overfitting: Low training loss, high validation loss

**Solutions:**

#### A. Architecture Guidelines for PINNs
```python
def design_pinn_architecture(input_dim, output_dim, problem_complexity='medium'):
    """Design PINN architecture based on problem complexity"""
    
    if problem_complexity == 'simple':
        hidden_dims = [32, 32]
    elif problem_complexity == 'medium':
        hidden_dims = [64, 64, 64]
    elif problem_complexity == 'complex':
        hidden_dims = [128, 128, 128, 128]
    
    model = PINNArchitecture(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation='tanh'  # Smooth activation for physics
    )
    
    return model
```

#### B. Regularization for PINNs
```python
class RegularizedPINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            
            # Add dropout only if specified (use sparingly in PINNs)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # L2 regularization through weight decay in optimizer
    
    def forward(self, x):
        return self.network(x)

# Use L2 regularization instead of dropout
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

---

## 5. Computational Issues

### Problem: Training Too Slow

**Symptoms:**
- Each epoch takes very long time
- GPU utilization is low
- Memory usage is inefficient

**Solutions:**

#### A. Efficient Batch Processing
```python
def efficient_batch_training(model, dataloader, physics_points, loss_fn):
    """Efficient batch processing for PINN training"""
    
    model.train()
    total_loss = 0
    
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        # Use subset of physics points for each batch
        n_physics_batch = min(len(physics_points), len(batch_X))
        physics_batch = physics_points[torch.randperm(len(physics_points))[:n_physics_batch]]
        
        # Compute loss
        loss, _ = loss_fn(model, batch_X, batch_y, physics_batch)
        
        # Accumulate gradients for larger effective batch size
        loss = loss / len(dataloader)  # Scale by number of batches
        loss.backward()
        
        total_loss += loss.item()
    
    return total_loss
```

#### B. Memory Optimization
```python
def memory_efficient_physics_loss(model, physics_points, chunk_size=100):
    """Compute physics loss in chunks to save memory"""
    
    total_physics_loss = 0
    n_chunks = len(physics_points) // chunk_size + 1
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(physics_points))
        
        if start_idx >= len(physics_points):
            break
        
        chunk = physics_points[start_idx:end_idx]
        chunk_loss = compute_physics_residual(model, chunk)
        total_physics_loss += chunk_loss * len(chunk)
    
    return total_physics_loss / len(physics_points)
```

### Problem: GPU Memory Issues

**Symptoms:**
- CUDA out of memory errors
- Training crashes with large datasets
- Inconsistent memory usage

**Solutions:**

#### A. Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientPINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        # ... initialize layers ...
    
    def forward(self, x):
        # Use gradient checkpointing for memory efficiency
        return checkpoint(self._forward_impl, x)
    
    def _forward_impl(self, x):
        # Actual forward implementation
        for layer in self.layers:
            x = layer(x)
        return x
```

#### B. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

def mixed_precision_training(model, optimizer, loss_fn, inputs, targets):
    """Use mixed precision to reduce memory usage"""
    
    scaler = GradScaler()
    
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast():
        loss, _ = loss_fn(model, inputs, targets)
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss
```

---

## 6. Debugging Strategies

### General Debugging Workflow

1. **Start Simple**: Begin with a minimal working example
2. **Isolate Components**: Test each component (data, model, physics) separately
3. **Monitor Everything**: Track all loss components, gradients, and activations
4. **Visualize**: Plot training curves, predictions, and residuals
5. **Compare Baselines**: Test against known solutions or simpler models

### Debugging Tools

#### A. Loss Component Monitoring
```python
class LossMonitor:
    def __init__(self):
        self.history = defaultdict(list)
    
    def log(self, epoch, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.history[key].append(value)
    
    def plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(self.history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_yscale('log')
        
        # Loss components
        axes[0, 1].plot(self.history['data_loss'], label='Data')
        axes[0, 1].plot(self.history['physics_loss'], label='Physics')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # Gradients
        if 'grad_norm' in self.history:
            axes[1, 0].plot(self.history['grad_norm'])
            axes[1, 0].set_title('Gradient Norm')
        
        # Learning rate
        if 'lr' in self.history:
            axes[1, 1].plot(self.history['lr'])
            axes[1, 1].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.show()
```

#### B. Gradient Analysis
```python
def analyze_gradients(model):
    """Analyze gradient statistics"""
    
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            print(f"{name}: grad_norm={param_norm:.6f}, "
                  f"param_norm={param.data.norm(2):.6f}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.6f}")
    
    return total_norm
```

#### C. Physics Residual Visualization
```python
def visualize_physics_residuals(model, test_points):
    """Visualize physics residuals spatially"""
    
    model.eval()
    with torch.no_grad():
        residuals = compute_physics_residual(model, test_points)
    
    # Extract coordinates
    x_coords = test_points[:, 0].cpu().numpy()
    y_coords = test_points[:, 1].cpu().numpy()
    residual_values = residuals.cpu().numpy()
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_coords, y_coords, c=residual_values, 
                         cmap='RdBu', s=20, alpha=0.7)
    plt.colorbar(scatter, label='PDE Residual')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Physics Residual Distribution')
    plt.show()
    
    print(f"Residual statistics:")
    print(f"  Mean: {np.mean(residual_values):.6f}")
    print(f"  Std:  {np.std(residual_values):.6f}")
    print(f"  Max:  {np.max(np.abs(residual_values)):.6f}")
```

---

## 7. Performance Optimization

### Training Speed Optimization

1. **Use appropriate batch sizes** (256-1024 for most problems)
2. **Implement efficient data loading** with PyTorch DataLoader
3. **Use GPU acceleration** when available
4. **Profile your code** to identify bottlenecks

### Memory Optimization

1. **Use gradient checkpointing** for large models
2. **Implement mixed precision training**
3. **Clear unnecessary gradients** with `optimizer.zero_grad()`
4. **Use `torch.no_grad()`** during validation

### Numerical Stability

1. **Implement gradient clipping**
2. **Use stable numerical methods** for physics computations
3. **Monitor for NaN/Inf values**
4. **Add small epsilon values** to prevent division by zero

---

## 8. When to Seek Help

If you've tried the solutions above and still have issues:

1. **Check the literature** for similar problems and solutions
2. **Post on forums** (PyTorch forums, Stack Overflow, Reddit)
3. **Consult domain experts** for physics-related issues
4. **Review recent PINN papers** for advanced techniques
5. **Consider simpler alternatives** if PINNs aren't working

Remember: PINNs are still an active research area, and some problems may not have established solutions yet. Don't hesitate to experiment and contribute to the community!

---

## Quick Reference Checklist

### Before Training
- [ ] Data is properly normalized
- [ ] Train/val/test splits are at well level
- [ ] Physics equations are correctly implemented
- [ ] Loss weights are reasonable
- [ ] Model architecture is appropriate

### During Training
- [ ] Monitor all loss components
- [ ] Check for NaN/Inf values
- [ ] Verify gradient norms are reasonable
- [ ] Ensure physics loss is decreasing
- [ ] Validate on held-out data

### After Training
- [ ] Test on completely unseen data
- [ ] Verify physics constraints are satisfied
- [ ] Compare with baseline methods
- [ ] Analyze failure cases
- [ ] Document lessons learned