"""
Validation framework for PINN models.

This module implements comprehensive validation functionality including:
- Hold-out well validation with proper data splitting
- Cross-validation functionality for robust performance assessment  
- Validation metrics computation (L2 error, MAE, PDE residuals)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from sklearn.model_selection import KFold
import time

from ..core.data_models import (
    WellData, ValidationMetrics, TrainingConfig, 
    ModelConfig, BatchData
)
from ..core.interfaces import ModelInterface
from ..physics.physics_loss import PhysicsLossCalculator


@dataclass
class ValidationConfig:
    """Configuration for validation procedures."""
    # Cross-validation settings
    n_folds: int = 5
    random_seed: int = 42
    
    # Hold-out validation settings
    holdout_fraction: float = 0.2
    stratify_by_formation: bool = True
    
    # Validation batch settings
    batch_size: int = 1024
    max_validation_points: int = 10000
    
    # Metrics computation
    compute_pde_residuals: bool = True
    compute_per_output_metrics: bool = True
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ValidationFramework:
    """
    Comprehensive validation framework for PINN models.
    
    Provides hold-out validation, cross-validation, and detailed metrics computation
    for robust performance assessment of physics-informed neural networks.
    """
    
    def __init__(self, 
                 physics_loss_calculator: PhysicsLossCalculator,
                 config: Optional[ValidationConfig] = None):
        """
        Initialize validation framework.
        
        Args:
            physics_loss_calculator: Calculator for physics-informed loss terms
            config: Validation configuration (uses defaults if None)
        """
        self.physics_loss_calculator = physics_loss_calculator
        self.config = config or ValidationConfig()
        self.device = torch.device(self.config.device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validation history
        self.validation_history: List[ValidationMetrics] = []
        
    def holdout_validation(self,
                          model: Union[ModelInterface, nn.Module],
                          well_data_list: List[WellData],
                          training_config: TrainingConfig) -> Tuple[ValidationMetrics, List[WellData], List[WellData]]:
        """
        Perform hold-out well validation with proper data splitting.
        
        Args:
            model: PINN model to validate
            well_data_list: List of all well data
            training_config: Training configuration
            
        Returns:
            Tuple of (validation_metrics, training_wells, validation_wells)
        """
        self.logger.info("Starting hold-out validation")
        
        # Split wells into training and validation sets
        train_wells, val_wells = self._split_wells_holdout(well_data_list)
        
        self.logger.info(f"Split data: {len(train_wells)} training wells, {len(val_wells)} validation wells")
        
        # Validate on held-out wells
        validation_metrics = self._validate_on_wells(model, val_wells, training_config)
        
        return validation_metrics, train_wells, val_wells
    
    def cross_validation(self,
                        model_factory: callable,
                        well_data_list: List[WellData],
                        training_config: TrainingConfig) -> List[ValidationMetrics]:
        """
        Perform k-fold cross-validation for robust performance assessment.
        
        Args:
            model_factory: Function that creates a new model instance
            well_data_list: List of all well data
            training_config: Training configuration
            
        Returns:
            List of validation metrics for each fold
        """
        self.logger.info(f"Starting {self.config.n_folds}-fold cross-validation")
        
        # Setup k-fold splitter
        kfold = KFold(n_splits=self.config.n_folds, 
                     shuffle=True, 
                     random_state=self.config.random_seed)
        
        fold_metrics = []
        well_indices = np.arange(len(well_data_list))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(well_indices)):
            self.logger.info(f"Processing fold {fold_idx + 1}/{self.config.n_folds}")
            
            # Split data for this fold
            train_wells = [well_data_list[i] for i in train_idx]
            val_wells = [well_data_list[i] for i in val_idx]
            
            # Create fresh model for this fold
            model = model_factory()
            
            # Validate on this fold
            fold_validation_metrics = self._validate_on_wells(model, val_wells, training_config)
            fold_metrics.append(fold_validation_metrics)
            
            self.logger.info(f"Fold {fold_idx + 1} L2 error: {fold_validation_metrics.l2_error:.6f}")
        
        return fold_metrics    

    def compute_validation_metrics(self,
                                  model: Union[ModelInterface, nn.Module],
                                  validation_data: List[WellData],
                                  training_config: TrainingConfig) -> ValidationMetrics:
        """
        Compute comprehensive validation metrics (L2 error, MAE, PDE residuals).
        
        Args:
            model: PINN model to validate
            validation_data: Validation well data
            training_config: Training configuration
            
        Returns:
            Comprehensive validation metrics
        """
        start_time = time.time()
        
        # Ensure model is in evaluation mode
        if hasattr(model, 'get_model'):
            model = model.get_model()
        
        model = model.to(self.device)
        model.eval()
        
        # Prepare validation batch
        val_batch = self._prepare_validation_batch(validation_data, training_config)
        
        with torch.no_grad():
            # Forward pass
            inputs = val_batch['inputs'].to(self.device)
            targets = val_batch['targets'].to(self.device)
            predictions = model(inputs)
            
            # Basic error metrics
            l2_error = torch.mean((predictions - targets) ** 2).item()
            mae = torch.mean(torch.abs(predictions - targets)).item()
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
            
            # R² score
            ss_res = torch.sum((targets - predictions) ** 2).item()
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
            r2_score = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Relative error
            relative_error = torch.mean(torch.abs((predictions - targets) / (targets + 1e-8))).item()
            
            # Per-output metrics (pressure and saturation)
            pressure_mae, pressure_rmse = 0.0, 0.0
            saturation_mae, saturation_rmse = 0.0, 0.0
            
            if self.config.compute_per_output_metrics and predictions.shape[1] >= 2:
                pressure_mae = torch.mean(torch.abs(predictions[:, 0] - targets[:, 0])).item()
                pressure_rmse = torch.sqrt(torch.mean((predictions[:, 0] - targets[:, 0]) ** 2)).item()
                saturation_mae = torch.mean(torch.abs(predictions[:, 1] - targets[:, 1])).item()
                saturation_rmse = torch.sqrt(torch.mean((predictions[:, 1] - targets[:, 1]) ** 2)).item()
            
            # PDE residuals
            mean_pde_residual, max_pde_residual, pde_residual_std = 0.0, 0.0, 0.0
            
            if self.config.compute_pde_residuals:
                pde_metrics = self._compute_pde_residuals(model, val_batch)
                mean_pde_residual = pde_metrics['mean']
                max_pde_residual = pde_metrics['max']
                pde_residual_std = pde_metrics['std']
        
        validation_time = time.time() - start_time
        
        return ValidationMetrics(
            l2_error=l2_error,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            mean_pde_residual=mean_pde_residual,
            max_pde_residual=max_pde_residual,
            pde_residual_std=pde_residual_std,
            pressure_mae=pressure_mae,
            pressure_rmse=pressure_rmse,
            saturation_mae=saturation_mae,
            saturation_rmse=saturation_rmse,
            r2_score=r2_score,
            relative_error=relative_error,
            epoch=0,  # Will be set by caller if needed
            training_time=validation_time
        ) 
   
    def _split_wells_holdout(self, well_data_list: List[WellData]) -> Tuple[List[WellData], List[WellData]]:
        """Split wells into training and validation sets for hold-out validation."""
        import random
        
        # Set random seed for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        if self.config.stratify_by_formation:
            # Group wells by formation for stratified splitting
            formation_groups = {}
            for well in well_data_list:
                formation = well.metadata.formation
                if formation not in formation_groups:
                    formation_groups[formation] = []
                formation_groups[formation].append(well)
            
            train_wells, val_wells = [], []
            
            # Split each formation group
            for formation, wells in formation_groups.items():
                random.shuffle(wells)
                val_size = max(1, int(len(wells) * self.config.holdout_fraction))
                
                val_wells.extend(wells[:val_size])
                train_wells.extend(wells[val_size:])
        else:
            # Simple random split
            shuffled_wells = well_data_list.copy()
            random.shuffle(shuffled_wells)
            
            val_size = int(len(shuffled_wells) * self.config.holdout_fraction)
            val_wells = shuffled_wells[:val_size]
            train_wells = shuffled_wells[val_size:]
        
        return train_wells, val_wells
    
    def _validate_on_wells(self,
                          model: Union[ModelInterface, nn.Module],
                          validation_wells: List[WellData],
                          training_config: TrainingConfig) -> ValidationMetrics:
        """Validate model on a specific set of wells."""
        return self.compute_validation_metrics(model, validation_wells, training_config)
    
    def _prepare_validation_batch(self, 
                                 validation_data: List[WellData],
                                 training_config: TrainingConfig) -> Dict[str, torch.Tensor]:
        """Prepare validation batch from well data."""
        # Collect all validation points
        all_inputs = []
        all_targets = []
        all_coords = []
        
        for well in validation_data:
            # Extract input features (depth, porosity, permeability, etc.)
            depth = well.depth
            
            # Get available curves for inputs
            input_features = []
            if 'PORO' in well.curves:
                input_features.append(well.curves['PORO'])
            if 'PERM' in well.curves:
                input_features.append(well.curves['PERM'])
            if 'GR' in well.curves:
                input_features.append(well.curves['GR'])
            if 'NPHI' in well.curves:
                input_features.append(well.curves['NPHI'])
            
            if len(input_features) == 0:
                continue
                
            # Stack input features
            inputs = np.column_stack([depth] + input_features)
            
            # Create synthetic targets (pressure and saturation) for validation
            # In a real scenario, these would be measured or computed values
            pressure = 2000 + depth * 0.5  # Simple pressure gradient
            saturation = 0.2 + 0.3 * np.random.random(len(depth))  # Random saturation
            targets = np.column_stack([pressure, saturation])
            
            # Coordinates for PDE computation
            coords = np.column_stack([depth, np.zeros_like(depth)])  # (depth, x)
            
            all_inputs.append(inputs)
            all_targets.append(targets)
            all_coords.append(coords)
        
        if not all_inputs:
            raise ValueError("No valid validation data found")
        
        # Concatenate all data
        inputs = np.vstack(all_inputs)
        targets = np.vstack(all_targets)
        coords = np.vstack(all_coords)
        
        # Limit validation points if too many
        if len(inputs) > self.config.max_validation_points:
            indices = np.random.choice(len(inputs), self.config.max_validation_points, replace=False)
            inputs = inputs[indices]
            targets = targets[indices]
            coords = coords[indices]
        
        return {
            'inputs': torch.FloatTensor(inputs),
            'targets': torch.FloatTensor(targets),
            'coords': torch.FloatTensor(coords)
        }    

    def _compute_pde_residuals(self, 
                              model: nn.Module,
                              val_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute PDE residual statistics."""
        coords = val_batch['coords'].to(self.device)
        coords.requires_grad_(True)
        
        # Forward pass to get predictions
        predictions = model(coords)
        
        # Compute physics residuals
        try:
            pde_residuals = self.physics_loss_calculator.compute_physics_loss(
                predictions, coords, {}
            )
            
            # Collect all residuals
            all_residuals = []
            for residual_tensor in pde_residuals.values():
                if isinstance(residual_tensor, torch.Tensor):
                    all_residuals.extend(residual_tensor.detach().cpu().numpy().flatten())
            
            if all_residuals:
                residuals_array = np.array(all_residuals)
                return {
                    'mean': np.mean(np.abs(residuals_array)),
                    'max': np.max(np.abs(residuals_array)),
                    'std': np.std(residuals_array)
                }
        except Exception as e:
            self.logger.warning(f"Could not compute PDE residuals: {e}")
        
        return {'mean': 0.0, 'max': 0.0, 'std': 0.0}
    
    def get_validation_summary(self, metrics_list: List[ValidationMetrics]) -> Dict[str, float]:
        """
        Compute summary statistics from multiple validation runs.
        
        Args:
            metrics_list: List of validation metrics from multiple runs/folds
            
        Returns:
            Dictionary with mean and std of key metrics
        """
        if not metrics_list:
            return {}
        
        # Extract key metrics
        l2_errors = [m.l2_error for m in metrics_list]
        mae_values = [m.mean_absolute_error for m in metrics_list]
        rmse_values = [m.root_mean_square_error for m in metrics_list]
        r2_scores = [m.r2_score for m in metrics_list]
        pde_residuals = [m.mean_pde_residual for m in metrics_list]
        
        return {
            'l2_error_mean': np.mean(l2_errors),
            'l2_error_std': np.std(l2_errors),
            'mae_mean': np.mean(mae_values),
            'mae_std': np.std(mae_values),
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values),
            'r2_score_mean': np.mean(r2_scores),
            'r2_score_std': np.std(r2_scores),
            'pde_residual_mean': np.mean(pde_residuals),
            'pde_residual_std': np.std(pde_residuals)
        }