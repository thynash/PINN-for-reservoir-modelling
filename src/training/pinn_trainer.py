"""
PINN Training Orchestrator

This module implements the core PINNTrainer class that manages the complete
training workflow for Physics-Informed Neural Networks, including training loops,
validation evaluation, and metrics computation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from ..core.interfaces import TrainerInterface
from ..core.data_models import (
    TrainingConfig, 
    ValidationMetrics, 
    TrainingResults,
    BatchData
)
from ..models.model_interface import PINNModelInterface
from ..physics.physics_loss import PhysicsLossCalculator, LossComponents
from .optimizer_manager import OptimizerManager, create_standard_optimizer_manager
from .batch_processor import BatchProcessor, create_standard_batch_processor
from .convergence_monitor import ConvergenceMonitor, create_standard_monitor


@dataclass
class TrainingState:
    """Container for training state information."""
    epoch: int = 0
    total_loss: float = 0.0
    data_loss: float = 0.0
    pde_loss: float = 0.0
    boundary_loss: float = 0.0
    learning_rate: float = 0.0
    optimizer_phase: str = "adam"
    best_validation_loss: float = float('inf')
    patience_counter: int = 0
    converged: bool = False


class PINNTrainer(TrainerInterface):
    """
    Core training orchestrator for Physics-Informed Neural Networks.
    
    Manages the complete training workflow including:
    - Training loop with proper loss computation and backpropagation
    - Validation evaluation during training with metrics computation
    - Checkpointing and model state management
    - Training progress monitoring and logging
    """
    
    def __init__(self,
                 physics_loss_calculator: PhysicsLossCalculator,
                 optimizer_manager: Optional[OptimizerManager] = None,
                 batch_processor: Optional[BatchProcessor] = None,
                 convergence_monitor: Optional[ConvergenceMonitor] = None,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: Optional[str] = None):
        """
        Initialize PINN trainer.
        
        Args:
            physics_loss_calculator: Physics loss computation engine
            optimizer_manager: Optimizer management system
            batch_processor: Batch processing system
            convergence_monitor: Convergence monitoring system
            device: Device for training computations
            checkpoint_dir: Directory for saving checkpoints
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.physics_loss_calculator = physics_loss_calculator
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training components
        self.optimizer_manager = optimizer_manager or create_standard_optimizer_manager()
        self.batch_processor = batch_processor or create_standard_batch_processor(str(self.device))
        self.convergence_monitor = convergence_monitor or create_standard_monitor()
        
        # Training state
        self.training_state = TrainingState()
        self.training_history = defaultdict(list)
        self.validation_history = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Training components (will be set during training)
        self.model_interface = None
        self.optimizer = None
        self.scheduler = None
        
    def train(self, 
              model: Union[PINNModelInterface, nn.Module],
              dataset: Any,
              config: TrainingConfig) -> TrainingResults:
        """
        Train the PINN model with given dataset and configuration.
        
        Args:
            model: PINN model to train (PINNModelInterface or nn.Module)
            dataset: Training dataset
            config: Training configuration
            
        Returns:
            Training results with loss history and metrics
        """
        self.logger.info("Starting PINN training...")
        start_time = time.time()
        
        # Setup model interface
        if isinstance(model, PINNModelInterface):
            self.model_interface = model
            model = model.get_model()
        else:
            self.model_interface = PINNModelInterface(model=model, device=self.device)
        
        model = model.to(self.device)
        model.train()
        
        # Initialize training state
        self.training_state = TrainingState()
        self.training_history = defaultdict(list)
        self.validation_history = []
        
        # Setup optimizer using OptimizerManager
        self.optimizer = self.optimizer_manager.setup_optimizer(model, "adam", lr=config.learning_rate)
        
        # Start convergence monitoring
        training_config_dict = {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'optimizer_switch_epoch': config.optimizer_switch_epoch
        }
        self.convergence_monitor.start_training(training_config_dict)
        
        # Training loop
        try:
            for epoch in range(config.num_epochs):
                self.training_state.epoch = epoch
                self.batch_processor.update_epoch()
                
                # Training step
                epoch_losses = self._training_step(model, dataset, config, epoch)
                
                # Update training history
                self._update_training_history(epoch_losses)
                
                # Validation step
                validation_metrics = None
                if epoch % config.validation_frequency == 0:
                    validation_metrics = self._validation_step(model, dataset, config)
                    self.validation_history.append(validation_metrics)
                    
                    # Check for improvement
                    if validation_metrics.l2_error < self.training_state.best_validation_loss:
                        self.training_state.best_validation_loss = validation_metrics.l2_error
                        self.training_state.patience_counter = 0
                        
                        # Save best model
                        self._save_best_model(model, epoch, validation_metrics)
                    else:
                        self.training_state.patience_counter += 1
                
                # Compute gradient norm for monitoring
                gradient_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        gradient_norm += param.grad.norm().item() ** 2
                gradient_norm = gradient_norm ** 0.5
                
                # Update convergence monitoring
                monitor_decisions = self.convergence_monitor.update(
                    loss_components=epoch_losses,
                    gradient_norm=gradient_norm,
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    optimizer_phase=self.optimizer_manager.current_phase.value,
                    epoch=epoch,
                    validation_metrics=validation_metrics
                )
                
                # Check for early stopping or convergence
                if monitor_decisions['should_stop_early'] or monitor_decisions['converged']:
                    self.training_state.converged = True
                    break
                
                # Check for optimizer switching
                if self.optimizer_manager.should_switch_optimizer(
                    epoch, epoch_losses.total_loss.item(), gradient_norm
                ):
                    old_phase = self.optimizer_manager.current_phase.value
                    self.optimizer = self.optimizer_manager.switch_optimizer(
                        model, self.optimizer, "lbfgs"
                    )
                    self.convergence_monitor.log_optimizer_switch(
                        epoch, old_phase, self.optimizer_manager.current_phase.value
                    )
                
                # Check L-BFGS stopping condition
                if (self.optimizer_manager.current_phase.value == "lbfgs" and 
                    self.optimizer_manager.should_stop_lbfgs()):
                    self.logger.info(f"L-BFGS phase completed at epoch {epoch}")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        # Finalize training
        total_time = time.time() - start_time
        final_metrics = self.validation_history[-1] if self.validation_history else None
        
        results = TrainingResults(
            total_loss_history=self.training_history['total_loss'],
            data_loss_history=self.training_history['data_loss'],
            pde_loss_history=self.training_history['pde_loss'],
            boundary_loss_history=self.training_history['boundary_loss'],
            validation_metrics_history=self.validation_history,
            final_metrics=final_metrics,
            total_epochs=self.training_state.epoch + 1,
            training_time=total_time,
            convergence_epoch=self.training_state.epoch if self.training_state.converged else None
        )
        
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        return results
    
    def _training_step(self, 
                      model: nn.Module,
                      dataset: Any,
                      config: TrainingConfig,
                      epoch: int) -> LossComponents:
        """
        Perform one training step (epoch).
        
        Args:
            model: Neural network model
            dataset: Training dataset
            config: Training configuration
            
        Returns:
            Loss components for this step
        """
        model.train()
        epoch_losses = []
        
        # Get batches from dataset using BatchProcessor
        batch_data = self._get_training_batch(dataset, config.batch_size)
        
        # Define loss function for optimizer manager
        def loss_fn():
            # Compute physics-informed loss
            loss_components = self.physics_loss_calculator.compute_total_loss(
                model=model,
                data_batch=batch_data['data'],
                physics_batch=batch_data['physics'],
                boundary_batch=batch_data.get('boundary'),
                initial_batch=batch_data.get('initial')
            )
            return loss_components.total_loss
        
        # Perform optimizer step using OptimizerManager
        step_info = self.optimizer_manager.step_optimizer(loss_fn, epoch)
        
        # Get final loss components for monitoring
        loss_components = self.physics_loss_calculator.compute_total_loss(
            model=model,
            data_batch=batch_data['data'],
            physics_batch=batch_data['physics'],
            boundary_batch=batch_data.get('boundary'),
            initial_batch=batch_data.get('initial')
        )
        
        # Update training state
        self.training_state.total_loss = loss_components.total_loss.item()
        self.training_state.data_loss = loss_components.data_loss.item()
        self.training_state.pde_loss = sum(loss.item() for loss in loss_components.pde_losses.values())
        self.training_state.boundary_loss = loss_components.boundary_loss.item()
        self.training_state.learning_rate = self.optimizer.param_groups[0]['lr']
        
        return loss_components
    
    def _validation_step(self,
                        model: nn.Module,
                        dataset: Any,
                        config: TrainingConfig) -> ValidationMetrics:
        """
        Perform validation evaluation.
        
        Args:
            model: Neural network model
            dataset: Dataset (validation split will be used)
            config: Training configuration
            
        Returns:
            Validation metrics
        """
        model.eval()
        
        with torch.no_grad():
            # Get validation batch
            val_batch = self._get_validation_batch(dataset, config.batch_size)
            
            # Forward pass
            inputs = val_batch['inputs'].to(self.device)
            targets = val_batch['targets'].to(self.device)
            predictions = model(inputs)
            
            # Compute validation metrics
            l2_error = torch.mean((predictions - targets) ** 2).item()
            mae = torch.mean(torch.abs(predictions - targets)).item()
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
            
            # Compute R² score
            ss_res = torch.sum((targets - predictions) ** 2).item()
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
            r2_score = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Compute physics residuals on validation points
            val_coords = val_batch.get('coords', inputs)
            val_coords.requires_grad_(True)
            val_predictions = model(val_coords)
            
            # Get PDE residuals
            pde_residuals = self.physics_loss_calculator.compute_physics_loss(
                val_predictions, val_coords, {}
            )
            
            # Compute residual statistics
            all_residuals = []
            for residual in pde_residuals.values():
                all_residuals.extend(residual.detach().cpu().numpy().flatten())
            
            if all_residuals:
                mean_pde_residual = np.mean(np.abs(all_residuals))
                max_pde_residual = np.max(np.abs(all_residuals))
                pde_residual_std = np.std(all_residuals)
            else:
                mean_pde_residual = 0.0
                max_pde_residual = 0.0
                pde_residual_std = 0.0
            
            # Per-output metrics (assuming 2 outputs: pressure, saturation)
            if predictions.shape[1] >= 2:
                pressure_mae = torch.mean(torch.abs(predictions[:, 0] - targets[:, 0])).item()
                pressure_rmse = torch.sqrt(torch.mean((predictions[:, 0] - targets[:, 0]) ** 2)).item()
                saturation_mae = torch.mean(torch.abs(predictions[:, 1] - targets[:, 1])).item()
                saturation_rmse = torch.sqrt(torch.mean((predictions[:, 1] - targets[:, 1]) ** 2)).item()
            else:
                pressure_mae = mae
                pressure_rmse = rmse
                saturation_mae = 0.0
                saturation_rmse = 0.0
            
            # Relative error
            relative_error = torch.mean(torch.abs((predictions - targets) / (targets + 1e-8))).item()
            
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
            epoch=self.training_state.epoch,
            training_time=0.0  # Will be updated by caller
        )
    
    def validate(self, 
                model: Union[PINNModelInterface, nn.Module],
                validation_data: Any) -> ValidationMetrics:
        """
        Validate model performance on validation dataset.
        
        Args:
            model: PINN model to validate
            validation_data: Validation dataset
            
        Returns:
            Validation metrics
        """
        if isinstance(model, PINNModelInterface):
            model = model.get_model()
        
        model = model.to(self.device)
        model.eval()
        
        # Create a dummy config for validation
        config = TrainingConfig(
            input_dim=4,
            hidden_dims=[100, 100, 100],
            output_dim=2,
            batch_size=1024
        )
        
        return self._validation_step(model, validation_data, config)
    
    def save_checkpoint(self, 
                       filepath: str,
                       epoch: int,
                       model_state: Dict[str, Any]) -> None:
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            model_state: Model state dictionary
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'training_state': {
                'total_loss': self.training_state.total_loss,
                'best_validation_loss': self.training_state.best_validation_loss,
                'patience_counter': self.training_state.patience_counter,
                'optimizer_phase': self.training_state.optimizer_phase
            },
            'training_history': dict(self.training_history),
            'validation_history': self.validation_history
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Tuple[int, Dict[str, Any]]:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Tuple of (epoch, model_state_dict)
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore training state
        if 'training_state' in checkpoint:
            state = checkpoint['training_state']
            self.training_state.best_validation_loss = state.get('best_validation_loss', float('inf'))
            self.training_state.patience_counter = state.get('patience_counter', 0)
            self.training_state.optimizer_phase = state.get('optimizer_phase', 'adam')
        
        # Restore history
        if 'training_history' in checkpoint:
            self.training_history = defaultdict(list, checkpoint['training_history'])
        
        if 'validation_history' in checkpoint:
            self.validation_history = checkpoint['validation_history']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
        
        return checkpoint['epoch'], checkpoint['model_state_dict']
    
    def _get_training_batch(self, dataset: Any, batch_size: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get training batch from dataset.
        
        This is a placeholder implementation. In practice, this would be handled
        by the BatchProcessor component.
        """
        # Placeholder implementation - returns dummy batch structure
        batch_data = {
            'data': {
                'inputs': torch.randn(batch_size, 4, device=self.device),
                'targets': torch.randn(batch_size, 2, device=self.device)
            },
            'physics': {
                'coords': torch.randn(batch_size, 4, device=self.device, requires_grad=True),
                'material_properties': {}
            }
        }
        return batch_data
    
    def _get_validation_batch(self, dataset: Any, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get validation batch from dataset.
        
        This is a placeholder implementation.
        """
        return {
            'inputs': torch.randn(batch_size, 4, device=self.device),
            'targets': torch.randn(batch_size, 2, device=self.device),
            'coords': torch.randn(batch_size, 4, device=self.device)
        }
    
    def _update_training_history(self, loss_components: LossComponents):
        """Update training history with current loss components."""
        self.training_history['total_loss'].append(loss_components.total_loss.item())
        self.training_history['data_loss'].append(loss_components.data_loss.item())
        self.training_history['boundary_loss'].append(loss_components.boundary_loss.item())
        
        # Add PDE losses
        pde_total = sum(loss.item() for loss in loss_components.pde_losses.values())
        self.training_history['pde_loss'].append(pde_total)
        
        # Add individual PDE losses
        for pde_name, pde_loss in loss_components.pde_losses.items():
            self.training_history[f'pde_{pde_name}_loss'].append(pde_loss.item())
    
    def _save_best_model(self, model: nn.Module, epoch: int, metrics: ValidationMetrics):
        """Save the best model checkpoint."""
        best_model_path = self.checkpoint_dir / "best_model.pt"
        
        if self.model_interface:
            self.model_interface.save_checkpoint(
                str(best_model_path),
                epoch,
                self.optimizer.state_dict() if self.optimizer else None,
                dict(self.training_history),
                {'validation_metrics': metrics}
            )
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'validation_metrics': metrics
            }, best_model_path)
    
    def _log_training_progress(self, epoch: int, loss_components: LossComponents):
        """Log training progress."""
        self.logger.info(
            f"Epoch {epoch:5d} | "
            f"Total Loss: {loss_components.total_loss.item():.6f} | "
            f"Data Loss: {loss_components.data_loss.item():.6f} | "
            f"PDE Loss: {sum(loss.item() for loss in loss_components.pde_losses.values()):.6f} | "
            f"Boundary Loss: {loss_components.boundary_loss.item():.6f} | "
            f"LR: {self.training_state.learning_rate:.2e}"
        )
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        return {
            'current_epoch': self.training_state.epoch,
            'total_loss': self.training_state.total_loss,
            'best_validation_loss': self.training_state.best_validation_loss,
            'patience_counter': self.training_state.patience_counter,
            'optimizer_phase': self.training_state.optimizer_phase,
            'converged': self.training_state.converged,
            'training_history_length': len(self.training_history['total_loss']),
            'validation_history_length': len(self.validation_history)
        }
    
    def reset_training_state(self):
        """Reset training state for new training run."""
        self.training_state = TrainingState()
        self.training_history = defaultdict(list)
        self.validation_history = []
        
        if self.physics_loss_calculator:
            self.physics_loss_calculator.reset_loss_history()