"""
Convergence Monitoring and Logging for PINN Training

This module implements comprehensive monitoring of training progress, convergence
detection, early stopping mechanisms, and detailed logging of loss components,
gradients, and training statistics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import logging
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

from ..core.data_models import ValidationMetrics, TrainingResults
from ..physics.physics_loss import LossComponents


@dataclass
class ConvergenceConfig:
    """Configuration for convergence monitoring."""
    # Early stopping
    early_stopping_patience: int = 500
    early_stopping_min_delta: float = 1e-6
    early_stopping_metric: str = "total_loss"  # "total_loss", "validation_loss", "pde_residual"
    
    # Convergence detection
    convergence_window: int = 100
    convergence_threshold: float = 1e-5
    convergence_relative_threshold: float = 1e-3
    
    # Gradient monitoring
    gradient_clip_threshold: float = 1.0
    gradient_explosion_threshold: float = 10.0
    gradient_vanishing_threshold: float = 1e-8
    
    # Loss monitoring
    loss_smoothing_window: int = 50
    loss_divergence_threshold: float = 1e6
    
    # Logging frequency
    log_frequency: int = 10
    validation_frequency: int = 100
    checkpoint_frequency: int = 1000
    plot_frequency: int = 500


@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific step."""
    epoch: int
    step: int
    timestamp: float
    
    # Loss components
    total_loss: float
    data_loss: float
    pde_loss: float
    boundary_loss: float
    individual_pde_losses: Dict[str, float]
    
    # Gradient information
    gradient_norm: float
    gradient_max: float
    gradient_mean: float
    
    # Learning rate
    learning_rate: float
    optimizer_phase: str
    
    # Validation metrics (optional)
    validation_metrics: Optional[ValidationMetrics] = None
    
    # Additional info
    batch_size: int = 0
    memory_usage: float = 0.0


class EarlyStoppingMonitor:
    """
    Early stopping monitor with configurable patience and criteria.
    """
    
    def __init__(self, 
                 patience: int = 500,
                 min_delta: float = 1e-6,
                 metric: str = "total_loss",
                 mode: str = "min"):
        """
        Initialize early stopping monitor.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor
            mode: "min" or "max" for improvement direction
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.should_stop = False
        self.best_epoch = 0
        
    def update(self, metrics: TrainingMetrics) -> bool:
        """
        Update early stopping monitor with new metrics.
        
        Args:
            metrics: Current training metrics
            
        Returns:
            True if training should stop
        """
        # Get current value
        if self.metric == "total_loss":
            current_value = metrics.total_loss
        elif self.metric == "validation_loss" and metrics.validation_metrics:
            current_value = metrics.validation_metrics.l2_error
        elif self.metric == "pde_residual" and metrics.validation_metrics:
            current_value = metrics.validation_metrics.mean_pde_residual
        else:
            current_value = metrics.total_loss
        
        # Check for improvement
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.patience_counter = 0
            self.best_epoch = metrics.epoch
        else:
            self.patience_counter += 1
        
        # Check if should stop
        self.should_stop = self.patience_counter >= self.patience
        
        return self.should_stop
    
    def get_state(self) -> Dict[str, Any]:
        """Get early stopping state."""
        return {
            'best_value': self.best_value,
            'patience_counter': self.patience_counter,
            'should_stop': self.should_stop,
            'best_epoch': self.best_epoch
        }


class ConvergenceDetector:
    """
    Detects convergence based on loss stability and gradient norms.
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 loss_threshold: float = 1e-5,
                 gradient_threshold: float = 1e-6,
                 relative_threshold: float = 1e-3):
        """
        Initialize convergence detector.
        
        Args:
            window_size: Window size for stability analysis
            loss_threshold: Absolute loss change threshold
            gradient_threshold: Gradient norm threshold
            relative_threshold: Relative loss change threshold
        """
        self.window_size = window_size
        self.loss_threshold = loss_threshold
        self.gradient_threshold = gradient_threshold
        self.relative_threshold = relative_threshold
        
        self.loss_history = deque(maxlen=window_size)
        self.gradient_history = deque(maxlen=window_size)
        self.converged = False
        
    def update(self, loss: float, gradient_norm: float) -> bool:
        """
        Update convergence detector.
        
        Args:
            loss: Current loss value
            gradient_norm: Current gradient norm
            
        Returns:
            True if converged
        """
        self.loss_history.append(loss)
        self.gradient_history.append(gradient_norm)
        
        if len(self.loss_history) < self.window_size:
            return False
        
        # Check loss stability
        recent_losses = list(self.loss_history)
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Absolute and relative thresholds
        loss_stable = loss_std < self.loss_threshold
        relative_stable = (loss_std / (loss_mean + 1e-8)) < self.relative_threshold
        
        # Check gradient magnitude
        recent_gradients = list(self.gradient_history)
        gradient_mean = np.mean(recent_gradients)
        gradient_small = gradient_mean < self.gradient_threshold
        
        # Convergence criteria
        self.converged = (loss_stable or relative_stable) and gradient_small
        
        return self.converged
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """Get convergence analysis metrics."""
        if len(self.loss_history) == 0:
            return {}
        
        recent_losses = list(self.loss_history)
        recent_gradients = list(self.gradient_history)
        
        return {
            'loss_std': np.std(recent_losses),
            'loss_mean': np.mean(recent_losses),
            'loss_trend': np.polyfit(range(len(recent_losses)), recent_losses, 1)[0],
            'gradient_mean': np.mean(recent_gradients),
            'gradient_std': np.std(recent_gradients),
            'gradient_trend': np.polyfit(range(len(recent_gradients)), recent_gradients, 1)[0],
            'converged': self.converged
        }


class TrainingLogger:
    """
    Comprehensive training logger with file and console output.
    """
    
    def __init__(self, 
                 log_dir: str,
                 experiment_name: str = "pinn_training",
                 log_level: str = "INFO"):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.start_time = time.time()
        
        # Setup logging
        self.logger = logging.getLogger(f"pinn_training_{experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # File handler
        log_file = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        self.metrics_history = []
        
    def log_training_start(self, config: Dict[str, Any]):
        """Log training start with configuration."""
        self.logger.info("="*80)
        self.logger.info(f"Starting PINN training experiment: {self.experiment_name}")
        self.logger.info("="*80)
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
    def log_epoch(self, metrics: TrainingMetrics):
        """Log epoch metrics."""
        elapsed_time = time.time() - self.start_time
        
        log_msg = (
            f"Epoch {metrics.epoch:5d} | "
            f"Loss: {metrics.total_loss:.6e} | "
            f"Data: {metrics.data_loss:.6e} | "
            f"PDE: {metrics.pde_loss:.6e} | "
            f"Boundary: {metrics.boundary_loss:.6e} | "
            f"Grad: {metrics.gradient_norm:.6e} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Time: {elapsed_time:.1f}s"
        )
        
        self.logger.info(log_msg)
        
        # Store metrics
        metrics_dict = asdict(metrics)
        metrics_dict['elapsed_time'] = elapsed_time
        self.metrics_history.append(metrics_dict)
        
        # Save to file periodically
        if len(self.metrics_history) % 10 == 0:
            self._save_metrics()
    
    def log_validation(self, epoch: int, validation_metrics: ValidationMetrics):
        """Log validation results."""
        self.logger.info(
            f"Validation Epoch {epoch} | "
            f"L2 Error: {validation_metrics.l2_error:.6e} | "
            f"MAE: {validation_metrics.mean_absolute_error:.6e} | "
            f"PDE Residual: {validation_metrics.mean_pde_residual:.6e} | "
            f"R²: {validation_metrics.r2_score:.4f}"
        )
    
    def log_convergence(self, epoch: int, convergence_metrics: Dict[str, float]):
        """Log convergence analysis."""
        self.logger.info(
            f"Convergence Analysis Epoch {epoch} | "
            f"Loss Std: {convergence_metrics.get('loss_std', 0):.6e} | "
            f"Grad Mean: {convergence_metrics.get('gradient_mean', 0):.6e} | "
            f"Converged: {convergence_metrics.get('converged', False)}"
        )
    
    def log_early_stopping(self, epoch: int, reason: str):
        """Log early stopping."""
        self.logger.info(f"Early stopping at epoch {epoch}: {reason}")
    
    def log_optimizer_switch(self, epoch: int, old_phase: str, new_phase: str):
        """Log optimizer phase switch."""
        self.logger.info(f"Optimizer switch at epoch {epoch}: {old_phase} -> {new_phase}")
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get complete metrics history."""
        return self.metrics_history.copy()


class ConvergenceMonitor:
    """
    Main convergence monitoring class that orchestrates all monitoring components.
    
    Features:
    - Training progress tracking with comprehensive metrics
    - Early stopping and convergence detection
    - Gradient monitoring and anomaly detection
    - Comprehensive logging with file and console output
    - Real-time plotting and visualization
    """
    
    def __init__(self, 
                 config: Optional[ConvergenceConfig] = None,
                 log_dir: str = "logs",
                 experiment_name: str = "pinn_training"):
        """
        Initialize convergence monitor.
        
        Args:
            config: Convergence monitoring configuration
            log_dir: Directory for logs and outputs
            experiment_name: Name of the experiment
        """
        self.config = config if config is not None else ConvergenceConfig()
        self.experiment_name = experiment_name
        
        # Initialize components
        self.early_stopping = EarlyStoppingMonitor(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            metric=self.config.early_stopping_metric
        )
        
        self.convergence_detector = ConvergenceDetector(
            window_size=self.config.convergence_window,
            loss_threshold=self.config.convergence_threshold,
            relative_threshold=self.config.convergence_relative_threshold
        )
        
        self.logger = TrainingLogger(log_dir, experiment_name)
        
        # State tracking
        self.training_start_time = None
        self.last_log_time = 0
        self.step_count = 0
        self.epoch_count = 0
        
        # Metrics storage
        self.loss_history = defaultdict(list)
        self.gradient_history = []
        self.lr_history = []
        self.validation_history = []
        
        # Anomaly detection
        self.gradient_anomalies = []
        self.loss_anomalies = []
        
    def start_training(self, config: Dict[str, Any]):
        """Initialize training monitoring."""
        self.training_start_time = time.time()
        self.logger.log_training_start(config)
        
    def update(self, 
              loss_components: LossComponents,
              gradient_norm: float,
              learning_rate: float,
              optimizer_phase: str,
              epoch: int,
              validation_metrics: Optional[ValidationMetrics] = None) -> Dict[str, bool]:
        """
        Update monitoring with current training metrics.
        
        Args:
            loss_components: Current loss components
            gradient_norm: Current gradient norm
            learning_rate: Current learning rate
            optimizer_phase: Current optimizer phase
            epoch: Current epoch
            validation_metrics: Optional validation metrics
            
        Returns:
            Dictionary with monitoring decisions
        """
        self.step_count += 1
        self.epoch_count = epoch
        
        # Create training metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            step=self.step_count,
            timestamp=time.time(),
            total_loss=loss_components.total_loss.item(),
            data_loss=loss_components.data_loss.item(),
            pde_loss=sum(loss.item() for loss in loss_components.pde_losses.values()),
            boundary_loss=loss_components.boundary_loss.item(),
            individual_pde_losses={k: v.item() for k, v in loss_components.pde_losses.items()},
            gradient_norm=gradient_norm,
            gradient_max=gradient_norm,  # Simplified
            gradient_mean=gradient_norm,  # Simplified
            learning_rate=learning_rate,
            optimizer_phase=optimizer_phase,
            validation_metrics=validation_metrics
        )
        
        # Update history
        self._update_history(metrics)
        
        # Check for anomalies
        self._check_anomalies(metrics)
        
        # Update early stopping
        should_stop_early = self.early_stopping.update(metrics)
        
        # Update convergence detection
        converged = self.convergence_detector.update(
            metrics.total_loss, 
            metrics.gradient_norm
        )
        
        # Logging
        if self.step_count % self.config.log_frequency == 0:
            self.logger.log_epoch(metrics)
        
        if validation_metrics and self.step_count % self.config.validation_frequency == 0:
            self.logger.log_validation(epoch, validation_metrics)
        
        # Convergence analysis
        if self.step_count % (self.config.log_frequency * 10) == 0:
            convergence_metrics = self.convergence_detector.get_convergence_metrics()
            self.logger.log_convergence(epoch, convergence_metrics)
        
        # Early stopping logging
        if should_stop_early:
            self.logger.log_early_stopping(epoch, "Patience exceeded")
        
        return {
            'should_stop_early': should_stop_early,
            'converged': converged,
            'gradient_anomaly': len(self.gradient_anomalies) > 0,
            'loss_anomaly': len(self.loss_anomalies) > 0
        }
    
    def _update_history(self, metrics: TrainingMetrics):
        """Update metrics history."""
        self.loss_history['total'].append(metrics.total_loss)
        self.loss_history['data'].append(metrics.data_loss)
        self.loss_history['pde'].append(metrics.pde_loss)
        self.loss_history['boundary'].append(metrics.boundary_loss)
        
        for pde_name, pde_loss in metrics.individual_pde_losses.items():
            self.loss_history[f'pde_{pde_name}'].append(pde_loss)
        
        self.gradient_history.append(metrics.gradient_norm)
        self.lr_history.append(metrics.learning_rate)
        
        if metrics.validation_metrics:
            self.validation_history.append(metrics.validation_metrics)
    
    def _check_anomalies(self, metrics: TrainingMetrics):
        """Check for training anomalies."""
        # Gradient anomalies
        if metrics.gradient_norm > self.config.gradient_explosion_threshold:
            anomaly = {
                'type': 'gradient_explosion',
                'epoch': metrics.epoch,
                'value': metrics.gradient_norm,
                'threshold': self.config.gradient_explosion_threshold
            }
            self.gradient_anomalies.append(anomaly)
            self.logger.log_warning(f"Gradient explosion detected: {metrics.gradient_norm:.2e}")
        
        if metrics.gradient_norm < self.config.gradient_vanishing_threshold:
            anomaly = {
                'type': 'gradient_vanishing',
                'epoch': metrics.epoch,
                'value': metrics.gradient_norm,
                'threshold': self.config.gradient_vanishing_threshold
            }
            self.gradient_anomalies.append(anomaly)
            self.logger.log_warning(f"Gradient vanishing detected: {metrics.gradient_norm:.2e}")
        
        # Loss anomalies
        if metrics.total_loss > self.config.loss_divergence_threshold:
            anomaly = {
                'type': 'loss_divergence',
                'epoch': metrics.epoch,
                'value': metrics.total_loss,
                'threshold': self.config.loss_divergence_threshold
            }
            self.loss_anomalies.append(anomaly)
            self.logger.log_warning(f"Loss divergence detected: {metrics.total_loss:.2e}")
    
    def log_optimizer_switch(self, epoch: int, old_phase: str, new_phase: str):
        """Log optimizer phase switch."""
        self.logger.log_optimizer_switch(epoch, old_phase, new_phase)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        elapsed_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': self.epoch_count,
            'total_steps': self.step_count,
            'elapsed_time': elapsed_time,
            'early_stopping_state': self.early_stopping.get_state(),
            'convergence_metrics': self.convergence_detector.get_convergence_metrics(),
            'gradient_anomalies': len(self.gradient_anomalies),
            'loss_anomalies': len(self.loss_anomalies),
            'final_loss': self.loss_history['total'][-1] if self.loss_history['total'] else None,
            'best_validation_loss': min([vm.l2_error for vm in self.validation_history]) if self.validation_history else None
        }
        
        return summary
    
    def create_training_plots(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create training progress plots.
        
        Args:
            save_dir: Directory to save plots (optional)
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        # Loss curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.loss_history['total'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        
        # Loss components
        axes[0, 1].plot(self.loss_history['data'], label='Data')
        axes[0, 1].plot(self.loss_history['pde'], label='PDE')
        axes[0, 1].plot(self.loss_history['boundary'], label='Boundary')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        
        # Gradient norm
        axes[1, 0].plot(self.gradient_history)
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_yscale('log')
        
        # Learning rate
        axes[1, 1].plot(self.lr_history)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        figures['training_curves'] = fig
        
        # Validation metrics
        if self.validation_history:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            l2_errors = [vm.l2_error for vm in self.validation_history]
            mae_errors = [vm.mean_absolute_error for vm in self.validation_history]
            pde_residuals = [vm.mean_pde_residual for vm in self.validation_history]
            r2_scores = [vm.r2_score for vm in self.validation_history]
            
            axes[0, 0].plot(l2_errors)
            axes[0, 0].set_title('L2 Error')
            axes[0, 0].set_xlabel('Validation Step')
            axes[0, 0].set_ylabel('L2 Error')
            axes[0, 0].set_yscale('log')
            
            axes[0, 1].plot(mae_errors)
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].set_xlabel('Validation Step')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].set_yscale('log')
            
            axes[1, 0].plot(pde_residuals)
            axes[1, 0].set_title('PDE Residuals')
            axes[1, 0].set_xlabel('Validation Step')
            axes[1, 0].set_ylabel('Mean PDE Residual')
            axes[1, 0].set_yscale('log')
            
            axes[1, 1].plot(r2_scores)
            axes[1, 1].set_title('R² Score')
            axes[1, 1].set_xlabel('Validation Step')
            axes[1, 1].set_ylabel('R² Score')
            
            plt.tight_layout()
            figures['validation_curves'] = fig
        
        # Save plots if directory provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            for name, fig in figures.items():
                fig.savefig(save_path / f"{name}.png", dpi=300, bbox_inches='tight')
        
        return figures
    
    def export_metrics(self, filepath: str):
        """Export all metrics to file."""
        metrics_data = {
            'loss_history': dict(self.loss_history),
            'gradient_history': self.gradient_history,
            'lr_history': self.lr_history,
            'validation_history': [asdict(vm) for vm in self.validation_history],
            'gradient_anomalies': self.gradient_anomalies,
            'loss_anomalies': self.loss_anomalies,
            'training_summary': self.get_training_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
    
    def reset(self):
        """Reset monitor for new training run."""
        self.early_stopping = EarlyStoppingMonitor(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            metric=self.config.early_stopping_metric
        )
        
        self.convergence_detector = ConvergenceDetector(
            window_size=self.config.convergence_window,
            loss_threshold=self.config.convergence_threshold,
            relative_threshold=self.config.convergence_relative_threshold
        )
        
        self.training_start_time = None
        self.step_count = 0
        self.epoch_count = 0
        
        self.loss_history = defaultdict(list)
        self.gradient_history = []
        self.lr_history = []
        self.validation_history = []
        self.gradient_anomalies = []
        self.loss_anomalies = []


# Factory functions for common configurations

def create_standard_monitor(log_dir: str = "logs", 
                          experiment_name: str = "pinn_training") -> ConvergenceMonitor:
    """Create convergence monitor with standard configuration."""
    config = ConvergenceConfig(
        early_stopping_patience=500,
        convergence_window=100,
        log_frequency=10,
        validation_frequency=100
    )
    return ConvergenceMonitor(config, log_dir, experiment_name)


def create_patient_monitor(log_dir: str = "logs",
                         experiment_name: str = "pinn_training") -> ConvergenceMonitor:
    """Create convergence monitor with high patience for long training."""
    config = ConvergenceConfig(
        early_stopping_patience=1000,
        convergence_window=200,
        log_frequency=50,
        validation_frequency=200
    )
    return ConvergenceMonitor(config, log_dir, experiment_name)


def create_sensitive_monitor(log_dir: str = "logs",
                           experiment_name: str = "pinn_training") -> ConvergenceMonitor:
    """Create convergence monitor with high sensitivity for quick stopping."""
    config = ConvergenceConfig(
        early_stopping_patience=200,
        early_stopping_min_delta=1e-5,
        convergence_window=50,
        convergence_threshold=1e-6,
        log_frequency=5,
        validation_frequency=50
    )
    return ConvergenceMonitor(config, log_dir, experiment_name)