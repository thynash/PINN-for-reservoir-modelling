"""
Optimizer Management System for PINN Training

This module implements the OptimizerManager class that handles the two-phase
optimization strategy for PINN training: Adam phase followed by L-BFGS refinement,
with automatic switching and adaptive learning rate scheduling.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.interfaces import OptimizerInterface


class OptimizerPhase(Enum):
    """Enumeration of optimizer phases."""
    ADAM = "adam"
    LBFGS = "lbfgs"
    CUSTOM = "custom"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer setup."""
    # Adam phase configuration
    adam_lr: float = 1e-3
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0
    adam_epochs: int = 3000
    
    # L-BFGS phase configuration
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_max_eval: Optional[int] = None
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_tolerance_change: float = 1e-9
    lbfgs_history_size: int = 100
    lbfgs_iterations: int = 1000
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "step", "exponential", "plateau"
    scheduler_patience: int = 100
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # Automatic switching
    auto_switch: bool = True
    switch_epoch: int = 3000
    switch_criteria: str = "epoch"  # "epoch", "loss_plateau", "gradient_norm"
    loss_plateau_patience: int = 200
    gradient_norm_threshold: float = 1e-4


class LearningRateScheduler:
    """
    Custom learning rate scheduler with multiple strategies.
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 scheduler_type: str = "cosine",
                 **kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler
            **kwargs: Scheduler-specific parameters
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        # Create PyTorch scheduler
        if scheduler_type == "cosine":
            T_max = kwargs.get('T_max', 1000)
            eta_min = kwargs.get('eta_min', 1e-6)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_type == "step":
            step_size = kwargs.get('step_size', 1000)
            gamma = kwargs.get('gamma', 0.5)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "exponential":
            gamma = kwargs.get('gamma', 0.95)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )
        elif scheduler_type == "plateau":
            patience = kwargs.get('patience', 100)
            factor = kwargs.get('factor', 0.5)
            min_lr = kwargs.get('min_lr', 1e-6)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=patience, factor=factor, min_lr=min_lr
            )
        else:
            self.scheduler = None
    
    def step(self, loss: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler is not None:
            if self.scheduler_type == "plateau" and loss is not None:
                self.scheduler.step(loss)
            elif self.scheduler_type != "plateau":
                self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class OptimizerManager(OptimizerInterface):
    """
    Manages the two-phase optimization strategy for PINN training.
    
    Features:
    - Adam phase (1000-5000 epochs, lr=1e-3) for initial training
    - L-BFGS refinement phase (500-1000 iterations) for fine-tuning
    - Automatic switching based on configurable criteria
    - Learning rate scheduling and adaptive optimization strategies
    - Gradient monitoring and convergence detection
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        Initialize optimizer manager.
        
        Args:
            config: Optimizer configuration
        """
        self.config = config if config is not None else OptimizerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_phase = OptimizerPhase.ADAM
        self.current_optimizer = None
        self.scheduler = None
        self.model = None
        
        # Monitoring
        self.loss_history = []
        self.gradient_norms = []
        self.lr_history = []
        self.phase_switch_epoch = None
        
        # L-BFGS specific
        self.lbfgs_iteration = 0
        self.lbfgs_closure = None
    
    def setup_optimizer(self, 
                       model: nn.Module,
                       optimizer_type: str = "adam",
                       **kwargs) -> torch.optim.Optimizer:
        """
        Set up optimizer for training.
        
        Args:
            model: Neural network model
            optimizer_type: Type of optimizer ("adam", "lbfgs")
            **kwargs: Additional optimizer parameters
            
        Returns:
            Configured optimizer
        """
        self.model = model
        
        if optimizer_type.lower() == "adam":
            return self._setup_adam_optimizer(model, **kwargs)
        elif optimizer_type.lower() == "lbfgs":
            return self._setup_lbfgs_optimizer(model, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _setup_adam_optimizer(self, model: nn.Module, **kwargs) -> torch.optim.Adam:
        """Set up Adam optimizer."""
        lr = kwargs.get('lr', self.config.adam_lr)
        betas = kwargs.get('betas', self.config.adam_betas)
        eps = kwargs.get('eps', self.config.adam_eps)
        weight_decay = kwargs.get('weight_decay', self.config.adam_weight_decay)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
        self.current_optimizer = optimizer
        self.current_phase = OptimizerPhase.ADAM
        
        # Setup scheduler
        if self.config.use_scheduler:
            self.scheduler = LearningRateScheduler(
                optimizer,
                scheduler_type=self.config.scheduler_type,
                T_max=self.config.adam_epochs,
                eta_min=self.config.scheduler_min_lr,
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor
            )
        
        self.logger.info(f"Adam optimizer initialized with lr={lr}")
        return optimizer
    
    def _setup_lbfgs_optimizer(self, model: nn.Module, **kwargs) -> torch.optim.LBFGS:
        """Set up L-BFGS optimizer."""
        lr = kwargs.get('lr', self.config.lbfgs_lr)
        max_iter = kwargs.get('max_iter', self.config.lbfgs_max_iter)
        max_eval = kwargs.get('max_eval', self.config.lbfgs_max_eval)
        tolerance_grad = kwargs.get('tolerance_grad', self.config.lbfgs_tolerance_grad)
        tolerance_change = kwargs.get('tolerance_change', self.config.lbfgs_tolerance_change)
        history_size = kwargs.get('history_size', self.config.lbfgs_history_size)
        
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size
        )
        
        self.current_optimizer = optimizer
        self.current_phase = OptimizerPhase.LBFGS
        self.lbfgs_iteration = 0
        
        self.logger.info(f"L-BFGS optimizer initialized with lr={lr}")
        return optimizer
    
    def update_learning_rate(self, 
                           optimizer: torch.optim.Optimizer,
                           new_lr: float) -> None:
        """
        Update optimizer learning rate.
        
        Args:
            optimizer: PyTorch optimizer
            new_lr: New learning rate
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.logger.info(f"Learning rate updated to {new_lr}")
    
    def switch_optimizer(self,
                        model: nn.Module,
                        current_optimizer: torch.optim.Optimizer,
                        new_optimizer_type: str) -> torch.optim.Optimizer:
        """
        Switch to different optimizer during training.
        
        Args:
            model: Neural network model
            current_optimizer: Current optimizer
            new_optimizer_type: New optimizer type
            
        Returns:
            New optimizer
        """
        self.logger.info(f"Switching from {self.current_phase.value} to {new_optimizer_type}")
        
        # Save current state
        if hasattr(current_optimizer, 'state_dict'):
            current_state = current_optimizer.state_dict()
        
        # Create new optimizer
        new_optimizer = self.setup_optimizer(model, new_optimizer_type)
        
        # Record phase switch
        self.phase_switch_epoch = len(self.loss_history)
        
        return new_optimizer
    
    def should_switch_optimizer(self, 
                              epoch: int,
                              loss: float,
                              gradient_norm: Optional[float] = None) -> bool:
        """
        Determine if optimizer should be switched.
        
        Args:
            epoch: Current training epoch
            loss: Current loss value
            gradient_norm: Current gradient norm
            
        Returns:
            True if optimizer should be switched
        """
        if not self.config.auto_switch:
            return False
        
        if self.current_phase == OptimizerPhase.LBFGS:
            return False  # Don't switch from L-BFGS
        
        # Update monitoring
        self.loss_history.append(loss)
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        
        # Check switching criteria
        if self.config.switch_criteria == "epoch":
            return epoch >= self.config.switch_epoch
        
        elif self.config.switch_criteria == "loss_plateau":
            if len(self.loss_history) < self.config.loss_plateau_patience:
                return False
            
            recent_losses = self.loss_history[-self.config.loss_plateau_patience:]
            loss_improvement = (max(recent_losses) - min(recent_losses)) / (max(recent_losses) + 1e-8)
            return loss_improvement < 0.01  # Less than 1% improvement
        
        elif self.config.switch_criteria == "gradient_norm":
            if gradient_norm is not None:
                return gradient_norm < self.config.gradient_norm_threshold
        
        return False
    
    def step_optimizer(self, 
                      loss_fn: Callable[[], torch.Tensor],
                      epoch: int) -> Dict[str, Any]:
        """
        Perform optimizer step with phase-specific logic.
        
        Args:
            loss_fn: Function that computes and returns loss
            epoch: Current epoch
            
        Returns:
            Step information dictionary
        """
        if self.current_phase == OptimizerPhase.ADAM:
            return self._step_adam(loss_fn, epoch)
        elif self.current_phase == OptimizerPhase.LBFGS:
            return self._step_lbfgs(loss_fn, epoch)
        else:
            raise ValueError(f"Unknown optimizer phase: {self.current_phase}")
    
    def _step_adam(self, loss_fn: Callable[[], torch.Tensor], epoch: int) -> Dict[str, Any]:
        """Perform Adam optimizer step."""
        # Compute loss and gradients
        self.current_optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        
        # Gradient clipping for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.current_optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            if self.config.scheduler_type == "plateau":
                self.scheduler.step(loss.item())
            else:
                self.scheduler.step()
        
        # Record learning rate
        current_lr = self.current_optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        return {
            'loss': loss.item(),
            'gradient_norm': grad_norm.item(),
            'learning_rate': current_lr,
            'phase': self.current_phase.value
        }
    
    def _step_lbfgs(self, loss_fn: Callable[[], torch.Tensor], epoch: int) -> Dict[str, Any]:
        """Perform L-BFGS optimizer step."""
        # L-BFGS requires a closure function
        def closure():
            self.current_optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss
        
        # Store closure for potential reuse
        self.lbfgs_closure = closure
        
        # Perform L-BFGS step
        loss = self.current_optimizer.step(closure)
        self.lbfgs_iteration += 1
        
        # Compute gradient norm for monitoring
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        current_lr = self.current_optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        return {
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'gradient_norm': grad_norm,
            'learning_rate': current_lr,
            'phase': self.current_phase.value,
            'lbfgs_iteration': self.lbfgs_iteration
        }
    
    def should_stop_lbfgs(self) -> bool:
        """Check if L-BFGS phase should be stopped."""
        return self.lbfgs_iteration >= self.config.lbfgs_iterations
    
    def get_optimizer_state(self) -> Dict[str, Any]:
        """Get current optimizer state information."""
        state = {
            'current_phase': self.current_phase.value,
            'phase_switch_epoch': self.phase_switch_epoch,
            'loss_history_length': len(self.loss_history),
            'gradient_norms_length': len(self.gradient_norms),
            'lr_history_length': len(self.lr_history)
        }
        
        if self.current_optimizer is not None:
            state['current_lr'] = self.current_optimizer.param_groups[0]['lr']
        
        if self.current_phase == OptimizerPhase.LBFGS:
            state['lbfgs_iteration'] = self.lbfgs_iteration
            state['lbfgs_max_iterations'] = self.config.lbfgs_iterations
        
        return state
    
    def get_convergence_metrics(self, window_size: int = 100) -> Dict[str, float]:
        """
        Get convergence metrics based on recent training history.
        
        Args:
            window_size: Size of window for computing metrics
            
        Returns:
            Dictionary of convergence metrics
        """
        metrics = {}
        
        if len(self.loss_history) >= window_size:
            recent_losses = self.loss_history[-window_size:]
            metrics['loss_mean'] = np.mean(recent_losses)
            metrics['loss_std'] = np.std(recent_losses)
            metrics['loss_trend'] = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        if len(self.gradient_norms) >= window_size:
            recent_grads = self.gradient_norms[-window_size:]
            metrics['grad_norm_mean'] = np.mean(recent_grads)
            metrics['grad_norm_std'] = np.std(recent_grads)
            metrics['grad_norm_trend'] = np.polyfit(range(len(recent_grads)), recent_grads, 1)[0]
        
        if len(self.lr_history) >= window_size:
            recent_lrs = self.lr_history[-window_size:]
            metrics['lr_mean'] = np.mean(recent_lrs)
            metrics['lr_current'] = recent_lrs[-1]
        
        return metrics
    
    def reset_state(self):
        """Reset optimizer manager state."""
        self.current_phase = OptimizerPhase.ADAM
        self.current_optimizer = None
        self.scheduler = None
        self.model = None
        self.loss_history.clear()
        self.gradient_norms.clear()
        self.lr_history.clear()
        self.phase_switch_epoch = None
        self.lbfgs_iteration = 0
        self.lbfgs_closure = None
    
    def save_state(self, filepath: str):
        """Save optimizer manager state."""
        state = {
            'config': self.config,
            'current_phase': self.current_phase.value,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'lr_history': self.lr_history,
            'phase_switch_epoch': self.phase_switch_epoch,
            'lbfgs_iteration': self.lbfgs_iteration
        }
        
        if self.current_optimizer is not None:
            state['optimizer_state_dict'] = self.current_optimizer.state_dict()
        
        if self.scheduler is not None and hasattr(self.scheduler, 'scheduler'):
            state['scheduler_state_dict'] = self.scheduler.scheduler.state_dict()
        
        torch.save(state, filepath)
    
    def load_state(self, filepath: str, model: nn.Module):
        """Load optimizer manager state."""
        state = torch.load(filepath)
        
        self.config = state.get('config', OptimizerConfig())
        self.current_phase = OptimizerPhase(state.get('current_phase', 'adam'))
        self.loss_history = state.get('loss_history', [])
        self.gradient_norms = state.get('gradient_norms', [])
        self.lr_history = state.get('lr_history', [])
        self.phase_switch_epoch = state.get('phase_switch_epoch')
        self.lbfgs_iteration = state.get('lbfgs_iteration', 0)
        
        # Recreate optimizer
        if self.current_phase == OptimizerPhase.ADAM:
            self.setup_optimizer(model, "adam")
        elif self.current_phase == OptimizerPhase.LBFGS:
            self.setup_optimizer(model, "lbfgs")
        
        # Load optimizer state
        if 'optimizer_state_dict' in state and self.current_optimizer is not None:
            self.current_optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in state and self.scheduler is not None:
            self.scheduler.scheduler.load_state_dict(state['scheduler_state_dict'])


# Factory functions for common configurations

def create_standard_optimizer_manager() -> OptimizerManager:
    """Create optimizer manager with standard PINN training configuration."""
    config = OptimizerConfig(
        adam_lr=1e-3,
        adam_epochs=3000,
        lbfgs_iterations=1000,
        auto_switch=True,
        switch_epoch=3000,
        use_scheduler=True,
        scheduler_type="cosine"
    )
    return OptimizerManager(config)


def create_fast_optimizer_manager() -> OptimizerManager:
    """Create optimizer manager for faster training (fewer epochs)."""
    config = OptimizerConfig(
        adam_lr=1e-3,
        adam_epochs=1000,
        lbfgs_iterations=500,
        auto_switch=True,
        switch_epoch=1000,
        use_scheduler=True,
        scheduler_type="step"
    )
    return OptimizerManager(config)


def create_conservative_optimizer_manager() -> OptimizerManager:
    """Create optimizer manager for conservative training (more epochs)."""
    config = OptimizerConfig(
        adam_lr=5e-4,
        adam_epochs=5000,
        lbfgs_iterations=1500,
        auto_switch=True,
        switch_epoch=5000,
        use_scheduler=True,
        scheduler_type="cosine"
    )
    return OptimizerManager(config)