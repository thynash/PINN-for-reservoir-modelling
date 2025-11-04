"""
Training visualization tools for PINN tutorial system.

Provides real-time training monitoring, learning curves, loss evolution,
and convergence analysis visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from pathlib import Path
import time

from ..core.data_models import TrainingResults, ValidationMetrics, TrainingConfig


class TrainingVisualizer:
    """
    Visualization tools for PINN training monitoring and analysis.
    
    Provides methods for creating learning curves, loss evolution plots,
    convergence analysis, and real-time training monitoring.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the training visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'total': '#1f77b4',
            'data': '#ff7f0e', 
            'pde': '#2ca02c',
            'boundary': '#d62728',
            'validation': '#9467bd'
        }
        self.setup_style()
    
    def setup_style(self):
        """Set up consistent plotting style for training visualizations."""
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10,
            'figure.titlesize': 15,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'lines.linewidth': 2,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def plot_learning_curves(self, 
                           training_results: TrainingResults,
                           log_scale: bool = True,
                           show_phases: bool = True,
                           title: str = "PINN Training Learning Curves",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive learning curves showing loss evolution.
        
        Args:
            training_results: TrainingResults object with loss histories
            log_scale: Whether to use logarithmic scale for y-axis
            show_phases: Whether to highlight optimization phases
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        epochs = range(1, len(training_results.total_loss_history) + 1)
        
        # Main loss curves
        ax_main = fig.add_subplot(gs[0, :])
        
        # Plot loss components
        ax_main.plot(epochs, training_results.total_loss_history, 
                    color=self.colors['total'], label='Total Loss', linewidth=2.5)
        ax_main.plot(epochs, training_results.data_loss_history, 
                    color=self.colors['data'], label='Data Loss', alpha=0.8)
        ax_main.plot(epochs, training_results.pde_loss_history, 
                    color=self.colors['pde'], label='PDE Loss', alpha=0.8)
        ax_main.plot(epochs, training_results.boundary_loss_history, 
                    color=self.colors['boundary'], label='Boundary Loss', alpha=0.8)
        
        if log_scale:
            ax_main.set_yscale('log')
        
        ax_main.set_xlabel('Epoch')
        ax_main.set_ylabel('Loss (log scale)' if log_scale else 'Loss')
        ax_main.set_title('Loss Evolution During Training')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # Highlight optimization phases if requested
        if show_phases and hasattr(training_results, 'optimizer_switch_epoch'):
            switch_epoch = getattr(training_results, 'optimizer_switch_epoch', len(epochs) // 2)
            ax_main.axvline(switch_epoch, color='red', linestyle='--', alpha=0.7, 
                           label='Optimizer Switch (Adam → L-BFGS)')
            ax_main.legend()
            
            # Add phase annotations
            ax_main.annotate('Adam Phase', xy=(switch_epoch/2, max(training_results.total_loss_history)/2),
                           xytext=(switch_epoch/2, max(training_results.total_loss_history)),
                           ha='center', fontsize=10, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            ax_main.annotate('L-BFGS Phase', xy=(switch_epoch + (len(epochs)-switch_epoch)/2, 
                           max(training_results.total_loss_history)/2),
                           xytext=(switch_epoch + (len(epochs)-switch_epoch)/2, 
                           max(training_results.total_loss_history)),
                           ha='center', fontsize=10, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # Loss component ratios
        ax_ratio = fig.add_subplot(gs[1, 0])
        total_losses = np.array(training_results.total_loss_history)
        data_ratios = np.array(training_results.data_loss_history) / total_losses
        pde_ratios = np.array(training_results.pde_loss_history) / total_losses
        boundary_ratios = np.array(training_results.boundary_loss_history) / total_losses
        
        ax_ratio.plot(epochs, data_ratios, color=self.colors['data'], label='Data/Total')
        ax_ratio.plot(epochs, pde_ratios, color=self.colors['pde'], label='PDE/Total')
        ax_ratio.plot(epochs, boundary_ratios, color=self.colors['boundary'], label='Boundary/Total')
        
        ax_ratio.set_xlabel('Epoch')
        ax_ratio.set_ylabel('Loss Ratio')
        ax_ratio.set_title('Loss Component Ratios')
        ax_ratio.legend()
        ax_ratio.grid(True, alpha=0.3)
        
        # Loss smoothing (moving average)
        ax_smooth = fig.add_subplot(gs[1, 1])
        window_size = max(1, len(epochs) // 50)  # Adaptive window size
        
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        if len(epochs) > window_size:
            smooth_epochs = epochs[window_size-1:]
            smooth_total = moving_average(training_results.total_loss_history, window_size)
            ax_smooth.plot(smooth_epochs, smooth_total, color=self.colors['total'], 
                          label=f'Smoothed Total Loss (window={window_size})')
            ax_smooth.set_xlabel('Epoch')
            ax_smooth.set_ylabel('Smoothed Loss')
            ax_smooth.set_title('Smoothed Loss Trend')
            if log_scale:
                ax_smooth.set_yscale('log')
            ax_smooth.legend()
            ax_smooth.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_validation_metrics(self, 
                              training_results: TrainingResults,
                              metrics_to_plot: Optional[List[str]] = None,
                              title: str = "Validation Metrics Evolution",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot validation metrics evolution during training.
        
        Args:
            training_results: TrainingResults object with validation history
            metrics_to_plot: List of metric names to plot
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if not training_results.validation_metrics_history:
            raise ValueError("No validation metrics history available")
        
        if metrics_to_plot is None:
            metrics_to_plot = ['l2_error', 'mean_pde_residual', 'pressure_mae', 'saturation_mae']
        
        # Extract validation data
        validation_epochs = [vm.epoch for vm in training_results.validation_metrics_history]
        
        n_metrics = len(metrics_to_plot)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric_name in enumerate(metrics_to_plot):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Extract metric values
            metric_values = []
            for vm in training_results.validation_metrics_history:
                if hasattr(vm, metric_name):
                    metric_values.append(getattr(vm, metric_name))
                else:
                    metric_values.append(np.nan)
            
            # Plot metric evolution
            ax.plot(validation_epochs, metric_values, 'o-', 
                   color=self.colors['validation'], linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Evolution')
            ax.grid(True, alpha=0.3)
            
            # Add trend line if enough points
            if len(metric_values) > 3:
                z = np.polyfit(validation_epochs, metric_values, 1)
                p = np.poly1d(z)
                ax.plot(validation_epochs, p(validation_epochs), '--', 
                       color='red', alpha=0.7, label='Trend')
                ax.legend()
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_convergence_analysis(self, 
                                training_results: TrainingResults,
                                convergence_window: int = 100,
                                title: str = "Convergence Analysis",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze and visualize training convergence patterns.
        
        Args:
            training_results: TrainingResults object
            convergence_window: Window size for convergence analysis
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)
        
        epochs = np.array(range(1, len(training_results.total_loss_history) + 1))
        total_loss = np.array(training_results.total_loss_history)
        
        # Main convergence plot
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.semilogy(epochs, total_loss, color=self.colors['total'], 
                        linewidth=2, label='Total Loss')
        
        # Calculate and plot convergence metrics
        if len(total_loss) > convergence_window:
            # Loss gradient (rate of change)
            loss_gradient = np.gradient(np.log(total_loss))
            
            # Moving average of gradient
            window_size = min(convergence_window, len(loss_gradient) // 10)
            if window_size > 1:
                smooth_gradient = np.convolve(loss_gradient, np.ones(window_size)/window_size, 
                                            mode='same')
                
                # Find potential convergence point
                convergence_threshold = -1e-4  # Threshold for convergence
                converged_mask = smooth_gradient > convergence_threshold
                if np.any(converged_mask):
                    convergence_epoch = epochs[converged_mask][0]
                    ax_main.axvline(convergence_epoch, color='red', linestyle='--', 
                                   alpha=0.7, label=f'Potential Convergence (Epoch {convergence_epoch})')
        
        ax_main.set_xlabel('Epoch')
        ax_main.set_ylabel('Loss (log scale)')
        ax_main.set_title('Training Convergence Analysis')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # Loss gradient analysis
        ax_grad = fig.add_subplot(gs[1, 0])
        if len(total_loss) > 1:
            loss_gradient = np.gradient(np.log(total_loss))
            ax_grad.plot(epochs, loss_gradient, color='orange', alpha=0.7)
            ax_grad.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax_grad.axhline(-1e-4, color='red', linestyle='--', alpha=0.7, 
                           label='Convergence Threshold')
            ax_grad.set_xlabel('Epoch')
            ax_grad.set_ylabel('Loss Gradient')
            ax_grad.set_title('Loss Rate of Change')
            ax_grad.legend()
            ax_grad.grid(True, alpha=0.3)
        
        # Loss variance analysis
        ax_var = fig.add_subplot(gs[1, 1])
        if len(total_loss) > convergence_window:
            # Rolling variance
            rolling_var = []
            for i in range(convergence_window, len(total_loss)):
                window_data = total_loss[i-convergence_window:i]
                rolling_var.append(np.var(window_data))
            
            var_epochs = epochs[convergence_window:]
            ax_var.semilogy(var_epochs, rolling_var, color='purple', alpha=0.7)
            ax_var.set_xlabel('Epoch')
            ax_var.set_ylabel('Rolling Variance (log scale)')
            ax_var.set_title(f'Loss Variance (window={convergence_window})')
            ax_var.grid(True, alpha=0.3)
        
        # Training phases analysis
        ax_phases = fig.add_subplot(gs[2, :])
        
        # Calculate loss improvement rate
        if len(total_loss) > 10:
            improvement_rate = []
            window = 10
            for i in range(window, len(total_loss)):
                old_loss = np.mean(total_loss[i-window:i-window//2])
                new_loss = np.mean(total_loss[i-window//2:i])
                if old_loss > 0:
                    rate = (old_loss - new_loss) / old_loss
                    improvement_rate.append(rate)
                else:
                    improvement_rate.append(0)
            
            rate_epochs = epochs[window:]
            ax_phases.plot(rate_epochs, improvement_rate, color='green', alpha=0.7)
            ax_phases.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax_phases.set_xlabel('Epoch')
            ax_phases.set_ylabel('Improvement Rate')
            ax_phases.set_title('Loss Improvement Rate')
            ax_phases.grid(True, alpha=0.3)
            
            # Highlight different training phases
            positive_improvement = np.array(improvement_rate) > 0
            if np.any(positive_improvement):
                ax_phases.fill_between(rate_epochs, 0, improvement_rate, 
                                     where=positive_improvement, alpha=0.3, 
                                     color='green', label='Improving')
            if np.any(~positive_improvement):
                ax_phases.fill_between(rate_epochs, 0, improvement_rate, 
                                     where=~positive_improvement, alpha=0.3, 
                                     color='red', label='Degrading')
            ax_phases.legend()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_optimizer_phases(self, 
                            training_results: TrainingResults,
                            config: TrainingConfig,
                            title: str = "Optimizer Phase Analysis",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the different optimization phases (Adam vs L-BFGS).
        
        Args:
            training_results: TrainingResults object
            config: TrainingConfig with optimizer settings
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = np.array(range(1, len(training_results.total_loss_history) + 1))
        total_loss = np.array(training_results.total_loss_history)
        switch_epoch = config.optimizer_switch_epoch
        
        # Phase 1: Adam optimizer
        ax_adam = axes[0, 0]
        adam_epochs = epochs[epochs <= switch_epoch]
        adam_loss = total_loss[:len(adam_epochs)]
        
        if len(adam_loss) > 0:
            ax_adam.semilogy(adam_epochs, adam_loss, color=self.colors['total'], linewidth=2)
            ax_adam.set_xlabel('Epoch')
            ax_adam.set_ylabel('Loss (log scale)')
            ax_adam.set_title(f'Adam Phase (Epochs 1-{switch_epoch})')
            ax_adam.grid(True, alpha=0.3)
            
            # Add learning rate annotation
            ax_adam.text(0.05, 0.95, f'Learning Rate: {config.learning_rate}', 
                        transform=ax_adam.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Phase 2: L-BFGS optimizer
        ax_lbfgs = axes[0, 1]
        lbfgs_epochs = epochs[epochs > switch_epoch]
        lbfgs_loss = total_loss[len(adam_epochs):]
        
        if len(lbfgs_loss) > 0:
            ax_lbfgs.semilogy(lbfgs_epochs, lbfgs_loss, color=self.colors['pde'], linewidth=2)
            ax_lbfgs.set_xlabel('Epoch')
            ax_lbfgs.set_ylabel('Loss (log scale)')
            ax_lbfgs.set_title(f'L-BFGS Phase (Epochs {switch_epoch+1}-{len(epochs)})')
            ax_lbfgs.grid(True, alpha=0.3)
            
            # Add iteration info
            ax_lbfgs.text(0.05, 0.95, f'Max Iterations: {config.lbfgs_iterations}', 
                         transform=ax_lbfgs.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Loss improvement comparison
        ax_comparison = axes[1, 0]
        
        # Calculate improvement rates for each phase
        if len(adam_loss) > 1:
            adam_improvement = (adam_loss[0] - adam_loss[-1]) / adam_loss[0]
            adam_rate = adam_improvement / len(adam_loss)
        else:
            adam_improvement = adam_rate = 0
        
        if len(lbfgs_loss) > 1:
            lbfgs_improvement = (lbfgs_loss[0] - lbfgs_loss[-1]) / lbfgs_loss[0]
            lbfgs_rate = lbfgs_improvement / len(lbfgs_loss)
        else:
            lbfgs_improvement = lbfgs_rate = 0
        
        phases = ['Adam', 'L-BFGS']
        improvements = [adam_improvement * 100, lbfgs_improvement * 100]
        rates = [adam_rate * 100, lbfgs_rate * 100]
        
        x = np.arange(len(phases))
        width = 0.35
        
        bars1 = ax_comparison.bar(x - width/2, improvements, width, 
                                 label='Total Improvement (%)', alpha=0.7)
        bars2 = ax_comparison.bar(x + width/2, rates, width, 
                                 label='Improvement Rate (%/epoch)', alpha=0.7)
        
        ax_comparison.set_xlabel('Optimizer Phase')
        ax_comparison.set_ylabel('Improvement (%)')
        ax_comparison.set_title('Optimization Phase Comparison')
        ax_comparison.set_xticks(x)
        ax_comparison.set_xticklabels(phases)
        ax_comparison.legend()
        ax_comparison.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax_comparison.annotate(f'{height:.2f}%',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax_comparison.annotate(f'{height:.4f}%',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=9)
        
        # Combined view with phase highlighting
        ax_combined = axes[1, 1]
        ax_combined.semilogy(epochs, total_loss, color=self.colors['total'], linewidth=2)
        
        # Highlight phases with different background colors
        ax_combined.axvspan(1, switch_epoch, alpha=0.2, color='blue', label='Adam Phase')
        ax_combined.axvspan(switch_epoch, len(epochs), alpha=0.2, color='green', label='L-BFGS Phase')
        ax_combined.axvline(switch_epoch, color='red', linestyle='--', alpha=0.7, 
                           label='Phase Transition')
        
        ax_combined.set_xlabel('Epoch')
        ax_combined.set_ylabel('Loss (log scale)')
        ax_combined.set_title('Complete Training with Phase Highlighting')
        ax_combined.legend()
        ax_combined.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def create_real_time_monitor(self, 
                               update_interval: float = 1.0,
                               max_points: int = 1000) -> Tuple[plt.Figure, callable]:
        """
        Create a real-time training monitor that can be updated during training.
        
        Args:
            update_interval: Update interval in seconds
            max_points: Maximum number of points to display
            
        Returns:
            Tuple of (figure, update_function)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Real-Time Training Monitor', fontsize=16, fontweight='bold')
        
        # Initialize empty plots
        loss_lines = {}
        metric_lines = {}
        
        # Loss plot
        ax_loss = axes[0, 0]
        loss_lines['total'] = ax_loss.plot([], [], color=self.colors['total'], 
                                          label='Total Loss')[0]
        loss_lines['data'] = ax_loss.plot([], [], color=self.colors['data'], 
                                         label='Data Loss')[0]
        loss_lines['pde'] = ax_loss.plot([], [], color=self.colors['pde'], 
                                        label='PDE Loss')[0]
        loss_lines['boundary'] = ax_loss.plot([], [], color=self.colors['boundary'], 
                                             label='Boundary Loss')[0]
        
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss (log scale)')
        ax_loss.set_title('Loss Evolution')
        ax_loss.set_yscale('log')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        # Gradient norm plot
        ax_grad = axes[0, 1]
        grad_line = ax_grad.plot([], [], color='orange', label='Gradient Norm')[0]
        ax_grad.set_xlabel('Epoch')
        ax_grad.set_ylabel('Gradient Norm')
        ax_grad.set_title('Gradient Magnitude')
        ax_grad.set_yscale('log')
        ax_grad.legend()
        ax_grad.grid(True, alpha=0.3)
        
        # Validation metrics plot
        ax_val = axes[1, 0]
        metric_lines['l2_error'] = ax_val.plot([], [], 'o-', color='purple', 
                                              label='L2 Error')[0]
        ax_val.set_xlabel('Epoch')
        ax_val.set_ylabel('L2 Error')
        ax_val.set_title('Validation Error')
        ax_val.legend()
        ax_val.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax_lr = axes[1, 1]
        lr_line = ax_lr.plot([], [], color='red', label='Learning Rate')[0]
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.set_title('Learning Rate Schedule')
        ax_lr.set_yscale('log')
        ax_lr.legend()
        ax_lr.grid(True, alpha=0.3)
        
        # Data storage for real-time updates
        data_store = {
            'epochs': [],
            'losses': {'total': [], 'data': [], 'pde': [], 'boundary': []},
            'gradients': [],
            'validation_epochs': [],
            'validation_errors': [],
            'learning_rates': []
        }
        
        def update_monitor(epoch: int, 
                          losses: Dict[str, float], 
                          gradient_norm: Optional[float] = None,
                          validation_error: Optional[float] = None,
                          learning_rate: Optional[float] = None):
            """
            Update the real-time monitor with new training data.
            
            Args:
                epoch: Current epoch number
                losses: Dictionary of loss values
                gradient_norm: Current gradient norm
                validation_error: Current validation error
                learning_rate: Current learning rate
            """
            # Update data store
            data_store['epochs'].append(epoch)
            for loss_type, value in losses.items():
                if loss_type in data_store['losses']:
                    data_store['losses'][loss_type].append(value)
            
            if gradient_norm is not None:
                data_store['gradients'].append(gradient_norm)
            
            if validation_error is not None:
                data_store['validation_epochs'].append(epoch)
                data_store['validation_errors'].append(validation_error)
            
            if learning_rate is not None:
                data_store['learning_rates'].append(learning_rate)
            
            # Limit data points
            if len(data_store['epochs']) > max_points:
                data_store['epochs'] = data_store['epochs'][-max_points:]
                for loss_type in data_store['losses']:
                    data_store['losses'][loss_type] = data_store['losses'][loss_type][-max_points:]
                data_store['gradients'] = data_store['gradients'][-max_points:]
                data_store['learning_rates'] = data_store['learning_rates'][-max_points:]
            
            # Update loss plots
            epochs = data_store['epochs']
            for loss_type, line in loss_lines.items():
                if loss_type in data_store['losses'] and data_store['losses'][loss_type]:
                    line.set_data(epochs, data_store['losses'][loss_type])
            
            # Update gradient plot
            if data_store['gradients']:
                grad_line.set_data(epochs[-len(data_store['gradients']):], data_store['gradients'])
            
            # Update validation plot
            if data_store['validation_errors']:
                metric_lines['l2_error'].set_data(data_store['validation_epochs'], 
                                                 data_store['validation_errors'])
            
            # Update learning rate plot
            if data_store['learning_rates']:
                lr_line.set_data(epochs[-len(data_store['learning_rates']):], 
                                data_store['learning_rates'])
            
            # Rescale axes
            for ax in axes.flat:
                ax.relim()
                ax.autoscale_view()
            
            # Refresh display
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        plt.tight_layout()
        plt.ion()  # Turn on interactive mode
        
        return fig, update_monitor
    
    def save_figure(self, fig: plt.Figure, filepath: str, dpi: int = 300):
        """
        Save figure with consistent formatting.
        
        Args:
            fig: matplotlib Figure object
            filepath: Path to save the figure
            dpi: Resolution for saved figure
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)