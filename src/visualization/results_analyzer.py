"""
Results analysis and comparison visualization tools for PINN tutorial system.

Provides comprehensive analysis of PINN predictions vs actual measurements,
error distribution visualization, and spatial error mapping.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ..core.data_models import WellData, ValidationMetrics


class ResultsAnalyzer:
    """
    Comprehensive results analysis and visualization tools.
    
    Provides methods for comparing predicted vs actual values, analyzing
    error distributions, creating spatial error maps, and generating
    statistical comparison plots.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the results analyzer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'predicted': '#1f77b4',
            'actual': '#ff7f0e',
            'error': '#d62728',
            'residual': '#2ca02c'
        }
        self.setup_style()
    
    def setup_style(self):
        """Set up consistent plotting style for results analysis."""
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
    
    def plot_prediction_comparison(self, 
                                 actual_values: np.ndarray,
                                 predicted_values: np.ndarray,
                                 variable_name: str = "Variable",
                                 depth: Optional[np.ndarray] = None,
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive predicted vs actual comparison plots.
        
        Args:
            actual_values: Array of actual/measured values
            predicted_values: Array of predicted values
            variable_name: Name of the variable being compared
            depth: Optional depth array for depth-based analysis
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1], 
                     hspace=0.3, wspace=0.3)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
        actual_clean = actual_values[valid_mask]
        predicted_clean = predicted_values[valid_mask]
        
        if depth is not None:
            depth_clean = depth[valid_mask]
        else:
            depth_clean = None
        
        if len(actual_clean) == 0:
            raise ValueError("No valid data points for comparison")
        
        # Calculate statistics
        r2 = r2_score(actual_clean, predicted_clean)
        mae = mean_absolute_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        
        # 1. Scatter plot with 1:1 line
        ax_scatter = fig.add_subplot(gs[0, 0])
        
        if depth_clean is not None:
            scatter = ax_scatter.scatter(actual_clean, predicted_clean, c=depth_clean, 
                                       cmap='viridis', alpha=0.6, s=20)
            cbar = plt.colorbar(scatter, ax=ax_scatter)
            cbar.set_label('Depth (ft)')
        else:
            ax_scatter.scatter(actual_clean, predicted_clean, alpha=0.6, s=20, 
                             color=self.colors['predicted'])
        
        # Add 1:1 line
        min_val = min(actual_clean.min(), predicted_clean.min())
        max_val = max(actual_clean.max(), predicted_clean.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', 
                       alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(actual_clean, predicted_clean, 1)
        p = np.poly1d(z)
        ax_scatter.plot(actual_clean, p(actual_clean), 'g-', alpha=0.8, 
                       linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        
        ax_scatter.set_xlabel(f'Actual {variable_name}')
        ax_scatter.set_ylabel(f'Predicted {variable_name}')
        ax_scatter.set_title('Predicted vs Actual')
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nMAPE = {mape:.2f}%'
        ax_scatter.text(0.05, 0.95, stats_text, transform=ax_scatter.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
        
        # 2. Residual plot
        ax_residual = fig.add_subplot(gs[0, 1])
        residuals = predicted_clean - actual_clean
        
        if depth_clean is not None:
            scatter_res = ax_residual.scatter(actual_clean, residuals, c=depth_clean, 
                                            cmap='viridis', alpha=0.6, s=20)
        else:
            ax_residual.scatter(actual_clean, residuals, alpha=0.6, s=20, 
                              color=self.colors['residual'])
        
        ax_residual.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax_residual.set_xlabel(f'Actual {variable_name}')
        ax_residual.set_ylabel('Residuals (Predicted - Actual)')
        ax_residual.set_title('Residual Plot')
        ax_residual.grid(True, alpha=0.3)
        
        # Add residual statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        ax_residual.text(0.05, 0.95, f'Mean: {residual_mean:.4f}\nStd: {residual_std:.4f}', 
                        transform=ax_residual.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Error distribution
        ax_error_dist = fig.add_subplot(gs[0, 2])
        
        # Histogram of residuals
        ax_error_dist.hist(residuals, bins=50, alpha=0.7, color=self.colors['error'], 
                          density=True, edgecolor='black', linewidth=0.5)
        
        # Overlay normal distribution
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax_error_dist.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                          label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')
        
        ax_error_dist.axvline(0, color='black', linestyle='--', alpha=0.8)
        ax_error_dist.set_xlabel('Residuals')
        ax_error_dist.set_ylabel('Density')
        ax_error_dist.set_title('Error Distribution')
        ax_error_dist.legend()
        ax_error_dist.grid(True, alpha=0.3)
        
        # 4. Depth-based analysis (if depth available)
        if depth_clean is not None:
            # Error vs depth
            ax_depth_error = fig.add_subplot(gs[1, 0])
            abs_errors = np.abs(residuals)
            ax_depth_error.scatter(abs_errors, depth_clean, alpha=0.6, s=20, 
                                 color=self.colors['error'])
            ax_depth_error.set_xlabel('Absolute Error')
            ax_depth_error.set_ylabel('Depth (ft)')
            ax_depth_error.set_title('Error vs Depth')
            ax_depth_error.invert_yaxis()
            ax_depth_error.grid(True, alpha=0.3)
            
            # Depth-binned statistics
            ax_depth_stats = fig.add_subplot(gs[1, 1])
            n_bins = 20
            depth_bins = np.linspace(depth_clean.min(), depth_clean.max(), n_bins + 1)
            depth_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
            
            bin_mae = []
            bin_rmse = []
            bin_counts = []
            
            for i in range(len(depth_bins) - 1):
                mask = (depth_clean >= depth_bins[i]) & (depth_clean < depth_bins[i + 1])
                if np.sum(mask) > 0:
                    bin_residuals = residuals[mask]
                    bin_mae.append(np.mean(np.abs(bin_residuals)))
                    bin_rmse.append(np.sqrt(np.mean(bin_residuals**2)))
                    bin_counts.append(np.sum(mask))
                else:
                    bin_mae.append(np.nan)
                    bin_rmse.append(np.nan)
                    bin_counts.append(0)
            
            ax_depth_stats.plot(bin_mae, depth_centers, 'o-', color='blue', 
                              label='MAE', linewidth=2)
            ax_depth_stats.plot(bin_rmse, depth_centers, 's-', color='red', 
                              label='RMSE', linewidth=2)
            ax_depth_stats.set_xlabel('Error Metric')
            ax_depth_stats.set_ylabel('Depth (ft)')
            ax_depth_stats.set_title('Error Metrics vs Depth')
            ax_depth_stats.invert_yaxis()
            ax_depth_stats.legend()
            ax_depth_stats.grid(True, alpha=0.3)
            
            # Sample count per depth bin
            ax_count = fig.add_subplot(gs[1, 2])
            ax_count.barh(depth_centers, bin_counts, height=np.diff(depth_bins)[0] * 0.8, 
                         alpha=0.7, color='skyblue', edgecolor='black')
            ax_count.set_xlabel('Sample Count')
            ax_count.set_ylabel('Depth (ft)')
            ax_count.set_title('Sample Distribution')
            ax_count.invert_yaxis()
            ax_count.grid(True, alpha=0.3)
        
        # 5. Q-Q plot for normality check
        ax_qq = fig.add_subplot(gs[2, 0])
        stats.probplot(residuals, dist="norm", plot=ax_qq)
        ax_qq.set_title('Q-Q Plot (Normality Check)')
        ax_qq.grid(True, alpha=0.3)
        
        # 6. Percentile error analysis
        ax_percentile = fig.add_subplot(gs[2, 1])
        percentiles = np.arange(0, 101, 5)
        error_percentiles = np.percentile(np.abs(residuals), percentiles)
        
        ax_percentile.plot(percentiles, error_percentiles, 'o-', color='purple', linewidth=2)
        ax_percentile.set_xlabel('Percentile')
        ax_percentile.set_ylabel('Absolute Error')
        ax_percentile.set_title('Error Percentiles')
        ax_percentile.grid(True, alpha=0.3)
        
        # Highlight key percentiles
        key_percentiles = [50, 90, 95, 99]
        for p in key_percentiles:
            if p <= 100:
                error_val = np.percentile(np.abs(residuals), p)
                ax_percentile.axhline(error_val, color='red', linestyle='--', alpha=0.5)
                ax_percentile.text(p, error_val, f'P{p}: {error_val:.3f}', 
                                 verticalalignment='bottom')
        
        # 7. Relative error analysis
        ax_rel_error = fig.add_subplot(gs[2, 2])
        relative_errors = np.abs(residuals) / np.abs(actual_clean) * 100
        
        ax_rel_error.hist(relative_errors, bins=50, alpha=0.7, color='orange', 
                         density=True, edgecolor='black', linewidth=0.5)
        ax_rel_error.set_xlabel('Relative Error (%)')
        ax_rel_error.set_ylabel('Density')
        ax_rel_error.set_title('Relative Error Distribution')
        ax_rel_error.grid(True, alpha=0.3)
        
        # Add median relative error
        median_rel_error = np.median(relative_errors)
        ax_rel_error.axvline(median_rel_error, color='red', linestyle='--', 
                           label=f'Median: {median_rel_error:.2f}%')
        ax_rel_error.legend()
        
        if title is None:
            title = f'{variable_name} Prediction Analysis'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_well_profile_comparison(self, 
                                   well_data: WellData,
                                   predictions: Dict[str, np.ndarray],
                                   variables_to_plot: Optional[List[str]] = None,
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create side-by-side well profile comparisons with statistical annotations.
        
        Args:
            well_data: WellData object with actual measurements
            predictions: Dictionary of variable_name -> predicted_values
            variables_to_plot: List of variables to compare
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if variables_to_plot is None:
            variables_to_plot = list(predictions.keys())
        
        # Filter available variables
        available_vars = [var for var in variables_to_plot 
                         if var in predictions and var in well_data.curves]
        n_vars = len(available_vars)
        
        if n_vars == 0:
            raise ValueError("No valid variables found for comparison")
        
        fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 12), sharey=True)
        if n_vars == 1:
            axes = [axes]
        
        depth = well_data.depth
        
        for i, var_name in enumerate(available_vars):
            ax = axes[i]
            
            actual_values = well_data.curves[var_name]
            predicted_values = predictions[var_name]
            
            # Remove NaN values for plotting
            valid_mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
            plot_depth = depth[valid_mask]
            plot_actual = actual_values[valid_mask]
            plot_predicted = predicted_values[valid_mask]
            
            if len(plot_actual) == 0:
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                continue
            
            # Plot actual and predicted curves
            ax.plot(plot_actual, plot_depth, color=self.colors['actual'], 
                   linewidth=2, label='Actual', alpha=0.8)
            ax.plot(plot_predicted, plot_depth, color=self.colors['predicted'], 
                   linewidth=2, label='Predicted', alpha=0.8)
            
            # Calculate and display statistics
            r2 = r2_score(plot_actual, plot_predicted)
            mae = mean_absolute_error(plot_actual, plot_predicted)
            rmse = np.sqrt(mean_squared_error(plot_actual, plot_predicted))
            
            # Add error shading
            error = np.abs(plot_predicted - plot_actual)
            ax.fill_betweenx(plot_depth, plot_actual - error, plot_actual + error, 
                           alpha=0.2, color=self.colors['error'], label='Error Band')
            
            # Customize axes
            ax.set_xlabel(f'{var_name}')
            if var_name in well_data.metadata.curve_units:
                unit = well_data.metadata.curve_units[var_name]
                ax.set_xlabel(f'{var_name} ({unit})')
            
            ax.invert_yaxis()  # Depth increases downward
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add statistics text box
            stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
            
            # Add depth correlation analysis
            correlation = np.corrcoef(plot_actual, plot_predicted)[0, 1]
            ax.text(0.02, 0.85, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Set depth label on leftmost axis
        axes[0].set_ylabel('Depth (ft)')
        
        if title is None:
            title = f'Well Profile Comparison: {well_data.well_id}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_error_spatial_mapping(self, 
                                 coordinates: np.ndarray,
                                 errors: np.ndarray,
                                 variable_name: str = "Variable",
                                 error_type: str = "Absolute Error",
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create spatial error mapping for multiple wells or spatial locations.
        
        Args:
            coordinates: Array of shape (n_points, 2) with (x, y) coordinates
            errors: Array of error values at each coordinate
            variable_name: Name of the variable
            error_type: Type of error being plotted
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Remove NaN values
        valid_mask = ~np.isnan(errors)
        coords_clean = coordinates[valid_mask]
        errors_clean = errors[valid_mask]
        
        if len(coords_clean) == 0:
            raise ValueError("No valid data points for spatial mapping")
        
        x_coords = coords_clean[:, 0]
        y_coords = coords_clean[:, 1]
        
        # 1. Scatter plot with error coloring
        ax_scatter = axes[0]
        scatter = ax_scatter.scatter(x_coords, y_coords, c=errors_clean, 
                                   cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_scatter)
        cbar.set_label(f'{error_type}')
        
        ax_scatter.set_xlabel('X Coordinate')
        ax_scatter.set_ylabel('Y Coordinate')
        ax_scatter.set_title(f'Spatial Distribution of {error_type}')
        ax_scatter.grid(True, alpha=0.3)
        
        # Add error statistics
        mean_error = np.mean(errors_clean)
        std_error = np.std(errors_clean)
        ax_scatter.text(0.02, 0.98, f'Mean: {mean_error:.3f}\nStd: {std_error:.3f}', 
                       transform=ax_scatter.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Error distribution by spatial regions
        ax_regions = axes[1]
        
        # Divide space into regions for analysis
        n_regions_x = 3
        n_regions_y = 3
        
        x_edges = np.linspace(x_coords.min(), x_coords.max(), n_regions_x + 1)
        y_edges = np.linspace(y_coords.min(), y_coords.max(), n_regions_y + 1)
        
        region_errors = []
        region_labels = []
        
        for i in range(n_regions_x):
            for j in range(n_regions_y):
                x_mask = (x_coords >= x_edges[i]) & (x_coords < x_edges[i + 1])
                y_mask = (y_coords >= y_edges[j]) & (y_coords < y_edges[j + 1])
                region_mask = x_mask & y_mask
                
                if np.sum(region_mask) > 0:
                    region_errors.append(errors_clean[region_mask])
                    region_labels.append(f'R{i+1}{j+1}')
        
        if region_errors:
            # Box plot of errors by region
            ax_regions.boxplot(region_errors, labels=region_labels)
            ax_regions.set_xlabel('Spatial Region')
            ax_regions.set_ylabel(f'{error_type}')
            ax_regions.set_title(f'{error_type} Distribution by Region')
            ax_regions.grid(True, alpha=0.3)
            ax_regions.tick_params(axis='x', rotation=45)
        
        if title is None:
            title = f'Spatial Error Analysis: {variable_name}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_multi_variable_comparison(self, 
                                     actual_data: Dict[str, np.ndarray],
                                     predicted_data: Dict[str, np.ndarray],
                                     title: str = "Multi-Variable Prediction Comparison",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive comparison across multiple variables.
        
        Args:
            actual_data: Dictionary of variable_name -> actual_values
            predicted_data: Dictionary of variable_name -> predicted_values
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Find common variables
        common_vars = set(actual_data.keys()) & set(predicted_data.keys())
        if not common_vars:
            raise ValueError("No common variables found between actual and predicted data")
        
        common_vars = list(common_vars)
        n_vars = len(common_vars)
        
        fig = plt.figure(figsize=(16, 4 * n_vars))
        gs = GridSpec(n_vars, 4, hspace=0.4, wspace=0.3)
        
        summary_stats = []
        
        for i, var_name in enumerate(common_vars):
            actual_values = actual_data[var_name]
            predicted_values = predicted_data[var_name]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
            actual_clean = actual_values[valid_mask]
            predicted_clean = predicted_values[valid_mask]
            
            if len(actual_clean) == 0:
                continue
            
            # Calculate statistics
            r2 = r2_score(actual_clean, predicted_clean)
            mae = mean_absolute_error(actual_clean, predicted_clean)
            rmse = np.sqrt(mean_squared_error(actual_clean, predicted_values))
            correlation = np.corrcoef(actual_clean, predicted_clean)[0, 1]
            
            summary_stats.append({
                'Variable': var_name,
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse,
                'Correlation': correlation,
                'N_points': len(actual_clean)
            })
            
            # 1. Scatter plot
            ax_scatter = fig.add_subplot(gs[i, 0])
            ax_scatter.scatter(actual_clean, predicted_clean, alpha=0.6, s=20)
            
            # Add 1:1 line
            min_val = min(actual_clean.min(), predicted_clean.min())
            max_val = max(actual_clean.max(), predicted_clean.max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax_scatter.set_xlabel(f'Actual {var_name}')
            ax_scatter.set_ylabel(f'Predicted {var_name}')
            ax_scatter.set_title(f'{var_name}: R² = {r2:.3f}')
            ax_scatter.grid(True, alpha=0.3)
            
            # 2. Residual plot
            ax_residual = fig.add_subplot(gs[i, 1])
            residuals = predicted_clean - actual_clean
            ax_residual.scatter(actual_clean, residuals, alpha=0.6, s=20)
            ax_residual.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax_residual.set_xlabel(f'Actual {var_name}')
            ax_residual.set_ylabel('Residuals')
            ax_residual.set_title(f'{var_name}: Residuals')
            ax_residual.grid(True, alpha=0.3)
            
            # 3. Error distribution
            ax_error = fig.add_subplot(gs[i, 2])
            ax_error.hist(residuals, bins=30, alpha=0.7, density=True, 
                         edgecolor='black', linewidth=0.5)
            
            # Overlay normal distribution
            mu, sigma = stats.norm.fit(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            ax_error.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
            
            ax_error.set_xlabel('Residuals')
            ax_error.set_ylabel('Density')
            ax_error.set_title(f'{var_name}: Error Distribution')
            ax_error.grid(True, alpha=0.3)
            
            # 4. Performance metrics
            ax_metrics = fig.add_subplot(gs[i, 3])
            metrics = ['R²', 'MAE', 'RMSE', 'Correlation']
            values = [r2, mae, rmse, correlation]
            
            bars = ax_metrics.bar(metrics, values, alpha=0.7, color=['blue', 'orange', 'red', 'green'])
            ax_metrics.set_ylabel('Metric Value')
            ax_metrics.set_title(f'{var_name}: Performance Metrics')
            ax_metrics.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax_metrics.annotate(f'{value:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3), textcoords="offset points",
                                   ha='center', va='bottom', fontsize=9)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path)
        
        # Print summary statistics
        if summary_stats:
            print("\nSummary Statistics:")
            print("-" * 80)
            df_stats = pd.DataFrame(summary_stats)
            print(df_stats.to_string(index=False, float_format='%.4f'))
        
        return fig
    
    def create_performance_dashboard(self, 
                                   validation_metrics: ValidationMetrics,
                                   variable_names: List[str] = None,
                                   title: str = "PINN Performance Dashboard",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive performance dashboard.
        
        Args:
            validation_metrics: ValidationMetrics object
            variable_names: List of variable names for display
            title: Dashboard title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if variable_names is None:
            variable_names = ['Pressure', 'Saturation']
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        
        # 1. Overall performance metrics
        ax_overall = fig.add_subplot(gs[0, :])
        
        metrics = ['L2 Error', 'MAE', 'RMSE', 'R² Score']
        values = [validation_metrics.l2_error, validation_metrics.mean_absolute_error,
                 validation_metrics.root_mean_square_error, validation_metrics.r2_score]
        
        bars = ax_overall.bar(metrics, values, alpha=0.7, 
                             color=['red', 'orange', 'blue', 'green'])
        ax_overall.set_ylabel('Metric Value')
        ax_overall.set_title('Overall Performance Metrics')
        ax_overall.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax_overall.annotate(f'{value:.4f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=10)
        
        # 2. Physics-informed metrics
        ax_physics = fig.add_subplot(gs[1, 0])
        
        physics_metrics = ['Mean PDE\nResidual', 'Max PDE\nResidual', 'PDE Residual\nStd']
        physics_values = [validation_metrics.mean_pde_residual, 
                         validation_metrics.max_pde_residual,
                         validation_metrics.pde_residual_std]
        
        bars_physics = ax_physics.bar(physics_metrics, physics_values, alpha=0.7, color='purple')
        ax_physics.set_ylabel('Residual Value')
        ax_physics.set_title('Physics Constraint Satisfaction')
        ax_physics.grid(True, alpha=0.3)
        ax_physics.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars_physics, physics_values):
            height = bar.get_height()
            ax_physics.annotate(f'{value:.2e}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
        
        # 3. Variable-specific performance
        ax_var_perf = fig.add_subplot(gs[1, 1])
        
        var_mae = [validation_metrics.pressure_mae, validation_metrics.saturation_mae]
        var_rmse = [validation_metrics.pressure_rmse, validation_metrics.saturation_rmse]
        
        x = np.arange(len(variable_names))
        width = 0.35
        
        bars1 = ax_var_perf.bar(x - width/2, var_mae, width, label='MAE', alpha=0.7)
        bars2 = ax_var_perf.bar(x + width/2, var_rmse, width, label='RMSE', alpha=0.7)
        
        ax_var_perf.set_xlabel('Variables')
        ax_var_perf.set_ylabel('Error Value')
        ax_var_perf.set_title('Variable-Specific Errors')
        ax_var_perf.set_xticks(x)
        ax_var_perf.set_xticklabels(variable_names)
        ax_var_perf.legend()
        ax_var_perf.grid(True, alpha=0.3)
        
        # 4. Training information
        ax_training = fig.add_subplot(gs[1, 2])
        
        training_info = ['Epoch', 'Training Time\n(minutes)', 'Relative Error\n(%)']
        training_values = [validation_metrics.epoch, 
                          validation_metrics.training_time / 60,
                          validation_metrics.relative_error * 100]
        
        bars_training = ax_training.bar(training_info, training_values, alpha=0.7, color='cyan')
        ax_training.set_ylabel('Value')
        ax_training.set_title('Training Information')
        ax_training.grid(True, alpha=0.3)
        ax_training.tick_params(axis='x', rotation=45)
        
        # 5. Performance radar chart
        ax_radar = fig.add_subplot(gs[2, :], projection='polar')
        
        # Normalize metrics for radar chart (0-1 scale)
        radar_metrics = ['Accuracy\n(1-L2)', 'Precision\n(1-MAE)', 'Physics\n(1-PDE)', 
                        'Correlation\n(R²)', 'Efficiency\n(1-RelErr)']
        
        # Normalize values (higher is better)
        radar_values = [
            1 - min(validation_metrics.l2_error, 1),  # Accuracy
            1 - min(validation_metrics.mean_absolute_error, 1),  # Precision
            1 - min(validation_metrics.mean_pde_residual, 1),  # Physics
            max(validation_metrics.r2_score, 0),  # Correlation
            1 - min(validation_metrics.relative_error, 1)  # Efficiency
        ]
        
        # Ensure values are in [0, 1]
        radar_values = [max(0, min(1, val)) for val in radar_values]
        
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
        radar_values += radar_values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        ax_radar.plot(angles, radar_values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax_radar.fill(angles, radar_values, alpha=0.25, color='blue')
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(radar_metrics)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Performance Radar Chart', pad=20)
        ax_radar.grid(True)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
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