"""
Scientific plotting utilities for PINN tutorial system.

Provides publication-quality plotting capabilities for well log data,
data distributions, and scientific visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from pathlib import Path

from ..core.data_models import WellData, WellMetadata


class ScientificPlotter:
    """
    Publication-quality scientific plotting utilities.
    
    Provides methods for creating well log visualizations, data distribution plots,
    and other scientific figures with consistent styling and formatting.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the scientific plotter.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 9))
        self.setup_style()
    
    def setup_style(self):
        """Set up consistent plotting style."""
        plt.style.use('default')  # Reset to default first
        
        # Set custom parameters
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def plot_data_distribution(self, 
                             data: Dict[str, np.ndarray], 
                             title: str = "Data Distribution Analysis",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive data distribution plots.
        
        Args:
            data: Dictionary of curve_name -> values
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        n_curves = len(data)
        n_cols = min(3, n_curves)
        n_rows = (n_curves + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        gs = GridSpec(n_rows * 2, n_cols, height_ratios=[3, 1] * n_rows, hspace=0.4, wspace=0.3)
        
        for i, (curve_name, values) in enumerate(data.items()):
            row = (i // n_cols) * 2
            col = i % n_cols
            
            # Remove NaN values for plotting
            clean_values = values[~np.isnan(values)]
            
            if len(clean_values) == 0:
                continue
            
            # Histogram
            ax_hist = fig.add_subplot(gs[row, col])
            ax_hist.hist(clean_values, bins=50, alpha=0.7, color=self.colors[i % len(self.colors)],
                        edgecolor='black', linewidth=0.5)
            ax_hist.set_title(f'{curve_name} Distribution')
            ax_hist.set_xlabel(curve_name)
            ax_hist.set_ylabel('Frequency')
            
            # Add statistics text
            mean_val = np.mean(clean_values)
            std_val = np.std(clean_values)
            ax_hist.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            ax_hist.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.8, label=f'±1σ')
            ax_hist.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.8)
            ax_hist.legend()
            
            # Box plot
            ax_box = fig.add_subplot(gs[row + 1, col])
            bp = ax_box.boxplot(clean_values, vert=False, patch_artist=True)
            bp['boxes'][0].set_facecolor(self.colors[i % len(self.colors)])
            bp['boxes'][0].set_alpha(0.7)
            ax_box.set_xlabel(curve_name)
            ax_box.set_yticklabels([])
            
            # Add outlier count
            q1, q3 = np.percentile(clean_values, [25, 75])
            iqr = q3 - q1
            outliers = clean_values[(clean_values < q1 - 1.5 * iqr) | (clean_values > q3 + 1.5 * iqr)]
            ax_box.text(0.02, 0.98, f'Outliers: {len(outliers)}', transform=ax_box.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_well_profiles(self, 
                          well_data: WellData,
                          curves_to_plot: Optional[List[str]] = None,
                          depth_range: Optional[Tuple[float, float]] = None,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create well profile visualization with multiple curves.
        
        Args:
            well_data: WellData object containing depth and curves
            curves_to_plot: List of curve names to plot (default: all available)
            depth_range: Tuple of (min_depth, max_depth) to plot
            title: Plot title (default: uses well_id)
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if curves_to_plot is None:
            curves_to_plot = list(well_data.curves.keys())
        
        # Filter available curves
        available_curves = [curve for curve in curves_to_plot if curve in well_data.curves]
        n_curves = len(available_curves)
        
        if n_curves == 0:
            raise ValueError("No valid curves found to plot")
        
        # Set up figure with subplots
        fig, axes = plt.subplots(1, n_curves, figsize=(4 * n_curves, 12), sharey=True)
        if n_curves == 1:
            axes = [axes]
        
        # Apply depth range filter
        depth = well_data.depth
        if depth_range:
            mask = (depth >= depth_range[0]) & (depth <= depth_range[1])
            depth = depth[mask]
        else:
            mask = np.ones(len(depth), dtype=bool)
        
        for i, curve_name in enumerate(available_curves):
            ax = axes[i]
            curve_data = well_data.curves[curve_name][mask]
            
            # Remove NaN values for plotting
            valid_mask = ~np.isnan(curve_data)
            plot_depth = depth[valid_mask]
            plot_data = curve_data[valid_mask]
            
            if len(plot_data) == 0:
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                continue
            
            # Plot the curve
            ax.plot(plot_data, plot_depth, color=self.colors[i % len(self.colors)], 
                   linewidth=1.5, label=curve_name)
            
            # Customize axes
            ax.set_xlabel(f'{curve_name}')
            if curve_name in well_data.metadata.curve_units:
                unit = well_data.metadata.curve_units[curve_name]
                ax.set_xlabel(f'{curve_name} ({unit})')
            
            ax.invert_yaxis()  # Depth increases downward
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(plot_data)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            ax.legend()
        
        # Set depth label on leftmost axis
        axes[0].set_ylabel('Depth (ft)')
        
        # Set title
        if title is None:
            title = f'Well Profile: {well_data.well_id}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_correlation_matrix(self, 
                               data: Dict[str, np.ndarray],
                               title: str = "Curve Correlation Matrix",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation matrix heatmap for well log curves.
        
        Args:
            data: Dictionary of curve_name -> values
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Create DataFrame for correlation calculation
        df_data = {}
        for curve_name, values in data.items():
            # Remove NaN values
            clean_values = values[~np.isnan(values)]
            if len(clean_values) > 0:
                df_data[curve_name] = values
        
        df = pd.DataFrame(df_data)
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_depth_correlation(self, 
                              well_data: WellData,
                              curve1: str, 
                              curve2: str,
                              depth_bins: int = 20,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create depth-binned correlation plot between two curves.
        
        Args:
            well_data: WellData object
            curve1: Name of first curve
            curve2: Name of second curve
            depth_bins: Number of depth bins for analysis
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if curve1 not in well_data.curves or curve2 not in well_data.curves:
            raise ValueError(f"Curves {curve1} and/or {curve2} not found in well data")
        
        depth = well_data.depth
        data1 = well_data.curves[curve1]
        data2 = well_data.curves[curve2]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(data1) | np.isnan(data2))
        depth_clean = depth[valid_mask]
        data1_clean = data1[valid_mask]
        data2_clean = data2[valid_mask]
        
        if len(data1_clean) == 0:
            raise ValueError("No valid data points for correlation analysis")
        
        # Create depth bins
        depth_edges = np.linspace(depth_clean.min(), depth_clean.max(), depth_bins + 1)
        depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2
        
        # Calculate correlation for each depth bin
        correlations = []
        counts = []
        
        for i in range(len(depth_edges) - 1):
            mask = (depth_clean >= depth_edges[i]) & (depth_clean < depth_edges[i + 1])
            if np.sum(mask) > 5:  # Need at least 5 points for meaningful correlation
                corr = np.corrcoef(data1_clean[mask], data2_clean[mask])[0, 1]
                correlations.append(corr)
                counts.append(np.sum(mask))
            else:
                correlations.append(np.nan)
                counts.append(0)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])
        
        # Main scatter plot
        ax_main = fig.add_subplot(gs[0, :2])
        scatter = ax_main.scatter(data1_clean, data2_clean, c=depth_clean, 
                                cmap='viridis', alpha=0.6, s=20)
        ax_main.set_xlabel(f'{curve1}')
        ax_main.set_ylabel(f'{curve2}')
        ax_main.set_title(f'{curve1} vs {curve2} (colored by depth)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_main)
        cbar.set_label('Depth (ft)')
        
        # Correlation vs depth
        ax_corr = fig.add_subplot(gs[0, 2])
        valid_corr_mask = ~np.isnan(correlations)
        ax_corr.plot(np.array(correlations)[valid_corr_mask], 
                    depth_centers[valid_corr_mask], 'o-', color='red', linewidth=2)
        ax_corr.set_xlabel('Correlation')
        ax_corr.set_ylabel('Depth (ft)')
        ax_corr.set_title('Correlation vs Depth')
        ax_corr.invert_yaxis()
        ax_corr.grid(True, alpha=0.3)
        ax_corr.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Sample count histogram
        ax_count = fig.add_subplot(gs[1, :])
        ax_count.bar(depth_centers, counts, width=np.diff(depth_edges)[0] * 0.8, 
                    alpha=0.7, color='skyblue', edgecolor='black')
        ax_count.set_xlabel('Depth (ft)')
        ax_count.set_ylabel('Sample Count')
        ax_count.set_title('Sample Count per Depth Bin')
        ax_count.grid(True, alpha=0.3)
        
        if title is None:
            title = f'Depth Correlation Analysis: {well_data.well_id}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_multi_well_comparison(self, 
                                  well_data_list: List[WellData],
                                  curve_name: str,
                                  normalize_depth: bool = True,
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare a specific curve across multiple wells.
        
        Args:
            well_data_list: List of WellData objects
            curve_name: Name of curve to compare
            normalize_depth: Whether to normalize depth to [0, 1] for comparison
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        
        # Individual well profiles
        for i, well_data in enumerate(well_data_list):
            if curve_name not in well_data.curves:
                continue
            
            depth = well_data.depth
            curve_data = well_data.curves[curve_name]
            
            # Remove NaN values
            valid_mask = ~np.isnan(curve_data)
            plot_depth = depth[valid_mask]
            plot_data = curve_data[valid_mask]
            
            if len(plot_data) == 0:
                continue
            
            if normalize_depth:
                # Normalize depth to [0, 1]
                plot_depth = (plot_depth - plot_depth.min()) / (plot_depth.max() - plot_depth.min())
            
            color = self.colors[i % len(self.colors)]
            ax1.plot(plot_data, plot_depth, color=color, alpha=0.7, 
                    linewidth=1.5, label=well_data.well_id)
        
        ax1.set_xlabel(f'{curve_name}')
        ax1.set_ylabel('Normalized Depth' if normalize_depth else 'Depth (ft)')
        ax1.set_title(f'{curve_name} Profiles')
        ax1.invert_yaxis()
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Statistical comparison
        all_values = []
        well_ids = []
        
        for well_data in well_data_list:
            if curve_name in well_data.curves:
                curve_data = well_data.curves[curve_name]
                clean_data = curve_data[~np.isnan(curve_data)]
                if len(clean_data) > 0:
                    all_values.extend(clean_data)
                    well_ids.extend([well_data.well_id] * len(clean_data))
        
        if all_values:
            df = pd.DataFrame({'value': all_values, 'well_id': well_ids})
            
            # Box plot comparison
            sns.boxplot(data=df, x='well_id', y='value', ax=ax2)
            ax2.set_xlabel('Well ID')
            ax2.set_ylabel(f'{curve_name}')
            ax2.set_title(f'{curve_name} Distribution by Well')
            ax2.tick_params(axis='x', rotation=45)
        
        if title is None:
            title = f'Multi-Well Comparison: {curve_name}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
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