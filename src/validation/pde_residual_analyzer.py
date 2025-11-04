"""
PDE residual analysis tools for PINN validation.

This module implements tools for:
- PDE residual distribution calculation for held-out points
- Residual visualization and statistical analysis
- Physics constraint violation detection and reporting
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..core.data_models import WellData, ValidationMetrics
from ..core.interfaces import ModelInterface
from ..physics.physics_loss import PhysicsLossCalculator


@dataclass
class ResidualAnalysisConfig:
    """Configuration for PDE residual analysis."""
    # Analysis settings
    n_test_points: int = 5000
    residual_threshold: float = 1e-2
    violation_threshold: float = 1e-1
    
    # Visualization settings
    create_plots: bool = True
    save_plots: bool = False
    plot_dir: str = "validation_plots"
    
    # Statistical analysis
    perform_normality_test: bool = True
    confidence_level: float = 0.95
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ResidualAnalysisResults:
    """Results from PDE residual analysis."""
    # Residual statistics
    residual_statistics: Dict[str, Dict[str, float]]
    
    # Violation analysis
    violation_count: int
    violation_percentage: float
    violation_locations: np.ndarray
    
    # Distribution analysis
    normality_test_results: Dict[str, Dict[str, float]]
    
    # Spatial analysis
    spatial_residual_map: Optional[np.ndarray] = None
    
    # Plots (if generated)
    residual_plots: Optional[Dict[str, plt.Figure]] = None


class PDEResidualAnalyzer:
    """
    Comprehensive PDE residual analysis for PINN validation.
    
    Analyzes physics constraint satisfaction and residual distributions
    to assess model performance on governing equations.
    """
    
    def __init__(self,
                 physics_loss_calculator: PhysicsLossCalculator,
                 config: Optional[ResidualAnalysisConfig] = None):
        """
        Initialize PDE residual analyzer.
        
        Args:
            physics_loss_calculator: Calculator for physics-informed loss terms
            config: Analysis configuration (uses defaults if None)
        """
        self.physics_loss_calculator = physics_loss_calculator
        self.config = config or ResidualAnalysisConfig()
        self.device = torch.device(self.config.device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Analysis history
        self.analysis_history: List[ResidualAnalysisResults] = []    

    def analyze_residuals(self,
                         model: Union[ModelInterface, nn.Module],
                         test_wells: List[WellData]) -> ResidualAnalysisResults:
        """
        Perform comprehensive PDE residual analysis on held-out test points.
        
        Args:
            model: PINN model to analyze
            test_wells: Test well data for residual computation
            
        Returns:
            Comprehensive residual analysis results
        """
        self.logger.info("Starting PDE residual analysis")
        
        # Ensure model is in evaluation mode
        if hasattr(model, 'get_model'):
            model = model.get_model()
        
        model = model.to(self.device)
        model.eval()
        
        # Generate test points from held-out wells
        test_points = self._generate_test_points(test_wells)
        
        # Compute PDE residuals
        residual_data = self._compute_detailed_residuals(model, test_points)
        
        # Perform statistical analysis
        residual_statistics = self._compute_residual_statistics(residual_data)
        
        # Analyze constraint violations
        violation_analysis = self._analyze_constraint_violations(residual_data)
        
        # Perform normality tests
        normality_results = {}
        if self.config.perform_normality_test:
            normality_results = self._perform_normality_tests(residual_data)
        
        # Create spatial residual map
        spatial_map = self._create_spatial_residual_map(test_points, residual_data)
        
        # Generate visualization plots
        plots = {}
        if self.config.create_plots:
            plots = self._create_residual_plots(residual_data, test_points)
        
        results = ResidualAnalysisResults(
            residual_statistics=residual_statistics,
            violation_count=violation_analysis['count'],
            violation_percentage=violation_analysis['percentage'],
            violation_locations=violation_analysis['locations'],
            normality_test_results=normality_results,
            spatial_residual_map=spatial_map,
            residual_plots=plots
        )
        
        self.analysis_history.append(results)
        return results
    
    def detect_physics_violations(self,
                                 model: Union[ModelInterface, nn.Module],
                                 test_points: np.ndarray,
                                 violation_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect and report physics constraint violations.
        
        Args:
            model: PINN model to analyze
            test_points: Test points for violation detection
            violation_threshold: Threshold for violation detection
            
        Returns:
            Dictionary with violation analysis results
        """
        threshold = violation_threshold or self.config.violation_threshold
        
        # Ensure model is in evaluation mode
        if hasattr(model, 'get_model'):
            model = model.get_model()
        
        model = model.to(self.device)
        model.eval()
        
        # Convert test points to tensor
        test_tensor = torch.FloatTensor(test_points).to(self.device)
        test_tensor.requires_grad_(True)
        
        # Compute residuals
        with torch.no_grad():
            predictions = model(test_tensor)
        
        # Get PDE residuals
        residuals = self.physics_loss_calculator.compute_physics_loss(
            predictions, test_tensor, {}
        )
        
        violations = {}
        total_violations = 0
        
        for pde_name, residual_tensor in residuals.items():
            residual_values = residual_tensor.detach().cpu().numpy().flatten()
            
            # Find violations
            violation_mask = np.abs(residual_values) > threshold
            violation_indices = np.where(violation_mask)[0]
            
            violations[pde_name] = {
                'count': len(violation_indices),
                'percentage': (len(violation_indices) / len(residual_values)) * 100,
                'max_violation': np.max(np.abs(residual_values)),
                'violation_indices': violation_indices,
                'violation_values': residual_values[violation_mask]
            }
            
            total_violations += len(violation_indices)
        
        return {
            'total_violations': total_violations,
            'total_percentage': (total_violations / len(test_points)) * 100,
            'per_pde_violations': violations,
            'threshold_used': threshold
        }    

    def _generate_test_points(self, test_wells: List[WellData]) -> np.ndarray:
        """Generate test points from held-out wells for residual analysis."""
        all_points = []
        
        for well in test_wells:
            depth = well.depth
            
            # Get available input features
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
            
            # Create input points
            points = np.column_stack([depth] + input_features)
            all_points.append(points)
        
        if not all_points:
            raise ValueError("No valid test points found in wells")
        
        # Concatenate and sample if needed
        all_points = np.vstack(all_points)
        
        if len(all_points) > self.config.n_test_points:
            indices = np.random.choice(len(all_points), self.config.n_test_points, replace=False)
            all_points = all_points[indices]
        
        return all_points
    
    def _compute_detailed_residuals(self, 
                                   model: nn.Module,
                                   test_points: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute detailed PDE residuals for all test points."""
        test_tensor = torch.FloatTensor(test_points).to(self.device)
        test_tensor.requires_grad_(True)
        
        # Forward pass
        predictions = model(test_tensor)
        
        # Compute PDE residuals
        residuals = self.physics_loss_calculator.compute_physics_loss(
            predictions, test_tensor, {}
        )
        
        # Convert to numpy arrays
        residual_data = {}
        for pde_name, residual_tensor in residuals.items():
            residual_data[pde_name] = residual_tensor.detach().cpu().numpy().flatten()
        
        return residual_data
    
    def _compute_residual_statistics(self, 
                                    residual_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute comprehensive statistics for each PDE residual."""
        statistics = {}
        
        for pde_name, residuals in residual_data.items():
            abs_residuals = np.abs(residuals)
            
            statistics[pde_name] = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'mean_abs': np.mean(abs_residuals),
                'max_abs': np.max(abs_residuals),
                'median': np.median(residuals),
                'q25': np.percentile(residuals, 25),
                'q75': np.percentile(residuals, 75),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            }
        
        return statistics
    
    def _analyze_constraint_violations(self, 
                                     residual_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze physics constraint violations across all PDEs."""
        all_residuals = []
        violation_locations = []
        
        for pde_name, residuals in residual_data.items():
            all_residuals.extend(residuals.flatten())
            
            # Find violation locations for this PDE
            violations = np.where(np.abs(residuals) > self.config.violation_threshold)[0]
            violation_locations.extend(violations)
        
        violation_count = len([r for r in all_residuals if abs(r) > self.config.violation_threshold])
        violation_percentage = (violation_count / len(all_residuals)) * 100 if all_residuals else 0
        
        return {
            'count': violation_count,
            'percentage': violation_percentage,
            'locations': np.array(violation_locations)
        }  
  
    def _perform_normality_tests(self, 
                                residual_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Perform normality tests on residual distributions."""
        normality_results = {}
        
        for pde_name, residuals in residual_data.items():
            # Shapiro-Wilk test (for smaller samples)
            if len(residuals) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(residuals, 'norm', 
                                        args=(np.mean(residuals), np.std(residuals)))
            
            # Anderson-Darling test
            ad_stat, ad_critical, ad_significance = stats.anderson(residuals, dist='norm')
            
            normality_results[pde_name] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'anderson_statistic': ad_stat,
                'anderson_critical_values': ad_critical.tolist(),
                'anderson_significance_levels': ad_significance.tolist()
            }
        
        return normality_results
    
    def _create_spatial_residual_map(self, 
                                    test_points: np.ndarray,
                                    residual_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Create spatial map of residual magnitudes."""
        # Combine all residuals into a single magnitude
        combined_residuals = np.zeros(len(test_points))
        
        for residuals in residual_data.values():
            combined_residuals += np.abs(residuals)
        
        # Create spatial map (depth vs residual magnitude)
        depths = test_points[:, 0]  # Assuming first column is depth
        
        # Create 2D map
        spatial_map = np.column_stack([depths, combined_residuals])
        
        return spatial_map
    
    def _create_residual_plots(self, 
                              residual_data: Dict[str, np.ndarray],
                              test_points: np.ndarray) -> Dict[str, plt.Figure]:
        """Create comprehensive residual visualization plots."""
        plots = {}
        
        # Set style for scientific plots
        plt.style.use('seaborn-v0_8')
        
        # 1. Residual distribution plots
        fig_dist, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (pde_name, residuals) in enumerate(residual_data.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Histogram with normal overlay
            ax.hist(residuals, bins=50, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black')
            
            # Overlay normal distribution
            mu, sigma = np.mean(residuals), np.std(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal fit')
            
            ax.set_title(f'{pde_name} Residual Distribution')
            ax.set_xlabel('Residual Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['residual_distributions'] = fig_dist
        
        # 2. Q-Q plots for normality assessment
        fig_qq, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (pde_name, residuals) in enumerate(residual_data.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title(f'{pde_name} Q-Q Plot')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['qq_plots'] = fig_qq
        
        # 3. Spatial residual plot
        fig_spatial, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        depths = test_points[:, 0]
        combined_residuals = np.zeros(len(test_points))
        for residuals in residual_data.values():
            combined_residuals += np.abs(residuals)
        
        scatter = ax.scatter(depths, combined_residuals, 
                           c=combined_residuals, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Depth')
        ax.set_ylabel('Combined Residual Magnitude')
        ax.set_title('Spatial Distribution of PDE Residuals')
        plt.colorbar(scatter, ax=ax, label='Residual Magnitude')
        ax.grid(True, alpha=0.3)
        
        plots['spatial_residuals'] = fig_spatial
        
        return plots
    
    def generate_residual_report(self, 
                               analysis_results: ResidualAnalysisResults) -> str:
        """Generate a comprehensive text report of residual analysis."""
        report = []
        report.append("=" * 60)
        report.append("PDE RESIDUAL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall violation summary
        report.append(f"CONSTRAINT VIOLATION SUMMARY:")
        report.append(f"Total violations: {analysis_results.violation_count}")
        report.append(f"Violation percentage: {analysis_results.violation_percentage:.2f}%")
        report.append("")
        
        # Per-PDE statistics
        report.append("PDE RESIDUAL STATISTICS:")
        report.append("-" * 40)
        
        for pde_name, stats in analysis_results.residual_statistics.items():
            report.append(f"\n{pde_name}:")
            report.append(f"  Mean: {stats['mean']:.6e}")
            report.append(f"  Std:  {stats['std']:.6e}")
            report.append(f"  Max absolute: {stats['max_abs']:.6e}")
            report.append(f"  Mean absolute: {stats['mean_abs']:.6e}")
        
        # Normality test results
        if analysis_results.normality_test_results:
            report.append("\nNORMALITY TEST RESULTS:")
            report.append("-" * 40)
            
            for pde_name, norm_stats in analysis_results.normality_test_results.items():
                report.append(f"\n{pde_name}:")
                if not np.isnan(norm_stats['shapiro_p_value']):
                    report.append(f"  Shapiro-Wilk p-value: {norm_stats['shapiro_p_value']:.6f}")
                report.append(f"  Kolmogorov-Smirnov p-value: {norm_stats['ks_p_value']:.6f}")
        
        return "\n".join(report)