"""
Prediction and comparison utilities for PINN validation.

This module implements tools for:
- Prediction generation for pressure and saturation profiles
- Comparison tools for predicted vs actual well measurements
- Error analysis and performance benchmarking utilities
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..core.data_models import WellData, ValidationMetrics
from ..core.interfaces import ModelInterface


@dataclass
class PredictionConfig:
    """Configuration for prediction generation and comparison."""
    # Prediction settings
    prediction_depth_resolution: float = 0.5  # meters
    extrapolation_factor: float = 1.2  # Factor for depth extrapolation
    
    # Comparison settings
    comparison_metrics: List[str] = None
    confidence_intervals: bool = True
    confidence_level: float = 0.95
    
    # Visualization settings
    create_comparison_plots: bool = True
    save_plots: bool = False
    plot_dir: str = "prediction_plots"
    
    # Performance benchmarking
    benchmark_against_baseline: bool = True
    baseline_method: str = "linear_interpolation"
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Set default comparison metrics if not provided."""
        if self.comparison_metrics is None:
            self.comparison_metrics = [
                'mse', 'mae', 'rmse', 'r2', 'mape', 'correlation'
            ]


@dataclass
class PredictionResults:
    """Results from prediction generation and comparison."""
    # Predictions
    predicted_pressure: np.ndarray
    predicted_saturation: np.ndarray
    prediction_depths: np.ndarray
    
    # Actual measurements (if available)
    actual_pressure: Optional[np.ndarray] = None
    actual_saturation: Optional[np.ndarray] = None
    actual_depths: Optional[np.ndarray] = None
    
    # Comparison metrics
    comparison_metrics: Optional[Dict[str, float]] = None
    
    # Error analysis
    pressure_errors: Optional[np.ndarray] = None
    saturation_errors: Optional[np.ndarray] = None
    
    # Confidence intervals
    pressure_ci_lower: Optional[np.ndarray] = None
    pressure_ci_upper: Optional[np.ndarray] = None
    saturation_ci_lower: Optional[np.ndarray] = None
    saturation_ci_upper: Optional[np.ndarray] = None
    
    # Benchmark comparison
    baseline_metrics: Optional[Dict[str, float]] = None
    
    # Plots (if generated)
    comparison_plots: Optional[Dict[str, plt.Figure]] = None


class PredictionComparator:
    """
    Comprehensive prediction and comparison utilities for PINN validation.
    
    Generates predictions for pressure and saturation profiles and provides
    detailed comparison with actual measurements and baseline methods.
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize prediction comparator.
        
        Args:
            config: Prediction configuration (uses defaults if None)
        """
        self.config = config or PredictionConfig()
        self.device = torch.device(self.config.device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Prediction history
        self.prediction_history: List[PredictionResults] = []    
 
    def generate_predictions(self,
                           model: Union[ModelInterface, nn.Module],
                           well_data: WellData,
                           depth_range: Optional[Tuple[float, float]] = None) -> PredictionResults:
        """
        Generate pressure and saturation predictions for a well profile.
        
        Args:
            model: PINN model for prediction generation
            well_data: Well data for prediction context
            depth_range: Optional depth range for predictions (min_depth, max_depth)
            
        Returns:
            Comprehensive prediction results
        """
        self.logger.info(f"Generating predictions for well {well_data.well_id}")
        
        # Ensure model is in evaluation mode
        if hasattr(model, 'get_model'):
            model = model.get_model()
        
        model = model.to(self.device)
        model.eval()
        
        # Determine depth range
        if depth_range is None:
            min_depth = np.min(well_data.depth)
            max_depth = np.max(well_data.depth)
            # Add extrapolation
            depth_span = max_depth - min_depth
            min_depth -= depth_span * (self.config.extrapolation_factor - 1) / 2
            max_depth += depth_span * (self.config.extrapolation_factor - 1) / 2
        else:
            min_depth, max_depth = depth_range
        
        # Create prediction depths
        prediction_depths = np.arange(min_depth, max_depth + self.config.prediction_depth_resolution,
                                    self.config.prediction_depth_resolution)
        
        # Prepare input features for prediction
        prediction_inputs = self._prepare_prediction_inputs(well_data, prediction_depths)
        
        # Generate predictions
        with torch.no_grad():
            input_tensor = torch.FloatTensor(prediction_inputs).to(self.device)
            predictions = model(input_tensor)
            predictions_np = predictions.cpu().numpy()
        
        # Extract pressure and saturation predictions
        predicted_pressure = predictions_np[:, 0] if predictions_np.shape[1] > 0 else np.zeros(len(prediction_depths))
        predicted_saturation = predictions_np[:, 1] if predictions_np.shape[1] > 1 else np.zeros(len(prediction_depths))
        
        # Create results object
        results = PredictionResults(
            predicted_pressure=predicted_pressure,
            predicted_saturation=predicted_saturation,
            prediction_depths=prediction_depths
        )
        
        self.prediction_history.append(results)
        return results
    
    def compare_with_measurements(self,
                                 predictions: PredictionResults,
                                 actual_measurements: WellData) -> PredictionResults:
        """
        Compare predictions with actual well measurements.
        
        Args:
            predictions: Generated predictions
            actual_measurements: Actual well measurements for comparison
            
        Returns:
            Updated prediction results with comparison metrics
        """
        self.logger.info("Comparing predictions with actual measurements")
        
        # Extract actual measurements
        actual_depths = actual_measurements.depth
        
        # Get actual pressure and saturation (or create synthetic for demo)
        actual_pressure = self._extract_or_synthesize_pressure(actual_measurements)
        actual_saturation = self._extract_or_synthesize_saturation(actual_measurements)
        
        # Interpolate predictions to match actual measurement depths
        pred_pressure_interp = np.interp(actual_depths, predictions.prediction_depths, predictions.predicted_pressure)
        pred_saturation_interp = np.interp(actual_depths, predictions.prediction_depths, predictions.predicted_saturation)
        
        # Compute comparison metrics
        comparison_metrics = self._compute_comparison_metrics(
            pred_pressure_interp, pred_saturation_interp,
            actual_pressure, actual_saturation
        )
        
        # Compute errors
        pressure_errors = pred_pressure_interp - actual_pressure
        saturation_errors = pred_saturation_interp - actual_saturation
        
        # Compute confidence intervals if requested
        pressure_ci_lower, pressure_ci_upper = None, None
        saturation_ci_lower, saturation_ci_upper = None, None
        
        if self.config.confidence_intervals:
            pressure_ci_lower, pressure_ci_upper = self._compute_confidence_intervals(
                pressure_errors, pred_pressure_interp
            )
            saturation_ci_lower, saturation_ci_upper = self._compute_confidence_intervals(
                saturation_errors, pred_saturation_interp
            )
        
        # Benchmark against baseline if requested
        baseline_metrics = None
        if self.config.benchmark_against_baseline:
            baseline_metrics = self._benchmark_against_baseline(
                actual_depths, actual_pressure, actual_saturation,
                pred_pressure_interp, pred_saturation_interp
            )
        
        # Create comparison plots
        plots = {}
        if self.config.create_comparison_plots:
            plots = self._create_comparison_plots(
                predictions, actual_depths, actual_pressure, actual_saturation,
                pred_pressure_interp, pred_saturation_interp, pressure_errors, saturation_errors
            )
        
        # Update results
        predictions.actual_pressure = actual_pressure
        predictions.actual_saturation = actual_saturation
        predictions.actual_depths = actual_depths
        predictions.comparison_metrics = comparison_metrics
        predictions.pressure_errors = pressure_errors
        predictions.saturation_errors = saturation_errors
        predictions.pressure_ci_lower = pressure_ci_lower
        predictions.pressure_ci_upper = pressure_ci_upper
        predictions.saturation_ci_lower = saturation_ci_lower
        predictions.saturation_ci_upper = saturation_ci_upper
        predictions.baseline_metrics = baseline_metrics
        predictions.comparison_plots = plots
        
        return predictions 
   
    def analyze_prediction_errors(self,
                                 predictions: PredictionResults) -> Dict[str, Any]:
        """
        Perform comprehensive error analysis on predictions.
        
        Args:
            predictions: Prediction results with comparison data
            
        Returns:
            Dictionary with detailed error analysis
        """
        if predictions.pressure_errors is None or predictions.saturation_errors is None:
            raise ValueError("Comparison with measurements must be performed first")
        
        analysis = {}
        
        # Pressure error analysis
        pressure_errors = predictions.pressure_errors
        analysis['pressure'] = {
            'mean_error': np.mean(pressure_errors),
            'std_error': np.std(pressure_errors),
            'max_abs_error': np.max(np.abs(pressure_errors)),
            'rmse': np.sqrt(np.mean(pressure_errors**2)),
            'mae': np.mean(np.abs(pressure_errors)),
            'error_percentiles': {
                '5th': np.percentile(pressure_errors, 5),
                '25th': np.percentile(pressure_errors, 25),
                '50th': np.percentile(pressure_errors, 50),
                '75th': np.percentile(pressure_errors, 75),
                '95th': np.percentile(pressure_errors, 95)
            }
        }
        
        # Saturation error analysis
        saturation_errors = predictions.saturation_errors
        analysis['saturation'] = {
            'mean_error': np.mean(saturation_errors),
            'std_error': np.std(saturation_errors),
            'max_abs_error': np.max(np.abs(saturation_errors)),
            'rmse': np.sqrt(np.mean(saturation_errors**2)),
            'mae': np.mean(np.abs(saturation_errors)),
            'error_percentiles': {
                '5th': np.percentile(saturation_errors, 5),
                '25th': np.percentile(saturation_errors, 25),
                '50th': np.percentile(saturation_errors, 50),
                '75th': np.percentile(saturation_errors, 75),
                '95th': np.percentile(saturation_errors, 95)
            }
        }
        
        # Spatial error analysis
        depths = predictions.actual_depths
        analysis['spatial'] = {
            'depth_range': (np.min(depths), np.max(depths)),
            'pressure_error_vs_depth_correlation': np.corrcoef(depths, pressure_errors)[0, 1],
            'saturation_error_vs_depth_correlation': np.corrcoef(depths, saturation_errors)[0, 1]
        }
        
        # Error distribution analysis
        analysis['distribution'] = {
            'pressure_error_normality': stats.shapiro(pressure_errors)[1] if len(pressure_errors) <= 5000 else np.nan,
            'saturation_error_normality': stats.shapiro(saturation_errors)[1] if len(saturation_errors) <= 5000 else np.nan,
            'pressure_error_skewness': stats.skew(pressure_errors),
            'saturation_error_skewness': stats.skew(saturation_errors),
            'pressure_error_kurtosis': stats.kurtosis(pressure_errors),
            'saturation_error_kurtosis': stats.kurtosis(saturation_errors)
        }
        
        return analysis
    
    def benchmark_performance(self,
                            predictions_list: List[PredictionResults],
                            benchmark_methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark PINN performance against multiple baseline methods.
        
        Args:
            predictions_list: List of prediction results from multiple wells
            benchmark_methods: List of baseline methods to compare against
            
        Returns:
            Comprehensive benchmarking results
        """
        if benchmark_methods is None:
            benchmark_methods = ['linear_interpolation', 'polynomial_fit', 'spline_interpolation']
        
        self.logger.info(f"Benchmarking performance against {len(benchmark_methods)} baseline methods")
        
        benchmark_results = {
            'pinn_performance': {},
            'baseline_performance': {method: {} for method in benchmark_methods},
            'relative_improvement': {}
        }
        
        # Aggregate PINN performance
        pinn_metrics = self._aggregate_prediction_metrics(predictions_list)
        benchmark_results['pinn_performance'] = pinn_metrics
        
        # Compute baseline performance for each method
        for method in benchmark_methods:
            baseline_metrics = self._compute_baseline_performance(predictions_list, method)
            benchmark_results['baseline_performance'][method] = baseline_metrics
            
            # Compute relative improvement
            improvement = {}
            for metric_name, pinn_value in pinn_metrics.items():
                baseline_value = baseline_metrics.get(metric_name, np.nan)
                if not np.isnan(baseline_value) and baseline_value != 0:
                    improvement[metric_name] = ((baseline_value - pinn_value) / baseline_value) * 100
                else:
                    improvement[metric_name] = np.nan
            
            benchmark_results['relative_improvement'][method] = improvement
        
        return benchmark_results    
 
   def _prepare_prediction_inputs(self, well_data: WellData, prediction_depths: np.ndarray) -> np.ndarray:
        """Prepare input features for prediction at specified depths."""
        # Interpolate well log curves to prediction depths
        input_features = [prediction_depths]
        
        # Interpolate available curves
        for curve_name in ['PORO', 'PERM', 'GR', 'NPHI']:
            if curve_name in well_data.curves:
                interpolated_curve = np.interp(prediction_depths, well_data.depth, well_data.curves[curve_name])
                input_features.append(interpolated_curve)
            else:
                # Use default values if curve not available
                default_values = {
                    'PORO': np.full_like(prediction_depths, 0.15),  # 15% porosity
                    'PERM': np.full_like(prediction_depths, 10.0),  # 10 mD permeability
                    'GR': np.full_like(prediction_depths, 50.0),    # 50 API gamma ray
                    'NPHI': np.full_like(prediction_depths, 0.15)   # 15% neutron porosity
                }
                input_features.append(default_values[curve_name])
        
        return np.column_stack(input_features)
    
    def _extract_or_synthesize_pressure(self, well_data: WellData) -> np.ndarray:
        """Extract actual pressure measurements or synthesize for demonstration."""
        if 'PRESSURE' in well_data.curves:
            return well_data.curves['PRESSURE']
        else:
            # Synthesize pressure using hydrostatic gradient
            depth = well_data.depth
            surface_pressure = 14.7  # psi
            pressure_gradient = 0.433  # psi/ft (typical for water)
            return surface_pressure + depth * pressure_gradient
    
    def _extract_or_synthesize_saturation(self, well_data: WellData) -> np.ndarray:
        """Extract actual saturation measurements or synthesize for demonstration."""
        if 'SW' in well_data.curves:
            return well_data.curves['SW']
        elif 'SATURATION' in well_data.curves:
            return well_data.curves['SATURATION']
        else:
            # Synthesize saturation based on porosity and resistivity
            depth = well_data.depth
            if 'PORO' in well_data.curves and 'RT' in well_data.curves:
                porosity = well_data.curves['PORO']
                resistivity = well_data.curves['RT']
                # Simplified Archie's equation
                saturation = np.clip(0.2 + 0.6 * (1 - porosity) + 0.1 * np.random.random(len(depth)), 0, 1)
            else:
                # Random saturation with depth trend
                saturation = 0.3 + 0.4 * np.random.random(len(depth))
            
            return saturation
    
    def _compute_comparison_metrics(self,
                                  pred_pressure: np.ndarray,
                                  pred_saturation: np.ndarray,
                                  actual_pressure: np.ndarray,
                                  actual_saturation: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive comparison metrics."""
        metrics = {}
        
        # Pressure metrics
        if 'mse' in self.config.comparison_metrics:
            metrics['pressure_mse'] = mean_squared_error(actual_pressure, pred_pressure)
        if 'mae' in self.config.comparison_metrics:
            metrics['pressure_mae'] = mean_absolute_error(actual_pressure, pred_pressure)
        if 'rmse' in self.config.comparison_metrics:
            metrics['pressure_rmse'] = np.sqrt(mean_squared_error(actual_pressure, pred_pressure))
        if 'r2' in self.config.comparison_metrics:
            metrics['pressure_r2'] = r2_score(actual_pressure, pred_pressure)
        if 'mape' in self.config.comparison_metrics:
            metrics['pressure_mape'] = np.mean(np.abs((actual_pressure - pred_pressure) / (actual_pressure + 1e-8))) * 100
        if 'correlation' in self.config.comparison_metrics:
            metrics['pressure_correlation'] = np.corrcoef(actual_pressure, pred_pressure)[0, 1]
        
        # Saturation metrics
        if 'mse' in self.config.comparison_metrics:
            metrics['saturation_mse'] = mean_squared_error(actual_saturation, pred_saturation)
        if 'mae' in self.config.comparison_metrics:
            metrics['saturation_mae'] = mean_absolute_error(actual_saturation, pred_saturation)
        if 'rmse' in self.config.comparison_metrics:
            metrics['saturation_rmse'] = np.sqrt(mean_squared_error(actual_saturation, pred_saturation))
        if 'r2' in self.config.comparison_metrics:
            metrics['saturation_r2'] = r2_score(actual_saturation, pred_saturation)
        if 'mape' in self.config.comparison_metrics:
            metrics['saturation_mape'] = np.mean(np.abs((actual_saturation - pred_saturation) / (actual_saturation + 1e-8))) * 100
        if 'correlation' in self.config.comparison_metrics:
            metrics['saturation_correlation'] = np.corrcoef(actual_saturation, pred_saturation)[0, 1]
        
        # Combined metrics
        combined_mse = (metrics.get('pressure_mse', 0) + metrics.get('saturation_mse', 0)) / 2
        combined_mae = (metrics.get('pressure_mae', 0) + metrics.get('saturation_mae', 0)) / 2
        
        metrics['combined_mse'] = combined_mse
        metrics['combined_mae'] = combined_mae
        
        return metrics    
 
   def _compute_confidence_intervals(self, 
                                    errors: np.ndarray,
                                    predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals for predictions."""
        error_std = np.std(errors)
        z_score = stats.norm.ppf((1 + self.config.confidence_level) / 2)
        
        margin = z_score * error_std
        ci_lower = predictions - margin
        ci_upper = predictions + margin
        
        return ci_lower, ci_upper
    
    def _benchmark_against_baseline(self,
                                  depths: np.ndarray,
                                  actual_pressure: np.ndarray,
                                  actual_saturation: np.ndarray,
                                  pred_pressure: np.ndarray,
                                  pred_saturation: np.ndarray) -> Dict[str, float]:
        """Benchmark PINN predictions against baseline method."""
        baseline_metrics = {}
        
        if self.config.baseline_method == 'linear_interpolation':
            # Simple linear interpolation baseline
            baseline_pressure = np.interp(depths, depths, actual_pressure)
            baseline_saturation = np.interp(depths, depths, actual_saturation)
        elif self.config.baseline_method == 'polynomial_fit':
            # Polynomial fit baseline
            p_coeffs = np.polyfit(depths, actual_pressure, deg=2)
            s_coeffs = np.polyfit(depths, actual_saturation, deg=2)
            baseline_pressure = np.polyval(p_coeffs, depths)
            baseline_saturation = np.polyval(s_coeffs, depths)
        else:
            # Default to mean values
            baseline_pressure = np.full_like(depths, np.mean(actual_pressure))
            baseline_saturation = np.full_like(depths, np.mean(actual_saturation))
        
        # Compute baseline metrics
        baseline_metrics['pressure_mse'] = mean_squared_error(actual_pressure, baseline_pressure)
        baseline_metrics['pressure_mae'] = mean_absolute_error(actual_pressure, baseline_pressure)
        baseline_metrics['saturation_mse'] = mean_squared_error(actual_saturation, baseline_saturation)
        baseline_metrics['saturation_mae'] = mean_absolute_error(actual_saturation, baseline_saturation)
        
        return baseline_metrics
    
    def _create_comparison_plots(self,
                               predictions: PredictionResults,
                               actual_depths: np.ndarray,
                               actual_pressure: np.ndarray,
                               actual_saturation: np.ndarray,
                               pred_pressure_interp: np.ndarray,
                               pred_saturation_interp: np.ndarray,
                               pressure_errors: np.ndarray,
                               saturation_errors: np.ndarray) -> Dict[str, plt.Figure]:
        """Create comprehensive comparison plots."""
        plots = {}
        
        # Set style for scientific plots
        plt.style.use('seaborn-v0_8')
        
        # 1. Side-by-side well profile comparison
        fig_profiles, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        
        # Pressure profile
        ax1.plot(actual_pressure, actual_depths, 'b-', linewidth=2, label='Actual', marker='o', markersize=3)
        ax1.plot(pred_pressure_interp, actual_depths, 'r--', linewidth=2, label='PINN Predicted')
        ax1.set_xlabel('Pressure (psi)')
        ax1.set_ylabel('Depth (ft)')
        ax1.set_title('Pressure Profile Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # Saturation profile
        ax2.plot(actual_saturation, actual_depths, 'b-', linewidth=2, label='Actual', marker='o', markersize=3)
        ax2.plot(pred_saturation_interp, actual_depths, 'r--', linewidth=2, label='PINN Predicted')
        ax2.set_xlabel('Water Saturation')
        ax2.set_ylabel('Depth (ft)')
        ax2.set_title('Saturation Profile Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plots['profile_comparison'] = fig_profiles
        
        # 2. Scatter plots for correlation analysis
        fig_scatter, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pressure scatter
        ax1.scatter(actual_pressure, pred_pressure_interp, alpha=0.6, color='blue')
        min_p, max_p = min(actual_pressure.min(), pred_pressure_interp.min()), max(actual_pressure.max(), pred_pressure_interp.max())
        ax1.plot([min_p, max_p], [min_p, max_p], 'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Pressure (psi)')
        ax1.set_ylabel('Predicted Pressure (psi)')
        ax1.set_title('Pressure Prediction Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Saturation scatter
        ax2.scatter(actual_saturation, pred_saturation_interp, alpha=0.6, color='green')
        min_s, max_s = min(actual_saturation.min(), pred_saturation_interp.min()), max(actual_saturation.max(), pred_saturation_interp.max())
        ax2.plot([min_s, max_s], [min_s, max_s], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Saturation')
        ax2.set_ylabel('Predicted Saturation')
        ax2.set_title('Saturation Prediction Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['scatter_comparison'] = fig_scatter
        
        # 3. Error analysis plots
        fig_errors, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Pressure error vs depth
        ax1.scatter(pressure_errors, actual_depths, alpha=0.6, color='red')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Pressure Error (psi)')
        ax1.set_ylabel('Depth (ft)')
        ax1.set_title('Pressure Error vs Depth')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # Saturation error vs depth
        ax2.scatter(saturation_errors, actual_depths, alpha=0.6, color='orange')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Saturation Error')
        ax2.set_ylabel('Depth (ft)')
        ax2.set_title('Saturation Error vs Depth')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # Pressure error histogram
        ax3.hist(pressure_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Pressure Error (psi)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Pressure Error Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Saturation error histogram
        ax4.hist(saturation_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Saturation Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Saturation Error Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['error_analysis'] = fig_errors
        
        return plots
    
    def _aggregate_prediction_metrics(self, predictions_list: List[PredictionResults]) -> Dict[str, float]:
        """Aggregate metrics across multiple prediction results."""
        aggregated = {}
        
        if not predictions_list:
            return aggregated
        
        # Collect all metrics
        all_metrics = {}
        for pred in predictions_list:
            if pred.comparison_metrics:
                for metric_name, value in pred.comparison_metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # Compute aggregated statistics
        for metric_name, values in all_metrics.items():
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)
            aggregated[f'{metric_name}_median'] = np.median(values)
        
        return aggregated
    
    def _compute_baseline_performance(self, 
                                    predictions_list: List[PredictionResults],
                                    method: str) -> Dict[str, float]:
        """Compute baseline performance using specified method."""
        baseline_metrics = {}
        
        # This is a simplified implementation
        # In practice, you would implement each baseline method properly
        baseline_metrics['pressure_mse_mean'] = 1000.0  # Placeholder
        baseline_metrics['pressure_mae_mean'] = 25.0    # Placeholder
        baseline_metrics['saturation_mse_mean'] = 0.1   # Placeholder
        baseline_metrics['saturation_mae_mean'] = 0.2   # Placeholder
        
        return baseline_metrics