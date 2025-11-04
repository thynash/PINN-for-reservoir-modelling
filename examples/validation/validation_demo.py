#!/usr/bin/env python3
"""
Validation and Benchmarking System Demo

This script demonstrates the comprehensive validation and benchmarking capabilities
of the PINN tutorial system, including:
- Hold-out well validation with proper data splitting
- Cross-validation functionality for robust performance assessment
- PDE residual analysis and physics constraint violation detection
- Prediction generation and comparison with actual measurements
- Error analysis and performance benchmarking utilities

Usage:
    python examples/validation_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import logging
from typing import List

from src.validation import ValidationFramework, PDEResidualAnalyzer, PredictionComparator
from src.validation.validation_framework import ValidationConfig
from src.validation.pde_residual_analyzer import ResidualAnalysisConfig
from src.validation.prediction_comparator import PredictionConfig
from src.models.pinn_architecture import PINNArchitecture
from src.physics.physics_loss import PhysicsLossCalculator
from src.core.data_models import WellData, WellMetadata, TrainingConfig
from src.data.las_reader import LASFileReader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_well_data() -> List[WellData]:
    """Create sample well data for demonstration."""
    logger.info("Creating sample well data for validation demo")
    
    wells = []
    
    for well_id in range(5):
        # Create synthetic well data
        depth = np.linspace(1000, 2000, 100)
        
        # Synthetic curves
        porosity = 0.15 + 0.1 * np.sin(depth / 100) + 0.02 * np.random.random(len(depth))
        permeability = 10 * np.exp(porosity * 5) + np.random.random(len(depth))
        gamma_ray = 50 + 30 * np.sin(depth / 200) + 5 * np.random.random(len(depth))
        neutron_porosity = porosity + 0.02 * np.random.random(len(depth))
        
        curves = {
            'PORO': porosity,
            'PERM': permeability,
            'GR': gamma_ray,
            'NPHI': neutron_porosity
        }
        
        metadata = WellMetadata(
            well_id=f"DEMO_WELL_{well_id:03d}",
            location=(39.0 + well_id * 0.1, -95.0 + well_id * 0.1),
            formation="Lansing-Kansas City",
            date_logged=None,
            curve_units={'PORO': 'fraction', 'PERM': 'mD', 'GR': 'API', 'NPHI': 'fraction'},
            total_depth=2000.0
        )
        
        well_data = WellData(
            well_id=f"DEMO_WELL_{well_id:03d}",
            depth=depth,
            curves=curves,
            metadata=metadata
        )
        
        wells.append(well_data)
    
    return wells


def create_demo_model() -> PINNArchitecture:
    """Create a demo PINN model for validation."""
    logger.info("Creating demo PINN model")
    
    model = PINNArchitecture(
        input_dim=4,  # depth, porosity, permeability, gamma ray
        hidden_dims=[100, 100, 100],
        output_dim=2,  # pressure, saturation
        activation='tanh'
    )
    
    # Initialize with random weights for demo
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    
    return model


def demo_validation_framework():
    """Demonstrate the validation framework capabilities."""
    logger.info("=" * 60)
    logger.info("VALIDATION FRAMEWORK DEMO")
    logger.info("=" * 60)
    
    # Create sample data and model
    well_data_list = create_sample_well_data()
    model = create_demo_model()
    
    # Setup physics loss calculator (simplified for demo)
    from src.physics.pde_formulator import PDEFormulator
    from src.physics.boundary_conditions import BoundaryConditionHandler
    
    pde_formulator = PDEFormulator()
    boundary_handler = BoundaryConditionHandler()
    physics_loss_calculator = PhysicsLossCalculator(pde_formulator, boundary_handler)
    
    # Create validation framework
    validation_config = ValidationConfig(
        n_folds=3,
        holdout_fraction=0.3,
        batch_size=512,
        compute_pde_residuals=True
    )
    
    validation_framework = ValidationFramework(physics_loss_calculator, validation_config)
    
    # Training configuration
    training_config = TrainingConfig(
        input_dim=4,
        hidden_dims=[100, 100, 100],
        output_dim=2,
        batch_size=512
    )
    
    # 1. Hold-out validation
    logger.info("Performing hold-out validation...")
    validation_metrics, train_wells, val_wells = validation_framework.holdout_validation(
        model, well_data_list, training_config
    )
    
    logger.info(f"Hold-out validation results:")
    logger.info(f"  L2 Error: {validation_metrics.l2_error:.6f}")
    logger.info(f"  MAE: {validation_metrics.mean_absolute_error:.6f}")
    logger.info(f"  RMSE: {validation_metrics.root_mean_square_error:.6f}")
    logger.info(f"  R² Score: {validation_metrics.r2_score:.6f}")
    logger.info(f"  Mean PDE Residual: {validation_metrics.mean_pde_residual:.6f}")
    
    # 2. Cross-validation
    logger.info("\nPerforming cross-validation...")
    
    def model_factory():
        return create_demo_model()
    
    cv_metrics = validation_framework.cross_validation(
        model_factory, well_data_list, training_config
    )
    
    # Compute cross-validation summary
    cv_summary = validation_framework.get_validation_summary(cv_metrics)
    
    logger.info(f"Cross-validation results ({len(cv_metrics)} folds):")
    logger.info(f"  L2 Error: {cv_summary['l2_error_mean']:.6f} ± {cv_summary['l2_error_std']:.6f}")
    logger.info(f"  MAE: {cv_summary['mae_mean']:.6f} ± {cv_summary['mae_std']:.6f}")
    logger.info(f"  R² Score: {cv_summary['r2_score_mean']:.6f} ± {cv_summary['r2_score_std']:.6f}")


def demo_pde_residual_analyzer():
    """Demonstrate PDE residual analysis capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("PDE RESIDUAL ANALYZER DEMO")
    logger.info("=" * 60)
    
    # Create sample data and model
    well_data_list = create_sample_well_data()
    model = create_demo_model()
    
    # Setup physics loss calculator
    from src.physics.pde_formulator import PDEFormulator
    from src.physics.boundary_conditions import BoundaryConditionHandler
    
    pde_formulator = PDEFormulator()
    boundary_handler = BoundaryConditionHandler()
    physics_loss_calculator = PhysicsLossCalculator(pde_formulator, boundary_handler)
    
    # Create residual analyzer
    residual_config = ResidualAnalysisConfig(
        n_test_points=1000,
        residual_threshold=1e-2,
        violation_threshold=1e-1,
        create_plots=False,  # Disable plots for demo
        perform_normality_test=True
    )
    
    residual_analyzer = PDEResidualAnalyzer(physics_loss_calculator, residual_config)
    
    # Perform residual analysis
    logger.info("Analyzing PDE residuals...")
    analysis_results = residual_analyzer.analyze_residuals(model, well_data_list[:2])
    
    logger.info(f"Residual analysis results:")
    logger.info(f"  Total violations: {analysis_results.violation_count}")
    logger.info(f"  Violation percentage: {analysis_results.violation_percentage:.2f}%")
    
    # Display per-PDE statistics
    for pde_name, stats in analysis_results.residual_statistics.items():
        logger.info(f"  {pde_name}:")
        logger.info(f"    Mean absolute residual: {stats['mean_abs']:.6e}")
        logger.info(f"    Max absolute residual: {stats['max_abs']:.6e}")
        logger.info(f"    Standard deviation: {stats['std']:.6e}")
    
    # Physics constraint violation detection
    logger.info("\nDetecting physics constraint violations...")
    test_points = np.random.random((500, 4))  # Random test points
    violation_analysis = residual_analyzer.detect_physics_violations(model, test_points)
    
    logger.info(f"Violation detection results:")
    logger.info(f"  Total violations: {violation_analysis['total_violations']}")
    logger.info(f"  Total violation percentage: {violation_analysis['total_percentage']:.2f}%")
    
    # Generate residual report
    report = residual_analyzer.generate_residual_report(analysis_results)
    logger.info("\nGenerated residual analysis report:")
    logger.info(report)


def demo_prediction_comparator():
    """Demonstrate prediction and comparison utilities."""
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION COMPARATOR DEMO")
    logger.info("=" * 60)
    
    # Create sample data and model
    well_data_list = create_sample_well_data()
    model = create_demo_model()
    
    # Create prediction comparator
    prediction_config = PredictionConfig(
        prediction_depth_resolution=1.0,
        create_comparison_plots=False,  # Disable plots for demo
        benchmark_against_baseline=True
    )
    
    prediction_comparator = PredictionComparator(prediction_config)
    
    # Generate predictions for a well
    test_well = well_data_list[0]
    logger.info(f"Generating predictions for well {test_well.well_id}...")
    
    predictions = prediction_comparator.generate_predictions(model, test_well)
    
    logger.info(f"Generated predictions:")
    logger.info(f"  Prediction depths: {len(predictions.prediction_depths)} points")
    logger.info(f"  Depth range: {predictions.prediction_depths.min():.1f} - {predictions.prediction_depths.max():.1f} ft")
    logger.info(f"  Pressure range: {predictions.predicted_pressure.min():.1f} - {predictions.predicted_pressure.max():.1f} psi")
    logger.info(f"  Saturation range: {predictions.predicted_saturation.min():.3f} - {predictions.predicted_saturation.max():.3f}")
    
    # Compare with measurements
    logger.info("\nComparing predictions with measurements...")
    predictions_with_comparison = prediction_comparator.compare_with_measurements(predictions, test_well)
    
    logger.info(f"Comparison metrics:")
    for metric_name, value in predictions_with_comparison.comparison_metrics.items():
        logger.info(f"  {metric_name}: {value:.6f}")
    
    # Perform error analysis
    logger.info("\nPerforming error analysis...")
    error_analysis = prediction_comparator.analyze_prediction_errors(predictions_with_comparison)
    
    logger.info(f"Error analysis results:")
    logger.info(f"  Pressure RMSE: {error_analysis['pressure']['rmse']:.3f} psi")
    logger.info(f"  Pressure MAE: {error_analysis['pressure']['mae']:.3f} psi")
    logger.info(f"  Saturation RMSE: {error_analysis['saturation']['rmse']:.6f}")
    logger.info(f"  Saturation MAE: {error_analysis['saturation']['mae']:.6f}")
    
    # Benchmark performance
    logger.info("\nBenchmarking performance...")
    benchmark_results = prediction_comparator.benchmark_performance([predictions_with_comparison])
    
    logger.info(f"Benchmark results:")
    logger.info(f"  PINN performance: {benchmark_results['pinn_performance']}")
    logger.info(f"  Baseline performance: {benchmark_results['baseline_performance']}")


def main():
    """Run the complete validation system demonstration."""
    logger.info("Starting PINN Validation and Benchmarking System Demo")
    logger.info("This demo showcases comprehensive validation capabilities for PINN models")
    
    try:
        # Demo each component
        demo_validation_framework()
        demo_pde_residual_analyzer()
        demo_prediction_comparator()
        
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SYSTEM DEMO COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("The validation system provides:")
        logger.info("✓ Hold-out and cross-validation capabilities")
        logger.info("✓ Comprehensive PDE residual analysis")
        logger.info("✓ Physics constraint violation detection")
        logger.info("✓ Prediction generation and comparison tools")
        logger.info("✓ Error analysis and performance benchmarking")
        logger.info("✓ Statistical analysis and reporting")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()