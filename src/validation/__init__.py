"""
Validation and benchmarking system for PINN models.

This module provides comprehensive validation tools including:
- Hold-out well validation with proper data splitting
- Cross-validation functionality for robust performance assessment
- Validation metrics computation (L2 error, MAE, PDE residuals)
- PDE residual analysis and physics constraint violation detection
- Prediction generation and comparison utilities
"""

from .validation_framework import ValidationFramework
from .pde_residual_analyzer import PDEResidualAnalyzer
from .prediction_comparator import PredictionComparator

__all__ = [
    'ValidationFramework',
    'PDEResidualAnalyzer', 
    'PredictionComparator'
]