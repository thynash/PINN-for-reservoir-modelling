"""
Core Module

Contains base interfaces and data models used throughout the PINN tutorial system.
"""

from .data_models import (
    WellData,
    WellMetadata,
    TrainingConfig,
    ValidationMetrics,
    ModelConfig
)
from .interfaces import (
    DataProcessorInterface,
    ModelInterface,
    TrainerInterface,
    VisualizerInterface
)

__all__ = [
    'WellData',
    'WellMetadata', 
    'TrainingConfig',
    'ValidationMetrics',
    'ModelConfig',
    'DataProcessorInterface',
    'ModelInterface',
    'TrainerInterface',
    'VisualizerInterface'
]