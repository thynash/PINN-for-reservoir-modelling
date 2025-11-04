"""
PINN Training Engine

This module provides comprehensive training capabilities for Physics-Informed Neural Networks,
including training orchestration, optimizer management, batch processing, and convergence monitoring.
"""

from .pinn_trainer import PINNTrainer, TrainingState
from .optimizer_manager import (
    OptimizerManager, 
    OptimizerConfig, 
    OptimizerPhase,
    LearningRateScheduler,
    create_standard_optimizer_manager,
    create_fast_optimizer_manager,
    create_conservative_optimizer_manager
)
from .batch_processor import (
    BatchProcessor,
    BatchConfig,
    SamplingStrategy,
    RandomSampling,
    StratifiedSampling,
    AdaptivePhysicsSampling,
    WellPointSampler,
    create_standard_batch_processor,
    create_large_batch_processor,
    create_memory_efficient_processor
)
from .convergence_monitor import (
    ConvergenceMonitor,
    ConvergenceConfig,
    TrainingMetrics,
    EarlyStoppingMonitor,
    ConvergenceDetector,
    TrainingLogger,
    create_standard_monitor,
    create_patient_monitor,
    create_sensitive_monitor
)
from .training_pipeline import (
    PINNTrainingPipeline,
    create_standard_pipeline,
    create_fast_pipeline,
    create_large_scale_pipeline
)

__all__ = [
    # Core training
    'PINNTrainer',
    'TrainingState',
    
    # Optimizer management
    'OptimizerManager',
    'OptimizerConfig',
    'OptimizerPhase',
    'LearningRateScheduler',
    'create_standard_optimizer_manager',
    'create_fast_optimizer_manager',
    'create_conservative_optimizer_manager',
    
    # Batch processing
    'BatchProcessor',
    'BatchConfig',
    'SamplingStrategy',
    'RandomSampling',
    'StratifiedSampling',
    'AdaptivePhysicsSampling',
    'WellPointSampler',
    'create_standard_batch_processor',
    'create_large_batch_processor',
    'create_memory_efficient_processor',
    
    # Convergence monitoring
    'ConvergenceMonitor',
    'ConvergenceConfig',
    'TrainingMetrics',
    'EarlyStoppingMonitor',
    'ConvergenceDetector',
    'TrainingLogger',
    'create_standard_monitor',
    'create_patient_monitor',
    'create_sensitive_monitor',
    
    # Training pipeline
    'PINNTrainingPipeline',
    'create_standard_pipeline',
    'create_fast_pipeline',
    'create_large_scale_pipeline'
]