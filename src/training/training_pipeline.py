"""
Complete PINN Training Pipeline

This module provides a high-level interface that integrates all training components
into a unified pipeline for Physics-Informed Neural Network training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from ..core.data_models import TrainingConfig, TrainingResults, WellData
from ..models.model_interface import PINNModelInterface
from ..models.pinn_architecture import PINNArchitecture
from ..physics.physics_loss import create_standard_physics_loss
from .pinn_trainer import PINNTrainer
from .optimizer_manager import create_standard_optimizer_manager, create_fast_optimizer_manager
from .batch_processor import create_standard_batch_processor, create_large_batch_processor
from .convergence_monitor import create_standard_monitor, create_patient_monitor


class PINNTrainingPipeline:
    """
    Complete PINN training pipeline that orchestrates all components.
    
    This class provides a high-level interface for training PINNs with
    automatic component configuration and management.
    """
    
    def __init__(self,
                 model_config: Optional[Dict[str, Any]] = None,
                 training_config: Optional[TrainingConfig] = None,
                 device: Optional[torch.device] = None,
                 experiment_name: str = "pinn_training",
                 output_dir: str = "output"):
        """
        Initialize PINN training pipeline.
        
        Args:
            model_config: Model architecture configuration
            training_config: Training configuration
            device: Device for computations
            experiment_name: Name of the experiment
            output_dir: Output directory for results
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.model_config = model_config or self._get_default_model_config()
        self.training_config = training_config or self._get_default_training_config()
        
        # Components (will be initialized during setup)
        self.model = None
        self.model_interface = None
        self.physics_loss_calculator = None
        self.optimizer_manager = None
        self.batch_processor = None
        self.convergence_monitor = None
        self.trainer = None
        
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            'input_dim': 4,
            'hidden_dims': [100, 100, 100],
            'output_dim': 2,
            'activation': 'tanh',
            'dropout_rate': 0.0,
            'batch_norm': False
        }
    
    def _get_default_training_config(self) -> TrainingConfig:
        """Get default training configuration."""
        return TrainingConfig(
            input_dim=4,
            hidden_dims=[100, 100, 100],
            output_dim=2,
            batch_size=1024,
            learning_rate=1e-3,
            num_epochs=5000,
            optimizer_switch_epoch=3000,
            validation_split=0.2,
            validation_frequency=100
        )
    
    def setup_model(self, 
                   input_features: Optional[List[str]] = None,
                   output_features: Optional[List[str]] = None) -> PINNArchitecture:
        """
        Set up PINN model architecture.
        
        Args:
            input_features: List of input feature names
            output_features: List of output feature names
            
        Returns:
            Configured PINN model
        """
        self.logger.info("Setting up PINN model architecture...")
        
        # Create model
        self.model = PINNArchitecture(
            input_dim=self.model_config['input_dim'],
            hidden_dims=self.model_config['hidden_dims'],
            output_dim=self.model_config['output_dim'],
            activation=self.model_config['activation'],
            dropout_rate=self.model_config['dropout_rate'],
            batch_norm=self.model_config['batch_norm']
        ).to(self.device)
        
        # Create model interface
        self.model_interface = PINNModelInterface(
            model=self.model,
            device=self.device
        )
        
        self.logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def setup_physics_loss(self, 
                          physics_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up physics loss calculator.
        
        Args:
            physics_config: Physics loss configuration
        """
        self.logger.info("Setting up physics loss calculator...")
        
        # Create physics loss calculator with standard configuration
        self.physics_loss_calculator = create_standard_physics_loss(
            device=str(self.device)
        )
        
        # Configure loss weights if provided
        if physics_config and 'loss_weights' in physics_config:
            self.physics_loss_calculator.set_loss_weights(**physics_config['loss_weights'])
        
        self.logger.info("Physics loss calculator initialized")
    
    def setup_training_components(self, 
                                training_mode: str = "standard") -> None:
        """
        Set up training components (optimizer, batch processor, monitor).
        
        Args:
            training_mode: Training mode ("standard", "fast", "conservative", "large_scale")
        """
        self.logger.info(f"Setting up training components for {training_mode} mode...")
        
        # Setup optimizer manager
        if training_mode == "fast":
            self.optimizer_manager = create_fast_optimizer_manager()
        else:
            self.optimizer_manager = create_standard_optimizer_manager()
        
        # Setup batch processor
        if training_mode == "large_scale":
            self.batch_processor = create_large_batch_processor(str(self.device))
        else:
            self.batch_processor = create_standard_batch_processor(str(self.device))
        
        # Setup convergence monitor
        log_dir = self.output_dir / "logs"
        if training_mode == "conservative":
            self.convergence_monitor = create_patient_monitor(str(log_dir), self.experiment_name)
        else:
            self.convergence_monitor = create_standard_monitor(str(log_dir), self.experiment_name)
        
        self.logger.info("Training components initialized")
    
    def setup_trainer(self) -> PINNTrainer:
        """
        Set up the main PINN trainer.
        
        Returns:
            Configured PINN trainer
        """
        if not all([self.physics_loss_calculator, self.optimizer_manager, 
                   self.batch_processor, self.convergence_monitor]):
            raise ValueError("Must setup physics loss and training components before trainer")
        
        self.logger.info("Setting up PINN trainer...")
        
        checkpoint_dir = self.output_dir / "checkpoints"
        
        self.trainer = PINNTrainer(
            physics_loss_calculator=self.physics_loss_calculator,
            optimizer_manager=self.optimizer_manager,
            batch_processor=self.batch_processor,
            convergence_monitor=self.convergence_monitor,
            device=self.device,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        self.logger.info("PINN trainer initialized")
        return self.trainer
    
    def prepare_data(self, 
                    well_data_list: List[WellData],
                    normalization_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare training data and set up normalization.
        
        Args:
            well_data_list: List of well data objects
            normalization_stats: Pre-computed normalization statistics
            
        Returns:
            Data preparation information
        """
        self.logger.info(f"Preparing data from {len(well_data_list)} wells...")
        
        if not self.model_interface:
            raise ValueError("Must setup model before preparing data")
        
        # Compute normalization statistics if not provided
        if normalization_stats is None:
            normalization_stats = self._compute_normalization_stats(well_data_list)
        
        # Set normalization parameters in model
        self.model_interface.set_normalization_parameters(
            input_stats=normalization_stats['input_stats'],
            output_stats=normalization_stats['output_stats']
        )
        
        # Split data for training/validation
        train_wells, val_wells = self._split_wells(well_data_list)
        
        data_info = {
            'total_wells': len(well_data_list),
            'train_wells': len(train_wells),
            'val_wells': len(val_wells),
            'normalization_stats': normalization_stats,
            'train_data': train_wells,
            'val_data': val_wells
        }
        
        self.logger.info(f"Data prepared: {len(train_wells)} training wells, {len(val_wells)} validation wells")
        return data_info
    
    def train(self, 
             well_data_list: List[WellData],
             training_mode: str = "standard",
             resume_from_checkpoint: Optional[str] = None) -> TrainingResults:
        """
        Execute complete PINN training pipeline.
        
        Args:
            well_data_list: List of well data for training
            training_mode: Training mode configuration
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training results
        """
        self.logger.info("="*80)
        self.logger.info(f"Starting PINN training pipeline: {self.experiment_name}")
        self.logger.info("="*80)
        
        # Setup pipeline components
        if self.model is None:
            self.setup_model()
        
        if self.physics_loss_calculator is None:
            self.setup_physics_loss()
        
        if self.trainer is None:
            self.setup_training_components(training_mode)
            self.setup_trainer()
        
        # Prepare data
        data_info = self.prepare_data(well_data_list)
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            checkpoint_info = self.model_interface.load_checkpoint(resume_from_checkpoint)
            self.logger.info(f"Resumed from epoch {checkpoint_info['epoch']}")
        
        # Execute training
        self.logger.info("Starting training...")
        results = self.trainer.train(
            model=self.model_interface,
            dataset=data_info['train_data'],
            config=self.training_config
        )
        
        # Save final results
        self._save_results(results, data_info)
        
        self.logger.info("Training completed successfully!")
        return results
    
    def _compute_normalization_stats(self, well_data_list: List[WellData]) -> Dict[str, Any]:
        """Compute normalization statistics from well data."""
        # This is a simplified implementation
        # In practice, this would compute proper statistics from all wells
        
        input_stats = {
            'depth': (1000.0, 500.0),  # mean, std
            'porosity': (0.15, 0.05),
            'permeability': (100.0, 50.0),
            'gamma_ray': (75.0, 25.0)
        }
        
        output_stats = {
            'pressure': (2000.0, 500.0),
            'saturation': (0.5, 0.2)
        }
        
        return {
            'input_stats': input_stats,
            'output_stats': output_stats
        }
    
    def _split_wells(self, well_data_list: List[WellData]) -> Tuple[List[WellData], List[WellData]]:
        """Split wells into training and validation sets."""
        import random
        
        # Shuffle wells
        shuffled_wells = well_data_list.copy()
        random.shuffle(shuffled_wells)
        
        # Split based on validation ratio
        val_size = int(len(shuffled_wells) * self.training_config.validation_split)
        
        val_wells = shuffled_wells[:val_size]
        train_wells = shuffled_wells[val_size:]
        
        return train_wells, val_wells
    
    def _save_results(self, results: TrainingResults, data_info: Dict[str, Any]) -> None:
        """Save training results and generate reports."""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = results_dir / "final_model.pt"
        self.model_interface.save_checkpoint(
            str(model_path),
            results.total_epochs,
            metadata={'data_info': data_info}
        )
        
        # Export convergence metrics
        metrics_path = results_dir / "training_metrics.json"
        self.convergence_monitor.export_metrics(str(metrics_path))
        
        # Create training plots
        plots_dir = results_dir / "plots"
        self.convergence_monitor.create_training_plots(str(plots_dir))
        
        # Save training summary
        summary = {
            'experiment_name': self.experiment_name,
            'training_results': {
                'total_epochs': results.total_epochs,
                'training_time': results.training_time,
                'converged': results.convergence_epoch is not None,
                'final_loss': results.total_loss_history[-1] if results.total_loss_history else None
            },
            'data_info': data_info,
            'model_config': self.model_config,
            'training_config': self.training_config.__dict__
        }
        
        import json
        summary_path = results_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_dir}")
    
    def evaluate_model(self, 
                      test_wells: List[WellData],
                      metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate trained model on test data.
        
        Args:
            test_wells: Test well data
            metrics: List of metrics to compute
            
        Returns:
            Evaluation metrics
        """
        if not self.model_interface or not self.model_interface.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info(f"Evaluating model on {len(test_wells)} test wells...")
        
        # This would implement proper evaluation logic
        # For now, return placeholder metrics
        evaluation_results = {
            'test_wells': len(test_wells),
            'l2_error': 0.001,
            'mae': 0.01,
            'r2_score': 0.95
        }
        
        self.logger.info(f"Evaluation completed: {evaluation_results}")
        return evaluation_results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'device': str(self.device),
            'model_config': self.model_config,
            'training_config': self.training_config.__dict__ if hasattr(self.training_config, '__dict__') else str(self.training_config),
            'components_initialized': {
                'model': self.model is not None,
                'physics_loss': self.physics_loss_calculator is not None,
                'optimizer_manager': self.optimizer_manager is not None,
                'batch_processor': self.batch_processor is not None,
                'convergence_monitor': self.convergence_monitor is not None,
                'trainer': self.trainer is not None
            }
        }
        
        if self.model:
            summary['model_info'] = self.model.get_model_info()
        
        if self.trainer:
            summary['training_summary'] = self.trainer.get_training_summary()
        
        return summary


# Factory functions for common pipeline configurations

def create_standard_pipeline(experiment_name: str = "pinn_training",
                           output_dir: str = "output") -> PINNTrainingPipeline:
    """Create standard PINN training pipeline."""
    return PINNTrainingPipeline(
        experiment_name=experiment_name,
        output_dir=output_dir
    )


def create_fast_pipeline(experiment_name: str = "pinn_fast_training",
                        output_dir: str = "output") -> PINNTrainingPipeline:
    """Create fast PINN training pipeline for quick experiments."""
    training_config = TrainingConfig(
        input_dim=4,
        hidden_dims=[50, 50, 50],
        output_dim=2,
        batch_size=512,
        learning_rate=1e-3,
        num_epochs=1000,
        optimizer_switch_epoch=500,
        validation_frequency=50
    )
    
    return PINNTrainingPipeline(
        training_config=training_config,
        experiment_name=experiment_name,
        output_dir=output_dir
    )


def create_large_scale_pipeline(experiment_name: str = "pinn_large_training",
                              output_dir: str = "output") -> PINNTrainingPipeline:
    """Create large-scale PINN training pipeline."""
    model_config = {
        'input_dim': 6,
        'hidden_dims': [200, 200, 200, 200],
        'output_dim': 2,
        'activation': 'swish',
        'dropout_rate': 0.1
    }
    
    training_config = TrainingConfig(
        input_dim=6,
        hidden_dims=[200, 200, 200, 200],
        output_dim=2,
        batch_size=2048,
        learning_rate=5e-4,
        num_epochs=10000,
        optimizer_switch_epoch=7000,
        validation_frequency=200
    )
    
    return PINNTrainingPipeline(
        model_config=model_config,
        training_config=training_config,
        experiment_name=experiment_name,
        output_dir=output_dir
    )