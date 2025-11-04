#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for the PINN tutorial system.
Tests performance on real KGS well data and provides reproducibility validation.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PINNBenchmarkSuite:
    """Comprehensive benchmarking suite for PINN tutorial system."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "output/benchmarks"):
        """Initialize benchmark suite."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.start_time = None
        
    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        
        logger.info("Starting PINN Tutorial System Benchmark Suite")
        logger.info("=" * 60)
        
        self.start_time = time.time()
        
        # Benchmark 1: Data Processing Performance
        logger.info("1. Benchmarking data processing performance...")
        data_results = self._benchmark_data_processing()
        self.results['data_processing'] = data_results
        
        # Benchmark 2: Model Performance
        logger.info("2. Benchmarking model performance...")
        model_results = self._benchmark_model_performance()
        self.results['model_performance'] = model_results
        
        # Benchmark 3: Training Performance
        logger.info("3. Benchmarking training performance...")
        training_results = self._benchmark_training_performance()
        self.results['training_performance'] = training_results
        
        # Benchmark 4: Validation Performance
        logger.info("4. Benchmarking validation performance...")
        validation_results = self._benchmark_validation_performance()
        self.results['validation_performance'] = validation_results
        
        # Benchmark 5: Reproducibility Tests
        logger.info("5. Running reproducibility tests...")
        reproducibility_results = self._benchmark_reproducibility()
        self.results['reproducibility'] = reproducibility_results
        
        # Benchmark 6: Memory and Resource Usage
        logger.info("6. Benchmarking resource usage...")
        resource_results = self._benchmark_resource_usage()
        self.results['resource_usage'] = resource_results
        
        # Generate summary report
        self._generate_benchmark_report()
        
        total_time = time.time() - self.start_time
        logger.info(f"Benchmark suite completed in {total_time:.2f} seconds")
        
        return self.results
    
    def _benchmark_data_processing(self) -> Dict[str, Any]:
        """Benchmark data processing pipeline performance."""
        
        results = {
            'las_files_processed': 0,
            'processing_time': 0.0,
            'files_per_second': 0.0,
            'memory_usage_mb': 0.0,
            'errors': []
        }
        
        try:
            # Import data processing modules
            from data.las_reader import LASFileReader
            from data.preprocessor import DataPreprocessor
            
            # Find LAS files
            las_files = list(self.data_dir.glob("*.las"))
            
            if not las_files:
                logger.warning("No LAS files found for benchmarking")
                return results
            
            # Limit to first 10 files for benchmarking
            test_files = las_files[:10]
            
            start_time = time.time()
            
            las_reader = LASFileReader()
            preprocessor = DataPreprocessor()
            
            processed_count = 0
            
            for las_file in test_files:
                try:
                    # Read LAS file
                    well_data = las_reader.read_las_file(str(las_file))
                    
                    if well_data is not None:
                        # Preprocess data
                        processed_data = preprocessor.process_well_data(well_data)
                        processed_count += 1
                        
                except Exception as e:
                    results['errors'].append(f"Error processing {las_file.name}: {str(e)}")
            
            processing_time = time.time() - start_time
            
            results.update({
                'las_files_processed': processed_count,
                'processing_time': processing_time,
                'files_per_second': processed_count / processing_time if processing_time > 0 else 0,
                'average_time_per_file': processing_time / processed_count if processed_count > 0 else 0
            })
            
            logger.info(f"   Processed {processed_count} files in {processing_time:.2f}s")
            logger.info(f"   Rate: {results['files_per_second']:.2f} files/second")
            
        except ImportError as e:
            results['errors'].append(f"Import error: {str(e)}")
            logger.error(f"   Data processing benchmark failed: {e}")
        except Exception as e:
            results['errors'].append(f"Benchmark error: {str(e)}")
            logger.error(f"   Unexpected error: {e}")
        
        return results
    
    def _benchmark_model_performance(self) -> Dict[str, Any]:
        """Benchmark PINN model performance."""
        
        results = {
            'model_creation_time': 0.0,
            'forward_pass_time': 0.0,
            'gradient_computation_time': 0.0,
            'throughput_samples_per_second': 0.0,
            'errors': []
        }
        
        try:
            # Try to import PyTorch and model modules
            import torch
            from models.pinn_architecture import PINNArchitecture
            from models.tensor_manager import TensorManager
            
            # Model creation benchmark
            start_time = time.time()
            model = PINNArchitecture(input_dim=4, hidden_dims=[50, 50, 50], output_dim=2)
            model_creation_time = time.time() - start_time
            
            # Tensor manager
            tensor_manager = TensorManager()
            
            # Create test data
            batch_sizes = [10, 50, 100, 500]
            forward_times = []
            gradient_times = []
            
            for batch_size in batch_sizes:
                # Forward pass benchmark
                test_input = torch.randn(batch_size, 4, requires_grad=True)
                
                start_time = time.time()
                output = model(test_input)
                forward_time = time.time() - start_time
                forward_times.append(forward_time)
                
                # Gradient computation benchmark
                start_time = time.time()
                loss = output.sum()
                loss.backward()
                gradient_time = time.time() - start_time
                gradient_times.append(gradient_time)
            
            results.update({
                'model_creation_time': model_creation_time,
                'forward_pass_time': np.mean(forward_times),
                'gradient_computation_time': np.mean(gradient_times),
                'throughput_samples_per_second': np.mean(batch_sizes) / np.mean(forward_times)
            })
            
            logger.info(f"   Model creation: {model_creation_time:.4f}s")
            logger.info(f"   Forward pass: {np.mean(forward_times):.4f}s (avg)")
            logger.info(f"   Gradient computation: {np.mean(gradient_times):.4f}s (avg)")
            
        except ImportError as e:
            results['errors'].append(f"PyTorch not available: {str(e)}")
            logger.warning(f"   Model benchmark skipped: {e}")
        except Exception as e:
            results['errors'].append(f"Model benchmark error: {str(e)}")
            logger.error(f"   Model benchmark failed: {e}")
        
        return results
    
    def _benchmark_training_performance(self) -> Dict[str, Any]:
        """Benchmark training performance."""
        
        results = {
            'training_setup_time': 0.0,
            'epoch_time': 0.0,
            'optimizer_switch_time': 0.0,
            'convergence_monitoring_overhead': 0.0,
            'errors': []
        }
        
        try:
            import torch
            import torch.nn as nn
            from models.pinn_architecture import PINNArchitecture
            from training.optimizer_manager import OptimizerManager
            from training.convergence_monitor import ConvergenceMonitor
            from physics.pde_formulator import PDEFormulator
            from physics.physics_loss import PhysicsLossCalculator
            from physics.boundary_conditions import BoundaryConditionHandler
            
            # Setup benchmark
            start_time = time.time()
            
            model = PINNArchitecture(4, [30, 30], 2)
            pde_formulator = PDEFormulator()
            boundary_handler = BoundaryConditionHandler()
            physics_loss_calc = PhysicsLossCalculator(pde_formulator, boundary_handler)
            
            optimizer_manager = OptimizerManager()
            convergence_monitor = ConvergenceMonitor()
            
            setup_time = time.time() - start_time
            
            # Create synthetic training data
            train_inputs = torch.randn(100, 4, requires_grad=True)
            train_targets = torch.randn(100, 2)
            
            # Benchmark single epoch
            optimizer = optimizer_manager.setup_adam_phase(model, lr=1e-3)
            
            start_time = time.time()
            
            # Forward pass
            outputs = model(train_inputs)
            
            # Compute losses
            data_loss = nn.MSELoss()(outputs, train_targets)
            physics_loss = physics_loss_calc.compute_physics_loss(outputs, train_inputs, {})
            total_loss = data_loss + physics_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_time = time.time() - start_time
            
            # Benchmark convergence monitoring
            start_time = time.time()
            convergence_monitor.update(1, {'total_loss': total_loss.item()})
            monitoring_time = time.time() - start_time
            
            results.update({
                'training_setup_time': setup_time,
                'epoch_time': epoch_time,
                'convergence_monitoring_overhead': monitoring_time
            })
            
            logger.info(f"   Training setup: {setup_time:.4f}s")
            logger.info(f"   Epoch time: {epoch_time:.4f}s")
            logger.info(f"   Monitoring overhead: {monitoring_time:.6f}s")
            
        except ImportError as e:
            results['errors'].append(f"Training modules not available: {str(e)}")
            logger.warning(f"   Training benchmark skipped: {e}")
        except Exception as e:
            results['errors'].append(f"Training benchmark error: {str(e)}")
            logger.error(f"   Training benchmark failed: {e}")
        
        return results
    
    def _benchmark_validation_performance(self) -> Dict[str, Any]:
        """Benchmark validation performance."""
        
        results = {
            'validation_setup_time': 0.0,
            'metrics_computation_time': 0.0,
            'pde_residual_analysis_time': 0.0,
            'prediction_generation_time': 0.0,
            'errors': []
        }
        
        try:
            import torch
            from validation.validation_framework import ValidationFramework, ValidationConfig
            from validation.pde_residual_analyzer import PDEResidualAnalyzer
            from validation.prediction_comparator import PredictionComparator
            from models.pinn_architecture import PINNArchitecture
            
            # Setup
            start_time = time.time()
            
            model = PINNArchitecture(4, [30, 30], 2)
            val_config = ValidationConfig(n_folds=3, holdout_fraction=0.2)
            validator = ValidationFramework(val_config)
            
            setup_time = time.time() - start_time
            
            # Create test data
            test_inputs = torch.randn(50, 4)
            test_targets = torch.randn(50, 2)
            
            # Benchmark metrics computation
            start_time = time.time()
            metrics = validator.compute_validation_metrics(model, test_inputs, test_targets)
            metrics_time = time.time() - start_time
            
            # Benchmark PDE residual analysis
            start_time = time.time()
            residual_analyzer = PDEResidualAnalyzer()
            residual_results = residual_analyzer.analyze_residuals(model, test_inputs)
            residual_time = time.time() - start_time
            
            # Benchmark prediction generation
            start_time = time.time()
            comparator = PredictionComparator()
            predictions = comparator.generate_predictions(model, test_inputs)
            prediction_time = time.time() - start_time
            
            results.update({
                'validation_setup_time': setup_time,
                'metrics_computation_time': metrics_time,
                'pde_residual_analysis_time': residual_time,
                'prediction_generation_time': prediction_time
            })
            
            logger.info(f"   Validation setup: {setup_time:.4f}s")
            logger.info(f"   Metrics computation: {metrics_time:.4f}s")
            logger.info(f"   PDE residual analysis: {residual_time:.4f}s")
            
        except ImportError as e:
            results['errors'].append(f"Validation modules not available: {str(e)}")
            logger.warning(f"   Validation benchmark skipped: {e}")
        except Exception as e:
            results['errors'].append(f"Validation benchmark error: {str(e)}")
            logger.error(f"   Validation benchmark failed: {e}")
        
        return results
    
    def _benchmark_reproducibility(self) -> Dict[str, Any]:
        """Test reproducibility across multiple runs."""
        
        results = {
            'reproducible_results': False,
            'max_difference': 0.0,
            'runs_tested': 0,
            'errors': []
        }
        
        try:
            import torch
            import numpy as np
            from models.pinn_architecture import PINNArchitecture
            
            # Test reproducibility with fixed seeds
            n_runs = 3
            outputs = []
            
            for run in range(n_runs):
                # Set seeds for reproducibility
                torch.manual_seed(42)
                np.random.seed(42)
                
                # Create model and data
                model = PINNArchitecture(4, [20, 20], 2)
                test_input = torch.randn(10, 4)
                
                # Get output
                with torch.no_grad():
                    output = model(test_input)
                    outputs.append(output.numpy())
            
            # Check if outputs are identical
            max_diff = 0.0
            reproducible = True
            
            for i in range(1, len(outputs)):
                diff = np.max(np.abs(outputs[0] - outputs[i]))
                max_diff = max(max_diff, diff)
                
                if diff > 1e-6:
                    reproducible = False
            
            results.update({
                'reproducible_results': reproducible,
                'max_difference': max_diff,
                'runs_tested': n_runs
            })
            
            logger.info(f"   Reproducibility test: {'PASS' if reproducible else 'FAIL'}")
            logger.info(f"   Max difference: {max_diff:.2e}")
            
        except ImportError as e:
            results['errors'].append(f"Reproducibility test skipped: {str(e)}")
            logger.warning(f"   Reproducibility test skipped: {e}")
        except Exception as e:
            results['errors'].append(f"Reproducibility test error: {str(e)}")
            logger.error(f"   Reproducibility test failed: {e}")
        
        return results
    
    def _benchmark_resource_usage(self) -> Dict[str, Any]:
        """Benchmark memory and CPU usage."""
        
        results = {
            'peak_memory_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'disk_usage_mb': 0.0,
            'errors': []
        }
        
        try:
            import psutil
            import os
            
            # Get current process
            process = psutil.Process(os.getpid())
            
            # Memory usage
            memory_info = process.memory_info()
            peak_memory_mb = memory_info.rss / 1024 / 1024
            
            # CPU usage (approximate)
            cpu_percent = process.cpu_percent()
            
            # Disk usage of output directory
            if self.output_dir.exists():
                disk_usage = sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file())
                disk_usage_mb = disk_usage / 1024 / 1024
            else:
                disk_usage_mb = 0.0
            
            results.update({
                'peak_memory_mb': peak_memory_mb,
                'cpu_usage_percent': cpu_percent,
                'disk_usage_mb': disk_usage_mb
            })
            
            logger.info(f"   Peak memory: {peak_memory_mb:.2f} MB")
            logger.info(f"   CPU usage: {cpu_percent:.1f}%")
            logger.info(f"   Disk usage: {disk_usage_mb:.2f} MB")
            
        except ImportError:
            results['errors'].append("psutil not available for resource monitoring")
            logger.warning("   Resource monitoring skipped (psutil not available)")
        except Exception as e:
            results['errors'].append(f"Resource monitoring error: {str(e)}")
            logger.error(f"   Resource monitoring failed: {e}")
        
        return results
    
    def _generate_benchmark_report(self) -> None:
        """Generate comprehensive benchmark report."""
        
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Add metadata
        self.results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': sys.platform,
            'total_runtime': time.time() - self.start_time if self.start_time else 0
        }
        
        # Save detailed results
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = self.output_dir / "benchmark_summary.md"
        self._generate_summary_markdown(summary_file)
        
        logger.info(f"Benchmark report saved to: {report_file}")
        logger.info(f"Summary report saved to: {summary_file}")
    
    def _generate_summary_markdown(self, output_file: Path) -> None:
        """Generate markdown summary report."""
        
        with open(output_file, 'w') as f:
            f.write("# PINN Tutorial System - Benchmark Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data Processing Results
            if 'data_processing' in self.results:
                dp = self.results['data_processing']
                f.write("## Data Processing Performance\n\n")
                f.write(f"- Files processed: {dp.get('las_files_processed', 0)}\n")
                f.write(f"- Processing time: {dp.get('processing_time', 0):.2f}s\n")
                f.write(f"- Rate: {dp.get('files_per_second', 0):.2f} files/second\n\n")
            
            # Model Performance Results
            if 'model_performance' in self.results:
                mp = self.results['model_performance']
                f.write("## Model Performance\n\n")
                f.write(f"- Model creation: {mp.get('model_creation_time', 0):.4f}s\n")
                f.write(f"- Forward pass: {mp.get('forward_pass_time', 0):.4f}s\n")
                f.write(f"- Throughput: {mp.get('throughput_samples_per_second', 0):.1f} samples/second\n\n")
            
            # Training Performance Results
            if 'training_performance' in self.results:
                tp = self.results['training_performance']
                f.write("## Training Performance\n\n")
                f.write(f"- Setup time: {tp.get('training_setup_time', 0):.4f}s\n")
                f.write(f"- Epoch time: {tp.get('epoch_time', 0):.4f}s\n\n")
            
            # Reproducibility Results
            if 'reproducibility' in self.results:
                rp = self.results['reproducibility']
                f.write("## Reproducibility\n\n")
                f.write(f"- Reproducible: {'PASS' if rp.get('reproducible_results', False) else 'FAIL'}\n")
                f.write(f"- Max difference: {rp.get('max_difference', 0):.2e}\n\n")
            
            # Resource Usage
            if 'resource_usage' in self.results:
                ru = self.results['resource_usage']
                f.write("## Resource Usage\n\n")
                f.write(f"- Peak memory: {ru.get('peak_memory_mb', 0):.2f} MB\n")
                f.write(f"- CPU usage: {ru.get('cpu_usage_percent', 0):.1f}%\n")
                f.write(f"- Disk usage: {ru.get('disk_usage_mb', 0):.2f} MB\n\n")
            
            # Overall Assessment
            f.write("## Overall Assessment\n\n")
            
            total_errors = sum(len(result.get('errors', [])) for result in self.results.values() if isinstance(result, dict))
            
            if total_errors == 0:
                f.write("PASS: All benchmarks completed successfully\n")
            else:
                f.write(f"WARNING: {total_errors} errors encountered during benchmarking\n")
            
            f.write(f"\nTotal benchmark runtime: {self.results['metadata']['total_runtime']:.2f} seconds\n")


def main():
    """Main entry point for benchmark suite."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="PINN Tutorial System Benchmark Suite")
    parser.add_argument("--data-dir", default="data", help="Directory containing LAS files")
    parser.add_argument("--output-dir", default="output/benchmarks", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run benchmark suite
    benchmark = PINNBenchmarkSuite(args.data_dir, args.output_dir)
    results = benchmark.run_complete_benchmark()
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUITE COMPLETED")
    print("=" * 60)
    
    total_errors = sum(len(result.get('errors', [])) for result in results.values() if isinstance(result, dict))
    
    if total_errors == 0:
        print("Status: ALL BENCHMARKS PASSED")
    else:
        print(f"Status: {total_errors} ERRORS ENCOUNTERED")
    
    print(f"Results saved to: {benchmark.output_dir}")


if __name__ == "__main__":
    main()