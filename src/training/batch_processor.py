"""
Batch Processing and Data Handling for PINN Training

This module implements the BatchProcessor class that handles efficient data sampling,
batching, gradient accumulation, and data shuffling strategies for well point selection
in Physics-Informed Neural Network training.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from ..core.data_models import BatchData, WellData
from ..core.interfaces import DatasetInterface


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    # Batch sizes
    data_batch_size: int = 512
    physics_batch_size: int = 1024
    boundary_batch_size: int = 256
    
    # Sampling strategies
    data_sampling_strategy: str = "random"  # "random", "stratified", "weighted"
    physics_sampling_strategy: str = "uniform"  # "uniform", "adaptive", "residual_based"
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    effective_batch_size: Optional[int] = None
    
    # Data shuffling
    shuffle_data: bool = True
    shuffle_physics_points: bool = True
    resample_physics_frequency: int = 10  # Resample physics points every N epochs
    
    # Well point selection
    wells_per_batch: int = 4
    points_per_well: int = 128
    depth_sampling_strategy: str = "uniform"  # "uniform", "log_uniform", "adaptive"
    
    # Memory management
    pin_memory: bool = True
    num_workers: int = 0
    prefetch_factor: int = 2


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""
    
    @abstractmethod
    def sample(self, data: Any, batch_size: int, **kwargs) -> Any:
        """Sample data according to the strategy."""
        pass


class RandomSampling(SamplingStrategy):
    """Random sampling strategy."""
    
    def sample(self, data: torch.Tensor, batch_size: int, **kwargs) -> torch.Tensor:
        """Randomly sample from data."""
        if len(data) <= batch_size:
            return data
        
        indices = torch.randperm(len(data))[:batch_size]
        return data[indices]


class StratifiedSampling(SamplingStrategy):
    """Stratified sampling based on data characteristics."""
    
    def __init__(self, stratify_column: int = 0, n_strata: int = 5):
        """
        Initialize stratified sampling.
        
        Args:
            stratify_column: Column index to use for stratification
            n_strata: Number of strata
        """
        self.stratify_column = stratify_column
        self.n_strata = n_strata
    
    def sample(self, data: torch.Tensor, batch_size: int, **kwargs) -> torch.Tensor:
        """Sample using stratified approach."""
        if len(data) <= batch_size:
            return data
        
        # Create strata based on specified column
        values = data[:, self.stratify_column]
        min_val, max_val = values.min(), values.max()
        
        # Define strata boundaries
        boundaries = torch.linspace(min_val, max_val, self.n_strata + 1)
        
        sampled_indices = []
        samples_per_stratum = batch_size // self.n_strata
        
        for i in range(self.n_strata):
            # Find points in this stratum
            if i == self.n_strata - 1:
                mask = (values >= boundaries[i]) & (values <= boundaries[i + 1])
            else:
                mask = (values >= boundaries[i]) & (values < boundaries[i + 1])
            
            stratum_indices = torch.where(mask)[0]
            
            if len(stratum_indices) > 0:
                # Sample from this stratum
                n_samples = min(samples_per_stratum, len(stratum_indices))
                if len(stratum_indices) > n_samples:
                    selected = stratum_indices[torch.randperm(len(stratum_indices))[:n_samples]]
                else:
                    selected = stratum_indices
                sampled_indices.extend(selected.tolist())
        
        # Fill remaining slots randomly if needed
        while len(sampled_indices) < batch_size:
            remaining_indices = [i for i in range(len(data)) if i not in sampled_indices]
            if remaining_indices:
                sampled_indices.append(random.choice(remaining_indices))
            else:
                break
        
        return data[torch.tensor(sampled_indices[:batch_size])]


class AdaptivePhysicsSampling(SamplingStrategy):
    """Adaptive sampling for physics points based on residual magnitudes."""
    
    def __init__(self, residual_threshold: float = 0.1, adaptive_ratio: float = 0.3):
        """
        Initialize adaptive physics sampling.
        
        Args:
            residual_threshold: Threshold for high-residual points
            adaptive_ratio: Ratio of batch to sample adaptively
        """
        self.residual_threshold = residual_threshold
        self.adaptive_ratio = adaptive_ratio
        self.residual_history = {}
    
    def update_residuals(self, points: torch.Tensor, residuals: torch.Tensor):
        """Update residual history for points."""
        for i, point in enumerate(points):
            point_key = tuple(point.cpu().numpy())
            self.residual_history[point_key] = residuals[i].item()
    
    def sample(self, data: torch.Tensor, batch_size: int, **kwargs) -> torch.Tensor:
        """Sample with adaptive strategy based on residuals."""
        if len(data) <= batch_size:
            return data
        
        adaptive_size = int(batch_size * self.adaptive_ratio)
        random_size = batch_size - adaptive_size
        
        # Adaptive sampling based on residuals
        adaptive_indices = []
        if self.residual_history:
            # Find high-residual points
            high_residual_indices = []
            for i, point in enumerate(data):
                point_key = tuple(point.cpu().numpy())
                if point_key in self.residual_history:
                    if self.residual_history[point_key] > self.residual_threshold:
                        high_residual_indices.append(i)
            
            # Sample from high-residual points
            if high_residual_indices:
                n_adaptive = min(adaptive_size, len(high_residual_indices))
                adaptive_indices = random.sample(high_residual_indices, n_adaptive)
        
        # Random sampling for remaining points
        remaining_indices = [i for i in range(len(data)) if i not in adaptive_indices]
        if remaining_indices:
            n_random = min(random_size + (adaptive_size - len(adaptive_indices)), len(remaining_indices))
            random_indices = random.sample(remaining_indices, n_random)
        else:
            random_indices = []
        
        # Combine indices
        all_indices = adaptive_indices + random_indices
        
        # Fill to batch size if needed
        while len(all_indices) < batch_size:
            remaining = [i for i in range(len(data)) if i not in all_indices]
            if remaining:
                all_indices.append(random.choice(remaining))
            else:
                break
        
        return data[torch.tensor(all_indices[:batch_size])]


class WellPointSampler:
    """
    Specialized sampler for well log data points.
    """
    
    def __init__(self, 
                 wells_per_batch: int = 4,
                 points_per_well: int = 128,
                 depth_strategy: str = "uniform"):
        """
        Initialize well point sampler.
        
        Args:
            wells_per_batch: Number of wells to sample per batch
            points_per_well: Number of points to sample per well
            depth_strategy: Depth sampling strategy
        """
        self.wells_per_batch = wells_per_batch
        self.points_per_well = points_per_well
        self.depth_strategy = depth_strategy
        
    def sample_wells(self, well_data_list: List[WellData]) -> List[WellData]:
        """Sample wells for batch."""
        if len(well_data_list) <= self.wells_per_batch:
            return well_data_list
        
        return random.sample(well_data_list, self.wells_per_batch)
    
    def sample_points_from_well(self, well_data: WellData) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Sample points from a single well.
        
        Args:
            well_data: Well data object
            
        Returns:
            Tuple of (sampled_depths, sampled_curves)
        """
        total_points = len(well_data.depth)
        
        if total_points <= self.points_per_well:
            return well_data.depth, well_data.curves
        
        # Sample depth indices based on strategy
        if self.depth_strategy == "uniform":
            indices = np.random.choice(total_points, self.points_per_well, replace=False)
        elif self.depth_strategy == "log_uniform":
            # Log-uniform sampling favors deeper points
            log_depths = np.log(well_data.depth - well_data.depth.min() + 1)
            probabilities = log_depths / log_depths.sum()
            indices = np.random.choice(total_points, self.points_per_well, replace=False, p=probabilities)
        elif self.depth_strategy == "adaptive":
            # Adaptive sampling based on curve variability
            variability_scores = self._compute_variability_scores(well_data)
            probabilities = variability_scores / variability_scores.sum()
            indices = np.random.choice(total_points, self.points_per_well, replace=False, p=probabilities)
        else:
            raise ValueError(f"Unknown depth strategy: {self.depth_strategy}")
        
        # Sort indices to maintain depth order
        indices = np.sort(indices)
        
        # Sample data
        sampled_depths = well_data.depth[indices]
        sampled_curves = {}
        for curve_name, curve_data in well_data.curves.items():
            sampled_curves[curve_name] = curve_data[indices]
        
        return sampled_depths, sampled_curves
    
    def _compute_variability_scores(self, well_data: WellData) -> np.ndarray:
        """Compute variability scores for adaptive sampling."""
        scores = np.ones(len(well_data.depth))
        
        # Compute local variability for each curve
        for curve_name, curve_data in well_data.curves.items():
            if len(curve_data) > 2:
                # Compute local gradient magnitude
                gradient = np.gradient(curve_data)
                gradient_magnitude = np.abs(gradient)
                
                # Smooth and normalize
                from scipy.ndimage import gaussian_filter1d
                smoothed_gradient = gaussian_filter1d(gradient_magnitude, sigma=2)
                normalized_gradient = smoothed_gradient / (smoothed_gradient.max() + 1e-8)
                
                scores += normalized_gradient
        
        # Normalize scores
        scores = scores / scores.sum()
        return scores


class BatchProcessor:
    """
    Main batch processor for PINN training data handling.
    
    Features:
    - Efficient data sampling and batching for different data types
    - Gradient accumulation support
    - Data shuffling and sampling strategies for well point selection
    - Memory-efficient batch generation
    - Support for multiple sampling strategies
    """
    
    def __init__(self, config: Optional[BatchConfig] = None, device: str = 'cpu'):
        """
        Initialize batch processor.
        
        Args:
            config: Batch processing configuration
            device: Device for tensor operations
        """
        self.config = config if config is not None else BatchConfig()
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize sampling strategies
        self.data_sampler = self._create_data_sampler()
        self.physics_sampler = self._create_physics_sampler()
        self.well_sampler = WellPointSampler(
            wells_per_batch=self.config.wells_per_batch,
            points_per_well=self.config.points_per_well,
            depth_strategy=self.config.depth_sampling_strategy
        )
        
        # State tracking
        self.epoch_count = 0
        self.batch_count = 0
        self.physics_points_cache = None
        
    def _create_data_sampler(self) -> SamplingStrategy:
        """Create data sampling strategy."""
        if self.config.data_sampling_strategy == "random":
            return RandomSampling()
        elif self.config.data_sampling_strategy == "stratified":
            return StratifiedSampling()
        else:
            return RandomSampling()
    
    def _create_physics_sampler(self) -> SamplingStrategy:
        """Create physics sampling strategy."""
        if self.config.physics_sampling_strategy == "uniform":
            return RandomSampling()
        elif self.config.physics_sampling_strategy == "adaptive":
            return AdaptivePhysicsSampling()
        else:
            return RandomSampling()
    
    def create_data_batch(self, 
                         well_data_list: List[WellData],
                         target_values: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Create data batch from well data.
        
        Args:
            well_data_list: List of well data objects
            target_values: Optional target values for supervised learning
            
        Returns:
            Dictionary containing batch tensors
        """
        # Sample wells for this batch
        sampled_wells = self.well_sampler.sample_wells(well_data_list)
        
        # Collect data from sampled wells
        batch_inputs = []
        batch_targets = []
        batch_coords = []
        batch_well_ids = []
        
        for well_data in sampled_wells:
            # Sample points from this well
            depths, curves = self.well_sampler.sample_points_from_well(well_data)
            
            # Create input features (depth + curve values)
            well_inputs = [depths]
            for curve_name in sorted(curves.keys()):
                well_inputs.append(curves[curve_name])
            
            well_input_array = np.column_stack(well_inputs)
            batch_inputs.append(well_input_array)
            
            # Coordinates (for physics loss)
            coords = np.column_stack([depths, np.full_like(depths, 0), np.full_like(depths, 0)])  # x, y, z
            batch_coords.append(coords)
            
            # Well IDs
            batch_well_ids.extend([well_data.well_id] * len(depths))
            
            # Targets (if provided)
            if target_values is not None:
                # This would need to be implemented based on how targets are structured
                pass
        
        # Concatenate all well data
        if batch_inputs:
            all_inputs = np.vstack(batch_inputs)
            all_coords = np.vstack(batch_coords)
            
            # Convert to tensors
            inputs_tensor = torch.tensor(all_inputs, dtype=torch.float32, device=self.device)
            coords_tensor = torch.tensor(all_coords, dtype=torch.float32, device=self.device)
            
            # Sample from combined data
            if len(inputs_tensor) > self.config.data_batch_size:
                sampled_inputs = self.data_sampler.sample(inputs_tensor, self.config.data_batch_size)
                sampled_coords = self.data_sampler.sample(coords_tensor, self.config.data_batch_size)
            else:
                sampled_inputs = inputs_tensor
                sampled_coords = coords_tensor
            
            batch = {
                'inputs': sampled_inputs,
                'coords': sampled_coords,
                'well_ids': batch_well_ids[:len(sampled_inputs)]
            }
            
            # Add targets if available
            if target_values is not None and len(target_values) >= len(sampled_inputs):
                batch['targets'] = target_values[:len(sampled_inputs)]
            
            return batch
        
        # Return empty batch if no data
        return {
            'inputs': torch.empty(0, 4, device=self.device),
            'coords': torch.empty(0, 3, device=self.device),
            'well_ids': []
        }
    
    def create_physics_batch(self, 
                           domain_bounds: Dict[str, Tuple[float, float]],
                           material_properties: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Create physics batch with collocation points.
        
        Args:
            domain_bounds: Dictionary of domain boundaries for each dimension
            material_properties: Optional material property tensors
            
        Returns:
            Dictionary containing physics batch tensors
        """
        # Generate or reuse physics points
        if (self.physics_points_cache is None or 
            self.epoch_count % self.config.resample_physics_frequency == 0):
            self.physics_points_cache = self._generate_physics_points(domain_bounds)
        
        # Sample from physics points
        physics_coords = self.physics_sampler.sample(
            self.physics_points_cache, 
            self.config.physics_batch_size
        )
        
        batch = {
            'coords': physics_coords,
            'material_properties': material_properties or {}
        }
        
        return batch
    
    def _generate_physics_points(self, 
                               domain_bounds: Dict[str, Tuple[float, float]]) -> torch.Tensor:
        """Generate physics collocation points."""
        # Generate points in the domain
        n_points = self.config.physics_batch_size * 10  # Generate more points for sampling
        
        points = []
        for dim_name, (min_val, max_val) in domain_bounds.items():
            if self.config.physics_sampling_strategy == "uniform":
                dim_points = torch.rand(n_points) * (max_val - min_val) + min_val
            else:
                dim_points = torch.rand(n_points) * (max_val - min_val) + min_val
            
            points.append(dim_points)
        
        # Stack dimensions
        physics_points = torch.stack(points, dim=1).to(self.device)
        
        return physics_points
    
    def create_boundary_batch(self, 
                            boundary_conditions: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Create boundary condition batch.
        
        Args:
            boundary_conditions: List of boundary condition specifications
            
        Returns:
            Dictionary containing boundary batch tensors
        """
        boundary_coords = []
        boundary_values = []
        boundary_types = []
        
        for bc in boundary_conditions:
            # Generate points on boundary
            coords = self._generate_boundary_points(bc)
            values = torch.full((len(coords), 1), bc.get('value', 0.0))
            types = [bc.get('type', 'dirichlet')] * len(coords)
            
            boundary_coords.append(coords)
            boundary_values.append(values)
            boundary_types.extend(types)
        
        if boundary_coords:
            all_coords = torch.cat(boundary_coords, dim=0)
            all_values = torch.cat(boundary_values, dim=0)
            
            # Sample if too many points
            if len(all_coords) > self.config.boundary_batch_size:
                indices = torch.randperm(len(all_coords))[:self.config.boundary_batch_size]
                all_coords = all_coords[indices]
                all_values = all_values[indices]
                boundary_types = [boundary_types[i] for i in indices]
            
            return {
                'coords': all_coords.to(self.device),
                'values': all_values.to(self.device),
                'types': boundary_types
            }
        
        return {
            'coords': torch.empty(0, 3, device=self.device),
            'values': torch.empty(0, 1, device=self.device),
            'types': []
        }
    
    def _generate_boundary_points(self, boundary_condition: Dict[str, Any]) -> torch.Tensor:
        """Generate points on a boundary."""
        # This is a simplified implementation
        # In practice, this would generate points based on boundary geometry
        n_points = 64
        
        if boundary_condition.get('location') == 'inlet':
            # Generate points at x=0
            coords = torch.zeros(n_points, 3)
            coords[:, 1] = torch.linspace(0, 1, n_points)  # y coordinates
            coords[:, 2] = torch.linspace(0, 1, n_points)  # z coordinates
        elif boundary_condition.get('location') == 'outlet':
            # Generate points at x=1
            coords = torch.ones(n_points, 3)
            coords[:, 1] = torch.linspace(0, 1, n_points)
            coords[:, 2] = torch.linspace(0, 1, n_points)
        else:
            # Default: random points in domain
            coords = torch.rand(n_points, 3)
        
        return coords
    
    def update_epoch(self):
        """Update epoch counter and trigger resampling if needed."""
        self.epoch_count += 1
        
        # Clear physics points cache if resampling is due
        if self.epoch_count % self.config.resample_physics_frequency == 0:
            self.physics_points_cache = None
    
    def update_physics_residuals(self, 
                               points: torch.Tensor,
                               residuals: torch.Tensor):
        """Update physics sampler with residual information."""
        if isinstance(self.physics_sampler, AdaptivePhysicsSampling):
            self.physics_sampler.update_residuals(points, residuals)
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size considering gradient accumulation."""
        if self.config.effective_batch_size is not None:
            return self.config.effective_batch_size
        return self.config.data_batch_size * self.config.gradient_accumulation_steps
    
    def should_accumulate_gradients(self) -> bool:
        """Check if gradients should be accumulated."""
        return self.config.gradient_accumulation_steps > 1
    
    def get_accumulation_steps(self) -> int:
        """Get number of gradient accumulation steps."""
        return self.config.gradient_accumulation_steps
    
    def get_batch_info(self) -> Dict[str, Any]:
        """Get information about batch processing configuration."""
        return {
            'data_batch_size': self.config.data_batch_size,
            'physics_batch_size': self.config.physics_batch_size,
            'boundary_batch_size': self.config.boundary_batch_size,
            'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
            'effective_batch_size': self.get_effective_batch_size(),
            'wells_per_batch': self.config.wells_per_batch,
            'points_per_well': self.config.points_per_well,
            'epoch_count': self.epoch_count,
            'batch_count': self.batch_count
        }


# Factory functions for common configurations

def create_standard_batch_processor(device: str = 'cpu') -> BatchProcessor:
    """Create batch processor with standard configuration."""
    config = BatchConfig(
        data_batch_size=512,
        physics_batch_size=1024,
        boundary_batch_size=256,
        wells_per_batch=4,
        points_per_well=128
    )
    return BatchProcessor(config, device)


def create_large_batch_processor(device: str = 'cpu') -> BatchProcessor:
    """Create batch processor for large-scale training."""
    config = BatchConfig(
        data_batch_size=1024,
        physics_batch_size=2048,
        boundary_batch_size=512,
        gradient_accumulation_steps=2,
        wells_per_batch=8,
        points_per_well=256
    )
    return BatchProcessor(config, device)


def create_memory_efficient_processor(device: str = 'cpu') -> BatchProcessor:
    """Create memory-efficient batch processor."""
    config = BatchConfig(
        data_batch_size=256,
        physics_batch_size=512,
        boundary_batch_size=128,
        gradient_accumulation_steps=4,
        wells_per_batch=2,
        points_per_well=64
    )
    return BatchProcessor(config, device)