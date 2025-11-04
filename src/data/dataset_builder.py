"""
Dataset creation and management for PINN training.

This module provides functionality to combine processed wells into training/validation
datasets with proper tensor conversion and data splitting.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

from ..core.interfaces import DatasetInterface
from ..core.data_models import WellData, BatchData

# Set up logging
logger = logging.getLogger(__name__)


class PINNDataset(Dataset, DatasetInterface):
    """
    PyTorch Dataset for PINN training with well log data.
    """
    
    def __init__(self, 
                 well_data_list: List[WellData],
                 input_features: List[str],
                 output_features: List[str],
                 coordinate_features: Optional[List[str]] = None,
                 device: str = 'cpu'):
        """
        Initialize the PINN dataset.
        
        Args:
            well_data_list: List of processed WellData objects
            input_features: List of input feature names
            output_features: List of output feature names  
            coordinate_features: List of coordinate feature names (e.g., ['depth'])
            device: Device to store tensors on
        """
        self.well_data_list = well_data_list
        self.input_features = input_features
        self.output_features = output_features
        self.coordinate_features = coordinate_features or ['depth']
        self.device = device
        
        # Build the dataset
        self._build_dataset()
        
        logger.info(f"Created PINN dataset with {len(self)} samples")
        logger.info(f"Input features: {self.input_features}")
        logger.info(f"Output features: {self.output_features}")
        logger.info(f"Coordinate features: {self.coordinate_features}")
    
    def _build_dataset(self):
        """Build the dataset from well data."""
        all_inputs = []
        all_outputs = []
        all_coordinates = []
        all_well_ids = []
        
        for well_data in self.well_data_list:
            # Extract input features
            input_data = self._extract_features(well_data, self.input_features)
            if input_data is None:
                continue
            
            # Extract output features (if available)
            output_data = self._extract_features(well_data, self.output_features)
            
            # Extract coordinate features
            coord_data = self._extract_coordinates(well_data)
            
            # Find valid samples (no NaN in inputs)
            valid_mask = ~np.any(np.isnan(input_data), axis=1)
            
            if np.sum(valid_mask) == 0:
                logger.warning(f"No valid samples in well {well_data.well_id}")
                continue
            
            # Apply valid mask
            valid_inputs = input_data[valid_mask]
            valid_coords = coord_data[valid_mask] if coord_data is not None else None
            valid_outputs = output_data[valid_mask] if output_data is not None else None
            
            # Store data
            all_inputs.append(valid_inputs)
            all_coordinates.append(valid_coords)
            if valid_outputs is not None:
                all_outputs.append(valid_outputs)
            
            # Store well IDs for each sample
            well_ids = [well_data.well_id] * len(valid_inputs)
            all_well_ids.extend(well_ids)
        
        # Combine all data
        if all_inputs:
            self.inputs = np.vstack(all_inputs)
            self.coordinates = np.vstack(all_coordinates) if all_coordinates[0] is not None else None
            self.outputs = np.vstack(all_outputs) if all_outputs else None
            self.well_ids = all_well_ids
        else:
            raise ValueError("No valid data found in any wells")
        
        # Convert to tensors
        self.inputs_tensor = torch.FloatTensor(self.inputs).to(self.device)
        self.coordinates_tensor = torch.FloatTensor(self.coordinates).to(self.device) if self.coordinates is not None else None
        self.outputs_tensor = torch.FloatTensor(self.outputs).to(self.device) if self.outputs is not None else None
        
        logger.info(f"Dataset built with {len(self.inputs)} samples from {len(self.well_data_list)} wells")
    
    def _extract_features(self, well_data: WellData, feature_names: List[str]) -> Optional[np.ndarray]:
        """Extract specified features from well data."""
        features = []
        
        for feature_name in feature_names:
            if feature_name == 'depth':
                feature_values = well_data.depth
            elif feature_name in well_data.curves:
                feature_values = well_data.curves[feature_name]
            else:
                logger.warning(f"Feature {feature_name} not found in well {well_data.well_id}")
                return None
            
            features.append(feature_values)
        
        # Stack features as columns
        feature_array = np.column_stack(features)
        return feature_array
    
    def _extract_coordinates(self, well_data: WellData) -> Optional[np.ndarray]:
        """Extract coordinate features from well data."""
        if not self.coordinate_features:
            return None
        
        coordinates = []
        
        for coord_name in self.coordinate_features:
            if coord_name == 'depth':
                coord_values = well_data.depth
            elif coord_name in well_data.curves:
                coord_values = well_data.curves[coord_name]
            else:
                logger.warning(f"Coordinate {coord_name} not found in well {well_data.well_id}")
                return None
            
            coordinates.append(coord_values)
        
        # Stack coordinates as columns
        coord_array = np.column_stack(coordinates)
        return coord_array
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> BatchData:
        """Get item at index."""
        inputs = self.inputs_tensor[idx]
        coordinates = self.coordinates_tensor[idx] if self.coordinates_tensor is not None else None
        outputs = self.outputs_tensor[idx] if self.outputs_tensor is not None else None
        well_id = self.well_ids[idx] if idx < len(self.well_ids) else None
        
        return BatchData(
            inputs=inputs.cpu().numpy(),
            targets=outputs.cpu().numpy() if outputs is not None else None,
            coordinates=coordinates.cpu().numpy() if coordinates is not None else None,
            well_ids=[well_id] if well_id else None
        )
    
    def split(self, train_ratio: float) -> Tuple['PINNDataset', 'PINNDataset']:
        """
        Split dataset into train and validation sets.
        
        Args:
            train_ratio: Fraction of data for training
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        # Split indices
        indices = np.arange(len(self))
        train_indices, val_indices = train_test_split(
            indices, train_size=train_ratio, random_state=42, shuffle=True
        )
        
        # Create subset datasets
        train_dataset = self._create_subset(train_indices)
        val_dataset = self._create_subset(val_indices)
        
        logger.info(f"Split dataset: {len(train_dataset)} training, {len(val_dataset)} validation")
        
        return train_dataset, val_dataset
    
    def _create_subset(self, indices: np.ndarray) -> 'PINNDataset':
        """Create a subset dataset with given indices."""
        # Create new dataset with subset of data
        subset_dataset = PINNDataset.__new__(PINNDataset)
        
        # Copy basic attributes
        subset_dataset.input_features = self.input_features
        subset_dataset.output_features = self.output_features
        subset_dataset.coordinate_features = self.coordinate_features
        subset_dataset.device = self.device
        
        # Subset the data
        subset_dataset.inputs = self.inputs[indices]
        subset_dataset.coordinates = self.coordinates[indices] if self.coordinates is not None else None
        subset_dataset.outputs = self.outputs[indices] if self.outputs is not None else None
        subset_dataset.well_ids = [self.well_ids[i] for i in indices]
        
        # Convert to tensors
        subset_dataset.inputs_tensor = torch.FloatTensor(subset_dataset.inputs).to(self.device)
        subset_dataset.coordinates_tensor = torch.FloatTensor(subset_dataset.coordinates).to(self.device) if subset_dataset.coordinates is not None else None
        subset_dataset.outputs_tensor = torch.FloatTensor(subset_dataset.outputs).to(self.device) if subset_dataset.outputs is not None else None
        
        return subset_dataset
    
    def get_batch(self, batch_size: int) -> BatchData:
        """
        Get random batch of specified size.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            BatchData object containing the batch
        """
        # Sample random indices
        indices = np.random.choice(len(self), size=min(batch_size, len(self)), replace=False)
        
        # Extract batch data
        batch_inputs = self.inputs_tensor[indices]
        batch_coordinates = self.coordinates_tensor[indices] if self.coordinates_tensor is not None else None
        batch_outputs = self.outputs_tensor[indices] if self.outputs_tensor is not None else None
        batch_well_ids = [self.well_ids[i] for i in indices]
        
        return BatchData(
            inputs=batch_inputs.cpu().numpy(),
            targets=batch_outputs.cpu().numpy() if batch_outputs is not None else None,
            coordinates=batch_coordinates.cpu().numpy() if batch_coordinates is not None else None,
            well_ids=batch_well_ids
        )
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for input and output features."""
        stats = {}
        
        # Input statistics
        stats['inputs'] = {}
        for i, feature_name in enumerate(self.input_features):
            feature_data = self.inputs[:, i]
            stats['inputs'][feature_name] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data))
            }
        
        # Output statistics (if available)
        if self.outputs is not None:
            stats['outputs'] = {}
            for i, feature_name in enumerate(self.output_features):
                feature_data = self.outputs[:, i]
                stats['outputs'][feature_name] = {
                    'mean': float(np.mean(feature_data)),
                    'std': float(np.std(feature_data)),
                    'min': float(np.min(feature_data)),
                    'max': float(np.max(feature_data))
                }
        
        return stats


class DatasetBuilder:
    """
    Builder class for creating PINN datasets from processed well data.
    """
    
    def __init__(self, 
                 input_features: Optional[List[str]] = None,
                 output_features: Optional[List[str]] = None,
                 coordinate_features: Optional[List[str]] = None):
        """
        Initialize the dataset builder.
        
        Args:
            input_features: Default input features
            output_features: Default output features
            coordinate_features: Default coordinate features
        """
        self.input_features = input_features or [
            'porosity', 'permeability', 'gamma_ray', 'neutron_porosity'
        ]
        self.output_features = output_features or [
            'pressure', 'saturation'
        ]
        self.coordinate_features = coordinate_features or ['depth']
    
    def build_dataset(self, 
                     well_data_list: List[WellData],
                     input_features: Optional[List[str]] = None,
                     output_features: Optional[List[str]] = None,
                     coordinate_features: Optional[List[str]] = None,
                     device: str = 'cpu') -> PINNDataset:
        """
        Build a PINN dataset from well data.
        
        Args:
            well_data_list: List of WellData objects
            input_features: Input feature names (overrides default)
            output_features: Output feature names (overrides default)
            coordinate_features: Coordinate feature names (overrides default)
            device: Device to store tensors on
            
        Returns:
            PINNDataset object
        """
        # Use provided features or defaults
        input_features = input_features or self.input_features
        output_features = output_features or self.output_features
        coordinate_features = coordinate_features or self.coordinate_features
        
        # Validate that wells have required features
        valid_wells = self._validate_wells(well_data_list, input_features, coordinate_features)
        
        if not valid_wells:
            raise ValueError("No wells contain the required features")
        
        logger.info(f"Building dataset from {len(valid_wells)} valid wells")
        
        # Create dataset
        dataset = PINNDataset(
            well_data_list=valid_wells,
            input_features=input_features,
            output_features=output_features,
            coordinate_features=coordinate_features,
            device=device
        )
        
        return dataset
    
    def build_train_val_datasets(self, 
                                well_data_list: List[WellData],
                                train_ratio: float = 0.8,
                                input_features: Optional[List[str]] = None,
                                output_features: Optional[List[str]] = None,
                                coordinate_features: Optional[List[str]] = None,
                                device: str = 'cpu') -> Tuple[PINNDataset, PINNDataset]:
        """
        Build training and validation datasets.
        
        Args:
            well_data_list: List of WellData objects
            train_ratio: Fraction of data for training
            input_features: Input feature names
            output_features: Output feature names
            coordinate_features: Coordinate feature names
            device: Device to store tensors on
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        # Build full dataset
        full_dataset = self.build_dataset(
            well_data_list=well_data_list,
            input_features=input_features,
            output_features=output_features,
            coordinate_features=coordinate_features,
            device=device
        )
        
        # Split into train and validation
        train_dataset, val_dataset = full_dataset.split(train_ratio)
        
        return train_dataset, val_dataset
    
    def build_well_holdout_datasets(self, 
                                   well_data_list: List[WellData],
                                   holdout_wells: Union[int, List[str]],
                                   input_features: Optional[List[str]] = None,
                                   output_features: Optional[List[str]] = None,
                                   coordinate_features: Optional[List[str]] = None,
                                   device: str = 'cpu') -> Tuple[PINNDataset, PINNDataset]:
        """
        Build datasets with specific wells held out for validation.
        
        Args:
            well_data_list: List of WellData objects
            holdout_wells: Number of wells to hold out or list of well IDs
            input_features: Input feature names
            output_features: Output feature names
            coordinate_features: Coordinate feature names
            device: Device to store tensors on
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        if isinstance(holdout_wells, int):
            # Randomly select wells to hold out
            well_ids = [well.well_id for well in well_data_list]
            holdout_well_ids = np.random.choice(well_ids, size=holdout_wells, replace=False)
        else:
            holdout_well_ids = holdout_wells
        
        # Split wells
        train_wells = [well for well in well_data_list if well.well_id not in holdout_well_ids]
        val_wells = [well for well in well_data_list if well.well_id in holdout_well_ids]
        
        logger.info(f"Well holdout: {len(train_wells)} training wells, {len(val_wells)} validation wells")
        
        # Build datasets
        train_dataset = self.build_dataset(
            well_data_list=train_wells,
            input_features=input_features,
            output_features=output_features,
            coordinate_features=coordinate_features,
            device=device
        )
        
        val_dataset = self.build_dataset(
            well_data_list=val_wells,
            input_features=input_features,
            output_features=output_features,
            coordinate_features=coordinate_features,
            device=device
        )
        
        return train_dataset, val_dataset
    
    def _validate_wells(self, 
                       well_data_list: List[WellData], 
                       required_features: List[str],
                       coordinate_features: List[str]) -> List[WellData]:
        """Validate that wells contain required features."""
        valid_wells = []
        
        for well_data in well_data_list:
            # Check input features
            missing_features = []
            for feature in required_features:
                if feature != 'depth' and feature not in well_data.curves:
                    missing_features.append(feature)
            
            # Check coordinate features
            for feature in coordinate_features:
                if feature != 'depth' and feature not in well_data.curves:
                    missing_features.append(feature)
            
            if missing_features:
                logger.debug(f"Well {well_data.well_id} missing features: {missing_features}")
            else:
                valid_wells.append(well_data)
        
        return valid_wells


class DataLoaderFactory:
    """
    Factory for creating PyTorch DataLoaders for PINN training.
    """
    
    @staticmethod
    def create_dataloader(dataset: PINNDataset,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 0,
                         pin_memory: bool = False) -> DataLoader:
        """
        Create a PyTorch DataLoader from a PINN dataset.
        
        Args:
            dataset: PINNDataset object
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=DataLoaderFactory._collate_fn
        )
    
    @staticmethod
    def _collate_fn(batch: List[BatchData]) -> Dict[str, torch.Tensor]:
        """Custom collate function for BatchData objects."""
        # Extract data from batch
        inputs = [item.inputs for item in batch]
        coordinates = [item.coordinates for item in batch if item.coordinates is not None]
        targets = [item.targets for item in batch if item.targets is not None]
        well_ids = [item.well_ids[0] for item in batch if item.well_ids is not None]
        
        # Stack into tensors
        batch_dict = {
            'inputs': torch.stack([torch.FloatTensor(inp) for inp in inputs])
        }
        
        if coordinates:
            batch_dict['coordinates'] = torch.stack([torch.FloatTensor(coord) for coord in coordinates])
        
        if targets:
            batch_dict['targets'] = torch.stack([torch.FloatTensor(tgt) for tgt in targets])
        
        if well_ids:
            batch_dict['well_ids'] = well_ids
        
        return batch_dict