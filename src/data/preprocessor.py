"""
Data preprocessing and quality filtering for well log data.

This module provides functionality for cleaning, normalizing, and filtering
well log data to prepare it for PINN training.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

from ..core.interfaces import PreprocessorInterface
from ..core.data_models import WellData

# Set up logging
logger = logging.getLogger(__name__)


class DataPreprocessor(PreprocessorInterface):
    """
    Data preprocessor for cleaning, normalization, and standardization of well log curves.
    """
    
    def __init__(self, 
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 3.0,
                 interpolation_method: str = 'linear',
                 max_gap_size: int = 10):
        """
        Initialize the data preprocessor.
        
        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            outlier_threshold: Threshold for outlier detection
            interpolation_method: Method for interpolating missing values
            max_gap_size: Maximum gap size to interpolate (in data points)
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.interpolation_method = interpolation_method
        self.max_gap_size = max_gap_size
        
        # Curve-specific valid ranges (typical values for well logs)
        self.valid_ranges = {
            'gamma_ray': (0, 300),      # GAPI
            'density': (1.0, 3.5),      # g/cc
            'neutron_porosity': (0, 60), # %
            'resistivity': (0.1, 10000), # ohm-m
            'porosity': (0, 50),        # %
            'permeability': (0.01, 10000) # mD
        }
        
        # Scalers for normalization
        self.scalers = {}
        
    def process(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Process curve data through complete preprocessing pipeline.
        
        Args:
            data: Dictionary of curve name -> values
            
        Returns:
            Processed curve data
        """
        # Step 1: Clean data
        cleaned_data = self.clean_data(data)
        
        # Step 2: Handle missing values
        interpolated_data = self.handle_missing_values(cleaned_data)
        
        # Step 3: Normalize curves
        normalized_data = self.normalize_curves(interpolated_data)
        
        return normalized_data
    
    def clean_data(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Clean data by removing invalid values and outliers.
        
        Args:
            curves: Dictionary mapping curve names to numpy arrays
            
        Returns:
            Cleaned curve data
        """
        cleaned_curves = {}
        
        for curve_name, values in curves.items():
            logger.debug(f"Cleaning curve: {curve_name}")
            
            # Convert to numpy array if not already
            values = np.array(values, dtype=float)
            
            # Step 1: Remove obviously invalid values
            cleaned_values = self._remove_invalid_values(curve_name, values)
            
            # Step 2: Detect and handle outliers
            cleaned_values = self._remove_outliers(curve_name, cleaned_values)
            
            cleaned_curves[curve_name] = cleaned_values
            
            # Log cleaning statistics
            original_count = len(values)
            valid_count = np.sum(~np.isnan(cleaned_values))
            logger.debug(f"Curve {curve_name}: {valid_count}/{original_count} valid values")
        
        return cleaned_curves
    
    def normalize_curves(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize curve values to standard ranges.
        
        Args:
            curves: Dictionary mapping curve names to numpy arrays
            
        Returns:
            Normalized curve data
        """
        normalized_curves = {}
        
        for curve_name, values in curves.items():
            logger.debug(f"Normalizing curve: {curve_name}")
            
            # Skip if all values are NaN
            if np.all(np.isnan(values)):
                normalized_curves[curve_name] = values
                continue
            
            # Get valid (non-NaN) values for fitting scaler
            valid_mask = ~np.isnan(values)
            valid_values = values[valid_mask]
            
            if len(valid_values) == 0:
                normalized_curves[curve_name] = values
                continue
            
            # Fit scaler on valid values
            scaler = StandardScaler()
            valid_values_reshaped = valid_values.reshape(-1, 1)
            scaler.fit(valid_values_reshaped)
            
            # Transform all values (NaN will remain NaN)
            normalized_values = np.full_like(values, np.nan)
            normalized_values[valid_mask] = scaler.transform(valid_values_reshaped).flatten()
            
            # Store scaler for potential inverse transform
            self.scalers[curve_name] = scaler
            
            normalized_curves[curve_name] = normalized_values
            
            logger.debug(f"Normalized {curve_name}: mean={np.nanmean(normalized_values):.3f}, "
                        f"std={np.nanstd(normalized_values):.3f}")
        
        return normalized_curves
    
    def handle_missing_values(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Handle missing values through interpolation or imputation.
        
        Args:
            curves: Dictionary mapping curve names to numpy arrays
            
        Returns:
            Curve data with missing values handled
        """
        interpolated_curves = {}
        
        for curve_name, values in curves.items():
            logger.debug(f"Handling missing values for curve: {curve_name}")
            
            # Count missing values
            missing_count = np.sum(np.isnan(values))
            total_count = len(values)
            missing_percent = (missing_count / total_count) * 100
            
            logger.debug(f"Curve {curve_name}: {missing_count}/{total_count} "
                        f"({missing_percent:.1f}%) missing values")
            
            if missing_count == 0:
                interpolated_curves[curve_name] = values
                continue
            
            if missing_count == total_count:
                logger.warning(f"Curve {curve_name} has no valid values")
                interpolated_curves[curve_name] = values
                continue
            
            # Interpolate missing values
            interpolated_values = self._interpolate_missing_values(values)
            interpolated_curves[curve_name] = interpolated_values
            
            # Log interpolation results
            remaining_missing = np.sum(np.isnan(interpolated_values))
            logger.debug(f"After interpolation: {remaining_missing} missing values remain")
        
        return interpolated_curves
    
    def _remove_invalid_values(self, curve_name: str, values: np.ndarray) -> np.ndarray:
        """Remove obviously invalid values based on physical constraints."""
        cleaned_values = values.copy()
        
        # Get valid range for this curve type
        if curve_name in self.valid_ranges:
            min_val, max_val = self.valid_ranges[curve_name]
            
            # Mark values outside valid range as NaN
            invalid_mask = (cleaned_values < min_val) | (cleaned_values > max_val)
            cleaned_values[invalid_mask] = np.nan
            
            invalid_count = np.sum(invalid_mask)
            if invalid_count > 0:
                logger.debug(f"Removed {invalid_count} values outside valid range "
                           f"[{min_val}, {max_val}] for {curve_name}")
        
        # Remove infinite values
        inf_mask = np.isinf(cleaned_values)
        cleaned_values[inf_mask] = np.nan
        
        inf_count = np.sum(inf_mask)
        if inf_count > 0:
            logger.debug(f"Removed {inf_count} infinite values for {curve_name}")
        
        return cleaned_values
    
    def _remove_outliers(self, curve_name: str, values: np.ndarray) -> np.ndarray:
        """Remove outliers using specified method."""
        if self.outlier_method == 'none':
            return values
        
        cleaned_values = values.copy()
        valid_mask = ~np.isnan(values)
        
        if np.sum(valid_mask) < 10:  # Need minimum data points for outlier detection
            return cleaned_values
        
        valid_values = values[valid_mask]
        
        if self.outlier_method == 'iqr':
            outlier_mask = self._detect_outliers_iqr(valid_values)
        elif self.outlier_method == 'zscore':
            outlier_mask = self._detect_outliers_zscore(valid_values)
        elif self.outlier_method == 'modified_zscore':
            outlier_mask = self._detect_outliers_modified_zscore(valid_values)
        else:
            logger.warning(f"Unknown outlier method: {self.outlier_method}")
            return cleaned_values
        
        # Apply outlier mask to original array
        valid_indices = np.where(valid_mask)[0]
        outlier_indices = valid_indices[outlier_mask]
        cleaned_values[outlier_indices] = np.nan
        
        outlier_count = np.sum(outlier_mask)
        if outlier_count > 0:
            logger.debug(f"Removed {outlier_count} outliers from {curve_name}")
        
        return cleaned_values
    
    def _detect_outliers_iqr(self, values: np.ndarray) -> np.ndarray:
        """Detect outliers using Interquartile Range method."""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr
        
        return (values < lower_bound) | (values > upper_bound)
    
    def _detect_outliers_zscore(self, values: np.ndarray) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(values))
        return z_scores > self.outlier_threshold
    
    def _detect_outliers_modified_zscore(self, values: np.ndarray) -> np.ndarray:
        """Detect outliers using Modified Z-score method."""
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        if mad == 0:
            return np.zeros(len(values), dtype=bool)
        
        modified_z_scores = 0.6745 * (values - median) / mad
        return np.abs(modified_z_scores) > self.outlier_threshold
    
    def _interpolate_missing_values(self, values: np.ndarray) -> np.ndarray:
        """Interpolate missing values in the array."""
        interpolated_values = values.copy()
        
        # Find valid (non-NaN) indices
        valid_mask = ~np.isnan(values)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2:
            return interpolated_values
        
        # Create interpolation function
        valid_values = values[valid_indices]
        
        try:
            if self.interpolation_method == 'linear':
                interp_func = interpolate.interp1d(
                    valid_indices, valid_values, 
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
            elif self.interpolation_method == 'cubic':
                if len(valid_indices) >= 4:
                    interp_func = interpolate.interp1d(
                        valid_indices, valid_values, 
                        kind='cubic', bounds_error=False, fill_value=np.nan
                    )
                else:
                    # Fall back to linear if not enough points for cubic
                    interp_func = interpolate.interp1d(
                        valid_indices, valid_values, 
                        kind='linear', bounds_error=False, fill_value=np.nan
                    )
            else:
                logger.warning(f"Unknown interpolation method: {self.interpolation_method}")
                return interpolated_values
            
            # Find gaps to interpolate
            missing_indices = np.where(~valid_mask)[0]
            
            for missing_idx in missing_indices:
                # Check if gap is within acceptable size
                if self._is_gap_acceptable(missing_idx, valid_indices):
                    interpolated_values[missing_idx] = interp_func(missing_idx)
            
        except Exception as e:
            logger.warning(f"Interpolation failed: {str(e)}")
        
        return interpolated_values
    
    def _is_gap_acceptable(self, missing_idx: int, valid_indices: np.ndarray) -> bool:
        """Check if a gap is small enough to interpolate."""
        # Find nearest valid points on both sides
        left_valid = valid_indices[valid_indices < missing_idx]
        right_valid = valid_indices[valid_indices > missing_idx]
        
        if len(left_valid) == 0 or len(right_valid) == 0:
            return False  # Can't interpolate at boundaries
        
        left_nearest = left_valid[-1]
        right_nearest = right_valid[0]
        
        gap_size = right_nearest - left_nearest - 1
        return gap_size <= self.max_gap_size
    
    def validate_data(self, data: Dict[str, np.ndarray]) -> bool:
        """Validate that data is suitable for preprocessing."""
        if not isinstance(data, dict):
            return False
        
        if len(data) == 0:
            return False
        
        # Check that all values are numpy arrays
        for curve_name, values in data.items():
            if not isinstance(values, np.ndarray):
                return False
            
            if len(values) == 0:
                return False
        
        return True
    
    def inverse_transform(self, curve_name: str, normalized_values: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized values back to original scale.
        
        Args:
            curve_name: Name of the curve
            normalized_values: Normalized values to transform back
            
        Returns:
            Values in original scale
        """
        if curve_name not in self.scalers:
            logger.warning(f"No scaler found for curve {curve_name}")
            return normalized_values
        
        scaler = self.scalers[curve_name]
        
        # Handle NaN values
        valid_mask = ~np.isnan(normalized_values)
        if np.sum(valid_mask) == 0:
            return normalized_values
        
        original_values = np.full_like(normalized_values, np.nan)
        valid_values = normalized_values[valid_mask].reshape(-1, 1)
        original_values[valid_mask] = scaler.inverse_transform(valid_values).flatten()
        
        return original_values


class WellDataFilter:
    """
    Filter wells based on data quality and required curves.
    """
    
    def __init__(self, 
                 required_curves: Optional[List[str]] = None,
                 min_data_completeness: float = 0.7,
                 min_depth_range: float = 100.0):
        """
        Initialize the well data filter.
        
        Args:
            required_curves: List of curves that must be present
            min_data_completeness: Minimum fraction of valid data required
            min_depth_range: Minimum depth range required (in feet/meters)
        """
        self.required_curves = required_curves or [
            'porosity', 'permeability', 'neutron_porosity', 'gamma_ray'
        ]
        self.min_data_completeness = min_data_completeness
        self.min_depth_range = min_depth_range
    
    def filter_wells(self, well_data_list: List[WellData]) -> List[WellData]:
        """
        Filter wells based on quality criteria.
        
        Args:
            well_data_list: List of WellData objects to filter
            
        Returns:
            Filtered list of WellData objects
        """
        filtered_wells = []
        
        for well_data in well_data_list:
            if self._passes_quality_check(well_data):
                filtered_wells.append(well_data)
            else:
                logger.debug(f"Well {well_data.well_id} filtered out due to quality issues")
        
        logger.info(f"Filtered {len(well_data_list)} wells to {len(filtered_wells)} wells")
        return filtered_wells
    
    def _passes_quality_check(self, well_data: WellData) -> bool:
        """Check if a well passes all quality criteria."""
        # Check depth range
        if not self._check_depth_range(well_data):
            return False
        
        # Check required curves
        if not self._check_required_curves(well_data):
            return False
        
        # Check data completeness
        if not self._check_data_completeness(well_data):
            return False
        
        return True
    
    def _check_depth_range(self, well_data: WellData) -> bool:
        """Check if well has sufficient depth range."""
        depth_range = np.max(well_data.depth) - np.min(well_data.depth)
        return depth_range >= self.min_depth_range
    
    def _check_required_curves(self, well_data: WellData) -> bool:
        """Check if well has all required curves."""
        available_curves = set(well_data.curves.keys())
        required_curves = set(self.required_curves)
        
        missing_curves = required_curves - available_curves
        if missing_curves:
            logger.debug(f"Well {well_data.well_id} missing curves: {missing_curves}")
            return False
        
        return True
    
    def _check_data_completeness(self, well_data: WellData) -> bool:
        """Check if well has sufficient data completeness."""
        for curve_name in self.required_curves:
            if curve_name not in well_data.curves:
                continue
            
            values = well_data.curves[curve_name]
            valid_fraction = np.sum(~np.isnan(values)) / len(values)
            
            if valid_fraction < self.min_data_completeness:
                logger.debug(f"Well {well_data.well_id} curve {curve_name} "
                           f"has low completeness: {valid_fraction:.2f}")
                return False
        
        return True
    
    def get_quality_summary(self, well_data_list: List[WellData]) -> Dict[str, any]:
        """
        Get summary of data quality across wells.
        
        Args:
            well_data_list: List of WellData objects
            
        Returns:
            Dictionary containing quality statistics
        """
        summary = {
            'total_wells': len(well_data_list),
            'wells_with_required_curves': 0,
            'average_completeness': {},
            'depth_range_stats': {
                'min': float('inf'),
                'max': 0,
                'mean': 0
            }
        }
        
        depth_ranges = []
        curve_completeness = {curve: [] for curve in self.required_curves}
        
        for well_data in well_data_list:
            # Check required curves
            if self._check_required_curves(well_data):
                summary['wells_with_required_curves'] += 1
            
            # Depth range
            depth_range = np.max(well_data.depth) - np.min(well_data.depth)
            depth_ranges.append(depth_range)
            
            # Curve completeness
            for curve_name in self.required_curves:
                if curve_name in well_data.curves:
                    values = well_data.curves[curve_name]
                    completeness = np.sum(~np.isnan(values)) / len(values)
                    curve_completeness[curve_name].append(completeness)
        
        # Calculate statistics
        if depth_ranges:
            summary['depth_range_stats'] = {
                'min': np.min(depth_ranges),
                'max': np.max(depth_ranges),
                'mean': np.mean(depth_ranges)
            }
        
        for curve_name, completeness_list in curve_completeness.items():
            if completeness_list:
                summary['average_completeness'][curve_name] = np.mean(completeness_list)
        
        return summary