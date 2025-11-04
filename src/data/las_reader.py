"""
LAS file reader and parser for well log data.

This module provides functionality to read and parse LAS (Log ASCII Standard) files
and extract curve data for PINN training.
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import lasio
import logging

from ..core.interfaces import LASReaderInterface
from ..core.data_models import WellData, WellMetadata

# Set up logging
logger = logging.getLogger(__name__)


class LASFileReader(LASReaderInterface):
    """
    LAS file reader that parses individual LAS files and extracts curve data.
    
    Handles different LAS file formats and versions with robust error handling.
    """
    
    def __init__(self):
        """Initialize the LAS file reader."""
        self.supported_curves = {
            'gamma_ray': ['GR', 'GAMMA', 'GAMMA_RAY', 'GRC'],
            'density': ['RHOB', 'DENSITY', 'BULK_DENSITY', 'DEN'],
            'neutron_porosity': ['CNPOR', 'NPHI', 'NEUTRON', 'NEUT', 'CNLS', 'CNSS', 'CNDL'],
            'resistivity': ['RILD', 'RES', 'RESISTIVITY', 'RT', 'RILM', 'RLL3'],
            'porosity': ['DPOR', 'POROSITY', 'PHI', 'PORO'],
            'permeability': ['PERM', 'PERMEABILITY', 'K'],
            'depth': ['DEPT', 'DEPTH', 'MD', 'TVDSS']
        }
        
    def read_las_file(self, filepath: str) -> WellData:
        """
        Read and parse a single LAS file.
        
        Args:
            filepath: Path to the LAS file
            
        Returns:
            WellData object containing parsed well data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed or is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LAS file not found: {filepath}")
            
        try:
            # Read LAS file using lasio
            las = lasio.read(filepath)
            
            # Extract well metadata
            metadata = self._extract_metadata(las, filepath)
            
            # Extract depth data
            depth = self._extract_depth(las)
            
            # Extract curve data
            curves = self._extract_all_curves(las)
            
            # Create WellData object
            well_data = WellData(
                well_id=metadata.well_id,
                depth=depth,
                curves=curves,
                metadata=metadata
            )
            
            logger.info(f"Successfully parsed LAS file: {filepath}")
            logger.info(f"Well ID: {well_data.well_id}, Depth range: {depth.min():.1f} - {depth.max():.1f}")
            logger.info(f"Available curves: {list(curves.keys())}")
            
            return well_data
            
        except Exception as e:
            logger.error(f"Error parsing LAS file {filepath}: {str(e)}")
            raise ValueError(f"Failed to parse LAS file {filepath}: {str(e)}")
    
    def extract_curves(self, well_data: WellData) -> Dict[str, np.ndarray]:
        """
        Extract specific curves from well data.
        
        Args:
            well_data: WellData object
            
        Returns:
            Dictionary mapping curve names to numpy arrays
        """
        return well_data.curves.copy()
    
    def get_well_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Extract metadata from LAS file without full parsing.
        
        Args:
            filepath: Path to the LAS file
            
        Returns:
            Dictionary containing well metadata
        """
        try:
            las = lasio.read(filepath)
            metadata = self._extract_metadata(las, filepath)
            
            return {
                'well_id': metadata.well_id,
                'location': metadata.location,
                'formation': metadata.formation,
                'date_logged': metadata.date_logged,
                'total_depth': metadata.total_depth,
                'curve_units': metadata.curve_units,
                'available_curves': list(las.curves.keys())
            }
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {filepath}: {str(e)}")
            return {}
    
    def _extract_metadata(self, las: lasio.LASFile, filepath: str) -> WellMetadata:
        """Extract metadata from LAS file object."""
        # Extract well ID
        well_id = self._safe_get_header_value(las, ['WELL', 'UWI', 'API'])
        if not well_id:
            # Use filename as fallback
            well_id = os.path.splitext(os.path.basename(filepath))[0]
        
        # Extract location information
        location = self._extract_location(las)
        
        # Extract formation
        formation = self._safe_get_header_value(las, ['FLD', 'FIELD', 'FORMATION'])
        if not formation:
            formation = "Unknown"
        
        # Extract date
        date_logged = self._extract_date(las)
        
        # Extract depth information
        total_depth = self._extract_total_depth(las)
        
        # Extract curve units
        curve_units = {}
        for curve in las.curves:
            curve_units[curve.mnemonic] = curve.unit if curve.unit else ""
        
        # Extract additional metadata
        surface_elevation = self._safe_get_numeric_value(las, ['EGL', 'EEL', 'ELEV'])
        operator = self._safe_get_header_value(las, ['COMP', 'COMPANY', 'OPERATOR'])
        field_name = self._safe_get_header_value(las, ['FLD', 'FIELD'])
        
        return WellMetadata(
            well_id=well_id,
            location=location,
            formation=formation,
            date_logged=date_logged,
            curve_units=curve_units,
            total_depth=total_depth,
            surface_elevation=surface_elevation,
            operator=operator,
            field_name=field_name
        )
    
    def _extract_depth(self, las: lasio.LASFile) -> np.ndarray:
        """Extract depth data from LAS file."""
        # Try to find depth curve
        depth_curve = None
        for curve_name in self.supported_curves['depth']:
            if curve_name in las.curves:
                depth_curve = las.curves[curve_name]
                break
        
        if depth_curve is None:
            # Use index if no explicit depth curve found
            if hasattr(las, 'index') and las.index is not None:
                depth = np.array(las.index)
            else:
                raise ValueError("No depth information found in LAS file")
        else:
            depth = np.array(depth_curve.data)
        
        # Remove null values
        null_value = las.well.NULL.value if las.well.NULL else -999.25
        valid_mask = depth != null_value
        depth = depth[valid_mask]
        
        if len(depth) == 0:
            raise ValueError("No valid depth data found")
        
        return depth
    
    def _extract_all_curves(self, las: lasio.LASFile) -> Dict[str, np.ndarray]:
        """Extract all available curves from LAS file."""
        curves = {}
        null_value = las.well.NULL.value if las.well.NULL else -999.25
        
        # Get depth for masking
        depth = self._extract_depth(las)
        valid_indices = np.arange(len(depth))
        
        for curve in las.curves:
            curve_name = curve.mnemonic
            curve_data = np.array(curve.data)
            
            # Skip depth curve as it's handled separately
            if curve_name in self.supported_curves['depth']:
                continue
            
            # Apply same masking as depth
            if len(curve_data) >= len(valid_indices):
                curve_data = curve_data[:len(valid_indices)]
            else:
                # Pad with null values if curve is shorter
                padded_data = np.full(len(valid_indices), null_value)
                padded_data[:len(curve_data)] = curve_data
                curve_data = padded_data
            
            # Replace null values with NaN
            curve_data = np.where(curve_data == null_value, np.nan, curve_data)
            
            # Store curve with standardized name
            standardized_name = self._standardize_curve_name(curve_name)
            curves[standardized_name] = curve_data
        
        return curves
    
    def _standardize_curve_name(self, curve_name: str) -> str:
        """Standardize curve names to common conventions."""
        curve_name_upper = curve_name.upper()
        
        # Check against supported curves
        for standard_name, aliases in self.supported_curves.items():
            if curve_name_upper in [alias.upper() for alias in aliases]:
                return standard_name
        
        # Return original name if no match found
        return curve_name.lower()
    
    def _extract_location(self, las: lasio.LASFile) -> Tuple[float, float]:
        """Extract location coordinates from LAS file."""
        # Try to extract coordinates from various fields
        lat = self._safe_get_numeric_value(las, ['LAT', 'LATITUDE', 'Y'])
        lon = self._safe_get_numeric_value(las, ['LON', 'LONGITUDE', 'X'])
        
        if lat is not None and lon is not None:
            return (lat, lon)
        
        # Try to parse location from LOC field
        loc_str = self._safe_get_header_value(las, ['LOC', 'LOCATION'])
        if loc_str:
            coords = self._parse_location_string(loc_str)
            if coords:
                return coords
        
        # Default to (0, 0) if no location found
        return (0.0, 0.0)
    
    def _parse_location_string(self, loc_str: str) -> Optional[Tuple[float, float]]:
        """Parse location string to extract coordinates."""
        # This is a simplified parser - real implementation would be more robust
        try:
            # Look for decimal coordinates
            coord_pattern = r'(-?\d+\.?\d*)'
            matches = re.findall(coord_pattern, loc_str)
            if len(matches) >= 2:
                return (float(matches[0]), float(matches[1]))
        except:
            pass
        
        return None
    
    def _extract_date(self, las: lasio.LASFile) -> Optional[datetime]:
        """Extract logging date from LAS file."""
        date_str = self._safe_get_header_value(las, ['DATE', 'DATE1', 'LOG_DATE'])
        if not date_str:
            return None
        
        # Try various date formats
        date_formats = [
            '%m/%d/%Y',
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%Y/%m/%d',
            '%d/%m/%Y'
        ]
        
        # Clean up date string
        date_str = re.sub(r'[^\d/\-]', '', date_str)
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def _extract_total_depth(self, las: lasio.LASFile) -> float:
        """Extract total depth from LAS file."""
        # Try STOP depth first
        stop_depth = self._safe_get_numeric_value(las, ['STOP', 'STPD', 'TD'])
        if stop_depth is not None:
            return stop_depth
        
        # Try TDD (Total Depth Driller)
        tdd = self._safe_get_numeric_value(las, ['TDD', 'TDD1', 'TOTAL_DEPTH'])
        if tdd is not None:
            return tdd
        
        # Use maximum depth from data
        try:
            depth = self._extract_depth(las)
            return float(np.max(depth))
        except:
            return 0.0
    
    def _safe_get_header_value(self, las: lasio.LASFile, keys: List[str]) -> Optional[str]:
        """Safely get header value from LAS file."""
        for key in keys:
            try:
                if hasattr(las.well, key):
                    value = getattr(las.well, key)
                    if hasattr(value, 'value'):
                        return str(value.value) if value.value is not None else None
                    return str(value) if value is not None else None
            except:
                continue
        return None
    
    def _safe_get_numeric_value(self, las: lasio.LASFile, keys: List[str]) -> Optional[float]:
        """Safely get numeric value from LAS file."""
        for key in keys:
            try:
                if hasattr(las.well, key):
                    value = getattr(las.well, key)
                    if hasattr(value, 'value'):
                        return float(value.value) if value.value is not None else None
                    return float(value) if value is not None else None
            except:
                continue
        return None
    
    def process(self, data: str) -> WellData:
        """Process LAS file path and return WellData."""
        return self.read_las_file(data)
    
    def validate_data(self, data: str) -> bool:
        """Validate that the file path exists and is a LAS file."""
        if not isinstance(data, str):
            return False
        
        if not os.path.exists(data):
            return False
        
        # Check file extension
        _, ext = os.path.splitext(data)
        return ext.lower() in ['.las', '.LAS']


class LASFileProcessor:
    """
    High-level processor for handling multiple LAS files.
    """
    
    def __init__(self):
        """Initialize the LAS file processor."""
        self.reader = LASFileReader()
    
    def process_directory(self, directory_path: str) -> List[WellData]:
        """
        Process all LAS files in a directory.
        
        Args:
            directory_path: Path to directory containing LAS files
            
        Returns:
            List of WellData objects
        """
        well_data_list = []
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return well_data_list
        
        las_files = [f for f in os.listdir(directory_path) 
                    if f.lower().endswith('.las')]
        
        logger.info(f"Found {len(las_files)} LAS files in {directory_path}")
        
        for filename in las_files:
            filepath = os.path.join(directory_path, filename)
            try:
                well_data = self.reader.read_las_file(filepath)
                well_data_list.append(well_data)
            except Exception as e:
                logger.warning(f"Failed to process {filename}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(well_data_list)} LAS files")
        return well_data_list
    
    def get_available_curves(self, well_data_list: List[WellData]) -> Dict[str, int]:
        """
        Get summary of available curves across all wells.
        
        Args:
            well_data_list: List of WellData objects
            
        Returns:
            Dictionary mapping curve names to count of wells containing them
        """
        curve_counts = {}
        
        for well_data in well_data_list:
            for curve_name in well_data.curves.keys():
                curve_counts[curve_name] = curve_counts.get(curve_name, 0) + 1
        
        return curve_counts