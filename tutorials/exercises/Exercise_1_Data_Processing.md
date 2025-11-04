# Exercise 1: Data Processing and Quality Assessment

## Objective
Learn to process real well log data, assess data quality, and prepare datasets for PINN training.

## Background
Data quality is crucial for successful PINN training. Poor quality data leads to poor model performance, regardless of the sophistication of the neural network architecture or physics constraints.

## Tasks

### Task 1.1: LAS File Analysis (20 minutes)

**Goal**: Analyze the structure and content of LAS files from the Kansas Geological Survey dataset.

**Instructions**:
1. Load 3-5 different LAS files from the `data/` directory
2. For each file, extract and analyze:
   - Available curves and their units
   - Depth range and sampling interval
   - Data completeness (percentage of valid measurements)
   - Well metadata (location, formation, date)

**Code Template**:
```python
import sys
sys.path.append('../src')
from data.las_reader import LASFileReader
from pathlib import Path

# Your code here
las_reader = LASFileReader()
data_dir = Path('../data')
las_files = list(data_dir.glob('*.las'))

# Analyze first 5 files
for i, las_file in enumerate(las_files[:5]):
    print(f"\n=== File {i+1}: {las_file.name} ===")
    # TODO: Implement analysis
```

**Expected Output**:
- Summary table of curve availability across files
- Identification of most common curves
- Assessment of data quality issues

### Task 1.2: Data Quality Metrics (25 minutes)

**Goal**: Implement comprehensive data quality assessment metrics.

**Instructions**:
1. Create a function `assess_curve_quality(curve_data)` that computes:
   - Completeness percentage
   - Outlier detection (3-sigma rule)
   - Gap analysis (consecutive missing values)
   - Statistical summary (mean, std, min, max)

2. Apply this function to all curves in your selected files
3. Create a quality scoring system (0-100) based on:
   - Completeness (40% weight)
   - Outlier percentage (30% weight)
   - Gap frequency (30% weight)

**Code Template**:
```python
import numpy as np

def assess_curve_quality(curve_data, curve_name):
    """
    Assess the quality of a single curve
    
    Args:
        curve_data: numpy array of curve measurements
        curve_name: name of the curve
    
    Returns:
        dict: Quality metrics
    """
    # TODO: Implement quality assessment
    pass

def compute_quality_score(quality_metrics):
    """Compute overall quality score (0-100)"""
    # TODO: Implement scoring
    pass

# Test your functions
```

**Expected Output**:
- Quality report for each curve
- Overall quality scores
- Recommendations for data preprocessing

### Task 1.3: Data Preprocessing Pipeline (30 minutes)

**Goal**: Implement a complete preprocessing pipeline for well log data.

**Instructions**:
1. Implement outlier removal using the IQR method
2. Handle missing values using:
   - Linear interpolation for small gaps (<5 points)
   - Forward/backward fill for edge cases
   - Median imputation for larger gaps
3. Normalize curves to zero mean and unit variance
4. Create a validation function to check preprocessing results

**Code Template**:
```python
def remove_outliers(data, method='iqr', factor=1.5):
    """Remove outliers using IQR or z-score method"""
    # TODO: Implement outlier removal
    pass

def handle_missing_values(data, max_gap=5):
    """Handle missing values with different strategies"""
    # TODO: Implement missing value handling
    pass

def normalize_curve(data):
    """Normalize curve to zero mean and unit variance"""
    # TODO: Implement normalization
    pass

def validate_preprocessing(original_data, processed_data):
    """Validate preprocessing results"""
    # TODO: Implement validation
    pass
```

**Expected Output**:
- Before/after comparison plots
- Preprocessing statistics
- Validation report

### Task 1.4: Multi-Well Dataset Creation (25 minutes)

**Goal**: Combine multiple wells into a coherent training dataset.

**Instructions**:
1. Filter wells to include only those with required curves (GR, PHIE, PERM)
2. Apply consistent preprocessing to all wells
3. Create train/validation/test splits at the well level (not sample level)
4. Ensure balanced representation across different geological conditions

**Code Template**:
```python
def filter_wells_by_curves(wells, required_curves):
    """Filter wells that have all required curves"""
    # TODO: Implement filtering
    pass

def create_balanced_splits(wells, train_ratio=0.7, val_ratio=0.15):
    """Create balanced train/val/test splits"""
    # TODO: Implement splitting
    pass

def combine_well_data(wells, curves_to_include):
    """Combine multiple wells into single dataset"""
    # TODO: Implement combination
    pass
```

**Expected Output**:
- Dataset statistics (number of wells, samples per split)
- Feature distribution plots
- Quality assessment of final dataset

## Solutions

### Solution 1.1: LAS File Analysis

```python
import sys
sys.path.append('../src')
from data.las_reader import LASFileReader
from pathlib import Path
import pandas as pd

def analyze_las_files(las_files, max_files=5):
    """Analyze multiple LAS files and create summary"""
    
    las_reader = LASFileReader()
    analysis_results = []
    
    for i, las_file in enumerate(las_files[:max_files]):
        try:
            print(f"\n=== Analyzing {las_file.name} ===")
            
            # Read LAS file
            well_data = las_reader.read_las_file(str(las_file))
            
            # Extract basic information
            result = {
                'file_name': las_file.name,
                'well_id': well_data.well_id,
                'depth_min': well_data.depth.min(),
                'depth_max': well_data.depth.max(),
                'depth_range': well_data.depth.max() - well_data.depth.min(),
                'n_samples': len(well_data.depth),
                'curves': list(well_data.curves.keys()),
                'n_curves': len(well_data.curves)
            }
            
            # Analyze each curve
            curve_stats = {}
            for curve_name, curve_data in well_data.curves.items():
                valid_mask = ~np.isnan(curve_data)
                completeness = np.sum(valid_mask) / len(curve_data) * 100
                
                curve_stats[curve_name] = {
                    'completeness': completeness,
                    'mean': np.nanmean(curve_data),
                    'std': np.nanstd(curve_data),
                    'min': np.nanmin(curve_data),
                    'max': np.nanmax(curve_data)
                }
            
            result['curve_stats'] = curve_stats
            analysis_results.append(result)
            
            print(f"  Well ID: {result['well_id']}")
            print(f"  Depth range: {result['depth_min']:.1f} - {result['depth_max']:.1f} ft")
            print(f"  Samples: {result['n_samples']}")
            print(f"  Curves: {result['curves']}")
            
        except Exception as e:
            print(f"  Error processing {las_file.name}: {e}")
    
    return analysis_results

# Run analysis
data_dir = Path('../data')
las_files = list(data_dir.glob('*.las'))
results = analyze_las_files(las_files)

# Create summary table
curve_availability = {}
for result in results:
    for curve in result['curves']:
        if curve not in curve_availability:
            curve_availability[curve] = 0
        curve_availability[curve] += 1

print(f"\n=== Curve Availability Summary ===")
for curve, count in sorted(curve_availability.items(), key=lambda x: x[1], reverse=True):
    print(f"{curve:12s}: {count}/{len(results)} files ({count/len(results)*100:.1f}%)")
```

### Solution 1.2: Data Quality Metrics

```python
import numpy as np
from scipy import stats

def assess_curve_quality(curve_data, curve_name):
    """Comprehensive curve quality assessment"""
    
    # Basic statistics
    total_points = len(curve_data)
    valid_mask = ~np.isnan(curve_data)
    valid_points = np.sum(valid_mask)
    completeness = valid_points / total_points * 100
    
    if valid_points == 0:
        return {
            'curve_name': curve_name,
            'completeness': 0,
            'quality_score': 0,
            'issues': ['No valid data']
        }
    
    valid_data = curve_data[valid_mask]
    
    # Outlier detection (3-sigma rule)
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)
    outlier_mask = np.abs(valid_data - mean_val) > 3 * std_val
    outlier_percentage = np.sum(outlier_mask) / len(valid_data) * 100
    
    # Gap analysis
    valid_indices = np.where(valid_mask)[0]
    gaps = np.diff(valid_indices) - 1
    gap_count = np.sum(gaps > 0)
    max_gap = np.max(gaps) if len(gaps) > 0 else 0
    
    # Statistical summary
    stats_summary = {
        'mean': mean_val,
        'std': std_val,
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'median': np.median(valid_data),
        'skewness': stats.skew(valid_data),
        'kurtosis': stats.kurtosis(valid_data)
    }
    
    # Quality issues
    issues = []
    if completeness < 80:
        issues.append(f'Low completeness ({completeness:.1f}%)')
    if outlier_percentage > 5:
        issues.append(f'High outlier rate ({outlier_percentage:.1f}%)')
    if max_gap > 10:
        issues.append(f'Large gaps (max: {max_gap} points)')
    if abs(stats_summary['skewness']) > 2:
        issues.append('Highly skewed distribution')
    
    return {
        'curve_name': curve_name,
        'completeness': completeness,
        'outlier_percentage': outlier_percentage,
        'gap_count': gap_count,
        'max_gap': max_gap,
        'statistics': stats_summary,
        'issues': issues
    }

def compute_quality_score(quality_metrics):
    """Compute overall quality score (0-100)"""
    
    # Weights for different factors
    completeness_weight = 0.4
    outlier_weight = 0.3
    gap_weight = 0.3
    
    # Completeness score (0-100)
    completeness_score = quality_metrics['completeness']
    
    # Outlier score (100 - outlier_percentage, capped at 0)
    outlier_score = max(0, 100 - quality_metrics['outlier_percentage'] * 2)
    
    # Gap score (penalize based on gap frequency and size)
    gap_penalty = quality_metrics['gap_count'] + quality_metrics['max_gap'] / 10
    gap_score = max(0, 100 - gap_penalty)
    
    # Weighted average
    quality_score = (completeness_weight * completeness_score +
                    outlier_weight * outlier_score +
                    gap_weight * gap_score)
    
    return min(100, max(0, quality_score))

# Example usage
def analyze_well_quality(well_data):
    """Analyze quality of all curves in a well"""
    
    quality_report = {}
    overall_scores = []
    
    for curve_name, curve_data in well_data.curves.items():
        quality_metrics = assess_curve_quality(curve_data, curve_name)
        quality_score = compute_quality_score(quality_metrics)
        quality_metrics['quality_score'] = quality_score
        
        quality_report[curve_name] = quality_metrics
        overall_scores.append(quality_score)
    
    # Overall well quality
    well_quality = np.mean(overall_scores) if overall_scores else 0
    
    return quality_report, well_quality
```

### Solution 1.3: Data Preprocessing Pipeline

```python
import numpy as np
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

def remove_outliers(data, method='iqr', factor=1.5):
    """Remove outliers using IQR or z-score method"""
    
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) == 0:
        return data
    
    valid_data = data[valid_mask]
    
    if method == 'iqr':
        Q1 = np.percentile(valid_data, 25)
        Q3 = np.percentile(valid_data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        z_scores = np.abs((data - mean_val) / std_val)
        outlier_mask = z_scores > factor
    
    # Replace outliers with NaN
    cleaned_data = data.copy()
    cleaned_data[outlier_mask] = np.nan
    
    return cleaned_data

def handle_missing_values(data, max_gap=5):
    """Handle missing values with different strategies"""
    
    data_filled = data.copy()
    
    # Find NaN positions
    nan_mask = np.isnan(data_filled)
    
    if not np.any(nan_mask):
        return data_filled
    
    # Get valid data positions
    valid_positions = np.where(~nan_mask)[0]
    
    if len(valid_positions) == 0:
        return data_filled
    
    # Linear interpolation for small gaps
    for i in range(len(data_filled)):
        if nan_mask[i]:
            # Find gap size
            gap_start = i
            gap_end = i
            while gap_end < len(data_filled) and nan_mask[gap_end]:
                gap_end += 1
            
            gap_size = gap_end - gap_start
            
            if gap_size <= max_gap:
                # Find surrounding valid points
                left_idx = gap_start - 1
                right_idx = gap_end
                
                if left_idx >= 0 and right_idx < len(data_filled):
                    if not nan_mask[left_idx] and not nan_mask[right_idx]:
                        # Linear interpolation
                        left_val = data_filled[left_idx]
                        right_val = data_filled[right_idx]
                        
                        for j in range(gap_start, gap_end):
                            alpha = (j - left_idx) / (right_idx - left_idx)
                            data_filled[j] = left_val + alpha * (right_val - left_val)
    
    # Forward/backward fill for remaining gaps
    # Forward fill
    last_valid = None
    for i in range(len(data_filled)):
        if not np.isnan(data_filled[i]):
            last_valid = data_filled[i]
        elif last_valid is not None:
            data_filled[i] = last_valid
    
    # Backward fill
    last_valid = None
    for i in range(len(data_filled) - 1, -1, -1):
        if not np.isnan(data_filled[i]):
            last_valid = data_filled[i]
        elif last_valid is not None:
            data_filled[i] = last_valid
    
    return data_filled

def normalize_curve(data):
    """Normalize curve to zero mean and unit variance"""
    
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) == 0:
        return data
    
    valid_data = data[valid_mask]
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)
    
    if std_val == 0:
        return data - mean_val
    
    normalized_data = (data - mean_val) / std_val
    return normalized_data

def validate_preprocessing(original_data, processed_data, curve_name):
    """Validate preprocessing results"""
    
    validation_report = {
        'curve_name': curve_name,
        'original_stats': {},
        'processed_stats': {},
        'changes': {}
    }
    
    # Original data statistics
    orig_valid = original_data[~np.isnan(original_data)]
    if len(orig_valid) > 0:
        validation_report['original_stats'] = {
            'count': len(orig_valid),
            'mean': np.mean(orig_valid),
            'std': np.std(orig_valid),
            'min': np.min(orig_valid),
            'max': np.max(orig_valid),
            'completeness': len(orig_valid) / len(original_data) * 100
        }
    
    # Processed data statistics
    proc_valid = processed_data[~np.isnan(processed_data)]
    if len(proc_valid) > 0:
        validation_report['processed_stats'] = {
            'count': len(proc_valid),
            'mean': np.mean(proc_valid),
            'std': np.std(proc_valid),
            'min': np.min(proc_valid),
            'max': np.max(proc_valid),
            'completeness': len(proc_valid) / len(processed_data) * 100
        }
    
    # Changes
    if len(orig_valid) > 0 and len(proc_valid) > 0:
        validation_report['changes'] = {
            'completeness_change': validation_report['processed_stats']['completeness'] - 
                                 validation_report['original_stats']['completeness'],
            'outliers_removed': len(orig_valid) - len(proc_valid),
            'mean_shift': abs(validation_report['processed_stats']['mean'] - 
                            validation_report['original_stats']['mean']),
            'std_change': validation_report['processed_stats']['std'] / 
                         validation_report['original_stats']['std']
        }
    
    return validation_report

# Complete preprocessing pipeline
def preprocess_well_data(well_data, outlier_method='iqr', max_gap=5):
    """Complete preprocessing pipeline for well data"""
    
    processed_curves = {}
    validation_reports = {}
    
    for curve_name, curve_data in well_data.curves.items():
        print(f"Processing {curve_name}...")
        
        # Step 1: Remove outliers
        cleaned_data = remove_outliers(curve_data, method=outlier_method)
        
        # Step 2: Handle missing values
        filled_data = handle_missing_values(cleaned_data, max_gap=max_gap)
        
        # Step 3: Normalize
        normalized_data = normalize_curve(filled_data)
        
        processed_curves[curve_name] = normalized_data
        
        # Validate
        validation_reports[curve_name] = validate_preprocessing(
            curve_data, normalized_data, curve_name
        )
    
    return processed_curves, validation_reports
```

### Solution 1.4: Multi-Well Dataset Creation

```python
import numpy as np
from sklearn.model_selection import train_test_split

def filter_wells_by_curves(wells, required_curves):
    """Filter wells that have all required curves with sufficient quality"""
    
    filtered_wells = []
    
    for well in wells:
        # Check if all required curves are present
        available_curves = set(well['curves'].keys())
        has_required = all(curve in available_curves for curve in required_curves)
        
        if not has_required:
            continue
        
        # Check data quality for required curves
        quality_ok = True
        for curve in required_curves:
            curve_data = well['curves'][curve]
            completeness = np.sum(~np.isnan(curve_data)) / len(curve_data) * 100
            
            if completeness < 70:  # Require at least 70% completeness
                quality_ok = False
                break
        
        if quality_ok:
            filtered_wells.append(well)
    
    return filtered_wells

def create_balanced_splits(wells, train_ratio=0.7, val_ratio=0.15):
    """Create balanced train/val/test splits at well level"""
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # First split: train vs (val + test)
    train_wells, temp_wells = train_test_split(
        wells, 
        test_size=(val_ratio + test_ratio),
        random_state=42
    )
    
    # Second split: val vs test
    val_wells, test_wells = train_test_split(
        temp_wells,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42
    )
    
    return {
        'train': train_wells,
        'validation': val_wells,
        'test': test_wells
    }

def combine_well_data(wells, curves_to_include):
    """Combine multiple wells into single dataset"""
    
    all_features = []
    all_targets = []
    all_well_ids = []
    
    for well in wells:
        # Extract features
        depth = well['depth']
        features = [depth]
        
        for curve in curves_to_include['features']:
            if curve in well['curves']:
                features.append(well['curves'][curve])
        
        # Stack features
        well_features = np.column_stack(features)
        
        # Extract targets
        targets = []
        for target in curves_to_include['targets']:
            if target in well['curves']:
                targets.append(well['curves'][target])
        
        if len(targets) > 0:
            well_targets = np.column_stack(targets)
            
            # Remove rows with any NaN values
            valid_mask = ~np.any(np.isnan(well_features), axis=1) & ~np.any(np.isnan(well_targets), axis=1)
            
            if np.sum(valid_mask) > 0:
                all_features.append(well_features[valid_mask])
                all_targets.append(well_targets[valid_mask])
                all_well_ids.extend([well['well_id']] * np.sum(valid_mask))
    
    # Combine all wells
    if len(all_features) > 0:
        combined_features = np.vstack(all_features)
        combined_targets = np.vstack(all_targets)
        combined_well_ids = np.array(all_well_ids)
        
        return {
            'X': combined_features,
            'y': combined_targets,
            'well_ids': combined_well_ids,
            'feature_names': ['depth'] + curves_to_include['features'],
            'target_names': curves_to_include['targets']
        }
    
    return None

# Example usage
def create_pinn_dataset(processed_wells):
    """Create complete PINN dataset from processed wells"""
    
    # Define required curves
    required_curves = ['GR', 'PHIE', 'PERM']
    curves_config = {
        'features': ['GR', 'PHIE', 'PERM'],
        'targets': ['PRESSURE', 'SATURATION']  # These would be computed/synthetic
    }
    
    # Filter wells
    filtered_wells = filter_wells_by_curves(processed_wells, required_curves)
    print(f"Filtered wells: {len(filtered_wells)}/{len(processed_wells)}")
    
    # Create splits
    well_splits = create_balanced_splits(filtered_wells)
    
    # Combine data for each split
    datasets = {}
    for split_name, wells in well_splits.items():
        dataset = combine_well_data(wells, curves_config)
        if dataset:
            datasets[split_name] = dataset
            print(f"{split_name}: {len(wells)} wells, {dataset['X'].shape[0]} samples")
    
    return datasets

# Test the complete pipeline
# datasets = create_pinn_dataset(processed_wells)
```

## Evaluation Criteria

### Task 1.1 (25 points)
- **Correct LAS file reading** (10 points)
- **Comprehensive analysis** (10 points)
- **Clear summary presentation** (5 points)

### Task 1.2 (25 points)
- **Proper quality metrics implementation** (15 points)
- **Reasonable scoring system** (5 points)
- **Clear quality assessment** (5 points)

### Task 1.3 (25 points)
- **Correct outlier removal** (8 points)
- **Appropriate missing value handling** (8 points)
- **Proper normalization** (4 points)
- **Validation implementation** (5 points)

### Task 1.4 (25 points)
- **Correct well filtering** (8 points)
- **Proper train/val/test splitting** (8 points)
- **Successful data combination** (6 points)
- **Dataset quality assessment** (3 points)

## Common Mistakes to Avoid

1. **Data Leakage**: Don't split at the sample level; always split at the well level
2. **Ignoring Physics**: Ensure preprocessing doesn't violate physical constraints
3. **Over-cleaning**: Don't remove too much data in the name of quality
4. **Poor Normalization**: Always normalize after handling missing values
5. **Inadequate Validation**: Always validate preprocessing results

## Extensions

1. **Advanced Outlier Detection**: Implement isolation forest or local outlier factor
2. **Physics-Informed Interpolation**: Use domain knowledge for missing value imputation
3. **Adaptive Quality Thresholds**: Adjust quality criteria based on curve type
4. **Cross-Well Validation**: Validate preprocessing consistency across wells
5. **Automated Pipeline**: Create automated preprocessing with minimal user input

## Resources

- [LAS File Format Specification](https://www.cwls.org/products/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas Data Cleaning](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Well Log Analysis Fundamentals](https://www.slb.com/technical-challenges/formation-evaluation-petrophysics)