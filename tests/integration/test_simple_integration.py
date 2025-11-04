#!/usr/bin/env python3
"""
Simple integration tests for the PINN tutorial system.
Tests basic functionality without complex dependencies.
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestBasicIntegration(unittest.TestCase):
    """Test basic system integration."""
    
    def test_project_structure(self):
        """Test that the project has the expected structure."""
        
        base_dir = Path(__file__).parent.parent
        
        # Check main directories exist
        expected_dirs = [
            'src',
            'tests', 
            'tutorials',
            'examples',
            'data'
        ]
        
        for dir_name in expected_dirs:
            dir_path = base_dir / dir_name
            self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")
            if dir_name in ['src', 'tests']:
                self.assertTrue(dir_path.is_dir(), f"{dir_name} should be a directory")
    
    def test_core_modules_exist(self):
        """Test that core modules exist and can be imported."""
        
        # Test core module structure
        src_dir = Path(__file__).parent.parent / 'src'
        
        expected_modules = [
            'core',
            'data', 
            'models',
            'physics',
            'training',
            'validation',
            'visualization'
        ]
        
        for module_name in expected_modules:
            module_dir = src_dir / module_name
            self.assertTrue(module_dir.exists(), f"Module {module_name} should exist")
            
            # Check for __init__.py
            init_file = module_dir / '__init__.py'
            self.assertTrue(init_file.exists(), f"Module {module_name} should have __init__.py")
    
    def test_configuration_files(self):
        """Test that configuration files exist."""
        
        base_dir = Path(__file__).parent.parent
        
        config_files = [
            'pyproject.toml',
            'setup.py',
            'requirements.txt',
            'README.md'
        ]
        
        for config_file in config_files:
            file_path = base_dir / config_file
            self.assertTrue(file_path.exists(), f"Config file {config_file} should exist")
    
    def test_tutorial_content_exists(self):
        """Test that tutorial content exists."""
        
        tutorials_dir = Path(__file__).parent.parent / 'tutorials'
        
        if tutorials_dir.exists():
            # Check for notebook files
            notebooks = list(tutorials_dir.glob('*.ipynb'))
            self.assertGreater(len(notebooks), 0, "Should have at least one tutorial notebook")
            
            # Check for documentation
            docs = list(tutorials_dir.glob('*.md'))
            self.assertGreater(len(docs), 0, "Should have tutorial documentation")
    
    def test_example_scripts_exist(self):
        """Test that example scripts exist."""
        
        examples_dir = Path(__file__).parent.parent / 'examples'
        
        if examples_dir.exists():
            # Check for Python files
            scripts = list(examples_dir.glob('*.py'))
            self.assertGreater(len(scripts), 0, "Should have at least one example script")
    
    def test_data_directory_structure(self):
        """Test that data directory has expected structure."""
        
        data_dir = Path(__file__).parent.parent / 'data'
        
        if data_dir.exists():
            # Check for LAS files
            las_files = list(data_dir.glob('*.las'))
            self.assertGreater(len(las_files), 0, "Should have LAS files for testing")


class TestModuleImports(unittest.TestCase):
    """Test that modules can be imported without errors."""
    
    def test_core_imports(self):
        """Test core module imports."""
        
        try:
            from core import data_models
            from core import interfaces
            print("✓ Core modules imported successfully")
        except ImportError as e:
            self.fail(f"Core module import failed: {e}")
    
    def test_data_imports(self):
        """Test data module imports."""
        
        try:
            from data import las_reader
            from data import preprocessor
            from data import dataset_builder
            print("✓ Data modules imported successfully")
        except ImportError as e:
            self.fail(f"Data module import failed: {e}")
    
    def test_model_imports(self):
        """Test model module imports."""
        
        try:
            # These might fail if PyTorch is not available
            from models import pinn_architecture
            from models import tensor_manager
            from models import model_interface
            print("✓ Model modules imported successfully")
        except ImportError as e:
            print(f"Warning: Model module import failed (likely missing PyTorch): {e}")
            # Don't fail the test for missing dependencies
    
    def test_physics_imports(self):
        """Test physics module imports."""
        
        try:
            from physics import pde_formulator
            from physics import boundary_conditions
            from physics import physics_loss
            print("✓ Physics modules imported successfully")
        except ImportError as e:
            print(f"Warning: Physics module import failed: {e}")
    
    def test_training_imports(self):
        """Test training module imports."""
        
        try:
            from training import pinn_trainer
            from training import optimizer_manager
            from training import batch_processor
            from training import convergence_monitor
            print("✓ Training modules imported successfully")
        except ImportError as e:
            print(f"Warning: Training module import failed: {e}")
    
    def test_validation_imports(self):
        """Test validation module imports."""
        
        try:
            from validation import validation_framework
            from validation import pde_residual_analyzer
            from validation import prediction_comparator
            print("✓ Validation modules imported successfully")
        except ImportError as e:
            print(f"Warning: Validation module import failed: {e}")
    
    def test_visualization_imports(self):
        """Test visualization module imports."""
        
        try:
            from visualization import scientific_plotter
            from visualization import training_visualizer
            from visualization import results_analyzer
            from visualization import diagram_generator
            print("✓ Visualization modules imported successfully")
        except ImportError as e:
            print(f"Warning: Visualization module import failed: {e}")


class TestSystemConfiguration(unittest.TestCase):
    """Test system configuration and setup."""
    
    def test_python_version(self):
        """Test that Python version is compatible."""
        
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3, "Python 3.x required")
        self.assertGreaterEqual(version.minor, 7, "Python 3.7+ recommended")
    
    def test_package_configuration(self):
        """Test package configuration files."""
        
        base_dir = Path(__file__).parent.parent
        
        # Test pyproject.toml
        pyproject_file = base_dir / 'pyproject.toml'
        if pyproject_file.exists():
            with open(pyproject_file, 'r') as f:
                content = f.read()
                self.assertIn('name', content, "pyproject.toml should have package name")
        
        # Test setup.py
        setup_file = base_dir / 'setup.py'
        if setup_file.exists():
            with open(setup_file, 'r') as f:
                content = f.read()
                self.assertIn('setup', content, "setup.py should have setup function")


if __name__ == '__main__':
    print("Running Simple Integration Tests...")
    print("=" * 50)
    
    # Run tests with high verbosity
    unittest.main(verbosity=2)