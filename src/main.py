"""
Main entry point for the PINN tutorial system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import get_config, load_config_from_file, LOGS_DIR


def setup_logging(config: dict) -> None:
    """Set up logging configuration."""
    log_config = config.get("logging", {})
    
    # Create formatter
    formatter = logging.Formatter(
        log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_config.get("level", "INFO")))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_config.get("file_logging", True):
        log_file = LOGS_DIR / "pinn_tutorial.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="PINN Tutorial: Physics-Informed Neural Networks for Multiphase Flow Modeling"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON or YAML)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing LAS files"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="output",
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "validate", "demo", "tutorial"],
        default="demo",
        help="Execution mode"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def run_demo(config: dict, data_dir: str, output_dir: str) -> None:
    """Run a demonstration of the PINN tutorial system."""
    logger = logging.getLogger(__name__)
    logger.info("Running PINN tutorial demonstration...")
    
    # This will be implemented in later tasks
    logger.info("Demo mode - basic system validation")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("System components initialized successfully!")


def run_training(config: dict, data_dir: str, output_dir: str) -> None:
    """Run PINN training pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting PINN training pipeline...")
    
    # This will be implemented in later tasks
    logger.info("Training mode - full PINN training pipeline")
    logger.info("Training pipeline will be implemented in subsequent tasks")


def run_validation(config: dict, data_dir: str, output_dir: str) -> None:
    """Run model validation and benchmarking."""
    logger = logging.getLogger(__name__)
    logger.info("Running model validation...")
    
    # This will be implemented in later tasks
    logger.info("Validation mode - model performance assessment")
    logger.info("Validation pipeline will be implemented in subsequent tasks")


def run_tutorial(config: dict, data_dir: str, output_dir: str) -> None:
    """Generate tutorial content and notebooks."""
    logger = logging.getLogger(__name__)
    logger.info("Generating tutorial content...")
    
    # This will be implemented in later tasks
    logger.info("Tutorial mode - educational content generation")
    logger.info("Tutorial generation will be implemented in subsequent tasks")


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_config()
    
    # Update logging level if verbose
    if args.verbose:
        config["logging"]["level"] = "DEBUG"
    
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting PINN Tutorial System")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Route to appropriate function based on mode
        if args.mode == "demo":
            run_demo(config, args.data_dir, args.output_dir)
        elif args.mode == "train":
            run_training(config, args.data_dir, args.output_dir)
        elif args.mode == "validate":
            run_validation(config, args.data_dir, args.output_dir)
        elif args.mode == "tutorial":
            run_tutorial(config, args.data_dir, args.output_dir)
        
        logger.info("PINN Tutorial System completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()