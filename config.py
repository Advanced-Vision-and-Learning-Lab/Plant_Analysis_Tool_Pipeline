"""
Configuration management for the Sorghum Pipeline.

This module handles all configuration settings, paths, and parameters
used throughout the pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Paths:
    """Configuration for all file paths."""
    input_folder: str
    output_folder: str
    boundingbox_dir: Optional[str] = None
    labels_folder: Optional[str] = None
    
    def __post_init__(self):
        """Ensure all paths are absolute where provided."""
        self.input_folder = os.path.abspath(self.input_folder)
        self.output_folder = os.path.abspath(self.output_folder)
        if self.boundingbox_dir:
            self.boundingbox_dir = os.path.abspath(self.boundingbox_dir)
        if self.labels_folder:
            self.labels_folder = os.path.abspath(self.labels_folder)


@dataclass
class ProcessingParams:
    """Parameters for image processing."""
    # Image processing
    target_size: tuple = (1024, 1024)
    gaussian_blur_kernel: int = 5
    morphology_kernel_size: int = 7
    min_component_area: int = 1000
    
    # Segmentation
    segmentation_threshold: float = 0.5
    max_components: int = 10
    
    # Texture analysis
    lbp_points: int = 8
    lbp_radius: int = 1
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple = (8, 8)
    hog_cells_per_block: tuple = (2, 2)
    lacunarity_window: int = 15
    ehd_threshold: float = 0.3
    angle_resolution: int = 45
    
    # Vegetation indices
    epsilon: float = 1e-10
    soil_factor: float = 0.16
    
    # Morphology
    pixel_to_cm: float = 0.1099609375
    prune_sizes: list = field(default_factory=lambda: [200, 100, 50, 30, 10])


@dataclass
class OutputSettings:
    """Settings for output generation."""
    save_images: bool = True
    save_plots: bool = True
    save_metadata: bool = True
    image_dpi: int = 150
    plot_dpi: int = 100
    image_format: str = "png"
    
    # Subdirectories
    segmentation_dir: str = "segmentation"
    features_dir: str = "features"
    texture_dir: str = "texture"
    morphology_dir: str = "morphology"
    vegetation_dir: str = "vegetation_indices"
    analysis_dir: str = "analysis"


@dataclass
class ModelSettings:
    """Settings for ML models."""
    device: str = "auto"  # auto, cpu, cuda
    model_name: str = "briaai/RMBG-2.0"
    batch_size: int = 1
    trust_remote_code: bool = True
    cache_dir: str = ""
    local_files_only: bool = False


class Config:
    """Main configuration class for the Sorghum Pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self.paths = Paths(
            input_folder="",
            output_folder="",
            boundingbox_dir=""
        )
        self.processing = ProcessingParams()
        self.output = OutputSettings()
        self.model = ModelSettings()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update paths
        if 'paths' in config_data:
            self.paths = Paths(**config_data['paths'])
        
        # Update processing parameters
        if 'processing' in config_data:
            for key, value in config_data['processing'].items():
                if hasattr(self.processing, key):
                    setattr(self.processing, key, value)
        
        # Update output settings
        if 'output' in config_data:
            for key, value in config_data['output'].items():
                if hasattr(self.output, key):
                    setattr(self.output, key, value)
        
        # Update model settings
        if 'model' in config_data:
            for key, value in config_data['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to YAML file."""
        config_data = {
            'paths': {
                'input_folder': self.paths.input_folder,
                'output_folder': self.paths.output_folder,
                'boundingbox_dir': self.paths.boundingbox_dir,
                'labels_folder': self.paths.labels_folder
            },
            'processing': {
                'target_size': self.processing.target_size,
                'gaussian_blur_kernel': self.processing.gaussian_blur_kernel,
                'morphology_kernel_size': self.processing.morphology_kernel_size,
                'min_component_area': self.processing.min_component_area,
                'segmentation_threshold': self.processing.segmentation_threshold,
                'max_components': self.processing.max_components,
                'lbp_points': self.processing.lbp_points,
                'lbp_radius': self.processing.lbp_radius,
                'hog_orientations': self.processing.hog_orientations,
                'hog_pixels_per_cell': self.processing.hog_pixels_per_cell,
                'hog_cells_per_block': self.processing.hog_cells_per_block,
                'lacunarity_window': self.processing.lacunarity_window,
                'ehd_threshold': self.processing.ehd_threshold,
                'angle_resolution': self.processing.angle_resolution,
                'epsilon': self.processing.epsilon,
                'soil_factor': self.processing.soil_factor,
                'pixel_to_cm': self.processing.pixel_to_cm,
                'prune_sizes': self.processing.prune_sizes
            },
            'output': {
                'save_images': self.output.save_images,
                'save_plots': self.output.save_plots,
                'save_metadata': self.output.save_metadata,
                'image_dpi': self.output.image_dpi,
                'plot_dpi': self.output.plot_dpi,
                'image_format': self.output.image_format,
                'segmentation_dir': self.output.segmentation_dir,
                'features_dir': self.output.features_dir,
                'texture_dir': self.output.texture_dir,
                'morphology_dir': self.output.morphology_dir,
                'vegetation_dir': self.output.vegetation_dir,
                'analysis_dir': self.output.analysis_dir
            },
            'model': {
                'device': self.model.device,
                'model_name': self.model.model_name,
                'batch_size': self.model.batch_size,
                'trust_remote_code': self.model.trust_remote_code,
                'cache_dir': self.model.cache_dir,
                'local_files_only': self.model.local_files_only,
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def get_device(self) -> str:
        """Get the appropriate device for processing."""
        if self.model.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.model.device
    
    def create_output_directories(self, base_path: str) -> None:
        """Ensure base output directory exists only.

        Subdirectories are created per plant in the output manager.
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        # Check if input directory exists
        if not os.path.exists(self.paths.input_folder):
            raise FileNotFoundError(f"Input folder does not exist: {self.paths.input_folder}")
        
        # Check if bounding box directory exists (optional)
        if hasattr(self.paths, 'boundingbox_dir') and self.paths.boundingbox_dir and not os.path.exists(self.paths.boundingbox_dir):
            raise FileNotFoundError(f"Bounding box directory does not exist: {self.paths.boundingbox_dir}")
        
        # Validate processing parameters
        if self.processing.target_size[0] <= 0 or self.processing.target_size[1] <= 0:
            raise ValueError("Target size must be positive")
        
        if self.processing.segmentation_threshold < 0 or self.processing.segmentation_threshold > 1:
            raise ValueError("Segmentation threshold must be between 0 and 1")
        
        return True


def create_default_config(output_path: str) -> None:
    """Create a default configuration file."""
    config = Config()
    config.paths = Paths(
        input_folder="Sorghum_dataset",
        output_folder="Sorghum_pipeline_Results",
        boundingbox_dir="boundingbox",
        labels_folder="labels"
    )
    config.save_to_file(output_path)
    print(f"Default configuration created at: {output_path}")
