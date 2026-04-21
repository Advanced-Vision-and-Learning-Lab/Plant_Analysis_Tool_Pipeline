"""
Sorghum Plant Phenotyping Pipeline

A comprehensive pipeline for analyzing sorghum plant images including:
- Data loading and preprocessing
- Image segmentation and masking
- Feature extraction (texture, morphology, vegetation indices)
- Results visualization and export

Author: Fahime Horvatinia
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Fahime Horvatinia"

from .pipeline import SorghumPipeline
from .config import Config
from .data import DataLoader
from .features import TextureExtractor, VegetationIndexExtractor, MorphologyExtractor
from .output import OutputManager

__all__ = [
    "SorghumPipeline",
    "Config", 
    "DataLoader",
    "TextureExtractor",
    "VegetationIndexExtractor",
    "MorphologyExtractor",
    "OutputManager"
]
