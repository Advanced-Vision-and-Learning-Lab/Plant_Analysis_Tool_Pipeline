"""
Vegetation index extraction for the Sorghum Pipeline.

This module handles extraction of various vegetation indices
from multispectral data.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class VegetationIndexExtractor:
    """Extracts vegetation indices from spectral data."""
    
    def __init__(self, epsilon: float = 1e-10, soil_factor: float = 0.16):
        """
        Initialize vegetation index extractor.
        
        Args:
            epsilon: Small value to avoid division by zero
            soil_factor: Soil factor for certain indices
        """
        # Coerce to float in case config passed strings like "1e-10"
        try:
            self.epsilon = float(epsilon)
        except Exception:
            self.epsilon = 1e-10
        try:
            self.soil_factor = float(soil_factor)
        except Exception:
            self.soil_factor = 0.16
        
        # Define vegetation index formulas
        self.index_formulas = {
            "NDVI": lambda nir, red: (nir - red) / (nir + red + self.epsilon),
            "GNDVI": lambda nir, green: (nir - green) / (nir + green + self.epsilon),
            "NDRE": lambda nir, red_edge: (nir - red_edge) / (nir + red_edge + self.epsilon),
            "GRNDVI": lambda nir, green, red: (nir - (green + red)) / (nir + (green + red) + self.epsilon),
            "TNDVI": lambda nir, red: np.sqrt(np.clip(((nir - red) / (nir + red + self.epsilon)) + 0.5, 0, None)),
            "MGRVI": lambda green, red: (green**2 - red**2) / (green**2 + red**2 + self.epsilon),
            "GRVI": lambda nir, green: nir / (green + self.epsilon),
            "NGRDI": lambda green, red: (green - red) / (green + red + self.epsilon),
            "MSAVI": lambda nir, red: 0.5 * (2.0 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))),
            "OSAVI": lambda nir, red: (nir - red) / (nir + red + self.soil_factor + self.epsilon),
            "TSAVI": lambda nir, red, s=0.33, a=0.5, X=1.5: (s * (nir - s * red - a)) / (a * nir + red - a * s + X * (1 + s**2) + self.epsilon),
            "GSAVI": lambda nir, green, l=0.5: (1 + l) * (nir - green) / (nir + green + l + self.epsilon),
            # Requested additions and aliases
            "GOSAVI": lambda nir, green: (nir - green) / (nir + green + 0.16 + self.epsilon),
            "GDVI": lambda nir, green: nir - green,
            "NDWI": lambda green, nir: (green - nir) / (green + nir + self.epsilon),
            "DSWI4": lambda green, red: green / (red + self.epsilon),
            "CIRE": lambda nir, red_edge: (nir / (red_edge + self.epsilon)) - 1.0,
            "LCI": lambda nir, red_edge: (nir - red_edge) / (nir + red_edge + self.epsilon),
            "CIgreen": lambda nir, green: (nir / (green + self.epsilon)) - 1,
            "MCARI": lambda red_edge, red, green: ((red_edge - red) - 0.2 * (red_edge - green)) * (red_edge / (red + self.epsilon)),
            "MCARI1": lambda nir, red, green: 1.2 * (2.5 * (nir - red) - 1.3 * (nir - green)),
            "MCARI2": lambda nir, red, green: (1.5 * (2.5 * (nir - red) - 1.3 * (nir - green))) / np.sqrt((2 * nir + 1)**2 - (6 * nir - 5 * np.sqrt(red + self.epsilon))),
            # MTVI variants per request
            "MTVI1": lambda nir, red, green: 1.2 * (1.2 * (nir - green) - 2.5 * (red - green)),
            "MTVI2": lambda nir, red, green: (1.5 * (1.2 * (nir - green) - 2.5 * (red - green))) / np.sqrt((2 * nir + 1)**2 - (6 * nir - 5 * np.sqrt(red + self.epsilon)) - 0.5 + self.epsilon),
            "CVI": lambda nir, red, green: (nir * red) / (green**2 + self.epsilon),
            "ARI": lambda green, red_edge: (1.0 / (green + self.epsilon)) - (1.0 / (red_edge + self.epsilon)),
            "ARI2": lambda nir, green, red_edge: nir * (1.0 / (green + self.epsilon)) - nir * (1.0 / (red_edge + self.epsilon)),
            "DVI": lambda nir, red: nir - red,
            "WDVI": lambda nir, red, a=0.5: nir - a * red,
            "SR": lambda nir, red: nir / (red + self.epsilon),
            "MSR": lambda nir, red: (nir / (red + self.epsilon) - 1) / np.sqrt(nir / (red + self.epsilon) + 1),
            "PVI": lambda nir, red, a=0.5, b=0.3: (nir - a * red - b) / (np.sqrt(1 + a**2) + self.epsilon),
            "GEMI": lambda nir, red: ((2 * (nir**2 - red**2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5 + self.epsilon)) * (1 - 0.25 * ((2 * (nir**2 - red**2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5 + self.epsilon))) - ((red - 0.125) / (1 - red + self.epsilon)),
            "ExR": lambda red, green: 1.3 * red - green,
            "RI": lambda red, green: (red - green) / (red + green + self.epsilon),
            "RRI1": lambda nir, red_edge: nir / (red_edge + self.epsilon),
            "RRI2": lambda red_edge, red: red_edge / (red + self.epsilon),
            "RRI": lambda nir, red_edge: nir / (red_edge + self.epsilon),
            "AVI": lambda nir, red: np.cbrt(nir * (1.0 - red) * (nir - red + self.epsilon)),
            "SIPI2": lambda nir, green, red: (nir - green) / (nir - red + self.epsilon),
            "TCARI": lambda red_edge, red, green: 3 * ((red_edge - red) - 0.2 * (red_edge - green) * (red_edge / (red + self.epsilon))),
            "TCARIOSAVI": lambda red_edge, red, green, nir: (3 * (red_edge - red) - 0.2 * (red_edge - green) * (red_edge / (red + self.epsilon))) / (1 + 0.16 * ((nir - red) / (nir + red + 0.16 + self.epsilon))),
            "CCCI": lambda nir, red_edge, red: (((nir - red_edge) * (nir + red)) / ((nir + red_edge) * (nir - red) + self.epsilon)),
            # Additional indices
            "RDVI": lambda nir, red: (nir - red) / (np.sqrt(nir + red + self.epsilon)),
            "NLI": lambda nir, red: ((nir**2) - red) / ((nir**2) + red + self.epsilon),
            "BIXS": lambda green, red: np.sqrt(((green**2) + (red**2)) / 2.0),
            "IPVI": lambda nir, red: nir / (nir + red + self.epsilon),
            "EVI2": lambda nir, red: 2.4 * (nir - red) / (nir + red + 1.0 + self.epsilon)
        }
        
        # Define required bands for each index
        self.index_bands = {
            "NDVI": ["nir", "red"],
            "GNDVI": ["nir", "green"],
            "NDRE": ["nir", "red_edge"],
            "GRNDVI": ["nir", "green", "red"],
            "TNDVI": ["nir", "red"],
            "MGRVI": ["green", "red"],
            "GRVI": ["nir", "green"],
            "NGRDI": ["green", "red"],
            "MSAVI": ["nir", "red"],
            "OSAVI": ["nir", "red"],
            "TSAVI": ["nir", "red"],
            "GSAVI": ["nir", "green"],
            "GOSAVI": ["nir", "green"],
            "GDVI": ["nir", "green"],
            "NDWI": ["green", "nir"],
            "DSWI4": ["green", "red"],
            "CIRE": ["nir", "red_edge"],
            "LCI": ["nir", "red_edge"],
            "CIgreen": ["nir", "green"],
            "MCARI": ["red_edge", "red", "green"],
            "MCARI1": ["nir", "red", "green"],
            "MCARI2": ["nir", "red", "green"],
            "MTVI1": ["nir", "red", "green"],
            "MTVI2": ["nir", "red", "green"],
            "CVI": ["nir", "red", "green"],
            "ARI": ["green", "red_edge"],
            "ARI2": ["nir", "green", "red_edge"],
            "DVI": ["nir", "red"],
            "WDVI": ["nir", "red"],
            "SR": ["nir", "red"],
            "MSR": ["nir", "red"],
            "PVI": ["nir", "red"],
            "GEMI": ["nir", "red"],
            "ExR": ["red", "green"],
            "RI": ["red", "green"],
            "RRI1": ["nir", "red_edge"],
            "RRI2": ["red_edge", "red"],
            "RRI": ["nir", "red_edge"],
            "AVI": ["nir", "red"],
            "SIPI2": ["nir", "green", "red"],
            "TCARI": ["red_edge", "red", "green"],
            "TCARIOSAVI": ["red_edge", "red", "green", "nir"],
            "CCCI": ["nir", "red_edge", "red"],
            "RDVI": ["nir", "red"],
            "NLI": ["nir", "red"],
            "BIXS": ["green", "red"],
            "IPVI": ["nir", "red"],
            "EVI2": ["nir", "red"]
        }
    
    def compute_vegetation_indices(self, spectral_stack: Dict[str, np.ndarray], 
                                 mask: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Compute vegetation indices from spectral data.
        
        Args:
            spectral_stack: Dictionary of spectral bands
            mask: Binary mask for the plant
            
        Returns:
            Dictionary of vegetation indices with values and statistics
        """
        indices = {}
        
        for index_name, formula in self.index_formulas.items():
            try:
                # Get required bands
                required_bands = self.index_bands.get(index_name, [])
                
                # Check if all required bands are available
                if not all(band in spectral_stack for band in required_bands):
                    logger.warning(f"Skipping {index_name}: missing required bands")
                    continue
                
                # Extract band data as float arrays
                band_data = []
                for band in required_bands:
                    arr = spectral_stack[band]
                    # Ensure numeric float np.ndarray
                    if isinstance(arr, np.ndarray):
                        # Squeeze all dimensions of size 1, not just the last one
                        # This handles cases where bands might be (H, W, 1) or (H, W)
                        arr = arr.squeeze()
                        # If still 3D, take first channel or flatten
                        if arr.ndim == 3:
                            if arr.shape[2] == 1:
                                arr = arr[:, :, 0]
                            else:
                                logger.warning(f"Band {band} has shape {arr.shape}, taking first channel")
                                arr = arr[:, :, 0]
                    arr = np.asarray(arr, dtype=np.float64)
                    # Ensure 2D
                    if arr.ndim != 2:
                        logger.error(f"Band {band} has unexpected shape {arr.shape} after processing")
                        raise ValueError(f"Band {band} must be 2D, got shape {arr.shape}")
                    band_data.append(arr)
                
                # Compute index (ensure float math)
                index_values = formula(*band_data).astype(np.float64)
                
                # Apply mask
                if mask is not None:
                    binary_mask = (np.asarray(mask).astype(np.int32) > 0)
                    masked_values = np.where(binary_mask, index_values, np.nan)
                else:
                    masked_values = index_values
                
                # Compute statistics
                valid_values = masked_values[~np.isnan(masked_values)]
                if len(valid_values) > 0:
                    stats = {
                        'mean': float(np.mean(valid_values)),
                        'std': float(np.std(valid_values)),
                        'min': float(np.min(valid_values)),
                        'max': float(np.max(valid_values)),
                        'median': float(np.median(valid_values)),
                        'q25': float(np.percentile(valid_values, 25)),
                        'q75': float(np.percentile(valid_values, 75)),
                        'nan_fraction': float(np.isnan(masked_values).sum() / masked_values.size)
                    }
                else:
                    stats = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                        'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'nan_fraction': 1.0
                    }
                
                indices[index_name] = {
                    'values': masked_values,
                    'statistics': stats
                }
                
                logger.debug(f"Computed {index_name}")
                
            except Exception as e:
                logger.error(f"Failed to compute {index_name}: {e}")
                continue
        
        return indices
    
    def create_vegetation_index_image(self, index_values: np.ndarray, 
                                    colormap: str = 'RdYlGn',
                                    vmin: Optional[float] = None,
                                    vmax: Optional[float] = None) -> np.ndarray:
        """
        Create visualization image for vegetation index.
        
        Args:
            index_values: Vegetation index values
            colormap: Matplotlib colormap name
            vmin: Minimum value for normalization
            vmax: Maximum value for normalization
            
        Returns:
            RGB image array
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            
            # Determine value range
            valid_values = index_values[~np.isnan(index_values)]
            if len(valid_values) == 0:
                return np.zeros((*index_values.shape, 3), dtype=np.uint8)
            
            if vmin is None:
                vmin = np.min(valid_values)
            if vmax is None:
                vmax = np.max(valid_values)
            
            # Normalize values
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap(colormap)
            
            # Apply colormap
            rgba_img = cmap(norm(index_values))
            rgba_img[np.isnan(index_values)] = [1, 1, 1, 1]  # White for NaN
            
            # Convert to RGB uint8
            rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
            
            return rgb_img
            
        except Exception as e:
            logger.error(f"Failed to create vegetation index image: {e}")
            return np.zeros((*index_values.shape, 3), dtype=np.uint8)
    
    def get_available_indices(self) -> list:
        """Get list of available vegetation indices."""
        return list(self.index_formulas.keys())
    
    def get_index_requirements(self, index_name: str) -> list:
        """
        Get required bands for a specific index.
        
        Args:
            index_name: Name of the vegetation index
            
        Returns:
            List of required band names
        """
        return self.index_bands.get(index_name, [])
    
    def validate_spectral_data(self, spectral_stack: Dict[str, np.ndarray]) -> bool:
        """
        Validate spectral data for vegetation index computation.
        
        Args:
            spectral_stack: Dictionary of spectral bands
            
        Returns:
            True if valid, False otherwise
        """
        if not spectral_stack:
            return False
        
        required_bands = ['nir', 'red', 'green', 'red_edge']
        if not all(band in spectral_stack for band in required_bands):
            logger.warning("Missing required spectral bands")
            return False
        
        # Check data shapes
        shapes = [arr.shape for arr in spectral_stack.values()]
        if not all(shape == shapes[0] for shape in shapes):
            logger.warning("Inconsistent spectral band shapes")
            return False
        
        return True
