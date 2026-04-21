"""
Spectral feature extraction for the Sorghum Pipeline.

This module handles extraction of spectral features and analysis
of multispectral data.
"""

import numpy as np
import cv2
from sklearn.decomposition import PCA
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SpectralExtractor:
    """Extracts spectral features from multispectral data."""
    
    def __init__(self, n_components: int = 3):
        """
        Initialize spectral extractor.
        
        Args:
            n_components: Number of PCA components to extract
        """
        self.n_components = n_components
    
    def extract_spectral_features(self, spectral_stack: Dict[str, np.ndarray], 
                                mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Extract spectral features from multispectral data.
        
        Args:
            spectral_stack: Dictionary of spectral bands
            mask: Optional binary mask
            
        Returns:
            Dictionary containing spectral features
        """
        features = {}
        
        try:
            # Extract individual band features
            features['band_features'] = self._extract_band_features(spectral_stack, mask)
            
            # Extract PCA features
            features['pca_features'] = self._extract_pca_features(spectral_stack, mask)
            
            # Extract spectral indices
            features['spectral_indices'] = self._extract_spectral_indices(spectral_stack, mask)
            
            # Extract texture features from spectral bands
            features['spectral_texture'] = self._extract_spectral_texture(spectral_stack, mask)
            
            logger.debug("Spectral features extracted successfully")
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
        
        return features
    
    def _extract_band_features(self, spectral_stack: Dict[str, np.ndarray], 
                             mask: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Extract features from individual spectral bands."""
        band_features = {}
        
        for band_name, band_data in spectral_stack.items():
            try:
                # Squeeze to 2D if needed
                if band_data.ndim > 2:
                    band_data = band_data.squeeze()
                
                # Apply mask if provided
                if mask is not None and mask.shape == band_data.shape:
                    masked_data = np.where(mask > 0, band_data, np.nan)
                else:
                    masked_data = band_data
                
                # Compute statistics
                valid_data = masked_data[~np.isnan(masked_data)]
                if len(valid_data) > 0:
                    band_features[band_name] = {
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'median': float(np.median(valid_data)),
                        'q25': float(np.percentile(valid_data, 25)),
                        'q75': float(np.percentile(valid_data, 75)),
                        'skewness': float(self._compute_skewness(valid_data)),
                        'kurtosis': float(self._compute_kurtosis(valid_data)),
                        'entropy': float(self._compute_entropy(valid_data))
                    }
                else:
                    band_features[band_name] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                        'median': 0.0, 'q25': 0.0, 'q75': 0.0,
                        'skewness': 0.0, 'kurtosis': 0.0, 'entropy': 0.0
                    }
                
            except Exception as e:
                logger.error(f"Band feature extraction failed for {band_name}: {e}")
                band_features[band_name] = {}
        
        return band_features
    
    def _extract_pca_features(self, spectral_stack: Dict[str, np.ndarray], 
                            mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Extract PCA features from spectral data."""
        try:
            # Stack all bands
            band_names = ['nir', 'red_edge', 'red', 'green']
            band_data = []
            
            for band_name in band_names:
                if band_name in spectral_stack:
                    arr = spectral_stack[band_name].squeeze().astype(float)
                    if mask is not None and mask.shape == arr.shape:
                        arr = np.where(mask > 0, arr, np.nan)
                    band_data.append(arr)
            
            if not band_data:
                return {}
            
            # Stack bands
            full_stack = np.stack(band_data, axis=-1)
            h, w, c = full_stack.shape
            
            # Reshape for PCA
            flat_data = full_stack.reshape(-1, c)
            valid_mask = ~np.isnan(flat_data).any(axis=1)
            
            if valid_mask.sum() == 0:
                return {}
            
            # Apply PCA
            valid_data = flat_data[valid_mask]
            pca = PCA(n_components=min(self.n_components, valid_data.shape[1]))
            pca_result = pca.fit_transform(valid_data)
            
            # Create full result array
            full_result = np.full((h * w, self.n_components), np.nan)
            full_result[valid_mask] = pca_result
            
            # Reshape back to image dimensions
            pca_components = {}
            for i in range(self.n_components):
                component = full_result[:, i].reshape(h, w)
                pca_components[f'pca_{i+1}'] = component
                
                # Compute statistics for this component
                valid_component = component[~np.isnan(component)]
                if len(valid_component) > 0:
                    pca_components[f'pca_{i+1}_stats'] = {
                        'mean': float(np.mean(valid_component)),
                        'std': float(np.std(valid_component)),
                        'min': float(np.min(valid_component)),
                        'max': float(np.max(valid_component))
                    }
            
            # Add PCA metadata
            pca_components['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
            pca_components['total_variance_explained'] = float(np.sum(pca.explained_variance_ratio_))
            
            return pca_components
            
        except Exception as e:
            logger.error(f"PCA feature extraction failed: {e}")
            return {}
    
    def _extract_spectral_indices(self, spectral_stack: Dict[str, np.ndarray], 
                                mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Extract basic spectral indices."""
        indices = {}
        
        try:
            # Get required bands
            nir = spectral_stack.get('nir', None)
            red = spectral_stack.get('red', None)
            green = spectral_stack.get('green', None)
            red_edge = spectral_stack.get('red_edge', None)
            
            if nir is not None:
                nir = nir.squeeze().astype(float)
            if red is not None:
                red = red.squeeze().astype(float)
            if green is not None:
                green = green.squeeze().astype(float)
            if red_edge is not None:
                red_edge = red_edge.squeeze().astype(float)
            
            # Apply mask
            if mask is not None:
                if nir is not None and mask.shape == nir.shape:
                    nir = np.where(mask > 0, nir, np.nan)
                if red is not None and mask.shape == red.shape:
                    red = np.where(mask > 0, red, np.nan)
                if green is not None and mask.shape == green.shape:
                    green = np.where(mask > 0, green, np.nan)
                if red_edge is not None and mask.shape == red_edge.shape:
                    red_edge = np.where(mask > 0, red_edge, np.nan)
            
            # Compute basic indices
            if nir is not None and red is not None:
                indices['nir_red_ratio'] = nir / (red + 1e-10)
                indices['nir_red_diff'] = nir - red
            
            if nir is not None and green is not None:
                indices['nir_green_ratio'] = nir / (green + 1e-10)
                indices['nir_green_diff'] = nir - green
            
            if red is not None and green is not None:
                indices['red_green_ratio'] = red / (green + 1e-10)
                indices['red_green_diff'] = red - green
            
            if nir is not None and red_edge is not None:
                indices['nir_red_edge_ratio'] = nir / (red_edge + 1e-10)
                indices['nir_red_edge_diff'] = nir - red_edge
            
            # Compute band ratios
            if nir is not None and red is not None and green is not None:
                indices['nir_red_green_sum'] = nir + red + green
                indices['nir_red_green_mean'] = (nir + red + green) / 3
            
        except Exception as e:
            logger.error(f"Spectral index extraction failed: {e}")
        
        return indices
    
    def _extract_spectral_texture(self, spectral_stack: Dict[str, np.ndarray], 
                                mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Extract texture features from spectral bands."""
        texture_features = {}
        
        try:
            from .texture import TextureExtractor
            
            texture_extractor = TextureExtractor()
            
            for band_name, band_data in spectral_stack.items():
                try:
                    # Prepare grayscale image
                    gray_data = band_data.squeeze().astype(float)
                    
                    # Apply mask
                    if mask is not None and mask.shape == gray_data.shape:
                        gray_data = np.where(mask > 0, gray_data, np.nan)
                    
                    # Normalize to 0-255
                    valid_data = gray_data[~np.isnan(gray_data)]
                    if len(valid_data) > 0:
                        m, M = np.min(valid_data), np.max(valid_data)
                        if M > m:
                            normalized = ((gray_data - m) / (M - m) * 255).astype(np.uint8)
                        else:
                            normalized = np.zeros_like(gray_data, dtype=np.uint8)
                    else:
                        normalized = np.zeros_like(gray_data, dtype=np.uint8)
                    
                    # Extract texture features
                    band_texture = texture_extractor.extract_all_texture_features(normalized)
                    texture_features[band_name] = band_texture
                    
                except Exception as e:
                    logger.error(f"Spectral texture extraction failed for {band_name}: {e}")
                    texture_features[band_name] = {}
        
        except ImportError:
            logger.warning("TextureExtractor not available for spectral texture analysis")
        
        return texture_features
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute entropy of data."""
        if len(data) == 0:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        
        # Remove zero probabilities
        hist = hist[hist > 0]
        
        # Compute entropy
        return -np.sum(hist * np.log2(hist))
    
    def create_spectral_visualization(self, spectral_stack: Dict[str, np.ndarray], 
                                    pca_features: Dict[str, Any]) -> np.ndarray:
        """
        Create visualization of spectral features.
        
        Args:
            spectral_stack: Original spectral data
            pca_features: PCA features
            
        Returns:
            Visualization image
        """
        try:
            # Preferred visualization: RGB = (Red, Red-Edge, Green)
            if 'red' in spectral_stack and 'red_edge' in spectral_stack and 'green' in spectral_stack:
                red = spectral_stack['red'].squeeze()
                red_edge = spectral_stack['red_edge'].squeeze()
                green = spectral_stack['green'].squeeze()

                # Normalize each band
                red_norm = self._normalize_band(red)
                red_edge_norm = self._normalize_band(red_edge)
                green_norm = self._normalize_band(green)

                # Create composite (Red, Red-Edge, Green)
                rgb_composite = np.stack([red_norm, red_edge_norm, green_norm], axis=-1)

                return rgb_composite.astype(np.uint8)

            # Fallback visualization: RGB = (NIR, Red, Green)
            if 'red' in spectral_stack and 'green' in spectral_stack and 'nir' in spectral_stack:
                red = spectral_stack['red'].squeeze()
                green = spectral_stack['green'].squeeze()
                nir = spectral_stack['nir'].squeeze()

                # Normalize each band
                red_norm = self._normalize_band(red)
                green_norm = self._normalize_band(green)
                nir_norm = self._normalize_band(nir)

                rgb_composite = np.stack([nir_norm, red_norm, green_norm], axis=-1)

                return rgb_composite.astype(np.uint8)
            
            # Fallback to first PCA component
            elif 'pca_1' in pca_features:
                pca1 = pca_features['pca_1']
                pca1_norm = self._normalize_band(pca1)
                return np.stack([pca1_norm, pca1_norm, pca1_norm], axis=-1).astype(np.uint8)
            
            else:
                # Return empty image
                return np.zeros((100, 100, 3), dtype=np.uint8)
        
        except Exception as e:
            logger.error(f"Spectral visualization creation failed: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        """Normalize band to 0-255 range."""
        valid_data = band[~np.isnan(band)]
        if len(valid_data) == 0:
            return np.zeros_like(band, dtype=np.uint8)
        
        m, M = np.min(valid_data), np.max(valid_data)
        if M > m:
            normalized = ((band - m) / (M - m) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(band, dtype=np.uint8)
        
        return normalized
