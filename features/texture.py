"""
Texture feature extraction for the Sorghum Pipeline.

This module handles extraction of texture features including:
- Local Binary Patterns (LBP)
- Histogram of Oriented Gradients (HOG)
- Lacunarity features
- Edge Histogram Descriptor (EHD)
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.feature import local_binary_pattern, hog
from skimage import exposure
from scipy import ndimage, signal
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TextureExtractor:
    """Extracts texture features from images."""
    
    def __init__(self, 
                 lbp_points: int = 8,
                 lbp_radius: int = 1,
                 hog_orientations: int = 9,
                 hog_pixels_per_cell: Tuple[int, int] = (8, 8),
                 hog_cells_per_block: Tuple[int, int] = (2, 2),
                 lacunarity_window: int = 15,
                 ehd_threshold: float = 0.3,
                 angle_resolution: int = 45):
        """
        Initialize texture extractor.
        
        Args:
            lbp_points: Number of points for LBP
            lbp_radius: Radius for LBP
            hog_orientations: Number of orientations for HOG
            hog_pixels_per_cell: Pixels per cell for HOG
            hog_cells_per_block: Cells per block for HOG
            lacunarity_window: Window size for lacunarity
            ehd_threshold: Threshold for EHD
            angle_resolution: Angle resolution for EHD
        """
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.lacunarity_window = lacunarity_window
        self.ehd_threshold = ehd_threshold
        self.angle_resolution = angle_resolution
    
    def extract_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern features.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            LBP feature map
        """
        try:
            lbp = local_binary_pattern(
                gray_image, 
                self.lbp_points, 
                self.lbp_radius, 
                method='uniform'
            )
            return self._convert_to_uint8(lbp)
        except Exception as e:
            logger.error(f"LBP extraction failed: {e}")
            return np.zeros_like(gray_image, dtype=np.uint8)
    
    def extract_hog(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients features.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            HOG feature map
        """
        try:
            _, vis = hog(
                gray_image,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                visualize=True,
                feature_vector=True
            )
            return exposure.rescale_intensity(vis, out_range=(0, 255)).astype(np.uint8)
        except Exception as e:
            logger.error(f"HOG extraction failed: {e}")
            return np.zeros_like(gray_image, dtype=np.uint8)
    
    def compute_local_lacunarity(self, gray_image: np.ndarray, window_size: int) -> np.ndarray:
        """
        Compute local lacunarity.
        
        Args:
            gray_image: Grayscale input image
            window_size: Size of the sliding window
            
        Returns:
            Local lacunarity map
        """
        try:
            arr = gray_image.astype(np.float32)
            m1 = ndimage.uniform_filter(arr, size=window_size)
            m2 = ndimage.uniform_filter(arr * arr, size=window_size)
            var = m2 - m1 * m1
            eps = 1e-6
            lac = var / (m1 * m1 + eps) + 1
            lac[m1 <= eps] = 0
            return lac
        except Exception as e:
            logger.error(f"Local lacunarity computation failed: {e}")
            return np.zeros_like(gray_image, dtype=np.float32)
    
    def compute_lacunarity_features(self, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute three types of lacunarity features.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Tuple of (lac1, lac2, lac3) lacunarity maps
        """
        try:
            # L1: Single window lacunarity
            lac1 = self.compute_local_lacunarity(gray_image, self.lacunarity_window)
            
            # L2: Multi-scale lacunarity
            scales = [max(3, self.lacunarity_window//2), self.lacunarity_window, self.lacunarity_window*2]
            lac2 = np.mean([
                self.compute_local_lacunarity(gray_image, s) for s in scales
            ], axis=0)
            
            # L3: DBC Lacunarity (if available)
            try:
                from ..models.dbc_lacunarity import DBC_Lacunarity
                x = torch.from_numpy(gray_image.astype(np.float32)/255.0)[None, None]
                layer = DBC_Lacunarity(window_size=self.lacunarity_window).eval()
                with torch.no_grad():
                    lac3 = layer(x).squeeze().cpu().numpy()
            except ImportError:
                logger.warning("DBC Lacunarity not available, using L2 as L3")
                lac3 = lac2.copy()
            
            return (
                self._convert_to_uint8(lac1),
                self._convert_to_uint8(lac2), 
                self._convert_to_uint8(lac3)
            )
        except Exception as e:
            logger.error(f"Lacunarity features computation failed: {e}")
            empty = np.zeros_like(gray_image, dtype=np.uint8)
            return empty, empty, empty
    
    def generate_ehd_masks(self, mask_size: int = 3) -> np.ndarray:
        """
        Generate masks for Edge Histogram Descriptor.
        
        Args:
            mask_size: Size of the mask
            
        Returns:
            Array of EHD masks
        """
        if mask_size < 3:
            mask_size = 3
        if mask_size % 2 == 0:
            mask_size += 1
        
        # Base gradient mask
        Gy = np.outer([1, 0, -1], [1, 2, 1])
        
        # Expand if needed
        if mask_size > 3:
            expd = np.outer([1, 2, 1], [1, 2, 1])
            for _ in range((mask_size - 3) // 2):
                Gy = signal.convolve2d(expd, Gy, mode='full')
        
        # Generate masks for different angles
        angles = np.arange(0, 360, self.angle_resolution)
        masks = np.zeros((len(angles), mask_size, mask_size), dtype=np.float32)
        
        for i, angle in enumerate(angles):
            masks[i] = ndimage.rotate(Gy, angle, reshape=False, mode='nearest')
        
        return masks
    
    def extract_ehd_features(self, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Edge Histogram Descriptor features.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Tuple of (ehd_features, ehd_map)
        """
        try:
            # Generate masks
            masks = self.generate_ehd_masks()
            
            # Convert to tensor
            X = torch.from_numpy(gray_image.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0)
            masks_tensor = torch.tensor(masks).unsqueeze(1).float()
            
            # Convolve with masks
            edge_responses = F.conv2d(X, masks_tensor, dilation=7)
            
            # Find maximum response
            values, indices = torch.max(edge_responses, dim=1)
            indices[values < self.ehd_threshold] = masks.shape[0]
            
            # Pool features
            feat_vect = []
            for edge in range(masks.shape[0] + 1):
                pooled = F.avg_pool2d(
                    (indices == edge).unsqueeze(1).float(),
                    kernel_size=5, stride=1, padding=2
                )
                feat_vect.append(pooled.squeeze(1))
            
            ehd_features = torch.stack(feat_vect, dim=1).squeeze(0).cpu().numpy()
            ehd_map = np.argmax(ehd_features, axis=0).astype(np.uint8)
            
            return ehd_features, ehd_map
            
        except Exception as e:
            logger.error(f"EHD features extraction failed: {e}")
            empty_features = np.zeros((9, gray_image.shape[0]-4, gray_image.shape[1]-4), dtype=np.float32)
            empty_map = np.zeros_like(gray_image, dtype=np.uint8)
            return empty_features, empty_map
    
    def extract_all_texture_features(self, gray_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all texture features from a grayscale image.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Dictionary of texture features
        """
        features = {}
        
        try:
            # LBP
            features['lbp'] = self.extract_lbp(gray_image)
            
            # HOG
            features['hog'] = self.extract_hog(gray_image)
            
            # Lacunarity
            lac1, lac2, lac3 = self.compute_lacunarity_features(gray_image)
            features['lac1'] = lac1
            features['lac2'] = lac2
            features['lac3'] = lac3
            
            # EHD
            ehd_features, ehd_map = self.extract_ehd_features(gray_image)
            features['ehd_features'] = ehd_features
            features['ehd_map'] = ehd_map
            
            logger.debug("All texture features extracted successfully")
            
        except Exception as e:
            logger.error(f"Texture feature extraction failed: {e}")
            # Return empty features
            features = {
                'lbp': np.zeros_like(gray_image, dtype=np.uint8),
                'hog': np.zeros_like(gray_image, dtype=np.uint8),
                'lac1': np.zeros_like(gray_image, dtype=np.uint8),
                'lac2': np.zeros_like(gray_image, dtype=np.uint8),
                'lac3': np.zeros_like(gray_image, dtype=np.uint8),
                'ehd_features': np.zeros((9, gray_image.shape[0]-4, gray_image.shape[1]-4), dtype=np.float32),
                'ehd_map': np.zeros_like(gray_image, dtype=np.uint8)
            }
        
        return features
    
    def _convert_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """Convert array to uint8 with proper normalization."""
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if np.ptp(arr) > 0:
            normalized = (arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255
        else:
            normalized = np.zeros_like(arr)
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    def compute_texture_statistics(self, features: Dict[str, np.ndarray], 
                                 mask: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for texture features.
        
        Args:
            features: Dictionary of texture features
            mask: Optional mask to apply
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        
        for feature_name, feature_data in features.items():
            if feature_name == 'ehd_features':
                # Special handling for EHD features
                if mask is not None:
                    # Apply mask to each channel
                    masked_features = []
                    for i in range(feature_data.shape[0]):
                        channel = feature_data[i]
                        if mask.shape != channel.shape:
                            # Resize mask to match channel
                            mask_resized = cv2.resize(mask, (channel.shape[1], channel.shape[0]), 
                                                    interpolation=cv2.INTER_NEAREST)
                            masked_channel = np.where(mask_resized > 0, channel, np.nan)
                        else:
                            masked_channel = np.where(mask > 0, channel, np.nan)
                        masked_features.append(masked_channel)
                    feature_data = np.stack(masked_features, axis=0)
                else:
                    feature_data = feature_data
                
                # Compute statistics for each EHD channel
                channel_stats = {}
                for i in range(feature_data.shape[0]):
                    channel = feature_data[i]
                    valid_data = channel[~np.isnan(channel)]
                    if len(valid_data) > 0:
                        channel_stats[f'channel_{i}'] = {
                            'mean': float(np.mean(valid_data)),
                            'std': float(np.std(valid_data)),
                            'min': float(np.min(valid_data)),
                            'max': float(np.max(valid_data)),
                            'median': float(np.median(valid_data))
                        }
                stats[feature_name] = channel_stats
            else:
                # Regular 2D features
                if mask is not None and mask.shape == feature_data.shape:
                    masked_data = np.where(mask > 0, feature_data, np.nan)
                else:
                    masked_data = feature_data
                
                valid_data = masked_data[~np.isnan(masked_data)]
                if len(valid_data) > 0:
                    stats[feature_name] = {
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'median': float(np.median(valid_data))
                    }
                else:
                    stats[feature_name] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0
                    }
        
        return stats
