"""
Image preprocessing functionality for the Sorghum Pipeline.

This module handles image preprocessing, composite creation,
and basic image transformations.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, Any, Optional
from itertools import product
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing and composite creation."""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target size for image resizing (width, height)
        """
        self.target_size = target_size
    
    def convert_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert array to uint8 format with proper normalization.
        
        Args:
            arr: Input array
            
        Returns:
            Normalized uint8 array
        """
        # Handle NaN and infinite values
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to 0-255 range
        if np.ptp(arr) > 0:
            normalized = (arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255
        else:
            normalized = np.zeros_like(arr)
        
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    def process_raw_image(self, pil_img: Image.Image) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Process raw 4-band image into composite and spectral bands.
        
        Args:
            pil_img: PIL Image object containing 4-band data
            
        Returns:
            Tuple of (composite_image, spectral_bands_dict)
        """
        # Split the 4-band RAW into tiles and stack them
        d = pil_img.size[0] // 2
        boxes = [
            (j, i, j + d, i + d)
            for i, j in product(
                range(0, pil_img.height, d),
                range(0, pil_img.width, d)
            )
        ]
        
        # Extract tiles and stack them
        stack = np.stack([
            np.array(pil_img.crop(box), dtype=float) 
            for box in boxes
        ], axis=-1)
        
        # Bands come in order: [green, red, red_edge, nir]
        green, red, red_edge, nir = np.split(stack, 4, axis=-1)
        
        # Build pseudo-RGB composite as (green, red_edge, red)
        composite = np.concatenate([green, red_edge, red], axis=-1)
        composite_uint8 = self.convert_to_uint8(composite)
        
        # Prepare spectral stack (ensure 2D format by squeezing last dimension)
        spectral_bands = {
            "green": green.squeeze(-1) if green.ndim == 3 and green.shape[2] == 1 else green,
            "red": red.squeeze(-1) if red.ndim == 3 and red.shape[2] == 1 else red,
            "red_edge": red_edge.squeeze(-1) if red_edge.ndim == 3 and red_edge.shape[2] == 1 else red_edge,
            "nir": nir.squeeze(-1) if nir.ndim == 3 and nir.shape[2] == 1 else nir
        }
        
        return composite_uint8, spectral_bands
    
    def create_composites(self, plants: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Create composites for all plants in the dataset.
        
        Args:
            plants: Dictionary of plant data
            
        Returns:
            Updated plant data with composites and spectral stacks
        """
        logger.info("Creating composites for all plants...")
        
        for key, pdata in plants.items():
            try:
                # Find the PIL Image
                if "raw_image" in pdata:
                    image, _ = pdata["raw_image"]
                elif "raw_images" in pdata and pdata["raw_images"]:
                    image, _ = pdata["raw_images"][0]
                else:
                    logger.warning(f"No raw image found for {key}")
                    continue
                
                # Process the image
                composite, spectral_stack = self.process_raw_image(image)
                
                # Store results
                pdata["composite"] = composite
                pdata["spectral_stack"] = spectral_stack
                
                logger.debug(f"Created composite for {key}")
                
            except Exception as e:
                logger.error(f"Failed to create composite for {key}: {e}")
                continue
        
        logger.info("Composite creation completed")
        return plants
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (width, height). If None, uses self.target_size
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        if target_size is None:
            return image
        
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normalize image using specified method.
        
        Args:
            image: Input image
            method: Normalization method ("minmax", "zscore", "robust")
            
        Returns:
            Normalized image
        """
        if method == "minmax":
            if image.dtype == np.uint8:
                return image.astype(np.float32) / 255.0
            else:
                img_min, img_max = image.min(), image.max()
                if img_max > img_min:
                    return (image - img_min) / (img_max - img_min)
                else:
                    return np.zeros_like(image, dtype=np.float32)
        
        elif method == "zscore":
            mean, std = image.mean(), image.std()
            if std > 0:
                return (image - mean) / std
            else:
                return np.zeros_like(image, dtype=np.float32)
        
        elif method == "robust":
            q25, q75 = np.percentile(image, [25, 75])
            if q75 > q25:
                return (image - q25) / (q75 - q25)
            else:
                return np.zeros_like(image, dtype=np.float32)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel
            
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter to image.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        
        return cv2.filter2D(image, -1, kernel)
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.2, beta: int = 15) -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0 = no change)
            beta: Brightness control (0 = no change)
            
        Returns:
            Enhanced image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def create_overlay(self, base_image: np.ndarray, mask: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0), 
                      alpha: float = 0.5) -> np.ndarray:
        """
        Create overlay of mask on base image.
        
        Args:
            base_image: Base image
            mask: Binary mask
            color: Overlay color (B, G, R)
            alpha: Overlay transparency
            
        Returns:
            Image with overlay
        """
        overlay = base_image.copy()
        overlay[mask == 255] = color
        return cv2.addWeighted(base_image, 1.0 - alpha, overlay, alpha, 0)
    
    def validate_composite(self, composite: np.ndarray) -> bool:
        """
        Validate composite image.
        
        Args:
            composite: Composite image to validate
            
        Returns:
            True if valid, False otherwise
        """
        if composite is None:
            return False
        
        if not isinstance(composite, np.ndarray):
            return False
        
        if composite.ndim != 3 or composite.shape[2] != 3:
            return False
        
        if composite.dtype != np.uint8:
            return False
        
        return True
