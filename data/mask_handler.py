"""
Mask handling functionality for the Sorghum Pipeline.

This module handles mask creation, processing, and validation
for plant segmentation tasks.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class MaskHandler:
    """Handles mask creation, processing, and validation."""
    
    def __init__(self, min_area: int = 1000, kernel_size: int = 7):
        """
        Initialize the mask handler.
        
        Args:
            min_area: Minimum area for connected components
            kernel_size: Kernel size for morphological operations
        """
        self.min_area = min_area
        self.kernel_size = kernel_size
    
    def create_bounding_box_mask(self, image_shape: Tuple[int, int], 
                                bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create a mask from bounding box coordinates.
        
        Args:
            image_shape: Shape of the image (height, width)
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Binary mask array
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x1, y1, x2, y2 = bbox
        # Clamp coordinates to image bounds
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        
        mask[y1:y2, x1:x2] = 255
        return mask
    
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess mask by cleaning and filtering.
        
        Args:
            mask: Input mask
            
        Returns:
            Cleaned mask
        """
        if mask is None:
            return None
        
        # Convert to binary if needed
        if isinstance(mask, tuple):
            mask = mask[0]
        
        # Ensure binary format
        mask = ((mask.astype(np.int32) > 0).astype(np.uint8)) * 255
        
        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.kernel_size, self.kernel_size)
        )
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            opened, connectivity=8
        )
        
        clean_mask = np.zeros_like(opened)
        for label in range(1, num_labels):  # Skip background
            if stats[label, cv2.CC_STAT_AREA] >= self.min_area:
                clean_mask[labels == label] = 255
        
        return clean_mask
    
    def keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected component in the mask.
        
        Args:
            mask: Input mask
            
        Returns:
            Mask with only the largest component
        """
        if mask is None:
            return None
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        
        if num_labels <= 1:
            return mask
        
        # Find the largest component (excluding background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        
        # Create mask with only the largest component
        largest_mask = (labels == largest_label).astype(np.uint8) * 255
        
        return largest_mask
    
    def apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to image.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Masked image
        """
        if mask is None:
            return image
        
        return cv2.bitwise_and(image, image, mask=mask)
    
    def create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0), 
                      alpha: float = 0.5) -> np.ndarray:
        """
        Create overlay of mask on image.
        
        Args:
            image: Base image
            mask: Binary mask
            color: Overlay color (B, G, R)
            alpha: Overlay transparency
            
        Returns:
            Image with mask overlay
        """
        overlay = image.copy()
        overlay[mask == 255] = color
        return cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)
    
    def get_mask_properties(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Get properties of the mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary of mask properties
        """
        if mask is None:
            return {}
        
        # Convert to binary
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Calculate properties
        area = np.sum(binary_mask)
        perimeter = cv2.arcLength(
            cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], 
            True
        ) if len(cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) > 0 else 0
        
        # Bounding box
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            bbox_area = w * h
            aspect_ratio = w / h if h > 0 else 0
        else:
            bbox_area = 0
            aspect_ratio = 0
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "bbox_area": float(bbox_area),
            "aspect_ratio": float(aspect_ratio),
            "coverage": float(area) / (mask.shape[0] * mask.shape[1]) if mask.size > 0 else 0.0
        }
    
    def validate_mask(self, mask: np.ndarray) -> bool:
        """
        Validate mask format and content.
        
        Args:
            mask: Mask to validate
            
        Returns:
            True if valid, False otherwise
        """
        if mask is None:
            return False
        
        if not isinstance(mask, np.ndarray):
            return False
        
        if mask.ndim != 2:
            return False
        
        if mask.dtype not in [np.uint8, np.bool_]:
            return False
        
        # Check if mask has any foreground pixels
        if np.sum(mask > 0) == 0:
            logger.warning("Mask has no foreground pixels")
            return False
        
        return True
    
    def resize_mask(self, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize mask to target size.
        
        Args:
            mask: Input mask
            target_size: Target size (width, height)
            
        Returns:
            Resized mask
        """
        if mask is None:
            return None
        
        return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    def dilate_mask(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Dilate mask to expand foreground regions.
        
        Args:
            mask: Input mask
            kernel_size: Size of dilation kernel
            
        Returns:
            Dilated mask
        """
        if mask is None:
            return None
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask, kernel, iterations=1)
    
    def erode_mask(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Erode mask to shrink foreground regions.
        
        Args:
            mask: Input mask
            kernel_size: Size of erosion kernel
            
        Returns:
            Eroded mask
        """
        if mask is None:
            return None
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.erode(mask, kernel, iterations=1)
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in the mask.
        
        Args:
            mask: Input mask
            
        Returns:
            Mask with filled holes
        """
        if mask is None:
            return None
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create filled mask
        filled_mask = np.zeros_like(mask)
        cv2.fillPoly(filled_mask, contours, 255)
        
        return filled_mask
