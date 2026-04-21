"""
Segmentation manager for the Sorghum Pipeline.

This module handles image segmentation using the BRIA model
and provides post-processing capabilities.
"""

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SegmentationManager:
    """Manages image segmentation using BRIA model."""
    
    def __init__(self, 
                 model_name: str = "briaai/RMBG-2.0",
                 device: str = "auto",
                 threshold: float = 0.5,
                 trust_remote_code: bool = True,
                 cache_dir: Optional[str] = None,
                 local_files_only: bool = False):
        """
        Initialize segmentation manager.
        
        Args:
            model_name: Name of the BRIA model
            device: Device to run model on ("auto", "cpu", "cuda")
            threshold: Segmentation threshold
            trust_remote_code: Whether to trust remote code
            cache_dir: Hugging Face cache directory for model weights
            local_files_only: If True, only load from local cache
        """
        self.model_name = model_name
        self.threshold = threshold
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load the BRIA segmentation model."""
        try:
            logger.info(f"Loading BRIA model: {self.model_name}")
            
            # Import token if available
            import os
            token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir if self.cache_dir else None,
                local_files_only=self.local_files_only,
                token=token,  # Use token for authenticated access (faster)
            ).eval().to(self.device)
            
            # Define image transform
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            logger.info("BRIA model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BRIA model: {e}")
            raise
    
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Segment an image using the BRIA model.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Binary mask (0/255)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Apply transform
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(input_tensor)[-1].sigmoid().cpu()[0].squeeze(0).numpy()
            
            # Apply threshold
            mask = (predictions > self.threshold).astype(np.uint8) * 255
            
            # Resize back to original size
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            return mask_resized
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            # Return empty mask
            return np.zeros(image.shape[:2], dtype=np.uint8)

    def segment_image_soft(self, image: np.ndarray) -> np.ndarray:
        """
        Segment an image and return a soft mask in [0, 1] resized to original size.
        No thresholding or post-processing is applied.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Float mask in [0,1] with shape (H, W)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid().cpu()[0].squeeze(0).numpy()
            original_size = (image.shape[1], image.shape[0])
            soft_mask = cv2.resize(preds.astype(np.float32), original_size, interpolation=cv2.INTER_LINEAR)
            return np.clip(soft_mask, 0.0, 1.0)
        except Exception as e:
            logger.error(f"Soft segmentation failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.float32)
    
    def post_process_mask(self, mask: np.ndarray, 
                         min_area: int = 1000,
                         kernel_size: int = 5) -> np.ndarray:
        """
        Post-process segmentation mask.
        
        Args:
            mask: Input mask
            min_area: Minimum area for connected components
            kernel_size: Kernel size for morphological operations
            
        Returns:
            Post-processed mask
        """
        try:
            # Morphological opening to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Remove small connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                opened, connectivity=8
            )
            
            processed_mask = np.zeros_like(opened)
            for label in range(1, num_labels):  # Skip background
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    processed_mask[labels == label] = 255
            
            return processed_mask
            
        except Exception as e:
            logger.error(f"Mask post-processing failed: {e}")
            return mask
    
    def keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected component.
        
        Args:
            mask: Input mask
            
        Returns:
            Mask with only the largest component
        """
        try:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            
            if num_labels <= 1:
                return mask
            
            # Find the largest component (excluding background)
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = 1 + np.argmax(areas)
            
            # Create mask with only the largest component
            largest_mask = (labels == largest_label).astype(np.uint8) * 255
            
            return largest_mask
            
        except Exception as e:
            logger.error(f"Largest component extraction failed: {e}")
            return mask
    
    def validate_mask(self, mask: np.ndarray) -> bool:
        """
        Validate segmentation mask.
        
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
    
    def get_mask_properties(self, mask: np.ndarray) -> dict:
        """
        Get properties of the segmentation mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary of mask properties
        """
        if not self.validate_mask(mask):
            return {}
        
        try:
            # Convert to binary
            binary_mask = (mask > 127).astype(np.uint8)
            
            # Calculate properties
            area = np.sum(binary_mask)
            perimeter = 0
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                
                # Bounding box
                x, y, w, h = cv2.boundingRect(contours[0])
                bbox_area = w * h
                aspect_ratio = w / h if h > 0 else 0
            else:
                bbox_area = 0
                aspect_ratio = 0
            
            return {
                "area": int(area),
                "perimeter": float(perimeter),
                "bbox_area": int(bbox_area),
                "aspect_ratio": float(aspect_ratio),
                "coverage": float(area) / (mask.shape[0] * mask.shape[1]) if mask.size > 0 else 0.0,
                "num_components": len(contours)
            }
            
        except Exception as e:
            logger.error(f"Mask property calculation failed: {e}")
            return {}
    
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
        try:
            overlay = image.copy()
            overlay[mask == 255] = color
            return cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)
        except Exception as e:
            logger.error(f"Overlay creation failed: {e}")
            return image
