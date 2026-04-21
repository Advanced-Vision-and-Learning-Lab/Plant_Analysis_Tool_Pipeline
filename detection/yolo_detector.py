"""
YOLOv12 detection for the Sorghum Pipeline.

This module handles object detection using YOLOv12 before segmentation.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not available. YOLO detection will be disabled.")

# Note: YOLOv12 models (yolov12n.pt, yolov12s.pt, etc.) need to be downloaded
# from the yolov12 repository: https://github.com/sunsmarterjie/yolov12
# Or use standard YOLO models (yolo8n.pt, yolo11n.pt, etc.) which work with ultralytics


class YOLODetector:
    """Handles YOLOv12 object detection."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_name: str = "yolov12n.pt",
                 device: str = "auto",
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to custom YOLO model weights (.pt file)
            model_name: Name of YOLO model (yolov12n, yolov12s, yolov12m, yolov12l, yolov12x)
            device: Device to run on ("auto", "cpu", "cuda", "0", "1", etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model = None
        self.model_path = model_path
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Determine device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        
        if YOLO_AVAILABLE:
            self._load_model()
        else:
            logger.warning("YOLO not available. Detection will be skipped.")
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading YOLO model from: {self.model_path}")
                logger.info(f"Model file exists: {Path(self.model_path).exists()}, size: {Path(self.model_path).stat().st_size / (1024*1024):.2f} MB")
                self.model = YOLO(self.model_path)
            else:
                if self.model_path:
                    logger.warning(f"YOLO model path specified but file not found: {self.model_path}")
                logger.info(f"Loading YOLO model: {self.model_name}")
                # Try to load the model (will download if not cached)
                # For YOLOv12 models, you may need to download them manually from:
                # https://github.com/sunsmarterjie/yolov12/releases
                try:
                    self.model = YOLO(self.model_name)
                except Exception as e:
                    # If YOLOv12 model not found, try alternative names or suggest download
                    logger.warning(f"Failed to load {self.model_name}: {e}")
                    logger.info("Trying alternative: yolov8n.pt (YOLOv8 nano)")
                    try:
                        self.model = YOLO("yolov8n.pt")
                        logger.info("Using YOLOv8n as fallback")
                    except Exception:
                        logger.error("Failed to load any YOLO model. Please install YOLOv12 or use YOLOv8/YOLOv11")
                        self.model = None
                        return
            
            logger.info(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.info("Note: For YOLOv12 models, download from: https://github.com/sunsmarterjie/yolov12/releases")
            self.model = None
    
    def detect(self, image: np.ndarray, 
               classes: Optional[List[int]] = None,
               return_boxes: bool = True,
               return_scores: bool = True) -> Dict[str, Any]:
        """
        Run object detection on an image.
        
        Args:
            image: Input image (RGB or BGR format, numpy array)
            classes: List of class IDs to detect (None = all classes)
            return_boxes: Whether to return bounding boxes
            return_scores: Whether to return confidence scores
            
        Returns:
            Dictionary containing:
                - boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
                - scores: List of confidence scores
                - class_ids: List of class IDs
                - class_names: List of class names
                - largest_box: Largest bounding box (x1, y1, x2, y2) or None
        """
        if self.model is None:
            logger.warning("YOLO model not loaded. Returning empty detections.")
            return {
                "boxes": [],
                "scores": [],
                "class_ids": [],
                "class_names": [],
                "largest_box": None
            }
        
        try:
            # YOLO (ultralytics) expects RGB images
            # Image might be RGB, BGR, or grayscale
            if len(image.shape) == 2:
                # Grayscale, convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                # Check if it's BGR (OpenCV default) or RGB
                # We'll assume it's already RGB if passed from pipeline (composite is RGB)
                # But if it's BGR, convert to RGB
                # Simple heuristic: if image comes from OpenCV operations, it's likely BGR
                # For now, assume RGB (pipeline passes RGB composites)
                rgb_image = image.copy()
                # If you know it's BGR, uncomment: rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            
            # Run inference
            # YOLOv12 models have compatibility issues with some ultralytics versions
            # If prediction fails, we'll catch and return empty detections
            try:
                results = self.model.predict(
                    rgb_image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=classes,
                    device=self.device,
                    verbose=False
                )
            except AttributeError as e:
                if "'AAttn' object has no attribute 'qkv'" in str(e):
                    logger.error(f"YOLOv12 compatibility issue detected. The model '{self.model_path or self.model_name}' is not compatible with ultralytics {__import__('ultralytics').__version__}. Please use YOLOv8/YOLOv11 models or update ultralytics.")
                    raise
                else:
                    raise
            
            # Extract detections
            boxes = []
            scores = []
            class_ids = []
            class_names = []
            vase_boxes = []  # Store vase boxes separately to exclude from potted plant
            
            # Identify vase/vast classes to exclude from primary selection but keep for exclusion
            excluded_from_primary = ['vase', 'vast']
            
            if len(results) > 0 and results[0].boxes is not None:
                num_detections = len(results[0].boxes)
                logger.debug(f"YOLO found {num_detections} raw detections")
                boxes_tensor = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
                scores_tensor = results[0].boxes.conf.cpu().numpy()  # (N,)
                class_ids_tensor = results[0].boxes.cls.cpu().numpy().astype(int)  # (N,)
                
                for i in range(len(boxes_tensor)):
                    class_id = int(class_ids_tensor[i])
                    class_name = results[0].names[class_id]
                    class_name_lower = class_name.lower()
                    
                    x1, y1, x2, y2 = boxes_tensor[i]
                    box_coords = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Store vase boxes separately (don't use as primary, but keep for exclusion)
                    if any(excluded in class_name_lower for excluded in excluded_from_primary):
                        vase_boxes.append(box_coords)
                        logger.debug(f"Found vase detection: {class_name} at {box_coords} (will be excluded from potted plant)")
                        continue  # Don't add to main boxes list
                    
                    # Add all other detections (potted plant, etc.)
                    boxes.append(box_coords)
                    scores.append(float(scores_tensor[i]))
                    class_ids.append(class_id)
                    class_names.append(class_name)
            
            # Find largest bounding box (by area) from remaining detections
            # Prefer "potted plant" class if available, otherwise use largest by area
            largest_box = None
            if boxes:
                areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
                
                # First, try to find a "potted plant" detection
                potted_plant_indices = [i for i, name in enumerate(class_names) 
                                       if 'potted plant' in name.lower() or ('potted' in name.lower() and 'plant' in name.lower())]
                
                if potted_plant_indices:
                    # Use the largest potted plant detection
                    potted_areas = [(i, areas[i]) for i in potted_plant_indices]
                    largest_potted_idx = max(potted_areas, key=lambda x: x[1])[0]
                    largest_box = boxes[largest_potted_idx]
                    logger.info(f"YOLO found {len(boxes)} detections after filtering. Selected potted plant as primary.")
                    logger.info(f"Primary box: {class_names[largest_potted_idx]} with area {areas[largest_potted_idx]}")
                else:
                    # No potted plant found, use largest by area
                    largest_idx = np.argmax(areas)
                    largest_box = boxes[largest_idx]
                    logger.info(f"YOLO returning {len(boxes)} detections after filtering. Classes: {class_names}")
                    logger.info(f"Largest box (primary) is: {class_names[largest_idx]} with area {areas[largest_idx]}")
            else:
                logger.warning("YOLO found no detections after filtering out vase. This may indicate no potted plant was detected.")
            
            return {
                "boxes": boxes,
                "scores": scores,
                "class_ids": class_ids,
                "class_names": class_names,
                "largest_box": largest_box,
                "vase_boxes": vase_boxes  # Store vase boxes to exclude from segmentation
            }
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return {
                "boxes": [],
                "scores": [],
                "class_ids": [],
                "class_names": [],
                "largest_box": None,
                "vase_boxes": []
            }
    
    def get_crop_box(self, image: np.ndarray, 
                     padding: int = 10,
                     min_size: int = 100) -> Optional[Tuple[int, int, int, int]]:
        """
        Get a bounding box to crop the image based on detections.
        Returns the largest detection with padding, or None if no detections.
        
        Args:
            image: Input image
            padding: Padding to add around the bounding box
            min_size: Minimum size for the bounding box
            
        Returns:
            Bounding box (x1, y1, x2, y2) or None
        """
        detections = self.detect(image)
        largest_box = detections.get("largest_box")
        
        if largest_box is None:
            return None
        
        x1, y1, x2, y2 = largest_box
        h, w = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Check minimum size
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
        
        return (x1, y1, x2, y2)

