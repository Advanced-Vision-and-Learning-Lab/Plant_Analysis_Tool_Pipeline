"""
Morphological feature extraction for the Sorghum Pipeline.

This module handles extraction of morphological features using PlantCV
and other computer vision techniques.
"""

import numpy as np
import cv2
import contextlib
import sys
from typing import Dict, Any, Optional, List, Tuple
import logging

# Try to import PlantCV, but don't fail if not available
try:
    from plantcv import plantcv as pcv
    PLANT_CV_AVAILABLE = True
except ImportError:
    PLANT_CV_AVAILABLE = False
    logger.warning("PlantCV not available. Morphological features will be limited.")

logger = logging.getLogger(__name__)


class MorphologyExtractor:
    """Extracts morphological features from plant images."""
    
    def __init__(self, pixel_to_cm: float = 0.1099609375, prune_sizes: List[int] = None):
        """
        Initialize morphology extractor.
        
        Args:
            pixel_to_cm: Conversion factor from pixels to centimeters
            prune_sizes: List of pruning sizes for skeleton processing
        """
        self.pixel_to_cm = pixel_to_cm
        self.prune_sizes = prune_sizes or [200, 100, 50, 30, 10]
        
        if PLANT_CV_AVAILABLE:
            # Configure PlantCV
            pcv.params.debug = None
            pcv.params.text_size = 0.7
            pcv.params.text_thickness = 2
            pcv.params.line_thickness = 3
            pcv.params.dpi = 100
    
    def extract_morphology_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract morphological features from plant image and mask.
        
        Args:
            image: Plant image (BGR format)
            mask: Binary mask of the plant
            
        Returns:
            Dictionary containing morphological features and images
        """
        features = {
            'traits': {},
            'images': {},
            'success': False
        }
        
        try:
            # Preprocess mask
            clean_mask = self._preprocess_mask(mask)
            if clean_mask is None:
                logger.warning("Failed to preprocess mask")
                return features
            
            # Extract basic morphological features
            basic_traits = self._extract_basic_features(clean_mask)
            features['traits'].update(basic_traits)
            
            # Extract skeleton-based features if PlantCV is available
            if PLANT_CV_AVAILABLE:
                skeleton_features = self._extract_skeleton_features(image, clean_mask)
                features['traits'].update(skeleton_features['traits'])
                features['images'].update(skeleton_features['images'])
            else:
                # Fallback to basic OpenCV features
                cv_features = self._extract_opencv_features(image, clean_mask)
                features['traits'].update(cv_features['traits'])
                features['images'].update(cv_features['images'])
            
            features['success'] = True
            logger.debug("Morphological features extracted successfully")
            
        except Exception as e:
            logger.error(f"Morphological feature extraction failed: {e}")
        
        return features
    
    def _preprocess_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess mask for morphological analysis."""
        if mask is None:
            return None
        
        # Convert to binary if needed
        if isinstance(mask, tuple):
            mask = mask[0]
        
        # Ensure binary format
        mask = ((mask.astype(np.int32) > 0).astype(np.uint8)) * 255
        
        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        clean_mask = np.zeros_like(opened)
        
        for label in range(1, num_labels):  # Skip background
            if stats[label, cv2.CC_STAT_AREA] >= 1000:
                clean_mask[labels == label] = 255
        
        return clean_mask
    
    def _extract_basic_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Extract basic morphological features using OpenCV."""
        features = {}
        
        try:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return features
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Basic measurements
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox_area = w * h
            
            # Ellipse fitting
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                (center, axes, angle) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
            else:
                major_axis = max(w, h)
                minor_axis = min(w, h)
            
            # Convert to centimeters
            features['area_cm2'] = area * (self.pixel_to_cm ** 2)
            features['perimeter_cm'] = perimeter * self.pixel_to_cm
            features['width_cm'] = w * self.pixel_to_cm
            features['height_cm'] = h * self.pixel_to_cm
            features['bbox_area_cm2'] = bbox_area * (self.pixel_to_cm ** 2)
            features['major_axis_cm'] = major_axis * self.pixel_to_cm
            features['minor_axis_cm'] = minor_axis * self.pixel_to_cm
            features['aspect_ratio'] = w / h if h > 0 else 0
            features['elongation'] = major_axis / minor_axis if minor_axis > 0 else 0
            features['circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            features['solidity'] = area / bbox_area if bbox_area > 0 else 0
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['convexity'] = area / hull_area if hull_area > 0 else 0
            
        except Exception as e:
            logger.error(f"Basic feature extraction failed: {e}")
        
        return features
    
    def _extract_skeleton_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Extract skeleton-based features using PlantCV."""
        features = {'traits': {}, 'images': {}}
        
        if not PLANT_CV_AVAILABLE:
            return features
        
        try:
            # Suppress PlantCV output
            with contextlib.redirect_stdout(self._FilteredStream(sys.stdout)), \
                 contextlib.redirect_stderr(self._FilteredStream(sys.stderr)):
                
                # Skeletonize - create complete skeleton without pruning
                # PlantCV skeletonize normally prunes by default, but there's no direct parameter in pcv.morphology.skeletonize to disable it easily 
                # unless we are using an older version or specific parameters. 
                # Actually, pcv.morphology.skeletonize just returns the skeleton. Pruning is usually a separate step `pcv.morphology.prune`.
                # So `skeleton` here IS the unpruned skeleton.
                skeleton = pcv.morphology.skeletonize(mask=mask)
                features['images']['skeleton'] = skeleton
                
                # Use complete skeleton (no pruning) for all operations
                # Set pruned_skeleton to the same as skeleton to maintain compatibility
                pruned_skel = skeleton.copy()
                features['images']['pruned_skeleton'] = pruned_skel
                
                # Find branch points and tips using complete skeleton
                # IMPORTANT: PlantCV's find_branch_pts and find_tips might expect a skeleton.
                branch_pts = pcv.morphology.find_branch_pts(skeleton, mask)
                features['images']['branch_points'] = branch_pts
                
                try:
                    tip_pts = pcv.morphology.find_tips(skeleton, mask)
                    features['images']['tip_points'] = tip_pts
                except Exception as e:
                    logger.warning(f"Tip detection failed: {e}")
                
                # Segment objects using complete skeleton
                try:
                    leaf_obj, stem_obj = pcv.morphology.segment_sort(
                        skeleton, [], mask
                    )
                    features['traits']['num_leaves'] = len(leaf_obj)
                    features['traits']['num_stems'] = len(stem_obj)
                except Exception as e:
                    logger.warning(f"Object segmentation failed: {e}")
                    features['traits']['num_leaves'] = 0
                    features['traits']['num_stems'] = 0
                
                # Size analysis - use the mask directly, not the skeleton
                # Size analysis should analyze the mask, not the skeleton
                try:
                    # Create labeled mask from the binary mask
                    labeled_mask, n_labels = pcv.create_labels(mask)
                    # Size analysis uses the original image and labeled mask
                    # This analyzes the actual plant size, not the skeleton
                    size_analysis = pcv.analyze.size(image, labeled_mask, n_labels, label="default")
                    features['images']['size_analysis'] = size_analysis
                    
                    # Get size traits
                    obs = pcv.outputs.observations.get("default_1", {})
                    for trait, info in obs.items():
                        if trait not in ["in_bounds", "object_in_frame"]:
                            val = info.get("value", None)
                            if val is not None:
                                if trait == "area":
                                    val = val * (self.pixel_to_cm ** 2)
                                elif trait in ["perimeter", "width", "height", "longest_path", 
                                            "ellipse_major_axis", "ellipse_minor_axis"]:
                                    val = val * self.pixel_to_cm
                                features['traits'][trait] = val
                
                except Exception as e:
                    logger.warning(f"Size analysis failed: {e}")
        
        except Exception as e:
            logger.error(f"Skeleton feature extraction failed: {e}")
        
        return features
    
    def _extract_opencv_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Extract features using only OpenCV (fallback when PlantCV is not available)."""
        features = {'traits': {}, 'images': {}}
        
        try:
            # Create skeleton using OpenCV
            skeleton = self._create_skeleton_opencv(mask)
            features['images']['skeleton'] = skeleton
            
            # Find branch points
            branch_points = self._find_branch_points_opencv(skeleton)
            features['images']['branch_points'] = branch_points
            features['traits']['num_branches'] = len(branch_points)
            
            # Find endpoints
            endpoints = self._find_endpoints_opencv(skeleton)
            features['images']['endpoints'] = endpoints
            features['traits']['num_endpoints'] = len(endpoints)
            
            # Skeleton length
            skeleton_length = np.sum(skeleton > 0)
            features['traits']['skeleton_length_pixels'] = skeleton_length
            features['traits']['skeleton_length_cm'] = skeleton_length * self.pixel_to_cm
            
        except Exception as e:
            logger.error(f"OpenCV feature extraction failed: {e}")
        
        return features
    
    def _create_skeleton_opencv(self, mask: np.ndarray) -> np.ndarray:
        """Create skeleton using OpenCV."""
        # Convert to binary
        binary = (mask > 0).astype(np.uint8)
        
        # Create skeleton using morphological operations
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()
            
            if cv2.countNonZero(binary) == 0:
                break
        
        return skeleton * 255
    
    def _find_branch_points_opencv(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find branch points in skeleton using OpenCV."""
        # Count neighbors for each pixel
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0  # Don't count center pixel
        
        neighbor_count = cv2.filter2D(skeleton, -1, kernel)
        
        # Branch points have 3 or more neighbors
        branch_points = np.where((skeleton > 0) & (neighbor_count >= 3))
        return list(zip(branch_points[1], branch_points[0]))  # (x, y) format
    
    def _find_endpoints_opencv(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints in skeleton using OpenCV."""
        # Count neighbors for each pixel
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0  # Don't count center pixel
        
        neighbor_count = cv2.filter2D(skeleton, -1, kernel)
        
        # Endpoints have exactly 1 neighbor
        endpoints = np.where((skeleton > 0) & (neighbor_count == 1))
        return list(zip(endpoints[1], endpoints[0]))  # (x, y) format
    
    class _FilteredStream:
        """Filter PlantCV output to reduce noise."""
        def __init__(self, stream):
            self.stream = stream
        
        def write(self, msg):
            skip = ("got pruned", "Slope of contour", "cannot be plotted")
            if not any(s in msg for s in skip):
                self.stream.write(msg)
        
        def flush(self):
            try:
                self.stream.flush()
            except Exception:
                pass
    
    def create_morphology_visualization(self, image: np.ndarray, mask: np.ndarray, 
                                      features: Dict[str, Any]) -> np.ndarray:
        """
        Create visualization of morphological features.
        
        Args:
            image: Original image
            mask: Binary mask
            features: Extracted features
            
        Returns:
            Visualization image
        """
        try:
            # Create visualization
            vis = image.copy()
            
            # Draw mask outline
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
            
            # Draw bounding box
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw skeleton if available
            if 'skeleton' in features.get('images', {}):
                skeleton = features['images']['skeleton']
                vis[skeleton > 0] = [0, 0, 255]  # Red skeleton
            
            # Draw branch points if available
            if 'branch_points' in features.get('images', {}):
                branch_img = features['images']['branch_points']
                vis[branch_img > 0] = [255, 255, 0]  # Yellow branch points
            
            return vis
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return image
