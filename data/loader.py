"""
Data loading functionality for the Sorghum Pipeline.

This module handles loading raw images, managing plant data,
and organizing data according to the pipeline requirements.
"""

import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and organizing plant image data."""
    
    # Plants to ignore completely (empty by default)
    IGNORE_PLANTS = set()
    
    # Plants where you want exactly one frame from their own folder
    EXACT_FRAME = {
        4: 7, 5: 5, 7: 5, 12: 5, 13: 5, 18: 7, 19: 2, 20: 3,
        24: 6, 25: 5, 26: 5, 30: 8, 37: 7
    }
    
    # Plants where you want to borrow a frame from a different plant folder
    BORROW_FRAME = {
        14: (13, 5), 15: (14, 5), 16: (15, 5), 33: (34, 7),
        34: (35, 7), 35: (35, 8), 36: (36, 6)
    }
    
    # Overrides provided by user: preferred frame per target plant name
    FRAME_OVERRIDE_BY_NAME = {
        'plant1': 9, 'plant2': 10, 'plant3': 9, 'plant5': 7, 'plant6': 9, 'plant8': 5,
        'plant7': 9, 'plant10': 9, 'plant11': 9, 'plant12': 9,
        'plant13': 10, 'plant14': 8, 'plant15': 11, 'plant19': 4, 'plant20': 7,
        'plant21': 9, 'plant22': 10, 'plant25': 4, 'plant26': 2, 'plant27': 10, 'plant28': 9, 'plant29': 2,
        'plant30': 9, 'plant31': 10, 'plant32': 9, 'plant33': 8,
        'plant35': 9, 'plant36': 4, 'plant38': 9, 'plant39': 9, 'plant41': 9,
        'plant42': 6, 'plant43': 10, 'plant44': 9, 'plant45': 7,
        'plant47': 10, 'plant48': 11,
    }
    
    # Substitutes provided by user: map target plant name -> source plant name
    PLANT_SUBSTITUTES_BY_NAME = {
        'plant16': 'plant15', 'plant15': 'plant14', 'plant14': 'plant13',
        'plant13': 'plant12', 'plant33': 'plant34', 'plant34': 'plant35',
        'plant24': 'plant25', 'plant25': 'plant25', 'plant35': 'plant36',
        'plant36': 'plant37', 'plant37': 'plant37', 'plant44': 'plant43',
        'plant45': 'plant44',
    }
    
    def __init__(self, input_folder: str, debug: bool = False, include_ignored: bool = False, strict_loader: bool = False, excluded_dates: Optional[List[str]] = None):
        """
        Initialize the data loader.
        
        Args:
            input_folder: Path to the input dataset folder
            debug: Enable debug logging
        """
        self.input_folder = Path(input_folder)
        self.debug = debug
        self.include_ignored = include_ignored
        self.strict_loader = strict_loader
        
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        # Normalize excluded dates as a set of folder names (with dashes)
        self.excluded_dates = set(excluded_dates or [])
    
    def load_selected_frames(self) -> Dict[str, Dict[str, Any]]:
        """
        Load selected frames according to predefined rules.
        If strict_loader is True, load only frame numbers from the plant's own folder (no borrowing/special picks).
        
        Returns:
            Dictionary with plant data organized by key format: "YYYY_MM_DD_plantX_frameY"
        """
        logger.info("Loading selected frames from dataset...")
        plants = {}

        # Detect if input folder is a direct date folder (contains plant folders)
        first_items = list(self.input_folder.iterdir())
        has_plant_folders = any(item.is_dir() and item.name.startswith('plant') for item in first_items)

        def choose_frame_and_source(pid: int) -> Tuple[int, str]:
            if self.strict_loader:
                # In strict mode, honor explicit frame overrides AND substitution of source plant
                plant_name_local = f"plant{pid}"
                frame_num = self.FRAME_OVERRIDE_BY_NAME.get(
                    plant_name_local,
                    self.EXACT_FRAME.get(pid, 8)
                )
                source_plant = self.PLANT_SUBSTITUTES_BY_NAME.get(plant_name_local, plant_name_local)
                return frame_num, source_plant
            # Original behavior
            frame_num = self._get_frame_number(pid)
            source_plant = self._get_source_plant(pid)
            return frame_num, source_plant

        if has_plant_folders:
            # Direct date folder structure
            date_name = self.input_folder.name
            date_path = self.input_folder
            for plant_name in sorted(os.listdir(date_path)):
                plant_path = date_path / plant_name
                if not plant_path.is_dir():
                    continue
                try:
                    plant_id = int(plant_name.replace("plant", ""))
                except ValueError:
                    continue
                if (plant_id in self.IGNORE_PLANTS) and (not self.include_ignored):
                    if self.debug:
                        logger.debug(f"Ignoring plant {plant_id}")
                    continue
                frame_num, source_plant = choose_frame_and_source(plant_id)
                frame_data = self._load_single_frame(date_path, source_plant, frame_num, plant_name)
                if frame_data:
                    key = f"{date_name.replace('-', '_')}_{plant_name}_frame{frame_num}"
                    plants[key] = frame_data
                    logger.debug(f"Loaded {key}")
        else:
            # Parent folder structure with date subfolders
            for date_name in sorted(os.listdir(self.input_folder)):
                date_path = self.input_folder / date_name
                if not date_path.is_dir():
                    continue
                if date_name in self.excluded_dates:
                    logger.info(f"Skipping excluded date: {date_name}")
                    continue
                for plant_name in sorted(os.listdir(date_path)):
                    plant_path = date_path / plant_name
                    if not plant_path.is_dir():
                        continue
                    try:
                        plant_id = int(plant_name.replace("plant", ""))
                    except ValueError:
                        continue
                    if (plant_id in self.IGNORE_PLANTS) and (not self.include_ignored):
                        if self.debug:
                            logger.debug(f"Ignoring plant {plant_id}")
                        continue
                    frame_num, source_plant = choose_frame_and_source(plant_id)
                    frame_data = self._load_single_frame(date_path, source_plant, frame_num, plant_name)
                    if frame_data:
                        key = f"{date_name.replace('-', '_')}_{plant_name}_frame{frame_num}"
                        plants[key] = frame_data
                        logger.debug(f"Loaded {key}")

        logger.info(f"Successfully loaded {len(plants)} plant frames")
        return plants
    
    def load_all_frames(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all available frames for each plant.
        
        Returns:
            Dictionary with all plant frames
        """
        logger.info("Loading all frames from dataset...")
        plants = {}
        
        # Check if we're directly in a date folder (contains plant folders)
        # or in a parent folder (contains date folders)
        first_items = list(self.input_folder.iterdir())
        has_plant_folders = any(item.is_dir() and item.name.startswith('plant') for item in first_items)
        
        if has_plant_folders:
            # We're directly in a date folder
            logger.info("Detected direct date folder structure")
            date_name = self.input_folder.name
            self._load_plants_from_date_folder(self.input_folder, date_name, plants)
        else:
            # We're in a parent folder with date subfolders
            logger.info("Detected parent folder structure")
            for date_name in sorted(os.listdir(self.input_folder)):
                date_path = self.input_folder / date_name
                if not date_path.is_dir():
                    continue
                if date_name in self.excluded_dates:
                    logger.info(f"Skipping excluded date: {date_name}")
                    continue
                
                logger.info(f"Processing date: {date_name}")
                self._load_plants_from_date_folder(date_path, date_name, plants)
        
        logger.info(f"Successfully loaded {len(plants)} plant frames")
        return plants
    
    def _load_plants_from_date_folder(self, date_path: Path, date_name: str, plants: Dict[str, Dict[str, Any]]) -> None:
        """Load plants from a date folder."""
        for plant_name in sorted(os.listdir(date_path)):
            plant_path = date_path / plant_name
            if not plant_path.is_dir():
                continue
            
            # Extract plant ID
            try:
                plant_id = int(plant_name.replace("plant", ""))
            except ValueError:
                logger.warning(f"Could not extract plant ID from {plant_name}")
                continue
            
            # Skip ignored plants
            if (plant_id in self.IGNORE_PLANTS) and (not self.include_ignored):
                logger.info(f"Skipping ignored plant {plant_id}")
                continue
            
            logger.info(f"Processing plant {plant_id}")
            
            # Load all frames for this plant
            pattern = str(plant_path / f"{plant_name}_frame*.tif")
            frame_files = sorted(glob.glob(pattern))
            logger.info(f"Found {len(frame_files)} frame files for {plant_name}")
            
            for frame_path in frame_files:
                frame_data = self._load_frame_from_path(frame_path, plant_name)
                if frame_data:
                    frame_id = Path(frame_path).stem.split("_frame")[-1]
                    key = f"{date_name.replace('-', '_')}_{plant_name}_frame{frame_id}"
                    plants[key] = frame_data
                    logger.debug(f"Loaded frame: {key}")
                else:
                    logger.warning(f"Failed to load frame: {frame_path}")
    
    def load_single_plant(self, date: str, plant: str, frame: int) -> Optional[Dict[str, Any]]:
        """
        Load a specific plant frame.
        
        Args:
            date: Date string (e.g., "2025-02-05")
            plant: Plant name (e.g., "plant1")
            frame: Frame number
            
        Returns:
            Plant data dictionary or None if not found
        """
        date_path = self.input_folder / date
        if not date_path.exists():
            logger.error(f"Date folder not found: {date}")
            return None
        
        plant_path = date_path / plant
        if not plant_path.exists():
            logger.error(f"Plant folder not found: {plant}")
            return None
        
        filename = f"{plant}_frame{frame}.tif"
        frame_path = plant_path / filename
        
        return self._load_frame_from_path(str(frame_path), plant)
    
    def _get_frame_number(self, plant_id: int) -> int:
        """Get the frame number for a plant ID."""
        plant_name = f"plant{plant_id}"
        # Highest priority: explicit overrides by plant name
        if plant_name in self.FRAME_OVERRIDE_BY_NAME:
            return int(self.FRAME_OVERRIDE_BY_NAME[plant_name])
        # Next: original exact/borrrow rules
        if plant_id in self.EXACT_FRAME:
            return self.EXACT_FRAME[plant_id]
        elif plant_id in self.BORROW_FRAME:
            return self.BORROW_FRAME[plant_id][1]
        else:
            return 8  # Default frame
    
    def _get_source_plant(self, plant_id: int) -> str:
        """Get the source plant name for a plant ID."""
        plant_name = f"plant{plant_id}"
        # Highest priority: explicit substitutes by plant name
        if plant_name in self.PLANT_SUBSTITUTES_BY_NAME:
            return self.PLANT_SUBSTITUTES_BY_NAME[plant_name]
        # Next: original borrow rules
        if plant_id in self.BORROW_FRAME:
            source_id = self.BORROW_FRAME[plant_id][0]
            return f"plant{source_id}"
        else:
            return f"plant{plant_id}"
    
    def _load_single_frame(self, date_path: Path, source_plant: str, 
                          frame_num: int, target_plant: str) -> Optional[Dict[str, Any]]:
        """Load a single frame from the specified path."""
        filename = f"{source_plant}_frame{frame_num}.tif"
        frame_path = date_path / source_plant / filename
        
        if not frame_path.exists():
            if self.debug:
                logger.warning(f"Frame not found: {frame_path}")
            return None
        
        return self._load_frame_from_path(str(frame_path), target_plant)
    
    def _load_frame_from_path(self, frame_path: str, plant_name: str) -> Optional[Dict[str, Any]]:
        """Load frame data from a file path."""
        try:
            logger.debug(f"Attempting to load: {frame_path}")
            image = Image.open(frame_path)
            
            # Force grayscale/single band mode for TIF files that should be single band
            # This is important for Dr. Mullet's dataset where images are 1024x1024 single band
            # that need to be split into 4 tiles of 512x512 for 4-band processing
            if image.mode in ('RGB', 'RGBA', 'P') and frame_path.lower().endswith('.tif'):
                # Check if it's actually a single band image by checking array shape
                import numpy as np
                test_arr = np.array(image)
                # If all channels are identical, it's likely a single band image saved as RGB
                if len(test_arr.shape) == 3 and test_arr.shape[2] == 3:
                    # Check if R, G, B channels are identical (single band saved as RGB)
                    if np.allclose(test_arr[:, :, 0], test_arr[:, :, 1]) and np.allclose(test_arr[:, :, 1], test_arr[:, :, 2]):
                        image = image.convert('L')  # Convert to grayscale
                        logger.debug(f"Converted RGB TIF to grayscale (single band): {frame_path}")
                # For single band TIF files, always convert to grayscale
                elif image.mode == 'P':  # Palette mode, often used for single band
                    image = image.convert('L')
                    logger.debug(f"Converted palette TIF to grayscale: {frame_path}")
            
            filename = Path(frame_path).name
            logger.debug(f"Successfully loaded image: {filename}, size: {image.size}, mode: {image.mode}")
            
            return {
                "raw_image": (image, filename),
                "plant_name": plant_name,
                "file_path": frame_path
            }
        except Exception as e:
            logger.error(f"Failed to load {frame_path}: {e}")
            return None
    
    def load_bounding_boxes(self, bbox_dir: str) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Load bounding box data from JSON files.
        
        Args:
            bbox_dir: Directory containing bounding box JSON files
            
        Returns:
            Dictionary mapping plant names to bounding box coordinates
        """
        bbox_path = Path(bbox_dir)
        if not bbox_path.exists():
            raise FileNotFoundError(f"Bounding box directory not found: {bbox_dir}")
        
        bbox_lookup = {}
        
        for json_file in bbox_path.glob("*.json"):
            stem = json_file.stem
            # Normalize stems like plant_33_new -> plant33
            if stem.startswith('plant_'):
                parts = stem.split('_')
                try:
                    idx = next(i for i,p in enumerate(parts) if p.isdigit())
                    plant_id = f"plant{parts[idx]}"
                except Exception:
                    plant_id = stem.replace('_', '')
            else:
                plant_id = stem
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                shapes = data.get('shapes', [])
                # Prefer rectangle labeled 'sorghum' (case-insensitive), else first rectangle
                def _is_sorghum_label(s: dict) -> bool:
                    for key in ('label', 'name', 'text'):
                        val = s.get(key)
                        if isinstance(val, str) and val.lower() == 'sorghum':
                            return True
                    return False
                rect = next((s for s in shapes if s.get('shape_type') == 'rectangle' and _is_sorghum_label(s)), None)
                if rect is None:
                    rect = next((s for s in shapes if s.get('shape_type') == 'rectangle'), None)
                
                if rect:
                    (x1, y1), (x2, y2) = rect['points']
                    bbox_lookup[plant_id] = (
                        int(max(0, x1)),
                        int(max(0, y1)),
                        int(min(1e9, x2)),
                        int(min(1e9, y2))
                    )
                else:
                    bbox_lookup[plant_id] = None
                    
            except Exception as e:
                logger.error(f"Failed to load bounding box {json_file}: {e}")
        
        logger.info(f"Loaded {len(bbox_lookup)} bounding boxes")
        return bbox_lookup
    
    def load_hand_labels(self, labels_dir: str) -> Dict[str, np.ndarray]:
        """
        Load hand-labeled masks from JSON files.
        
        Args:
            labels_dir: Directory containing label JSON files
            
        Returns:
            Dictionary mapping plant names to mask arrays
        """
        labels_path = Path(labels_dir)
        if not labels_path.exists():
            logger.warning(f"Labels directory not found: {labels_dir}")
            return {}
        
        masks = {}
        
        for json_file in labels_path.glob("*.json"):
            plant_id = json_file.stem
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Create mask from shapes (assuming we have image dimensions)
                # This would need to be adapted based on your label format
                mask = self._create_mask_from_shapes(data)
                if mask is not None:
                    masks[plant_id] = mask
                    
            except Exception as e:
                logger.error(f"Failed to load label {json_file}: {e}")
        
        logger.info(f"Loaded {len(masks)} hand labels")
        return masks
    
    def _create_mask_from_shapes(self, data: Dict) -> Optional[np.ndarray]:
        """Create a mask array from shape data."""
        # This is a placeholder - implement based on your label format
        # For now, return None
        return None
    
    def validate_data(self, plants: Dict[str, Dict[str, Any]]) -> bool:
        """
        Validate loaded plant data.
        
        Args:
            plants: Dictionary of plant data
            
        Returns:
            True if data is valid, False otherwise
        """
        if not plants:
            logger.error("No plant data loaded")
            return False
        
        for key, data in plants.items():
            if "raw_image" not in data:
                logger.error(f"Missing raw_image in {key}")
                return False
            
            image, filename = data["raw_image"]
            if not isinstance(image, Image.Image):
                logger.error(f"Invalid image type in {key}")
                return False
        
        logger.info("Data validation passed")
        return True
