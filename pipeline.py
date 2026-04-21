"""
Main pipeline class for the Sorghum Plant Phenotyping Pipeline.

This module orchestrates the entire pipeline from data loading
to feature extraction and result output.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import numpy as np
import cv2
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from sklearn.decomposition import PCA
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from .config import Config
from .data import DataLoader, ImagePreprocessor, MaskHandler
from .features import TextureExtractor, VegetationIndexExtractor, MorphologyExtractor
from .output import OutputManager
from .segmentation import SegmentationManager
try:
    from .detection import YOLODetector
    YOLO_DETECTOR_AVAILABLE = True
except ImportError:
    YOLODetector = None
    YOLO_DETECTOR_AVAILABLE = False
# Make occlusion handling optional if the module is not present
try:
    from .segmentation.occlusion_handler import OcclusionHandler  # type: ignore
except Exception:
    OcclusionHandler = None  # type: ignore


class SorghumPipeline:
    """
    Main pipeline class for sorghum plant phenotyping.
    
    This class orchestrates the entire pipeline from data loading
    to feature extraction and result output.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Config] = None, include_ignored: bool = False, enable_occlusion_handling: bool = False, enable_instance_integration: bool = False, strict_loader: bool = False, excluded_dates: Optional[List[str]] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
            config: Configuration object (if not using file)
            include_ignored: Whether to include ignored plants
            enable_occlusion_handling: Whether to enable SAM2Long occlusion handling
        """
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = Config(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Validate configuration
        self.config.validate()
        
        # Store settings
        self.enable_occlusion_handling = enable_occlusion_handling
        self.enable_instance_integration = enable_instance_integration
        self.strict_loader = strict_loader
        self.excluded_dates = excluded_dates or []
        self.enable_yolo_detection = getattr(self.config.model, 'enable_yolo_detection', False)
        self.yolo_model_path = getattr(self.config.model, 'yolo_model_path', None)
        
        # Initialize components
        self._initialize_components(include_ignored)
        
        logger.info("Sorghum Pipeline initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('sorghum_pipeline.log')
            ]
        )
        global logger
        logger = logging.getLogger(__name__)
    
    def _initialize_components(self, include_ignored: bool = False):
        """Initialize all pipeline components."""
        # Data components
        self.data_loader = DataLoader(
            input_folder=self.config.paths.input_folder,
            debug=True,
            include_ignored=include_ignored,
            strict_loader=self.strict_loader,
            excluded_dates=self.excluded_dates,
        )
        self.preprocessor = ImagePreprocessor(
            target_size=self.config.processing.target_size
        )
        self.mask_handler = MaskHandler(
            min_area=self.config.processing.min_component_area,
            kernel_size=self.config.processing.morphology_kernel_size
        )
        
        # Feature extractors
        self.texture_extractor = TextureExtractor(
            lbp_points=self.config.processing.lbp_points,
            lbp_radius=self.config.processing.lbp_radius,
            hog_orientations=self.config.processing.hog_orientations,
            hog_pixels_per_cell=self.config.processing.hog_pixels_per_cell,
            hog_cells_per_block=self.config.processing.hog_cells_per_block,
            lacunarity_window=self.config.processing.lacunarity_window,
            ehd_threshold=self.config.processing.ehd_threshold,
            angle_resolution=self.config.processing.angle_resolution
        )
        
        self.vegetation_extractor = VegetationIndexExtractor(
            epsilon=self.config.processing.epsilon,
            soil_factor=self.config.processing.soil_factor
        )
        
        self.morphology_extractor = MorphologyExtractor(
            pixel_to_cm=self.config.processing.pixel_to_cm,
            prune_sizes=self.config.processing.prune_sizes
        )
        
        # YOLO Detection (optional, before segmentation)
        self.yolo_detector = None
        if YOLO_DETECTOR_AVAILABLE and self.enable_yolo_detection:
            try:
                self.yolo_detector = YOLODetector(
                    model_path=self.yolo_model_path,
                    model_name=getattr(self.config.model, 'yolo_model_name', 'yolov12n.pt'),
                    device=self.config.get_device(),
                    conf_threshold=getattr(self.config.model, 'yolo_conf_threshold', 0.25),
                    iou_threshold=getattr(self.config.model, 'yolo_iou_threshold', 0.45)
                )
                logger.info("YOLO detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize YOLO detector: {e}")
                self.yolo_detector = None
        
        # Segmentation
        self.segmentation_manager = SegmentationManager(
            model_name=self.config.model.model_name,
            device=self.config.get_device(),
            threshold=self.config.processing.segmentation_threshold,
            trust_remote_code=self.config.model.trust_remote_code,
            cache_dir=self.config.model.cache_dir if getattr(self.config.model, 'cache_dir', '') else None,
            local_files_only=getattr(self.config.model, 'local_files_only', False),
        )
        
        # Occlusion handling (optional)
        self.occlusion_handler = None
        if self.enable_occlusion_handling and OcclusionHandler is not None:
            try:
                self.occlusion_handler = OcclusionHandler(
                    device=self.config.get_device(),
                    model="tiny",  # Can be made configurable
                    confidence_threshold=0.5,
                    iou_threshold=0.1
                )
                logger.info("Occlusion handler initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize occlusion handler: {e}")
                logger.warning("Continuing without occlusion handling")
                self.occlusion_handler = None
        elif self.enable_occlusion_handling and OcclusionHandler is None:
            logger.warning("Occlusion handler module not found; continuing without occlusion handling")
        
        # Output manager
        self.output_manager = OutputManager(
            output_folder=self.config.paths.output_folder,
            settings=self.config.output
        )
        # Pass Dr. Mullet experiment flag to output manager
        self.output_manager.is_dr_mullet_experiment = getattr(self.config.model, 'is_dr_mullet_experiment', False)

    def _free_gpu_memory_before_instance(self) -> None:
        """Attempt to free GPU memory prior to running SAM2Long in a subprocess.

        - Moves BRIA segmentation model to CPU if present
        - Deletes the model reference to release VRAM
        - Calls torch.cuda.empty_cache()
        """
        try:
            import torch as _torch  # type: ignore
            # Move BRIA model to CPU and drop reference
            try:
                if getattr(self, 'segmentation_manager', None) is not None:
                    mdl = getattr(self.segmentation_manager, 'model', None)
                    if mdl is not None:
                        try:
                            mdl.to('cpu')
                        except Exception:
                            pass
                        try:
                            delattr(self.segmentation_manager, 'model')
                        except Exception:
                            pass
                        # Ensure attribute exists but is None for future checks
                        try:
                            self.segmentation_manager.model = None  # type: ignore
                        except Exception:
                            pass
            except Exception:
                pass
            # Free CUDA cache
            try:
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Freed GPU memory before SAM2Long invocation (moved BRIA to CPU and emptied cache)")
        except Exception as e:
            logger.warning(f"Failed to free GPU memory before instance segmentation: {e}")
    
    def run(self, load_all_frames: bool = False, segmentation_only: bool = False, filter_plants: Optional[List[str]] = None, filter_frames: Optional[List[str]] = None, run_instance_segmentation: bool = False, features_frame_only: Optional[int] = None, reuse_instance_results: bool = False, instance_mapping_path: Optional[str] = None, force_reprocess: bool = False, respect_instance_frame_rules_for_features: bool = False, substitute_feature_image_from_instance_src: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            load_all_frames: Whether to load all frames or selected frames
            segmentation_only: If True, run segmentation only and skip feature extraction
            
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting Sorghum Pipeline...")
        
        try:
            import time
            total_start = time.perf_counter()
            # Step 1: Load data
            logger.info("Step 1/6: Loading data...")
            # In reuse mode we need all frames to select the mapped frame per plant
            if reuse_instance_results:
                plants = self.data_loader.load_all_frames()
            else:
                # If specific frames are requested, we must load all frames to filter correctly
                if load_all_frames or (filter_frames is not None and len(filter_frames) > 0):
                    plants = self.data_loader.load_all_frames()
                else:
                    plants = self.data_loader.load_selected_frames()
            
            # Optional filter by specific plant names (e.g., ["plant1"]) 
            if filter_plants:
                allowed = set(filter_plants)
                plants = {
                    key: pdata for key, pdata in plants.items()
                    if len(key.split('_')) > 3 and key.split('_')[3] in allowed
                }

            # Optional filter by specific frame numbers (e.g., ["9"] or ["frame9"]) 
            if filter_frames:
                # Normalize to 'frameX' tokens
                wanted = set(
                    [f if str(f).startswith('frame') else f"frame{str(f)}" for f in filter_frames]
                )
                plants = {
                    key: pdata for key, pdata in plants.items()
                    if key.split('_')[-1] in wanted
                }

            if not plants:
                raise ValueError("No plant data loaded")
            
            logger.info(f"Loaded {len(plants)} plants")

            # If reusing instance results with mapping, restrict to exactly the mapped frame per plant (default frame8)
            if reuse_instance_results:
                try:
                    import json as _json
                    if instance_mapping_path is None:
                        raise ValueError("instance_mapping_path is required in reuse mode")
                    _map = _json.load(open(instance_mapping_path, 'r'))
                    # Normalize mapping plant keys and compute target frame (default 8)
                    target_frame_by_plant = {}
                    for pk, pv in _map.items():
                        k_norm = pk if str(pk).startswith('plant') else f"plant{int(pk)}" if str(pk).isdigit() else str(pk)
                        try:
                            target_frame_by_plant[k_norm] = int(pv.get('frame', 8))
                        except Exception:
                            target_frame_by_plant[k_norm] = 8
                    before = len(plants)
                    plants = {
                        key: pdata for key, pdata in plants.items()
                        if (len(key.split('_')) > 3 and key.split('_')[3] in target_frame_by_plant
                            and key.split('_')[-1] == f"frame{target_frame_by_plant[key.split('_')[3]]}")
                    }
                    logger.info(f"Restricted loaded data by mapping frames: {before} -> {len(plants)} items")
                except Exception as e:
                    logger.warning(f"Failed to restrict loaded data by mapping frames: {e}")

            # Skip plants that already have saved results (unless force_reprocess)
            if not force_reprocess:
                try:
                    before = len(plants)
                    filtered = {}
                    for key, pdata in plants.items():
                        parts = key.split('_')
                        if len(parts) < 5:
                            filtered[key] = pdata
                            continue
                        date_key = "_".join(parts[:3])
                        plant_name = parts[3]
                        plant_dir = Path(self.config.paths.output_folder) / date_key / plant_name
                        meta_ok = (plant_dir / 'metadata.json').exists()
                        seg_mask_ok = (plant_dir / self.config.output.segmentation_dir / 'mask.png').exists()
                        if meta_ok or seg_mask_ok:
                            continue
                        filtered[key] = pdata
                    plants = filtered
                    logger.info(f"Skip-existing filter: {before} -> {len(plants)} items to process")
                except Exception as e:
                    logger.warning(f"Skip-existing filter failed: {e}")
            
            # Pre-segmentation borrowing: use plant12 images for plant13 from the start
            try:
                rewired = 0
                borrow_map: Dict[str, str] = {
                    'plant13': 'plant12',
                    'plant14': 'plant13',
                    'plant15': 'plant14',
                    'plant16': 'plant15',
                }
                for _k in list(plants.keys()):
                    _parts = _k.split('_')
                    # Expect keys like YYYY_MM_DD_plantX_frameY
                    if len(_parts) < 5:
                        continue
                    _date_key = "_".join(_parts[:3])
                    _plant_name = _parts[3]
                    _frame_token = _parts[4]
                    # Do NOT borrow on 2025_05_08
                    if _date_key == '2025_05_08':
                        continue
                    if _plant_name not in borrow_map:
                        continue
                    _src_plant = borrow_map[_plant_name]
                    _src_key = f"{_date_key}_{_src_plant}_{_frame_token}"
                    _src = plants.get(_src_key)
                    if not _src:
                        # Fallback: load raw image for source plant directly from disk
                        try:
                            from PIL import Image as _Image
                            _date_folder = _date_key.replace('_', '-')
                            _frame_num = int(_frame_token.replace('frame', ''))
                            _date_dir = Path(self.config.paths.input_folder)
                            # If input folder is a parent of dates, append date folder
                            if _date_dir.name != _date_folder:
                                _date_dir = _date_dir / _date_folder
                            _frame_path = _date_dir / _src_plant / f"{_src_plant}_frame{_frame_num}.tif"
                            if _frame_path.exists():
                                _img = _Image.open(str(_frame_path))
                                _src = {"raw_image": (_img, _frame_path.name), "plant_name": _plant_name, "file_path": str(_frame_path)}
                            else:
                                _src = None
                        except Exception:
                            _src = None
                    if not _src:
                        continue
                    _tgt = plants[_k]
                    # Preserve original raw image once
                    if 'raw_image' in _tgt and 'raw_image_original' not in _tgt:
                        _tgt['raw_image_original'] = _tgt['raw_image']
                    if 'raw_image' in _src:
                        _tgt['raw_image'] = _src['raw_image']
                        _tgt['borrowed_from'] = _src_plant
                        rewired += 1
                if rewired > 0:
                    logger.info(f"Pre-seg borrowing applied: rewired {rewired} frames for plants 13/14/15/16")
            except Exception as e:
                logger.warning(f"Pre-seg borrowing failed: {e}")

            # Step 2: Create composites
            logger.info("Step 2/6: Creating composites...")
            step_start = time.perf_counter()
            plants = self.preprocessor.create_composites(plants)
            logger.info(f"Composites done in {(time.perf_counter()-step_start):.2f}s")
            
            # Step 2.5: YOLO Detection (before segmentation)
            if self.yolo_detector is not None:
                logger.info("Step 2.5/6: Running YOLO detection...")
                step_start = time.perf_counter()
                plants = self._run_yolo_detection(plants)
                logger.info(f"YOLO detection done in {(time.perf_counter()-step_start):.2f}s")
            
            # Step 3: Segment plants (optionally with bounding boxes from YOLO or file)
            logger.info("Step 3/6: Segmenting plants...")
            step_start = time.perf_counter()
            bbox_lookup = None
            try:
                bbox_dir = getattr(self.config.paths, 'boundingbox_dir', None)
                # Default to project BoundingBox dir if unset or falsy
                if not bbox_dir:
                    try:
                        self.config.paths.boundingbox_dir = "/home/grads/f/fahimehorvatinia/Documents/my_full_project/BoundingBox"
                        bbox_dir = self.config.paths.boundingbox_dir
                    except Exception:
                        bbox_dir = None
                if bbox_dir:
                    bbox_lookup = self.data_loader.load_bounding_boxes(bbox_dir)
                    logger.info(f"Loaded bounding boxes from {bbox_dir}")
            except Exception as e:
                logger.warning(f"Failed to load bounding boxes: {e}")
                bbox_lookup = None
            plants = self._segment_plants(plants, bbox_lookup)
            logger.info(f"Segmentation done in {(time.perf_counter()-step_start):.2f}s")
            
            # Step 3.5: Handle occlusion if enabled
            if self.enable_occlusion_handling and self.occlusion_handler is not None:
                logger.info("Step 3.5/6: Handling occlusion with SAM2Long...")
                step_start = time.perf_counter()
                plants = self._handle_occlusion(plants)
                logger.info(f"Occlusion handling done in {(time.perf_counter()-step_start):.2f}s")

            # Optional: Export RMBG maskouts with white background and run instance segmentation
            if (run_instance_segmentation or self.enable_instance_integration) and not reuse_instance_results:
                if not load_all_frames:
                    logger.warning("Instance segmentation expects all 13 frames; consider running with load_all_frames=True.")
                logger.info("Step 3.6: Exporting white-background RMBG images for instance segmentation...")
                # Derive date-specific export/result directories when a single date is present
                date_keys = set()
                try:
                    for _k in plants.keys():
                        _p = _k.split('_')
                        if len(_p) >= 3:
                            date_keys.add("_".join(_p[:3]))
                except Exception:
                    pass
                if len(date_keys) == 1:
                    date_key = next(iter(date_keys))
                    base_dir = Path(self.config.paths.output_folder) / date_key
                    export_dir = base_dir / "instance_input_maskouts"
                    instance_results_dir = base_dir / "instance_results"
                else:
                    export_dir = Path(self.config.paths.output_folder) / "instance_input_maskouts"
                    instance_results_dir = Path(self.config.paths.output_folder) / "instance_results"
                export_dir.mkdir(parents=True, exist_ok=True)
                instance_results_dir.mkdir(parents=True, exist_ok=True)
                self._export_white_background_maskouts(plants, export_dir)

                logger.info("Invoking final SAM2Long instance segmentation on exported images...")
                # Free GPU memory before launching SAM2Long to avoid CUDA OOM
                self._free_gpu_memory_before_instance()
                env = os.environ.copy()
                env["SAM2LONG_IMAGES_DIR"] = str(export_dir)
                env["SAM2LONG_RESULTS_DIR"] = str(instance_results_dir)
                # Ensure instance outputs include all frames for all dates
                try:
                    env.pop("INSTANCE_OUTPUT_FRAMES", None)
                except Exception:
                    pass
                script_path = "/home/grads/f/fahimehorvatinia/Documents/my_full_project/Experiments3_code/sam2long_instance_integration.py"
                try:
                    subprocess.run(["python", script_path], check=True, env=env)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Instance segmentation failed: {e}")
                else:
                    # Integrate instance masks (track_0 as target) into pdata before feature extraction
                    try:
                        self._apply_instance_masks(plants, instance_results_dir)
                        logger.info("Applied instance segmentation masks to pipeline data")
                    except Exception as e:
                        logger.warning(f"Failed to apply instance masks: {e}")
            elif reuse_instance_results:
                # Reuse existing instance masks from mapping file
                if instance_mapping_path is None:
                    raise ValueError("reuse_instance_results=True requires instance_mapping_path to be provided")
                try:
                    self._apply_instance_masks_from_mapping(plants, Path(instance_mapping_path))
                    logger.info("Applied instance masks from mapping file")
                except Exception as e:
                    logger.error(f"Failed to apply instance masks from mapping: {e}")
            
            if not segmentation_only:
                # If reusing instance results with a mapping, restrict features to mapped frames per plant
                if reuse_instance_results and instance_mapping_path is not None:
                    try:
                        import json as _json
                        _map = _json.load(open(instance_mapping_path, 'r'))
                        # Normalize map
                        _norm = {}
                        for pk, pv in _map.items():
                            k_norm = pk if str(pk).startswith('plant') else f"plant{int(pk)}" if str(pk).isdigit() else str(pk)
                            _norm[k_norm] = int(pv.get('frame', 8))
                        before = len(plants)
                        plants = {
                            k: v for k, v in plants.items()
                            if len(k.split('_')) > 3 and k.split('_')[3] in _norm and k.split('_')[-1] == f"frame{_norm[k.split('_')[3]]}"
                        }
                        logger.info(f"Restricted feature extraction by mapping: {before} -> {len(plants)} items")
                    except Exception as e:
                        logger.warning(f"Failed to restrict by mapping frames: {e}")
                # Optional: restrict features to per-plant preferred frame using internal frame rules
                if respect_instance_frame_rules_for_features:
                    try:
                        # Keep this in sync with _apply_instance_masks frame_rules
                        frame_rules: Dict[str, int] = {
                            "plant33": 2,
                            "plant16": 4,
                            "plant19": 5,
                            "plant26": 8,
                            "plant27": 8,
                            "plant29": 8,
                            "plant35": 7,
                            "plant36": 6,
                            "plant37": 2,
                            "plant45": 5,
                        }
                        before = len(plants)
                        def _keep(k: str) -> bool:
                            parts = k.split('_')
                            if len(parts) < 2:
                                return False
                            plant_name = parts[-2]
                            frame_token = parts[-1]
                            if not (plant_name.startswith('plant') and frame_token.startswith('frame')):
                                return False
                            desired = frame_rules.get(plant_name, 8)
                            return frame_token == f"frame{desired}"
                        plants = {k: v for k, v in plants.items() if _keep(k)}
                        logger.info(f"Restricted feature extraction by per-plant frame rules: {before} -> {len(plants)} items")
                    except Exception as e:
                        logger.warning(f"Failed to apply per-plant frame restriction for features: {e}")

                # Optional: if features_frame_only set, keep only that frame's entries (global single frame)
                if features_frame_only is not None:
                    frame_token = f"frame{features_frame_only}"
                    plants = {k: v for k, v in plants.items() if k.split('_')[-1] == frame_token}
                    logger.info(f"Restricted feature extraction to {len(plants)} items for {frame_token}")

                # Optional: substitute feature input image from instance src_rules mapping (e.g., plant14 <- plant13)
                if substitute_feature_image_from_instance_src:
                    try:
                        src_rules: Dict[str, str] = {
                            "plant13": "plant12",
                            "plant14": "plant13",
                            "plant15": "plant14",
                            "plant16": "plant15",
                        }
                        switched = 0
                        for key in list(plants.keys()):
                            parts = key.split('_')
                            if len(parts) < 5:
                                continue
                            date_key = "_".join(parts[:3])
                            plant_name = parts[3]
                            frame_token = parts[-1]
                            if plant_name not in src_rules:
                                continue
                            src_plant = src_rules[plant_name]
                            src_key = f"{date_key}_{src_plant}_{frame_token}"
                            if src_key not in plants:
                                continue
                            src_pdata = plants[src_key]
                            tgt_pdata = plants[key]
                            # Preserve the original composite used for segmentation for correct overlays later
                            try:
                                if 'composite' in tgt_pdata and 'segmentation_composite' not in tgt_pdata:
                                    tgt_pdata['segmentation_composite'] = tgt_pdata['composite']
                            except Exception:
                                pass
                            # Swap feature inputs: composite and spectral bands
                            if 'composite' in src_pdata:
                                tgt_pdata['composite'] = src_pdata['composite']
                            if 'spectral_stack' in src_pdata:
                                tgt_pdata['spectral_stack'] = src_pdata['spectral_stack']
                            # Ensure mask aligns with substituted composite; resize if needed
                            try:
                                import cv2 as _cv2
                                import numpy as _np
                                comp = tgt_pdata.get('composite')
                                msk = tgt_pdata.get('mask')
                                if comp is not None and msk is not None:
                                    ch, cw = comp.shape[:2]
                                    mh, mw = msk.shape[:2]
                                    if (mh, mw) != (ch, cw):
                                        resized = _cv2.resize(msk.astype('uint8'), (cw, ch), interpolation=_cv2.INTER_NEAREST)
                                        tgt_pdata['mask'] = resized
                                        if 'soft_mask' in tgt_pdata and isinstance(tgt_pdata['soft_mask'], _np.ndarray):
                                            tgt_pdata['soft_mask'] = (resized > 0).astype(_np.float32)
                                    # Precompute masked composite with white background for saving
                                    white = _np.full_like(comp, 255, dtype=_np.uint8)
                                    result = white.copy()
                                    result[tgt_pdata['mask'] > 0] = comp[tgt_pdata['mask'] > 0]
                                    tgt_pdata['masked_composite'] = result
                            except Exception:
                                pass
                            switched += 1
                        if switched > 0:
                            logger.info(f"Substituted feature images from src_rules for {switched} items")
                    except Exception as e:
                        logger.warning(f"Failed feature-image substitution via src_rules: {e}")
                # Step 4: Extract features
                logger.info("Step 4/6: Extracting features...")
                step_start = time.perf_counter()
                # Stream-save mode: save outputs immediately after each plant's features when fast output is enabled
                stream_save = False
                try:
                    import os as _os
                    stream_save = bool(int(_os.environ.get('STREAM_SAVE', '0'))) or bool(getattr(self.output_manager, 'fast_mode', False))
                except Exception:
                    stream_save = False

                plants = self._extract_features(plants, stream_save=stream_save)
                logger.info(f"Features done in {(time.perf_counter()-step_start):.2f}s")
                
                # Step 5: Generate outputs (skip if already stream-saved)
                if not stream_save:
                    logger.info("Step 5/6: Generating outputs...")
                    step_start = time.perf_counter()
                    self._generate_outputs(plants)
                    logger.info(f"Outputs done in {(time.perf_counter()-step_start):.2f}s")
                
                # Step 6: Create summary
                logger.info("Step 6/6: Creating summary...")
                summary = self._create_summary(plants)
            else:
                logger.info("Segmentation-only mode: skipping texture/vegetation/morphology features and plots")
                # Segmentation-only: generate only segmentation outputs and a minimal summary
                logger.info("Step 4/4: Generating segmentation outputs (segmentation-only mode)...")
                self._generate_outputs(plants)
                summary = {
                    "total_plants": len(plants),
                    "successful_plants": len(plants),
                    "failed_plants": 0,
                    "features_extracted": {
                        "texture": 0,
                        "vegetation": 0,
                        "morphology": 0
                    }
                }
            
            total_time = time.perf_counter() - total_start
            logger.info(f"Pipeline completed successfully in {total_time:.2f}s!")
            return {
                "plants": plants,
                "summary": summary,
                "config": self.config,
                "timing_seconds": total_time
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _export_white_background_maskouts(self, plants: Dict[str, Any], out_dir: Path) -> None:
        """Export RMBG composites with white background using the soft/binary masks.

        Filenames follow: plantX_plantX_frameY_maskout.png so the final instance script can detect plants.
        """
        # Clear any previous maskouts to avoid processing stale plants
        try:
            if out_dir.exists():
                for p in out_dir.glob("*_maskout.png"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
        count = 0
        # Previously certain plants were forced to bbox-only and skipped from instance export.
        # This constraint is now removed so that all plants can be exported for instance segmentation.
        bbox_only_plants: Set[str] = set()
        date_exception = "2025_05_08"
        for key, pdata in plants.items():
            try:
                # key format: "YYYY_MM_DD_plantX_frameY"
                parts = key.split('_')
                if len(parts) < 3:
                    continue
                plant_name = parts[-2]
                frame_token = parts[-1]  # e.g., frame8
                if not plant_name.startswith('plant') or not frame_token.startswith('frame'):
                    continue
                date_key = "_".join(parts[:3])
                # No special-case skip; allow instance export for all plants
                # Extract frame number
                frame_num = int(frame_token.replace('frame', ''))
                composite = pdata.get('composite')
                mask = pdata.get('mask')
                if composite is None or mask is None:
                    continue
                # Ensure 3-channel BGR
                if len(composite.shape) == 2:
                    composite_bgr = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
                else:
                    composite_bgr = composite
                out_img = composite_bgr.copy()
                # Set background to white where mask == 0
                out_img[mask == 0] = (255, 255, 255)
                out_path = out_dir / f"{plant_name}_{plant_name}_{frame_token}_maskout.png"
                cv2.imwrite(str(out_path), out_img)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to export maskout for {key}: {e}")
        logger.info(f"Exported {count} white-background maskouts to {out_dir}")
    
    def _run_yolo_detection(self, plants: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run YOLO detection on all plant composites to get bounding boxes.
        Stores detection results in plant_data for use in segmentation.
        
        Args:
            plants: Dictionary of plant data
            
        Returns:
            Updated plant data with YOLO detection bounding boxes
        """
        if self.yolo_detector is None:
            return plants
        
        total = len(plants)
        iterator = plants.items()
        if tqdm is not None:
            iterator = tqdm(list(plants.items()), desc="YOLO Detection", total=total, unit="img", leave=False)
        
        for idx, (key, pdata) in enumerate(iterator):
            try:
                composite = pdata.get('composite')
                if composite is None:
                    continue
                
                # Composite from preprocessor is RGB (uint8)
                # YOLO expects RGB, so we can pass it directly
                composite_for_yolo = composite.copy()
                if len(composite_for_yolo.shape) == 2:
                    # Grayscale, convert to RGB
                    composite_for_yolo = cv2.cvtColor(composite_for_yolo, cv2.COLOR_GRAY2RGB)
                elif len(composite_for_yolo.shape) == 3 and composite_for_yolo.shape[2] == 3:
                    # Already RGB, use as-is
                    pass
                else:
                    logger.warning(f"Unexpected composite shape for {key}: {composite_for_yolo.shape}")
                    continue
                
                # Run YOLO detection (composite is already RGB)
                detections = self.yolo_detector.detect(composite_for_yolo)
                
                # Store detection results
                pdata['yolo_detections'] = detections
                
                # Store bounding box for segmentation (use largest detection - should be potted plant)
                if detections.get('largest_box') is not None:
                    x1, y1, x2, y2 = detections['largest_box']
                    # Store as (x1, y1, x2, y2) tuple for bbox_lookup format
                    parts = key.split('_')
                    plant_name = parts[-2] if len(parts) >= 2 else None
                    if plant_name:
                        pdata['yolo_bbox'] = (x1, y1, x2, y2)
                        logger.debug(f"YOLO detected bbox for {key}: ({x1}, {y1}, {x2}, {y2})")
                
                # Store vase boxes to exclude from segmentation
                if detections.get('vase_boxes'):
                    pdata['yolo_vase_boxes'] = detections['vase_boxes']
                    logger.debug(f"YOLO found {len(detections['vase_boxes'])} vase boxes to exclude for {key}")
                
            except Exception as e:
                logger.warning(f"YOLO detection failed for {key}: {e}")
                # Store empty detections so we know YOLO was attempted
                pdata['yolo_detections'] = {
                    'boxes': [],
                    'scores': [],
                    'class_ids': [],
                    'class_names': [],
                    'largest_box': None,
                    'vase_boxes': []
                }
                pdata['yolo_error'] = str(e)
                continue
        
        return plants
    
    def _segment_plants(self, plants: Dict[str, Any], 
                       bbox_lookup: Optional[Dict[str, tuple]]) -> Dict[str, Any]:
        """Segment plants using BRIA model.

        If bbox_lookup is provided and contains an entry for the plant (e.g., 'plant1'),
        the image is cropped/masked to the bounding box region before segmentation and the
        predicted mask is mapped back to the full image size. In bbox mode a largest
        connected component post-processing is applied to obtain a clean target mask.
        """
        total = len(plants)
        iterator = plants.items()
        if tqdm is not None:
            iterator = tqdm(list(plants.items()), desc="Segmenting", total=total, unit="img", leave=False)
        for idx, (key, pdata) in enumerate(iterator):
            try:
                # Get composite image - ensure it exists
                composite = pdata.get('composite')
                if composite is None:
                    logger.error(f"No composite found for {key}, skipping segmentation")
                    continue
                h, w = composite.shape[:2]
                
                # Store segmentation composite (the one used for segmentation)
                if 'segmentation_composite' not in pdata:
                    pdata['segmentation_composite'] = composite.copy()
                
                # Determine bbox for this plant if available
                # Priority: 1) YOLO detection bbox, 2) bbox_lookup from file, 3) None
                parts = key.split('_')
                plant_name = parts[-2] if len(parts) >= 2 else None
                date_key = "_".join(parts[:3]) if len(parts) >= 3 else None  # e.g., 2025_04_16
                bbox = None
                
                # First check for YOLO detection bbox
                if 'yolo_bbox' in pdata:
                    bbox = pdata['yolo_bbox']
                    logger.debug(f"Using YOLO detection bbox for {key}: {bbox}")
                # Then check bbox_lookup from file
                elif bbox_lookup is not None and plant_name is not None:
                    # keys in bbox_lookup are typically like 'plant1'
                    bbox = bbox_lookup.get(plant_name)
                # Use bounding boxes when available; plant33 no longer ignores bbox
                
                # No plants are forced to bbox-only by default; allow env override per run
                bbox_only_plants: Set[str] = set()
                # Read env override: comma-separated list of plants to force bbox-only
                force_bbox_only_all = False
                force_bbox_only_plants: Set[str] = set()
                try:
                    import os as _os
                    try:
                        force_bbox_only_all = bool(int(_os.environ.get('FORCE_BBOX_ONLY_ALL', '0')))
                    except Exception:
                        force_bbox_only_all = False
                    try:
                        env_list = _os.environ.get('FORCE_BBOX_ONLY_PLANTS', '')
                        if env_list:
                            force_bbox_only_plants = {p.strip() for p in env_list.split(',') if p.strip()}
                    except Exception:
                        force_bbox_only_plants = set()
                except Exception:
                    force_bbox_only_all = False
                    force_bbox_only_plants = set()

                use_bbox_only = force_bbox_only_all or (plant_name in force_bbox_only_plants) or (plant_name in bbox_only_plants)

                # Do not use bounding boxes for date 2025_05_08
                if date_key == '2025_05_08':
                    bbox = None

                if bbox is not None:
                    # Clamp bbox to image
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(w, int(x1)))
                    x2 = max(0, min(w, int(x2)))
                    y1 = max(0, min(h, int(y1)))
                    y2 = max(0, min(h, int(y2)))
                    if x2 <= x1 or y2 <= y1:
                        x1, y1, x2, y2 = 0, 0, w, h

                    if use_bbox_only:
                        # Use the bbox as the mask directly (255 inside, 0 outside)
                        soft_full = np.zeros((h, w), dtype=np.float32)
                        soft_full[y1:y2, x1:x2] = 1.0
                        bin_full = np.zeros((h, w), dtype=np.uint8)
                        bin_full[y1:y2, x1:x2] = 255
                        pdata['soft_mask'] = soft_full
                        pdata['mask'] = bin_full
                    else:
                        # Segment inside the bbox region and map back
                        crop = composite[y1:y2, x1:x2]
                        
                        # Exclude vase areas from the crop before segmentation
                        crop_mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
                        if 'yolo_vase_boxes' in pdata and pdata['yolo_vase_boxes']:
                            # Create exclusion mask for vase boxes within the potted plant bbox
                            for vx1, vy1, vx2, vy2 in pdata['yolo_vase_boxes']:
                                # Convert vase box coordinates to crop-relative coordinates
                                crop_vx1 = max(0, vx1 - x1)
                                crop_vy1 = max(0, vy1 - y1)
                                crop_vx2 = min(x2 - x1, vx2 - x1)
                                crop_vy2 = min(y2 - y1, vy2 - y1)
                                # Only exclude if vase box overlaps with crop region
                                if crop_vx2 > crop_vx1 and crop_vy2 > crop_vy1:
                                    crop_mask[crop_vy1:crop_vy2, crop_vx1:crop_vx2] = 0.0
                                    logger.debug(f"Excluding vase box from crop: ({crop_vx1}, {crop_vy1}, {crop_vx2}, {crop_vy2})")
                        
                        # Apply exclusion mask to crop (set vase area to black/background)
                        crop_excluded = crop.copy()
                        if crop_mask.min() < 1.0:  # If there are exclusions
                            crop_excluded = crop_excluded.astype(np.float32)
                            if len(crop.shape) == 3:
                                # RGB image - need 3D mask
                                crop_mask_3d = np.stack([crop_mask] * crop.shape[2], axis=2)
                                crop_excluded = (crop_excluded * crop_mask_3d).astype(np.uint8)
                            else:
                                # Grayscale image - use 2D mask directly
                                crop_excluded = (crop_excluded * crop_mask).astype(np.uint8)
                        
                        soft_mask_crop = self.segmentation_manager.segment_image_soft(crop_excluded)
                        soft_full = np.zeros((h, w), dtype=np.float32)
                        soft_resized = cv2.resize(soft_mask_crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                        soft_full[y1:y2, x1:x2] = soft_resized
                        
                        # Also exclude vase areas from final mask
                        if 'yolo_vase_boxes' in pdata and pdata['yolo_vase_boxes']:
                            for vx1, vy1, vx2, vy2 in pdata['yolo_vase_boxes']:
                                # Clamp vase box to image bounds
                                vx1 = max(0, min(w, int(vx1)))
                                vx2 = max(0, min(w, int(vx2)))
                                vy1 = max(0, min(h, int(vy1)))
                                vy2 = max(0, min(h, int(vy2)))
                                if vx2 > vx1 and vy2 > vy1:
                                    soft_full[vy1:vy2, vx1:vx2] = 0.0
                                    logger.debug(f"Excluding vase box from final mask: ({vx1}, {vy1}, {vx2}, {vy2})")
                        
                        bin_full = (soft_full > 0.5).astype(np.uint8) * 255
                        try:
                            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(bin_full, 8)
                            if n_lbl > 1:
                                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                                bin_full = (labels == largest).astype(np.uint8) * 255
                        except Exception:
                            pass
                        pdata['soft_mask'] = soft_full.astype(np.float32)
                        pdata['mask'] = bin_full.astype(np.uint8)
                else:
                    # Full-image segmentation (no bbox)
                    soft_mask = self.segmentation_manager.segment_image_soft(composite)
                    pdata['soft_mask'] = soft_mask
                    pdata['mask'] = (soft_mask * 255.0).astype(np.uint8)
                
                # Ensure features_mask is set for all plants (not just selected ones)
                # This is critical for feature extraction when mask3 doesn't exist
                if 'mask' in pdata and isinstance(pdata['mask'], np.ndarray):
                    if 'features_mask' not in pdata:
                        pdata['features_mask'] = pdata['mask'].copy()
                    # Store original mask before any instance segmentation modifications
                    if 'original_mask' not in pdata:
                        pdata['original_mask'] = pdata['mask'].copy()
                
                # For selected plants, define mask3 to be exactly mask
                try:
                    selected_mask3_plants: Set[str] = {"plant19", "plant20", "plant27", "plant28", "plant33", "plant39", "plant40", "plant42", "plant44", "plant46"}
                    if plant_name in selected_mask3_plants and isinstance(pdata.get('mask'), np.ndarray):
                        pdata['mask3'] = pdata['mask']
                    # For non-selected plants (like Dr. Mullet's), ensure mask3 = mask for consistency
                    elif isinstance(pdata.get('mask'), np.ndarray):
                        pdata['mask3'] = pdata['mask'].copy()
                except Exception:
                    pass
                
                # Progress log every 25 items and for first/last
                if tqdm is None and (idx == 0 or (idx + 1) % 25 == 0 or (idx + 1) == total):
                    logger.info(f"Segmented {idx + 1}/{total}: {key}")
                
            except Exception as e:
                logger.error(f"Segmentation failed for {key}: {e}")
                try:
                    composite = pdata.get('composite')
                    if composite is not None:
                        pdata['soft_mask'] = np.zeros(composite.shape[:2], dtype=np.float32)
                        pdata['mask'] = np.zeros(composite.shape[:2], dtype=np.uint8)
                        # Ensure features_mask and mask3 are set even on failure
                        pdata['features_mask'] = pdata['mask'].copy()
                        pdata['mask3'] = pdata['mask'].copy()
                        pdata['original_mask'] = pdata['mask'].copy()
                except Exception:
                    pass
        
        return plants
    
    def _handle_occlusion(self, plants: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle occlusion problems using SAM2Long.
        
        This method groups plants by their base plant ID and processes
        each plant's 13-frame sequence to differentiate target plant
        from neighboring plants.
        
        Args:
            plants: Dictionary of plant data
            
        Returns:
            Updated plant data with occlusion handling results
        """
        if self.occlusion_handler is None:
            logger.warning("Occlusion handler not available, skipping occlusion handling")
            return plants
        
        # Group plants by base plant ID (e.g., "plant1" from "plant1_plant1_frame1")
        plant_groups = {}
        for key, pdata in plants.items():
            # Extract plant ID from key like "plant1_plant1_frame1"
            parts = key.split('_')
            if len(parts) >= 3:
                plant_id = parts[0]  # e.g., "plant1"
                if plant_id not in plant_groups:
                    plant_groups[plant_id] = []
                plant_groups[plant_id].append((key, pdata))
        
        logger.info(f"Processing {len(plant_groups)} plant groups for occlusion handling")
        
        # Process each plant group
        for plant_id, plant_frames in plant_groups.items():
            try:
                # Sort frames by frame number
                plant_frames.sort(key=lambda x: int(x[0].split('_')[-1].replace('frame', '')))
                
                if len(plant_frames) < 2:
                    logger.warning(f"Plant {plant_id} has only {len(plant_frames)} frames, skipping")
                    continue
                
                # Extract frames and keys
                frame_keys = [x[0] for x in plant_frames]
                frames = [x[1]['composite'] for x in plant_frames]
                
                logger.info(f"Processing plant {plant_id} with {len(frames)} frames")
                
                # Process with SAM2Long
                occlusion_results = self.occlusion_handler.segment_plant_sequence(
                    frames=frames,
                    target_plant_id=plant_id
                )
                
                # Update plant data with occlusion results
                target_masks = occlusion_results['target_masks']
                neighbor_masks = occlusion_results['neighbor_masks']
                
                for i, (key, pdata) in enumerate(plant_frames):
                    if i < len(target_masks):
                        # Update mask with target plant only
                        pdata['original_mask'] = pdata.get('mask', np.zeros_like(target_masks[i]))
                        pdata['mask'] = target_masks[i]
                        pdata['neighbor_mask'] = neighbor_masks[i]
                        pdata['occlusion_handled'] = True
                        
                        # Update soft mask as well
                        pdata['original_soft_mask'] = pdata.get('soft_mask', np.zeros_like(target_masks[i], dtype=np.float32))
                        pdata['soft_mask'] = (target_masks[i] / 255.0).astype(np.float32)
                
                # Calculate and store occlusion metrics
                metrics = self.occlusion_handler.get_occlusion_metrics(occlusion_results)
                for key, pdata in plant_frames:
                    pdata['occlusion_metrics'] = metrics
                
                logger.info(f"Plant {plant_id} occlusion handling completed")
                logger.info(f"  - Average occlusion ratio: {metrics['average_occlusion_ratio']:.3f}")
                logger.info(f"  - Frames with occlusion: {metrics['frames_with_occlusion']}")
                
            except Exception as e:
                logger.error(f"Occlusion handling failed for plant {plant_id}: {e}")
                # Mark as failed but continue
                for key, pdata in plant_frames:
                    pdata['occlusion_handled'] = False
                    pdata['occlusion_error'] = str(e)
        
        return plants
    
    def _extract_features(self, plants: Dict[str, Any], stream_save: bool = False) -> Dict[str, Any]:
        """Extract all features from plants.
        
        If stream_save is True, save outputs for each plant immediately after
        its features are computed to improve throughput and reduce peak memory.
        """
        total = len(plants)
        logger.info(f"Extracting features for {total} plants...")
        iterator = plants.items()
        if tqdm is not None:
            iterator = tqdm(list(plants.items()), desc="Extracting features", total=total, unit="img", leave=False)

        # Prepare output directories once if we're streaming saves
        if stream_save:
            try:
                self.output_manager.create_output_directories()
            except Exception:
                pass

        for idx, (key, pdata) in enumerate(iterator):
            try:
                logger.debug(f"Extracting features for {key}")
                
                # Extract texture features
                pdata['texture_features'] = self._extract_texture_features(pdata)
                
                # Extract vegetation indices
                pdata['vegetation_indices'] = self._extract_vegetation_indices(pdata)
                
                # Extract morphological features
                pdata['morphology_features'] = self._extract_morphology_features(pdata)
                
                # Immediately save outputs for this plant if streaming is enabled
                if stream_save:
                    try:
                        self.output_manager.save_plant_results(key, pdata)
                    except Exception as _e:
                        logger.error(f"Stream-save failed for {key}: {_e}")
                
                logger.debug(f"Features extracted for {key}")
                if tqdm is None and (idx == 0 or (idx + 1) % 25 == 0 or (idx + 1) == total):
                    logger.info(f"Extracted features for {idx + 1}/{total}: {key}")
                
            except Exception as e:
                logger.error(f"Feature extraction failed for {key}: {e}")
                # Add empty features
                pdata['texture_features'] = {}
                pdata['vegetation_indices'] = {}
                pdata['morphology_features'] = {}
        
        return plants
    
    def _extract_texture_features(self, pdata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract texture features for a single plant."""
        features = {}
        
        # Get bands to process
        bands = ['color', 'nir', 'red_edge', 'red', 'green', 'pca']
        
        for band in bands:
            try:
                # Prepare grayscale image
                gray_image = self._prepare_band_image(pdata, band)
                
                # Extract texture features
                band_features = self.texture_extractor.extract_all_texture_features(gray_image)
                
                # Compute statistics using mask3 → features_mask → mask
                mask = pdata.get('mask3', pdata.get('features_mask', pdata.get('mask')))
                stats = self.texture_extractor.compute_texture_statistics(band_features, mask)
                
                features[band] = {
                    'features': band_features,
                    'statistics': stats
                }
                
            except Exception as e:
                logger.error(f"Texture extraction failed for band {band}: {e}")
                features[band] = {'features': {}, 'statistics': {}}
        
        return features
    
    def _extract_vegetation_indices(self, pdata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract vegetation indices for a single plant."""
        try:
            spectral_stack = pdata.get('spectral_stack', {})
            # Prefer mask3 → features_mask → mask
            mask = pdata.get('mask3', pdata.get('features_mask', pdata.get('mask')))
            
            if not spectral_stack or mask is None:
                return {}
            
            return self.vegetation_extractor.compute_vegetation_indices(
                spectral_stack, mask
            )
            
        except Exception as e:
            logger.error(f"Vegetation index extraction failed: {e}")
            return {}
    
    def _extract_morphology_features(self, pdata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract morphological features for a single plant."""
        try:
            composite = pdata.get('composite')
            # Prefer mask3 → features_mask → mask
            mask = pdata.get('mask3', pdata.get('features_mask', pdata.get('mask')))
            
            if composite is None or mask is None:
                return {}
            
            return self.morphology_extractor.extract_morphology_features(
                composite, mask
            )
            
        except Exception as e:
            logger.error(f"Morphology feature extraction failed: {e}")
            return {}
    
    def _prepare_band_image(self, pdata: Dict[str, Any], band: str) -> np.ndarray:
        """Prepare grayscale image for a specific band."""
        if band == 'color':
            composite = pdata['composite']
            # Prefer mask3 → features_mask → mask
            mask = pdata.get('mask3', pdata.get('features_mask', pdata.get('mask')))
            if mask is not None:
                masked = self.mask_handler.apply_mask_to_image(composite, mask)
                return cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            else:
                return cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
        
        elif band == 'pca':
            # Create PCA from spectral bands
            spectral_stack = pdata.get('spectral_stack', {})
            # Prefer mask3 → features_mask → mask
            mask = pdata.get('mask3', pdata.get('features_mask', pdata.get('mask')))
            
            if not spectral_stack:
                return np.zeros((512, 512), dtype=np.uint8)
            
            # Stack bands
            bands_data = []
            for b in ['nir', 'red_edge', 'red', 'green']:
                if b in spectral_stack:
                    arr = spectral_stack[b]
                    # Squeeze all dimensions of size 1
                    arr = arr.squeeze()
                    # If still 3D, take first channel
                    if arr.ndim == 3:
                        arr = arr[:, :, 0]
                    arr = arr.astype(float)
                    if mask is not None:
                        arr = np.where(mask > 0, arr, np.nan)
                    bands_data.append(arr)
            
            if not bands_data:
                return np.zeros((512, 512), dtype=np.uint8)
            
            # Create PCA
            full_stack = np.stack(bands_data, axis=-1)
            h, w, c = full_stack.shape
            flat = full_stack.reshape(-1, c)
            valid = ~np.isnan(flat).any(axis=1)
            
            if valid.sum() == 0:
                return np.zeros((h, w), dtype=np.uint8)
            
            vec = np.zeros(h * w)
            vec[valid] = PCA(n_components=1, whiten=True).fit_transform(
                flat[valid]
            ).squeeze()
            
            gray_f = vec.reshape(h, w)
            if mask is not None:
                m, M = gray_f[mask > 0].min(), gray_f[mask > 0].max()
            else:
                m, M = gray_f.min(), gray_f.max()
            
            if M > m:
                gray = ((gray_f - m) / (M - m) * 255).astype(np.uint8)
            else:
                gray = np.zeros_like(gray_f, dtype=np.uint8)
            
            return gray
        
        else:
            # Individual spectral band
            spectral_stack = pdata.get('spectral_stack', {})
            # Prefer mask3 → features_mask → mask
            mask = pdata.get('mask3', pdata.get('features_mask', pdata.get('mask')))
            
            if band not in spectral_stack:
                return np.zeros((512, 512), dtype=np.uint8)
            
            arr = spectral_stack[band]
            # Squeeze all dimensions of size 1
            arr = arr.squeeze()
            # If still 3D, take first channel
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            arr = arr.astype(float)
            if mask is not None:
                arr = np.where(mask > 0, arr, np.nan)
            
            if mask is not None:
                m, M = np.nanmin(arr), np.nanmax(arr)
            else:
                m, M = arr.min(), arr.max()
            
            if M > m:
                gray = ((np.nan_to_num(arr, nan=m) - m) / (M - m) * 255).astype(np.uint8)
            else:
                gray = np.zeros_like(arr, dtype=np.uint8)
            
            return gray
    
    def _generate_outputs(self, plants: Dict[str, Any]) -> None:
        """Generate all output files and visualizations."""
        self.output_manager.create_output_directories()
        
        for key, pdata in plants.items():
            try:
                logger.debug(f"Generating outputs for {key}")
                self.output_manager.save_plant_results(key, pdata)
            except Exception as e:
                logger.error(f"Output generation failed for {key}: {e}")
    
    def _create_summary(self, plants: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of pipeline results."""
        summary = {
            "total_plants": len(plants),
            "successful_plants": 0,
            "failed_plants": 0,
            "features_extracted": {
                "texture": 0,
                "vegetation": 0,
                "morphology": 0
            }
        }
        
        for key, pdata in plants.items():
            try:
                # Check if features were extracted
                if pdata.get('texture_features'):
                    summary["features_extracted"]["texture"] += 1
                if pdata.get('vegetation_indices'):
                    summary["features_extracted"]["vegetation"] += 1
                if pdata.get('morphology_features'):
                    summary["features_extracted"]["morphology"] += 1
                
                summary["successful_plants"] += 1
                
            except Exception:
                summary["failed_plants"] += 1
        
        return summary

    def _apply_instance_masks(self, plants: Dict[str, Any], instance_results_dir: Path) -> None:
        """Replace segmentation masks with SAM2Long instance masks using track_1.

        Expects files under instance_results_dir/plantX/track_1/frame_YY_mask.png.
        """
        # Default and per-plant overrides for source plant, track and preferred frame
        default_track = "track_0"
        src_rules: Dict[str, str] = {
            "plant13": "plant12",
            "plant14": "plant13",
            "plant15": "plant14",
            "plant16": "plant15",
        }
        track_rules: Dict[str, str] = {
            # explicit track rules
            "plant1": "track_0",
            "plant4": "track_0",
            "plant9": "track_3",
            "plant13": "track_1",
            "plant14": "track_0",
            "plant15": "track_0",
            "plant16": "track_0",
            "plant18": "track_0",
            "plant19": "track_0",
            "plant23": "track_1",
            "plant26": "track_0",
            "plant27": "track_0",
            "plant29": "track_0",
            "plant31": "track_1",
            "plant34": "track_1",
            "plant35": "track_1",
            "plant36": "track_0",
            "plant37": "track_1",
            "plant38": "track_0",
            "plant39": "track_1",
            "plant40": "track_0",
            "plant41": "track_1",
            "plant42": "track_0",
            "plant43": "track_0",
            "plant45": "track_0",
        }
        frame_rules: Dict[str, int] = {
            # preferred frame overrides (1-based)
            "plant13": 8,
            "plant14": 8,
            "plant15": 8,
            "plant33": 2,
            "plant16": 4,
            "plant19": 5,
            "plant26": 8,
            "plant27": 8,
            "plant29": 8,
            "plant35": 7,
            "plant36": 6,
            "plant37": 2,
            "plant45": 5,
        }
        # Allow applying instance masks for all plants (no bbox-only exceptions)
        bbox_only_plants: Set[str] = set()
        date_exception = "2025_05_08"

        for key, pdata in plants.items():
            try:
                parts = key.split('_')
                if len(parts) < 3:
                    continue
                plant_name = parts[-2]
                frame_token = parts[-1]  # frame8
                if not (plant_name.startswith('plant') and frame_token.startswith('frame')):
                    continue
                date_key = "_".join(parts[:3])
                # No special-case skip when applying instance masks
                frame_num = int(frame_token.replace('frame', ''))
                # Resolve source plant, track and desired frame
                src_plant = src_rules.get(plant_name, plant_name)
                track_name = track_rules.get(plant_name, default_track)
                desired_frame = frame_rules.get(plant_name, frame_num)
                plant_dir = Path(instance_results_dir) / src_plant / track_name
                mask_path = plant_dir / f"frame_{desired_frame:02d}_mask.png"
                if not mask_path.exists():
                    # Fallback to current frame if override not found
                    fallback = plant_dir / f"frame_{frame_num:02d}_mask.png"
                    if fallback.exists():
                        mask_path = fallback
                    else:
                        # Last-resort: pick any available frame mask in the track directory
                        try:
                            candidates = sorted(plant_dir.glob("frame_*_mask.png"))
                            if len(candidates) > 0:
                                mask_path = candidates[0]
                            else:
                                continue
                        except Exception:
                            continue
                inst_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if inst_mask is None:
                    continue
                # Ensure binary uint8 0/255
                inst_mask_bin = (inst_mask > 0).astype(np.uint8) * 255
                pdata['original_mask'] = pdata.get('mask', inst_mask_bin.copy())
                pdata['mask'] = inst_mask_bin
                pdata['original_soft_mask'] = pdata.get('soft_mask', (inst_mask_bin / 255.0).astype(np.float32))
                pdata['soft_mask'] = (inst_mask_bin / 255.0).astype(np.float32)
                pdata['instance_applied'] = True

                # For selected plants, define mask3 to be exactly mask; otherwise use intersection logic
                try:
                    parts = key.split('_')
                    plant_name = parts[-2] if len(parts) >= 2 else ""
                    selected_mask3_plants: Set[str] = {"plant19", "plant20", "plant27", "plant28", "plant33", "plant39", "plant40", "plant42", "plant44", "plant46"}
                    if plant_name in selected_mask3_plants and isinstance(pdata.get('mask'), np.ndarray):
                        pdata['mask3'] = pdata['mask']
                        pdata['features_mask'] = pdata['mask']
                    else:
                        _m1 = pdata.get('mask')
                        _m2 = pdata.get('original_mask')
                        if isinstance(_m1, np.ndarray) and isinstance(_m2, np.ndarray):
                            _m1b = (_m1.astype(np.uint8) > 0)
                            _m2b = (_m2.astype(np.uint8) > 0)
                            mask3 = (_m1b & _m2b).astype(np.uint8) * 255
                            pdata['mask3'] = mask3
                            pdata['features_mask'] = mask3
                except Exception:
                    pass

                # After applying instance masks, also overwrite the composite and spectral stack
                # with the source plant's raw image (desired frame preferred) so that
                # feature extraction and saved originals/overlays are consistent with the mask source.
                try:
                    if plant_name in src_rules:
                        date_key = "_".join(parts[:3])
                        src_key_desired = f"{date_key}_{src_plant}_frame{desired_frame}"
                        src_key_same = f"{date_key}_{src_plant}_{frame_token}"
                        copy_from = plants.get(src_key_desired) or plants.get(src_key_same)
                        if copy_from is None:
                            # Fallback: load source composite from filesystem if not present in plants dict
                            try:
                                from PIL import Image as _Image
                                _date_folder = date_key.replace('_', '-')
                                _date_dir = Path(self.config.paths.input_folder)
                                if _date_dir.name != _date_folder:
                                    _date_dir = _date_dir / _date_folder
                                _frame_path = _date_dir / src_plant / f"{src_plant}_frame{desired_frame}.tif"
                                if not _frame_path.exists():
                                    _frame_path = _date_dir / src_plant / f"{src_plant}_frame{frame_num}.tif"
                                if _frame_path.exists():
                                    _img = _Image.open(str(_frame_path))
                                    # Process to composite using preprocessor
                                    comp, spec = self.preprocessor.process_raw_image(_img)
                                    copy_from = {"composite": comp, "spectral_stack": spec}
                            except Exception:
                                copy_from = None
                        if copy_from is not None:
                            # Preserve the segmentation-time composite once
                            if 'composite' in pdata and 'segmentation_composite' not in pdata:
                                pdata['segmentation_composite'] = pdata['composite']
                            if 'composite' in copy_from:
                                pdata['composite'] = copy_from['composite']
                            if 'spectral_stack' in copy_from:
                                pdata['spectral_stack'] = copy_from['spectral_stack']
                            # Ensure mask size matches the copied composite
                            ch, cw = pdata['composite'].shape[:2]
                            mh, mw = pdata['mask'].shape[:2]
                            if (mh, mw) != (ch, cw):
                                pdata['mask'] = cv2.resize(pdata['mask'].astype('uint8'), (cw, ch), interpolation=cv2.INTER_NEAREST)
                                pdata['soft_mask'] = (pdata['mask'] > 0).astype(np.float32)
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Instance mask apply failed for {key}: {e}")

    def _apply_instance_masks_from_mapping(self, plants: Dict[str, Any], mapping_file: Path) -> None:
        """Apply instance masks using an explicit mapping file with absolute paths.

        mapping JSON structure:
        {
          "plant1": {"frame": 8, "mask_path": "/abs/path/to/plant1/track_X/frame_08_mask.png"},
          "plant2": {"frame": 8, "mask_path": "/abs/path/.../frame_08_mask.png"},
          ...
        }
        If a plant's mapping specifies a different frame, only entries matching that frame are updated.
        """
        import json
        if not mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        # Normalize mapping plant keys to names like 'plantX'
        norm_map = {}
        for k, v in mapping.items():
            k_norm = k if str(k).startswith("plant") else f"plant{int(k)}" if str(k).isdigit() else str(k)
            norm_map[k_norm] = v

        for key, pdata in plants.items():
            try:
                parts = key.split('_')
                if len(parts) < 3:
                    continue
                plant_name = parts[-2]
                frame_token = parts[-1]
                if not (plant_name.startswith('plant') and frame_token.startswith('frame')):
                    continue
                frame_num = int(frame_token.replace('frame', ''))
                if plant_name not in norm_map:
                    continue
                entry = norm_map[plant_name]
                target_frame = int(entry.get("frame", frame_num))
                if frame_num != target_frame:
                    # Only update the designated frame for this plant
                    continue
                mask_path_str = entry.get("mask_path")
                if not mask_path_str:
                    continue
                mask_path = Path(mask_path_str)
                if not mask_path.exists():
                    logger.warning(f"Mask path not found for {plant_name} {frame_token}: {mask_path}")
                    continue
                inst_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if inst_mask is None:
                    continue
                inst_mask_bin = (inst_mask > 0).astype(np.uint8) * 255
                pdata['original_mask'] = pdata.get('mask', inst_mask_bin.copy())
                pdata['mask'] = inst_mask_bin
                pdata['original_soft_mask'] = pdata.get('soft_mask', (inst_mask_bin / 255.0).astype(np.float32))
                pdata['soft_mask'] = (inst_mask_bin / 255.0).astype(np.float32)
                pdata['instance_applied'] = True

                # Build mask3 = external(mask) AND BRIA(original_mask)
                try:
                    _m1 = pdata.get('mask')
                    _m2 = pdata.get('original_mask')
                    if isinstance(_m1, np.ndarray) and isinstance(_m2, np.ndarray):
                        _m1b = (_m1.astype(np.uint8) > 0)
                        _m2b = (_m2.astype(np.uint8) > 0)
                        mask3 = (_m1b & _m2b).astype(np.uint8) * 255
                        pdata['mask3'] = mask3
                        pdata['features_mask'] = mask3
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Instance mapping apply failed for {key}: {e}")


def run_pipeline(config_path: str, load_all_frames: bool = False, segmentation_only: bool = False, filter_plants: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to run the pipeline.
    
    Args:
        config_path: Path to configuration file
        load_all_frames: Whether to load all frames or selected frames
        segmentation_only: If True, run segmentation only and skip feature extraction
        
    Returns:
        Pipeline results
    """
    pipeline = SorghumPipeline(config_path)
    return pipeline.run(load_all_frames, segmentation_only, filter_plants)


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    load_all = "--all" in sys.argv
    seg_only = "--seg-only" in sys.argv
    # Basic arg parse for --plant=<name>
    plant_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--plant="):
            plant_filter = [arg.split("=", 1)[1]]
    
    try:
        results = run_pipeline(config_path, load_all, seg_only, plant_filter)
        print("Pipeline completed successfully!")
        print(f"Processed {results['summary']['total_plants']} plants")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
