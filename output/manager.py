"""
Output manager for the Sorghum Pipeline.

This module handles saving results, generating visualizations,
and creating reports.
"""

import os
import json
import numpy as np
import cv2

# Use a non-GUI backend to avoid segmentation faults in headless runs
try:
    import matplotlib
    if os.environ.get('MPLBACKEND') is None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
except Exception:
    # Fallback safe imports (should not happen normally)
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages output generation and saving."""
    
    def __init__(self, output_folder: str, settings: Any):
        """
        Initialize output manager.
        
        Args:
            output_folder: Base output folder
            settings: Output settings from config
        """
        self.output_folder = Path(output_folder)
        self.settings = settings
        # Flag for Dr. Mullet's experiment (simplified output - only mask.png and one maskout)
        self.is_dr_mullet_experiment = False
        # Fast mode and parallel save controls
        try:
            self.fast_mode: bool = bool(int(os.environ.get('FAST_OUTPUT', '0'))) or bool(getattr(settings, 'fast_mode', False))
        except Exception:
            self.fast_mode = False
        try:
            self.max_workers: int = int(os.environ.get('FAST_SAVE_WORKERS', '4'))
        except Exception:
            self.max_workers = 4
        try:
            self.png_compression: int = int(os.environ.get('PNG_COMPRESSION', '1'))  # 0-9; 1 is fast
        except Exception:
            self.png_compression = 1
        
        # Reduce thread usage to lower risk of native library segfaults
        try:
            import os as _os
            _os.environ.setdefault('OMP_NUM_THREADS', '1')
            _os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
            _os.environ.setdefault('MKL_NUM_THREADS', '1')
            _os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        except Exception:
            pass
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        # Create base directories
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _imwrite_fast(self, dest: Path, img: np.ndarray) -> None:
        try:
            cv2.imwrite(str(dest), img, [cv2.IMWRITE_PNG_COMPRESSION, int(self.png_compression)])
        except Exception:
            cv2.imwrite(str(dest), img)
    
    def create_output_directories(self) -> None:
        """Ensure base output directory exists.

        Note: Do NOT create subdirectories at the root (e.g., 'analysis').
        Subdirectories are created within each plant's directory only.
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def save_plant_results(self, plant_key: str, plant_data: Dict[str, Any]) -> None:
        """
        Save all results for a single plant.
        
        Args:
            plant_key: Plant identifier (e.g., "2025_02_05_plant1_frame8")
            plant_data: Plant data dictionary
        """
        try:
            # Parse plant key
            parts = plant_key.split('_')
            date_key = "_".join(parts[:3])
            plant_name = parts[3]
            frame_key = parts[4] if len(parts) > 4 else "frame0"
            
            # Create plant-specific directory
            plant_dir = self.output_folder / date_key / plant_name
            plant_dir.mkdir(parents=True, exist_ok=True)
            
            # Save segmentation results
            self._save_segmentation_results(plant_dir, plant_name, plant_data)
            
            # Save texture features
            self._save_texture_features(plant_dir, plant_data)
            
            # Save vegetation indices
            self._save_vegetation_indices(plant_dir, plant_data)
            
            # Save morphology features
            self._save_morphology_features(plant_dir, plant_data)
            
            # Save analysis plots
            self._save_analysis_plots(plant_dir, plant_data)
            
            # Save metadata
            self._save_metadata(plant_dir, plant_key, plant_data)
            
            logger.debug(f"Results saved for {plant_key}")
            
        except Exception as e:
            logger.error(f"Failed to save results for {plant_key}: {e}")
    
    def _save_segmentation_results(self, plant_dir: Path, plant_name: str, plant_data: Dict[str, Any]) -> None:
        """Save segmentation results."""
        if not self.settings.save_images:
            return
        
        seg_dir = plant_dir / self.settings.segmentation_dir
        seg_dir.mkdir(exist_ok=True)
        
        try:
            tasks: List[Tuple[Path, np.ndarray]] = []
            # Choose which base image to present in original/overlay
            use_feature_image = False
            try:
                # Allow env override, and special-case plants 13-16 per user requirement
                use_feature_image = bool(int(os.environ.get('OUTPUT_USE_FEATURE_IMAGE', '0'))) or plant_name in { 'plant13','plant14','plant15','plant16' }
            except Exception:
                use_feature_image = plant_name in { 'plant13','plant14','plant15','plant16' }
            if use_feature_image:
                base_image = plant_data.get('composite', plant_data.get('segmentation_composite'))
            else:
                base_image = plant_data.get('segmentation_composite', plant_data.get('composite'))
            # For Dr. Mullet's experiment: only save mask.png and one maskout
            if self.is_dr_mullet_experiment:
                # Save only essential outputs
                if base_image is not None:
                    tasks.append((seg_dir / 'original.png', base_image))
                if 'mask' in plant_data:
                    tasks.append((seg_dir / 'mask.png', plant_data['mask']))
                if base_image is not None and 'mask' in plant_data:
                    overlay = self._create_overlay(base_image, plant_data['mask'])
                    tasks.append((seg_dir / 'overlay.png', overlay))
                    # Only one maskout: using the final mask (this masks out the detected plant)
                    maskout = self._create_maskout_white_background(base_image, plant_data['mask'])
                    tasks.append((seg_dir / 'maskout.png', maskout))
                    
                    # Also save YOLO detection visualization if available
                    # Save even if detections are empty (to show that YOLO ran)
                    if 'yolo_detections' in plant_data:
                        yolo_bbox = plant_data.get('yolo_bbox')
                        # If no bbox but we have detections, try to get largest_box from detections
                        if yolo_bbox is None and plant_data['yolo_detections'].get('largest_box'):
                            yolo_bbox = plant_data['yolo_detections']['largest_box']
                        
                        if yolo_bbox is not None:
                            yolo_vis = self._create_yolo_visualization(base_image, plant_data['yolo_detections'], yolo_bbox)
                        else:
                            # Create visualization with all detections even if no bbox
                            yolo_vis = self._create_yolo_visualization(base_image, plant_data['yolo_detections'], None)
                        
                        if yolo_vis is not None:
                            tasks.append((seg_dir / 'yolo_detection.png', yolo_vis))
            else:
                # Standard output (for other experiments)
                if base_image is not None:
                    tasks.append((seg_dir / 'original.png', base_image))
                if 'mask' in plant_data:
                    tasks.append((seg_dir / 'mask.png', plant_data['mask']))
                if 'mask3' in plant_data and isinstance(plant_data['mask3'], np.ndarray):
                    tasks.append((seg_dir / 'mask3.png', plant_data['mask3']))
                # Save the BRIA-generated mask (if present before overrides) as mask2.png
                if 'original_mask' in plant_data and isinstance(plant_data['original_mask'], np.ndarray):
                    tasks.append((seg_dir / 'mask2.png', plant_data['original_mask']))
                if base_image is not None and 'mask' in plant_data:
                    overlay = self._create_overlay(base_image, plant_data['mask'])
                    tasks.append((seg_dir / 'overlay.png', overlay))
                if 'masked_composite' in plant_data:
                    tasks.append((seg_dir / 'masked_composite.png', plant_data['masked_composite']))

                # Create white-background maskouts
                try:
                    if base_image is not None and 'mask' in plant_data:
                        maskout_external = self._create_maskout_white_background(base_image, plant_data['mask'])
                        tasks.append((seg_dir / 'maskout_external.png', maskout_external))
                    # BRIA-only maskout directly on original composite
                    if base_image is not None and 'original_mask' in plant_data and isinstance(plant_data['original_mask'], np.ndarray):
                        maskout_bria = self._create_maskout_white_background(base_image, plant_data['original_mask'])
                        tasks.append((seg_dir / 'maskout_bria.png', maskout_bria))
                    # mask3 maskout on original composite
                    if base_image is not None and 'mask3' in plant_data and isinstance(plant_data['mask3'], np.ndarray):
                        maskout_mask3 = self._create_maskout_white_background(base_image, plant_data['mask3'])
                        tasks.append((seg_dir / 'maskout_mask3.png', maskout_mask3))
                except Exception as _e:
                    logger.debug(f"Failed to create double maskouts: {_e}")

            if self.max_workers > 1 and len(tasks) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    futures = [ex.submit(self._imwrite_fast, p, img) for p, img in tasks]
                    for _ in as_completed(futures):
                        pass
            else:
                for p, img in tasks:
                    self._imwrite_fast(p, img)
        except Exception as e:
            logger.error(f"Failed to save segmentation results: {e}")
    
    def _save_texture_features(self, plant_dir: Path, plant_data: Dict[str, Any]) -> None:
        """Save texture features."""
        if not self.settings.save_images or 'texture_features' not in plant_data:
            return
        
        texture_dir = plant_dir / self.settings.texture_dir
        texture_dir.mkdir(exist_ok=True)
        
        def save_feature_png(feature_name: str, values: Any, dest: Path, cmap_name: str = 'viridis') -> None:
            try:
                arr = np.asarray(values)
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    self._imwrite_fast(dest, cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    return
                if self.fast_mode:
                    # Fast path: simple normalization, no matplotlib
                    normalized = self._normalize_to_uint8(np.nan_to_num(arr.astype(np.float64), nan=0.0))
                    self._imwrite_fast(dest, normalized)
                else:
                    arr = arr.astype(np.float64)
                    masked = np.ma.masked_invalid(arr)
                    # For LBP and HOG, ensure proper value range to avoid constant output
                    if feature_name in ('lbp', 'hog'):
                        # Use full range including zeros to ensure proper colormap mapping
                        vmin = float(arr.min())
                        vmax = float(arr.max())
                        # Ensure there's some variation
                        if vmax <= vmin:
                            vmax = vmin + 1.0
                    else:
                        vmin, vmax = None, None
                    
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.set_axis_off()
                    ax.set_facecolor('white')
                    if vmin is not None and vmax is not None and vmax > vmin:
                        im = ax.imshow(masked, cmap=cmap_name, vmin=vmin, vmax=vmax)
                    else:
                        im = ax.imshow(masked, cmap=cmap_name)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="2%", pad=0.02)
                    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
                    cbar.set_label(feature_name, fontsize=7)
                    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
                    if hasattr(cbar, 'outline') and cbar.outline is not None:
                        cbar.outline.set_linewidth(0.5)
                    plt.tight_layout()
                    plt.savefig(dest, dpi=self.settings.plot_dpi, bbox_inches='tight')
                    plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to save texture feature image for {feature_name}: {e}")
                try:
                    normalized = self._normalize_to_uint8(np.nan_to_num(arr, nan=0.0))
                    self._imwrite_fast(dest, normalized)
                except Exception:
                    pass

        try:
            texture_features = plant_data['texture_features']
            
            for band, band_data in texture_features.items():
                if 'features' not in band_data:
                    continue
                
                band_dir = texture_dir / band
                band_dir.mkdir(exist_ok=True)
                
                features = band_data['features']
                
                # Save individual feature maps (optionally in parallel)
                items: List[Tuple[str, np.ndarray, Path, str]] = []
                for feature_name, feature_map in features.items():
                    if feature_name == 'ehd_features':
                        for i in range(feature_map.shape[0]):
                            channel = feature_map[i]
                            if isinstance(channel, np.ndarray) and channel.size > 0:
                                items.append((f'ehd_channel_{i}', channel, band_dir / f'ehd_channel_{i}.png', 'magma'))
                    else:
                        if isinstance(feature_map, np.ndarray) and feature_map.size > 0:
                            cmap_choice = 'gray' if feature_name in ('lbp', 'hog') else 'plasma' if feature_name.startswith('lac') else 'viridis'
                            items.append((feature_name, feature_map, band_dir / f'{feature_name}.png', cmap_choice))

                if self.max_workers > 1 and len(items) > 1:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                        futures = [ex.submit(save_feature_png, n, m, p, c) for (n, m, p, c) in items]
                        for _ in as_completed(futures):
                            pass
                else:
                    for (n, m, p, c) in items:
                        save_feature_png(n, m, p, c)
                
                # Create feature summary plot
                self._create_texture_summary_plot(band_dir, features, band)
                
                # Save texture statistics if available
                if 'statistics' in band_data and isinstance(band_data['statistics'], dict):
                    try:
                        with open(band_dir / 'texture_statistics.json', 'w') as f:
                            json.dump(band_data['statistics'], f, indent=2)
                    except Exception as e:
                        logger.error(f"Failed to save texture statistics for {band}: {e}")
                
        except Exception as e:
            logger.error(f"Failed to save texture features: {e}")
    
    def _save_vegetation_indices(self, plant_dir: Path, plant_data: Dict[str, Any]) -> None:
        """Save vegetation indices."""
        if not self.settings.save_images or 'vegetation_indices' not in plant_data:
            return
        
        veg_dir = plant_dir / self.settings.vegetation_dir
        veg_dir.mkdir(exist_ok=True)
        
        # Colormap and range settings per index
        index_cmap_settings = {
            "NDVI": (cm.RdYlGn, -1, 1),
            "GNDVI": (cm.RdYlGn, -1, 1),
            "NDRE": (cm.RdYlGn, -1, 1),
            "GRNDVI": (cm.RdYlGn, -1, 1),
            "TNDVI": (cm.RdYlGn, -1, 1),
            "MGRVI": (cm.RdYlGn, -1, 1),
            "GRVI": (cm.RdYlGn, -1, 1),
            "NGRDI": (cm.RdYlGn, -1, 1),
            "MSAVI": (cm.YlGn, 0, 1),
            "OSAVI": (cm.YlGn, 0, 1),
            "TSAVI": (cm.YlGn, 0, 1),
            "GSAVI": (cm.YlGn, 0, 1),
            "NDWI": (cm.Blues, -1, 1),
            "DSWI4": (cm.Blues, -1, 1),
            "CIRE": (cm.viridis, 0, 10),
            "LCI": (cm.viridis, 0, 5),
            "CIgreen": (cm.viridis, 0, 5),
            "MCARI": (cm.viridis, 0, 1.5),
            "MCARI1": (cm.viridis, 0, 1.5),
            "MCARI2": (cm.viridis, 0, 1.5),
            "CVI": (cm.plasma, 0, 10),
            "TCARI": (cm.viridis, 0, 1),
            "TCARIOSAVI": (cm.viridis, 0, 1),
            "AVI": (cm.magma, 0, 1),
            "SIPI2": (cm.inferno, 0, 1),
            "ARI": (cm.magma, 0, 1),
            "ARI2": (cm.magma, 0, 1),
            "DVI": (cm.Greens, 0, None),
            "WDVI": (cm.Greens, 0, None),
            "SR": (cm.viridis, 0, 10),
            "MSR": (cm.viridis, 0, 10),
            "PVI": (cm.cividis, None, None),
            "GEMI": (cm.cividis, 0, 1),
            "ExR": (cm.Reds, -1, 1),
            "RI": (cm.Reds, 0, None),
            "RRI1": (cm.Reds, 0, 1)
        }

        def save_index_png(index_name: str, values: Any, dest: Path) -> None:
            try:
                arr = values
                if not isinstance(arr, (list, tuple,)) and isinstance(arr, (float, int)):
                    return
                arr = np.asarray(arr, dtype=np.float64)
                if self.fast_mode:
                    normalized = self._normalize_to_uint8(np.nan_to_num(arr, nan=0.0))
                    self._imwrite_fast(dest, normalized)
                else:
                    cmap, vmin, vmax = index_cmap_settings.get(index_name, (cm.viridis, np.nanmin(arr), np.nanmax(arr)))
                    if vmin is None:
                        vmin = np.nanmin(arr)
                    if vmax is None:
                        vmax = np.nanmax(arr)
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                        vmin, vmax = 0.0, 1.0
                    masked = np.ma.masked_invalid(arr)
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.set_axis_off()
                    ax.set_facecolor('white')
                    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="2%", pad=0.02)
                    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
                    cbar.set_label(index_name, fontsize=7)
                    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
                    if hasattr(cbar, 'outline') and cbar.outline is not None:
                        cbar.outline.set_linewidth(0.5)
                    plt.tight_layout()
                    plt.savefig(dest, dpi=self.settings.plot_dpi, bbox_inches='tight')
                    plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to save vegetation index image for {index_name}: {e}")
                try:
                    # Fallback simple normalization
                    normalized = self._normalize_to_uint8(np.nan_to_num(arr, nan=0.0))
                    self._imwrite_fast(dest, normalized)
                except Exception:
                    pass

        try:
            vegetation_indices = plant_data['vegetation_indices']
            
            items_png: List[Tuple[str, np.ndarray, Path]] = []
            items_stats: List[Tuple[Path, Dict[str, Any]]] = []
            for index_name, index_data in vegetation_indices.items():
                if isinstance(index_data, dict) and 'values' in index_data:
                    values = index_data['values']
                    if isinstance(values, np.ndarray) and values.size > 0:
                        items_png.append((index_name, values, veg_dir / f'{index_name}.png'))
                    stats = index_data.get('statistics')
                    if isinstance(stats, dict):
                        items_stats.append((veg_dir / f'{index_name}_stats.json', stats))

            # Save sequentially to avoid matplotlib thread-safety issues
            for (name, arr, dest) in items_png:
                save_index_png(name, arr, dest)
            for (path, stats) in items_stats:
                try:
                    with open(path, 'w') as f:
                        json.dump(stats, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save stats for {path.name.split('.')[0]}: {e}")
                
            # Create vegetation index summary (skip in fast mode)
            if not self.fast_mode:
                self._create_vegetation_summary_plot(veg_dir, vegetation_indices)
            
            # Save aggregated vegetation statistics
            try:
                all_stats = {k: v.get('statistics', {}) for k, v in vegetation_indices.items() if isinstance(v, dict)}
                with open(veg_dir / 'vegetation_statistics.json', 'w') as f:
                    json.dump(all_stats, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save aggregated vegetation statistics: {e}")
            
        except Exception as e:
            logger.error(f"Failed to save vegetation indices: {e}")
    
    def _save_morphology_features(self, plant_dir: Path, plant_data: Dict[str, Any]) -> None:
        """Save morphological features."""
        if not self.settings.save_images or 'morphology_features' not in plant_data:
            return
        
        morph_dir = plant_dir / self.settings.morphology_dir
        morph_dir.mkdir(exist_ok=True)
        
        try:
            morphology_features = plant_data['morphology_features']
            
            # Save morphological images
            if 'images' in morphology_features:
                for image_name, image_data in morphology_features['images'].items():
                    if isinstance(image_data, np.ndarray) and image_data.size > 0:
                        cv2.imwrite(str(morph_dir / f'{image_name}.png'), image_data)
            
            # Save morphological data
            if 'traits' in morphology_features:
                traits = morphology_features['traits']
                with open(morph_dir / 'traits.json', 'w') as f:
                    json.dump(traits, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save morphology features: {e}")
    
    def _save_analysis_plots(self, plant_dir: Path, plant_data: Dict[str, Any]) -> None:
        """Save analysis plots."""
        if not self.settings.save_plots or self.fast_mode:
            return
        
        analysis_dir = plant_dir / self.settings.analysis_dir
        analysis_dir.mkdir(exist_ok=True)
        
        try:
            # Create comprehensive analysis plot
            self._create_comprehensive_analysis_plot(analysis_dir, plant_data)
            
        except Exception as e:
            logger.error(f"Failed to save analysis plots: {e}")
    
    def _save_metadata(self, plant_dir: Path, plant_key: str, plant_data: Dict[str, Any]) -> None:
        """Save metadata for the plant."""
        if not self.settings.save_metadata:
            return
        
        try:
            metadata = {
                'plant_key': plant_key,
                'timestamp': pd.Timestamp.now().isoformat(),
                'image_shape': plant_data.get('composite', np.array([])).shape if 'composite' in plant_data else None,
                'has_mask': 'mask' in plant_data and plant_data['mask'] is not None,
                'features_available': {
                    'texture': 'texture_features' in plant_data,
                    'vegetation': 'vegetation_indices' in plant_data,
                    'morphology': 'morphology_features' in plant_data
                }
            }
            
            with open(plant_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                       color: Tuple[int, int, int] = (0, 255, 0), 
                       alpha: float = 0.5) -> np.ndarray:
        """Return a strictly masked image: pixels where mask>0 keep original; others set to 0."""
        if mask is None:
            return image
        # Resize mask to image size if needed
        if mask.shape[:2] != image.shape[:2]:
            try:
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass
        binary = (mask.astype(np.int32) > 0).astype(np.uint8) * 255
        return cv2.bitwise_and(image, image, mask=binary)
    
    def _create_yolo_visualization(self, image: np.ndarray, detections: Dict[str, Any], bbox: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
        """Create visualization of YOLO detection bounding boxes on image."""
        try:
            vis = image.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
            elif len(vis.shape) == 3 and vis.shape[2] == 3:
                # Ensure RGB format
                if vis.dtype != np.uint8:
                    vis = (vis * 255).astype(np.uint8) if vis.max() <= 1.0 else vis.astype(np.uint8)
            else:
                logger.warning(f"Unexpected image shape for YOLO visualization: {vis.shape}")
                return None
            
            # Draw all detections
            boxes = detections.get('boxes', [])
            scores = detections.get('scores', [])
            class_names = detections.get('class_names', [])
            
            if bbox is not None:
                # Draw the primary bounding box (used for segmentation) in blue
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)  # Blue box, thicker
                # Find matching class name and score for the primary box
                label = "YOLO Detection (Primary)"
                if len(scores) > 0 and len(class_names) > 0:
                    # Find matching box to get class name
                    for i, box in enumerate(boxes):
                        if abs(box[0] - x1) < 5 and abs(box[1] - y1) < 5:
                            if i < len(scores) and i < len(class_names):
                                label = f"{class_names[i]} ({scores[i]:.2f})"
                                break
                cv2.putText(vis, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw all other detections in green (non-vase detections)
            excluded_classes = ['vase', 'vast']
            if boxes:
                for i, box in enumerate(boxes):
                    bx1, by1, bx2, by2 = box
                    # Skip if it's the primary box (already drawn)
                    if bbox is None or (abs(bx1 - bbox[0]) > 5 or abs(by1 - bbox[1]) > 5):
                        # Double-check: skip vase/vast detections if any somehow got through
                        if i < len(class_names):
                            class_name_lower = class_names[i].lower()
                            if any(excluded in class_name_lower for excluded in excluded_classes):
                                continue  # Skip drawing vase detections
                        cv2.rectangle(vis, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)  # Green boxes
                        if i < len(scores) and i < len(class_names):
                            label = f"{class_names[i]}: {scores[i]:.2f}"
                            cv2.putText(vis, label, (int(bx1), int(by1) - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw vase boxes in red to show they are detected but excluded
            vase_boxes = detections.get('vase_boxes', [])
            if vase_boxes:
                for vx1, vy1, vx2, vy2 in vase_boxes:
                    cv2.rectangle(vis, (int(vx1), int(vy1)), (int(vx2), int(vy2)), (0, 0, 255), 2)  # Red boxes for vase
                    cv2.putText(vis, "vase (excluded)", (int(vx1), int(vy1) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # If no detections, add text indicating YOLO ran but found nothing
            if not boxes and bbox is None:
                h, w = vis.shape[:2]
                cv2.putText(vis, "YOLO Detection: No objects found", (w//4, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return vis
        except Exception as e:
            logger.warning(f"Failed to create YOLO visualization: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _create_maskout_white_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create maskout image with white background.
        
        This masks out the detected plant (keeps the plant, sets background to white).
        The mask is created from BRIA segmentation within the YOLO detection bounding box.
        """
        # Create white background
        white_background = np.full_like(image, 255, dtype=np.uint8)
        
        # Apply mask to original image (keep only masked regions)
        masked_image = image.copy()
        masked_image[mask == 0] = 0  # Set non-masked regions to black
        
        # Combine: white background + masked image
        result = white_background.copy()
        result[mask > 0] = masked_image[mask > 0]
        
        return result
    
    def _normalize_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to uint8 range."""
        if arr.size == 0:
            return arr.astype(np.uint8)
        
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.ptp(arr) > 0:
            normalized = (arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255
        else:
            normalized = np.zeros_like(arr)
        
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    def _create_texture_summary_plot(self, output_dir: Path, features: Dict[str, np.ndarray], band: str) -> None:
        """Create texture feature summary plot."""
        try:
            # Get available features
            available_features = [k for k, v in features.items() 
                                if isinstance(v, np.ndarray) and v.size > 0 and k != 'ehd_features']
            
            if not available_features:
                return
            
            # Create subplot
            n_features = len(available_features)
            cols = min(3, n_features)
            rows = (n_features + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if n_features == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, feature_name in enumerate(available_features):
                row, col = divmod(i, cols)
                ax = axes[row, col] if rows > 1 else axes[col]
                
                feature_map = features[feature_name]
                ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'{band.upper()} - {feature_name.upper()}')
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(n_features, rows * cols):
                row, col = divmod(i, cols)
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{band}_texture_summary.png', 
                       dpi=self.settings.plot_dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create texture summary plot: {e}")
    
    def _create_vegetation_summary_plot(self, output_dir: Path, vegetation_indices: Dict[str, Any]) -> None:
        """Create vegetation index summary plot."""
        try:
            # Get available indices
            available_indices = [k for k, v in vegetation_indices.items() 
                               if isinstance(v, dict) and 'values' in v and isinstance(v['values'], np.ndarray)]
            
            if not available_indices:
                return
            
            # Create subplot
            n_indices = len(available_indices)
            cols = min(3, n_indices)
            rows = (n_indices + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if n_indices == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, index_name in enumerate(available_indices):
                row, col = divmod(i, cols)
                ax = axes[row, col] if rows > 1 else axes[col]
                
                values = vegetation_indices[index_name]['values']
                im = ax.imshow(values, cmap='RdYlGn')
                ax.set_title(f'{index_name}')
                ax.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.02)
                cbar = plt.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
                if hasattr(cbar, 'outline') and cbar.outline is not None:
                    cbar.outline.set_linewidth(0.5)
            
            # Hide unused subplots
            for i in range(n_indices, rows * cols):
                row, col = divmod(i, cols)
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'vegetation_indices_summary.png', 
                       dpi=self.settings.plot_dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create vegetation summary plot: {e}")
    
    def _create_comprehensive_analysis_plot(self, output_dir: Path, plant_data: Dict[str, Any]) -> None:
        """Create comprehensive analysis plot."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            if 'composite' in plant_data:
                axes[0, 0].imshow(cv2.cvtColor(plant_data['composite'], cv2.COLOR_BGR2RGB))
                axes[0, 0].set_title('Original Composite')
                axes[0, 0].axis('off')
            
            # Mask
            if 'mask' in plant_data:
                axes[0, 1].imshow(plant_data['mask'], cmap='gray')
                axes[0, 1].set_title('Segmentation Mask')
                axes[0, 1].axis('off')
            
            # Overlay
            if 'composite' in plant_data and 'mask' in plant_data:
                overlay = self._create_overlay(plant_data['composite'], plant_data['mask'])
                axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                axes[0, 2].set_title('Overlay')
                axes[0, 2].axis('off')
            
            # Texture features (if available)
            if 'texture_features' in plant_data and 'color' in plant_data['texture_features']:
                color_features = plant_data['texture_features']['color'].get('features', {})
                if 'lbp' in color_features:
                    axes[1, 0].imshow(color_features['lbp'], cmap='viridis')
                    axes[1, 0].set_title('LBP Texture')
                    axes[1, 0].axis('off')
            
            # Vegetation indices (if available)
            if 'vegetation_indices' in plant_data:
                veg_indices = plant_data['vegetation_indices']
                if 'NDVI' in veg_indices and 'values' in veg_indices['NDVI']:
                    axes[1, 1].imshow(veg_indices['NDVI']['values'], cmap='RdYlGn')
                    axes[1, 1].set_title('NDVI')
                    axes[1, 1].axis('off')
            
            # Morphology (if available)
            if 'morphology_features' in plant_data and 'images' in plant_data['morphology_features']:
                morph_images = plant_data['morphology_features']['images']
                if 'skeleton' in morph_images:
                    axes[1, 2].imshow(morph_images['skeleton'], cmap='gray')
                    axes[1, 2].set_title('Skeleton')
                    axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'comprehensive_analysis.png', 
                       dpi=min(getattr(self.settings, 'plot_dpi', 100), 100), bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create comprehensive analysis plot: {e}")
    
    def create_pipeline_summary(self, results: Dict[str, Any]) -> None:
        """Create a summary of the entire pipeline run."""
        try:
            summary_file = self.output_folder / 'pipeline_summary.json'
            
            with open(summary_file, 'w') as f:
                json.dump(results['summary'], f, indent=2)
            
            logger.info(f"Pipeline summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to create pipeline summary: {e}")
