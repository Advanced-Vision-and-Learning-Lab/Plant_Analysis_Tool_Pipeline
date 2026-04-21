## UNL-CPPD → YOLO-Pose + Mullet pseudo-labeling

This folder contains utilities to:

- Convert **UNL-CPPD** (base + leaf tip/collar keypoints) into **Ultralytics YOLO-Pose** training format.
- Run a trained pose model on your unlabeled **Mullet** images to generate **auto keypoint annotations**.

### 1) Prepare UNL-CPPD for Ultralytics YOLO-Pose

UNL-CPPD ground-truth structure keypoints:

- **base**: one point per image
- For each leaf id \(1..11\): **collar** + **tip**

Ultralytics needs a fixed keypoint layout, so we encode **one plant instance per image** with:

- \(K = 23\) keypoints = `base` + `11 * (collar, tip)`
- Missing leaves are stored as \(v=0\) (not labeled) for their keypoints

Run:

```bash
python3 sorghum_pipeline/tools/unl_cppd_pose/prepare_unl_cppd_pose_dataset.py
```

This writes an Ultralytics dataset (default):

- `/home/grads/f/fahimehorvatinia/Documents/my_full_project/Dataset/_ultralytics_unl_cppd_pose/`
  - `images/train|val/*.png`
  - `labels/train|val/*.txt`
  - `leaf_status/train|val/*.json` (sidecar, preserves `alive/dead` per leaf)
  - `unl_cppd_pose.yaml`

### 2) Train / fine-tune YOLO pose on it

Example (starting from a pretrained Ultralytics pose model):

```bash
yolo pose train model=yolov8n-pose.pt data=/home/grads/f/fahimehorvatinia/Documents/my_full_project/Dataset/_ultralytics_unl_cppd_pose/unl_cppd_pose.yaml imgsz=960 epochs=100 batch=8
```

If you already have a pose checkpoint (your “YOLO26 pose”), replace `model=...` with that `.pt`.

### 3) Auto-annotate (pseudo-label) Mullet images

After training, run:

```bash
python3 sorghum_pipeline/tools/unl_cppd_pose/predict_mullet_keypoints.py \\
  --model /path/to/best.pt \\
  --images /home/grads/f/fahimehorvatinia/Documents/my_full_project/Main_Pipeline/Mullet_with_white_bg \\
  --out /home/grads/f/fahimehorvatinia/Documents/my_full_project/Main_Pipeline/Mullet_keypoints_predictions
```

This creates one JSON per image with predicted `base` + `leaf_i collar/tip` keypoints.

