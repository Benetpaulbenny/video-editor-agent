# Porotta

## Purpose

Porotta is a serial video-analysis pipeline with a Gradio interface. It accepts multiple videos, saves them locally, samples one frame per second, analyzes each frame layer by layer, and updates one CSV file per video in realtime.

## Running

```bash
uv run porotta
```

On startup, the application clears the contents of:

```text
~/Downloads/Pookie
```

The folder itself is preserved. This keeps every run isolated and prevents stale videos, thumbnails, and CSV files from being mixed with a new run.

## Architecture

- `app.py` builds the Gradio interface and controls the page flow.
- `storage.py` resets the Pookie workspace at startup.
- `app.py` stores uploaded videos in `~/Downloads/Pookie/Video` and creates thumbnails.
- `chippi.py` is the internal orchestration layer. It is called by `app.py` and is not a separate command.
- `frame_analysis/` contains one class per analysis feature.
- `csv_manager.py` creates, updates, and renders one CSV per video.

The code follows an object-oriented structure so each analysis layer can be developed independently and called by Chippi in a fixed order.

## Processing flow

```text
Upload videos
    ↓
Save videos and create thumbnails
    ↓
Press Next
    ↓
Process video 1 serially
    ↓
Extract one frame per second
    ↓
Run the implemented layers on the same frame
    ↓
Write one CSV row and update the UI
    ↓
Continue to the next frame, then the next video
```

Serial execution was selected so every layer receives the exact same frame before processing moves forward. The generator-based Chippi pipeline yields after every frame, which enables realtime UI and CSV updates.

FFmpeg and ffprobe are used for the current frame extraction implementation because they support many video formats and provide reliable frame metadata. They can be optimized later without changing the layer interface.

## CSV output

Each video receives a separate file:

```text
~/Downloads/Pookie/video_csv/<video_name>.csv
```

The CSV keeps the original 12-column feature layout. Structured layer values are stored as JSON arrays inside their grouped column, while the header documents the order of each value.

### Frame number

```text
Frame_number[S.No, Frame number, Timestamp]
```

Example value:

```json
[1, 120, 4.0]
```

### Image quality

```text
Image_quality[Brightness, Contrast, Blur score, Noise]
```

Example value:

```json
[128.4, 61.2, 0.82, 0.13]
```

### Color analysis

```text
Color_analysis[Color temp[kelvin, classification], Dominant colors[r, g, b, percentage], Avg RGB[r, g, b], Saturation[mean, median, low_saturation_ratio, high_saturation_ratio]]
```

Example value:

```json
[
  [5337, "neutral"],
  [[32, 22, 20, 0.3873], [107, 71, 75, 0.3266]],
  [107.3195, 84.6321, 80.1282],
  [0.37, 0.3765, 0.234, 0.0284]
]
```

### Face detection

```text
Face detection[face_id, bbox[x1, y1, x2, y2], confidence, embedding[vector]]
```

Face detection uses InsightFace with the `buffalo_l` model and its RetinaFace detector. Each detected face records its bounding box, confidence, and ArcFace embedding vector directly in the CSV cell.

The remaining feature columns are present for the planned layers and currently contain `0` placeholders.

### Face landmarks

```text
Face landmark[face_id, eyes[left, right], nose, jaw, mouth, normalized[eyes[left, right], nose, jaw, mouth], landmark_confidence[detector confidence]]
```

Layer 4 uses InsightFace's 68-point landmark output and groups it into eyes, nose, jaw, and mouth regions. Coordinates are stored in pixels and normalized by frame width and height. If the 68-point output is unavailable, the five-point keypoint output is used for eyes, nose, and mouth, while jaw remains empty. InsightFace does not expose an independent confidence for every landmark, so the detector confidence is preserved as the explicitly defined overall landmark confidence.

### Head pose

```text
Head pose[face_id, yaw, pitch, roll, confidence]
```

Layer 5 uses InsightFace's 3D facial landmarks and geometric pose solving. Yaw, pitch, and roll are reported in degrees. Negative yaw means looking left, positive yaw means looking right; negative pitch means looking down, positive pitch means looking up; negative roll means tilted left, positive roll means tilted right. Confidence is the detector confidence reduced by the geometric reprojection error. No separate pose model is used.

### Pose estimation

```text
Pose estimation[person_id, confidence, keypoints[17x[x, y, confidence]], hands[left[x, y, confidence], right[x, y, confidence]], body_orientation[rotation, direction], movement[moving, direction, speed]]
```

Layer 6 uses YOLO11 Pose for person detection and the standard 17 body keypoints. StrongSORT assigns person IDs across sampled frames. Each person value is ordered as:

```text
[person_id, confidence, keypoints, hands, body_orientation, movement]
```

Keypoints follow the standard YOLO order from nose through ankles. Hands reuse the left and right wrist points. Body orientation is derived from shoulder and hip geometry. Movement is derived from tracked bounding-box centers between the one-second samples, so speed is in pixels per second and is intentionally a coarse sampled estimate.

StrongSORT is configured with a one-observation confirmation threshold so a detected person can appear in the CSV immediately. A frame with no detected person correctly stores an empty list.

## Current layers

### Layer 1 — Image Quality

OpenCV calculates brightness, global contrast, Laplacian-based blur score, and estimated noise score. This layer measures technical image quality and does not interpret the scene.

### Layer 2 — Color Analysis

OpenCV calculates estimated color temperature, dominant colors using K-means, average RGB, and saturation statistics. This layer records visual color characteristics without making editing decisions.

### Layer 3 — Face Detection

InsightFace with `buffalo_l` provides RetinaFace detection, bounding boxes, confidence scores, and ArcFace embeddings. Layer 3 does not repeat Layer 1 image-quality calculations. The model is initialized lazily when the first frame reaches this layer so application startup remains lightweight.

### Layer 4 — Face Landmarks

InsightFace landmark output provides geometric facial points for each detected face. Layer 4 records eyes, nose, jaw, and mouth coordinates in pixels and normalized coordinates. It does not calculate head orientation, emotion, or other interpretation; those belong to later layers.

### Layer 5 — Head Pose

InsightFace's 3D landmark geometry is used with geometric pose solving to calculate yaw, pitch, roll, and confidence. Layer 5 does not calculate emotion, identity, or image quality.

### Layer 6 — Pose Estimation

YOLO11 Pose detects people and supplies body keypoints with coordinates and confidence. StrongSORT maintains person IDs between sampled frames. Body orientation and movement are derived from those outputs. The layer does not infer emotion, identity by name, or editing decisions.

### Layer 7 — Object Detection

YOLO11 object detection records every supported object class, bounding box, normalized bounding box, confidence, and size. A separate StrongSORT tracker assigns object IDs across sampled frames. Object detection does not decide whether an object is important, what a person is doing, or what editing action should follow.

The YOLO object confidence threshold is `0.20`. StrongSORT uses a stricter `0.70` confidence threshold. If StrongSORT cannot return a track for a valid current-frame detection, the detection is still recorded with a local fallback ID so it is not silently lost.

### Layer 8 — Segmentation

Layer 8 combines SAM 2 and SegFormer. SAM 2 receives the Layer 7 object boxes as prompts and produces precise person/object masks. SegFormer supplies semantic sky and ground masks. Foreground is the union of SAM 2 object masks, and background is its complement. Every mask is stored as a named `mask_rle` object containing the original `[height, width]` and a comma-separated column-major `counts` string. Person records reuse the existing pose/tracking `person_id` and include `area_ratio`.

### Layer 9 — OCR

Layer 9 uses PaddleOCR PP-OCRv5 server models to detect and recognize visible text. Each detected text record stores the exact recognized text, polygon, bounding box, OCR confidence, and bounding-region area ratio. Recognition results are filtered at `0.5` confidence to reduce obvious false positives. Language is stored only when PaddleOCR supplies it; the configured English model is not treated as language detection. This layer records objective text data and does not interpret meaning, importance, subtitles, logos, or editing decisions.

## Design decisions

- JSON arrays are used inside grouped CSV cells so nested measurements stay together while the CSV remains easy to inspect.
- Grouped headers document the position of every value, avoiding repeated keys in every row.
- Separate CSV files keep each video’s analysis independent.
- Realtime updates expose progress immediately and make long video analysis observable.
- Placeholder columns preserve the planned schema while later layers are implemented one at a time.
- Blur and noise normalization references are provisional and can be calibrated against real footage later.
- InsightFace was selected instead of Haar Cascade, dlib, YOLO, or DeepFace because RetinaFace plus ArcFace matches the required face detection and identity-representation responsibilities.
- Embeddings are stored directly in the CSV as JSON arrays so each video CSV is self-contained and easy for the agent to read.
- Layer 4 reuses Layer 3's detected face objects so the same frame is not passed through the InsightFace detector twice.
- Layer 5 reuses the same InsightFace 3D landmarks from Layer 3 instead of adding a separate pose model.
- Layer 6 uses YOLO11 Pose for body structure and StrongSORT for temporal identity because a single frame cannot reliably provide movement direction.
- Layer 6 is evaluated on the existing one-frame-per-second stream so every layer remains serial and receives the same frame.
- StrongSORT confirms a track after one observation because the output is sampled once per second and delaying confirmation would hide early valid detections.
- Layer 7 uses a separate StrongSORT tracker from Layer 6 so person and object identities remain independent.
- Layer 7 keeps all YOLO-supported classes instead of restricting the detector to a small manually selected category list.
- Layer 7 uses separate YOLO (`0.20`) and StrongSORT (`0.70`) confidence thresholds so detector recall and tracker strictness can be tuned independently.
- Layer 7 records unmatched YOLO detections with local fallback IDs when StrongSORT cannot confirm a track on a sparse one-second sample.
- SAM 2 is pinned as an official Git dependency so a fresh uv environment has the required implementation. Its Hugging Face weights are downloaded on first use, or a local checkpoint/config can be supplied through `POROTTA_SAM2_CHECKPOINT` and `POROTTA_SAM2_CONFIG`.
- SAM 2 loads on CPU by default because the application processes the existing InsightFace and YOLO stack on CPU and should not fail from GPU memory contention.
- SAM 2 is prompted with Layer 7 boxes so its precise masks stay aligned with the object IDs already recorded in the same frame.
- SegFormer is used for sky and ground because SAM 2 segments prompted regions but does not assign semantic region labels by itself.
- Masks use run-length encoding instead of raw per-pixel boolean arrays to keep CSV cells manageable.
- PaddleOCR is used for both text detection and recognition because Layer 9 records objective text regions before any semantic interpretation layer.
- OCR polygons are retained instead of reducing text regions to bounding boxes alone.
- InsightFace and ONNX Runtime startup logs are suppressed so the terminal shows major application output while real errors remain visible.
