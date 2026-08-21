# Porotta LLM Data Guide

The companion file `achu.json` is the machine-readable schema. Use this document for explanations and `achu.json` for validation and exact array contracts.

## What this project does

Porotta is a serial video-analysis system. It accepts multiple videos through a Gradio interface, samples one frame per second, runs analysis layers on the same frame, and writes one CSV file per video.

The system records objective measurements. It does not currently make editing decisions, identify people by name, or recommend changes to the video.

## How to run

```bash
uv run porotta
```

Every run starts by clearing the contents of:

```text
~/Downloads/Pookie
```

The Pookie folder is preserved, but all previous videos, CSV files, and other contents inside it are removed. Treat every run as a fresh analysis session.

## File locations

Uploaded videos:

```text
~/Downloads/Pookie/Video
```

Per-video CSV files:

```text
~/Downloads/Pookie/video_csv/<video_stem>.csv
```

Temporary thumbnails:

```text
/tmp/porotta_thumbnails
```

There are currently no separate embedding files. Face embeddings are stored inside the Face Detection CSV cell.

## Execution order

Videos are processed serially. The complete sequence is:

```text
video 1
  frame 1
    Frame Number
    Image Quality
    Color Analysis
    Face Detection
    write one CSV row
    update the Gradio page
  frame 2
  ...
video 2
  frame 1
  ...
```

Only one frame is active at a time. Every active layer receives the same extracted frame before the next frame is selected.

`ffprobe` reads frame metadata and `ffmpeg` extracts one frame per second. `Chippi` is the internal orchestrator called by `app.py`; it is not a separate command.

The pipeline is generator-based. After each frame is analyzed and written, Chippi yields an update, so the CSV table and analysis status appear in the UI immediately.

## CSV structure

Every CSV has 12 top-level columns. Grouped headers describe the order of values stored in a JSON array inside one cell.

```text
1. Frame_number[S.No, Frame number, Timestamp]
2. Image_quality[Brightness, Contrast, Blur score, Noise]
3. Color_analysis[Color temp[kelvin, classification], Dominant colors[r, g, b, percentage], Avg RGB[r, g, b], Saturation[mean, median, low_saturation_ratio, high_saturation_ratio]]
4. Face detection[face_id, bbox[x1, y1, x2, y2], confidence, embedding[vector]]
5. Face landmark[face_id, eyes[left, right], nose, jaw, mouth, normalized[eyes[left, right], nose, jaw, mouth], landmark_confidence[detector confidence]]
6. Head pose[face_id, yaw, pitch, roll, confidence]
7. Pose estimation[person_id, confidence, keypoints[17x[x, y, confidence]], hands[left[x, y, confidence], right[x, y, confidence]], body_orientation[rotation, direction], movement[moving, direction, speed]]
8. Object detection[object_id, class, bbox[x1, y1, x2, y2], bbox_normalized[x1, y1, x2, y2], confidence, size[width, height, area_ratio]]
9. Segmentation
10. OCR
11. Scene classification
12. Camera analysis
```

The CSV library quotes cells automatically when their JSON contains commas. An LLM or parser must first read the CSV cell as a string and then parse that string as JSON.

## Column value contracts

### Frame Number

Value order:

```json
[serial_number, exact_frame_number, timestamp_seconds]
```

Example:

```json
[1, 120, 4.0]
```

The serial number starts at `1` for each video CSV. The exact frame number is calculated from the video frame rate. The timestamp is the sampled second.

### Image Quality

Tool: OpenCV.

Value order:

```json
[mean_brightness, global_contrast, blur_score, noise_score]
```

Example:

```json
[128.4, 61.2, 0.82, 0.13]
```

These measurements describe technical frame quality. They do not describe the scene. Blur and noise normalization references are provisional and may be calibrated later.

### Color Analysis

Tool: OpenCV.

Value order:

```json
[
  [kelvin, classification],
  [[r, g, b, percentage], ...],
  [r, g, b],
  [mean_saturation, median_saturation, low_saturation_ratio, high_saturation_ratio]
]
```

Example:

```json
[
  [5337, "neutral"],
  [[32, 22, 20, 0.3873], [107, 71, 75, 0.3266]],
  [107.3195, 84.6321, 80.1282],
  [0.37, 0.3765, 0.234, 0.0284]
]
```

The color temperature is an estimated perceived temperature, not a physical camera white-balance reading. Dominant colors are produced with OpenCV K-means clustering.

### Face Detection

Tool: InsightFace with the `buffalo_l` model. The detector is RetinaFace-based and the face representation is produced by ArcFace.

Value order for the cell:

```json
[
  [face_id, [x1, y1, x2, y2], confidence, embedding_vector],
  ...
]
```

Example:

```json
[
  [
    "face_001",
    [1261.2626, 382.5294, 1299.6416, 432.6128],
    0.7439,
    [0.12, -0.44, 0.87]
  ]
]
```

The real embedding is a large numerical vector. It is stored directly in the CSV as a JSON array. `face_001` is a detection label for that frame; it is not a confirmed person identity and is not persistent tracking across frames.

If no face is detected, the Face Detection cell is an empty list:

```json
[]
```

### Future columns

Segmentation, OCR, Scene classification, and Camera analysis are reserved for future layers. Their current row values are `0` placeholders and must not be interpreted as measured results.

### Face Landmarks

Tool: InsightFace landmark output from the same face-analysis result used by Face Detection.

Value order for the cell:

```json
[
  [face_id, [eyes_left, eyes_right], nose, jaw, mouth, [normalized_eyes, normalized_nose, normalized_jaw, normalized_mouth], landmark_confidence],
  ...
]
```

The implementation uses InsightFace's 68-point landmarks for semantic regions and falls back to five-point keypoints when necessary. The normalized coordinates divide `x` by frame width and `y` by frame height. The landmark confidence is the detector confidence because InsightFace does not expose a separate confidence for each landmark. Layer 4 does not calculate yaw, pitch, roll, or emotion.

### Head Pose

Tool: InsightFace 3D facial landmarks with geometric pose solving.

Value order for the cell:

```json
[
  [face_id, yaw, pitch, roll, confidence],
  ...
]
```

Yaw, pitch, and roll are degrees. Negative yaw means looking left; positive yaw means looking right. Negative pitch means looking down; positive pitch means looking up. Negative roll means tilted left; positive roll means tilted right. Confidence combines the InsightFace detector confidence with the solve-projection reprojection error. No separate pose model is used.

### Pose Estimation

Tools: YOLO11 Pose and StrongSORT.

The value order for the cell is:

```json
[
  [person_id, confidence, keypoints, hands, body_orientation, movement],
  ...
]
```

`keypoints` always follows this order:

```text
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

Every keypoint is `[x, y, confidence]`. `hands` is `[left_wrist, right_wrist]`. `body_orientation` is `[rotation, direction]`, derived from the shoulder and hip centers; its direction is `left`, `right`, `center`, or `unknown`. `movement` is `[moving, direction, speed]`, where direction is `left`, `right`, `up`, `down`, or `stationary`, and speed is pixels per second.

StrongSORT provides the temporal person ID. Movement is calculated between the one-second sampled frames, so it is a coarse movement estimate rather than a full-frame-rate measurement. The first observation of a person is reported as stationary with speed `0.0` because no previous position exists.

The tracker uses a one-observation confirmation threshold so a detected person can be recorded immediately. If no person is detected in a sampled frame, the Pose Estimation cell is `[]`.

### Object Detection

Tools: YOLO11 object detection and StrongSORT.

The value order for the cell is:

```json
[
  [object_id, class, bbox, bbox_normalized, confidence, size],
  ...
]
```

`bbox` is `[x1, y1, x2, y2]` in pixels. `bbox_normalized` uses the same order with each coordinate divided by frame width or height and clamped to `[0, 1]`. `size` is `[width, height, area_ratio]`, where `area_ratio` is the bounding-box area divided by the frame area. YOLO-supported classes are retained; the required examples such as `person`, `vehicle`, `building`, `tree`, `phone`, and `sign` are not an exclusive filter.

StrongSORT provides `object_id` across the sampled frames. A frame with no tracked objects stores `[]`. This layer records detections only and does not decide object importance, activity, scene meaning, or editing actions.

The YOLO confidence threshold is `0.20` and the StrongSORT confidence threshold is `0.70`. If StrongSORT does not return a track for a current-frame YOLO detection, the detection is retained with a local fallback ID. Fallback IDs are unique within each frame and are matched to recent same-class boxes when overlap is sufficient.

## LLM interpretation rules

- Use the CSV header to determine the order of every JSON-array value.
- Parse grouped cells as JSON before using their values.
- Do not infer missing feature results from placeholder `0` values.
- Treat Layer 1 and Layer 2 as objective measurements, not editing instructions.
- Treat face embeddings as numerical similarity representations, not names or personal identities.
- Treat `face_id` as frame-local. Treat `person_id` as persistent only within the current video analysis run because it comes from StrongSORT.
- Treat `object_id` as persistent only within the current video analysis run because it comes from the separate object StrongSORT tracker.
- Use the timestamp and frame number to locate the original sampled frame.
- Expect one CSV per video and process rows in order.
- The latest row may appear while analysis is still running; do not assume the CSV is complete until processing finishes.

## Current design choices

- CSV is the primary per-video record because it is easy to inspect and display in the Gradio page.
- Structured values are stored as key-free JSON arrays because their field order is documented in the header.
- Processing is serial so every layer sees the same frame and realtime updates remain ordered.
- Chippi is the single layer orchestrator so future analysis modules can be added without changing the interface flow.
- Layer 6 is lazy-loaded so the app can start without loading YOLO11 and StrongSORT models until pose analysis is reached.
- Layer 7 uses its own YOLO11 model and StrongSORT tracker so object IDs cannot be confused with person IDs from Layer 6.
