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
5. Face landmark
6. Head pose
7. Pose estimation
8. Object detection
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

Face landmark, Head pose, Pose estimation, Object detection, Segmentation, OCR, Scene classification, and Camera analysis are reserved for future layers. Their current row values are `0` placeholders and must not be interpreted as measured results.

## LLM interpretation rules

- Use the CSV header to determine the order of every JSON-array value.
- Parse grouped cells as JSON before using their values.
- Do not infer missing feature results from placeholder `0` values.
- Treat Layer 1 and Layer 2 as objective measurements, not editing instructions.
- Treat face embeddings as numerical similarity representations, not names or personal identities.
- Treat `face_id` as frame-local unless a future tracking layer defines persistent identity logic.
- Use the timestamp and frame number to locate the original sampled frame.
- Expect one CSV per video and process rows in order.
- The latest row may appear while analysis is still running; do not assume the CSV is complete until processing finishes.

## Current design choices

- CSV is the primary per-video record because it is easy to inspect and display in the Gradio page.
- Structured values are stored as key-free JSON arrays because their field order is documented in the header.
- Processing is serial so every layer sees the same frame and realtime updates remain ordered.
- Chippi is the single layer orchestrator so future analysis modules can be added without changing the interface flow.
