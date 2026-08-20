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
Run Layer 1 and Layer 2 on the same frame
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

## Current layers

### Layer 1 — Image Quality

OpenCV calculates brightness, global contrast, Laplacian-based blur score, and estimated noise score. This layer measures technical image quality and does not interpret the scene.

### Layer 2 — Color Analysis

OpenCV calculates estimated color temperature, dominant colors using K-means, average RGB, and saturation statistics. This layer records visual color characteristics without making editing decisions.

### Layer 3 — Face Detection

InsightFace with `buffalo_l` provides RetinaFace detection, bounding boxes, confidence scores, and ArcFace embeddings. Layer 3 does not repeat Layer 1 image-quality calculations. The model is initialized lazily when the first frame reaches this layer so application startup remains lightweight.

## Design decisions

- JSON arrays are used inside grouped CSV cells so nested measurements stay together while the CSV remains easy to inspect.
- Grouped headers document the position of every value, avoiding repeated keys in every row.
- Separate CSV files keep each video’s analysis independent.
- Realtime updates expose progress immediately and make long video analysis observable.
- Placeholder columns preserve the planned schema while later layers are implemented one at a time.
- Blur and noise normalization references are provisional and can be calibrated against real footage later.
- InsightFace was selected instead of Haar Cascade, dlib, YOLO, or DeepFace because RetinaFace plus ArcFace matches the required face detection and identity-representation responsibilities.
- Embeddings are stored directly in the CSV as JSON arrays so each video CSV is self-contained and easy for the agent to read.
