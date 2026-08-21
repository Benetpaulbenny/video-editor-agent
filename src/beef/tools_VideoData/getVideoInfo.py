"""
Get resolution and FPS of a video file using ffprobe.

Reports both r_frame_rate and avg_frame_rate (per your VFR learning: these
can differ, and the gap indicates variable frame rate footage).
"""

import ffmpeg
from pathlib import Path
from fractions import Fraction

# ---------------- CONFIG ----------------
INPUT_VIDEO = "media/inputs/R-Clip_V1-0001.mp4"   # path to video file
# -----------------------------------------

class GetVideoInfo:
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self.width = None
        self.height = None
        self.r_fps = None
        self.avg_fps = None

    def analyze(self) -> "getVideoInfo":
        probe = ffmpeg.probe(str(self.in_path))
        video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)

        width = video_stream.get("width")
        height = video_stream.get("height")
        r_frame_rate = video_stream.get("r_frame_rate")
        avg_frame_rate = video_stream.get("avg_frame_rate")

        r_fps = float(Fraction(r_frame_rate)) if r_frame_rate else None
        avg_fps = float(Fraction(avg_frame_rate)) if avg_frame_rate else None

        is_vfr = r_fps is not None and avg_fps is not None and abs(r_fps - avg_fps) > 0.01

        return {"width": width, "height": height, "resolution": f"{width}x{height}", "r_frame_rate": r_frame_rate, "r_fps": r_fps, "avg_frame_rate": avg_frame_rate, "avg_fps": avg_fps, "is_vfr": is_vfr}

    def __str__(self) -> str:
        return