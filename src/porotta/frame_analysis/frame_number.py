import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class FrameSample:
    frame_number: int
    timestamp: float
    frame_path: Path


class FrameNumber:
    def execute(self, video_path: str | Path) -> Iterator[FrameSample]:
        video_path = Path(video_path)
        frame_rate, frame_count = self._get_metadata(video_path)
        for second in range(math.ceil(frame_count / frame_rate)):
            frame_number = round(second * frame_rate) + 1
            frame_path = self._extract_frame(video_path, second, frame_number)
            yield FrameSample(frame_number, float(second), frame_path)

    def _get_metadata(self, video_path: Path) -> tuple[float, int]:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries", "stream=avg_frame_rate,nb_read_frames",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream = json.loads(result.stdout)["streams"][0]
        return float(Fraction(stream["avg_frame_rate"])), int(stream["nb_read_frames"])

    def _extract_frame(self, video_path: Path, second: int, frame_number: int) -> Path:
        frame_path = Path("/tmp") / f"porotta_frame_{video_path.stem}_{frame_number}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ss", str(second), "-frames:v", "1", str(frame_path)],
            capture_output=True,
            check=True,
        )
        return frame_path
