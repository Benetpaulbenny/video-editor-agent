import json
from pathlib import Path

from .csv_manager import CsvManager
from .frame_analysis.frame_number import FrameNumber
from .frame_analysis.image_quality import ImageQuality


class Chippi:
    def __init__(self, csv_manager: CsvManager) -> None:
        self.csv_manager = csv_manager
        self.frame_number = FrameNumber()
        self.image_quality = ImageQuality()

    def run(self, videos: list[str] | None):
        videos = videos or []
        for video in videos:
            self.csv_manager.reset(video)
        yield "Frame analysis started", self.csv_manager.render(videos)
        for video in videos:
            for serial_number, sample in enumerate(self.frame_number.execute(Path(video)), start=1):
                quality = self.image_quality.execute(sample.frame_path)
                frame_data = [serial_number, sample.frame_number, sample.timestamp]
                row = [json.dumps(frame_data, separators=(",", ":")), json.dumps(quality, separators=(",", ":"))]
                row.extend([0] * (len(self.csv_manager.columns) - len(row)))
                self.csv_manager.append_row(video, row)
                yield f"Processed frame {sample.frame_number} from {Path(video).name}", self.csv_manager.render(videos)
