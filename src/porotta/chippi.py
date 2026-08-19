from pathlib import Path

from .csv_manager import CsvManager
from .frame_analysis.frame_number import FrameNumber


class Chippi:
    def __init__(self, csv_manager: CsvManager) -> None:
        self.csv_manager = csv_manager
        self.frame_number = FrameNumber()

    def run(self, videos: list[str] | None):
        videos = videos or []
        for video in videos:
            self.csv_manager.reset(video)
        yield "Frame analysis started", self.csv_manager.render(videos)
        for video in videos:
            for sample in self.frame_number.execute(Path(video)):
                row = [sample.frame_number] + [0] * (len(self.csv_manager.columns) - 1)
                self.csv_manager.append_row(video, row)
                yield f"Processed frame {sample.frame_number} from {Path(video).name}", self.csv_manager.render(videos)
