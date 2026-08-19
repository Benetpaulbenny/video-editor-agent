import csv
from pathlib import Path


class CsvManager:
    def __init__(self, directory: str = "~/Downloads/Pookie/video_csv") -> None:
        self.directory = Path(directory).expanduser()
        self.path = self.directory / "videos.csv"
        self.columns = [
            "Frame_number",
            "Image_quality",
            "Color_analysis",
            "Face detection",
            "Face landmark",
            "Head pose",
            "Pose estimation",
            "Object detection",
            "Segmentation",
            "OCR",
            "Scene classification",
            "Camera analysis",
        ]

    def load_or_create(self) -> list[list[int]]:
        if not self.path.exists() or not self._has_expected_columns():
            self._create()
        return self._read()

    def _has_expected_columns(self) -> bool:
        with self.path.open(newline="") as file:
            return next(csv.reader(file), []) == self.columns

    def _create(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.columns)
            writer.writerow([0] * len(self.columns))

    def _read(self) -> list[list[int]]:
        with self.path.open(newline="") as file:
            rows = list(csv.reader(file))
        return [[int(value) for value in row] for row in rows[1:] if len(row) == len(self.columns)]
