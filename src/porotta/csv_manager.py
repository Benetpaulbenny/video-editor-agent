import csv
from html import escape
from pathlib import Path


class CsvManager:
    def __init__(self, directory: str = "~/Downloads/Pookie/video_csv") -> None:
        self.directory = Path(directory).expanduser()
        self.columns = [
            "Frame_number[S.No, Frame number, Timestamp]",
            "Image_quality[Brightness, Contrast, Blur score, Noise]",
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

    def prepare(self, video_path: str | Path) -> Path:
        path = self.path_for(video_path)
        if not path.exists() or not self._has_expected_columns(path):
            self._create(path)
        return path

    def path_for(self, video_path: str | Path) -> Path:
        return self.directory / f"{Path(video_path).stem}.csv"

    def append_row(self, video_path: str | Path, row: list[object]) -> None:
        path = self.prepare(video_path)
        with path.open("a", newline="") as file:
            csv.writer(file).writerow(row)

    def reset(self, video_path: str | Path) -> Path:
        path = self.path_for(video_path)
        self._create(path)
        return path

    def render(self, videos: list[str] | None) -> str:
        sections = [self._render_video(video) for video in videos or []]
        return "<div class='csv-results'>" + "".join(sections) + "</div>"

    def _create(self, path: Path) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.columns)

    def _has_expected_columns(self, path: Path) -> bool:
        with path.open(newline="") as file:
            return next(csv.reader(file), []) == self.columns

    def _render_video(self, video_path: str | Path) -> str:
        path = self.prepare(video_path)
        with path.open(newline="") as file:
            rows = list(csv.reader(file))
        header = "".join(f"<th>{escape(value)}</th>" for value in rows[0])
        body = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in rows[1:])
        return f"<section><h3>{escape(Path(video_path).name)}</h3><p>{escape(str(path))}</p><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></section>"
