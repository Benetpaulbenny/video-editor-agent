import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

from .csv_manager import CsvManager


class VideoManager:
    def __init__(self, directory: str = "~/Downloads/Pookie/Video") -> None:
        self.directory = Path(directory).expanduser()
        self.thumbnail_directory = Path(tempfile.gettempdir()) / "porotta_thumbnails"

    def save_videos(self, videos: list[str] | None) -> tuple[str, list[str]]:
        self.directory.mkdir(parents=True, exist_ok=True)
        saved_videos = []
        thumbnails = []
        for index, video in enumerate(videos or [], start=1):
            source = Path(video)
            target = self._unique_path(source)
            shutil.copy2(source, target)
            saved_videos.append(target)
            thumbnail = self._create_thumbnail(target, index)
            if thumbnail.exists():
                thumbnails.append(str(thumbnail))
        return f"{len(saved_videos)} video(s) saved", thumbnails

    def _unique_path(self, video: Path) -> Path:
        target = self.directory / video.name
        index = 1
        while target.exists():
            target = self.directory / f"{video.stem}_{index}{video.suffix}"
            index += 1
        return target

    def _create_thumbnail(self, video: Path, index: int) -> Path:
        self.thumbnail_directory.mkdir(parents=True, exist_ok=True)
        thumbnail = self.thumbnail_directory / f"{index}_{video.stem}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-frames:v", "1", "-vf", "scale=320:-1", str(thumbnail)],
            capture_output=True,
            check=False,
        )
        return thumbnail


class PorottaApp:
    def __init__(self) -> None:
        self.video_manager = VideoManager()
        self.csv_manager = CsvManager()

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="Porotta Video Editor") as interface:
            with gr.Column(visible=True) as home_page:
                gr.Markdown("# Video Input")
                with gr.Row():
                    with gr.Column():
                        videos = gr.File(label="Upload videos", file_count="multiple", file_types=["video"], type="filepath")
                        result = gr.Textbox(label="Result", interactive=False)
                    with gr.Column():
                        gr.Textbox(label="Video save path", value=str(self.video_manager.directory), interactive=False)
                        thumbnails = gr.Gallery(label="Thumbnails", columns=3, object_fit="contain")
                next_button = gr.Button("Next", variant="primary")
            with gr.Column(visible=False) as next_page:
                gr.Markdown("# Video Data")
                csv_path = gr.Textbox(label="CSV save path", interactive=False)
                table = gr.Dataframe(headers=self.csv_manager.columns, datatype=["number"] * 12, interactive=False)
            videos.change(self.video_manager.save_videos, inputs=videos, outputs=[result, thumbnails])
            next_button.click(self._open_next_page, outputs=[home_page, next_page, table, csv_path])
        return interface

    def _open_next_page(self) -> tuple:
        data = self.csv_manager.load_or_create()
        return gr.update(visible=False), gr.update(visible=True), data, str(self.csv_manager.path)

    def launch(self) -> None:
        self.build().launch()


def main() -> None:
    PorottaApp().launch()
