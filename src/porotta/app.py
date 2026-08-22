import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from html import escape
from pathlib import Path

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface(\.|$)")
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

import gradio as gr

from .csv_manager import CsvManager
from .chippi import Chippi
from .storage import StorageManager


class VideoManager:
    def __init__(self, storage: StorageManager) -> None:
        self.directory = storage.root / "Video"
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
    CSS = """
.gradio-container{box-sizing:border-box;max-width:1600px!important;overflow-x:hidden;width:100%!important}
.porotta-home-row{align-items:flex-start;gap:16px}
.porotta-home-column{min-width:0}
.porotta-gallery{max-width:100%;overflow:hidden}
.csv-results{box-sizing:border-box;max-width:100%;overflow-x:auto;padding-bottom:12px;width:100%}
.csv-results section{box-sizing:border-box;min-width:0;width:100%}
.csv-results table{border-collapse:collapse;min-width:100%;table-layout:auto;width:max-content}
.csv-results th,.csv-results td{max-width:420px;padding:8px 10px;vertical-align:top;white-space:nowrap}
.csv-results th{white-space:normal}
@media (max-width:768px){.porotta-home-row{flex-direction:column!important}.porotta-home-column{width:100%!important}}
"""

    def __init__(self) -> None:
        self.storage = StorageManager()
        self.storage.reset()
        self.video_manager = VideoManager(self.storage)
        self.csv_manager = CsvManager()
        self.chippi = Chippi(self.csv_manager)

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="Porotta Video Editor") as interface:
            with gr.Column(visible=True) as home_page:
                gr.Markdown("# Video Input")
                with gr.Row(elem_classes="porotta-home-row"):
                    with gr.Column(elem_classes="porotta-home-column"):
                        videos = gr.File(label="Upload videos", file_count="multiple", file_types=["video"], type="filepath")
                        result = gr.Textbox(label="Result", interactive=False)
                    with gr.Column(elem_classes="porotta-home-column"):
                        gr.Textbox(label="Video save path", value=str(self.video_manager.directory), interactive=False)
                        thumbnails = gr.Gallery(label="Thumbnails", columns=3, object_fit="contain", elem_classes="porotta-gallery")
                next_button = gr.Button("Next", variant="primary")
            with gr.Column(visible=False) as next_page:
                gr.Markdown("# Video Data")
                status = gr.HTML()
                csv_tables = gr.HTML()
                clip_analysis_button = gr.Button("Next: Clip Analysis", visible=False, variant="primary")
            with gr.Column(visible=False) as clip_analysis_page:
                gr.Markdown("# Clip Analysis")
                gr.Markdown("Clip analysis will be added here.")
            videos.change(self.video_manager.save_videos, inputs=videos, outputs=[result, thumbnails])
            next_button.click(
                self._open_next_page,
                inputs=videos,
                outputs=[home_page, next_page, status, csv_tables, clip_analysis_button],
            )
            clip_analysis_button.click(
                self._open_clip_analysis,
                outputs=[next_page, clip_analysis_page],
            )
        return interface

    def _open_next_page(self, videos: list[str] | None):
        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            self._status_markup("Preparing CSV files"),
            self.csv_manager.render(videos),
            gr.update(visible=False),
        )
        for status, tables in self.chippi.run(videos):
            complete = status.startswith("Analysis complete")
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                self._status_markup(status),
                tables,
                gr.update(visible=complete),
            )

    def _open_clip_analysis(self):
        return gr.update(visible=False), gr.update(visible=True)

    def _status_markup(self, message: str) -> str:
        complete = message.startswith("Analysis complete")
        state = "complete" if complete else "processing"
        icon = "✓" if complete else ""
        return f"""
<style>
.porotta-status{{align-items:center;border-radius:12px;display:flex;font-size:16px;gap:12px;margin:8px 0;padding:14px 18px}}
.porotta-status.processing{{background:#fff7ed;color:#9a3412}}
.porotta-status.complete{{animation:porotta-complete 1.2s ease-out;background:#f0fdf4;color:#166534}}
.porotta-icon{{align-items:center;border-radius:50%;display:flex;font-weight:700;height:22px;justify-content:center;width:22px}}
.processing .porotta-icon{{animation:porotta-pulse 1.1s infinite;background:#f97316}}
.complete .porotta-icon{{background:#22c55e;color:white}}
@keyframes porotta-pulse{{0%,100%{{box-shadow:0 0 0 0 #fdba74;opacity:.65}}50%{{box-shadow:0 0 0 8px #fed7aa;opacity:1}}}}
@keyframes porotta-complete{{0%{{transform:scale(.98);opacity:.5}}100%{{transform:scale(1);opacity:1}}}}
</style>
<div class="porotta-status {state}">
<span class="porotta-icon">{icon}</span>
<span>{escape(message)}</span>
</div>
"""

    def launch(self) -> None:
        self.build().launch(css=self.CSS)


def main() -> None:
    PorottaApp().launch()
