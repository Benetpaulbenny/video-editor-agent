import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

VIDEO_DIR = Path("~/Downloads/Pookie/Video").expanduser()
CSV_DIR = Path("~/Downloads/Pookie/video_csv").expanduser()
THUMBNAIL_DIR = Path(tempfile.gettempdir()) / "porotta_thumbnails"
CSV_PATH = CSV_DIR / "videos.csv"
CSV_COLUMNS = [f"column_{index}" for index in range(1, 13)]


def unique_video_path(video: Path) -> Path:
    target = VIDEO_DIR / video.name
    index = 1
    while target.exists():
        target = VIDEO_DIR / f"{video.stem}_{index}{video.suffix}"
        index += 1
    return target


def create_thumbnail(video: Path, index: int) -> Path:
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    thumbnail = THUMBNAIL_DIR / f"{index}_{video.stem}.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-frames:v", "1", "-vf", "scale=320:-1", str(thumbnail)],
        capture_output=True,
        check=False,
    )
    return thumbnail


def save_videos(videos: list[str] | None) -> tuple[str, list[str]]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    saved_videos = []
    thumbnails = []
    for index, video in enumerate(videos or [], start=1):
        source = Path(video)
        target = unique_video_path(source)
        shutil.copy2(source, target)
        saved_videos.append(target)
        thumbnail = create_thumbnail(target, index)
        if thumbnail.exists():
            thumbnails.append(str(thumbnail))
    return f"{len(saved_videos)} video(s) saved", thumbnails


def create_csv() -> list[list[int]]:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    row = [0] * 12
    with CSV_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        writer.writerow(row)
    return [row]


def open_next_page() -> tuple[gr.update, gr.update, list[list[int]], str]:
    data = create_csv()
    return gr.update(visible=False), gr.update(visible=True), data, str(CSV_PATH)


def create_interface() -> gr.Blocks:
    with gr.Blocks(title="Porotta Video Editor") as interface:
        with gr.Column(visible=True) as home_page:
            gr.Markdown("# Video Input")
            with gr.Row():
                with gr.Column():
                    videos = gr.File(label="Upload videos", file_count="multiple", file_types=["video"], type="filepath")
                    result = gr.Textbox(label="Result", interactive=False)
                with gr.Column():
                    path = gr.Textbox(label="Video save path", value=str(VIDEO_DIR), interactive=False)
                    thumbnails = gr.Gallery(label="Thumbnails", columns=3, object_fit="contain")
            next_button = gr.Button("Next", variant="primary")
        with gr.Column(visible=False) as next_page:
            gr.Markdown("# Video Data")
            csv_path = gr.Textbox(label="CSV save path", interactive=False)
            table = gr.Dataframe(headers=CSV_COLUMNS, datatype=["number"] * 12, interactive=False)
        videos.change(save_videos, inputs=videos, outputs=[result, thumbnails])
        next_button.click(open_next_page, outputs=[home_page, next_page, table, csv_path])
    return interface


def main() -> None:
    create_interface().launch()
