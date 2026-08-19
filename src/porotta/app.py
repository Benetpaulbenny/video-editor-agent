import gradio as gr


def show_video_count(videos: list[str] | None) -> str:
    return f"{len(videos or [])} video(s) uploaded"


def create_interface() -> gr.Blocks:
    with gr.Blocks(title="Video Input") as interface:
        gr.Markdown("# Video Input")
        videos = gr.File(label="Upload videos", file_count="multiple", file_types=["video"], type="filepath")
        result = gr.Textbox(label="Result", interactive=False)
        videos.change(show_video_count, inputs=videos, outputs=result)
    return interface


def main() -> None:
    create_interface().launch()
