"""
Convert a video file to an audio file using ffmpeg.

Run with: uv run src/beef/video_to_audio.py
"""

import ffmpeg
from pathlib import Path


INPUT_VIDEO = "media/inputs/R-Clip_V1-0001.mp4"  
F_NAME = Path(INPUT_VIDEO).stem
OUTPUT_PATH = f"media/outputs/V2A_{F_NAME}.wav"  
AUDIO_CODEC = "pcm_s16le"      # e.g. libmp3lame (mp3), pcm_s16le (wav), aac (m4a)
AUDIO_BITRATE = "192k"          # ignored for lossless codecs like pcm_s16le
SAMPLE_RATE = 44100             # Hz
CHANNELS = 2                    # 1 = mono, 2 = stereo
OVERWRITE = True               


def video_to_audio(input_path: str, output_path: str, codec: str = AUDIO_CODEC, bitrate: str = AUDIO_BITRATE, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS, overwrite: bool = OVERWRITE) -> None:
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stream = ffmpeg.input(str(in_path))

    stream = ffmpeg.output(stream.audio, str(out_path), acodec=codec, audio_bitrate=bitrate, ar=sample_rate, ac=channels)
    stream = ffmpeg.overwrite_output(stream) if overwrite else stream

    print(f"Extracting audio: {in_path} -> {out_path}")
    ffmpeg.run(stream, quiet=False)
    print("Done.")


if __name__ == "__main__":
    video_to_audio(INPUT_VIDEO, OUTPUT_PATH)