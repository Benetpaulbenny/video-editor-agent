"""
Convert a video file to an audio file using ffmpeg.
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


class Converter:
    def __init__(self,input_path: str, codec: str = AUDIO_CODEC, bitrate: str = AUDIO_BITRATE, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS, overwrite: bool = OVERWRITE):
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self.codec = codec
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        self.channels = channels
        self.overwrite = overwrite

        self.output_path = None

    def VideoToAudio(self, output_path: str) -> "Converter":
        
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        stream = ffmpeg.input(str(self.input_path))

        stream = ffmpeg.output(stream.audio, str(out_path), acodec=self.codec, audio_bitrate=self.bitrate, ar=self.sample_rate, ac=self.channels)
        stream = ffmpeg.overwrite_output(stream) if self.overwrite else stream

        print(f"Extracting audio")
        ffmpeg.run(stream, quiet=False)
        print("Done.")
        return self