from sarvamai import SarvamAI
import json
import tempfile
from pathlib import Path
import os
from dotenv import load_dotenv
import csv

load_dotenv()

class GenerateTranscript:
    def __init__(self, file_name: str, audio_path: str, temp_dir: str="media/temp", output_dir: str="media/outputs", api_key: str | None = None, model: str ="saaras:v3", language_code: str="unknown", mode: str="transcribe"):
        self.audio_path = Path(audio_path)
        self.file_name = file_name
        self.temp_dir = temp_dir
        if not self.audio_path.exists:
            raise FileNotFoundError(f"Audio not found: {self.audio_path}")
        self.output_dir = output_dir
        api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("No Sarvam API key provided. Pass api_key= or set SARVAM_API_KEY.")
        self.client = SarvamAI(api_subscription_key = api_key)
        self.model = model
        self.language_code = language_code
        self.mode = mode

    def TranscribeWithDiarization(self)-> "GenerateTranscript":
        job = self.client.speech_to_text_job.create_job(model=self.model, language_code=self.language_code, mode=self.mode, with_diarization=True)
        print(f"Job created with diarization enabled: {job.job_id}")

        job.upload_files(file_paths=[self.audio_path])
        print("File uploaded")

        job.start()
        status = job.wait_until_complete()
        print(f"Job completed with state: {status.job_state}")

        job.download_outputs(temp_dir=self.temp_dir)

        with tempfile.TemporaryDirectory() as tmp:
            job.download_outputs(temp_dir=tmp)
            result = json.loads((Path(tmp) / f"{self.audio_path.name}.json").read_text())
        
        print(f"\nLanguage: {result['language_code']} (confidence: {result['language_probability']:.0%})\n")
        
        current_speaker = None

        csv_path = f"{self.output_dir}/A2T_{self.file_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["speaker", "start_time_seconds", "end_time_seconds", "transcript"])
            for entry in result["diarized_transcript"]["entries"]:
                spk = f"SPEAKER_{int(entry['speaker_id']):02d}"
                writer.writerow([spk, f"{entry['start_time_seconds']:.2f}", f"{entry['end_time_seconds']:.2f}", entry["transcript"]])

        print(f"\nSaved diarized transcript to {csv_path}")
        
        return self