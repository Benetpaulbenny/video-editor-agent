from sarvamai import SarvamAI
import json
import tempfile
from pathlib import Path

client = SarvamAI(
    api_subscription_key="sk_99lvdbkb_9DEKPS9N7Ct834PmnpiTRxB8",
)


job = client.speech_to_text_job.create_job(
    model="saaras:v3",
    language_code="unknown",
    mode="transcribe",
    with_diarization=True,
)
print(f"Job created: {job.job_id}")


FILE = "media/outputs/V2A_R-Clip_V1-0001.wav"
job.upload_files(file_paths=[FILE])
F_NAME = Path(FILE).name
print(F_NAME)
print("File uploaded")

job.start()


status = job.wait_until_complete()
print(f"Job completed with state: {status.job_state}")

job.download_outputs(output_dir="media/outputs")
print("Outputs saved to media/outputs")

with tempfile.TemporaryDirectory() as tmp:
    job.download_outputs(output_dir=tmp)
    result = json.loads((Path(tmp) / f"{F_NAME}.json").read_text())
 
print(f"\nLanguage: {result['language_code']} (confidence: {result['language_probability']:.0%})\n")
 
current_speaker = None

import csv


csv_path = f"media/outputs/A2T_{F_NAME}.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["speaker", "start_time_seconds", "end_time_seconds", "transcript"])
    for entry in result["diarized_transcript"]["entries"]:
        spk = f"SPEAKER_{int(entry['speaker_id']):02d}"
        writer.writerow([spk, f"{entry['start_time_seconds']:.2f}", f"{entry['end_time_seconds']:.2f}", entry["transcript"]])

print(f"\nSaved diarized transcript to {csv_path}")