import csv
import numpy as np
import librosa
from pathlib import Path

# ---------------- CONFIG ----------------
INPUT_AUDIO = "media/outputs/V2A_R-Clip_V1-0001.wav"                  # path to source audio
F_NAME = Path(INPUT_AUDIO).stem
OUTPUT_CSV = f"media/outputs/FA_{F_NAME}.csv"      # condensed anomaly output for LLM
SAMPLE_RATE = None                                # None = native sample rate
N_FFT = 2048
HOP_LENGTH = 512
MIN_FREQ_HZ = 20
MAX_FREQ_HZ = 20000

ROLLING_WINDOW_FRAMES = 43     # ~0.5s at sr=22050, hop=512; local trend window for z-score
Z_SCORE_THRESHOLD = 3.0        # frames beyond this many std devs from local trend are flagged
MERGE_GAP_SEC = 0.3            # merge flagged frames into one event if this close together
CONTEXT_SEC = 0.25             # seconds of context before/after each event to report avg freq for
# -----------------------------------------

class FindFrequency:
    """
        Detect abnormal frequency changes in an audio file and produce a condensed
        CSV of anomaly candidates, suitable for passing to an LLM for review.

        Approach:
        1. Compute dominant frequency + spectral centroid per frame (same as
            extract_audio_frequency.py).
        2. Compute frame-to-frame deltas and a rolling z-score for each signal.
        3. Flag frames where the z-score exceeds a threshold (i.e. a sudden jump
            relative to the recent local trend, not just a high absolute value).
        4. Group consecutive/nearby flagged frames into events, and output one row
            per event (with surrounding context) rather than one row per frame.
            This keeps the output small and focused for LLM consumption.
        """
    def __init__(self, input_path: str, output_path: str,sample_rate: int | None = SAMPLE_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, min_freq_hz: float = MIN_FREQ_HZ, max_freq_hz: float = MAX_FREQ_HZ, rolling_window_frames: int = ROLLING_WINDOW_FRAMES, z_threshold: float = Z_SCORE_THRESHOLD, merge_gap_sec: float = MERGE_GAP_SEC, context_sec: float = CONTEXT_SEC):
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input audio not found: {self.input_path}")
 
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
 
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.rolling_window_frames = rolling_window_frames
        self.z_threshold = z_threshold
        self.merge_gap_sec = merge_gap_sec
        self.context_sec = context_sec

    @staticmethod
    def rolling_zscore(signal: np.ndarray, window: int) -> np.ndarray:
        """Z-score of each point relative to a trailing rolling window (causal, no lookahead bias)."""
        z = np.zeros_like(signal, dtype=float)
        for i in range(len(signal)):
            start = max(0, i - window)
            local = signal[start:i] if i > start else signal[start : i + 1]
            if len(local) < 2:
                continue
            mean, std = local.mean(), local.std()
            if std > 1e-8:
                z[i] = (signal[i] - mean) / std
        return z


    def detect_anomalies(self) -> "FindFrequency":

        print(f"Loading audio: {self.input_path}")
        y, sr = librosa.load(str(self.input_path), sr=self.sample_rate, mono=True)

        print("Computing STFT and frequency signals...")
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=self.hop_length)

        band_mask = (freqs >= self.min_freq_hz) & (freqs <= self.max_freq_hz)
        freqs_band = freqs[band_mask]
        stft_band = stft[band_mask, :]

        dominant_bin_idx = np.argmax(stft_band, axis=0)
        dominant_freq = freqs_band[dominant_bin_idx]

        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)[0]

        print("Scoring anomalies...")
        dom_delta = np.abs(np.diff(dominant_freq, prepend=dominant_freq[0]))
        cen_delta = np.abs(np.diff(centroid, prepend=centroid[0]))

        dom_z = self.rolling_zscore(dom_delta, self.rolling_window_frames)
        cen_z = self.rolling_zscore(cen_delta, self.rolling_window_frames)

        combined_z = np.maximum(dom_z, cen_z)
        flagged = combined_z >= self.z_threshold

        print(f"Flagged {flagged.sum()} / {len(flagged)} frames above z={self.z_threshold}")

        # group flagged frames into events (merge frames within merge_gap_sec of each other)
        flagged_idx = np.where(flagged)[0]
        events = []
        if len(flagged_idx) > 0:
            event_start = flagged_idx[0]
            prev = flagged_idx[0]
            for idx in flagged_idx[1:]:
                if times[idx] - times[prev] > self.merge_gap_sec:
                    events.append((event_start, prev))
                    event_start = idx
                prev = idx
            events.append((event_start, prev))

        print(f"Grouped into {len(events)} events")

        print(f"Writing CSV: {self.out_path}")
        with open(self.out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "event_start_sec",
                    "event_end_sec",
                    "peak_zscore",
                    "freq_before_hz",
                    "freq_during_hz",
                    "freq_after_hz",
                    "centroid_before_hz",
                    "centroid_during_hz",
                    "centroid_after_hz",
                    "rms_before",
                    "rms_during",
                    "rms_after",
                ]
            )
            for start_i, end_i in events:
                t_start, t_end = times[start_i], times[end_i]
                peak_z = combined_z[start_i : end_i + 1].max()

                def window_avg(signal, center_start, center_end, before=False, after=False):
                    if before:
                        mask = (times >= center_start - self.context_sec) & (times < center_start)
                    elif after:
                        mask = (times > center_end) & (times <= center_end + self.context_sec)
                    else:
                        mask = (times >= center_start) & (times <= center_end)
                    vals = signal[mask]
                    return float(vals.mean()) if len(vals) else float("nan")

                writer.writerow(
                    [
                        f"{t_start:.3f}",
                        f"{t_end:.3f}",
                        f"{peak_z:.2f}",
                        f"{window_avg(dominant_freq, t_start, t_end, before=True):.1f}",
                        f"{window_avg(dominant_freq, t_start, t_end):.1f}",
                        f"{window_avg(dominant_freq, t_start, t_end, after=True):.1f}",
                        f"{window_avg(centroid, t_start, t_end, before=True):.1f}",
                        f"{window_avg(centroid, t_start, t_end):.1f}",
                        f"{window_avg(centroid, t_start, t_end, after=True):.1f}",
                        f"{window_avg(rms, t_start, t_end, before=True):.4f}",
                        f"{window_avg(rms, t_start, t_end):.4f}",
                        f"{window_avg(rms, t_start, t_end, after=True):.4f}",
                    ]
                )

        print("Finished.")
        return self


# if __name__ == "__main__":
#     detect_anomalies(INPUT_AUDIO, OUTPUT_CSV)