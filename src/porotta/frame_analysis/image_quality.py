import cv2
import numpy as np


class ImageQuality:
    BLUR_REFERENCE = 500.0
    NOISE_REFERENCE = 25.0

    def execute(self, frame_path: str) -> list[float]:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unable to read frame: {frame_path}")
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_variance = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        noise_sigma = self._estimate_noise(grayscale)
        return [
            round(float(grayscale.mean()), 4),
            round(float(grayscale.std()), 4),
            round(min(laplacian_variance / self.BLUR_REFERENCE, 1.0), 4),
            round(min(noise_sigma / self.NOISE_REFERENCE, 1.0), 4),
        ]

    def _estimate_noise(self, grayscale: np.ndarray) -> float:
        smoothed = cv2.GaussianBlur(grayscale, (3, 3), 0).astype(np.float32)
        residual = grayscale.astype(np.float32) - smoothed
        median = np.median(residual)
        return float(np.median(np.abs(residual - median)) / 0.6745)
