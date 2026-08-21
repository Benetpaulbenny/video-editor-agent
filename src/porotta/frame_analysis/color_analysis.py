import cv2
import numpy as np


class ColorAnalysis:
    DOMINANT_COLOR_COUNT = 3

    def execute(self, frame_path: str) -> list:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unable to read frame: {frame_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return [
            self._color_temperature(rgb),
            self._dominant_colors(rgb),
            self._average_rgb(rgb),
            self._saturation(hsv),
        ]

    def _average_rgb(self, rgb: np.ndarray) -> dict:
        average = rgb.mean(axis=(0, 1))
        return [round(float(value), 4) for value in average]

    def _color_temperature(self, rgb: np.ndarray) -> dict:
        red, green, blue = rgb.mean(axis=(0, 1)) / 255
        x = -0.14282 * red + 1.54924 * green - 0.95641 * blue
        y = -0.32466 * red + 1.57837 * green - 0.73191 * blue
        z = -0.68202 * red + 0.77073 * green + 0.56332 * blue
        total = x + y + z
        if total <= 0:
            kelvin = 4000
        else:
            chromaticity_x = x / total
            chromaticity_y = y / total
            denominator = 0.1858 - chromaticity_y
            offset = (chromaticity_x - 0.3320) / denominator if denominator else 0
            kelvin = 449 * offset**3 + 3525 * offset**2 + 6823.3 * offset + 5520.33
        kelvin = int(np.clip(kelvin, 1000, 12000))
        classification = "cool" if kelvin < 4000 else "neutral" if kelvin <= 5500 else "warm"
        return [kelvin, classification]

    def _dominant_colors(self, rgb: np.ndarray) -> list[dict]:
        pixels = rgb.reshape((-1, 3)).astype(np.float32)
        if len(pixels) > 10000:
            pixels = pixels[np.linspace(0, len(pixels) - 1, 10000).astype(int)]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels,
            self.DOMINANT_COLOR_COUNT,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        counts = np.bincount(labels.flatten(), minlength=self.DOMINANT_COLOR_COUNT)
        total = counts.sum()
        return [
            [*[int(round(value)) for value in centers[index]], round(float(counts[index] / total), 4)]
            for index in np.argsort(counts)[::-1]
        ]

    def _saturation(self, hsv: np.ndarray) -> dict:
        saturation = hsv[:, :, 1].astype(np.float32) / 255
        return [
            round(float(saturation.mean()), 4),
            round(float(np.median(saturation)), 4),
            round(float(np.mean(saturation < 0.25)), 4),
            round(float(np.mean(saturation > 0.75)), 4),
        ]
