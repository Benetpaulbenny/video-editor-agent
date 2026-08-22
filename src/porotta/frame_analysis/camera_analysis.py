import cv2
import numpy as np


class CameraAnalysis:
    def __init__(self) -> None:
        self.previous_gray = None

    def start_video(self) -> None:
        self.previous_gray = None

    def execute(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return self._result("static", 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

        previous_points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if previous_points is None or len(previous_points) < 4:
            self.previous_gray = gray
            return self._result("unknown", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or status is None:
            self.previous_gray = gray
            return self._result("unknown", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        valid = status.ravel() == 1
        source = previous_points.reshape(-1, 2)[valid]
        target = current_points.reshape(-1, 2)[valid]
        if len(source) < 4:
            self.previous_gray = gray
            return self._result("unknown", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        transform, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        self.previous_gray = gray
        if transform is None:
            return self._result("unknown", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        translation_x = float(transform[0, 2])
        translation_y = float(transform[1, 2])
        rotation = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
        scale = float(np.sqrt(transform[0, 0] ** 2 + transform[1, 0] ** 2))
        scale_change = scale - 1.0
        inlier_ratio = float(np.mean(inliers)) if inliers is not None else 0.0
        magnitude = float(np.hypot(translation_x, translation_y))
        stability_score = float(np.clip(1.0 / (1.0 + magnitude / 8.0 + abs(rotation) / 8.0 + abs(scale_change) * 20.0), 0.0, 1.0))
        motion_type, confidence = self._motion_type(
            translation_x,
            translation_y,
            rotation,
            scale_change,
            inlier_ratio,
        )
        stability_classification = self._stability_classification(stability_score)
        return {
            "camera_analysis": {
                "camera_angle": {"horizontal": "front", "vertical": "eye_level"},
                "camera_motion": {"type": motion_type, "confidence": round(confidence, 4)},
                "stability": {"score": round(stability_score, 4), "classification": stability_classification},
                "motion": {
                    "translation_x": round(translation_x, 4),
                    "translation_y": round(translation_y, 4),
                    "rotation": round(rotation, 4),
                    "scale_change": round(scale_change, 4),
                },
            }
        }

    def _result(self, motion_type: str, confidence: float, stability: float, x: float, y: float, rotation: float, scale: float) -> dict:
        return {
            "camera_analysis": {
                "camera_angle": {"horizontal": "front", "vertical": "eye_level"},
                "camera_motion": {"type": motion_type, "confidence": round(confidence, 4)},
                "stability": {"score": round(stability, 4), "classification": self._stability_classification(stability)},
                "motion": {
                    "translation_x": round(x, 4),
                    "translation_y": round(y, 4),
                    "rotation": round(rotation, 4),
                    "scale_change": round(scale, 4),
                },
            }
        }

    def _motion_type(self, x: float, y: float, rotation: float, scale: float, inlier_ratio: float) -> tuple[str, float]:
        translation = float(np.hypot(x, y))
        if translation < 1.5 and abs(rotation) < 0.5 and abs(scale) < 0.01:
            motion_type = "static"
        elif abs(scale) >= 0.02 and translation >= 1.5:
            motion_type = "dolly"
        elif abs(scale) >= 0.02:
            motion_type = "zoom"
        elif abs(x) >= abs(y) * 1.25:
            motion_type = "pan"
        elif abs(y) >= abs(x) * 1.25:
            motion_type = "tilt"
        else:
            motion_type = "handheld"
        confidence = float(np.clip(inlier_ratio * (1.0 + min(translation, 10.0) / 20.0), 0.0, 1.0))
        return motion_type, confidence

    def _stability_classification(self, score: float) -> str:
        if score >= 0.8:
            return "stable"
        if score >= 0.5:
            return "moderately_stable"
        return "unstable"
