import contextlib
import io
from pathlib import Path

import cv2
import insightface
import numpy as np


class FaceDetection:
    def __init__(self) -> None:
        self._analyzer = None

    def detect(self, frame_path: str | Path) -> tuple[np.ndarray, list]:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unable to read frame: {frame_path}")
        return frame, self._get_analyzer().get(frame)

    def execute(self, frame_path: str | Path, faces: list | None = None) -> list[list]:
        if faces is None:
            _, faces = self.detect(frame_path)
        return [
            [
                f"face_{face_number:03d}",
                [round(float(value), 4) for value in face.bbox],
                round(float(face.det_score), 4),
                [round(float(value), 8) for value in np.asarray(face.embedding).reshape(-1)],
            ]
            for face_number, face in enumerate(faces, start=1)
        ]

    def _get_analyzer(self):
        if self._analyzer is None:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._analyzer = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                )
                self._analyzer.prepare(ctx_id=0, det_size=(640, 640))
        return self._analyzer
