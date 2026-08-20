from pathlib import Path

import cv2
import insightface
import numpy as np


class FaceDetection:
    def __init__(self, embedding_directory: str = "~/Downloads/Pookie/face_embeddings") -> None:
        self.embedding_directory = Path(embedding_directory).expanduser()
        self._analyzer = None

    def execute(self, frame_path: str | Path, video_path: str | Path, serial_number: int) -> list[dict]:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unable to read frame: {frame_path}")
        faces = self._get_analyzer().get(frame)
        results = []
        for face_number, face in enumerate(faces, start=1):
            embedding_path = self._save_embedding(video_path, serial_number, face_number, face.embedding)
            results.append(
                [
                    f"face_{face_number:03d}",
                    [round(float(value), 4) for value in face.bbox],
                    round(float(face.det_score), 4),
                    "stored_separately",
                    str(embedding_path),
                ]
            )
        return results

    def _get_analyzer(self):
        if self._analyzer is None:
            self._analyzer = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self._analyzer.prepare(ctx_id=0, det_size=(640, 640))
        return self._analyzer

    def _save_embedding(
        self,
        video_path: str | Path,
        serial_number: int,
        face_number: int,
        embedding: np.ndarray,
    ) -> Path:
        self.embedding_directory.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem
        path = self.embedding_directory / f"{video_name}_frame_{serial_number:04d}_face_{face_number:03d}.npy"
        np.save(path, np.asarray(embedding, dtype=np.float32))
        return path
