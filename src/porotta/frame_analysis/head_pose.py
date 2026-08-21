import cv2
import numpy as np


class HeadPose:
    LANDMARK_INDICES = (30, 8, 36, 45, 48, 54)
    MODEL_POINTS = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0],
        ],
        dtype=np.float64,
    )

    def execute(self, faces: list, frame_shape: tuple[int, ...]) -> list[list]:
        height, width = frame_shape[:2]
        return [
            self._estimate(face, index, width, height)
            for index, face in enumerate(faces, start=1)
        ]

    def _estimate(self, face, face_number: int, width: int, height: int) -> list:
        landmarks = getattr(face, "landmark_3d_68", None)
        if landmarks is None or len(landmarks) < 68:
            return [f"face_{face_number:03d}", 0.0, 0.0, 0.0, 0.0]
        landmarks = np.asarray(landmarks)[:, :2].astype(np.float64)
        image_points = landmarks[list(self.LANDMARK_INDICES)]
        camera_matrix = np.array(
            [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return [f"face_{face_number:03d}", 0.0, 0.0, 0.0, 0.0]
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        yaw, pitch, roll = self._angles(rotation_matrix)
        projected, _ = cv2.projectPoints(
            self.MODEL_POINTS,
            rotation_vector,
            translation_vector,
            camera_matrix,
            np.zeros((4, 1)),
        )
        error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))
        confidence = float(np.clip(float(face.det_score) * np.exp(-error / 20), 0, 1))
        return [
            f"face_{face_number:03d}",
            round(yaw, 4),
            round(pitch, 4),
            round(roll, 4),
            round(confidence, 4),
        ]

    def _angles(self, rotation_matrix: np.ndarray) -> tuple[float, float, float]:
        scale = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        if scale > 1e-6:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], scale)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], scale)
            roll = 0.0
        return tuple(float(np.degrees(angle)) for angle in (yaw, pitch, roll))
