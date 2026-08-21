import numpy as np


class FaceLandmark:
    def execute(self, faces: list, frame_shape: tuple[int, ...]) -> list[list]:
        height, width = frame_shape[:2]
        return [self._face_landmarks(face, index, width, height) for index, face in enumerate(faces, start=1)]

    def _face_landmarks(self, face, face_number: int, width: int, height: int) -> list:
        regions = self._regions(face)
        normalized_regions = self._normalize_regions(regions, width, height)
        return [
            f"face_{face_number:03d}",
            regions,
            normalized_regions,
            round(float(face.det_score), 4),
        ]

    def _regions(self, face) -> list:
        points = getattr(face, "landmark_3d_68", None)
        if points is not None and len(points) >= 68:
            points = np.asarray(points)[:, :2]
            return [
                [self._points(points[42:48]), self._points(points[36:42])],
                self._points(points[27:36]),
                self._points(points[0:17]),
                self._points(points[48:68]),
            ]
        keypoints = getattr(face, "kps", None)
        if keypoints is not None and len(keypoints) >= 5:
            keypoints = np.asarray(keypoints)[:, :2]
            return [
                [[*self._point(keypoints[0])], [*self._point(keypoints[1])]],
                [self._point(keypoints[2])],
                [],
                [[*self._point(keypoints[3])], [*self._point(keypoints[4])]],
            ]
        return [[], [], [], []]

    def _normalize_regions(self, regions: list, width: int, height: int) -> list:
        return [self._normalize_region(region, width, height) for region in regions]

    def _normalize_region(self, region, width: int, height: int):
        if not region:
            return []
        if isinstance(region[0], (int, float)):
            return [round(float(region[0] / width), 6), round(float(region[1] / height), 6)]
        return [self._normalize_region(point, width, height) for point in region]

    def _points(self, points: np.ndarray) -> list[list[float]]:
        return [self._point(point) for point in points]

    def _point(self, point: np.ndarray) -> list[float]:
        return [round(float(point[0]), 4), round(float(point[1]), 4)]
