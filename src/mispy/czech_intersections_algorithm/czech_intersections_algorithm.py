from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
import numpy as np
from numpy import linalg as LA
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==================================================================================================
class IntersectionSegment:
    def __init__(
        self,
        face_a: Face,
        face_b: Face,
        point1: np.ndarray,
        point2: np.ndarray,
        classification: str,
        zones: Tuple[Zone, Zone]
    ) -> None:
        self.face_a = face_a
        self.face_b = face_b
        self.point1 = point1
        self.point2 = point2
        self.classification = classification
        self.zones = zones

    def describe(self) -> None:
        print("IntersectionSegment:")
        print(f"  Face A: {[node.p for node in self.face_a.nodes]}")
        print(f"  Face B: {[node.p for node in self.face_b.nodes]}")
        print(f"  Point 1: {self.point1}")
        print(f"  Point 2: {self.point2}")
        print(f"  Classification: {self.classification}")
        print(f"  Zones: {self.zones[0].name}, {self.zones[1].name}")

class CzechIntersectionsAlgorithm:
    def __init__(self, mesh):
        self.mesh = mesh
        self.intersection_segments = self.find_all_intersections()
# ==================================================================================================
# Геометрические утилиты
# ==================================================================================================

    def normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def triangle_normal(self, v0, v1, v2):
        return self.normalize(np.cross(v1 - v0, v2 - v0))

# ==================================================================================================
# Алгоритм Muller–Trumbore для луча и треугольника
# ==================================================================================================
    def ray_triangle_intersection(self, orig, dir, triangle, epsilon=1e-8):
        v0, v1, v2 = triangle
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(dir, edge2)
        a = np.dot(edge1, h)
        if -epsilon < a < epsilon:
            return None  # Прямая параллельна треугольнику
        f = 1.0 / a
        s = orig - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = np.cross(s, edge1)
        v = f * np.dot(dir, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = f * np.dot(edge2, q)
        if t > epsilon:
            return t, u, v
        return None

# ==================================================================================================
# Поиск пересечения двух треугольников
# ==================================================================================================
    def intersect_faces(self, face_a: Face, face_b: Face) -> Optional[IntersectionSegment]:
        tri_a = [node.p for node in face_a.nodes]
        tri_b = [node.p for node in face_b.nodes]

        intersection_points = []

        for i in range(3):
            p0 = tri_a[i]
            p1 = tri_a[(i + 1) % 3]
            dir = p1 - p0

            result = self.ray_triangle_intersection(p0, dir, tri_b)
            if result:
                t, u, v = result
                if 0 <= t <= 1:
                    intersection_points.append(p0 + t * dir)

        for i in range(3):
            p0 = tri_b[i]
            p1 = tri_b[(i + 1) % 3]
            dir = p1 - p0

            result = self.ray_triangle_intersection(p0, dir, tri_a)
            if result:
                t, u, v = result
                if 0 <= t <= 1:
                    intersection_points.append(p0 + t * dir)

        # Убираем дубли
        unique_points = []
        for pt in intersection_points:
            if not any(np.allclose(pt, up, atol=1e-8) for up in unique_points):
                unique_points.append(pt)

        if len(unique_points) == 0:
            return None
        elif len(unique_points) == 1:
            return IntersectionSegment(face_a, face_b, unique_points[0], unique_points[0], "need_classification", (face_a.zone, face_b.zone))
        elif len(unique_points) == 2:
            return IntersectionSegment(face_a, face_b, unique_points[0], unique_points[1], "need_classification", (face_a.zone, face_b.zone))
        else:
            sorted_pts = sorted(unique_points, key=lambda p: np.linalg.norm(p - unique_points[0]))
            return IntersectionSegment(face_a, face_b, sorted_pts[0], sorted_pts[-1], "need_classification", (face_a.zone, face_b.zone))

# ==================================================================================================
# Основная функция: найти все пересечения
# ==================================================================================================
    def find_all_intersections(self) -> List[IntersectionSegment]:
        segments = []
        for i, f1 in enumerate(self.mesh.faces):
            for j, f2 in enumerate(self.mesh.faces):
                if j <= i or f1.zone == f2.zone:
                    continue
                result = self.intersect_faces(f1, f2)
                if result:
                    segments.append(result)
        self.intersection_segments = segments
        return segments

    def get_border_faces(self):
        seen = set()
        list_of_faces = []
        for segment in self.intersection_segments:
            for face in (segment.face_a, segment.face_b):
                if face not in seen:
                    seen.add(face)
                    list_of_faces.append(face)
        return list_of_faces
    
    def describe_border_faces(self):
        for segment in self.intersection_segments:
            segment.describe()

    
    def change_mesh(self):
        new_faces = []
        border_faces_ids = {f.glo_id for f in self.get_border_faces()}
        for face_main in self.mesh.faces:
            if face_main.glo_id not in border_faces_ids:
                new_faces.append(face_main)
        self.mesh.faces = new_faces
        self.mesh.title = "combined_" + self.mesh.title
        return self.mesh

# ==================================================================================================

if __name__ == '__main__':
    pass

# ==================================================================================================
