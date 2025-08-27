from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
import numpy as np
from numpy import linalg as LA
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon, LineString
from shapely.ops import split
from scipy.spatial import Delaunay

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
        self.border_multiline = self.find_border_multiline()
        self.border_faces = self.get_border_faces()
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
    
    def get_or_create_node(self, nodes: Node, point: np.ndarray):
        # Проверяем, есть ли уже узел с такими координатами
        for n in nodes:
            if np.allclose(n.p, point):  # сравнение с допуском
                return n
        # Создаём новый узел
        node = Node(point)
        nodes.append(node)
        return node
        
    def find_border_multiline(self) -> List[Edge]:
        nodes = []
        edges = []

        for segment in self.intersection_segments:
            # Получаем/создаём узлы для концов отрезка
            n1 = self.get_or_create_node(nodes, segment.point1)
            n2 = self.get_or_create_node(nodes, segment.point2)
            # Создаём ребро
            edge = Edge()
            edge.nodes = [n1, n2]
            edge.faces = [segment.face_a, segment.face_b]

            # Добавляем в коллекции
            edges.append(edge)

            # Сохраняем связи
            for f in [segment.face_a, segment.face_b]:
                if edge not in f.edges:
                    f.edges.append(edge)
            for n in [n1, n2]:
                if edge not in n.edges:
                    n.edges.append(edge)
                for f in [segment.face_a, segment.face_b]:
                    if f not in n.faces:
                        n.faces.append(f)

        return edges
    
    
    def delaunay_triangulation(self):
        """
        Делает разбиение пересекающихся треугольников вдоль линии пересечения
        и триангулирует каждую часть в 3D (через проекцию в локальную 2D-плоскость).
        """
        new_faces = []

        for segment in self.intersection_segments:
            face_a = segment.face_a
            face_b = segment.face_b

            tri_a = Polygon([node.p for node in face_a.nodes])
            tri_b = Polygon([node.p for node in face_b.nodes])
            line = LineString([segment.point1, segment.point2])

            split_a = split(tri_a, line)
            split_b = split(tri_b, line)

            polys = []
            for g in split_a.geoms:
                if g.geom_type == "Polygon":
                    polys.append(g)
            for g in split_b.geoms:
                if g.geom_type == "Polygon":
                    polys.append(g)

            for poly in polys:
                coords3d = np.array(poly.exterior.coords)

                if len(coords3d) < 3:
                    continue

                # --- 1. Находим базис плоскости ---
                # Берём первые три точки, строим нормаль
                v1 = coords3d[1] - coords3d[0]
                v2 = coords3d[2] - coords3d[0]
                normal = np.cross(v1, v2)
                normal /= np.linalg.norm(normal)

                # Первая ось в плоскости
                axis_x = v1 / np.linalg.norm(v1)
                # Вторая ось в плоскости (ортогональна первой и нормали)
                axis_y = np.cross(normal, axis_x)
                axis_y /= np.linalg.norm(axis_y)

                # --- 2. Проецируем в локальную 2D-плоскость ---
                coords2d = np.array([
                    [np.dot(p - coords3d[0], axis_x), np.dot(p - coords3d[0], axis_y)]
                    for p in coords3d
                ])

                # --- 3. Делаем Delaunay в 2D ---
                try:
                    delaunay = Delaunay(coords2d, qhull_options="QJ")
                except Exception:
                    continue

                # --- 4. Восстанавливаем треугольники в 3D ---
                for simplex in delaunay.simplices:
                    pts = coords3d[simplex]
                    nodes = [Node(p) for p in pts]
                    f = Face()
                    f.nodes = nodes
                    new_faces.append(f)
        
        # Собираем множество координат граничных узлов
        border_points = {
            tuple(node.p) for edge in self.border_multiline for node in edge.nodes
        }

        # Фильтруем новые треугольники
        filtered_faces = []
        for face in new_faces:
            if any(tuple(node.p) in border_points for node in face.nodes):
                filtered_faces.append(face)

        # Результат
        new_faces = filtered_faces
        return new_faces 
        

    def delaunay_triangulation_v2(self):
        """
        Делает разбиение пересекающихся треугольников вдоль линии пересечения.
        Для каждого треугольника создается 4-точечный многоугольник (2 точки линии + 2 вершины треугольника,
        которые не лежат на линии), затем выполняется Delaunay триангуляция.
        """
        def point_on_edge(p, a, b, tol=1e-8):
            """Проверка, лежит ли точка p на отрезке ab"""
            ab = b - a
            ap = p - a
            cross = np.linalg.norm(np.cross(ab, ap))
            if cross > tol:
                return False
            dot = np.dot(ap, ab)
            if dot < -tol or dot > np.dot(ab, ab) + tol:
                return False
            return True

        new_faces = []
        
        polygons_coord = []

        for segment in self.intersection_segments:
            line_points = [segment.point1, segment.point2]

            for face in [segment.face_a, segment.face_b]:
                tri_coords = np.array([node.p for node in face.nodes])

                # выбираем две вершины треугольника, которые НЕ лежат на линии пересечения
                remaining_points = []
                for i in range(3):
                    p = tri_coords[i]
                    if not any(point_on_edge(lp, tri_coords[j], tri_coords[(j+1)%3]) for j in range(3) for lp in line_points):
                        remaining_points.append(p)

                if len(remaining_points) != 2:
                    # вырожденный случай: берём любые две точки треугольника
                    remaining_points = tri_coords[:2]

                # формируем 4-точечный многоугольник: 2 точки линии + 2 вершины треугольника
                quad_points = np.vstack([line_points, remaining_points])
                if len(quad_points) < 4:
                    raise Exception(f"quad_points < 4: {quad_points}, line_points: {line_points} but remaining_points: {remaining_points}")
                polygons_coord.append(quad_points)
                
                # --- 1. Находим базис плоскости ---
                v1 = quad_points[1] - quad_points[0]
                v2 = quad_points[2] - quad_points[0]
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm < 1e-8:
                    continue  # вырожденный случай
                normal /= norm

                axis_x = v1 / np.linalg.norm(v1)
                axis_y = np.cross(normal, axis_x)
                axis_y /= np.linalg.norm(axis_y)

                # --- 2. Проекция в локальную 2D плоскость ---
                coords2d = np.array([
                    [np.dot(p - quad_points[0], axis_x), np.dot(p - quad_points[0], axis_y)]
                    for p in quad_points
                ])

                # --- 3. Delaunay триангуляция ---
                try:
                    delaunay = Delaunay(coords2d, qhull_options="QJ")
                except Exception:
                    raise Exception("degenerate triangle")

                # --- 4. Создаём новые Face ---
                for simplex in delaunay.simplices:
                    pts = quad_points[simplex]
                    nodes = [Node(p) for p in pts]
                    f = Face()
                    f.nodes = nodes
                    f.zone = face.zone
                    new_faces.append(f)

        return new_faces, polygons_coord

    
    
    def get_border_faces(self):
        seen = set()
        list_of_faces = []
        for segment in self.intersection_segments:
            for face in (segment.face_a, segment.face_b):
                if face not in seen:
                    seen.add(face)
                    list_of_faces.append(face)
        return list_of_faces

    
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
