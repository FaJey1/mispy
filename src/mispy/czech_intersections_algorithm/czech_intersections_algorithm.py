import logging
import time


import numpy as np
from numpy import linalg as LA
from typing import List, Tuple, Optional
from collections import defaultdict
from itertools import combinations

from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
from itertools import combinations

# ==================================================================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ==================================================================================================

class CzechIntersectionsAlgorithm:
    def __init__(self, mesh):
        self.mesh = mesh
        self.border_faces_pairs= []
        self.border_faces_classification= []
        self.border_faces = []
        self.border_multiline = []
        self.new_zones = []
        self.impossible_couples = []
        
        self.last_face = max([face.glo_id for face in self.mesh.faces]) + 1
        self.last_edge = max([edge.glo_id for edge in self.mesh.edges]) + 1
        self.new_mesh = None
        self.new_faces = []
        self.new_edges = []
        self.new_nodes = []

    
    def make_pairs(self, zones: list = []):
        """Вычисление пар сеток для устранения пересечений"""
        pairs = {}
        
        last_pair = zones[0].name
        self.new_zones.append(zones[0])
        
        for z in zones[1:]:
            pair = (self.new_zones.pop(), z)
            last_pair += f"_{z.name}"
            self.new_zones.append(Zone(last_pair))
            pairs[last_pair] = pair
        
        return pairs
            
            
    def face_neighbours(self, face):
        """Соседи через все рёбра"""
        neighbours = []
        for e in face.edges:
            neigh = face.neighbour(e)
            if neigh is not None:
                neighbours.append(neigh)
        return neighbours
    
    
    def aabb_intersect(self, aabb1, aabb2, tol=1e-12):
        """Построение AABB параллелепипеда"""
        (min1, max1), (min2, max2) = aabb1, aabb2
        overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
        return np.all(overlap > tol)
    
    
    def face_aabb(self, face):
        """Возвращает (min, max) по координатам для грани""" 
        coords = np.array([node.p for node in face.nodes]) 
        return coords.min(axis=0), coords.max(axis=0)
    
    
    def face_center(self, face):
        """Центр треугольника"""
        coords = np.array([node.p for node in face.nodes])
        return coords.mean(axis=0)


    def max_dist_from_aabb(self, aabb):
        """Максимальное расстояние на основе объёма AABB"""
        (mn, mx) = aabb
        size = mx - mn
        volume = np.prod(size)  # dx * dy * dz
        return np.cbrt(volume)  # кубический корень


    def classify_intersections(self, faces_pair):
        """
        Для каждой пары граней [face0, face1] из faces_pair
        ищет точки пересечения рёбер face0 с плоскостью face1 и наоборот.
        
        Возвращает словарь:
        {
            "faces": (face0, face1),
            "face0_vs_face1": [ (point, cls),  ],
            "face1_vs_face0": [ (point, cls), ... ]
        },
        """
        results = []

        def plane_from_face(face):
            """Возвращает нормаль и точку (n, p0) для плоскости треугольника."""
            p0, p1, p2 = [np.array(node.p) for node in face.nodes]
            n = np.cross(p1 - p0, p2 - p0)
            n /= np.linalg.norm(n)
            return n, p0

        def intersect_edge_with_plane(p1, p2, n, p0, eps=1e-9):
            """
            Пересечение ребра (p1,p2) с плоскостью (n,p0).
            Возвращает (point, cls, <face>_<edge_id>).
            """
            p1, p2, p0 = map(np.array, (p1, p2, p0))
            u = p2 - p1
            denom = np.dot(n, u)

            # ребро параллельно плоскости
            if abs(denom) < eps:
                return None

            t = np.dot(n, p0 - p1) / denom
            point = p1 + t * u

            if t < -eps or t > 1 + eps:
                cls = 0  # вне отрезка
            elif abs(t) < eps or abs(t - 1) < eps:
                cls = 1  # вершина
            else:
                cls = 2  # внутри ребра
            return point, cls

        n1, p01 = plane_from_face(faces_pair[1])
        n0, p00 = plane_from_face(faces_pair[0])

        # face0 против плоскости face1
        intersections0 = []
        for e in faces_pair[0].edges:
            p1, p2 = e.nodes[0].p, e.nodes[1].p
            result = intersect_edge_with_plane(p1, p2, n1, p01)
            if result is not None:
                result = result + (f"{faces_pair[0].glo_id}_{e.glo_id}", )
                intersections0.append(result)

        # face1 против плоскости face0
        intersections1 = []
        for e in faces_pair[1].edges:
            p1, p2 = e.nodes[0].p, e.nodes[1].p
            result = intersect_edge_with_plane(p1, p2, n0, p00)
            if result is not None:
                result = result + (f"{faces_pair[1].glo_id}_{e.glo_id}", )
                intersections1.append(result)

        results = {
            "faces": (faces_pair[0], faces_pair[1]),
            "face0_vs_face1": intersections0,
            "face1_vs_face0": intersections1
        }

        return results
    
    
    def classify_faces(self, face_info: dict):
        impossible = {"001", "002", "012", "122", "222"}
        case000 = "000"

        results = {}

        for key in ["face0_vs_face1", "face1_vs_face0"]:
            # у нас кортеж из трех элементов, берём только cls
            cls_list = [str(cls) for _, cls, _ in face_info[key]]
            if len(cls_list) != 3:
                logging.warning("classify_faces: %s must contain exactly 3 classifiers", key)
                logging.warning("face1 id: %s, face2 id: %s", face_info["faces"][0].glo_id, face_info["faces"][1].glo_id)
                logging.warning(face_info["face0_vs_face1"])
                logging.warning(face_info["face1_vs_face0"])
                
                # про это не сказано, но если 2 точки, то 3 точно 0
                logging.warning("classify_faces: perhaps the planes are perpendicular or parallel")
                while len(cls_list) < 3:
                    cls_list.append("0")
                
                # raise ValueError(f"{key} должен содержать ровно 3 классификатора")
            code = "".join(sorted(cls_list))  # сортируем, чтобы порядок точек не мешал
            
            if code == case000:
                results[key] = 0
            elif code in impossible:
                results[key] = 2
            else:
                results[key] = 1

        return results
    
    
    def face_intersection_line(self, face_info, eps=1e-9):
        """
        Находит пересечение двух отрезков, образованных точками с cls=2
        из face0_vs_face1 и face1_vs_face0.

        Возвращает np.array с двумя точками пересечения или одной точкой, если отрезки пересекаются в точке.
        Если пересечения нет — пустой массив.
        """
        # Выбираем точки с cls=2
        pts0 = [np.array(pt) for pt, cls, _ in face_info["face0_vs_face1"] if cls == 2 or cls == 1]
        pts1 = [np.array(pt) for pt, cls, _ in face_info["face1_vs_face0"] if cls == 2 or cls == 1]

        if len(pts0) != 2 or len(pts1) != 2:
            raise ValueError("Каждая грань должна содержать ровно 2 точки с cls=1,2")

        p1, p2 = pts0
        q1, q2 = pts1

        # Выбираем ось с наибольшим размахом отрезка p1-p2
        axis = np.argmax(np.abs(p2 - p1))

        # Проекции на выбранную ось
        p_proj = sorted([p1[axis], p2[axis]])
        q_proj = sorted([q1[axis], q2[axis]])

        # Интервал пересечения проекций
        lo = max(p_proj[0], q_proj[0])
        hi = min(p_proj[1], q_proj[1])

        if hi < lo - eps:
            return np.array([])  # пересечения нет

        # Функция для восстановления координаты по параметру t
        def point_from_param(val, a, b):
            if abs(b[axis] - a[axis]) < eps:
                return a.copy()  # вырожденный случай
            t = (val - a[axis]) / (b[axis] - a[axis])
            return a + t * (b - a)

        pt_lo = point_from_param(lo, p1, p2)
        pt_hi = point_from_param(hi, p1, p2)

        if np.linalg.norm(pt_hi - pt_lo) < eps:
            return np.array([pt_lo])  # пересечение в точке

        return np.array([pt_lo, pt_hi])
    

    def candidate_face_pairs(self, zone1, zone2):
        """Кандидаты для пересечения zone1 × zone2 через обход соседей"""
        pairs = []
        visited = set()
        num = 0
        for f1 in zone1.faces:
            num += 1
            logging.info("Cheking face num: %s, id: %s", num , f1.glo_id)
            
            aabb1 = self.face_aabb(f1)
            center1 = self.face_center(f1)
            max_dist = self.max_dist_from_aabb(aabb1)

            # стартовые кандидаты только по AABB
            stack = [f for f in zone2.faces if self.aabb_intersect(aabb1, self.face_aabb(f))]

            while stack:
                f2 = stack.pop()
                if (f1.glo_id, f2.glo_id) in visited:
                    continue
                visited.add((f1.glo_id, f2.glo_id))

                # доп. фильтр: расстояние между центрами
                center2 = self.face_center(f2)
                    
                # проверка через расстояние между центроидами граней для сокращения числа пар
                # if np.linalg.norm(center1 - center2) > max_dist:
                #     continue
                
                # поиск отрезков попарного пересечения граней-кандидатов 
                # !!! на этом этапе вставить чехов!!!!
                pair = self.classify_intersections(faces_pair = [f1, f2])
                if self.classify_faces(pair)["face0_vs_face1"] == 0 or self.classify_faces(pair)["face1_vs_face0"] == 0:
                    logging.info("Czech classification: combination 000 -> planes do not intersect, faces can not be viewed")
                    continue
                elif self.classify_faces(pair)["face0_vs_face1"] == 2 or self.classify_faces(pair)["face1_vs_face0"] == 2:
                    logging.info("Czech classification: floating point error found, impossible combination -> going through the neighbors")
                    self.impossible_couples.append([f1, f2])
                    # МЕТКА А: переместил сюда код
                    # проверяем соседей f2, но относительно исходной f1
                    for neigh in self.face_neighbours(f2):
                        if neigh.zone == zone2 and self.aabb_intersect(aabb1, self.face_aabb(neigh)):
                            stack.append(neigh)
                intersection_line = self.face_intersection_line(pair)
                if intersection_line.size == 0:
                    logging.info("Czech classification: classification case is acceptable, but the edges do not intersect.")
                    continue
                logging.info("Czech classification: intersection was found correctly")
                self.border_multiline.append(intersection_line)
                pairs.append(pair)

                # pair = {}
                # pair["faces"] = [f1, f2]
                # pairs.append(pair)
                # МЕТКА А: раньше код был тут
                # проверяем соседей f2, но относительно исходной f1
                # for neigh in self.face_neighbours(f2):
                #     if neigh.zone == zone2 and self.aabb_intersect(aabb1, self.face_aabb(neigh)):
                #         stack.append(neigh)
            
        return pairs

    
    def triangulation_pair(self):
        """
        Триангуляция каждой пары пересекающихся граней через линию пересечения.
        Новые треугольники сохраняются в self.new_face, новые ребра — в self.new_edge.
        """
        for face_pair_tuple, intersection_edge in zip(self.border_faces_pairs, self.border_multiline):
            face0, face1 = face_pair_tuple["faces"]

            # 1. копируем исходные вершины граней
            polygon0_nodes = face0.nodes.copy()
            polygon1_nodes = face1.nodes.copy()
            
            # 2. вставляем концы линии пересечения
            line_points = [intersection_edge[0], intersection_edge[1]]

            def insert_line_points(polygon_nodes, line_pts):
                """
                Вставляем концы линии пересечения в вершины полигона в порядке следования вдоль ребер
                """
                new_polygon_nodes = polygon_nodes.copy()
                for pt in line_pts:
                    min_dist = float("inf")
                    insert_idx = 0
                    for i in range(len(new_polygon_nodes)):
                        a = new_polygon_nodes[i].p
                        b = new_polygon_nodes[(i + 1) % len(new_polygon_nodes)].p
                        ab = b - a
                        ap = pt - a
                        t = np.dot(ap, ab) / np.dot(ab, ab) if np.dot(ab, ab) != 0 else 0
                        t = np.clip(t, 0, 1)
                        proj = a + t * ab
                        dist = np.linalg.norm(pt - proj)
                        if dist < min_dist:
                            min_dist = dist
                            insert_idx = i + 1
                    new_polygon_nodes.insert(insert_idx, Node(pt))
                return new_polygon_nodes

            polygon0_nodes = insert_line_points(polygon0_nodes, line_points)
            polygon1_nodes = insert_line_points(polygon1_nodes, line_points)

            # 3. триангуляция методом обрезания уха
            def triangulate_polygon(polygon_nodes):
                faces = []
                nodes = polygon_nodes.copy()
                while len(nodes) >= 3:
                    # берем первые три вершины как "ухо"
                    face_nodes = nodes[:3]
                    new_face = Face()
                    new_face.glo_id = self.last_face
                    self.last_face += 1
                    new_face.nodes = face_nodes

                    # создаем ребра
                    new_face.edges = []
                    for i in range(3):
                        edge = Edge()
                        edge.glo_id = self.last_edge
                        self.last_edge += 1
                        edge.nodes = [face_nodes[i], face_nodes[(i + 1) % 3]]
                        edge.faces = [new_face]
                        new_face.edges.append(edge)
                        self.new_edges.append(edge)

                    faces.append(new_face)
                    self.new_faces.append(new_face)
                    # убираем среднюю вершину для следующего треугольника
                    nodes.pop(1)
                return faces

            # триангулируем каждую грань
            triangulate_polygon(polygon0_nodes)
            triangulate_polygon(polygon1_nodes)
        
        
    def fix_mesh(self):
        pass
    
    
    def border_faces_pairs_to_list(self):
        """Список граней образующих границу"""
        ids = set()
        for pair in  self.border_faces_pairs:
            ids.add(pair["faces"][0].glo_id)
            ids.add(pair["faces"][1].glo_id)
        
        for id in ids:
            self.border_faces.append(self.mesh.find_face_by_id(id))
        
        
    def builder(self):
        """Устранение пересечений сеток"""
        # формирование пар для поиска взаимопересения
        zones = self.mesh.zones
        pairs = self.make_pairs(zones)
        
        # поиск кандидатов
        logging.info("Search for intersection candidates")
        for pair in pairs:
            start = time.time()
            logging.info("New zone calculated: %s, parent zones: '%s', '%s'", pair, pairs[pair][0].name, pairs[pair][1].name)
            self.border_faces_pairs = self.candidate_face_pairs(pairs[pair][0], pairs[pair][1])
            self.triangulation_pair()
            self.fix_mesh()
            end = time.time()
            logging.info(f"Zone {pair}. search time for candidates: {(end - start):.6f}s")
            
            self.border_faces_pairs_to_list()
            logging.info("Zone %s. intersection candidates found, pair count: %s, border pair cls: %s, impossible pair: %s, line pair count %s", pair, len(self.border_faces_pairs), len(self.border_faces), len(self.impossible_couples), len(self.border_multiline))
        
        # триангуляция и восстановление связей
        # триангуляция граней с сохранением привязки к исходным ребрам 
        # новые ребра и вершины нужно добавить во временную структуру
        self.triangulation_pair()
        
        
# ==================================================================================================

if __name__ == '__main__':
    pass

# ==================================================================================================
