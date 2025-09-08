import logging

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

class Plane:
    """
    Класс Plane — контейнер для уравнения плоскости, построенной по грани (Face).
    """

    def __init__(self, face, size: float = 1.0):
        """
        Конструктор. Строит уравнение плоскости Ax + By + Cz + D = 0 
        и опорный прямоугольник вокруг грани.

        Parameters
        ----------
        face : Face
            Объект Face, содержащий как минимум 3 вершины (nodes).
        size : float
            Масштаб прямоугольника в плоскости.
        """
        
        self.glo_id = face.glo_id
        self.intersection = False
        
        if len(face.nodes) < 3:
            raise ValueError("Грань должна иметь минимум 3 вершины")

        # Берём три точки
        p1, p2, p3 = [node.p for node in face.nodes[:3]]

        # Нормаль = нормированное векторное произведение
        n = np.cross(p2 - p1, p3 - p1)
        n /= np.linalg.norm(n)

        # Коэффициенты уравнения
        self.A, self.B, self.C = n
        self.D = -np.dot(n, p1)

        # Опорные данные
        self.points = [p1, p2, p3]
        self.normal = n

        # Два базисных вектора в плоскости
        # просто берём первый ненулевой вектор и ортогонализуем
        u = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        u = u - np.dot(u, n) * n   # ортогонализация
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        # Центр (центроид треугольника)
        center = (p1 + p2 + p3) / 3.0

        # Углы прямоугольника в плоскости
        self.corners = [
            center + size * ( u + v),
            center + size * (-u + v),
            center + size * (-u - v),
            center + size * ( u - v),
        ]


    def equation(self):
        """
        Возвращает коэффициенты уравнения плоскости (A, B, C, D),
        где Ax + By + Cz + D = 0.
        """
        return self.A, self.B, self.C, self.D


    def distance(self, point):
        """
        Вычисляет *ориентированное* расстояние от произвольной точки до плоскости.

        Если результат положительный — точка находится по направлению нормали,
        если отрицательный — с противоположной стороны,
        если равно нулю — точка лежит в плоскости.
        """
        return self.A*point[0] + self.B*point[1] + self.C*point[2] + self.D


# ==================================================================================================

class Line:
    """
    Класс Line — контейнер для прямой, полученной как пересечение двух плоскостей (Plane).
    """

    def __init__(self, plane1, plane2, t_min=-1, t_max=1):
        """
        Конструктор. Строит прямую как пересечение двух плоскостей.

        Parameters
        ----------
        plane1, plane2 : Plane
            Две плоскости.
        t_min, t_max : float
            Диапазон параметра t для генерации отрезка (для визуализации).
        """

        self.plane1 = plane1
        self.plane2 = plane2
        self.t_min = t_min
        self.t_max = t_max
        self.direction = 0
        self.point = 0

    def build_intersection_line(self):
        # Направляющий вектор прямой = векторное произведение нормалей
        d = np.cross([self.plane1.A, self.plane1.B, self.plane1.C],
                     [self.plane2.A, self.plane2.B, self.plane2.C])

        norm_d = np.linalg.norm(d)
        if norm_d < 1e-10:
            logging.info("INFO: Planes are parallel or coincide — there is no direct intersection.")
            return False
        self.direction = d / norm_d

        # Находим точку на прямой:
        # Решаем систему: plane1.equation(x)=0 и plane2.equation(x)=0
        # Составляем матрицу коэффициентов
        A = np.array([
            [self.plane1.A, self.plane1.B, self.plane1.C],
            [self.plane2.A, self.plane2.B, self.plane2.C],
            self.direction  # третья строка для однозначности
        ])
        b = np.array([-self.plane1.D, -self.plane2.D, 0.0])

        # Решаем через lstsq (устойчиво даже если матрица не квадратная)
        p0, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        self.point = p0  # точка на прямой

        # Для отрисовки сохраним отрезок
        self.t_min, self.t_max = self.t_min, self.t_max
        self.segment = np.array([
            self.point + self.t_min * self.direction,
            self.point + self.t_max * self.direction
        ])
        return True
        
    
    def parametric(self, t):
        """
        Параметрическое уравнение прямой: r(t) = point + t * direction
        """
        return self.point + t * self.direction


# ==================================================================================================

class CzechIntersectionsAlgorithm:
    def __init__(self, mesh):
        self.mesh = mesh
        self.new_mesh = self.mesh
        self.mesh_pairs = []
        self.sub_meshes = []
        
        self.border_candidates = {}
        self.border_candidates_list = {}
        self.border_planes = {}
        self.border_intersection_planes_line = {}
        self.border_intersection_pairs_cls = {}
        
        self.border_multiline = {}
        self.border_faces = {}
        
        self.border_cls_points = {}
        self.triangulation_faces = {}
        self.border_triangulation_faces = {}
        logging.info("Total face count: %s", len(self.mesh.faces))
    
    
    def pair_count(self):
        """
        Формирование списка пар зон (Mesh) для дальнейшей обработки.
        
        Алгоритм:
        ---------
        1. Получаем список имён всех зон в текущей mesh (self.mesh.zones).
        2. Проверяем, что зон как минимум две; если меньше — выводим ошибку.
        3. Формируем пары зон:
        - Первая пара: первая и вторая зона в списке.
        - Далее создаём цепочку: текущий префикс объединяет предыдущие зоны через дефис,
            к нему добавляется следующая зона, и формируется новая пара (префикс, новая зона).
        4. Все сформированные пары сохраняются в self.mesh_pairs.
        """
        zone_names = [zone.name for zone in self.mesh.zones]
        if len(zone_names) < 2:
            logging.error("Count of zones < 2, list: %s", zone_names)
            return
        current_prefix = zone_names[0]
        second_element = zone_names[1]

        self.mesh_pairs.append((current_prefix, second_element))
        current_prefix += "-" + second_element
        for name in zone_names[2:]:
            current_prefix += '-' + name
            self.mesh_pairs.append((current_prefix, name))
        
        logging.info("Mesh pairs: %s", self.mesh_pairs)
    
    
    def builder(self):
        self.pair_count()
        for pair in self.mesh_pairs:
            sub_faces = [face for face in self.mesh.faces if face.zone.name in pair]
            all_nodes = [node for face in sub_faces for node in face.nodes]
            all_edges = [edge for face in sub_faces for edge in face.edges]
            sub_nodes = list({id(node): node for node in all_nodes}.values())
            sub_edges = list({id(edge): edge for edge in all_edges}.values())
            
            sub_mesh = Mesh()
            sub_mesh.title = f"{pair[0]}-{pair[1]}"
            sub_mesh.zones = list(pair)
            sub_mesh.faces = sub_faces
            sub_mesh.edges = sub_edges
            sub_mesh.nodes = sub_nodes
            self.sub_meshes.append(sub_mesh)
            self.tnoi_filtering(sub_mesh = sub_mesh)
            self.defining_intersections(mesh_title = sub_mesh.title)
            self.make_triangulation(mesh_title = sub_mesh.title)
            self.new_mesh.faces = self.remove_intersections(sub_mesh = sub_mesh)
            self.new_mesh.faces += self.triangulation_faces[sub_mesh.title]
        self.mesh.faces = self.new_mesh.faces
    
    
    def tnoi_filtering(self, sub_mesh: Mesh, max_normal_angle = 0.99, max_threshold_dist = 0.3):
        """
        Алгоритм поиска пар кандидатов на пересечение граней по методам Lo & Wang, 2004; McLaurin et al., 2013.
        Результат: self.border_candidates[sub_mesh.title] = list of tuples (face1, face2)
        
        Шаги:
        1. Построение граничного словаря по зонам и граней.
        2. Вычисление центроидов и нормалей для каждой грани.
        3. Фильтрация пар граней по:
        a) пространственной близости (bounding box / расстояние между центроидами)
        b) ориентации нормалей (угол между нормалями)
        c) принадлежности разным зонам (если есть зоны)
        4. Сохранение пар кандидатов в список self.border_candidates[sub_mesh.title].
        5. Формирование списка уникальных граней-кандидатов для дополнительного анализа.
        """
        self.border_candidates[sub_mesh.title] = []
        
        # Шаг 1: вычисляем центроиды и нормали для всех граней
        face_data = []
        for face in sub_mesh.faces:
            nodes_coords = np.array([node.p for node in face.nodes])
            centroid = np.mean(nodes_coords, axis=0)
            
            # Вычисление нормали (для треугольника)
            if len(nodes_coords) >= 3:
                v1 = nodes_coords[1] - nodes_coords[0]
                v2 = nodes_coords[2] - nodes_coords[0]
                normal = np.cross(v1, v2)
                norm_length = np.linalg.norm(normal)
                if norm_length > 1e-12:
                    normal /= norm_length
                else:
                    normal = np.zeros(3)
            else:
                normal = np.zeros(3)
            
            face_data.append((face, centroid, normal))
        
        # Шаг 2: перебор всех уникальных пар граней
        for (face1, c1, n1), (face2, c2, n2) in combinations(face_data, 2):
            
            # Фильтр по зонам (если разные зоны, то можем рассматривать)
            if face1.zone == face2.zone:
                # Получаем множества glo_id ребер обеих граней
                shared_edges = {e.glo_id for e in face1.edges} & {e.glo_id for e in face2.edges}
                # Если есть общие ребра, пропускаем эту пару
                if shared_edges:
                    continue
            
            # Фильтр по расстоянию центроидов (bounding box / радиус)
            distance = np.linalg.norm(c1 - c2)
            threshold_dist = max_threshold_dist  # пример: подбирается под масштаб сетки
            if distance > threshold_dist:
                continue
            
            # Фильтр по нормалям (угол между нормалями)
            dot = np.dot(n1, n2)
            if abs(dot) > max_normal_angle:  # почти параллельные, можно игнорировать
                continue
            
            # Если пара прошла фильтры, добавляем как кандидата
            self.border_candidates[sub_mesh.title].append((face1, face2))
        
        # Шаг 3: формируем список уникальных граней-кандидатов
        unique_faces = set()
        for f1, f2 in self.border_candidates[sub_mesh.title]:
            unique_faces.add(f1)
            unique_faces.add(f2)
        
        self.border_candidates_list[sub_mesh.title] = list(unique_faces)
        logging.info("INFO: Candidates for checking the intersection formed, pairs count: %s, unique faces count: %s",
                 len(self.border_candidates[sub_mesh.title]),
                 len(self.border_candidates_list[sub_mesh.title]))


    def defining_plane(self, faces_pair, mesh_title):
        border_intersection_pairs  = []
        if not mesh_title in self.border_planes.keys():
            self.border_planes[mesh_title] = []
        
        plane1 = Plane(faces_pair[0])
        plane2 = Plane(faces_pair[1])
        
        intersection = self.defining_intersection_planes_line(plane1, plane2, mesh_title)
        plane1.intersection = intersection
        plane2.intersection = intersection
        
        if intersection:
            border_intersection_pairs.append(faces_pair)
            self.border_planes[mesh_title].append([plane1, plane2])
        
        return border_intersection_pairs


    def defining_intersection_planes_line(self, plane1, plane2, mesh_title):
        if not mesh_title in self.border_intersection_planes_line.keys():
            self.border_intersection_planes_line[mesh_title] = []
            
        line = Line(plane1, plane2)
        intersection = line.build_intersection_line()
        
        if intersection:
            self.border_intersection_planes_line[mesh_title].append(line)
            return True
        return False
        
        
    def classify_point_on_edge(self, p, edge, eps=1e-9):
        """
        Классифицирует точку пересечения p относительно ребра edge.
        Возвращает 0 (вне ребра), 1 (вершина), 2 (внутри ребра).
        """
        a = edge.nodes[0].p
        b = edge.nodes[1].p

        ab = b - a
        ap = p - a

        # Проверяем, лежит ли p на прямой AB (через скалярное произведение)
        cross = np.linalg.norm(np.cross(ab, ap))
        if cross > eps:  # не лежит на линии ребра
            return 0

        # Считаем параметр t для отрезка [a,b]
        ab_len2 = np.dot(ab, ab)
        if ab_len2 < eps:  # вырожденный отрезок
            return 1 if np.linalg.norm(p - a) < eps else 0

        t = np.dot(ap, ab) / ab_len2

        if t < -eps or t > 1 + eps:  # вне отрезка
            return 0
        if abs(t) < eps or abs(t - 1) < eps:  # совпадает с вершиной
            return 1
        return 2  # внутри ребра
    
    
    def edge_line_intersection(self, edge, line, eps=1e-9):
        """
        Ищет пересечение линии (line) с прямой, содержащей edge.
        Возвращает точку пересечения или None.
        """
        a = edge.nodes[0].p
        b = edge.nodes[1].p
        ab = b - a

        # Решаем систему: a + s*ab = line.point + t*line.direction
        M = np.column_stack([ab, -line.direction])
        rhs = line.point - a

        if np.linalg.matrix_rank(M) < 2:
            return None  # параллельны или совпадают

        try:
            sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
            s = sol[0]
            p_int = a + s * ab
            return p_int
        except Exception:
            return None

    
    def check_classifiers(self, cls):
        impossible_cases = [
            [0, 0, 1],
            [0, 0, 2],
            [1, 1, 2],
            [2, 2, 2],
            [1, 2, 2],
            [2, 2, 2]
        ]
        if cls in impossible_cases:
            logging.info("cls is impossible case, need correction")
            return 2

        if cls == [0, 0, 0]:
            logging.info("skip, points do not lie in triangles")
            return 0
        logging.info("classification is correct")
        return 1
    
    
    def intersect_collinear_edges(self, edge1, edge2, eps=1e-9):
        p1, p2 = edge1.nodes[0].p, edge1.nodes[1].p
        q1, q2 = edge2.nodes[0].p, edge2.nodes[1].p

        # берём направление
        d = p2 - p1
        d /= np.linalg.norm(d)

        def proj(v):
            return np.dot(v - p1, d)

        a1, a2 = proj(p1), proj(p2)
        b1, b2 = proj(q1), proj(q2)

        left = max(min(a1, a2), min(b1, b2))
        right = min(max(a1, a2), max(b1, b2))

        if left > right + eps:
            return None  # реально не пересекаются

        p_left = p1 + d * left
        p_right = p1 + d * right

        new_edge = Edge()
        new_edge.nodes = [Node(p_left), Node(p_right)]
        return new_edge
    

    def correct_classifiers(self, cls, eps=1e-8):
        """
        Корректирует классификацию в случае невозможного варианта
        (из-за численных ошибок). Возвращает исправленный список.
        """
        # Если случай невозможный (например 022) -> пробуем поправить
        cls_str = "".join(map(str, cls))

        # Случай 022 -> часто это 011 (ошибка из-за близости к вершине)
        if cls_str == "022":
            return [0, 1, 1]

        # Случай 122 -> невозможно, но заменим на 112
        if cls_str == "122":
            return [1, 1, 2]

        # Случай 002 -> скорее всего точка была на вершине -> исправляем на 011
        if cls_str == "002":
            return [0, 1, 1]

        # Если было 012 -> скорее всего 011
        if cls_str == "012":
            return [0, 1, 1]

        # Если всё корректно — возвращаем как есть
        return cls


    def try_neighbor(self, face_id, cls):
        """
        Пытается продолжить трассировку через соседнюю грань.
        """
        face = self.mesh.find_face_by_id(face_id)
        for edge in face.edges:
            for neighbor in edge.faces:
                if neighbor.glo_id != face_id:
                    logging.info("Let's try to continue tracing through the face of %s (the neighbor for %s)",
                                neighbor.glo_id, face_id)
                    # тут запускается повторная классификация для соседа
                    # (по сути, рекурсивный вызов классификатора)
                    return neighbor
        return None


    # ----------------------------------------------------------------------------------------------

    def defining_intersections(self, mesh_title):
        # Шаг 1. Построение плоскостей и линий пересечения по 3-м точкам для пар кандидатов
        border_intersection_pairs = []
        for pair in  self.border_candidates[mesh_title]:
            border_intersection_pairs += self.defining_plane(pair, mesh_title)
        
        self.border_candidates[mesh_title] = border_intersection_pairs
        logging.info("INFO: Candidates planes for checking the intersection have been formed, count: %s", len(self.border_planes[mesh_title]))
        logging.info("INFO: Candidates intersection planes line for checking the intersection have been formed, count: %s", len(self.border_intersection_planes_line[mesh_title]))
        logging.info("INFO: Candidates for checking the intersection have been formed after intersection checking, count: %s", len(self.border_candidates[mesh_title]))
        # Шаг 2. Вычисление отрезка пересечения для каждого из треугольников
        # Определяем и классифицируем отрезок линии пересечения плоскости и треугольника
        # Идем по каждому ребру (уравнение прямой + поиск пересечения) и классифицируем точки пресечения
        # 0 - точка пересечения лежит за пределами ребра
        # 1 - точка пересечениялежитввершине
        # 2 - точка пересечениялежитвнутрикромки
        # Сохраняем полученный результат в струкутре [face_id, [(edge_id, классификация, coords), ...]]
        self.border_cls_points[mesh_title] = []
        for pair, line in zip(self.border_candidates[mesh_title], self.border_intersection_planes_line[mesh_title]):
            line.build_intersection_line()

            new_pair = []

            for face in pair:
                edge_data = []

                for edge in face.edges:
                    p_int = self.edge_line_intersection(edge, line)
                    if p_int is not None:
                        cls = self.classify_point_on_edge(p_int, edge)
                        edge_data.append([edge.glo_id, cls, p_int.tolist()])

                if edge_data:
                    edge_ids, cls_values, point_coords = zip(*edge_data)
                else:
                   edge_ids, cls_values, point_coords = [], [], []

                #new_pair.append([face.glo_id, list(cls_values), list(point_coords), list(edge_ids)])
                new_pair.append([face.glo_id, edge_data])
                #print(edge_data)
            self.border_cls_points[mesh_title].append(new_pair)
        logging.info(
            "INFO: Candidates for checking the intersection have been formed after classification, count: %s",
            len(self.border_cls_points[mesh_title])
        )        
    
        # Шаг 3. Коррекция ошибок и построение линии пересечения
        self.border_multiline[mesh_title] = []
        self.border_faces[mesh_title] = []
        for bcp in self.border_cls_points[mesh_title]:
            # сортировка классификаторов
            bcp[0][1] = sorted(bcp[0][1], key=lambda data: data[1])
            bcp[1][1] = sorted(bcp[1][1], key=lambda data: data[1])
            cls1 = [cls[1] for cls in bcp[0][1]]
            cls2 = [cls[1] for cls in bcp[1][1]]
            
            # првоерка корректности классификаторов
            status = self.check_classifiers(cls = cls1)
            status2 = self.check_classifiers(cls = cls2)
            
            logging.info("First pair element: face id: %s, classification: %s, status: %s", bcp[0][0], cls1, status)
            logging.info("Second pair element: face id: %s, classification: %s, status: %s", bcp[1][0], cls2, status2)
            
            # построение отрезка пересечения двух граней
            if status and status2:
                sub_edge1 = Edge()
                sub_edge2 = Edge()
                # строим отрезок прямой пересечения плоскостей с первым треугольником
                sub_edge1.nodes = [Node(np.array(bcp[0][1][1][2])), Node(np.array(bcp[0][1][2][2]))]
                # строим отрезок прямой пересечения плоскостей со вторым треугольником
                sub_edge2.nodes = [Node(np.array(bcp[1][1][1][2])), Node(np.array(bcp[1][1][2][2]))]
                
                # грани которые пересекает искомый отрезок
                intersection_edge = self.intersect_collinear_edges(sub_edge1, sub_edge2)
                if intersection_edge:
                    intersection_edge.faces = [self.mesh.find_face_by_id(bcp[0][0]), self.mesh.find_face_by_id(bcp[1][0])]
                    self.border_multiline[mesh_title].append(intersection_edge)
                    self.border_faces[mesh_title].append((intersection_edge, [self.mesh.find_face_by_id(bcp[0][0]), self.mesh.find_face_by_id(bcp[1][0])]))
                # self.border_multiline[mesh_title].append(sub_edge1)
                # self.border_multiline[mesh_title].append(sub_edge2)
                # print(bcp[1][1][0])
                # print(bcp[1][1][0])
                #break
            # нужна коррекция через другие грани
            else:
                pass
                # # шаг "соседнего треугольника" — если классификация всё ещё невалидна
                # corrected_edge = self.try_neighbor(face_id=bcp[0][0], cls=cls1)

                # if corrected_edge:
                #     corrected_edge.faces = [
                #         self.mesh.find_face_by_id(bcp[0][0]),
                #         self.mesh.find_face_by_id(bcp[1][0])
                #     ]
                #     self.border_multiline[mesh_title].append(corrected_edge)
                #     self.border_faces[mesh_title].append(
                #         (corrected_edge, [
                #             self.mesh.find_face_by_id(bcp[0][0]),
                #             self.mesh.find_face_by_id(bcp[1][0])
                #         ])
                #     )
                # else:
                #     logging.error("Failed to correct intersection for faces %s and %s",
                #                 bcp[0][0], bcp[1][0])
    
    
    # ----------------------------------------------------------------------------------------------
    
    def make_triangulation(self, mesh_title):
        """
        Выполняет триангуляцию всех граней из self.border_faces[mesh_title].
        Результат сохраняется в self.triangulation_faces[mesh_title].
        """
        self.triangulation_faces[mesh_title] = []
        self.border_triangulation_faces[mesh_title] = []

        if mesh_title not in self.border_faces:
            return

        for intersection_edge, faces in self.border_faces[mesh_title]:
            for face in faces:
                # исходные вершины треугольника
                tri_vertices = [node for node in face.nodes]

                # добавляем точки пересечения в список вершин
                new_vertices = self.insert_intersection_points(tri_vertices, intersection_edge)

                # проекция многоугольника в плоскость грани
                polygon_2d, to3d = self.project_to_face_plane(new_vertices)

                # триангуляция в 2D
                triangles_2d = self.ear_clipping(polygon_2d)

                # поднимаем обратно в 3D и создаём новые Face
                pair = []
                for tri in triangles_2d:
                    face_new = Face()
                    face_new.nodes = [Node(to3d(p)) for p in tri]
                    face_new.zone = Zone(mesh_title)
                    self.triangulation_faces[mesh_title].append(face_new)
                    pair.append(face_new)
                self.border_triangulation_faces[mesh_title].append(pair)
        print(len(self.border_triangulation_faces[mesh_title]))


    # ----------------------------------------------------------------------------------------------

    def insert_intersection_points(self, tri_vertices, edge):
        """
        Вставляет точки пересечения (edge.nodes[0], edge.nodes[1])
        в список вершин треугольника tri_vertices (Node).
        """
        verts = tri_vertices.copy()

        for pt_node in edge.nodes:
            pt = pt_node.p
            for i in range(len(verts)):
                a, b = verts[i].p, verts[(i + 1) % len(verts)].p
                if self.point_on_segment(pt, a, b):
                    verts.insert(i + 1, Node(pt.copy()))
                    break

        return verts

    
    # ----------------------------------------------------------------------------------------------

    def project_to_face_plane(self, vertices):
        """
        Строит локальную 2D-систему координат для проекции многоугольника в плоскость Face.
        Возвращает:
          polygon_2d: список np.array([x,y])
          to3d: функция для обратного перехода
        """
        # берём первые три точки
        p0, p1, p2 = vertices[0].p, vertices[1].p, vertices[2].p

        # базис в плоскости
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        axis_x = v1 / np.linalg.norm(v1)
        axis_y = np.cross(normal, axis_x)

        def to2d(p):
            v = p - p0
            return np.array([np.dot(v, axis_x), np.dot(v, axis_y)])

        def to3d(p2d):
            return p0 + axis_x * p2d[0] + axis_y * p2d[1]

        polygon_2d = [to2d(v.p) for v in vertices]
        return polygon_2d, to3d

    
    # ----------------------------------------------------------------------------------------------

    def ear_clipping(self, polygon):
        """
        Триангуляция многоугольника методом "обрезания уха" в 2D.
        polygon — список np.array([x, y]).
        Возвращает список треугольников (каждый треугольник — список точек 2D).
        """
        triangles = []
        verts = polygon.copy()

        def is_convex(a, b, c):
            return np.cross(b - a, c - b) >= 0

        while len(verts) > 3:
            n = len(verts)
            ear_found = False
            for i in range(n):
                a, b, c = verts[i - 1], verts[i], verts[(i + 1) % n]
                if is_convex(a, b, c):
                    triangles.append([a, b, c])
                    del verts[i]
                    ear_found = True
                    break
            if not ear_found:
                break

        if len(verts) == 3:
            triangles.append(verts)

        return triangles

    
    # ----------------------------------------------------------------------------------------------

    def point_on_segment(self, pt, a, b, eps=1e-9):
        """
        Проверка, что точка pt лежит на отрезке (a, b).
        """
        cross = np.cross(b - a, pt - a)
        if np.linalg.norm(cross) > eps:
            return False
        dot = np.dot(pt - a, b - a)
        if dot < 0:
            return False
        if dot > np.dot(b - a, b - a):
            return False
        return True

    
    # ----------------------------------------------------------------------------------------------
    
    def remove_intersections(self, sub_mesh):
        remove_faces = set()
        new_faces = []
        [remove_faces.add(f[1][0].glo_id) for f in self.border_faces[sub_mesh.title]]
        [remove_faces.add(f[1][1].glo_id) for f in self.border_faces[sub_mesh.title]]
        for cur_face in sub_mesh.faces:
            if not cur_face.glo_id in remove_faces:
                new_faces.append(cur_face)
        
        logging.info("Total face count: %s", len(self.mesh.faces))
        logging.info("Total face count after remove intersections: %s", len(new_faces))
        return new_faces
        
                
    

# ==================================================================================================

if __name__ == '__main__':
    pass

# ==================================================================================================
