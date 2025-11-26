import logging
import math

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import defaultdict

from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node


class CzechClassify:
    def __init__(self, candidates: Tuple, checked_pairs=None):
        # пары, которые могут пересекаться
        self.face1, self.face2 = candidates
        self.impossible_couples = []
        self.points = []
        self.checked_pairs = checked_pairs if checked_pairs is not None else set()
        self.impossible_cases = {"001", "002", "012", "122", "222"}
        self.case000 = "000"
    
    def segment_triangle_intersection(self, a, b, tri) -> List[np.ndarray] or None:
        p0 = tri.nodes[0].p
        u = tri.nodes[1].p - p0
        v = tri.nodes[2].p - p0
        n = np.cross(u, v)
        w0 = a - p0
        w1 = b - p0

        # Проверка на одну сторону
        g0 = np.dot(n, w0)
        g1 = np.dot(n, w1)
        if g0 * g1 > 0:
            return None

        # Точка пересечения с плоскостью
        t = g0 / (g0 - g1)
        P = a + t * (b - a)

        # Проверка, принадлежит ли P треугольнику
        uu = np.dot(u, u)
        uv = np.dot(u, v)
        vv = np.dot(v, v)
        w = P - p0
        wu = np.dot(w, u)
        wv = np.dot(w, v)
        D = uv * uv - uu * vv
        s = (uv * wv - vv * wu) / D
        if s < 0 or s > 1:
            return None
        t2 = (uv * wu - uu * wv) / D
        if t2 < 0 or s + t2 > 1:
            return None
        return [P]

    def coplanar_intersection_segment(self, f1, f2):
        pts = []

        # Ребра f1 vs f2
        for i in range(3):
            p, q = f1.nodes[i].p, f1.nodes[(i+1)%3].p
            ip = self.segment_triangle_intersection(p, q, f2)
            if ip: pts.extend(ip)

        # Ребра f2 vs f1
        for i in range(3):
            p, q = f2.nodes[i].p, f2.nodes[(i+1)%3].p
            ip = self.segment_triangle_intersection(p, q, f1)
            if ip: pts.extend(ip)

        if len(pts) == 0:
            return None
        if len(pts) == 1:
            return [pts[0], pts[0]]

        # Отрезок — две крайние точки
        pts = np.array(pts)
        d = np.linalg.norm(pts[None,:,:] - pts[:,None,:], axis=2)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        return [pts[i], pts[j]]
    
    # def classify(self):
    #     f1, f2 = self.face1, self.face2

    #     # расчет нормалей
    #     if f1.normal is None:
    #         f1.calculate_normal()
    #     if f2.normal is None:
    #         f2.calculate_normal()
            
    #     # уравнение плоскости для f2
    #     p1 = f1.nodes[0].p
    #     n1 = f1.normal
    #     d1 = -np.dot(n1, p1)
        
    #     p2 = f2.nodes[0].p
    #     n2 = f2.normal
    #     d2 = -np.dot(n2, p2)

    #     classification = []
    #     intersection_points = []
        
    #     # --- 0. Параллельность / Копланарность ---
    #     if self.plane_parallel(n1, n2):
    #         if self.plane_coplanar(d1, d2, n1, n2):
    #             # --- копланарный случай (настоящий случай 222 или 000) ---
    #             seg = self.coplanar_intersection_segment(f1, f2)
    #             if seg is None:
    #                 return False, []
    #             else:
    #                 return True, seg   # <-- возвращаем отрезок [p1, p2]
    #         else:
    #             # плоскости параллельны, но не копланарны → не пересекаются
    #             return False, []
        
    #     # 1. для каждого ребра грани 1 вычислить пересечение с плоскостью грани 2
    #     for i in range(3):
    #         a = f1.nodes[i].p
    #         b = f1.nodes[(i + 1) % 3].p

    #         result = self.edge_plane_intersection(a, b, n2, d2)
    #         if result is not None:
    #             t, point = result
    #             intersection_points.append(point)
    #             classification.append(self.classify_point(t))
        
    #     if len(classification) < 3:
    #         classification += ["0"] * (3 - len(classification))
    #     code = "".join(sorted(classification))
    #     print(code)
        
        
    #     # --- 3. Применяем таблицу классификаций ---
    #     if code == self.case000:
    #         # Нет пересечения
    #         return False, []

    #     elif code in self.impossible_cases:
    #         # Невозможный случай — пробуем уточнить по соседним граням
    #         self.impossible_couples.append((f1, f2))
    #         return self.recheck_with_neighbours(f1, f2)

    #     else:
    #         # Валидное пересечение
    #         self.points = intersection_points
    #         return True, intersection_points
    def classify(self):
        f1, f2 = self.face1, self.face2

        # Расчёт нормалей
        if f1.normal is None: f1.calculate_normal()
        if f2.normal is None: f2.calculate_normal()

        # Уравнения плоскостей
        p1, n1 = f1.nodes[0].p, f1.normal
        p2, n2 = f2.nodes[0].p, f2.normal
        d1, d2 = -np.dot(n1, p1), -np.dot(n2, p2)

        intersection_points = []
        classification = []

        # --- 0. Параллельность / копланарность ---
        if self.plane_parallel(n1, n2):
            if self.plane_coplanar(d1, d2):
                seg = self.coplanar_intersection_segment(f1, f2)
                code = "222" if seg else "000"
                return False, seg if seg else []
            else:
                return False, []

        # --- 1. Пересечение рёбер f1 с плоскостью f2 ---
        for i in range(3):
            a, b = f1.nodes[i].p, f1.nodes[(i+1) % 3].p
            result = self.edge_plane_intersection(a, b, n2, d2)
            if result:
                t, pt = result
                intersection_points.append(pt)
                classification.append(self.classify_point(t))

        if len(classification) < 3:
            classification += ["0"] * (3 - len(classification))
        code = "".join(sorted(classification))

        # --- 2. Обработка результатов ---
        if code == self.case000:
            return False, []
        elif code in self.impossible_cases:
            self.impossible_couples.append((f1, f2))
            self.checked_pairs.add((id(f1), id(f2)))
            has_intersection, pts = self.recheck_with_neighbours(f1, f2)
            return False if has_intersection else "000", pts
        else:
            self.points = intersection_points
            return True, intersection_points
    
    # -----------------------------------------------------------
    # def recheck_with_neighbours(self, f1, f2):
    #     """
    #     Повторная проверка пересечения через соседние грани,
    #     если текущая пара попала в невозможный случай.
    #     """
    #     logging.debug("Called imp")
    #     # Получаем соседей первой грани
    #     neighbours = self.face_neighbours(f1)
    #     # Перебираем соседние грани
    #     for nb in neighbours:
    #         # Пропускаем, если сравниваем саму f2
    #         if nb is f2:
    #             continue

    #         sub_result = CzechClassify((nb, f2))
    #         has_intersection, pts = sub_result.classify()

    #         if has_intersection:
    #             # Если сосед пересекается — уточняем результат
    #             self.points = pts
    #             return True, pts

    #     # Ни один сосед не дал пересечение — подтверждаем отсутствие
    #     return False, []
    def recheck_with_neighbours(self, f1, f2):
        neighbours = self.face_neighbours(f1)
        for nb in neighbours:
            if nb is f2:
                continue
            pair_key = (id(nb), id(f2))
            if pair_key in self.checked_pairs:
                continue  # уже проверяли
            self.checked_pairs.add(pair_key)
            sub_result = CzechClassify((nb, f2), checked_pairs=self.checked_pairs)
            code, pts = sub_result.classify()
            if pts:
                self.points = pts
                return True, pts
        return False, []


    @staticmethod
    def edge_plane_intersection(a, b, n, d, eps=1e-9):
        """
        Вычисляет пересечение ребра AB с плоскостью n·x + d = 0.
        Возвращает (t, точку) или None, если пересечения нет или оно вне отрезка.
        """
        ab = b - a
        denom = np.dot(n, ab)
        if abs(denom) < eps:
            return None  # ребро параллельно плоскости

        t = -(np.dot(n, a) + d) / denom
        if t < -eps or t > 1 + eps:
            return None  # пересечение вне ребра

        point = a + t * ab
        return t, point


    @staticmethod
    def classify_point(t, eps=1e-6):
        """
        Классифицирует параметр пересечения t вдоль ребра:
        "0" — вне ребра;
        "1" — вершина (t≈0 или t≈1);
        "2" — внутри ребра (0<t<1).
        """
        if t < -eps or t > 1 + eps:
            return "0"
        elif abs(t) <= eps or abs(1 - t) <= eps:
            return "1"
        else:
            return "2"

    @staticmethod
    def plane_parallel(n1, n2, eps=1e-12):
        # |n1 × n2| == 0 → нормали коллинеарны
        return np.linalg.norm(np.cross(n1, n2)) < eps

    @staticmethod
    def plane_coplanar(d1, d2, eps=1e-12):
        # Плоскости параллельны: проверяем совпадение d
        return abs(d1 - d2) < eps

    def face_neighbours(self, face):
        """
        Возвращает список соседних граней для заданной грани face.
        Ожидается, что у объекта Face есть метод .neighbourhood(),
        который возвращает соседей по общим рёбрам.
        """
        if hasattr(face, "neighbourhood"):
            return face.neighbourhood()
        return []


@dataclass
class Node:
    node_id: int = 0
    # ссылка на левую/правую ноды
    children_nodes: Tuple[Node, Node] = ()
    # ограничивающий объем AABB
    bouding_box: Tuple[np.ndarray, np.ndarray] = ()
    is_leaf: bool = False
    faces: List[Face] = field(default_factory=list)


class BVHTree:
    def __init__(self, mesh, leaf_in_node: int = 1):
        self.mesh = mesh
        self.leaf_in_node = leaf_in_node
        self.nodes_counter = 0
        self.nodes = []
        self.root_node = None
        self.faces_to_fix = defaultdict(list)
    
    
    def aabb(self, nodes_coord: np.ndarray):
        node_min = nodes_coord.min(axis=0)
        node_max = nodes_coord.max(axis=0)
        centroid = (node_max + node_min) * 0.5
        return (node_min, node_max), centroid


    def merge_aabb(self, box1, box2):
        if box1 is None:
            return box2
        if box2 is None:
            return box1
        minv = np.minimum(box1[0], box2[0])
        maxv = np.maximum(box1[1], box2[1])
        return (minv, maxv)
    
    
    def surface_area_aabb(self, box):
        # np.array(len_x, len_y, len_z)
        side_lengths = box[1] - box[0]
        # S (AABB) = 2 * (xy + xz + yz)
        return 2.0 * (side_lengths[0] * side_lengths[1] + side_lengths[0] * side_lengths[2] + side_lengths[1] * side_lengths[2])
    
    
    def volume(self, box):
        # np.array(len_x, len_y, len_z)
        side_lengths = box[1] - box[0]
        # V (AABB) = x * y * z
        return side_lengths[0] * side_lengths[1] * side_lengths[2]
    
    
    def compute_faces_bounding_box(self, faces: List[Face]):
        all_coords = np.vstack([node.p for face in faces for node in face.nodes])
        all_coords_unique = np.unique(all_coords, axis=0)
        box, centroid = self.aabb(all_coords_unique)
        return box, centroid
    
    
    def aabb_overlap(self, box1, box2):
        for i in range(3):
            if box1[1][i] <= box2[0][i] or box2[1][i] <= box1[0][i]:
                return False
        return True
    
    
    def axis_of_largest_extent(self, box):
        extent = box[1] - box[0]
        return np.argmax(extent)


    def early_split_clipping(self, face, SAmax):
        """
        Алгоритм ранней раздельной обрезки (Early Split Clipping) для одного треугольника.

        Возвращает:
            boxes, centroids — списки AABB и центроидов
        """
        boxes = []
        centroids = []
        stack = [face]  # стек с исходным примитивом

        while stack:
            current = stack.pop()
            nodes_coord = np.array([node.p for node in current.nodes])
            box, centroid = self.aabb(nodes_coord)
            SA = self.surface_area_aabb(box)

            if SA <= SAmax:
                boxes.append(box)
                centroids.append(centroid)
                continue

            # делим по оси наибольшей протяжённости
            axis = self.axis_of_largest_extent(box)
            split_pos = 0.5 * (box[0][axis] + box[1][axis])

            # обрезаем примитив на две половины
            P_neg = self.clip_primitive(nodes_coord, axis, split_pos, positive=False)
            P_pos = self.clip_primitive(nodes_coord, axis, split_pos, positive=True)

            # создаём новые "Face" только если валидные вершины
            if P_neg is not None and P_neg.shape[0] >= 3:
                new_face_neg = type(current)()  # создаём пустой Face
                new_face_neg.nodes = [type(current.nodes[0])(p) for p in P_neg]
                stack.append(new_face_neg)

            if P_pos is not None and P_pos.shape[0] >= 3:
                new_face_pos = type(current)()
                new_face_pos.nodes = [type(current.nodes[0])(p) for p in P_pos]
                stack.append(new_face_pos)

        return boxes, centroids


    def clip_primitive(self, coords: np.ndarray, axis: int, split_pos: float, positive: bool = True):
        if positive:
            mask = coords[:, axis] >= split_pos
        else:
            mask = coords[:, axis] <= split_pos

        if np.all(mask) or not np.any(mask):
            # разделение не произошло — возвращаем исходные координаты
            return coords.copy()

        # "обрезаем" координаты по split_pos
        new_coords = coords.copy()
        if positive:
            new_coords[:, axis] = np.clip(new_coords[:, axis], split_pos, None)
        else:
            new_coords[:, axis] = np.clip(new_coords[:, axis], None, split_pos)

        return new_coords

    
    def prepare_mesh(self, esc_enable=False):
        nodes_coord = np.array([node.p for node in self.mesh.faces[0].nodes])
        box, _ = self.aabb(nodes_coord)
        SAmax = self.surface_area_aabb(box)
        logging.debug("prepare_mesh, Smax: %s", SAmax)
        
        for face in self.mesh.faces:
            nodes_coord = np.array([node.p for node in face.nodes])
            if esc_enable:
                boxes, centroids = self.early_split_clipping(face, SAmax)
                face.bounding_boxes = boxes
                face.centroids = centroids
                face.calculate_normal()
                # boxes_max =  max(self.surface_area_aabb(b) for b in boxes)
                # if SAmax > boxes_max:
                #     SAmax = boxes_max
            else:
                box, centroid = self.aabb(nodes_coord)
                face.bounding_boxes = [box]
                face.centroids = [centroid]
                face.calculate_normal()
    
    
    def build_tree(self, split_func: str = "sah"):
        # создаем корень
        root_box, _ = self.compute_faces_bounding_box(self.mesh.faces)
        self.root_node = Node(
            node_id=self.nodes_counter,
            bouding_box=root_box,
            # т.к. не лист
            faces=self.mesh.faces,
            is_leaf=False
        )
        
        # создаем рекурсивно дочерние узлы 
        self.nodes.append(self.root_node)
        self.nodes_counter += 1
        self.build_node(node=self.root_node, split_func=split_func)
    
    
    def build_node(self, node: Node, split_func: str):
        # если в faces кол-во <= leaf_in_node, то помечаем текущую ноду как лист
        faces = node.faces
        if len(faces) <= self.leaf_in_node:
            node.is_leaf = True
            return
        
        # ищем разбиение по выбранной функции [sah, vah, median, half] 
        axis, index, (left_box, right_box) = self.find_best_split(faces=faces, split_func=split_func)
        
        # ?
        if axis == -1 or index is None:
            node.is_leaf = True
            return
        
        # сортировка по максимому осей
        # сортировка по максимальной координате всех боксов face
        sorted_faces = sorted(
            faces, key=lambda f: max(box[1][axis] for box in f.bounding_boxes)
        )
        left_faces = sorted_faces[:index]
        right_faces = sorted_faces[index:]
        
        # создаем левый и правый узлы
        left_node = Node(
            node_id=self.nodes_counter,
            bouding_box=left_box,
            faces=left_faces,
        )
        self.nodes.append(left_node)
        self.nodes_counter += 1
        
        right_node = Node(
            node_id=self.nodes_counter,
            bouding_box=right_box,
            faces=right_faces,
        )
        self.nodes.append(right_node)
        self.nodes_counter += 1
        
        # к текущему узлу добавляем ссылки на дочерние узлы
        node.children_nodes = (left_node, right_node)
        
        # рекурсивно строит левую и правую ветку каждого уровня
        self.build_node(left_node, split_func=split_func)
        self.build_node(right_node, split_func=split_func)
    
    
    # def find_best_split(self, faces: List[Face], split_func: str):
    #     # best_score — это минимальное значение критерия разбиения (SAH или VAH) на текущем наборе faces.
    #     # изначально мы не знаем ни одного разбиения, поэтому ставим его в бесконечность, чтобы любое первое вычисленное значение стало «лучшим».
    #     # когда в цикле мы найдём score меньше best_score, мы обновим best_score.
    #     best_score = float("inf")
    #     # хранит ось (0=X, 1=Y, 2=Z), вдоль которой найдено лучшее разбиение.
    #     # -1 используется как «флаг», что ещё не найдено ни одного разбиения.
    #     # eсли в конце остаётся -1, значит разбиение невозможно (например, faces слишком мало), и узел станет листом.
    #     best_axis = -1
    #     # индекс разбиения в отсортированном массиве faces.
    #     # разделяет массив на [0..best_index] | [best_index..n].
    #     # изначально None, пока мы не найдём хотя бы одно допустимое разбиение.
    #     best_index = None
    #     # (left_box, right_box) для текущего лучшего разбиения.
    #     # эти AABB сразу будут использоваться для создания дочерних узлов, чтобы не пересчитывать их заново.
    #     # изначально None, потому что ещё нет вычисленных боксов.
    #     best_boxes = None
        
    #     n_faces = len(faces)
    #     # нельзя делить
    #     if n_faces <= 1:
    #         return best_axis, best_index, best_boxes
        
    #     # x=0, y=1, z=2
    #     for axis in range(3):
    #         # сортируем faces по координате максимальной точки bounding box вдоль axis
    #         # sorted_faces[i].bounding_boxes[0][1] = vmax
    #         sorted_faces = sorted(faces, key=lambda f: f.bounding_boxes[0][1][axis])

    #         # подготовка массивов для sweep
    #         left_bounds = [None] * n_faces   # левая граница [0..i]
    #         right_bounds = [None] * n_faces  # правая граница [i..n]

    #         # sweep справа налево: вычисляем right_bounds[i] = AABB всех элементов с i+1 до конца
    #         box = None
    #         for i in range(n_faces - 1, 0, -1):
    #             box = self.merge_aabb(box, sorted_faces[i].bounding_boxes[0])
    #             right_bounds[i - 1] = box

    #         # weep слева направо: вычисляем left_bounds и оцениваем критерий
    #         box = None
    #         for i in range(1, n_faces):
    #             box = self.merge_aabb(box, sorted_faces[i - 1].bounding_boxes[0])

    #             # вычисляем "стоимость" разбиения
    #             if split_func == "sah":
    #                 # стандартный SAH = площадь бокса * количество примитивов
    #                 score = self.surface_area_aabb(box) * i + self.surface_area_aabb(right_bounds[i - 1]) * (n_faces - i)
    #             elif split_func == "vah":
    #                 # Volume-Area Heuristic = объём бокса * количество примитивов
    #                 left_extent = box[1] - box[0]
    #                 right_extent = right_bounds[i - 1][1] - right_bounds[i - 1][0]
    #                 score = np.prod(left_extent) * i + np.prod(right_extent) * (n_faces - i)
    #             else:
    #                 raise ValueError(f"Unknown split_func: {split_func}")

    #             # запоминаем лучшее разбиение
    #             if score < best_score:
    #                 best_score = score
    #                 best_axis = axis
    #                 best_index = i
    #                 best_boxes = (box, right_bounds[i - 1])

    #     return best_axis, best_index, best_boxes
    
    
    def find_best_split(self, faces: List[Face], split_func: str):
        best_score = float("inf")
        best_axis = -1
        best_index = None
        best_boxes = None
        n_faces = len(faces)
        if n_faces <= 1:
            return best_axis, best_index, best_boxes

        for axis in range(3):
            # сортируем по максимальной координате всех AABB face
            sorted_faces = sorted(
                faces,
                key=lambda f: max(box[1][axis] for box in f.bounding_boxes)
            )

            right_bounds = [None] * n_faces

            # sweep справа налево
            box = None
            for i in range(n_faces - 1, 0, -1):
                # объединяем все bounding_boxes face[i] в один
                b = None
                for bb in sorted_faces[i].bounding_boxes:
                    b = self.merge_aabb(b, bb)
                box = self.merge_aabb(box, b)
                right_bounds[i - 1] = box

            # sweep слева направо
            box = None
            for i in range(1, n_faces):
                b = None
                for bb in sorted_faces[i - 1].bounding_boxes:
                    b = self.merge_aabb(b, bb)
                box = self.merge_aabb(box, b)

                # вычисляем стоимость
                if split_func == "sah":
                    score = self.surface_area_aabb(box) * i + self.surface_area_aabb(right_bounds[i - 1]) * (n_faces - i)
                elif split_func == "vah":
                    left_extent = box[1] - box[0]
                    right_extent = right_bounds[i - 1][1] - right_bounds[i - 1][0]
                    score = np.prod(left_extent) * i + np.prod(right_extent) * (n_faces - i)
                else:
                    raise ValueError(f"Unknown split_func: {split_func}")

                if score < best_score:
                    best_score = score
                    best_axis = axis
                    best_index = i
                    best_boxes = (box, right_bounds[i - 1])

        return best_axis, best_index, best_boxes

    
    
    def traversal_tree(self):
        if self.root_node is None:
            return []

        candidate_pairs = []
        checked_pairs = set()  # для избежания дубликатов

        # стек элементов для обхода
        # элемент = (node, ancestors_to_check)
        stack = [(self.root_node, [])]

        while stack:
            node, ancestors = stack.pop()

            # # === 1. Если узел листовой ===
            # if node.is_leaf:
            #     faces = node.faces
            #     # внутри листа проверяем пары
            #     for i in range(len(faces)):
            #         for j in range(i + 1, len(faces)):
            #             f1, f2 = faces[i], faces[j]
            #             pair_id = tuple(sorted((f1.glo_id, f2.glo_id)))
            #             if pair_id not in checked_pairs:
            #                 # проверяем AABB
            #                 if self.check_intersection(f1, f2):
            #                     candidate_pairs.append((f1, f2))
            #                 checked_pairs.add(pair_id)

            #     # === 2. Проверяем соседей через родителей ===
            #     for ancestor, sibling_node in ancestors:
            #         if sibling_node is None:
            #             continue
            #         if self.aabb_overlap(node.bouding_box, sibling_node.bouding_box):
            #             # спускаемся по второму ребенку до листа и добавляем все пары
            #             leaves = self.collect_leaf_faces(sibling_node)
            #             for f1 in faces:
            #                 for f2 in leaves:
            #                     pair_id = tuple(sorted((f1.glo_id, f2.glo_id)))
            #                     if pair_id not in checked_pairs:
            #                         if self.check_intersection(f1, f2):
            #                             candidate_pairs.append((f1, f2))
            #                         checked_pairs.add(pair_id)
            #     continue
            # === 1. Если узел листовой ===
            if node.is_leaf:
                # Пары внутри самого листа
                checked_pairs, candidate_pairs = self.process_face_pairs(node.faces, node.faces, checked_pairs, candidate_pairs)

                # === 2. Проверяем соседей через родителей ===
                for ancestor, sibling_node in ancestors:
                    if sibling_node is None:
                        continue
                    if self.aabb_overlap(node.bouding_box, sibling_node.bouding_box):
                        leaves = self.collect_leaf_faces(sibling_node)
                        checked_pairs, candidate_pairs = self.process_face_pairs(node.faces, leaves, checked_pairs, candidate_pairs)
                continue

            # === 3. Если узел не лист ===
            left_node, right_node = node.children_nodes

            # формируем список родителей для поднятия
            new_ancestors_left = list(ancestors)
            new_ancestors_right = list(ancestors)

            if right_node:
                new_ancestors_left.append((node, right_node))
            if left_node:
                new_ancestors_right.append((node, left_node))

            # помещаем в стек
            if left_node:
                stack.append((left_node, new_ancestors_left))
            if right_node:
                stack.append((right_node, new_ancestors_right))

        return candidate_pairs


    def collect_leaf_faces(self, node):
        """
        Рекурсивно собирает все faces в листах под узлом.
        """
        if node.is_leaf:
            return node.faces
        faces = []
        for child in node.children_nodes:
            if child:
                faces.extend(self.collect_leaf_faces(child))
        return faces
    
    
    def process_face_pairs(self, faces1, faces2, checked_pairs, candidate_pairs):
        for f1 in faces1:
            for f2 in faces2:
                if f1 is f2:  # защита от самих себя
                    continue

                pair_id = tuple(sorted((f1.glo_id, f2.glo_id)))
                if pair_id in checked_pairs:
                    continue

                if self.check_intersection(f1, f2):
                    #print(f1.glo_id, f2.glo_id)
                    # чехи
                    cs = CzechClassify((f1,f2), checked_pairs=checked_pairs)
                    result, intersection_points = cs.classify()
                    if result:
                        candidate_pairs.append(((f1, f2), intersection_points))
                        self.faces_to_fix[f1.glo_id].append(intersection_points)
                        self.faces_to_fix[f2.glo_id].append(intersection_points)
                            

                checked_pairs.add(pair_id)
        
        return checked_pairs,candidate_pairs
    
    
    def check_intersection(self, f1: Face, f2: Face):
        # проверка пересечения AABB
        box1, box2 = f1.bounding_boxes[0], f2.bounding_boxes[0]
        for i in range(3):
            if box1[1][i] <= box2[0][i] or box2[1][i] <= box1[0][i]:
                return False  # нет пересечения (или только касание)

        # проверка на отсутствие общих вершин
        f1_vertex_ids = {n.glo_id for n in f1.nodes}
        f2_vertex_ids = {n.glo_id for n in f2.nodes}
        if f1_vertex_ids & f2_vertex_ids:
            return False  # есть общие вершины

        return True
    
    
    def build_graph(self, node, graph=None):
        if graph is None:
            graph = nx.DiGraph()

        # подпись узла
        if node.is_leaf:
            #node_label = f"Leaf\nID:{node.node_id}\nFace_ids:{', '.join(str(face.glo_id) for face in node.faces)}"
            node_label = f"Leaf\nID:{node.node_id}\nFaces: {len(node.faces)}"
        else:
            node_label = f"Node\nID:{node.node_id}"

        # добавляем узел в граф
        graph.add_node(node.node_id, label=node_label, is_leaf=node.is_leaf)

        # если есть дочерние узлы, добавляем ребра и рекурсивно строим граф
        if node.children_nodes:
            left_node, right_node = node.children_nodes
            if left_node:
                graph.add_edge(node.node_id, left_node.node_id)
                self.build_graph(left_node, graph)
            if right_node:
                graph.add_edge(node.node_id, right_node.node_id)
                self.build_graph(right_node, graph)

        return graph

    
if __name__ == '__main__':
    #mesh = Mesh("tests/examples/small_sphere_double.dat")
    mesh = Mesh("tests/examples/sphere_double.dat")
    # bvh = BVHTree(mesh)
    # bvh.prepare_mesh(esc_enable=False)
    # bvh.build_tree()
    # graph = bvh.build_graph(bvh.root_node)
    #face.glo_id in [149, 75, 149, 66]:
    #[7349, 7481]
    #[4838 4841]
    cs = CzechClassify((mesh.find_face_by_id(4838), mesh.find_face_by_id(4841)))
    result, points = cs.classify()
    print(result)
