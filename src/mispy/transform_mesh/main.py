import logging
import time
import math


import numpy as np
from numpy import linalg as LA
from typing import List, Tuple, Optional
from collections import defaultdict
from itertools import combinations

from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node


class CzechIntersectionsAlgorithm:
    def __init__(self, ):
        pass


class BVH:
    def __init__(self, mesh, eps_scale=1e-9, aspect_threshold=2.0, max_leaf_size=2, max_depth=64):
        """
        BVH-конструктор
        :param mesh: объект Mesh
        :param eps_scale: масштаб для расширения bbox (устранение float ошибок)
        :param aspect_threshold: порог отношения сторон (aspect ratio)
        """
        self.mesh = mesh
        
        self.eps_scale = eps_scale
        self.aspect_threshold = aspect_threshold
        self.max_leaf_size = max_leaf_size
        self.max_depth = max_depth
        self.primitives = []  # все AABB-примитивы
        self.nodes = []       # узлы BVH
        self.candidates = []  # candidate-пары (face_id_a, face_id_b)


    def preprocessing(self, face):
        """
        Препроцессинг треугольной грани:
        - Вычисление AABB, центроида, площади
        - Расширение bbox на eps
        - Early Split Clipping при вытянутых треугольниках
        """
        coords = np.array([node.p for node in face.nodes], dtype=float)
        p0, p1, p2 = coords

        # базовые вычисления
        vmin = coords.min(axis=0)
        vmax = coords.max(axis=0)
        centroid = coords.mean(axis=0)
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        scale = np.linalg.norm(vmax - vmin)
        eps = self.eps_scale * (scale if scale > 0 else 1.0)
        vmin -= eps
        vmax += eps
        zone_name = face.zone.name if hasattr(face, "zone") else None
        adjacency_info = [face.neighbour(edge) for edge in face.edges]

        # вычисляем соотношение сторон (aspect ratio)
        edge_lengths = [np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p0 - p2)]
        longest = max(edge_lengths)
        # высота через площадь: h = 2A / a
        shortest_height = 2 * area / longest if longest > 0 else 1.0
        aspect = longest / shortest_height if shortest_height > 0 else 1.0

        primitives = []

        # если треугольник не вытянутый, просто возвращаем один primitive
        if aspect <= self.aspect_threshold:
            primitives.append({
                "face_id": face.glo_id,
                "face": face,
                "box_min": vmin,
                "box_max": vmax,
                "centroid": centroid,
                "area": area,
                "zone_name": zone_name,
                "adjacency_info": adjacency_info
            })
        else:
            # Early Split Clipping
            # определяем количество делений вдоль длинной оси
            k = min(math.ceil(aspect), 4)
            axis = np.argmax(vmax - vmin)  # ось наибольшего размера bbox

            step = (vmax[axis] - vmin[axis]) / k
            for i in range(k):
                sub_min = vmin.copy()
                sub_max = vmax.copy()
                sub_min[axis] = vmin[axis] + i * step
                sub_max[axis] = vmin[axis] + (i + 1) * step

                # создаём под-примитив
                primitives.append({
                    "face_id": face.glo_id,
                    "face": face,
                    "box_min": sub_min,
                    "box_max": sub_max,
                    "centroid": centroid,  # общий для face
                    "area": area / k,      # делим площадь поровну
                    "zone_name": zone_name,
                    "adjacency_info": adjacency_info
                })

            logging.debug(
                "Early Split: face_id=%s, aspect=%.2f, k=%d, axis=%d",
                face.glo_id, aspect, k, axis
            )

        # лог без adjacency_info
        for prim in primitives:
            log_data = {k: v for k, v in prim.items() if k not in ["adjacency_info", "face"]}
            logging.debug("Processing face_id %s, parameters: %s", face.glo_id, log_data)

        return primitives


    # вспомогательные функции для AABB и Surface Area Heuristic (SAH)
    @staticmethod
    def surface_area(box_min, box_max):
        """Площадь поверхности AABB"""
        d = np.maximum(box_max - box_min, 0.0)
        return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])

    @staticmethod
    def box_union(a_min, a_max, b_min, b_max):
        """Объединение двух AABB"""
        return np.minimum(a_min, b_min), np.maximum(a_max, b_max)

    @staticmethod
    def boxes_intersect(a_min, a_max, b_min, b_max):
        """Проверка пересечения двух AABB"""
        return np.all(a_max >= b_min) and np.all(b_max >= a_min)


    def collect_primitives(self):
        """
        Собирает все AABB-примитивы из faces.
        """
        logging.info("Total mesh faces: %s", len(self.mesh.faces))
        logging.info("Collecting primitives from mesh...")
        for face in self.mesh.faces:
            prims = self.preprocessing(face)
            self.primitives.extend(prims)
        logging.info("Total primitives collected: %d", len(self.primitives))
    
    
    def build_tree(self):
        """
        Рекурсивное построение BVH с SAH-разбиением.
        """
        self.collect_primitives()
        indices = np.arange(len(self.primitives))
        sorted_indices = [
            sorted(indices, key=lambda i: self.primitives[i]["centroid"][axis])
            for axis in range(3)
        ]
        
        logging.info("Building BVH tree...")
        # рекурсивное построение дерева
        root_index = self.build_node(sorted_indices, depth=0)
        logging.info("BVH construction completed. Root index: %d", root_index)
    
    
    def build_node(self, sorted_indices_by_axis, depth):
        """
        Рекурсивное построение одного узла BVH.
        :param sorted_indices_by_axis: список из трёх списков индексов, отсортированных по X/Y/Z
        """
        prim_indices = sorted_indices_by_axis[0]
        n = len(prim_indices)
        
        # вычисляем AABB текущего узла
        vmin = np.min([self.primitives[i]["box_min"] for i in prim_indices], axis=0)
        vmax = np.max([self.primitives[i]["box_max"] for i in prim_indices], axis=0)
        
        # условие выхода
        if n <= self.max_leaf_size or depth >= self.max_depth:
            node = {
                "box_min": vmin,
                "box_max": vmax,
                "is_leaf": True,
                "prim_indices": prim_indices,
                "depth": depth,
            }
            self.nodes.append(node)
            return len(self.nodes) - 1

        # SAH-разбиение
        best_cost = math.inf
        best_axis = -1
        best_split = -1
        
        for axis in range(3):
            plist = sorted_indices_by_axis[axis]
            left_bounds = []
            right_bounds = [None] * n

            # sweep справа налево (right bounds)
            cur_vmin = np.full(3, np.inf)
            cur_vmax = np.full(3, -np.inf)
            for i in reversed(range(n)):
                prim = self.primitives[plist[i]]
                cur_vmin = np.minimum(cur_vmin, prim["box_min"])
                cur_vmax = np.maximum(cur_vmax, prim["box_max"])
                right_bounds[i] = (cur_vmin.copy(), cur_vmax.copy())

            # sweep слева направо (left bounds)
            cur_vmin = np.full(3, np.inf)
            cur_vmax = np.full(3, -np.inf)
            for i in range(1, n):
                prim = self.primitives[plist[i - 1]]
                cur_vmin = np.minimum(cur_vmin, prim["box_min"])
                cur_vmax = np.maximum(cur_vmax, prim["box_max"])
                left_sa = self.surface_area(cur_vmin, cur_vmax)
                right_sa = self.surface_area(*right_bounds[i])
                cost = left_sa * i + right_sa * (n - i)
                if cost < best_cost:
                    best_cost, best_axis, best_split = cost, axis, i
                    
        # если разбиение невыгодно, создаем лист
        if best_axis == -1:
            node = {
                "box_min": vmin,
                "box_max": vmax,
                "is_leaf": True,
                "prim_indices": prim_indices,
                "depth": depth,
            }
            self.nodes.append(node)
            return len(self.nodes) - 1

        # делим примитивы на левую и правую часть
        plist = sorted_indices_by_axis[best_axis]
        left_ids = plist[:best_split]
        right_ids = plist[best_split:]
        
        # перестраиваем сортированные списки для потомков
        left_sorted = []
        right_sorted = []
        for axis in range(3):
            plist_axis = sorted_indices_by_axis[axis]
            left_sorted.append([i for i in plist_axis if i in left_ids])
            right_sorted.append([i for i in plist_axis if i in right_ids])
            
        # создаем узел и рекурсивно строим потомков
        node_index = len(self.nodes)
        node = {
            "box_min": vmin,
            "box_max": vmax,
            "is_leaf": False,
            "depth": depth,
            "left": None,
            "right": None,
        }
        self.nodes.append(node)

        node["left"] = self.build_node(left_sorted, depth + 1)
        node["right"] = self.build_node(right_sorted, depth + 1)

        self.nodes[node_index] = node
        return node_index
        

    def test_node_pair(self, idx_a, idx_b, skip_adjacent = True, zone_filter = True):
        """
        Рекурсивное сравнение двух узлов дерева
        """
        node_a = self.nodes[idx_a]
        node_b = self.nodes[idx_b]
        
        # проверка пересечения AABB
        if not self.boxes_intersect(node_a["box_min"], node_a["box_max"],
                                    node_b["box_min"], node_b["box_max"]):
            return
        # лист-лист
        if node_a["is_leaf"] and node_b["is_leaf"]:
            for i in node_a["prim_indices"]:
                for j in node_b["prim_indices"]:
                    fa = self.primitives[i]
                    fb = self.primitives[j]

                    # исключаем дубликаты и самосравнение
                    if fa["face_id"] >= fb["face_id"]:
                        continue

                    # фильтры adjacency и zone
                    if skip_adjacent and set(fa["adjacency_info"]) & set(fb["adjacency_info"]):
                        continue
                    if zone_filter == "same" and fa["zone_name"] != fb["zone_name"]:
                        continue
                    if zone_filter == "different" and fa["zone_name"] == fb["zone_name"]:
                        continue

                    self.candidates.append((fa["face"], fb["face"]))
            return

        # выбираем узел для рекурсии
        if not node_a["is_leaf"] and (node_b["is_leaf"] or node_a["depth"] <= node_b["depth"]):
            self.test_node_pair(node_a["left"], idx_b, skip_adjacent, zone_filter)
            self.test_node_pair(node_a["right"], idx_b, skip_adjacent, zone_filter)
        else:
            self.test_node_pair(idx_a, node_b["left"], skip_adjacent, zone_filter)
            self.test_node_pair(idx_a, node_b["right"], skip_adjacent, zone_filter)
        
    
    def find_candidates(self, skip_adjacent=True, zone_filter=None):
        """
        Находит все пары пересекающихся AABB (candidate пары).
        :param skip_adjacent: пропускать ли смежные грани
        :param zone_filter: 'same' | 'different' | None
        """
        logging.info("Starting pairwise traversal for candidate detection...")
        self.candidates = []
        if not self.nodes:
            return []

        self.test_node_pair(0, 0, skip_adjacent, zone_filter)
        logging.info("Candidate pairs found: %d", len(self.candidates))
        return self.candidates
    

if __name__ == '__main__':
    mesh = Mesh("tests/examples/small_sphere_double.dat")
    bvh_tree = BVH(mesh)
    bvh_tree.build_tree()
    candidates = bvh_tree.find_candidates(zone_filter="different")
    print(type(candidates[0][0]))
