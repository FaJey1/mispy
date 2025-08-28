import logging

import numpy as np
from numpy import linalg as LA
from typing import List, Tuple, Optional
from collections import defaultdict
from itertools import combinations

from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node


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
        logging.info("Total face count: %s", len(self.mesh.faces))
        self.border_candidates = []
        self.border_candidates_help = []
        self.border_planes = []
        self.border_intersection_planes_line = []
        self.border_faces = []
        self.border_multiline = []
    
    def tnoi_filtering(self) :
        """
        Filtering intersection candidates
        
        Returns
        -------
        candidates
            List of border candidates.
        """
        # Алгоритм background grid
        bbox_min = np.min([node.p for node in self.mesh.nodes], axis=0)
        bbox_max = np.max([node.p for node in self.mesh.nodes], axis=0)
        bbox_size = bbox_max - bbox_min
        logging.info("tnoi_filtering >> bbox_size: %s", bbox_size)
        
        # Размер ячейки фоновой сетки (cell_size) = среднему размеру face
        cell_size = np.cbrt(np.mean([np.linalg.norm(f.nodes[0].p - f.nodes[1].p) for f in self.mesh.faces]))
        logging.info("tnoi_filtering >> cell_size: %s", cell_size)
        
        # Привязка треугольников к ячейкам
        # Для каждого Face считаем его локальный bounding box
        # Находим, какие ячейки сетки он пересекает
        # Записываем face в список кандидатов ячейки
        grid = defaultdict(list)

        for face in self.mesh.faces:
            f_min = np.min([n.p for n in face.nodes], axis=0)
            f_max = np.max([n.p for n in face.nodes], axis=0)
            cell_min = np.floor((f_min - bbox_min) / cell_size).astype(int)
            cell_max = np.floor((f_max - bbox_min) / cell_size).astype(int)

            for ix in range(cell_min[0], cell_max[0]+1):
                for iy in range(cell_min[1], cell_max[1]+1):
                    for iz in range(cell_min[2], cell_max[2]+1):
                        grid[(ix, iy, iz)].append(face)
        logging.info("The triangles are bound to the cells, count: %s", len(grid))
        
        # Для каждой ячейки берём все грани (faces)
        # Если там несколько разных сеток -> формируем все пары (face_i, face_j)
        # Если в ячейке три и более сетки (A, B, D), то берём комбинации: (A, B), (A, D), (B, D)
        # Если в ячейке 1 зона, но несколько граней -> проверяем самопересечение
        candidates_set = set()
        for cell_faces in grid.values():
            # Группируем грани по zone
            by_zone = defaultdict(list)
            for f in cell_faces:
                by_zone[f.zone.name].append(f)

            zones = list(by_zone.keys())

            if len(zones) == 1:
                # Одна зона, но несколько граней -> проверяем самопересечения
                # faces = by_zone[zones[0]]
                # if len(faces) > 1:
                #     for f1, f2 in combinations(faces, 2):
                #         key = tuple(sorted((id(f1), id(f2))))
                #         candidates_set.add(key)
                pass 
            else:
                # Несколько зон -> формируем пары между зонами
                for i, zone1 in enumerate(zones):
                    for zone2 in zones[i+1:]:
                        for f1 in by_zone[zone1]:
                            for f2 in by_zone[zone2]:
                                key = tuple(sorted((id(f1), id(f2))))
                                candidates_set.add(key)

        # Восстанавливаем объекты граней из id
        all_faces = [f for faces in grid.values() for f in faces]
        id_to_face = {id(f): f for f in all_faces}
        self.border_candidates = [(id_to_face[i1], id_to_face[i2]) for i1, i2 in candidates_set]

        logging.info("INFO: Candidates for checking the intersection have been formed, count: %s", len(self.border_candidates))
        
        sub = {f.glo_id for pair in self.border_candidates for f in pair}
        return [f for f in self.mesh.faces if f.glo_id in sub]


    def defining_plane(self, faces_pair):
        """
        
        """
        plane1 = Plane(faces_pair[0])
        plane2 = Plane(faces_pair[1])
        
        intersection = self.defining_intersection_planes_line(plane1, plane2)
        plane1.intersection = intersection
        plane2.intersection = intersection
        
        if intersection:
            self.border_candidates_help.append(faces_pair)
            self.border_planes.append([plane1, plane2])


    def defining_intersection_planes_line(self, plane1, plane2):
        """
        
        """
        line = Line(plane1, plane2)
        intersection = line.build_intersection_line()
        if intersection:
            self.border_intersection_planes_line.append(line)
            return True
        return False


    def defining_intersections(self):
        """
        
        """
        # Шаг 1. Построение плоскостей и линий пересечения по 3-м точкам для пар кандидатов
        for pair in  self.border_candidates:
            self.defining_plane(pair)
        
        self.border_candidates = self.border_candidates_help
        logging.info("INFO: Candidates planes for checking the intersection have been formed, count: %s", len(self.border_planes))
        logging.info("INFO: Candidates intersection planes line for checking the intersection have been formed, count: %s", len(self.border_intersection_planes_line))
        logging.info("INFO: Candidates for checking the intersection have been formed after intersection checking, count: %s", len(self.border_candidates))
        
        # Шаг 2. Вычисление отрезка пересечения для каждого из треугольников
        # Определяем и классифицируем отрезок линии пересечения плоскости и треугольника
        # Идем по каждому ребру (уравнение прямой + поиск пересечения) и классифицируем точки пресечения
        # 0 - точка пересечения лежит за пределами ребра
        # 1 - точка пересечениялежитввершине
        # 2 - точка пересечениялежитвнутрикромки
        # Сохраняем полученный результат в струкутре <пара>-<грань>-<ребро>-<прямая с ребром>-<точка>
            

# ==================================================================================================

if __name__ == '__main__':
    pass

# ==================================================================================================
