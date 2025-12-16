import logging

import numpy as np
from typing import List, Tuple


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
