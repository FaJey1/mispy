import logging

import numpy as np
from typing import List, Tuple, Union

from mispy.extract_mesh import Node, Face

# Настройка логирования для этого модуля
# Используем getLogger вместо basicConfig, чтобы не конфликтовать с другими модулями
logger = logging.getLogger(__name__)


def _edge_plane_intersection(a, b, n, d, eps=1e-22):
    """
    Вычисляет пересечение ребра AB с плоскостью n·x + d = 0.
    
    Использует прямой расчёт пересечения ребра с плоскостью для избежания
    числовых неточностей, которые возникают при поиске линии пересечения
    двух плоскостей и последующем поиске точки на этой линии.
    
    Parameters
    ----------
    a : np.ndarray
        Начальная точка ребра (3D координаты).
    b : np.ndarray
        Конечная точка ребра (3D координаты).
    n : np.ndarray
        Нормаль плоскости (3D вектор).
    d : float
        Константа плоскости (d = -n·p0, где p0 - точка на плоскости).
    eps : float, optional
        Порог для проверки параллельности и границ отрезка (по умолчанию 1e-9).
        
    Returns
    -------
    tuple | None
        (t, point) если пересечение найдено:
        - t: параметр вдоль ребра (0 <= t <= 1)
        - point: точка пересечения (3D координаты)
        None если пересечения нет или ребро параллельно плоскости.
        
    Notes
    -----
    Алгоритм:
    1. Вычисляем направляющий вектор ребра: ab = b - a
    2. Проверяем параллельность: если |n·ab| < eps, ребро параллельно плоскости
    3. Вычисляем параметр t: t = -(n·a + d) / (n·ab)
    4. Проверяем, что t находится в пределах [0, 1] (с учётом eps)
    5. Вычисляем точку пересечения: point = a + t * ab
    """
    ab = b - a
    denom = np.dot(n, ab)
    
    # Проверка на параллельность ребра и плоскости
    if abs(denom) < eps:
        return None  # Ребро параллельно плоскости
    
    # Вычисление параметра t вдоль ребра
    t = -(np.dot(n, a) + d) / denom
    
    # Проверка, что пересечение находится в пределах отрезка [0, 1]
    if t < -eps or t > 1 + eps:
        return None  # Пересечение вне ребра
    
    # Вычисление точки пересечения
    point = a + t * ab
    return t, point


def _classify_point(t, eps=1e-22):
    """
    Классифицирует параметр пересечения t вдоль ребра.
    
    Согласно теории Czech алгоритма, классификатор определяет положение
    точки пересечения относительно ребра:
    - 0: точка лежит вне ребра (t < 0 или t > 1)
    - 1: точка лежит в вершине ребра (t ≈ 0 или t ≈ 1)
    - 2: точка лежит внутри ребра (0 < t < 1)
    
    Parameters
    ----------
    t : float
        Параметр вдоль ребра (0 соответствует началу, 1 - концу).
    eps : float, optional
        Порог для определения близости к вершине (по умолчанию 1e-6).
        
    Returns
    -------
    int
        Классификатор: 0 (вне), 1 (вершина), или 2 (внутри).
    """
    if t < -eps or t > 1 + eps:
        return 0  # Точка вне ребра
    elif abs(t) <= eps or abs(1 - t) <= eps:
        return 1  # Точка в вершине (начало или конец ребра)
    else:
        return 2  # Точка внутри ребра


def _plane_parallel(n1, n2, eps=1e-22):
    """
    Проверяет, являются ли две плоскости параллельными.
    
    Плоскости параллельны, если их нормали коллинеарны (векторное произведение
    нормалей близко к нулю).
    
    Parameters
    ----------
    n1 : np.ndarray
        Нормаль первой плоскости (3D вектор).
    n2 : np.ndarray
        Нормаль второй плоскости (3D вектор).
    eps : float, optional
        Порог для проверки коллинеарности (по умолчанию 1e-12).
        
    Returns
    -------
    bool
        True если плоскости параллельны, False иначе.
    """
    # |n1 × n2| == 0 → нормали коллинеарны → плоскости параллельны
    return np.linalg.norm(np.cross(n1, n2)) < eps


def _point_in_triangle(point: np.ndarray, face: Face, eps: float = 1e-22) -> bool:
    """
    Проверяет, принадлежит ли точка треугольнику (в 3D пространстве).
    
    Использует барицентрические координаты для проверки принадлежности точки
    треугольнику. Точка принадлежит треугольнику, если все барицентрические
    координаты неотрицательны и их сумма равна 1.
    
    Parameters
    ----------
    point : np.ndarray
        Точка для проверки (3D координаты).
    face : Face
        Треугольная грань.
    eps : float, optional
        Порог для проверки принадлежности (по умолчанию 1e-9).
        
    Returns
    -------
    bool
        True если точка принадлежит треугольнику, False иначе.
        
    Notes
    -----
    Алгоритм использует барицентрические координаты:
    1. Вычисляем векторы от первой вершины к двум другим и к точке
    2. Находим барицентрические координаты u, v, w
    3. Точка внутри треугольника, если u >= 0, v >= 0, w >= 0 и u + v + w ≈ 1
    """
    # Получаем вершины треугольника
    v0, v1, v2 = face.nodes[0].p, face.nodes[1].p, face.nodes[2].p
    
    # Векторы от первой вершины
    edge1 = v1 - v0
    edge2 = v2 - v0
    point_vec = point - v0
    
    # Вычисляем барицентрические координаты
    # Используем метод решения системы уравнений через векторное произведение
    # P = v0 + u*(v1-v0) + v*(v2-v0), где u, v >= 0 и u + v <= 1
    dot00 = np.dot(edge1, edge1)
    dot01 = np.dot(edge1, edge2)
    dot02 = np.dot(edge1, point_vec)
    dot11 = np.dot(edge2, edge2)
    dot12 = np.dot(edge2, point_vec)
    
    # Решение системы для u и v
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        # Треугольник вырожденный (площадь близка к нулю)
        return False
    
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    w = 1.0 - u - v
    
    # Точка внутри треугольника, если все барицентрические координаты >= -eps
    # (используем -eps для учёта числовых ошибок)
    return u >= -eps and v >= -eps and w >= -eps

class CzechClassify:
    def __init__(self, candidates: Tuple = None, checked_pairs: set = None, pair_index: int = 0):
        """
        Инициализирует классификатор пересечения двух треугольников.
        
        Создаёт экземпляр CzechClassify для классификации и проверки пересечения
        двух треугольных граней согласно теории Czech алгxоритма. Классификатор
        использует edge-plane intersection для определения пересечения рёбер граней
        с плоскостями противоположных граней, избегая числовых неточностей.
        
        Parameters
        ----------
        candidates : Tuple[Face, Face], optional
            Кортеж из двух граней (face_a, face_b) для проверки пересечения.
            По умолчанию None. Обе грани должны быть объектами типа Face.
        checked_pairs : set, optional
            Множество уже проверенных пар граней, используемое для избежания
            дублирования проверок при проходе через соседние грани (neighbor tracing).
            Хранит кортежи из идентификаторов граней (id(face_a), id(face_b)).
            По умолчанию None (создаётся пустое множество).
        pair_index : int, optional
            Индекс пары граней для отладки и логирования. Используется в логах
            для идентификации конкретной пары при анализе результатов классификации.
            По умолчанию 0.
        
        Attributes
        ----------
        face_a : Face
            Первая грань для проверки пересечения.
        face_b : Face
            Вторая грань для проверки пересечения.
        pair_index : int
            Индекс пары граней для отладки и логирования.
        impossible_couples : List[Tuple[Face, Face]]
            Список пар граней, которые попали в невозможные случаи классификации
            (например, коды "001", "002", "012", "122", "222"). Используется для
            последующей обработки через neighbor tracing.
        points : List[np.ndarray]
            Список найденных точек пересечения двух граней. Может содержать:
            - 0 точек: пересечения нет
            - 1 точку: грани касаются в одной точке (касание вне вершин)
            - 2 точки: грани пересекаются по отрезку
            Если одна из точек лежит в вершине, необходимо искать вторую точку
            через классификацию. Если второй точки нет, пересечения нет.
        checked_pairs : set
            Множество уже проверенных пар граней для избежания дублирования
            при neighbor tracing. Хранит кортежи (id(face_a), id(face_b)).
        impossible_cases : set[str]
            Множество невозможных случаев классификации согласно теории:
            {"001", "002", "012", "122", "222"}. Эти случаи не могут возникнуть
            при правильном вычислении пересечения и указывают на числовую неточность.
            Требуют обработки через neighbor tracing.
        special_cases : set[str]
            Множество особых случаев классификации: {"000"} - нет пересечения
            (все точки пересечения лежат вне рёбер).
        
        Notes
        -----
        - Классификатор использует алгоритм Czech для определения пересечения
          треугольников, основанный на edge-plane intersection.
        - Для избежания числовых неточностей используется прямой расчёт пересечения
          рёбер с плоскостями, а не поиск линии пересечения двух плоскостей.
        - При обнаружении невозможных случаев используется neighbor tracing для
          корректировки результатов через проверку соседних граней.
        """
        # пары, которые могут пересекаться
        self.face_a, self.face_b = candidates
        
        # Номер пары для дебага
        self.pair_index = pair_index
        
        # Невозможные классификации
        self.impossible_couples = []
        
        # Найденные точки/точка пересечения двух ячеек, 1 точка если грани "касаются",
        # То есть точка пересечения лежит ВНЕ вершины, 
        # если одна из точек лежит в вершине, то ОБЯЗАТЕЛЬНО ищем вторую, иначе пропускаем
        self.points = []
        
        # При проходе через соседа проверить, что такую пару смотрели уже checked_pairs 
        self.checked_pairs = checked_pairs if checked_pairs is not None else set()
        
        # Невозможные случаи классификации
        self.impossible_cases = {"001", "002", "012", "122", "222"}
        # Особые случаи классификации
        self.special_cases = {"000"}
        
        self.eps = 1e-11
    
    # ----------------------------------------------------------------------------------
    
    def classify(self):
        """
        Классифицирует пересечение двух треугольников согласно теории Czech алгоритма.
        
        Метод определяет классификаторы пересечения для каждого ребра грани с плоскостью
        противоположной грани. Классификатор может быть:
        - 0: точка пересечения лежит вне ребра
        - 1: точка пересечения лежит в вершине ребра
        - 2: точка пересечения лежит внутри ребра
        
        Алгоритм основан на edge-plane intersection для избежания числовых неточностей,
        которые возникают при поиске линии пересечения двух плоскостей.
        
        Returns
        -------
        dict
            Словарь с классификаторами для обеих граней:
            {
                "face_a": [(edge.glo_id, classifier), ...],  # 3 элемента
                "face_b": [(edge.glo_id, classifier), ...],  # 3 элемента
                "coplanar": bool  # True если плоскости совпадают
            }
            
        Notes
        -----
        Метод обрабатывает три основных случая:
        1. Плоскости совпадают (coplanar=True): грани лежат в одной плоскости,
           требуется специальная обработка копланарного пересечения.
        2. Плоскости параллельны или почти параллельны: нормали коллинеарны,
           но плоскости не совпадают - пересечения нет.
        3. Плоскости пересекаются: выполняется классификация пересечений рёбер
           с плоскостями для обеих граней.
           
        Если для какой-либо грани найдено меньше 3 классификаторов (например,
        из-за параллельности рёбер плоскости), недостающие классификаторы
        дополняются значением 0, и выводится INFO сообщение.
        """
        f1, f2 = self.face_a, self.face_b
        
        # Расчёт нормалей граней (если ещё не вычислены)
        if f1.normal is None:
            f1.calculate_normal()
        if f2.normal is None:
            f2.calculate_normal()
        
        # Уравнения плоскостей: n·x + d = 0
        # n - нормаль плоскости, d = -n·p0 (где p0 - точка на плоскости)
        p1, n1 = f1.nodes[0].p, f1.normal
        p2, n2 = f2.nodes[0].p, f2.normal
        d1 = -np.dot(n1, p1)  # Константа для плоскости f1
        d2 = -np.dot(n2, p2)  # Константа для плоскости f2
        
        # --- 1. Проверка на совпадение плоскостей ---
        # Плоскости совпадают, если нормали коллинеарны И константы d равны
        # (или противоположны, если нормали направлены в разные стороны)
        is_parallel = _plane_parallel(n1, n2)
        is_coplanar = False
        
        if is_parallel:
            # Проверяем совпадение плоскостей: d1 ≈ d2 или d1 ≈ -d2
            # (второй случай - нормали противоположны, но плоскости совпадают)
            if abs(d1 - d2) < self.eps or abs(d1 + d2) < self.eps:
                is_coplanar = True
                logger.debug(
                    "CzechClassify: Pair %d, faces %d and %d are coplanar",
                    self.pair_index,
                    f1.glo_id,
                    f2.glo_id,
                )
                # Для копланарных граней возвращаем специальный результат
                # Классификаторы будут обработаны отдельно в методе обработки копланарности
                return {
                    "face_a": [(e.glo_id, 0) for e in f1.edges],
                    "face_b": [(e.glo_id, 0) for e in f2.edges],
                    "coplanar": True,
                }
            else:
                # Плоскости параллельны, но не совпадают - пересечения нет
                logger.debug(
                    "CzechClassify: Pair %d, faces %d and %d are parallel but not coplanar",
                    self.pair_index,
                    f1.glo_id,
                    f2.glo_id,
                )
                return {
                    "face_a": [(e.glo_id, 0) for e in f1.edges],
                    "face_b": [(e.glo_id, 0) for e in f2.edges],
                    "coplanar": False,
                }
        
        # --- 2. Плоскости почти параллельны (малый угол между нормалями) ---
        # Проверяем угол между нормалями для предупреждения о возможных неточностях
        cross_norm = np.linalg.norm(np.cross(n1, n2))
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm > 0 and n2_norm > 0:
            sin_angle = cross_norm / (n1_norm * n2_norm)
            if sin_angle < 1e-6:  # Угол меньше ~0.0001 градуса
                logger.debug(
                    "CzechClassify: Pair %d, faces %d and %d are nearly parallel (sin(angle)=%.2e)",
                    self.pair_index,
                    f1.glo_id,
                    f2.glo_id,
                    sin_angle,
                )
        
        # --- 3. Плоскости пересекаются - классифицируем пересечения рёбер ---
        # Для каждой грани находим пересечения её рёбер с плоскостью противоположной грани
        
        # Классификация рёбер f1 с плоскостью f2
        # Сохраняем не только классификаторы, но и точки пересечения для дальнейшего использования
        classification_a = []
        intersection_points_a = []  # Точки пересечения рёбер f1 с плоскостью f2
        for i, edge in enumerate(f1.edges):
            # Получаем координаты узлов ребра
            a = edge.nodes[0].p
            b = edge.nodes[1].p
            
            # Ищем пересечение ребра с плоскостью f2
            result = _edge_plane_intersection(a, b, n2, d2)
            
            if result is not None:
                t, point = result
                classifier = _classify_point(t)
                classification_a.append((edge.glo_id, classifier, point))  # Добавляем точку пересечения
                intersection_points_a.append(point)
            else:
                # Ребро не пересекает плоскость или параллельно ей
                classification_a.append((edge.glo_id, 0, None))
        
        # Классификация рёбер f2 с плоскостью f1
        # Сохраняем не только классификаторы, но и точки пересечения для дальнейшего использования
        classification_b = []
        intersection_points_b = []  # Точки пересечения рёбер f2 с плоскостью f1
        for i, edge in enumerate(f2.edges):
            # Получаем координаты узлов ребра
            a = edge.nodes[0].p
            b = edge.nodes[1].p
            
            # Ищем пересечение ребра с плоскостью f1
            result = _edge_plane_intersection(a, b, n1, d1)
            
            if result is not None:
                t, point = result
                classifier = _classify_point(t)
                classification_b.append((edge.glo_id, classifier, point))  # Добавляем точку пересечения
                intersection_points_b.append(point)
            else:
                # Ребро не пересекает плоскость или параллельно ей
                classification_b.append((edge.glo_id, 0, None))
        
        # Проверка: если классификаторов меньше 3, дополняем нулями
        if len(classification_a) < 3:
            logger.debug(
                "CzechClassify: Pair %d, face_a (%d) has only %d classifiers, padding with zeros",
                self.pair_index,
                f1.glo_id,
                len(classification_a),
            )
            # Дополняем до 3 элементов (используем edge.glo_id = -1 для отсутствующих)
            while len(classification_a) < 3:
                classification_a.append((-1, 0, None))
        
        if len(classification_b) < 3:
            logger.debug(
                "CzechClassify: Pair %d, face_b (%d) has only %d classifiers, padding with zeros",
                self.pair_index,
                f2.glo_id,
                len(classification_b),
            )
            # Дополняем до 3 элементов (используем edge.glo_id = -1 для отсутствующих)
            while len(classification_b) < 3:
                classification_b.append((-1, 0, None))
        
        return {
            "face_a": classification_a,  # [(edge.glo_id, classifier, point), ...]
            "face_b": classification_b,  # [(edge.glo_id, classifier, point), ...]
            "intersection_points_a": intersection_points_a,  # Точки пересечения рёбер f1 с плоскостью f2
            "intersection_points_b": intersection_points_b,  # Точки пересечения рёбер f2 с плоскостью f1
            "coplanar": False,
        }

    # ----------------------------------------------------------------------------------
    
    def find_intersection_segment(self, result: dict):
        """
        Находит сегмент пересечения двух треугольников на основе результата классификации.
        
        Метод обрабатывает классификаторы пересечения рёбер граней с плоскостями,
        определяет тип пересечения согласно теории Czech алгоритма и формирует
        сегмент пересечения (отрезок или точку).
        
        Алгоритм:
        1. Извлекает классификаторы из result для обеих граней
        2. Формирует строку классификации путём сортировки классификаторов по возрастанию
        3. Проверяет попадание в impossible_cases (требуют neighbor tracing)
        4. Проверяет попадание в special_cases (например, "000" - нет пересечения)
        5. Обрабатывает валидные случаи для формирования сегмента пересечения
        
        Parameters
        ----------
        result : dict
            Результат классификации от метода classify():
            {
                "face_a": [(edge.glo_id, classifier), ...],  # 3 элемента
                "face_b": [(edge.glo_id, classifier), ...],  # 3 элемента
                "coplanar": bool  # True если плоскости совпадают
            }
            где classifier: 0 (вне ребра), 1 (в вершине), 2 (внутри ребра)
        
        Returns
        -------
        Tuple[bool, List[Node]]
            Кортеж (has_intersection, intersection_segment), где:
            - has_intersection: True если есть пересечение, False если нет
            - intersection_segment: список точек пересечения в виде объектов Node
              (пустой список [] если нет пересечения, [Node1, Node2] для отрезка, [Node, Node] для точки касания)
              Если точка не совпадает с вершиной грани, создаётся новый объект Node с координатами точки
        
        Notes
        -----
        Классификаторы сортируются по возрастанию для формирования строки классификации,
        например, "022", "011", "112". Порядок рёбер не важен для классификации согласно теории.
        
        Impossible cases ({"001", "002", "012", "122", "222"}) указывают на числовую неточность
        и требуют обработки через neighbor tracing для корректировки результатов.
        """
        # Проверка на копланарность - требует специальной обработки
        if result.get("coplanar", False):
            logger.debug(
                "CzechClassify: Pair %d, coplanar case detected, requires special handling",
                self.pair_index,
            )
            # TODO: Реализовать обработку копланарного случая
            return False, []
        
        # Извлекаем классификаторы из result для обеих граней
        # Классификаторы представлены как список кортежей [(edge.glo_id, classifier, point), ...]
        face_a_classifiers = result.get("face_a", [])
        face_b_classifiers = result.get("face_b", [])
        
        # Извлекаем только классификаторы (значения 0, 1, 2), игнорируя edge.glo_id и точки
        # и формируем строку классификации путём сортировки по возрастанию
        classifiers_a = [item[1] for item in face_a_classifiers]  # item = (edge.glo_id, classifier, point)
        classifiers_b = [item[1] for item in face_b_classifiers]  # item = (edge.glo_id, classifier, point)
        
        # Сортируем классификаторы по возрастанию для формирования строки классификации
        # Согласно теории, порядок точек не важен для классификации
        classifiers_a_sorted = sorted(classifiers_a)
        classifiers_b_sorted = sorted(classifiers_b)
        
        # Формируем строку классификации из отсортированных классификаторов
        classification_code_a = "".join(map(str, classifiers_a_sorted))
        classification_code_b = "".join(map(str, classifiers_b_sorted))
        
        # Согласно теории: треугольники пересекаются только если ОБЕ грани пересекают
        # линию пересечения плоскостей. Если одна грань имеет классификацию '000'
        # (нет пересечения), то и треугольники не пересекаются.
        # Проверяем обе классификации!
        if classification_code_a == "000" or classification_code_b == "000":
            # Если хотя бы одна грань не пересекает линию пересечения, треугольники не пересекаются
            logger.debug(
                "CzechClassify: Pair %d, no intersection - face_a: '%s', face_b: '%s' (one or both have '000')",
                self.pair_index,
                classification_code_a,
                classification_code_b,
            )
            return False, []
        
        # Используем классификацию из face_a для дальнейшей обработки
        # (обе классификации уже проверили на '000' выше)
        classification_code = classification_code_a
        
        logger.debug(
            "CzechClassify: Pair %d, classification codes - face_a: '%s', face_b: '%s', using: '%s'",
            self.pair_index,
            classification_code_a,
            classification_code_b,
            classification_code,
        )
        
        # Проверка на невозможные случаи (impossible cases)
        # Эти случаи указывают на числовую неточность и требуют neighbor tracing
        if classification_code in self.impossible_cases:
            logger.debug(
                "CzechClassify: Pair %d, impossible case '%s' detected, requires neighbor tracing (faces %d and %d)",
                self.pair_index,
                classification_code,
                self.face_a.glo_id,
                self.face_b.glo_id,
            )
            # Согласно теории: при невозможном случае для исходной пары граней
            # пересечение не может быть корректно определено из-за числовой неточности.
            # Neighbor tracing используется для поиска правильного продолжения границы
            # пересечения через соседние треугольники, но это пересечение относится
            # к другой паре граней, а не к исходной.
            # Поэтому для исходной пары возвращаем False и добавляем её в impossible_couples
            # для последующей обработки (например, через neighbor tracing на уровне выше).
            self.impossible_couples.append((self.face_a, self.face_b))
            return False, []
        
        # Проверка на особые случаи (special cases)
        # Например, "000" - все пересечения вне рёбер, пересечения нет
        if classification_code in self.special_cases:
            logger.debug(
                "CzechClassify: Pair %d, special case '%s' detected, no intersection (faces %d and %d)",
                self.pair_index,
                classification_code,
                self.face_a.glo_id,
                self.face_b.glo_id,
            )
            # Случай "000" - нет пересечения
            return False, []
        
        # Валидный случай классификации - есть пересечение
        # Формируем сегмент пересечения из найденных точек
        logger.debug(
            "CzechClassify: Pair %d, valid classification '%s', intersection detected (faces %d and %d)",
            self.pair_index,
            classification_code,
            self.face_a.glo_id,
            self.face_b.glo_id,
        )
        
        # Формируем список точек пересечения из найденных точек
        # Используем точки из intersection_points_a или intersection_points_b
        # (они лежат на линии пересечения плоскостей, но нужно отфильтровать дубликаты)
        all_intersection_points = []
        
        # Добавляем точки пересечения из рёбер f1 с плоскостью f2
        # Фильтруем точки, которые не являются None (None означает, что пересечения нет)
        for item in face_a_classifiers:
            if len(item) >= 3 and item[2] is not None:  # item = (edge.glo_id, classifier, point)
                all_intersection_points.append(item[2])
        
        # Добавляем точки пересечения из рёбер f2 с плоскостью f1
        for item in face_b_classifiers:
            if len(item) >= 3 and item[2] is not None:  # item = (edge.glo_id, classifier, point)
                all_intersection_points.append(item[2])
        
        # Удаляем дубликаты точек (проверка на близость с epsilon)
        # Две точки считаются одинаковыми, если расстояние между ними < eps
        unique_points = []
        for pt in all_intersection_points:
            is_duplicate = False
            for existing_pt in unique_points:
                if np.linalg.norm(pt - existing_pt) < self.eps:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(pt)
        
        # Согласно теории, сегмент пересечения - это две крайние точки из найденных
        # Но нужно проверить, что точки действительно принадлежат ОБОИМ треугольникам
        # (лежат внутри треугольников или на их границах)
        # Классификация показывает только пересечение рёбер с плоскостями,
        # но не гарантирует, что точки лежат внутри треугольников
        
        if len(unique_points) == 0:
            logger.debug(
                "CzechClassify: Pair %d, valid classification but no intersection points found",
                self.pair_index,
            )
            return False, []
        
        # Проверяем принадлежность найденных точек обоим треугольникам
        # Точка должна лежать внутри или на границе обоих треугольников для реального пересечения
        valid_intersection_points = []
        
        for pt in unique_points:
            # Проверяем принадлежность точки face_a (включая границы)
            in_face_a = _point_in_triangle(pt, self.face_a, self.eps)
            # Проверяем принадлежность точки face_b (включая границы)
            in_face_b = _point_in_triangle(pt, self.face_b, self.eps)
            
            if in_face_a and in_face_b:
                # Точка принадлежит обоим треугольникам - это валидная точка пересечения
                valid_intersection_points.append(pt)
            else:
                # Точка не принадлежит хотя бы одному треугольнику - пропускаем
                logger.debug(
                    "CzechClassify: Pair %d, point %s filtered out - in_face_a=%s, in_face_b=%s",
                    self.pair_index,
                    pt,
                    in_face_a,
                    in_face_b,
                )
        
        # Если после фильтрации не осталось валидных точек, пересечения нет
        if len(valid_intersection_points) == 0:
            logger.debug(
                "CzechClassify: Pair %d, classification '%s' valid but no points belong to both triangles - no intersection",
                self.pair_index,
                classification_code,
            )
            return False, []
        
        # Преобразуем точки np.ndarray в объекты Node
        # Если точка совпадает с вершиной грани - используем существующий Node
        # Иначе создаём новый Node с координатами точки
        intersection_segment_nodes = []
        all_face_nodes = list(self.face_a.nodes) + list(self.face_b.nodes)
        
        for pt in valid_intersection_points:
            # Пытаемся найти существующий Node с такими же координатами
            found_node = None
            for node in all_face_nodes:
                if np.linalg.norm(node.p - pt) < self.eps:
                    found_node = node
                    break
            
            if found_node is not None:
                # Используем существующий Node
                intersection_segment_nodes.append(found_node)
            else:
                # Создаём новый Node для точки пересечения
                new_node = Node(pt)
                new_node.glo_id = -1  # Временный ID, будет присвоен при необходимости
                intersection_segment_nodes.append(new_node)
        
        # Формируем сегмент пересечения
        if len(intersection_segment_nodes) == 1:
            # Касание в одной точке - возвращаем одну точку дважды для совместимости
            return True, [intersection_segment_nodes[0], intersection_segment_nodes[0]]
        else:
            # Находим две крайние точки (с максимальным расстоянием между ними)
            max_dist = -1
            i_max, j_max = 0, 1
            for i in range(len(intersection_segment_nodes)):
                for j in range(i + 1, len(intersection_segment_nodes)):
                    dist = np.linalg.norm(intersection_segment_nodes[i].p - intersection_segment_nodes[j].p)
                    if dist > max_dist:
                        max_dist = dist
                        i_max, j_max = i, j
            # Возвращаем две крайние точки в виде объектов Node
            return True, [intersection_segment_nodes[i_max], intersection_segment_nodes[j_max]]
    
    # ----------------------------------------------------------------------------------
    
    def get_intersection(self):
        """
        Определяет наличие и сегмент пересечения двух треугольников.
        
        Метод является главной точкой входа для проверки пересечения двух граней.
        Он выполняет классификацию пересечения через метод classify(), затем анализирует
        результат и формирует сегмент пересечения (отрезок или точку) с учётом наличия
        общих вершин между гранями.
        
        Алгоритм:
        1. Выполняет классификацию пересечения рёбер граней с плоскостями через classify()
        2. Определяет количество общих вершин между гранями
        3. Обрабатывает два основных случая:
           a) 1 общая вершина: первая точка сегмента - общая вершина (Node),
              вторая точка находится через классификацию (точка на ребре/в плоскости)
           b) 0 общих вершин: обе точки сегмента находятся через классификацию
        4. Формирует сегмент пересечения из найденных точек
        
        Returns
        -------
        Tuple[bool, List[Node], List[Tuple[Face, Face]]]
            Кортеж из трёх элементов:
            - has_intersection (bool): True если есть пересечение, False если нет
            - intersection_segment (List[Node]): список точек пересечения в виде объектов Node:
              * Пустой список [] - нет пересечения
              * [Node, Node] - сегмент из 2 точек (общая вершина + найденная точка, или обе найденные точки)
              * Для касания в одной точке: [Node, Node] с одинаковыми координатами
              * Если точка не совпадает с вершиной грани, создаётся новый Node с координатами точки
            - impossible_couples (List[Tuple[Face, Face]]): список пар граней, попавших
              в невозможные случаи классификации (требуют neighbor tracing)
        
        Raises
        ------
        ValueError
            Если грани имеют 2 или более общих вершин (недопустимая конфигурация)
            или если обнаружено пересечение, но сегмент содержит меньше 2 точек.
        
        Notes
        -----
        Метод обрабатывает три основных сценария:
        
        1. **1 общая вершина**:
           - Первая точка сегмента - общая вершина (объект Node из сетки)
           - Вторая точка находится через классификацию пересечений рёбер с плоскостями
           - Если вторая точка не найдена, пересечения нет
        
        2. **0 общих вершин**:
           - Обе точки сегмента находятся через классификацию
           - Метод find_intersection_segment() возвращает две крайние точки пересечения
           - Если точки не найдены, пересечения нет
        
        3. **2+ общих вершин**:
           - Вызывается ValueError, так как такая конфигурация недопустима
           - Такие пары должны были быть отсеяны на этапе BVH traversal
        
        Согласно теории Czech алгоритма, треугольники пересекаются только если ОБЕ грани
        пересекают линию пересечения плоскостей. Если одна грань имеет классификацию '000'
        (все пересечения вне рёбер), то треугольники не пересекаются.
        
        Сегмент пересечения формируется из точек, найденных при классификации пересечений
        рёбер граней с плоскостями противоположных граней. Дубликаты точек удаляются
        (проверка на близость с epsilon = 1e-9).
        
        Examples
        --------
        >>> cz = CzechClassify(candidates=(face_a, face_b))
        >>> has_intersection, segment, impossible = cz.get_intersection()
        >>> if has_intersection:
        ...     print(f"Intersection segment: {segment[0]} -> {segment[1]}")
        """
        # Наличие пересечения
        has_intersection = False
        # Пересечение в одной точке/отрезке
        intersection_segment = []

        # Возможно три случая
        # касание в 1 точке, при этом если есть общая вершина, то обязательно ищем вторую точку через классифкацию, 
        # если ее нет, то пересечения нет. Такая точка принадлежит обоим плоскостям 
        # реально пересечение или пересечения нет - чисто метод классификации
        
        # поиск общей вершины, при этом common_vertices либо 0 либо 1, иначе ошибка (должны были отсеить ранее)
        common_vertices = set(self.face_a.nodes) & set(self.face_b.nodes)
        
        # Проверка классификации
        result = self.classify()
        logger.debug("CzechClassify: Pair %d, classify: %s", self.pair_index, result)
        
        if len(common_vertices) == 1:
            # Случай с 1 общей вершиной: первая точка - общая вершина (Node),
            # вторая точка должна быть найдена через классификацию (точка на ребре/в плоскости треугольника)
            common_vertex = list(common_vertices)[0]  # Получаем общую вершину (Node объект)
            logger.debug(
                "CzechClassify: Pair %d, have 1 common vertex (node_id=%d), looking for second intersection point",
                self.pair_index,
                common_vertex.glo_id,
            )
            
            has_intersection, intersection_segment = self.find_intersection_segment(result)
            
            if not has_intersection:
                # Вторая точка не найдена - пересечения нет
                logger.debug(
                    "CzechClassify: Pair %d, common vertex found but no second point - no intersection",
                    self.pair_index,
                )
                return False, [], self.impossible_couples
            
            # Проверка: если все найденные точки пересечения совпадают с общей вершиной,
            # то пересечения нет (это касание в одной точке, а не пересечение)
            # Согласно теории: если одна точка лежит в вершине, нужно обязательно найти
            # вторую точку, которая НЕ является вершиной
            common_vertex_coords = common_vertex.p
            all_points_match_vertex = True
            
            # Проверяем все найденные точки пересечения (теперь это объекты Node)
            for node in intersection_segment:
                if np.linalg.norm(node.p - common_vertex_coords) > self.eps:
                    all_points_match_vertex = False
                    break
            
            if all_points_match_vertex:
                # Все найденные точки совпадают с общей вершиной - пересечения нет
                logger.debug(
                    "CzechClassify: Pair %d, all intersection points match common vertex (node_id=%d) - no intersection",
                    self.pair_index,
                    common_vertex.glo_id,
                )
                return False, [], self.impossible_couples
            
            # Формируем сегмент: первая точка - общая вершина (Node), вторая - точка из классификации
            # Берём первую найденную точку пересечения, которая отличается от общей вершины
            second_point_node = None
            for node in intersection_segment:
                if np.linalg.norm(node.p - common_vertex_coords) > self.eps:
                    second_point_node = node
                    break
            
            if second_point_node is None:
                # Не нашли точку, отличную от общей вершины - пересечения нет
                logger.debug(
                    "CzechClassify: Pair %d, no intersection point different from common vertex - no intersection",
                    self.pair_index,
                )
                return False, [], self.impossible_couples
            
            # intersection_segment теперь содержит объекты Node, берём нужную точку
            intersection_segment = [common_vertex, second_point_node]
            
            logger.info(
                "CzechClassify: Pair %d, intersection segment: common_vertex (node_id=%d) + second point (node_id=%s)",
                self.pair_index,
                common_vertex.glo_id,
                second_point_node.glo_id if hasattr(second_point_node, 'glo_id') else 'new',
            )
            return True, intersection_segment, self.impossible_couples
        elif len(common_vertices) >= 2:
            raise ValueError(
                f"CzechClassify: Pair {self.pair_index}, incorrect pair - faces {self.face_a.glo_id} and {self.face_b.glo_id} "
                f"have {common_vertices} common vertices (should be 0 or 1)"
            )
        
        # Если common_vertices == 0 просто ищем отрезок
        # find_intersection_segment() теперь возвращает объекты Node
        has_intersection, intersection_segment = self.find_intersection_segment(result)
        
        # Проверка на корректность сегмента пересечения
        if has_intersection and len(intersection_segment) < 2:
            raise ValueError(
                f"CzechClassify: Pair {self.pair_index}, intersection detected but segment has less than 2 points (count={len(intersection_segment)})"
            )
        
        return has_intersection, intersection_segment, self.impossible_couples
