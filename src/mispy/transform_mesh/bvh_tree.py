import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Literal
from collections import defaultdict

import numpy as np
import networkx as nx

from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
from .czech_classify import CzechClassify

# Настройка логирования для этого модуля
# Используем getLogger вместо basicConfig, чтобы не конфликтовать с другими модулями
logger = logging.getLogger(__name__)

def _aabb_empty() -> Tuple[np.ndarray, np.ndarray]:
    """
    Создаёт пустой AABB бокс.
    
    Пустой бокс имеет bb_min = [+inf, +inf, +inf] и bb_max = [-inf, -inf, -inf],
    что гарантирует, что любой реальный бокс будет его расширять при использовании
    _aabb_include. Это позволяет использовать пустой бокс как начальное значение
    при итеративном построении AABB из множества примитивов.
    
    Используется для инициализации AABB перед включением примитивов при построении
    BVH-дерева.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Кортеж (bb_min, bb_max), где:
        bb_min - массив минимальных координат [+inf, +inf, +inf],
        bb_max - массив максимальных координат [-inf, -inf, -inf].
    """
    bb_min = np.array([np.inf, np.inf, np.inf], dtype=float)
    bb_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    return bb_min, bb_max


def _aabb_include(bb_min: np.ndarray, bb_max: np.ndarray, other_min: np.ndarray, other_max: np.ndarray):
    """
    Расширяет AABB, включая другой бокс, и возвращает новый (min, max).
    
    Вычисляет минимальный ограничивающий бокс, который содержит оба входных AABB.
    Новый бокс формируется как пересечение границ исходных боксов по каждой оси:
    новый min = min(bb_min, other_min) по каждой оси,
    новый max = max(bb_max, other_max) по каждой оси.
    
    Используется при построении BVH для объединения AABB примитивов в узлы,
    а также при вычислении AABB дочерних узлов из AABB их примитивов.
    
    Parameters
    ----------
    bb_min : np.ndarray
        Минимальные координаты первого AABB по осям X, Y, Z.
    bb_max : np.ndarray
        Максимальные координаты первого AABB по осям X, Y, Z.
    other_min : np.ndarray
        Минимальные координаты второго AABB по осям X, Y, Z.
    other_max : np.ndarray
        Максимальные координаты второго AABB по осям X, Y, Z.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Кортеж (new_min, new_max), где:
        new_min - минимальные координаты объединённого AABB,
        new_max - максимальные координаты объединённого AABB.
    """
    return np.minimum(bb_min, other_min), np.maximum(bb_max, other_max)


def _aabb_surface_area(bb_min: np.ndarray, bb_max: np.ndarray) -> float:
    """
    Вычисляет площадь поверхности AABB.
    
    Площадь поверхности вычисляется как сумма площадей всех шести граней
    прямоугольного параллелепипеда. Формула: 2*(xy + yz + xz), где x, y, z -
    размеры AABB по соответствующим осям.
    
    Используется в метрике SAH (Surface Area Heuristic) для оценки качества
    разбиения узла BVH-дерева. Формула SAH: cost = C_traverse + S_left*N_left + S_right*N_right,
    где S_left, S_right - площади поверхности левого/правого боксов,
    N_left, N_right - количество примитивов в левой/правой части.
    
    Parameters
    ----------
    bb_min : np.ndarray
        Минимальные координаты AABB по осям X, Y, Z.
    bb_max : np.ndarray
        Максимальные координаты AABB по осям X, Y, Z.
    
    Returns
    -------
    float
        Площадь поверхности AABB.
        Формула: 2*(d[0]*d[1] + d[1]*d[2] + d[0]*d[2]),
        где d = bb_max - bb_min - размеры AABB по осям (dx, dy, dz).
    """
    d = np.maximum(bb_max - bb_min, 0.0)  # размеры по осям (dx, dy, dz)
    return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[0] * d[2])


def _aabb_volume(bb_min: np.ndarray, bb_max: np.ndarray) -> float:
    """
    Вычисляет объём AABB.
    
    Объём вычисляется как произведение размеров AABB по всем трём осям.
    Формула: V = dx * dy * dz, где dx, dy, dz - размеры AABB по осям X, Y, Z.
    
    Используется в метрике VAH (Volume Area Heuristic) как альтернатива SAH
    для оценки качества разбиения узла BVH-дерева. VAH использует объём вместо
    площади поверхности, что может быть более эффективно для некоторых типов сцен.
    
    Parameters
    ----------
    bb_min : np.ndarray
        Минимальные координаты AABB по осям X, Y, Z.
    bb_max : np.ndarray
        Максимальные координаты AABB по осям X, Y, Z.
    
    Returns
    -------
    float
        Объём AABB.
        Формула: d[0] * d[1] * d[2],
        где d = bb_max - bb_min - размеры AABB по осям (dx, dy, dz).
    """
    d = np.maximum(bb_max - bb_min, 0.0)  # размеры по осям (dx, dy, dz)
    return float(d[0] * d[1] * d[2])


def _aabb_intersect(a_min, a_max, b_min, b_max):
    """
    Проверяет пересечение двух AABB.
    
    Два AABB пересекаются, если они имеют общую область в пространстве.
    Для пересечения необходимо и достаточно, чтобы по всем осям выполнялось условие:
    a_min[i] <= b_max[i] И a_max[i] >= b_min[i] для i = 0, 1, 2 (X, Y, Z).
    
    Используется в BVH для быстрого отсечения непересекающихся пар узлов/примитивов
    при обходе дерева, что значительно ускоряет поиск потенциально пересекающихся
    треугольников.
    
    Parameters
    ----------
    a_min : np.ndarray
        Минимальные координаты первого AABB по осям X, Y, Z.
    a_max : np.ndarray
        Максимальные координаты первого AABB по осям X, Y, Z.
    b_min : np.ndarray
        Минимальные координаты второго AABB по осям X, Y, Z.
    b_max : np.ndarray
        Максимальные координаты второго AABB по осям X, Y, Z.
    
    Returns
    -------
    bool
        True, если AABB пересекаются, False в противном случае.
    """
    return np.all((a_min <= b_max) & (a_max >= b_min))


def _is_neighbour(face_a, face_b):
    """
    Проверяет, являются ли две грани соседями (имеют общее ребро).
    
    Две грани считаются соседями, если у них есть общее ребро (Edge).
    Если грани имеют только общую вершину (Node), но не имеют общего ребра,
    они не считаются соседями.
    
    Parameters
    ----------
    face_a : Face
        Первая грань для проверки.
    face_b : Face
        Вторая грань для проверки.
    
    Returns
    -------
    bool
        True, если грани имеют общее ребро (являются соседями),
        False, если у граней нет общего ребра или есть только общая вершина.
    
    Notes
    -----
    Проверка выполняется через пересечение множеств рёбер граней.
    Если множества рёбер имеют общий элемент (Edge), грани являются соседями.
    Проверка только общих вершин (nodes) не выполняется, так как наличие
    общей вершины без общего ребра не означает, что грани соседи.
    
    Важно: Сравнение объектов Edge работает корректно, потому что:
    1. В структуре Mesh метод add_edge() возвращает существующий объект Edge,
       если ребро между двумя узлами уже существует (через find_edge()).
    2. Если две грани имеют общее ребро (соединяют одни и те же два узла),
       они ссылаются на один и тот же объект Edge в памяти.
    3. В Python объекты без переопределения __hash__ и __eq__ сравниваются
       по идентичности (identity) - то есть по id() объекта.
    4. Множество (set) использует id объекта для хеширования, поэтому
       один и тот же объект Edge будет найден в пересечении множеств.
    """
    common_edges = set(face_a.edges) & set(face_b.edges)
    return len(common_edges) > 0


@dataclass
class BVHNode:
    """
    Узел BVH-дерева.
    
    Используется для представления узла в иерархической структуре BVH.
    Листовые узлы (is_leaf=True) содержат список примитивов (primitives),
    которые находятся внутри их ограничивающего бокса (bounding_box).
    Внутренние узлы (is_leaf=False) содержат два дочерних узла (children),
    которые разделяют пространство по выбранной оси разбиения.
    
    Attributes
    ----------
    node_id : int
        Уникальный идентификатор узла в дереве.
    is_leaf : bool
        Флаг, указывающий, является ли узел листовым (True) или внутренним (False).
    bounding_box : Tuple[np.ndarray, np.ndarray]
        Ограничивающий бокс узла в формате (bb_min, bb_max), где
        bb_min - минимальные координаты по осям X, Y, Z,
        bb_max - максимальные координаты по осям X, Y, Z.
    primitives : List[PrimitiveRef]
        Список примитивов (PrimitiveRef), содержащихся в узле (только для листовых узлов).
        Каждый PrimitiveRef содержит грань (face), её AABB (bb_min, bb_max) и
        координаты вершин (nodes_coords) для быстрого доступа без пересчёта.
        Для доступа к грани используется primitives[i].face.
    children : Tuple[Optional[BVHNode], Optional[BVHNode]]
        Кортеж из двух дочерних узлов (left_child, right_child).
        Для листовых узлов оба значения равны None.
    """
    node_id: int
    is_leaf: bool
    bounding_box: Tuple[np.ndarray, np.ndarray,]
    primitives: List["PrimitiveRef"] = field(default_factory=list)
    children: Tuple[Optional["BVHNode"], Optional["BVHNode"]] = (None, None)


@dataclass
class PrimitiveRef:
    """
    Обёртка над гранью и её AABB, чтобы каждый раз не пересчитывать bbox.
    
    Используется при построении BVH-дерева для хранения грани вместе с
    предвычисленным ограничивающим боксом (AABB) и координатами вершин.
    Это позволяет избежать многократного пересчёта AABB и извлечения координат
    для одной и той же грани при сортировке, оценке разбиений и обходе дерева.
    
    Attributes
    ----------
    face : Face
        Геометрическая грань (треугольник) из сетки.
    bb_min : np.ndarray
        Минимальные координаты AABB грани по осям X, Y, Z.
    bb_max : np.ndarray
        Максимальные координаты AABB грани по осям X, Y, Z.
    nodes_coords : np.ndarray
        Массив координат вершин грани в формате (3, 3), где каждая строка
        содержит координаты одной вершины (x, y, z). Используется для
        быстрого доступа к координатам без обращения к face.nodes[i].p
        при обходе дерева и проверке пересечений.
    """

    face: Face
    bb_min: np.ndarray
    bb_max: np.ndarray
    nodes_coords: np.ndarray

@dataclass
class SplitResult:
    """
    Результат поиска оптимального разбиения узла BVH-дерева.
    
    Содержит информацию о наилучшем разбиении, найденном с помощью
    метрики SAH (Surface Area Heuristic) или VAH (Volume Area Heuristic).
    Разбиение определяется осью (X, Y или Z) и индексом в отсортированном
    списке примитивов, после которого происходит разделение на левую и правую части.
    
    Attributes
    ----------
    cost : float
        Значение метрики SAH/VAH для данного разбиения.
        Формула: cost = C_traverse + S_left*N_left + S_right*N_right,
        где S_left/S_right - площадь поверхности (SAH) или объём (VAH)
        левого/правого боксов, N_left/N_right - количество примитивов.
    axis : int
        Ось разбиения: 0 = X, 1 = Y, 2 = Z.
        Если axis == -1, то оптимальное разбиение не найдено.
    index : int
        Индекс в отсортированном списке примитивов, после которого
        происходит разбиение. Примитивы с индексами [0, index) идут
        в левый дочерний узел, [index, n) - в правый.
    left_bb_min : np.ndarray
        Минимальные координаты AABB левой части после разбиения.
    left_bb_max : np.ndarray
        Максимальные координаты AABB левой части после разбиения.
    right_bb_min : np.ndarray
        Минимальные координаты AABB правой части после разбиения.
    right_bb_max : np.ndarray
        Максимальные координаты AABB правой части после разбиения.
    subdivide_next : bool
        Флаг, указывающий, следует ли продолжать разбиение узла.
        True, если найденное разбиение даёт выигрыш по метрике и
        стоит создавать дочерние узлы. False, если разбиение невыгодно
        и узел должен стать листовым.
    """
    best_cost: float
    axis: int
    index: int
    left_bb_min: np.ndarray
    left_bb_max: np.ndarray
    right_bb_min: np.ndarray
    right_bb_max: np.ndarray
    subdivide_next: bool


def split_other_list(
    full_list: List[PrimitiveRef],
    left_face_ids: set[int]
) -> Tuple[List[PrimitiveRef], List[PrimitiveRef]]:
    """
    Разбивает список примитивов на левую и правую части по идентификаторам граней.
    
    Используется при построении BVH-дерева для корректного разбиения списков примитивов,
    отсортированных по осям, отличным от выбранной оси разбиения. Функция разделяет
    примитивы на две части, проверяя принадлежность каждой грани к левой части по её
    идентификатору (face.glo_id).
    
    Parameters
    ----------
    full_list : List[PrimitiveRef]
        Полный список примитивов для разбиения. Список отсортирован по одной из осей
        (X, Y или Z), отличной от выбранной оси разбиения.
    left_face_ids : set[int]
        Множество идентификаторов граней (face.glo_id), которые должны попасть
        в левую часть. Эти идентификаторы определяются из списка примитивов,
        попавших в левую часть по выбранной оси разбиения.
    
    Returns
    -------
    Tuple[List[PrimitiveRef], List[PrimitiveRef]]
        Кортеж (left_part, right_part), где:
        left_part - список примитивов, грани которых имеют идентификаторы из left_face_ids,
        right_part - список примитивов, грани которых не входят в left_face_ids.
    
    Notes
    -----
    Эта функция необходима, потому что при разбиении узла BVH-дерева:
    1. Для выбранной оси разбиение выполняется напрямую по индексу в отсортированном списке.
    2. Для остальных двух осей списки отсортированы по другим координатам, поэтому
       прямой срез по индексу не даст правильного разбиения (один и тот же примитив
       может быть на разных позициях в разных списках).
    3. Используя face.glo_id как уникальный идентификатор, мы можем корректно
       разделить примитивы в списках, отсортированных по другим осям.
    
    Пример:
        Если по оси X примитивы с индексами [0, 5) попали в левую часть, то для
        списков, отсортированных по осям Y и Z, мы используем face.glo_id этих
        примитивов, чтобы найти их в других списках и правильно разделить.
    """
    left_part: List[PrimitiveRef] = []
    right_part: List[PrimitiveRef] = []
    
    # Разделяем примитивы на левую и правую части, проверяя принадлежность
    # грани к левой части по её идентификатору
    for p in full_list:
        if p.face.glo_id in left_face_ids:
            left_part.append(p)
        else:
            right_part.append(p)
    return left_part, right_part


class BVHTree:
    """
    BVH‑дерево для поиска потенциально пересекающихся пар треугольников.
    
    Реализует иерархическую структуру данных для эффективного поиска пересечений
    между треугольниками в сетке. Дерево строится с использованием метрики SAH
    (Surface Area Heuristic) или VAH (Volume Area Heuristic) для оптимизации
    качества разбиения пространства.
    
    Алгоритм построения:
    - На каждом уровне поддерживаются 3 списка примитивов, отсортированных по
      максимальной координате AABB (vmax) по осям X, Y, Z.
    - Для каждой оси выполняется двухпроходный "sweep" (справа‑налево и слева‑направо)
      для оценки всех возможных разбиений с помощью SAH или VAH.
    - Выбирается разбиение с наилучшей метрикой, после чего списки переупорядочиваются
      и рекурсивно строятся дочерние узлы.
    
    Attributes
    ----------
    OVERSPLIT_THRESHOLD : float
        Пороговое значение для определения, стоит ли продолжать разбиение узла.
        По умолчанию 1.0. Если SAH разбиения превышает это значение, умноженное
        на площадь/объём узла и количество примитивов, разбиение считается невыгодным.
    EMPTY_NODE_TRAVERSAL_COST : float
        Стоимость обхода пустого узла в формуле SAH/VAH. По умолчанию 1.0.
    SplitMetric : Literal["sah", "vah"]
        Тип метрики для оценки разбиения: "sah" (Surface Area Heuristic) или
        "vah" (Volume Area Heuristic).
    mesh : Mesh
        Исходная сетка, для которого строится BVH-дерево.
    faces_in_node : int
        Максимальное количество граней в листовом узле. Если количество примитивов
        в узле меньше или равно этому значению, узел становится листовым.
    nodes_counter : int
        Счётчик для генерации уникальных идентификаторов узлов.
    nodes : List[BVHNode]
        Список всех узлов дерева, созданных при построении.
    root_node : Optional[BVHNode]
        Корневой узел BVH-дерева. None до вызова build_tree().
    faces_to_fix : Dict[int, Tuple[Face, List[List[np.ndarray]]]]
        Словарь, где ключ - идентификатор грани (face.glo_id), значение - кортеж
        (Face, List[List[np.ndarray]]), содержащий саму грань и список отрезков пересечения.
        Каждый отрезок представлен списком из 2 точек (np.ndarray).
        Заполняется при обходе дерева (traversal_tree) после геометрической проверки
        пересечений через CzechClassify.
    candidate_pairs : Dict[int, Dict[str, Tuple[Face, Face]]]
        Словарь кандидатных пар граней для проверки пересечения. Ключ - индекс пары,
        значение - словарь с ключом "faces", содержащим кортеж (face1, face2).
        Заполняется при обходе дерева для пар, AABB которых пересекаются.
    _primitives : List[PrimitiveRef]
        Список подготовленных примитивов (граней с предвычисленными AABB).
        Заполняется в методе prepare_mesh().
    _mesh_bb_min : Optional[np.ndarray]
        Минимальные координаты общего AABB всей сетки. None до вызова prepare_mesh().
    _mesh_bb_max : Optional[np.ndarray]
        Максимальные координаты общего AABB всей сетки. None до вызова prepare_mesh().
    """

    OVERSPLIT_THRESHOLD: float = 1.0
    EMPTY_NODE_TRAVERSAL_COST: float = 1.0
    SplitMetric = Literal["sah", "vah"]
    
    def __init__(self, mesh: Mesh, faces_in_node: int = 1):
        """
        Инициализирует BVH-дерево для заданной сетки.
        
        Создаёт экземпляр BVH-дерева и инициализирует все необходимые структуры данных.
        Перед использованием дерева необходимо вызвать prepare_mesh() для подготовки
        примитивов и build_tree() для построения дерева.
        
        Parameters
        ----------
        mesh : Mesh
            Меш, для которого будет построено BVH-дерево.
        faces_in_node : int, optional
            Максимальное количество граней в листовом узле. По умолчанию 1.
            Узлы с количеством примитивов <= faces_in_node становятся листовыми.
        """
        self.mesh: Mesh = mesh
        self.faces_in_node: int = faces_in_node

        self.nodes_counter: int = 0
        self.nodes: List[BVHNode] = []
        self.root_node: Optional[BVHNode] = None

        # Для последующей геометрической проверки
        # Ключ: face.glo_id, значение: кортеж (Face, List[List[np.ndarray]])
        # Каждый отрезок - это список из 2 точек (np.ndarray)
        self.faces_to_fix: Dict[int, Tuple[Face, List[List[np.ndarray]]]] = {}
        # Пары полученные в результате обхода дерева
        self.candidate_pairs: Dict[int, Tuple[Face, Face]] = {}
        self.candidate_pairs_after_czech: Dict[int, Tuple[Face, Face]] = {}
        self.impossible_couples: Dict[int, List[Tuple[Face, Face]]] = {}
        
        # Подготовленные примитивы (заполняются в prepare_mesh)
        self._primitives: List[PrimitiveRef] = []
        self._mesh_bb_min: Optional[np.ndarray] = None
        self._mesh_bb_max: Optional[np.ndarray] = None
        
    # ----------------------------------------------------------------------------------

    def prepare_mesh(self, esc_enable: bool = False) -> None:
        """
        Предварительная подготовка сетки для построения BVH-дерева.
        
        Для каждой грани вычисляет AABB (ограничивающий бокс) и сохраняет его вместе
        с гранью в списке примитивов. Также вычисляет общий AABB всего сетки путём
        итеративного расширения: начиная с пустого AABB, для каждой грани расширяет
        общий AABB, включая AABB текущей грани. После обработки всех граней получается
        общий AABB, содержащий всю сетку.
        
        Параметр esc_enable зарезервирован под возможные эвристики "раннего останова/разбиения".
        Сейчас он влияет только на логирование.
        
        Parameters
        ----------
        esc_enable : bool, optional
            Флаг включения эвристик раннего останова. По умолчанию False.
            В текущей реализации влияет только на уровень логирования.
        
        Notes
        -----
        Логика вычисления общего AABB:
        - Начинается с пустого AABB (bb_min=[+inf, +inf, +inf], bb_max=[-inf, -inf, -inf])
        - Для каждой грани вычисляется её AABB как min/max координат вершин
        - Общий AABB расширяется через _aabb_include(), который итеративно включает
          AABB каждой грани
        - После цикла mesh_bb_min и mesh_bb_max содержат общий AABB всей сетки,
          так как _aabb_include возвращает расширенный AABB, включающий все предыдущие грани
        """
        logger.info(
                "BVHTree: prepare_mesh started (ESC enabled: %s), faces=%d",
                esc_enable,
                len(self.mesh.faces),
            )
        
        self._primitives.clear()
        mesh_bb_min, mesh_bb_max = _aabb_empty()
        for face in self.mesh.faces:
            coords = np.array([node.p for node in face.nodes], dtype=float)
            bb_min = coords.min(axis=0)
            bb_max = coords.max(axis=0)
            self._primitives.append(PrimitiveRef(face=face, bb_min=bb_min, bb_max=bb_max, nodes_coords=coords))
            # Итеративно расширяем общий AABB, включая AABB текущей грани
            # После цикла mesh_bb_min и mesh_bb_max будут содержать общий AABB всей сетки
            mesh_bb_min, mesh_bb_max = _aabb_include(mesh_bb_min, mesh_bb_max, bb_min, bb_max)
        
        self._mesh_bb_min = mesh_bb_min
        self._mesh_bb_max = mesh_bb_max
        logger.info(
                    "BVHTree: prepare_mesh finished; primitives=%d, mesh_bb_min=%s, mesh_bb_max=%s",
                    len(self._primitives),
                    self._mesh_bb_min,
                    self._mesh_bb_max,
                )

    # ----------------------------------------------------------------------------------
    
    def build_tree(self, split_func: SplitMetric = "sah") -> None:
        """
        Построение BVH-дерева из подготовленных примитивов.
        
        Строит иерархическую структуру данных для эффективного поиска пересечений
        между треугольниками. Алгоритм использует метрику SAH (Surface Area Heuristic)
        или VAH (Volume Area Heuristic) для оптимизации качества разбиения пространства.
        
        Алгоритм построения:
        1. Создаются три списка примитивов, отсортированных по максимальной координате
           AABB (vmax) вдоль каждой оси X, Y, Z.
        2. Рекурсивно строится дерево, начиная с корневого узла (depth=0).
        3. На каждом уровне выполняется поиск оптимального разбиения с помощью
           двухпроходного "sweep" алгоритма для оценки всех возможных разбиений.
        4. Выбирается разбиение с наилучшей метрикой, после чего списки переупорядочиваются
           и рекурсивно строятся дочерние узлы.
        
        Parameters
        ----------
        split_func : SplitMetric, optional
            Метрика для оценки качества разбиения узлов. По умолчанию "sah".
            - "sah" (Surface Area Heuristic): использует площадь поверхности AABB
            - "vah" (Volume Area Heuristic): использует объём AABB
        
        Raises
        ------
        RuntimeError
            Если примитивы не подготовлены. Необходимо вызвать prepare_mesh() перед build_tree().
        ValueError
            Если указана неподдерживаемая метрика разбиения.
        
        Notes
        -----
        Перед вызовом этого метода необходимо вызвать prepare_mesh() для подготовки
        примитивов и вычисления их AABB.
        
        После построения дерева корневой узел доступен через self.root_node,
        а все узлы дерева хранятся в self.nodes.
        
        Параметр depth=0 для корневого узла - стандартная практика в деревьях,
        где корень имеет глубину 0, а каждый уровень вложенности увеличивает depth на 1.
        """
        # Проверка на пустые примитивы
        if not self._primitives:
            raise RuntimeError(
                "BVH: primitives are not prepared. Call prepare_mesh() before build_tree()."
            )
        
        # Проверка на допустимую функцию стоимости
        split_func = split_func.lower()
        if split_func not in ("sah", "vah"):
            raise ValueError(f"Unsupported split_func '{split_func}', expected 'sah' or 'vah'.")
        
        logger.info("BVHTree: build_tree started with split_func='%s'", split_func)
        
        # Инициализируем три списка, отсортированных по vmax AABB вдоль каждой оси
        # Сортировка по максимальной координате (vmax) используется для эффективного
        # двухпроходного sweep алгоритма при поиске оптимального разбиения
        plist_x = sorted(self._primitives, key=lambda p: p.bb_max[0])
        plist_y = sorted(self._primitives, key=lambda p: p.bb_max[1])
        plist_z = sorted(self._primitives, key=lambda p: p.bb_max[2])
        
        # Очищаем список узлов дерева, ставим счетчик идентификаторов в 0
        self.nodes.clear()
        self.nodes_counter = 0
        
        # Начинаем рекурсивное построение с корневого узла
        # Параметр depth=0 для корня - стандартная практика в деревьях
        self.root_node = self._build_node_recursive(
            plist_x,
            plist_y,
            plist_z,
            metric=split_func,
            depth=0,  # Корневой узел всегда имеет глубину 0
        )

        logger.info(
            "BVHTree: build_tree finished; total_nodes=%d, root_id=%s",
            len(self.nodes),
            self.root_node.node_id if self.root_node else None,
        )
    
    # ----------------------------------------------------------------------------------
    
    def _build_node_recursive(
        self,
        plist_x: List[PrimitiveRef],
        plist_y: List[PrimitiveRef],
        plist_z: List[PrimitiveRef],
        metric: SplitMetric,
        depth: int,  # Глубина узла в дереве: 0 для корня, увеличивается на 1 для каждого уровня
    ) -> BVHNode:
        """
        Рекурсивное построение узла BVH-дерева по трём спискам примитивов.
        
        Метод строит узел дерева, начиная с вычисления общего AABB для всех примитивов.
        Затем проверяет критерии остановки (количество примитивов или невыгодность разбиения)
        и либо создаёт листовой узел, либо находит оптимальное разбиение и рекурсивно
        строит дочерние узлы.
        
        Parameters
        ----------
        plist_x : List[PrimitiveRef]
            Список примитивов, отсортированный по максимальной координате AABB по оси X.
            Все три списка содержат одни и те же примитивы, только отсортированные по разным осям.
        plist_y : List[PrimitiveRef]
            Список примитивов, отсортированный по максимальной координате AABB по оси Y.
            Все три списка содержат одни и те же примитивы, только отсортированные по разным осям.
        plist_z : List[PrimitiveRef]
            Список примитивов, отсортированный по максимальной координате AABB по оси Z.
            Все три списка содержат одни и те же примитивы, только отсортированные по разным осям.
        metric : SplitMetric
            Метрика для оценки качества разбиения ("sah" или "vah").
            Используется в методе _find_object_split для выбора оптимальной оси и индекса разбиения.
        depth : int
            Глубина узла в дереве: 0 для корня, увеличивается на 1 для каждого уровня.
            Используется для логирования уровня вложенности узлов при отладке,
            отслеживания структуры дерева и потенциального ограничения максимальной глубины.
        
        Returns
        -------
        BVHNode
            Построенный узел дерева (листовой или внутренний).
            Листовой узел содержит грани в поле faces, внутренний узел содержит
            дочерние узлы в поле children.
        
        Notes
        -----
        Общий алгоритм:
        1. Вычисляется общий AABB для всех примитивов узла путём итеративного
           включения AABB всех примитивов через _aabb_include().
        
        2. Проверяется первый критерий остановки: если количество примитивов
           <= faces_in_node, создаётся листовой узел с этими примитивами.
        
        3. Ищется оптимальное разбиение с помощью _find_object_split (SAH/VAH).
           Метод выполняет двухпроходный sweep алгоритм для всех трёх осей (X, Y, Z)
           и возвращает лучшее разбиение или информацию о том, что разбиение не найдено.
        
        4. Обработка результата поиска разбиения:
           a) Если subdivide_next=True: разбиение выгодно по метрике SAH/VAH,
              используем найденное оптимальное разбиение.
           b) Если subdivide_next=False:
              - Если count <= faces_in_node: создаём листовой узел (эта ветка
                теоретически не должна выполняться, так как проверка уже была выше,
                но оставлена для безопасности).
              - Если split.axis == -1 (разбиение не найдено) и count > faces_in_node:
                создаём листовой узел с предупреждением, так как разбиение невозможно,
                но количество примитивов превышает faces_in_node.
              - Если split.axis != -1 (разбиение найдено, но невыгодно по метрике)
                и count > faces_in_node: принудительно используем найденное разбиение,
                чтобы гарантировать, что листовые узлы содержат не больше faces_in_node граней.
        
        5. Выполнение разбиения по выбранной оси и индексу:
           - Для выбранной оси разбиение выполняется напрямую по индексу в отсортированном списке.
           - Для остальных двух осей используется split_other_list() по face.glo_id,
             так как списки отсортированы по другим осям и прямой срез не даст правильного разбиения.
        
        6. Создаётся внутренний узел (is_leaf=False) с пустым списком граней.
        
        7. Рекурсивно строятся дочерние узлы для левой и правой частей разбиения
           с увеличенной глубиной (depth + 1).
        
        Разбиение списков по осям:
        После выбора оптимальной оси разбиения (axis) и индекса (index), необходимо
        корректно разделить все три списка на левую и правую части:
        - Для выбранной оси: прямое разбиение по индексу (axis_list[:index] и axis_list[index:]).
        - Для остальных двух осей: разбиение по идентификаторам граней (face.glo_id),
          так как списки отсортированы по другим осям и прямой срез по индексу не даст
          правильного разбиения.
        
        Внутренние узлы и primitives=[]:
        Внутренние узлы (is_leaf=False) имеют primitives=[], потому что:
        - Внутренние узлы не содержат примитивы напрямую, они содержат только дочерние узлы (children).
        - Примитивы (PrimitiveRef) хранятся только в листовых узлах (is_leaf=True).
        - Это стандартная структура BVH-дерева: листья содержат примитивы,
          внутренние узлы содержат только структуру разбиения пространства.
        - Все примитивы, которые попадают в область внутреннего узла, распределяются
          между его дочерними узлами при рекурсивном построении.
        """
        # Проверяем, что все три списка содержат одинаковое количество примитивов
        # (они содержат одни и те же примитивы, только отсортированные по разным осям)
        count = len(plist_x)
        assert count == len(plist_y) == len(plist_z)
        
        # Вычисляем общий AABB для узла, итеративно включая AABB всех примитивов
        # Начинаем с пустого AABB, который будет расширен до минимального бокса,
        # содержащего все примитивы узла
        bb_min, bb_max = _aabb_empty()
        for p in plist_x:
            bb_min, bb_max = _aabb_include(bb_min, bb_max, p.bb_min, p.bb_max)
            
        # Критерий остановки рекурсии — создаём листовой узел
        # Если количество примитивов меньше или равно faces_in_node,
        # дальнейшее разбиение не имеет смысла
        if count <= self.faces_in_node:
            node = self._create_node(
                bb_min=bb_min,
                bb_max=bb_max,
                is_leaf=True,
                primitives=plist_x.copy(),  # Копируем список примитивов для листового узла
            )
            logger.debug(
                "BVHTree: created leaf node %d at depth=%d with %d primitives",
                node.node_id,
                depth,
                len(node.primitives),
            )
            return node

        # Ищем оптимальное разбиение узла по одной из осей (X, Y или Z)
        # Метод _find_object_split выполняет двухпроходный sweep алгоритм для всех осей
        # и выбирает разбиение с наилучшей метрикой SAH или VAH
        split = self._find_object_split(plist_x, plist_y, plist_z, bb_min, bb_max, metric)

        # Обработка результата поиска разбиения
        # Если метрика SAH/VAH показала, что разбиение невыгодно (subdivide_next=False),
        # проверяем, нужно ли всё равно разбивать из-за ограничения faces_in_node
        if not split.subdivide_next:
            # Случай 1: Количество примитивов уже <= faces_in_node
            # Теоретически эта ветка не должна выполняться, так как проверка была выше,
            # но оставлена для безопасности и ясности логики
            if count <= self.faces_in_node:
                node = self._create_node(
                    bb_min=bb_min,
                    bb_max=bb_max,
                    is_leaf=True,
                    primitives=plist_x.copy(),  # Копируем список примитивов для листового узла
                )
                logger.debug(
                    "BVHTree: SAH/VAH subdivision not profitable; created leaf node %d at depth=%d with %d primitives, split.subdivide_next=%s",
                    node.node_id,
                    depth,
                    len(node.primitives),
                    split.subdivide_next,
                )
                return node
            
            # Случай 2: SAH не нашёл разбиения (split.axis == -1), но count > faces_in_node
            # Это означает, что разбиение невозможно или не найдено алгоритмом,
            # но количество примитивов превышает ограничение - создаём листовой узел с предупреждением
            elif split.axis == -1:
                node = self._create_node(
                    bb_min=bb_min,
                    bb_max=bb_max,
                    is_leaf=True,
                    primitives=plist_x.copy(),  # Копируем список примитивов для листового узла
                )
                logger.debug(
                    "BVHTree: no split found but count=%d > faces_in_node=%d; created leaf node %d at depth=%d with %d primitives",
                    count,
                    self.faces_in_node,
                    node.node_id,
                    depth,
                    len(node.primitives),
                )
                return node
            
            # Случай 3: SAH нашёл разбиение (split.axis != -1), но оно считается невыгодным по метрике,
            # однако количество примитивов всё ещё > faces_in_node
            # Принудительно используем найденное разбиение, чтобы гарантировать,
            # что листовые узлы будут содержать не больше faces_in_node граней
            else:
                # Используем разбиение, найденное SAH, даже если оно считается невыгодным по метрике
                # Это необходимо для соблюдения ограничения faces_in_node
                axis = split.axis
                index = split.index
                logger.debug(
                    "BVHTree: using SAH split despite not profitable (count=%d > faces_in_node=%d); axis=%d, index=%d at depth=%d",
                    count,
                    self.faces_in_node,
                    axis,
                    index,
                    depth,
                )
        else:
            # Случай 4: Разбиение выгодно по метрике SAH/VAH (subdivide_next=True)
            # Используем оптимальное разбиение, найденное алгоритмом
            axis = split.axis  # Выбранная ось разбиения: 0=X, 1=Y, 2=Z
            index = split.index  # Индекс разбиения в отсортированном списке
        
        # Выбираем список примитивов для выбранной оси
        # Этот список уже отсортирован по максимальной координате AABB (vmax) вдоль этой оси,
        # что позволяет эффективно выполнить разбиение по индексу
        if axis == 0:
            axis_list = plist_x  # Ось X
        elif axis == 1:
            axis_list = plist_y  # Ось Y
        else:
            axis_list = plist_z  # Ось Z
        
        # Разбиваем список выбранной оси напрямую по индексу
        # left_axis содержит примитивы с индексами [0, index) - левая часть
        # right_axis содержит примитивы с индексами [index, n) - правая часть
        left_axis = axis_list[:index]
        right_axis = axis_list[index:]
        
        # Чтобы корректно разрезать остальные два списка (которые отсортированы по другим осям),
        # запоминаем множество идентификаторов граней (face.glo_id) примитивов, попавших в левую часть.
        # Это необходимо, потому что:
        # 1. Остальные списки отсортированы по другим осям, поэтому прямой срез по индексу
        #    не даст правильного разбиения (один и тот же примитив может быть на разных позициях)
        # 2. Мы используем face.glo_id как уникальный идентификатор для поиска примитива
        #    в других списках и определения, в какую часть (левую/правую) он должен попасть
        left_face_ids = {p.face.glo_id for p in left_axis}
        
        # Разбиваем все три списка на левую и правую части:
        # - Для выбранной оси используем прямое разбиение по индексу (уже сделано выше)
        # - Для остальных двух осей используем функцию split_other_list(), которая
        #   разделяет список, проверяя принадлежность каждого примитива к левой части
        #   по его face.glo_id
        #
        # Пример для axis=0 (ось X):
        # - left_x, right_x = left_axis, right_axis (прямое разбиение по индексу)
        # - left_y, right_y = split_other_list(plist_y, left_face_ids) (по face.glo_id)
        # - left_z, right_z = split_other_list(plist_z, left_face_ids) (по face.glo_id)
        #
        # Это гарантирует, что все три списка содержат одни и те же примитивы в левой/правой части,
        # только отсортированные по разным осям
        if axis == 0:
            left_x, right_x = left_axis, right_axis
            left_y, right_y = split_other_list(plist_y, left_face_ids)
            left_z, right_z = split_other_list(plist_z, left_face_ids)
        elif axis == 1:
            left_y, right_y = left_axis, right_axis
            left_x, right_x = split_other_list(plist_x, left_face_ids)
            left_z, right_z = split_other_list(plist_z, left_face_ids)
        else:
            left_z, right_z = left_axis, right_axis
            left_x, right_x = split_other_list(plist_x, left_face_ids)
            left_y, right_y = split_other_list(plist_y, left_face_ids)
        
        # Создаём внутренний узел (is_leaf=False) с пустым списком примитивов
        # Внутренние узлы не содержат примитивы напрямую, они содержат только дочерние узлы
        # Все примитивы, попадающие в область этого узла, будут распределены между дочерними узлами
        node = self._create_node(
            bb_min=bb_min,
            bb_max=bb_max,
            is_leaf=False,
            primitives=[],  # Внутренние узлы не содержат примитивы, только дочерние узлы
        )

        logger.debug(
            "BVHTree: internal node %d at depth=%d; axis=%d, index=%d, left_count=%d, right_count=%d",
            node.node_id,
            depth,
            axis,
            index,
            len(left_x),
            len(right_x),
        )

        # Рекурсивно строим дочерние узлы для левой и правой частей разбиения
        # Каждый дочерний узел получает соответствующие списки примитивов и
        # глубину, увеличенную на 1 (стандартная практика в деревьях)
        left_child = self._build_node_recursive(
            left_x,
            left_y,
            left_z,
            metric=metric,
            depth=depth + 1,  # Увеличиваем глубину для дочернего узла
        )
        right_child = self._build_node_recursive(
            right_x,
            right_y,
            right_z,
            metric=metric,
            depth=depth + 1,  # Увеличиваем глубину для дочернего узла
        )

        # Связываем дочерние узлы с текущим внутренним узлом
        node.children = (left_child, right_child)

        return node
    
    # ----------------------------------------------------------------------------------
    
    def _create_node(
        self,
        bb_min: np.ndarray,
        bb_max: np.ndarray,
        is_leaf: bool,
        primitives: List[PrimitiveRef],
    ) -> BVHNode:
        """
        Создаёт новый узел BVH-дерева и добавляет его в список узлов.
        
        Создаёт экземпляр BVHNode с заданными параметрами, присваивает ему уникальный
        идентификатор (node_id) на основе счётчика nodes_counter, и добавляет узел
        в глобальный список узлов дерева (self.nodes).
        
        Parameters
        ----------
        bb_min : np.ndarray
            Минимальные координаты ограничивающего бокса узла по осям X, Y, Z.
        bb_max : np.ndarray
            Максимальные координаты ограничивающего бокса узла по осям X, Y, Z.
        is_leaf : bool
            Флаг, указывающий, является ли узел листовым (True) или внутренним (False).
        primitives : List[PrimitiveRef]
            Список примитивов (PrimitiveRef), содержащихся в узле. Для листовых узлов
            содержит примитивы с предвычисленными AABB и координатами вершин,
            для внутренних узлов должен быть пустым списком ([]).
        
        Returns
        -------
        BVHNode
            Созданный узел дерева с присвоенным node_id и добавленный в self.nodes.
        
        Notes
        -----
        - Уникальный идентификатор node_id присваивается автоматически на основе
          текущего значения nodes_counter, после чего счётчик увеличивается на 1.
        - Узел автоматически добавляется в список self.nodes для последующего
          доступа и анализа структуры дерева.
        - Для внутренних узлов (is_leaf=False) параметр primitives должен быть пустым
          списком, так как внутренние узлы содержат только дочерние узлы, а не примитивы.
        - Для доступа к грани из примитива используется primitives[i].face.
        """
        # Создаём узел с уникальным идентификатором на основе текущего счётчика
        node = BVHNode(
            node_id=self.nodes_counter,
            bounding_box=(bb_min, bb_max),
            is_leaf=is_leaf,
            primitives=primitives,
        )
        # Увеличиваем счётчик для следующего узла
        self.nodes_counter += 1
        # Добавляем узел в глобальный список для последующего доступа
        self.nodes.append(node)
        return node
    
    # ----------------------------------------------------------------------------------
    
    def _find_object_split(
        self,
        plist_x: List[PrimitiveRef],
        plist_y: List[PrimitiveRef],
        plist_z: List[PrimitiveRef],
        node_bb_min: np.ndarray,
        node_bb_max: np.ndarray,
        metric: SplitMetric,
    ) -> SplitResult:
        """
        Находит оптимальное разбиение узла BVH-дерева с использованием SAH или VAH.
        
        Реализует двухпроходный "sweep" алгоритм для поиска оптимального разбиения узла
        по одной из осей (X, Y или Z). Для каждой оси выполняется:
        1. Сортировка примитивов по максимальной координате AABB (vmax) вдоль этой оси.
        2. Sweep справа налево: вычисление AABB правых частей для всех возможных разбиений.
        3. Sweep слева направо: вычисление AABB левых частей и оценка метрики SAH/VAH
           для каждого возможного разбиения.
        4. Выбор разбиения с наименьшей стоимостью.
        
        Parameters
        ----------
        plist_x : List[PrimitiveRef]
            Список примитивов, отсортированный по максимальной координате AABB по оси X.
        plist_y : List[PrimitiveRef]
            Список примитивов, отсортированный по максимальной координате AABB по оси Y.
        plist_z : List[PrimitiveRef]
            Список примитивов, отсортированный по максимальной координате AABB по оси Z.
        node_bb_min : np.ndarray
            Минимальные координаты AABB узла по осям X, Y, Z.
        node_bb_max : np.ndarray
            Максимальные координаты AABB узла по осям X, Y, Z.
        metric : SplitMetric
            Метрика для оценки качества разбиения: "sah" (Surface Area Heuristic) или
            "vah" (Volume Area Heuristic).
        
        Returns
        -------
        SplitResult
            Результат поиска оптимального разбиения, содержащий:
            - best_cost: стоимость лучшего разбиения (SAH/VAH)
            - axis: выбранная ось разбиения (0=X, 1=Y, 2=Z, -1 если не найдено)
            - index: индекс разбиения в отсортированном списке
            - left_bb_min/max, right_bb_min/max: AABB левой и правой частей
            - subdivide_next: флаг, указывающий, стоит ли продолжать разбиение
        
        Notes
        -----
        Алгоритм двухпроходного sweep:
        1. Для каждой оси (X, Y, Z):
           - Сортировка примитивов по vmax (максимальная координата AABB) вдоль оси.
           - Sweep справа налево: для каждого индекса i вычисляется AABB примитивов
             с индексами [i, n), сохраняется в right_bounds_min/max[i-1].
           - Sweep слева направо: для каждого индекса i вычисляется AABB примитивов
             с индексами [0, i) и оценивается метрика разбиения.
        
        2. Формула метрики SAH/VAH:
           cost = C_traverse + S_left * N_left + S_right * N_right
           где:
           - C_traverse = EMPTY_NODE_TRAVERSAL_COST (стоимость обхода узла)
           - S_left/S_right: площадь поверхности (SAH) или объём (VAH) левого/правого AABB
           - N_left/N_right: количество примитивов в левой/правой части
        
        3. Выбор оптимального разбиения:
           - Выбирается разбиение с минимальной стоимостью среди всех осей и индексов.
           - Разбиение считается выгодным, если его стоимость меньше порогового значения:
             OVERSPLIT_THRESHOLD * node_measure * primitives_count
        
        4. Если оптимальное разбиение не найдено (best_axis == -1), возвращаются
           защитные значения AABB, равные AABB всего узла.
        """
        # Вычисляем меру узла (площадь поверхности для SAH или объём для VAH)
        # Это используется для определения порогового значения стоимости разбиения
        node_measure = (
            _aabb_surface_area(node_bb_min, node_bb_max)
            if metric == "sah"
            else _aabb_volume(node_bb_min, node_bb_max)
        )
        primitives_count = len(plist_x)
        
        # Начальное "худшее" значение стоимости разбиения
        # Используется как начальное значение для поиска минимальной стоимости разбиения
        # Инициализируем очень большим значением, чтобы любое найденное разбиение было лучше
        # Используем стоимость неразбитого узла (node_measure * N) как верхнюю границу
        cost_no_split = node_measure * float(primitives_count)
        best_split_cost = float('inf')  # Начинаем с бесконечности, чтобы найти любое разбиение
        
        # Инициализируем параметры лучшего разбиения
        # best_axis = -1 означает, что разбиение ещё не найдено
        best_axis = -1
        best_index = -1
        best_left_min, best_left_max = None, None
        best_right_min, best_right_max = None, None
        
        plists = [plist_x, plist_y, plist_z]
        n = primitives_count

        # Временный массив для хранения AABB правых частей для всех возможных разбиений
        # right_bounds_min[i-1] и right_bounds_max[i-1] содержат AABB примитивов [i, n)
        right_bounds_min = [None] * n
        right_bounds_max = [None] * n
        
        # Перебираем все три оси (X, Y, Z) для поиска оптимального разбиения
        for dim in range(3):
            plist = plists[dim]

            # Сортируем примитивы по максимальной координате AABB (vmax) вдоль оси dim
            # Это необходимо для эффективного двухпроходного sweep алгоритма
            plist.sort(key=lambda p: p.bb_max[dim])
            
            # Первый проход (sweep справа налево): вычисляем AABB правых частей
            # Для каждого индекса i вычисляем AABB примитивов с индексами [i, n)
            # и сохраняем в right_bounds_min/max[i-1]
            rb_min, rb_max = _aabb_empty()
            for i in range(n - 1, 0, -1):
                # Итеративно расширяем AABB, включая примитив с индексом i
                rb_min, rb_max = _aabb_include(rb_min, rb_max, plist[i].bb_min, plist[i].bb_max)
                # Сохраняем AABB правой части для разбиения по индексу i
                right_bounds_min[i - 1] = rb_min
                right_bounds_max[i - 1] = rb_max
                
            # Второй проход (sweep слева направо): вычисляем AABB левых частей и оцениваем метрику
            # Для каждого индекса i вычисляем AABB примитивов [0, i) и оцениваем стоимость разбиения
            lb_min, lb_max = _aabb_empty()
            for i in range(1, n):
                # Включаем примитив с индексом i-1 в левую часть
                p = plist[i - 1]
                lb_min, lb_max = _aabb_include(lb_min, lb_max, p.bb_min, p.bb_max)
                
                # Проверяем, что справа есть примитивы (разбиение имеет смысл)
                if right_bounds_min[i - 1] is None:
                    # означает, что справа нет примитивов — разбиение бессмысленно
                    continue
                
                # Вычисляем меру (площадь поверхности или объём) для левой и правой частей
                if metric == "sah":
                    left_measure = _aabb_surface_area(lb_min, lb_max)
                    right_measure = _aabb_surface_area(
                        right_bounds_min[i - 1], right_bounds_max[i - 1]
                    )
                else:
                    left_measure = _aabb_volume(lb_min, lb_max)
                    right_measure = _aabb_volume(
                        right_bounds_min[i - 1], right_bounds_max[i - 1]
                    )
                
                # Вычисляем стоимость разбиения по формуле SAH/VAH:
                # cost = C_traverse + S_left * N_left + S_right * N_right
                split_cost = (
                    self.EMPTY_NODE_TRAVERSAL_COST  # C_traverse
                    + left_measure * float(i)  # S_left * N_left
                    + right_measure * float(n - i)  # S_right * N_right
                )
                
                # Если найденное разбиение лучше текущего лучшего, обновляем параметры
                if split_cost < best_split_cost:
                    best_split_cost = split_cost
                    best_axis = dim  # Сохраняем ось разбиения
                    best_index = i  # Сохраняем индекс разбиения
                    # Сохраняем AABB левой и правой частей (копируем, чтобы избежать изменений)
                    best_left_min, best_left_max = lb_min.copy(), lb_max.copy()
                    best_right_min = right_bounds_min[i - 1].copy()
                    best_right_max = right_bounds_max[i - 1].copy()
        
        # Определяем, стоит ли продолжать разбиение узла
        # Разбиение выгодно, если:
        # 1. Найдено оптимальное разбиение (best_axis != -1)
        # 2. Стоимость разбиения меньше стоимости неразбитого узла, умноженной на порог
        # Стоимость неразбитого узла = node_measure * N (просто обход всех примитивов)
        cost_no_split = node_measure * float(primitives_count)
        subdivide_next = (
            best_axis != -1
            and best_split_cost < self.OVERSPLIT_THRESHOLD * cost_no_split
        )
        
        # Если оптимальное разбиение не найдено, устанавливаем защитные значения AABB
        # (равные AABB всего узла), чтобы избежать ошибок при использовании результата
        if best_axis == -1:
            # Защитное значение AABB, когда разбиение не найдено
            best_left_min, best_left_max = node_bb_min, node_bb_max
            best_right_min, best_right_max = node_bb_min, node_bb_max
            
        logger.debug(
            "BVHTree: _find_object_split metric=%s, primitives=%d, best_axis=%d, "
            "best_index=%d, best_cost=%.6f, subdivide_next=%s",
            metric,
            primitives_count,
            best_axis,
            best_index,
            best_split_cost,
            subdivide_next,
        )

        return SplitResult(
            best_cost=best_split_cost,
            axis=best_axis,
            index=best_index,
            left_bb_min=best_left_min,
            left_bb_max=best_left_max,
            right_bb_min=best_right_min,
            right_bb_max=best_right_max,
            subdivide_next=subdivide_next,
        )
    
    # ----------------------------------------------------------------------------------

    def build_graph(self, root: Optional[BVHNode] = None) -> nx.DiGraph:
        """
        Строит ориентированный граф BVH-дерева для последующей визуализации.
        
        Создаёт граф (networkx.DiGraph), представляющий структуру BVH-дерева,
        где узлы графа соответствуют узлам дерева, а рёбра - связям родитель-потомок.
        Граф может быть использован для визуализации структуры дерева с помощью
        библиотек визуализации графов.
        
        Parameters
        ----------
        root : Optional[BVHNode], optional
            Корневой узел дерева для построения графа. Если не указан (None),
            используется self.root_node. По умолчанию None.
        
        Returns
        -------
        nx.DiGraph
            Ориентированный граф, представляющий структуру BVH-дерева.
            Узлы графа содержат атрибуты:
            - node_id: идентификатор узла дерева
            - label: строковое представление идентификатора
            - is_leaf: флаг, является ли узел листовым
        
        Raises
        ------
        RuntimeError
            Если root не указан и дерево не построено (self.root_node is None).
            Необходимо вызвать build_tree() перед build_graph().
        
        Notes
        -----
        Граф строится с помощью обхода дерева в глубину (DFS - Depth-First Search).
        Для каждого узла дерева:
        1. Добавляется узел в граф с атрибутами node_id, label, is_leaf.
        2. Для каждого дочернего узла добавляется ориентированное ребро
           от родителя к потомку и выполняется рекурсивный обход.
        
        Граф может быть использован для:
        - Визуализации структуры дерева (модуль visualization_mesh.visualization)
        - Анализа глубины и балансировки дерева (модуль visualization_mesh.statistics) 
        """
        # Если корневой узел не указан, используем корневой узел дерева
        if root is None:
            if self.root_node is None:
                raise RuntimeError("BVH: tree is not built. Call build_tree() first.")
            root = self.root_node

        # Создаём пустой ориентированный граф
        graph = nx.DiGraph()

        # Внутренняя функция для обхода дерева в глубину (DFS)
        # Рекурсивно обходит все узлы дерева и добавляет их в граф
        def _dfs(node: BVHNode) -> None:
            # Добавляем узел в граф с атрибутами для последующей визуализации
            graph.add_node(
                node.node_id,
                label=str(node.node_id),  # Строковое представление ID для меток
                is_leaf=node.is_leaf,  # Флаг листового узла для стилизации
            )
            # Обходим дочерние узлы и добавляем рёбра в граф
            for child in node.children:
                if child is not None:
                    # Добавляем ориентированное ребро от родителя к потомку
                    graph.add_edge(node.node_id, child.node_id)
                    # Рекурсивно обходим дочерний узел
                    _dfs(child)

        # Начинаем обход дерева с корневого узла
        _dfs(root)

        logger.info(
            "BVHTree: build_graph finished; nodes=%d, edges=%d",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        return graph
    
    # ----------------------------------------------------------------------------------    
    
    def traversal_tree(self) -> Dict[int, Tuple[Face, List[List[Node]]]]:
        """
        Выполняет обход BVH дерева для поиска пересечений граней между собой.
        
        Метод является основным алгоритмом для обнаружения самопересечений сетки.
        Он использует пространственную структуру BVH дерева для эффективного
        отсеивания заведомо непересекающихся пар граней, а затем применяет
        точный геометрический алгоритм CzechClassify для проверки пересечений.
        
        Алгоритм состоит из двух основных этапов:
        
        1. **Обход BVH дерева с проверкой AABB пересечений**:
           - Использует итеративный обход с помощью стека
           - Начинает с пары (root_node, root_node) для проверки всех пар граней
           - Для каждой пары узлов проверяет пересечение их AABB
           - Если AABB не пересекаются, поддерево пропускается
           - Когда оба узла - листья, перебирает все пары примитивов (граней) в них
        
        2. **Геометрическая проверка пересечений через CzechClassify**:
           - Для каждой пары граней из листовых узлов применяет CzechClassify
           - Фильтрует пары граней-соседей (имеющих общее ребро)
           - Выполняет дополнительную проверку AABB на уровне примитивов
           - При обнаружении пересечения добавляет сегмент пересечения в faces_to_fix
        
        3. **Обработка невозможных случаев через neighbor tracing**:
           - Если классификация попадает в невозможный случай (impossible case),
             пара сохраняется для последующей обработки
           - После основного обхода проверяются соседние грани для пар из impossible cases
           - Согласно теории Czech алгоритма, граница пересечения продолжается через
             соседние треугольники, поэтому проверяются пары (сосед face_a, face_b)
             и (face_a, сосед face_b)
        
        Returns
        -------
        Dict[int, Tuple[Face, List[List[Node]]]]
            Словарь с гранями, имеющими пересечения, и их сегментами пересечения:
            - Ключ: int - идентификатор грани (face.glo_id)
            - Значение: Tuple[Face, List[List[Node]]] - кортеж из грани и списка
              сегментов пересечения. Каждый сегмент - это список из 2 объектов Node
              (или 1 Node, дублированного для точки касания)
        
        Raises
        ------
        RuntimeError
            Если BVH дерево не построено (self.root_node is None). Необходимо сначала
            вызвать build_tree().
        
        ValueError
            Если CzechClassify обнаружил пересечение (has_intersection = True),
            но intersection_result пустой. Это недопустимая ситуация, указывающая
            на ошибку в логике CzechClassify.
        
        Notes
        -----
        Метод использует множество checked_pairs для избежания дублирования проверок:
        - Пары хранятся как кортежи из отсортированных glo_id граней: (min_id, max_id)
        - Это гарантирует, что пара (a, b) и (b, a) проверяются только один раз
        
        Метод также использует фильтрацию пар граней-соседей через _is_neighbour():
        - Грани-соседи (имеющие общее ребро) не проверяются на пересечение,
          так как они по определению не могут пересекаться (кроме как по границе)
        
        Обработка impossible cases:
        - Пары граней, попавшие в impossible cases классификации (например, '001', '002'),
          сохраняются в _impossible_pairs_queue для последующей обработки
        - После основного цикла эти пары обрабатываются через проверку соседних граней
        - Это реализация neighbor tracing согласно теории Czech алгоритма
        
        Examples
        --------
        >>> bvh = BVHTree(mesh, faces_in_node=1)
        >>> bvh.prepare_mesh(esc_enable=False)
        >>> bvh.build_tree(split_func="sah")
        >>> faces_to_fix = bvh.traversal_tree()
        >>> print(f"Found {len(faces_to_fix)} faces with intersections")
        """
        
        # Проверка на построение BVH дерева
        # Метод требует, чтобы дерево было построено заранее через build_tree()
        if self.root_node is None:
            raise RuntimeError("BVH: tree is not built. Call build_tree() first.")

        logger.info("BVHTree: traversal_tree started")
        
        # Множество проверенных пар граней для избежания дублирования проверок
        # Хранит кортежи из отсортированных glo_id граней: (min_id, max_id)
        # Это гарантирует, что пара (a, b) и (b, a) проверяются только один раз
        checked_pairs: set = set()
        # Счётчик обращений к checked_pairs (для статистики и отладки)
        checked_pairs_count = 0
        
        # Стек для итеративного обхода BVH дерева
        # Содержит пары узлов (node_a, node_b) для проверки пересечения их AABB
        # Начинаем с пары (root_node, root_node) для проверки всех пересечений в дереве
        # Алгоритм: если AABB двух узлов пересекаются, добавляем их дочерние узлы в стек
        stack: List[Tuple[BVHNode, BVHNode]] = [(self.root_node, self.root_node)]
        
        # Основной цикл обхода дерева
        # Итеративно обрабатываем пары узлов из стека до его опустошения
        while stack:
            node_a, node_b = stack.pop()

            # Получаем AABB (Axis-Aligned Bounding Box) для обоих узлов
            # AABB используется для быстрого отсеивания заведомо непересекающихся пар
            a_min, a_max = node_a.bounding_box
            b_min, b_max = node_b.bounding_box

            # Проверка пересечения AABB узлов
            # Если AABB не пересекаются, то грани в поддеревьях заведомо не пересекаются
            # и можно пропустить всю ветку дерева
            if not _aabb_intersect(a_min, a_max, b_min, b_max):
                logger.debug("BVHTree: node_a: %d, node_b: %d has't intersection, skip check", node_a.node_id, node_b.node_id)
                continue
            
            logger.debug("BVHTree: node_a: %d, node_b: %d has intersection, start check", node_a.node_id, node_b.node_id)
            
            # --- Случай 1: Оба узла - листья ---
            # В листовых узлах хранятся примитивы (PrimitiveRef), содержащие грани
            # Перебираем все пары примитивов из обоих листовых узлов
            if node_a.is_leaf and node_b.is_leaf:
                # Перебираем все пары примитивов из листовых узлов
                # Каждый примитив (PrimitiveRef) содержит ссылку на грань (Face)
                for p1 in node_a.primitives:
                    for p2 in node_b.primitives:
                        # Получаем грани из примитивов
                        f1, f2 = p1.face, p2.face
                        
                        # --- Фильтрация 1: Проверка на совпадение граней ---
                        # Если грани совпадают (в разных листьях оказалась одна и та же грань),
                        # пропускаем эту пару, так как грань не может пересекаться сама с собой
                        if f1 is f2:
                            logger.debug("BVHTree: node_a: %d, node_b: %d, f1: %d is f2: %d", node_a.node_id, node_b.node_id, f1.glo_id, f2.glo_id)
                            continue
                        
                        # Формируем ключ пары из отсортированных идентификаторов граней
                        # Это гарантирует, что пара (a, b) и (b, a) имеют одинаковый ключ
                        key = tuple(sorted((f1.glo_id, f2.glo_id)))
                        
                        # --- Фильтрация 2: Проверка на дублирование пар ---
                        # Если такая пара уже проверялась ранее, пропускаем её
                        # Это происходит, когда одна и та же пара граней попадает в разные листовые узлы
                        if key in checked_pairs:
                            logger.debug("BVHTree: node_a: %d, node_b: %d, count call to checked_pairs_count, %d", node_a.node_id, node_b.node_id, checked_pairs_count)
                            checked_pairs_count += 1
                            continue
                        
                        # Добавляем ключ проверенной пары в множество checked_pairs
                        # Это гарантирует, что пара не будет проверяться повторно
                        checked_pairs.add(key)
                        
                        # --- Фильтрация 3: Проверка на соседей ---
                        # Грани-соседи (имеющие общее ребро) не проверяются на пересечение,
                        # так как они по определению не могут пересекаться (кроме как по границе ребра)
                        # Соседние грани должны быть обработаны другими механизмами
                        if _is_neighbour(f1, f2):
                            logger.debug("BVHTree: node_a: %d, node_b: %d, f1: %d, f2: %d is neighbour, skip pair", node_a.node_id, node_b.node_id, f1.glo_id, f2.glo_id)
                            continue
                        
                        # --- Фильтрация 4: Дополнительная проверка AABB на уровне примитивов ---
                        # Хотя AABB узлов пересекаются, AABB отдельных граней могут не пересекаться
                        # Используем предвычисленные AABB из PrimitiveRef (bb_min, bb_max),
                        # которые были вычислены в prepare_mesh(), вместо пересчёта
                        f1_min, f1_max = p1.bb_min, p1.bb_max
                        f2_min, f2_max = p2.bb_min, p2.bb_max
                        if not _aabb_intersect(f1_min, f1_max, f2_min, f2_max):
                            logger.debug("BVHTree: node_a: %d, node_b: %d, check intersection for f1: %d, f2: %d", node_a.node_id, node_b.node_id, f1.glo_id, f2.glo_id)
                            continue
                        
                        # --- Пары прошли все фильтры - добавляем в кандидаты ---
                        # Пара граней является кандидатом на пересечение и будет проверена
                        # через точный геометрический алгоритм CzechClassify
                        idx = len(self.candidate_pairs)
                        self.candidate_pairs[idx] = (f1, f2)
                        logger.debug("BVHTree: Added new candidate pair {%d: (%d, %d)}", idx, f1.glo_id, f2.glo_id)
                        
                        # --- Геометрическая проверка пересечения через CzechClassify ---
                        # CzechClassify использует edge-plane intersection для классификации
                        # пересечения двух треугольников согласно теории Czech алгоритма
                        # Метод get_intersection() возвращает:
                        # - has_intersection: True если найдено пересечение, False иначе
                        # - intersection_result: список из 2 объектов Node (сегмент пересечения)
                        # - impossible_couple: список пар граней, попавших в impossible cases
                        czc = CzechClassify(candidates=(f1, f2), checked_pairs=checked_pairs, pair_index=idx)
                        has_intersection, intersection_result, impossible_couple = czc.get_intersection()
                        
                        # --- Обработка результата пересечения ---
                        if has_intersection:
                            # Пересечение найдено - добавляем сегмент пересечения к обеим граням
                            # intersection_result содержит список из 2 объектов Node (сегмент пересечения)
                            # или [Node, Node] с одинаковыми координатами для точки касания
                            
                            # Проверка на ошибку: если has_intersection = True,
                            # intersection_result не должен быть пустым
                            if not intersection_result:
                                raise ValueError("has_intersection = True and intersection_result None is impossible case, need to check this")
                            
                            # Инициализируем запись в faces_to_fix для f1, если её ещё нет
                            if f1.glo_id not in self.faces_to_fix:
                                self.faces_to_fix[f1.glo_id] = (f1, [])

                            # Инициализируем запись в faces_to_fix для f2, если её ещё нет
                            if f2.glo_id not in self.faces_to_fix:
                                self.faces_to_fix[f2.glo_id] = (f2, [])

                            # Добавляем сегмент пересечения к обеим граням
                            # intersection_result - это список из 2 объектов Node, представляющий сегмент
                            self.faces_to_fix[f1.glo_id][1].append(intersection_result)
                            self.faces_to_fix[f2.glo_id][1].append(intersection_result)
                            
                            # Сохраняем пару граней, для которых найдено пересечение
                            # для последующего анализа и визуализации
                            idx = len(self.candidate_pairs_after_czech)
                            self.candidate_pairs_after_czech[idx] = (f1, f2)
                        else:
                            # Пересечение не найдено - возможно, это impossible case
                            # Согласно теории Czech алгоритма, impossible cases (например, '001', '002')
                            # указывают на числовую неточность вычислений
                            # В таких случаях граница пересечения продолжается через соседние треугольники
                            
                            # Сохраняем пары из impossible_couple для последующей обработки
                            # через neighbor tracing после основного цикла обхода дерева
                            if impossible_couple:
                                # impossible_couple - это список кортежей (Face, Face) пар,
                                # которые попали в impossible cases классификации
                                # Добавляем их в очередь для обработки после основного цикла
                                if not hasattr(self, '_impossible_pairs_queue'):
                                    self._impossible_pairs_queue = []
                                self._impossible_pairs_queue.extend(impossible_couple)
                # Продолжаем цикл, так как оба узла были листьями и обработаны
                continue
                                  
            # --- Случаи 2-4: Раскрытие внутренних узлов ---
            # Если хотя бы один из узлов является внутренним (не листом),
            # добавляем в стек пары дочерних узлов для дальнейшей проверки
            
            # Случай 2: node_a - лист, node_b - внутренний узел
            # Добавляем в стек все пары (node_a, child_b) для каждого дочернего узла node_b
            if node_a.is_leaf and not node_b.is_leaf:
                for child in node_b.children:
                    if child is not None:
                        stack.append((node_a, child))
            
            # Случай 3: node_a - внутренний узел, node_b - лист
            # Добавляем в стек все пары (child_a, node_b) для каждого дочернего узла node_a
            elif not node_a.is_leaf and node_b.is_leaf:
                for child in node_a.children:
                    if child is not None:
                        stack.append((child, node_b))
            
            # Случай 4: Оба узла - внутренние
            # Добавляем в стек все пары (child_a, child_b) для всех комбинаций
            # дочерних узлов node_a и node_b
            else:
                for child_a in node_a.children:
                    for child_b in node_b.children:
                        if child_a is not None and child_b is not None:
                            stack.append((child_a, child_b))
        
        # --- Этап 2: Обработка impossible_couples через neighbor tracing ---
        # Согласно теории Czech алгоритма (CZECH_THEORY.md, строки 100-104):
        # при невозможном случае классификации граница пересечения продолжается через соседние треугольники
        # Алгоритм neighbor tracing проверяет соседние грани для пар, попавших в impossible cases,
        # чтобы найти правильное продолжение границы пересечения
        
        if hasattr(self, '_impossible_pairs_queue') and self._impossible_pairs_queue:
            logger.debug("BVHTree: Processing %d impossible pairs through neighbor tracing", len(self._impossible_pairs_queue))
            
            # Обрабатываем каждую пару граней из очереди impossible cases
            for face_a, face_b in self._impossible_pairs_queue:
                # Получаем списки соседних граней для обеих граней
                # Соседние грани - это грани, имеющие общее ребро с текущей гранью
                neighbors_a = face_a.neighbourhood()
                neighbors_b = face_b.neighbourhood()
                
                # --- Вариант 1: Проверка пар (сосед face_a, face_b) ---
                # Проверяем пересечение между соседними гранями face_a и face_b
                # Это соответствует случаю, когда граница пересечения продолжается через
                # соседнюю грань face_a, а не через саму face_a
                for neighbor_face_a in neighbors_a:
                    # Проверяем, не проверяли ли мы уже эту пару ранее в основном цикле
                    pair_key = tuple(sorted((neighbor_face_a.glo_id, face_b.glo_id)))
                    if pair_key in checked_pairs:
                        continue
                    
                    # Фильтруем пары граней-соседей (они не могут пересекаться)
                    if _is_neighbour(neighbor_face_a, face_b):
                        continue
                    
                    # Помечаем пару как проверенную
                    checked_pairs.add(pair_key)
                    
                    # Выполняем геометрическую проверку пересечения для новой пары
                    neighbor_cz = CzechClassify(
                        candidates=(neighbor_face_a, face_b),
                        checked_pairs=set(),  # Создаём новый set для checked_pairs (локальный для neighbor tracing)
                        pair_index=-1,  # Используем специальный индекс для neighbor tracing
                    )
                    has_intersection, intersection_result, _ = neighbor_cz.get_intersection()
                    
                    # Если найдено пересечение через соседа, добавляем его к соответствующим граням
                    if has_intersection and intersection_result:
                        neighbor_id = neighbor_face_a.glo_id
                        face_b_id = face_b.glo_id
                        
                        # Инициализируем записи в faces_to_fix, если их ещё нет
                        if neighbor_id not in self.faces_to_fix:
                            self.faces_to_fix[neighbor_id] = (neighbor_face_a, [])
                        if face_b_id not in self.faces_to_fix:
                            self.faces_to_fix[face_b_id] = (face_b, [])
                        
                        # Добавляем сегмент пересечения к обеим граням
                        self.faces_to_fix[neighbor_id][1].append(intersection_result)
                        self.faces_to_fix[face_b_id][1].append(intersection_result)
                        
                        logger.debug(
                            "BVHTree: neighbor_tracing found intersection via neighbor face_a (%d -> %d, face_b=%d)",
                            face_a.glo_id,
                            neighbor_id,
                            face_b_id,
                        )
                        # Нашли пересечение через соседа face_a, переходим к следующей паре из очереди
                        break
                
                # --- Вариант 2: Проверка пар (face_a, сосед face_b) ---
                # Проверяем пересечение между face_a и соседними гранями face_b
                # Это соответствует случаю, когда граница пересечения продолжается через
                # соседнюю грань face_b, а не через саму face_b
                for neighbor_face_b in neighbors_b:
                    # Проверяем, не проверяли ли мы уже эту пару ранее в основном цикле
                    pair_key = tuple(sorted((face_a.glo_id, neighbor_face_b.glo_id)))
                    if pair_key in checked_pairs:
                        continue
                    
                    # Фильтруем пары граней-соседей (они не могут пересекаться)
                    if _is_neighbour(face_a, neighbor_face_b):
                        continue
                    
                    # Помечаем пару как проверенную
                    checked_pairs.add(pair_key)
                    
                    # Выполняем геометрическую проверку пересечения для новой пары
                    neighbor_cz = CzechClassify(
                        candidates=(face_a, neighbor_face_b),
                        checked_pairs=set(),  # Создаём новый set для checked_pairs (локальный для neighbor tracing)
                        pair_index=-1,  # Используем специальный индекс для neighbor tracing
                    )
                    has_intersection, intersection_result, _ = neighbor_cz.get_intersection()
                    
                    # Если найдено пересечение через соседа, добавляем его к соответствующим граням
                    if has_intersection and intersection_result:
                        face_a_id = face_a.glo_id
                        neighbor_id = neighbor_face_b.glo_id
                        
                        # Инициализируем записи в faces_to_fix, если их ещё нет
                        if face_a_id not in self.faces_to_fix:
                            self.faces_to_fix[face_a_id] = (face_a, [])
                        if neighbor_id not in self.faces_to_fix:
                            self.faces_to_fix[neighbor_id] = (neighbor_face_b, [])
                        
                        # Добавляем сегмент пересечения к обеим граням
                        self.faces_to_fix[face_a_id][1].append(intersection_result)
                        self.faces_to_fix[neighbor_id][1].append(intersection_result)
                        
                        logger.debug(
                            "BVHTree: neighbor_tracing found intersection via neighbor face_b (face_a=%d, %d -> %d)",
                            face_a_id,
                            face_b.glo_id,
                            neighbor_id,
                        )
                        # Нашли пересечение через соседа face_b, переходим к следующей паре из очереди
                        break
            
            # Очищаем очередь impossible pairs после обработки
            # Эта очередь использовалась только для neighbor tracing и больше не нужна
            del self._impossible_pairs_queue
                                                         
        logger.info(
            "BVHTree: traversal_tree finished; candidate_pairs=%d, candidate_pairs_after_czech=%d, faces_to_fix=%d, checked_pairs=%d",
            len(self.candidate_pairs),
            len(self.candidate_pairs_after_czech),
            len(self.faces_to_fix),
            len(checked_pairs),
        )
        logger.debug("BVHTree: Values in checked_pairs=%s", len(checked_pairs))
        return self.faces_to_fix
                        

if __name__ == '__main__':
    mesh = Mesh("examples/small_sphere_double.dat")
    # Проверка функции _is_neighbour, 
    # при общем ребре, должна возвращать True
    # при общей одной вершине и отсутствия общих ребер, должна возвращать False
    print(_is_neighbour(mesh.faces[0], mesh.faces[0]))
    print(_is_neighbour(mesh.faces[0], mesh.faces[1]))
    