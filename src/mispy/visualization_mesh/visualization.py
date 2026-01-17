"""
Модуль визуализации сетки и результатов поиска пересечений.

Модуль предоставляет функции для визуализации треугольных сеток, граней с пересечениями
и пар граней-кандидатов на пересечение. Использует matplotlib для 3D визуализации.

Иерархия функций:
- Базовые функции: draw_face, draw_intersection_seg
- Функции среднего уровня: face_with_intersection_segment_plotter
- Функции высокого уровня: mesh_plotter, pairs_broken_face_plotter
"""

import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mispy.extract_mesh import Mesh, Face, Node

logger = logging.getLogger(__name__)


# ==================================================================================================
# Базовые функции визуализации
# ==================================================================================================

def draw_face(ax,
              faces_coord=[],
              colors=[],
              default_color="blue",
              edge_enable=False,
              alpha=0.3,
              draw_aabb=False):
    """
    Рисует треугольные грани на 3D графике.
    
    Базовая функция для отрисовки треугольных граней. Использует Poly3DCollection
    для эффективной отрисовки множества граней одновременно.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Ось 3D графика matplotlib для отрисовки.
    faces_coord : List[np.ndarray], optional
        Список массивов координат вершин граней. Каждый массив имеет форму (3, 3),
        где первая размерность - 3 вершины, вторая - 3 координаты (x, y, z).
        По умолчанию пустой список.
    colors : List[np.ndarray], optional
        Список цветов для каждой грани. Каждый элемент - массив RGB значений [r, g, b]
        в диапазоне [0, 1]. Если список пуст, используется default_color.
        По умолчанию пустой список.
    default_color : str, optional
        Цвет по умолчанию для граней, если colors не указан (по умолчанию "blue").
    edge_enable : bool, optional
        Включить отрисовку рёбер граней чёрным цветом (по умолчанию False).
    alpha : float, optional
        Прозрачность граней в диапазоне [0, 1] (по умолчанию 0.3).
    draw_aabb : bool, optional
        Включить отрисовку AABB (Axis-Aligned Bounding Box) для каждой грани красным цветом
        (по умолчанию False).
    
    Notes
    -----
    Функция использует Poly3DCollection для эффективной отрисовки множества граней.
    AABB рисуется как параллелепипед с 12 рёбрами, если draw_aabb=True.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
    >>> draw_face(ax, faces_coord=[coords], colors=[[1, 0, 0]], alpha=0.5)
    """
    # Рисуем сами грани через Poly3DCollection для эффективной отрисовки
    poly_collection = Poly3DCollection(
        faces_coord,
        alpha=alpha,
        facecolors=colors if colors else default_color,
        edgecolors="black" if edge_enable else "none",
        linewidths=0.3 if edge_enable else 0.0
    )
    ax.add_collection3d(poly_collection)

    # Рисуем AABB для каждой грани, если включено
    if draw_aabb:
        for coords in faces_coord:
            # coords: (3,3) np.array - координаты 3 вершин треугольника
            min_corner = coords.min(axis=0)  # Минимальные координаты по каждой оси
            max_corner = coords.max(axis=0)  # Максимальные координаты по каждой оси

            # 8 вершин параллелепипеда AABB
            corners = np.array([
                [min_corner[0], min_corner[1], min_corner[2]],  # 0: min, min, min
                [max_corner[0], min_corner[1], min_corner[2]],  # 1: max, min, min
                [max_corner[0], max_corner[1], min_corner[2]],  # 2: max, max, min
                [min_corner[0], max_corner[1], min_corner[2]],  # 3: min, max, min
                [min_corner[0], min_corner[1], max_corner[2]],  # 4: min, min, max
                [max_corner[0], min_corner[1], max_corner[2]],  # 5: max, min, max
                [max_corner[0], max_corner[1], max_corner[2]],  # 6: max, max, max
                [min_corner[0], max_corner[1], max_corner[2]],  # 7: min, max, max
            ])

            # Рёбра бокса: пары индексов вершин для 12 рёбер параллелепипеда
            edges = [
                [0,1],[1,2],[2,3],[3,0],  # нижняя грань (z = min)
                [4,5],[5,6],[6,7],[7,4],  # верхняя грань (z = max)
                [0,4],[1,5],[2,6],[3,7]   # вертикальные рёбра
            ]

            # Рисуем рёбра AABB красным цветом
            for e in edges:
                ax.plot(*zip(corners[e[0]], corners[e[1]]), color="red", linewidth=0.5)


def draw_intersection_seg(ax,
            segment_coord=[],
            color="red",
            linewidths=0.3,
            alpha=0.3):
    """
    Рисует сегменты пересечения на 3D графике.
    
    Базовая функция для отрисовки отрезков пересечения граней. Каждый сегмент
    представляет собой отрезок между двумя точками в 3D пространстве.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Ось 3D графика matplotlib для отрисовки.
    segment_coord : List[List[Union[np.ndarray, Node]]], optional
        Список сегментов пересечения. Каждый сегмент - список из 2 элементов:
        [point1, point2], где каждая точка может быть np.ndarray или объектом Node.
        По умолчанию пустой список.
    color : str, optional
        Цвет линий сегментов (по умолчанию "red").
    linewidths : float, optional
        Толщина линий сегментов (по умолчанию 0.3).
    alpha : float, optional
        Прозрачность линий в диапазоне [0, 1] (по умолчанию 0.3).
    
    Notes
    -----
    Функция автоматически преобразует объекты Node в np.ndarray, извлекая координаты
    через атрибут .p. Если segment_coord пуст или None, функция ничего не рисует.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> seg1 = [np.array([0, 0, 0]), np.array([1, 1, 1])]
    >>> seg2 = [node1, node2]  # Node объекты
    >>> draw_intersection_seg(ax, segment_coord=[seg1, seg2], color="red", linewidths=1.0)
    """
    if segment_coord is None or len(segment_coord) == 0:
        return

    # Отрисовываем каждый сегмент пересечения
    for seg in segment_coord:
        p1, p2 = seg

        # Преобразуем точки в numpy массивы
        # Если точка - объект Node, извлекаем координаты через атрибут .p
        p1 = np.asarray(p1.p if hasattr(p1, 'p') else p1)
        p2 = np.asarray(p2.p if hasattr(p2, 'p') else p2)

        # Координаты для plot() - списки x, y, z координат
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        zs = [p1[2], p2[2]]

        # Рисуем отрезок на графике
        ax.plot(xs, ys, zs,
                color=color,
                linewidth=linewidths,
                alpha=alpha)


# ==================================================================================================
# Функции среднего уровня
# ==================================================================================================

def face_with_intersection_segment_plotter(ax,
                                          face: Face,
                                          intersection_segments: List[List[Node]],
                                          edge_enable: bool = False,
                                          draw_aabb: bool = False,
                                          alpha: float = 0.3,
                                          segment_color: str = "red",
                                          segment_linewidth: float = 1.2):
    """
    Рисует одну грань с её сегментами пересечения на заданной оси.
    
    Функция среднего уровня, которая объединяет отрисовку грани и её сегментов
    пересечения. Используется как строительный блок для более высокоуровневых
    функций визуализации.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Ось 3D графика matplotlib для отрисовки.
    face : Face
        Грань для отрисовки.
    intersection_segments : List[List[Node]]
        Список сегментов пересечения для данной грани. Каждый сегмент - список
        из 2 объектов Node, представляющих концы отрезка пересечения.
    edge_enable : bool, optional
        Включить отрисовку рёбер грани (по умолчанию False).
    draw_aabb : bool, optional
        Включить отрисовку AABB для грани (по умолчанию False).
    alpha : float, optional
        Прозрачность грани в диапазоне [0, 1] (по умолчанию 0.3).
    segment_color : str, optional
        Цвет сегментов пересечения (по умолчанию "red").
    segment_linewidth : float, optional
        Толщина линий сегментов пересечения (по умолчанию 1.2).
    
    Notes
    -----
    Функция использует draw_face() для отрисовки грани и draw_intersection_seg()
    для отрисовки сегментов пересечения. Сегменты рисуются с полной непрозрачностью
    (alpha=1.0) для лучшей видимости.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> face = mesh.find_face_by_id(123)
    >>> segments = [[node1, node2], [node3, node4]]
    >>> face_with_intersection_segment_plotter(ax, face, segments, edge_enable=True)
    """
    # Получаем координаты вершин грани
    coords = np.array([node.p for node in face.nodes])
    
    # Генерируем случайный цвет для грани
    colors = np.random.rand(1, 3)
    
    # Рисуем грань через базовую функцию
    draw_face(
        ax=ax,
        faces_coord=[coords],
        colors=list(colors),
        alpha=alpha,
        edge_enable=edge_enable,
        draw_aabb=draw_aabb
    )
    
    # Рисуем сегменты пересечения, если они есть
    if intersection_segments:
        # Преобразуем сегменты в формат для draw_intersection_seg
        # intersection_segments уже содержит объекты Node, поэтому просто передаём их
        segment_coord = []
        for segment in intersection_segments:
            # segment - это список из 2 объектов Node
            segment_coord.append(segment)
        
        # Рисуем сегменты через базовую функцию
        draw_intersection_seg(
            ax=ax,
            segment_coord=segment_coord,
            color=segment_color,
            linewidths=segment_linewidth,
            alpha=1.0  # Полная непрозрачность для лучшей видимости
        )


# ==================================================================================================
# Функции высокого уровня
# ==================================================================================================

def mesh_plotter(mesh: Mesh,
                 faces_enable: bool = True,
                 draw_aabb: bool = False, 
                 edge_enable: bool = False,
                 faces_to_fix: Optional[Dict[int, Tuple[Face, List[List[Node]]]]] = None,
                 faces_to_fix_enable: bool = False,
                 intersection_segments_enable: bool = True,
                 intersection_linewidth: float = 1.2,
                 alpha: float = 0.3):
    """
    Визуализирует всю сетку с возможностью выделения граней с пересечениями.
    
    Основная функция для визуализации треугольной сетки. Может отображать все грани сетки,
    выделять грани с пересечениями и рисовать сегменты пересечения.
    
    Parameters
    ----------
    mesh : Mesh
        Объект сетки для визуализации.
    faces_enable : bool, optional
        Включить отрисовку всех граней сетки (по умолчанию True).
    draw_aabb : bool, optional
        Включить отрисовку AABB для всех граней (по умолчанию False).
    edge_enable : bool, optional
        Включить отрисовку рёбер всех граней (по умолчанию False).
    faces_to_fix : Dict[int, Tuple[Face, List[List[Node]]]], optional
        Словарь граней с пересечениями из BVHTree.faces_to_fix:
        - Ключ: int - идентификатор грани (face.glo_id)
        - Значение: Tuple[Face, List[List[Node]]] - грань и список сегментов пересечения
        По умолчанию None.
    faces_to_fix_enable : bool, optional
        Включить выделение граней с пересечениями (по умолчанию False).
        Если True, грани из faces_to_fix будут отрисованы отдельно.
    intersection_segments_enable : bool, optional
        Включить отрисовку сегментов пересечения для граней из faces_to_fix
        (по умолчанию True). Работает только если faces_to_fix_enable=True.
    intersection_linewidth : float, optional
        Толщина линий сегментов пересечения (по умолчанию 1.2).
    alpha : float, optional
        Прозрачность граней в диапазоне [0, 1] (по умолчанию 0.3).
    
    Notes
    -----
    Функция использует иерархию вызовов:
    1. Для всех граней сетки: draw_face()
    2. Для граней с пересечениями: face_with_intersection_segment_plotter()
       (которая внутри использует draw_face() и draw_intersection_seg())
    
    Грани с пересечениями отрисовываются поверх всех остальных граней для лучшей видимости.
    Цвета граней определяются по зонам (zone.name) сетки.
    
    Examples
    --------
    >>> bvh = BVHTree(mesh, faces_in_node=1)
    >>> bvh.prepare_mesh(esc_enable=False)
    >>> bvh.build_tree(split_func="sah")
    >>> faces_to_fix = bvh.traversal_tree()
    >>> mesh_plotter(mesh, faces_enable=True, faces_to_fix=faces_to_fix, 
    ...              faces_to_fix_enable=True, intersection_segments_enable=True)
    """
    # Создаём фигуру и ось для 3D визуализации
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Создаём карту цветов по зонам сетки
    # Каждая зона получает случайный цвет для визуального различия
    zones = list({face.zone.name for face in mesh.faces})
    colors = np.random.rand(len(zones), 3)
    color_map = {z: c for z, c in zip(zones, colors)}

    # --- Этап 1: Отрисовка всех граней сетки (если включено) ---
    if faces_enable:
        polys = []
        face_colors = []
        
        for face in mesh.faces:
            coords = np.array([node.p for node in face.nodes])
            polys.append(coords)
            face_colors.append(color_map.get(face.zone.name))
        
        # Рисуем все грани одним вызовом через базовую функцию
        draw_face(
            ax=ax,
            faces_coord=polys,
            colors=face_colors,
            alpha=alpha,
            edge_enable=edge_enable,
            draw_aabb=draw_aabb
        )
    
    # --- Этап 2: Отрисовка граней с пересечениями (если включено) ---
    if faces_to_fix_enable and faces_to_fix:
        # Отрисовываем каждую грань с пересечениями через функцию среднего уровня
        for face_id, (face, intersection_segments) in faces_to_fix.items():
            # Используем face_with_intersection_segment_plotter для отрисовки
            # грани и её сегментов пересечения
            face_with_intersection_segment_plotter(
                ax=ax,
                face=face,
                intersection_segments=intersection_segments if intersection_segments_enable else [],
                edge_enable=edge_enable if not faces_enable else False,  # Рёбра только если не рисуем все грани
                draw_aabb=draw_aabb,
                alpha=alpha,
                segment_color="red",
                segment_linewidth=intersection_linewidth
            )
    
    # Настройка осей и заголовка
    ax.set_title(mesh.title if mesh.title else "Mesh Visualization")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()


def pairs_broken_face_plotter(face_pairs: Dict[int, Tuple[Face, Face]],
                              faces_to_fix: Optional[Dict[int, Tuple[Face, List[List[Node]]]]] = None,
                              draw_intersection: bool = False,
                              edge_enable: bool = False,
                              stop_draw: int = 0,
                              draw_aabb: bool = False,
                              alpha: float = 0.3,
                              intersection_linewidth: float = 1.2):
    """
    Визуализирует пары граней-кандидатов на пересечение.
    
    Функция отрисовывает пары граней из candidate_pairs или candidate_pairs_after_czech
    для анализа результатов работы BVH алгоритма. Может дополнительно отображать
    сегменты пересечения, если они найдены в faces_to_fix.
    
    Parameters
    ----------
    face_pairs : Dict[int, Tuple[Face, Face]]
        Словарь пар граней для визуализации:
        - Ключ: int - индекс пары
        - Значение: Tuple[Face, Face] - пара граней (face_a, face_b)
        Обычно это bvh.candidate_pairs или bvh.candidate_pairs_after_czech.
    faces_to_fix : Dict[int, Tuple[Face, List[List[Node]]]], optional
        Словарь граней с пересечениями из BVHTree.faces_to_fix:
        - Ключ: int - идентификатор грани (face.glo_id)
        - Значение: Tuple[Face, List[List[Node]]] - грань и список сегментов пересечения
        Используется для отрисовки сегментов пересечения, если draw_intersection=True.
        По умолчанию None.
    draw_intersection : bool, optional
        Включить отрисовку сегментов пересечения для пар граней (по умолчанию False).
        Если True, функция ищет сегменты пересечения в faces_to_fix для обеих граней пары.
    edge_enable : bool, optional
        Включить отрисовку рёбер граней (по умолчанию False).
    stop_draw : int, optional
        Остановить отрисовку после N пар. 0 означает отрисовать все пары (по умолчанию 0).
    draw_aabb : bool, optional
        Включить отрисовку AABB для граней (по умолчанию False).
    alpha : float, optional
        Прозрачность граней в диапазоне [0, 1] (по умолчанию 0.3).
    intersection_linewidth : float, optional
        Толщина линий сегментов пересечения (по умолчанию 1.2).
    
    Notes
    -----
    Функция использует базовые функции draw_face() и draw_intersection_seg() для отрисовки.
    Каждая пара граней отрисовывается на отдельном графике для детального анализа.
    Если draw_intersection=True, функция ищет сегменты пересечения в faces_to_fix
    для обеих граней пары и объединяет их для отрисовки.
    
    Examples
    --------
    >>> bvh = BVHTree(mesh, faces_in_node=1)
    >>> bvh.prepare_mesh(esc_enable=False)
    >>> bvh.build_tree(split_func="sah")
    >>> faces_to_fix = bvh.traversal_tree()
    >>> # Визуализация пар с пересечениями
    >>> pairs_broken_face_plotter(bvh.candidate_pairs_after_czech, 
    ...                           faces_to_fix=faces_to_fix,
    ...                           draw_intersection=True,
    ...                           stop_draw=5)
    """
    num = 0
    for key, (face_a, face_b) in face_pairs.items():
        # Создаём отдельную фигуру для каждой пары граней
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Получаем координаты вершин обеих граней
        coords1 = np.array([node.p for node in face_a.nodes])
        coords2 = np.array([node.p for node in face_b.nodes])
        
        # Генерируем случайные цвета для каждой грани
        colors = np.random.rand(2, 3)
        
        # Рисуем обе грани через базовую функцию
        draw_face(
            ax=ax,
            faces_coord=[coords1, coords2],
            colors=list(colors),
            alpha=alpha,
            edge_enable=edge_enable,
            draw_aabb=draw_aabb
        )
        
        # Отрисовка сегментов пересечения, если включено и faces_to_fix предоставлен
        if draw_intersection and faces_to_fix:
            # Собираем все сегменты пересечения для обеих граней пары
            all_segments = []
            
            # Ищем сегменты для face_a
            if face_a.glo_id in faces_to_fix:
                _, segments_a = faces_to_fix[face_a.glo_id]
                all_segments.extend(segments_a)
            
            # Ищем сегменты для face_b
            if face_b.glo_id in faces_to_fix:
                _, segments_b = faces_to_fix[face_b.glo_id]
                all_segments.extend(segments_b)
            
            # Рисуем все найденные сегменты пересечения
            if all_segments:
                draw_intersection_seg(
                    ax=ax,
                    segment_coord=all_segments,
                    color="red",
                    linewidths=intersection_linewidth,
                    alpha=1.0
                )
        
        # Настройка осей и заголовка
        ax.set_title(f"Pair num: {num} (faces {face_a.glo_id} and {face_b.glo_id})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()
        
        # Проверка условия остановки
        if stop_draw > 0 and num + 1 >= stop_draw:
            break
        num += 1


def visualize_bvh_tree_graph(graph):
    """
    Визуализирует структуру BVH дерева в виде графа.
    
    Функция создаёт графическое представление структуры BVH дерева, где узлы
    представляют узлы дерева, а рёбра - связи между родительскими и дочерними узлами.
    
    Parameters
    ----------
    graph : networkx.Graph
        Граф структуры BVH дерева, созданный через BVHTree.build_graph().
    
    Notes
    -----
    Функция пытается использовать graphviz для лучшей визуализации структуры дерева.
    Если graphviz недоступен, используется spring_layout из networkx.
    """
    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    except:
        pos = nx.spring_layout(graph)

    labels = nx.get_node_attributes(graph, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=350, node_color="lightblue", arrows=False, font_size=6)
    plt.title("BVH Tree Structure", fontsize=12)
    plt.show()


if __name__ == '__main__':
    pass
