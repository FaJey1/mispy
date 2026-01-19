import logging
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Literal
from collections import defaultdict
import numpy as np
import networkx as nx
from tabulate import tabulate

from mispy.extract_mesh import *
from mispy.visualization_mesh import *
from .czech_classify import CzechClassify
from .bvh_tree import BVHTree, _is_neighbour

# Настройка логирования для всех модулей
# Устанавливаем уровень DEBUG для root logger, чтобы собирать все логи
# Handlers будут фильтровать: консоль - INFO, файл - DEBUG
# Создаём formatter для логов
formatter = logging.Formatter("%(levelname)s: %(message)s")

# Настройка логирования в консоль (только INFO и выше)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Настройка логирования в файл (DEBUG и выше)
file_handler = logging.FileHandler("log.txt", mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Настраиваем root logger на DEBUG, чтобы все логи попадали в handlers
# Handlers сами отфильтруют по своим уровням
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()  # Очищаем существующие handlers
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Явно устанавливаем уровни для дочерних logger'ов модулей,
# чтобы они наследовали настройки от root logger
logging.getLogger("mispy.transform_mesh.bvh_tree").setLevel(logging.DEBUG)
logging.getLogger("mispy.transform_mesh.czech_classify").setLevel(logging.DEBUG)  
    
if __name__ == '__main__':
    mesh = Mesh("examples/small_sphere_double.dat")
    mesh = Mesh("examples/sphere_double.dat")
    #mesh = Mesh("examples/bunny_double.dat")
    
    times = {}
    bvh = BVHTree(mesh, faces_in_node=1)
    _, times["prepare"] = measure_time(bvh.prepare_mesh, esc_enable=False)
    _, times["build"] = measure_time(bvh.build_tree, split_func="vah")
    _, times["traversal"] = measure_time(bvh.traversal_tree)
    
    graph = bvh.build_graph()
    #visualize_bvh_tree_graph(graph)
    table = [(name, f"{t:.6f} сек") for name, t in times.items()]
    logging.info("=== Результаты BVH алгоритма ===\n%s\n%s",
                 tabulate(table, headers=["Параметр", "Значение"], tablefmt="grid"),
                 tabulate(statistic_bvh_tree_graph(graph), headers=["Параметр", "Значение"], tablefmt="grid"),
                )
    
    # Визуализация всей сетки с гранями, имеющими пересечения, и их сегментами
    mesh_plotter(
        mesh=mesh,
        faces_enable=False,  # Показать все грани сетки
        draw_aabb=False,
        edge_enable=False,
        faces_to_fix=bvh.faces_to_fix,  # Грани с пересечениями
        faces_to_fix_enable=True,  # Выделить грани с пересечениями
        intersection_segments_enable=True,  # Показать сегменты пересечения
        intersection_linewidth=1.2,
        alpha=0.3
    )
    
    # Визуализация пар граней-кандидатов с пересечениями
    # pairs_broken_face_plotter(
    #     face_pairs=bvh.candidate_pairs_after_czech,
    #     faces_to_fix=bvh.faces_to_fix,
    #     draw_intersection=True,  # Показать сегменты пересечения
    #     edge_enable=False,
    #     draw_aabb=False,
    #     stop_draw=5  # Показать только первые 5 пар
    # )
    
    # Пример пары с невалидной классификацией
    # test = {485: (mesh.find_face_by_id(7232), mesh.find_face_by_id(7360))}
    # pairs_broken_face_plotter(test, edge_enable=False, draw_aabb=True, stop_draw=0)
    # cz = CzechClassify(candidates=test[485])
    # result = cz.get_intersection()
    # has_intersection, intersection_segment, impossible_couples = result
    # print(f"has_intersection: {has_intersection}")
    # print(f"intersection_segment: {intersection_segment}")
    # print(f"impossible_couples count: {len(impossible_couples)}")
    # coords1 = np.array([node.glo_id for node in bvh.candidate_pairs[485][0].nodes])
    # coords2 = np.array([node.glo_id for node in bvh.candidate_pairs[485][1].nodes])
    # print(coords1, coords2)
    # if impossible_couples:
    #     for i, (face_a, face_b) in enumerate(impossible_couples):
    #         print(f"  [{i}] Face {face_a.glo_id} <-> Face {face_b.glo_id}")
    # pairs_broken_face_plotter(test, edge_enable=False, draw_aabb=True, stop_draw=0)
