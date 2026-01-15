import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Literal
from collections import defaultdict

import numpy as np
import networkx as nx
from tabulate import tabulate

from mispy.extract_mesh import *
from mispy.visualization_mesh import *
from .czech_classify import CzechClassify
from .bvh_tree import BVHTree

# Настройка логирования для всех модулей
# Устанавливаем уровень DEBUG, чтобы видеть DEBUG логи из bvh_tree.py
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")    
    
if __name__ == '__main__':
    mesh = Mesh("examples/small_sphere_double.dat")
    #mesh = Mesh("examples/sphere_double.dat")
    #mesh = Mesh("examples/bunny_double.dat")
    
    times = {}
    bvh = BVHTree(mesh, faces_in_node=1)
    _, times["prepare"] = measure_time(bvh.prepare_mesh, esc_enable=False)
    _, times["build"] = measure_time(bvh.build_tree, split_func="vah")
    _, times["traversal"] = measure_time(bvh.traversal_tree)
    
    graph = bvh.build_graph()
    #visualize_bvh_tree_graph(graph)
    table = [(name, f"{t:.6f} сек") for name, t in times.items()]
    logging.info("=== Результаты BVH алгоритма ===\n%s",
                 tabulate(table, headers=["Параметр", "Значение"], tablefmt="grid"))
    logging.info("=== Результаты BVH алгоритма ===\n%s",
                 tabulate(statistic_bvh_tree_graph(graph), headers=["Параметр", "Значение"], tablefmt="grid"))
    
    #candidate_pairs = bvh.traversal_tree() 
    #graph = bvh.build_graph(bvh.root_node)
    #mesh_plotter(mesh, faces_enable=False, draw_aabb=False, edge_enable=True, faces_to_fix=bvh.faces_to_fix, faces_to_fix_enable=True)
    #face.glo_id in [149, 75, 149, 66]:
    #[7349, 7481]
    #[4838 4841]
    #cs = CzechClassify((mesh.find_face_by_id(4838), mesh.find_face_by_id(4841)))
    #result, points = cs.classify()
    #print(result)
