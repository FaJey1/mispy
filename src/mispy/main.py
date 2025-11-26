import logging
import time
from yaspin import yaspin
from yaspin.spinners import Spinners

from mispy.extract_mesh import *
from mispy.transform_mesh import *
from mispy.visualization_mesh import *


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def measure_time(func, *args, **kwargs):
    """
    Измеряет время выполнения функции.
    
    Параметры:
        func: callable — функция для вызова
        *args, **kwargs — аргументы для функции
    
    Возвращает:
        tuple(result, elapsed_time)
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def alg(mesh: Mesh, split_func: str = "sah", esc_enable: bool = False, draw_aabb: bool = False, edge_enable: bool = False, faces_enable: bool = True, faces_to_fix_enable: bool = False):
    times = {}
    bvh = BVHTree(mesh)

    # --- prepare_mesh ---
    with yaspin(Spinners.arc, text="Подготовка mesh...") as sp:
        _, times["prepare_time"] = measure_time(bvh.prepare_mesh, esc_enable=False)
        sp.ok("DONE")

    # --- build_tree ---
    with yaspin(Spinners.arc, text=f"Построение BVH (split={split_func})...") as sp:
        _, times["build_time"] = measure_time(bvh.build_tree, split_func=split_func)
        sp.ok("DONE")

    # --- traversal ---
    with yaspin(Spinners.arc, text="Трассировка BVH...") as sp:
        candidate_pairs, times["traversal_time"] = measure_time(bvh.traversal_tree)
        faces_to_fix = bvh.faces_to_fix
        sp.ok("DONE")

    # --- Вывод результатов ---
    logging.info("=== Результаты BVH алгоритма ===")
    logging.info("Split function: %s", split_func)
    logging.info("Finded faces to fix %s", len(faces_to_fix))

    for name, t in times.items():
        logging.info(f"{name:<20} {t:>10.6f} сек")

    # построение графа (без спиннера)
    graph = bvh.build_graph(bvh.root_node)
    visualize_bvh_tree_graph(graph)
    #mesh_plotter(mesh=mesh, faces_enable=faces_enable, draw_aabb=draw_aabb, edge_enable=edge_enable, faces_to_fix = faces_to_fix, faces_to_fix_enable=faces_to_fix_enable)
    

def main():
    mesh = Mesh("tests/examples/small_sphere_double.dat")
    #mesh = Mesh("tests/examples/sphere_double.dat")
    #mesh = Mesh("tests/examples/bunny_double.dat")
    #mesh = Mesh("tests/examples/air_inlet_010000000000.dat")
    alg(mesh = mesh, faces_enable = True, draw_aabb = False, esc_enable = False, edge_enable = True, faces_to_fix_enable = True, split_func = "vah")
    #pairs_plotter([(mesh.find_face_by_id(4838), mesh.find_face_by_id(4841))])


if __name__ == '__main__':
    main()
