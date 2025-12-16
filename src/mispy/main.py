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


def alg(mesh: Mesh, split_func: str = "sah", esc_enable: bool = False, draw_aabb: bool = False, edge_enable: bool = False, faces_enable: bool = True, faces_to_fix_enable: bool = False, leaf_in_node = 1):
    times = {}
    bvh = BVHTree(mesh, leaf_in_node=1)
    # --- prepare_mesh ---
    with yaspin(Spinners.arc, text="Подготовка сетки...") as sp:
        _, times["Время подготовки сетки"] = measure_time(bvh.prepare_mesh, esc_enable=False)
        sp.ok("DONE")

    # --- build_tree ---
    with yaspin(Spinners.arc, text=f"Построение BVH (split={split_func})...") as sp:
        _, times["Время построения дерева"] = measure_time(bvh.build_tree, split_func=split_func)
        sp.ok("DONE")

    # --- traversal ---
    with yaspin(Spinners.arc, text="Трассировка BVH...") as sp:
        candidate_pairs, times["Время обхода дерева"] = measure_time(bvh.traversal_tree)
        faces_to_fix = bvh.faces_to_fix
        sp.ok("DONE")
        
    # --- triangulation ---
    # with yaspin(Spinners.arc, text="Триангуляция 'сломанных' ячеек...") as sp:
    #     times["Время триангуляции"] = measure_time()
    #     sp.ok("DONE")

    # --- build graph
    graph = bvh.build_graph(bvh.root_node)
    
    # --- Вывод результатов ---
    table = [
        ["Функция разбиения", split_func],
        ["Количество ячеек в листе", leaf_in_node],
        ["Использование раннего разбиения", esc_enable],
        ["Найдено пар ячеек для коррекции", len(faces_to_fix)],
    ]
    for name, t in times.items():
        table.append([name, f"{t:.6f} сек"])

    logging.info("=== Результаты BVH алгоритма ===\n%s\n%s\n%s", 
                 tabulate(
                    table,
                    headers=["Параметр", "Значение"],
                    tablefmt="grid"
                ),
                statistic_bvh_tree_graph(graph),
                statistic_mesh(mesh))
    
    #mesh_plotter(mesh=mesh, faces_enable=faces_enable, draw_aabb=draw_aabb, edge_enable=edge_enable, faces_to_fix = faces_to_fix, faces_to_fix_enable=faces_to_fix_enable)
    #fix_face_plotter(mesh=mesh, faces_to_fix=faces_to_fix, stop_draw=1)
    
    #pairs_broken_face_plotter(candidate_pairs, stop_draw=1)
    #pairs_plotter([(mesh.find_face_by_id(4838), mesh.find_face_by_id(4841))])
    

def main():
    mesh1 = Mesh("tests/examples/small_sphere_double.dat")
    mesh2 = Mesh("tests/examples/sphere_double.dat")
    mesh3 = Mesh("tests/examples/bunny_double.dat")
    #mesh = Mesh("tests/examples/air_inlet_010000000000.dat")
    tests = {
        1: [
            mesh1, False, "vah", 5],
        2: [
            mesh2, False, "vah", 5],
        3: [
            mesh3, False, "vah", 5],
        4: [
            mesh1, False, "vah", 1],
        5: [
            mesh2, False, "vah", 1],
        6: [
            mesh3, False, "vah", 1],
        7: [
            mesh1, False, "sah", 5],
        8: [
            mesh2, False, "sah", 5],
        9: [
            mesh3, False, "sah", 5],
        10: [
            mesh1, False, "sah", 1],
        11: [
            mesh2, False, "sah", 1],
        12: [
            mesh3, False, "sah", 1],
        13: [
            mesh1, True, "vah", 5],
        14: [
            mesh2, True, "vah", 5],
        15: [
            mesh3, True, "vah", 5],
        16: [
            mesh1, True, "vah", 1],
        17: [
            mesh2, True, "vah", 1],
        18: [
            mesh3, True, "vah", 1],
        19: [
            mesh1, True, "sah", 5],
        20: [
            mesh2, True, "sah", 5],
        21: [
            mesh3, True, "sah", 5],
        22: [
            mesh1, True, "sah", 1],
        23: [
            mesh2, True, "sah", 1],
        24: [
            mesh3, True, "sah", 1],
    }
    for key in tests:
        logging.info("=== ТЕСТ %s, СЕТКА %s ===", key, tests[key][0].title)
        alg(mesh = tests[key][0], faces_enable = True, draw_aabb = False, esc_enable = tests[key][1], edge_enable = True, faces_to_fix_enable = True, split_func = tests[key][2], leaf_in_node = tests[key][3] )
    #alg(mesh = mesh1, faces_enable = True, draw_aabb = False, esc_enable = False, edge_enable = True, faces_to_fix_enable = True, split_func = "vah", leaf_in_node = 5 )


if __name__ == '__main__':
    main()
