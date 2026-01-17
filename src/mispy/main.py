import logging
import time

from tabulate import tabulate

from mispy.extract_mesh import *
from mispy.transform_mesh import *
from mispy.visualization_mesh.statistics import (
    statistic_bvh_tree_graph,
    statistic_mesh,
    save_results,
    visualization_results,
    measure_time,
    
)


#logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def alg(mesh: Mesh, test_id: int, split_func: str = "sah",
        esc_enable: bool = False, faces_in_node: int = 1):
    """Запускает BVH алгоритм и возвращает результаты."""
    times = {}
    bvh = BVHTree(mesh, faces_in_node=faces_in_node)

    _, times["prepare"] = measure_time(bvh.prepare_mesh, esc_enable=esc_enable)
    _, times["build"] = measure_time(bvh.build_tree, split_func=split_func)
    _, times["traversal"] = measure_time(bvh.traversal_tree)

    graph = bvh.build_graph(bvh.root_node)
    bvh_table, bvh_stats = statistic_bvh_tree_graph(graph)
    mesh_table, mesh_stats = statistic_mesh(mesh)

    table = [
        ["Функция разбиения", split_func],
        ["Количество ячеек в листе", faces_in_node],
        ["Использование раннего разбиения", esc_enable],
        ["Найдено пар ячеек для коррекции", len(bvh.faces_to_fix)],
    ]
    for name, t in times.items():
        table.append([name, f"{t:.6f} сек"])

    logging.info("=== Результаты BVH алгоритма ===\n%s\n%s\n%s",
                 tabulate(table, headers=["Параметр", "Значение"], tablefmt="grid"),
                 bvh_table, mesh_table)

    total_time = times["prepare"] + times["build"] + times["traversal"]

    return {
        "test_id": test_id,
        "mesh": mesh.title,
        "faces": mesh_stats["faces"],
        "edges": mesh_stats["edges"],
        "nodes": mesh_stats["nodes"],
        "prepare_time": times["prepare"],
        "build_time": times["build"],
        "traversal_time": times["traversal"],
        "total_time": total_time,
        "split_func": split_func.upper(),
        "esc": esc_enable,
        "faces_in_node": faces_in_node,
        "pairs_to_fix": len(bvh.faces_to_fix),
        "bvh_vertices": bvh_stats["bvh_nodes"],
        "bvh_edges": bvh_stats["bvh_edges"],
        "bvh_depth": bvh_stats["bvh_depth"],
        "bvh_balance": bvh_stats["bvh_balance"],
    }


def main():
    meshes = {
        "small": Mesh("examples/small_sphere_double.dat"),
        "sphere": Mesh("examples/sphere_double.dat"),
        "bunny": Mesh("examples/bunny_double.dat"),
    }

    tests = {
        # --- small ---
        1: ("small", False, "vah", 5),

        # --- sphere ---
        9:  ("sphere", False, "vah", 5),

        # --- bunny ---
        # 17: ("bunny", False, "vah", 5),
    }

    results = []
    for test_id, (mesh_key, esc, split, leaf) in tests.items():
        mesh = meshes[mesh_key]
        logging.info("=== ТЕСТ %d, СЕТКА %s ===", test_id, mesh.title)
        results.append(alg(mesh, test_id, split, esc, leaf))

    save_results(results)
    #visualization_results(results)


if __name__ == '__main__':
    main()
