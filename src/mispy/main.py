import logging
import sys
import time

from mispy.extract_mesh import *
from mispy.transform_mesh import *
from mispy.visualization_mesh.statistics import (
    statistic_bvh_tree_graph,
    statistic_mesh,
    save_results,
    measure_time,
    _load_csv,
)
from mispy.visualization_mesh.visualization import visualization_results, visualization_results_percents


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


def alg(mesh: Mesh, test_id: int, split_func: str = "sah",
        esc_enable: bool = False, faces_in_node: int = 1):
    """Запускает BVH алгоритм и возвращает результаты."""
    from tabulate import tabulate
    
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
    
    # Вычисляем общее количество пар в impossible_couples
    # impossible_couples: Dict[int, List[Tuple[Face, Face]]]
    impossible_couples_count = sum(len(pairs) for pairs in bvh.impossible_couples.values())

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
        "candidate_pairs_count": len(bvh.candidate_pairs),
        "candidate_pairs_after_czech_count": len(bvh.candidate_pairs_after_czech),
        "impossible_couples_count": impossible_couples_count,
        "bvh_vertices": bvh_stats["bvh_nodes"],
        "bvh_edges": bvh_stats["bvh_edges"],
        "bvh_depth": bvh_stats["bvh_depth"],
        "bvh_balance": bvh_stats["bvh_balance"],
    }


def main():
    meshes = {
        "small_sphere_double": Mesh("examples/small_sphere_double.dat"),
        "sphere_double": Mesh("examples/sphere_double.dat"),
        "bunny_double": Mesh("examples/bunny_double.dat"),
    }

    tests = {
        # --- small ---
        1: ("small_sphere_double", False, "vah", 5),
        2: ("small_sphere_double", False, "vah", 1),
        3: ("small_sphere_double", False, "sah", 5),
        4: ("small_sphere_double", False, "sah", 1),
        5: ("small_sphere_double", True, "vah", 5),
        6: ("small_sphere_double", True, "vah", 1),
        7: ("small_sphere_double", True, "sah", 5),
        8: ("small_sphere_double", True, "sah", 1),

        # --- sphere ---
        9: ("sphere_double", False, "vah", 5),
        10: ("sphere_double", False, "vah", 1),
        11: ("sphere_double", False, "sah", 5),
        12: ("sphere_double", False, "sah", 1),
        13: ("sphere_double", True, "vah", 5),
        14: ("sphere_double", True, "vah", 1),
        15: ("sphere_double", True, "sah", 5),
        16: ("sphere_double", True, "sah", 1),

        # --- bunny ---
        17: ("bunny_double", False, "vah", 5),
        18: ("bunny_double", False, "vah", 1),
        19: ("bunny_double", False, "sah", 5),
        20: ("bunny_double", False, "sah", 1),
        21: ("bunny_double", True, "vah", 5),
        22: ("bunny_double", True, "vah", 1),
        23: ("bunny_double", True, "sah", 5),
        24: ("bunny_double", True, "sah", 1),
    }

    results = []
    for test_id, (mesh_key, esc, split, leaf) in tests.items():
        mesh = meshes[mesh_key]
        logging.info("=== ТЕСТ %d, СЕТКА %s ===", test_id, mesh.title)
        results.append(alg(mesh, test_id, split, esc, leaf))

    save_results(results)
    visualization_results(results)
    visualization_results(results)

def visualization_passed_tests_result():
    results = _load_csv("table_summary.csv")
    visualization_results(results)
    visualization_results_percents(results)


if __name__ == '__main__':
    #main()
    visualization_passed_tests_result()
    
