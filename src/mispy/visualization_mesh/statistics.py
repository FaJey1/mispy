import logging
import os
import csv
import time

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Используем getLogger вместо basicConfig, чтобы не конфликтовать с настройками из других модулей
logger = logging.getLogger(__name__)


def measure_time(func, *args, **kwargs):
    """Измеряет время выполнения функции."""
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start

def statistic_bvh_tree_graph(graph):
    """Статистика BVH-дерева по графу."""
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    roots = [n for n, d in graph.in_degree() if d == 0]
    root = roots[0] if roots else list(graph.nodes)[0]

    leaves = [n for n, attr in graph.nodes(data=True) if attr.get('is_leaf', False)]
    leaf_depths = [nx.shortest_path_length(graph, root, leaf) for leaf in leaves]

    depth = max(leaf_depths) if leaf_depths else 0
    balance = np.std(leaf_depths) if leaf_depths else 0

    stats = [
        ["Количество вершин", num_nodes],
        ["Количество рёбер", num_edges],
        ["Глубина дерева", depth],
        ["Сбалансированность (std глубин листьев)", f"{balance:.2f}"]
    ]
    
    return stats

def statistic_mesh(mesh):
    """Статистика сетки."""
    table = [
        ["zones", len(mesh.zones)],
        ["nodes", len(mesh.nodes)],
        ["edges", len(mesh.edges)],
        ["faces", len(mesh.faces)]
    ]
    
    stats = {
        "zones": len(mesh.zones),
        "nodes": len(mesh.nodes),
        "edges": len(mesh.edges),
        "faces": len(mesh.faces)
    }
    
    return table, stats

def _build_table_summary(results):
    """Строит сводную таблицу результатов."""
    return [
        {
            "Тест": r["test_id"],
            "Сетка": r["mesh"],
            "Общее время, сек": r["total_time"],
            "Подготовка, сек": r["prepare_time"],
            "Построение, сек": r["build_time"],
            "Обход, сек": r["traversal_time"],
            "Ячеек": r["faces"],
            "Рёбер": r["edges"],
            "Вершин": r["nodes"],
            "Пар для коррекции": r["pairs_to_fix"],
            "ESC": r["esc"],
            "Функция разбиения": r["split_func"],
            "Ячеек в листе": r["faces_in_node"],
        }
        for r in results
    ]

def _save_csv(filename, rows):
    """Сохраняет таблицу в CSV."""
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def save_results(results):
    """Сохраняет результаты тестов в CSV."""
    _save_csv("table_summary.csv", _build_table_summary(results))
    logging.info("Результаты сохранены в results/table_summary.csv")

def visualization_results(results):
    """Визуализирует результаты тестов."""
    df = pd.DataFrame(_build_table_summary(results))

    colors = {
        "prepare": "#FFA94D",
        "build": "#5B8DB8",
        "traversal": "#7C6AB9",
    }

    for grid in df["Сетка"].unique():
        df_grid = df[df["Сетка"] == grid].sort_values("Общее время, сек")

        tests = [
            f"Тест {int(t)}\nESC:{e}\nSF:{sf}\nFoL:{fol}\nCF:{cf}"
            for t, e, sf, fol, cf in zip(
                df_grid["Тест"],
                df_grid["ESC"],
                df_grid["Функция разбиения"],
                df_grid["Ячеек в листе"],
                df_grid["Пар для коррекции"]
            )
        ]
        prepare = df_grid["Подготовка, сек"]
        build = df_grid["Построение, сек"]
        traversal = df_grid["Обход, сек"]
        total_time = df_grid["Общее время, сек"]

        plt.figure(figsize=(12, 6))
        plt.bar(tests, prepare, label="Время подготовки", color=colors["prepare"])
        plt.bar(tests, build, bottom=prepare, label="Время построения", color=colors["build"])
        plt.bar(tests, traversal, bottom=prepare + build, label="Время обхода", color=colors["traversal"])

        for i, t in enumerate(total_time):
            plt.hlines(y=t, xmin=i-0.4, xmax=i+0.4, colors='gray', linestyles='dashed', linewidth=1)
            plt.text(i, t + 0.001, f"{t:.4f}", ha='center', va='bottom', fontsize=8, color='gray')

        plt.ylabel("Время выполнения, сек")
        plt.xlabel("Тесты")
        plt.title(f"Время выполнения BVH по этапам для сетки '{grid}'")
        plt.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
