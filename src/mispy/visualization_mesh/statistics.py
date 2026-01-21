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
    """
    Вычисляет статистику BVH-дерева по графу.
    
    Статистика включает количество узлов, рёбер, глубину дерева и сбалансированность.
    Сбалансированность вычисляется как стандартное отклонение глубин листьев:
    - 0.0 означает идеально сбалансированное дерево (все листья на одной глубине)
    - Малые значения (< 1.0) означают хорошо сбалансированное дерево
    - Большие значения (> 2.0) означают несбалансированное дерево
    
    Parameters
    ----------
    graph : networkx.DiGraph
        Направленный граф структуры BVH-дерева, созданный через BVHTree.build_graph().
    
    Returns
    -------
    List[List[Union[str, int, float]]]
        Список строк статистики в формате [["Параметр", значение], ...].
    
    Notes
    -----
    Глубина листа вычисляется как длина кратчайшего пути от корня до листа.
    Для бинарного дерева с N листьями:
    - Идеально сбалансированное: все листья на глубине log2(N) или log2(N)+1
    - Несбалансированное: листья разбросаны по глубинам от 1 до максимальной
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Находим корневой узел (узел без входящих рёбер)
    roots = [n for n, d in graph.in_degree() if d == 0]
    root = roots[0] if roots else list(graph.nodes)[0]

    # Находим все листовые узлы (is_leaf=True в атрибутах узла)
    leaves = [n for n, attr in graph.nodes(data=True) if attr.get('is_leaf', False)]
    
    # Вычисляем глубину каждого листа как длину кратчайшего пути от корня
    leaf_depths = [nx.shortest_path_length(graph, root, leaf) for leaf in leaves]

    # Максимальная глубина дерева = глубина самого глубокого листа
    depth = max(leaf_depths) if leaf_depths else 0
    
    # Сбалансированность = стандартное отклонение глубин листьев
    # Чем меньше значение, тем более сбалансировано дерево
    # 0.0 = идеально сбалансированное (все листья на одной глубине)
    balance = np.std(leaf_depths) if leaf_depths else 0.0
    
    # Дополнительная статистика для понимания распределения глубин
    min_depth = min(leaf_depths) if leaf_depths else 0
    mean_depth = np.mean(leaf_depths) if leaf_depths else 0.0

    stats_table = [
        ["Количество вершин", num_nodes],
        ["Количество рёбер", num_edges],
        ["Глубина дерева (макс)", depth],
        ["Глубина листьев (мин)", min_depth],
        ["Глубина листьев (средн)", f"{mean_depth:.2f}"],
        ["Сбалансированность (std глубин листьев)", f"{balance:.2f}"]
    ]
    
    stats_dict = {
        "bvh_nodes": num_nodes,
        "bvh_edges": num_edges,
        "bvh_depth": depth,
        "bvh_balance": balance
    }
    
    return stats_table, stats_dict

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


def _build_table_prepare_build(results):
    """
    Строит таблицу для этапов подготовки и построения BVH дерева.
    
    Столбцы: Номер теста, Сетка, Кол-во ячеек сетки, кол-во ребер сетки, 
    кол-во вершин сетки, сколько ячеек в листе, включен ли ESC, 
    Вершин bvh, Глубина bvh, сбалансированность bvh, 
    Время подготовки, Время построения
    """
    return [
        {
            "Номер теста": r["test_id"],
            "Сетка": r["mesh"],
            "Кол-во ячеек сетки": r["faces"],
            "кол-во ребер сетки": r["edges"],
            "кол-во вершин сетки": r["nodes"],
            "сколько ячеек в листе": r["faces_in_node"],
            "включен ли ESC": r["esc"],
            "Вершин bvh": r["bvh_vertices"],
            "Глубина bvh": r["bvh_depth"],
            "сбалансированность bvh": f"{r['bvh_balance']:.2f}",
            "Время подготовки": r["prepare_time"],
            "Время построения": r["build_time"],
        }
        for r in results
    ]


def _build_table_traversal_classification(results):
    """
    Строит таблицу для этапов обхода и классификации.
    
    Столбцы: Номер теста, Сетка, Кол-во ячеек сетки, кол-во ребер сетки,
    кол-во вершин сетки, сколько ячеек в листе, включен ли ESC, 
    функция разбиения, найдено пар для коррекции (faces_to_fix),
    кол-во невозможных пар, кол-во пар найденных до классификации bvh,
    время обхода с учетом классификации
    """
    return [
        {
            "Номер теста": r["test_id"],
            "Сетка": r["mesh"],
            "Кол-во ячеек сетки": r["faces"],
            "кол-во ребер сетки": r["edges"],
            "кол-во вершин сетки": r["nodes"],
            "сколько ячеек в листе": r["faces_in_node"],
            "включен ли ESC": r["esc"],
            "функция разбиения": r["split_func"],
            "найдено пар для коррекции": r["pairs_to_fix"],
            "кол-во невозможных пар": r["impossible_couples_count"],
            "кол-во проверенных пар checked_pairs": r["checked_pairs"],
            "кол-во пар найденных без checheked_pairs": r["candidate_pairs_without_checked_pairs"],
            "cумма кол-во пар найденных пар и списка checheked_pairs": r["sum_candidate_pairs_count_checked_pairs"],
            "кол-во пар найденных до классификации bvh": r["candidate_pairs_count"],
            "кол-во пар найденных без классификации": r["candidate_pairs_after_czech_count"],
            "время обхода с учетом классификации": r["traversal_time"],
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

def _load_csv(filename):
    """
    Загружает таблицу из CSV файла и преобразует в формат результатов alg().
    
    Parameters
    ----------
    filename : str
        Путь к CSV файлу (относительно корня проекта или абсолютный).
    
    Returns
    -------
    List[Dict]
        Список словарей с результатами в формате, который возвращает функция alg().
        Ключи: test_id, mesh, faces, edges, nodes, prepare_time, build_time,
        traversal_time, total_time, split_func, esc, faces_in_node, pairs_to_fix.
    """
    path = filename if os.path.isabs(filename) else os.path.join("results", filename)
    
    results = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Преобразуем данные из CSV формата в формат alg()
            result = {
                "test_id": int(row["Тест"]),
                "mesh": row["Сетка"],
                "faces": int(row["Ячеек"]),
                "edges": int(row["Рёбер"]),
                "nodes": int(row["Вершин"]),
                "prepare_time": float(row["Подготовка, сек"]),
                "build_time": float(row["Построение, сек"]),
                "traversal_time": float(row["Обход, сек"]),
                "total_time": float(row["Общее время, сек"]),
                "split_func": row["Функция разбиения"],
                "esc": row["ESC"].lower() == "true" if isinstance(row["ESC"], str) else bool(row["ESC"]),
                "faces_in_node": int(row["Ячеек в листе"]),
                "pairs_to_fix": int(row["Пар для коррекции"]),
            }
            results.append(result)
    
    return results

def save_results(results):
    """
    Сохраняет результаты тестов в CSV файлы.
    
    Создаёт три таблицы:
    1. table_summary.csv - сводная таблица всех результатов
    2. table1_prepare_build.csv - таблица для этапов подготовки и построения
    3. table2_traversal_classification.csv - таблица для этапов обхода и классификации
    """
    _save_csv("table_summary.csv", _build_table_summary(results))
    _save_csv("table1_prepare_build.csv", _build_table_prepare_build(results))
    _save_csv("table2_traversal_classification.csv", _build_table_traversal_classification(results))
    logging.info("Результаты сохранены в results/: table_summary.csv, table1_prepare_build.csv, table2_traversal_classification.csv")

