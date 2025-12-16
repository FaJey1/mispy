import logging

import numpy as np
import networkx as nx
from tabulate import tabulate


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def statistic_bvh_tree_graph(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Находим корень: узел без входящих рёбер
    roots = [n for n, d in graph.in_degree() if d == 0]
    if not roots:
        root = list(graph.nodes)[0]
    else:
        root = roots[0]

    # Находим листья
    leaves = [n for n, attr in graph.nodes(data=True) if attr.get('is_leaf', False)]

    # Считаем глубину каждого листа
    leaf_depths = [nx.shortest_path_length(graph, root, leaf) for leaf in leaves]

    # Глубина дерева = максимальная глубина листа
    depth = max(leaf_depths) if leaf_depths else 0

    # Сбалансированность = стандартное отклонение глубин листьев
    balance = np.std(leaf_depths) if leaf_depths else 0

    # Подготовка данных для вывода через tabulate
    table = [
        ["Количество вершин", num_nodes],
        ["Количество рёбер", num_edges],
        ["Глубина дерева", depth],
        ["Сбалансированность (std глубин листьев)", f"{balance:.2f}"]
    ]
    return tabulate(table, headers=["Показатель", "Значение"], tablefmt="grid")


def statistic_mesh(mesh):
    table = [
        ["zones", len(mesh.zones)],
        ["nodes", len(mesh.nodes)],
        ["edges", len(mesh.edges)],
        ["faces", len(mesh.faces)]
    ]
    
    return table