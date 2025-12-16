import logging

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from .statistics import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def draw_face(ax,
              faces_coord=[],
              colors=[],
              default_color="blue",
              edge_enable=False,
              alpha=0.3,
              draw_aabb=False):
    # Рисуем сами грани
    poly_collection = Poly3DCollection(
        faces_coord,
        alpha=alpha,
        facecolors=colors if colors else default_color,
        edgecolors="k" if edge_enable else "none",
        linewidths=0.3 if edge_enable else 0.0
    )
    ax.add_collection3d(poly_collection)

    # Рисуем AABB для каждой грани
    if draw_aabb:
        for coords in faces_coord:
            # coords: (3,3) np.array
            min_corner = coords.min(axis=0)
            max_corner = coords.max(axis=0)

            # 8 вершин бокса
            corners = np.array([
                [min_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], max_corner[1], max_corner[2]],
                [min_corner[0], max_corner[1], max_corner[2]],
            ])

            # Рёбра бокса: пары индексов вершин
            edges = [
                [0,1],[1,2],[2,3],[3,0],  # нижняя грань
                [4,5],[5,6],[6,7],[7,4],  # верхняя грань
                [0,4],[1,5],[2,6],[3,7]   # вертикальные рёбра
            ]

            # рисуем рёбра
            for e in edges:
                ax.plot(*zip(corners[e[0]], corners[e[1]]), color="red", linewidth=0.5)


def draw_intersection_seg(ax,
            segment_coord=[],
            color="red",
            linewidths=0.3,
            alpha=0.3):
    if segment_coord is None or len(segment_coord) == 0:
        return

    for seg in segment_coord:
        # if len(seg) != 2:
        #     print("aswfedf")
        #     continue  # защита от повреждённых данных

        p1, p2 = seg

        # Преобразуем точки в numpy
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)

        # Координаты для plot()
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        zs = [p1[2], p2[2]]

        ax.plot(xs, ys, zs,
                color=color,
                linewidth=linewidths,
                alpha=alpha)
    

def mesh_plotter(mesh,
                 faces_enable=True,
                 draw_aabb=False, 
                 edge_enable=False,
                 faces_to_fix=[],
                 faces_to_fix_enable=False,
                 alpha=0.3):
    
    # vertices = np.array([node.p for node in mesh.nodes])
    # x_range = vertices[:,0].max() - vertices[:,0].min()
    # y_range = vertices[:,1].max() - vertices[:,1].min()
    # z_range = vertices[:,2].max() - vertices[:,2].min()

    # scale = 6 / max(x_range, y_range, z_range)  # коэффициент, чтобы не было слишком большого рисунка
    # figsize = (x_range * scale, y_range * scale)

    # fig = plt.figure(figsize=figsize)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([x_range, y_range, z_range])
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Карта цветов по зонам
    zones = list({face.zone.name for face in mesh.faces})
    colors = np.random.rand(len(zones), 3)
    color_map = {z: c for z,c in zip(zones, colors)}

    # Все грани одним Poly3DCollection
    polys = []
    colors = []

    if faces_enable:
        for face in mesh.faces:
            #if face.glo_id in [149, 75, 66]:
            ##[4838 4841] [7349, 7481]
            #if face.glo_id in [4838, 4841]:
            coords = np.array([node.p for node in face.nodes])
            polys.append(coords)
            colors.append(color_map.get(face.zone.name))
    draw_face(ax = ax, faces_coord = polys, colors = colors, alpha = alpha, edge_enable = edge_enable, draw_aabb = draw_aabb)
    
    polys = []
    if faces_to_fix_enable:
        for face_id in faces_to_fix:
            #if face.glo_id in [149, 75, 66]:
            ##[4838 4841] [7349, 7481]
            #if face.glo_id in [4838, 4841]:
            coords = np.array([node.p for node in mesh.find_face_by_id(face_id).nodes])
            polys.append(coords)
    draw_face(ax = ax, faces_coord = polys,alpha = alpha)
    # Оси и подпись
    ax.set_title(mesh.title)
    data = statistic_mesh(mesh=mesh)

    table = ax.table(
        cellText=data,
        loc="bottom",
        cellLoc="center",
    )

    table.scale(1, 1.2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()


def fix_face_plotter(mesh,
                faces_to_fix=[],
                draw_aabb=False, 
                edge_enable=False,
                alpha=0.3,
                stop_draw = 0):
    
    num = 0
    for broken_face_id in faces_to_fix:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = np.random.rand(2, 3)
        print(broken_face_id, faces_to_fix[broken_face_id])
        coords1 = np.array([node.p for node in mesh.find_face_by_id(broken_face_id).nodes])
        draw_face(ax = ax, faces_coord = [coords1], colors = list(colors), alpha = alpha, edge_enable = edge_enable, draw_aabb=draw_aabb)
        draw_intersection_seg(
            ax=ax,
            segment_coord=faces_to_fix[broken_face_id],
            color="red",
            linewidths=1.2,
            alpha=1.0
        )
        
        ax.set_title(f"Face id: {broken_face_id}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()
        if stop_draw == num + 1 and stop_draw != 0:
            break
        num += 1
    


def pairs_broken_face_plotter(face_pairs,
                edge_enable=False,
                stop_draw = 0,
                draw_aabb = False,
                alpha=0.3):
    num = 0
    for key in face_pairs:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        coords1 = np.array([node.p for node in face_pairs[key]["faces"][0].nodes])
        coords2 = np.array([node.p for node in face_pairs[key]["faces"][1].nodes])
        colors = np.random.rand(2, 3)
        draw_face(ax = ax, faces_coord = [coords1, coords2], colors = list(colors), alpha = alpha, edge_enable = edge_enable, draw_aabb=draw_aabb)
        draw_intersection_seg(
            ax=ax,
            segment_coord=[face_pairs[key]["intersection_points"]],
            color="red",
            linewidths=0.3,
            alpha=1.0
        )
        ax.set_title(f"Pair num: {num}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()
        if stop_draw == num + 1 and stop_draw != 0:
            break
        num += 1


def visualize_bvh_tree_graph(graph):
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
