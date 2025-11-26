import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def draw_face(ax,
              faces_coord=[],
              colors=[],
              default_color="blue",
              edge_enable=False,
              alpha=0.3,
              draw_aabb=False):
    """
    Рисует полигоны (faces) и, если draw_aabb=True, их AABB.
    """
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
    # ax.set_title(mesh.title)
    # data = [[k, v] for k, v in mesh_statistics(mesh).items()]

    # table = ax.table(
    #     cellText=data,
    #     loc="bottom",
    #     cellLoc="center",
    # )

    # table.scale(1, 1.2)
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()


def pairs_plotter(face_pairs,
                nodes_enable=False, 
                edge_enable=False,
                stop_draw = 0,
                alpha=0.3):
    num = 0
    for face_pair in face_pairs:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        coords1 = np.array([node.p for node in face_pair[0].nodes])
        coords2 = np.array([node.p for node in face_pair[1].nodes])
        colors = np.random.rand(2, 3)
        draw_face(ax = ax, faces_coord = [coords1, coords2], colors = list(colors), alpha = alpha, edge_enable = edge_enable, draw_aabb=True)
        ax.set_title(f"Pair num: {num}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()
        if stop_draw == num + 1 and stop_draw != 0:
            break
        num += 1


if __name__ == '__main__':
    pass
