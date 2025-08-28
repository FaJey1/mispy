from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
from mispy.czech_intersections_algorithm import Plane, Line
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy import linalg as LA


# ==================================================================================================

def zone_color(zone):
    colors = [
        "#e6194b",  # красный
        "#3cb44b",  # зеленый
        "#ffe119",  # желтый
        "#4363d8",  # синий
        "#f58231",  # оранжевый
        "#911eb4",  # фиолетовый
        "#46f0f0",  # голубой
        "#f032e6",  # розовый
        "#bcf60c",  # лаймовый
        "#fabebe",  # светло-розовый
    ]
    return colors[zone]


# ==================================================================================================

def draw_lines(ax, line, color="red"):
    p1, p2 = line.segment
    ax.plot([p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color)
    

# ==================================================================================================

def draw_faces_planes(ax, planes, alpha=0.1):
    for plane in planes:
        poly = Poly3DCollection([plane.corners], color=np.random.rand(3,), alpha=alpha)
        ax.add_collection3d(poly)


# ==================================================================================================

def draw_faces(ax, faces, color_map = None, default_color = 'blue', alpha=0.3):
    for face in faces:
        zone_name = face.zone.name
        color = color_map.get(zone_name, default_color) if color_map else default_color

        # Координаты узлов
        node_coords = np.array([node.p for node in face.nodes])
        ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], color=color, s=20)

        # Рёбра
        for edge in face.edges:
            p1, p2 = edge.nodes[0].p, edge.nodes[1].p
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)

        # Грань
        poly = Poly3DCollection([node_coords], alpha=alpha, facecolor=color, edgecolors='none')
        ax.add_collection3d(poly)


# ==================================================================================================

def draw_candidates(candidate_pairs: list, candidate_plane_pairs: list, candidate_intersection_planes_line: list, stop_draw = 1):
    pair_num = -1
    for pair_faces, pair_planes, intersection_line in zip(candidate_pairs, candidate_plane_pairs, candidate_intersection_planes_line):
        pair_num += 1
        if pair_num == stop_draw:
            break
        
        zones = {}
        next_id = 0
        for face in pair_faces:
            zone_name = face.zone.name
            if zone_name not in zones:
                zones[zone_name] = next_id
                next_id += 1
        color_map = {name: zone_color(idx) for name, idx in zones.items()}
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        draw_faces(ax, [pair_faces[0], pair_faces[1]], color_map=color_map)
        draw_faces_planes(ax, pair_planes)
        draw_lines(ax, intersection_line)
        ax.set_title(f"candidate_pairs_{pair_num}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()


# ==================================================================================================

def draw_multiline(ax, edges, color_map=None, default_color='blue'):
    for p1, p2 in ((edge.nodes[0].p, edge.nodes[1].p) for edge in edges):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red')


# ==================================================================================================

def mesh_plotter(mesh: Mesh, border: list, border_enable: bool = False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    zones = {}
    next_id = 0
    for face in mesh.faces:
        zone_name = face.zone.name
        if zone_name not in zones:
            zones[zone_name] = next_id
            next_id += 1
    color_map = {name: zone_color(idx) for name, idx in zones.items()}
    draw_faces(ax, mesh.faces, color_map=color_map)

    if border_enable:
        draw_faces(ax, border, default_color='blue')


    ax.set_title(mesh.title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()


# ==================================================================================================

if __name__ == '__main__':
    pass

# ==================================================================================================