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

def draw_intersection_points(ax, cls_points, color_map, default_color = 'blue'):
    for face in cls_points:
        face_id = face[0]
        cls_list = [data[1] for data in face[1]]
        coords_list = [data[2] for data in face[1]]
        for cls, coords in zip(cls_list, coords_list):
            x, y, z = coords
            ax.scatter(x, y, z, color="black", s=15)              # точка
            ax.text(x, y, z, f"{face_id}_{cls}", color="black") # подпись


# ==================================================================================================

def draw_edge(ax, edge, color, enable_text = False):
    p1, p2 = edge.nodes[0].p, edge.nodes[1].p
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)
    if enable_text:
        mid = [(p1[0] + p2[0]) / 2,
    (p1[1] + p2[1]) / 2,
    (p1[2] + p2[2]) / 2]
        ax.text(mid[0], mid[1], mid[2], edge.glo_id, color="purple", fontsize=9)


# ==================================================================================================

def draw_faces(ax, faces, color_map = None, default_color = 'blue', alpha=0.3, enable_text = False):
    for face in faces:
        zone_name = face.zone.name
        color = color_map.get(zone_name, default_color) if color_map else default_color

        # Координаты узлов
        node_coords = np.array([node.p for node in face.nodes])
        ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], color=color, s=20)

        # Рёбра
        for edge in face.edges:
            # p1, p2 = edge.nodes[0].p, edge.nodes[1].p
            # ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)
            # if enable_text:
            #     mid = [(p1[0] + p2[0]) / 2,
            # (p1[1] + p2[1]) / 2,
            # (p1[2] + p2[2]) / 2]
            #     ax.text(mid[0], mid[1], mid[2], edge.glo_id, color="purple", fontsize=9)
            draw_edge(ax, edge, color)

        # Грань
        poly = Poly3DCollection([node_coords], alpha=alpha, facecolor=color, edgecolors='none')
        ax.add_collection3d(poly)
        
        if enable_text:
            centroid = node_coords.mean(axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], face.glo_id, color="black", fontsize=10)


# ==================================================================================================

def draw_candidates(candidate_pairs: list = [], candidate_plane_pairs: list = [], candidate_intersection_planes_line: list = [], candidate_cls_points: list = [], stop_draw = 0):
    pair_num = -1
    if not stop_draw:
        stop_draw = len(candidate_pairs)
    for pair_faces, pair_planes, intersection_line, cls_points_pair in zip(candidate_pairs, candidate_plane_pairs, candidate_intersection_planes_line, candidate_cls_points):
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
        # if pair_faces[0].glo_id != 108 and pair_faces[1].glo_id != 108:
        #     continue
        color_map = {name: zone_color(idx) for name, idx in zones.items()}
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        draw_faces(ax, [pair_faces[0], pair_faces[1]], color_map=color_map, enable_text = True)
        draw_faces_planes(ax, pair_planes)
        draw_lines(ax, intersection_line)
        draw_intersection_points(ax, cls_points_pair, color_map=color_map)
        ax.set_title(f"candidate_pair_{pair_num}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()


# ==================================================================================================

def draw_border_faces(border_faces: list = [], border_triangulation_faces: list = [], stop_draw = 0):
    pair_num = -1
    if not stop_draw:
        stop_draw = len(border_faces)
        
    for bf, tf in zip(border_faces, border_triangulation_faces):
        pair_num += 1
        if pair_num == stop_draw:
            break
        
        zones = {}
        next_id = 0
        for face in bf[1]:
            zone_name = face.zone.name
            if zone_name not in zones:
                zones[zone_name] = next_id
                next_id += 1
        color_map = {name: zone_color(idx) for name, idx in zones.items()}
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        draw_faces(ax, [bf[1][0], bf[1][1]], color_map=color_map, enable_text = True)
        draw_faces(ax, [tf[0], tf[1]], default_color="yellow", enable_text = True)
        draw_edge(ax = ax, edge = bf[0], color = "black", enable_text = False)
        ax.set_title(f"border_pair_{pair_num}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()
        
    print(len(border_faces))


# ==================================================================================================

def draw_multiline(ax, edges):
    for p1, p2 in ((edge.nodes[0].p, edge.nodes[1].p) for edge in edges):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')


# ==================================================================================================

def mesh_plotter(mesh: Mesh, border: list = [], border_multiline: list = [], border_enable: bool = False):
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
    # draw_faces(ax, mesh.faces, color_map=color_map)
    draw_multiline(ax, border_multiline)

    # if border_enable:
    #     draw_faces(ax, border, default_color='blue', enable_text = False)


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