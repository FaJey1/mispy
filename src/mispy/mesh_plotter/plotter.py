from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy import linalg as LA


# Линия пересечения двух ячеек не пересейкается 3й
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

def border_plotter(faces: list, edges: list, polygons_coord: list):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    zones = {}
    next_id = 0
    
    if faces:
        for face in faces:
            zone_name = face.zone.name
            if zone_name not in zones:
                zones[zone_name] = next_id
                next_id += 1
            rand_color = np.random.rand(3,)
            
            # Get&draw face nodes
            node_coords = [node.p for node in face.nodes]
            node_coords = np.array(node_coords)
            ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], color="blue", s=20)
            
            # Get&draw face edges
            for p1, p2 in ((edge.nodes[0].p, edge.nodes[1].p) for edge in face.edges):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="blue")
            
            # Draw face
            poly = Poly3DCollection([node_coords], alpha=0.2, facecolor="blue", edgecolors='none')
            ax.add_collection3d(poly)
            
    if polygons_coord:
        for polygon_coord in polygons_coord:
            rand_color = np.random.rand(3,)
            ax.scatter(polygon_coord[:,0], polygon_coord[:,1], polygon_coord[:,2], color=rand_color, s=20)
            poly = Poly3DCollection([polygon_coord], alpha=0.4, facecolor=rand_color, edgecolor='k')
            ax.add_collection3d(poly)
            
        
    for p1, p2 in ((edge.nodes[0].p, edge.nodes[1].p) for edge in edges):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red')
            
    ax.set_title("border")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()


# ==================================================================================================

def draw_multiline(ax, edges, color_map=None, default_color='blue'):
    pass


# ==================================================================================================

def draw_faces(ax, faces, color_map=None, default_color='blue'):
    """
    Рисует грани на 3D-оси.
    
    Parameters
    ----------
    ax : Axes3D
        Объект matplotlib 3D axes.
    faces : list[Face]
        Список граней для отрисовки.
    color_map : dict, optional
        Словарь {zone_name: color} для зон.
    default_color : str
        Цвет, если zone_name не указан в color_map.
    """
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
        poly = Poly3DCollection([node_coords], alpha=0.2, facecolor=color, edgecolors='none')
        ax.add_collection3d(poly)


# ==================================================================================================

def mesh_plotter(mesh: Mesh, border: dict = {}, border_enable: bool = False):
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
        candidate_faces = border.get("border_candidate_faces", [])
        if candidate_faces:
            draw_faces(ax, candidate_faces, default_color='purple')

        border_faces = border.get("border_faces", [])
        if border_faces:
            draw_faces(ax, border_faces, default_color='blue')

        border_multiline = border.get("border_multiline", [])
        if border_multiline:
            draw_multiline(ax, border_multiline, default_color='red')


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