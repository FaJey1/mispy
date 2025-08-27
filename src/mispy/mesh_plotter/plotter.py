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

def mesh_plotter(mesh: Mesh, border_faces: list = [], border_multiline: list = [], border_enable: bool = False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    zones = {}
    next_id = 0
    
    for face in mesh.faces:
        zone_name = face.zone.name
        if zone_name not in zones:
            zones[zone_name] = next_id
            next_id += 1
        
        # Get&draw face nodes
        node_coords = [node.p for node in face.nodes]
        node_coords = np.array(node_coords)
        ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], color='red', s=20)
        
        # Get&draw face edges
        for p1, p2 in ((edge.nodes[0].p, edge.nodes[1].p) for edge in face.edges):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='orange')
        
        # Draw face
        poly = Poly3DCollection([node_coords], alpha=0.2, facecolor=zone_color(zones[zone_name]), edgecolors='none')
        ax.add_collection3d(poly)
    
    # Draw intersection border
    if border_enable:
        for face in border_faces:
            zone_name = face.zone.name
            if zone_name not in zones:
                zones[zone_name] = next_id
                next_id += 1
            
            # Get&draw face nodes
            node_coords = [node.p for node in face.nodes]
            node_coords = np.array(node_coords)
            ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], color='blue', s=20)
            
            # Get&draw face edges
            for p1, p2 in ((edge.nodes[0].p, edge.nodes[1].p) for edge in face.edges):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='blue')
            
            # Draw face
            poly = Poly3DCollection([node_coords], alpha=0.2, facecolor="blue", edgecolors='none')
            ax.add_collection3d(poly)
        for p1, p2 in ((edge.nodes[0].p, edge.nodes[1].p) for edge in border_multiline):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red')
    
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