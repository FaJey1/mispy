from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy import linalg as LA

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
    

def mesh_plotter(mesh):
    fig = plt.figure(figsize=(10, 10))
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