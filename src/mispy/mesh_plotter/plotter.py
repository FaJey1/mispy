from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from numpy import linalg as LA
from mispy.mesh_statistics import *


def draw_nodes(ax,
               coords = [],
               color = None,
               texts = [],
               text_enable = False):
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
            color=color, s=10)
    # Подписи граней и вершин
    if text_enable:
        for id, (x, y, z) in zip(texts, coords):
            ax.text(x, y, z, f"fec_{id}", color="black", fontsize=8)


def draw_line(ax, coords=None, color="red", linewidth=2):
    segs = [np.array(coords)]
    line_collection = Line3DCollection(segs, colors=color, linewidths=linewidth)
    ax.add_collection3d(line_collection)


def draw_planes(ax, 
                faces_coord=[], 
                colors=None,
                alpha=0.2,
                size_ratio=0.7,      # доля размера грани для квадрата
                draw_intersection=True,
                line_ratio=1):     # доля размера грани для линии

    planes = []  # (нормаль, центр, размер)

    for face_coord, color in zip(faces_coord, colors):
        face_coord = np.array(face_coord)
        center = face_coord.mean(axis=0)

        # Векторы и нормаль
        v1 = face_coord[1] - face_coord[0]
        v2 = face_coord[2] - face_coord[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # размер грани: максимум длины стороны
        side_lengths = [np.linalg.norm(face_coord[i] - face_coord[(i + 1) % 3]) for i in range(3)]
        size = max(side_lengths) * size_ratio
        line_len = max(side_lengths) * line_ratio

        #planes.append((normal, center, size, line_len))

        # базис
        u = v1 / np.linalg.norm(v1)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)

        half_size = size
        rect_points = [
            center + u * half_size + v * half_size,
            center - u * half_size + v * half_size,
            center - u * half_size - v * half_size,
            center + u * half_size - v * half_size
        ]

        rect = Poly3DCollection([rect_points], color=color, alpha=alpha*0.5)
        ax.add_collection3d(rect)

    # линия пересечения для 2 плоскостей
    if draw_intersection and len(planes) == 2:
        n1, p1, size1, line_len1 = planes[0]
        n2, p2, size2, line_len2 = planes[1]

        # направление
        direction = np.cross(n1, n2)
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-10:
            return
        direction /= norm_dir

        # решаем систему [n1; n2] x = [n1·p1; n2·p2]
        A = np.array([n1, n2])
        b = np.array([np.dot(n1, p1), np.dot(n2, p2)])

        # ищем частное решение через псевдообратную
        point = np.linalg.lstsq(A, b, rcond=None)[0]

        # две точки на линии (берем среднее line_len от двух граней)
        line_len = (line_len1 + line_len2) / 2
        p_start = point - direction * line_len
        p_end   = point + direction * line_len

        ax.plot([p_start[0], p_end[0]],
                [p_start[1], p_end[1]],
                [p_start[2], p_end[2]],
                color="black", linewidth=1)


def draw_face(ax,
              faces_coord = [], 
              colors = [],
              default_color = "blue",
              edge_enable = False,
              alpha = 0.3):
    poly_collection = Poly3DCollection(
            faces_coord,
            alpha=alpha,
            facecolors=colors if colors else default_color,
            edgecolors="k" if edge_enable else "none",
            linewidths=0.3 if edge_enable else 0.0
    )
    ax.add_collection3d(poly_collection)
    

def mesh_plotter(mesh, 
                 border_faces=[], 
                 border_multiline=[],
                 new_faces = [],
                 faces_enable=True, 
                 border_enable=False, 
                 border_face_enable = True,
                 nodes_enable=False, 
                 edge_enable=False,
                 draw_new_faces_enable = False,
                 alpha=0.3):
    
    vertices = np.array([node.p for node in mesh.nodes])
    x_range = vertices[:,0].max() - vertices[:,0].min()
    y_range = vertices[:,1].max() - vertices[:,1].min()
    z_range = vertices[:,2].max() - vertices[:,2].min()

    scale = 6 / max(x_range, y_range, z_range)  # коэффициент, чтобы не было слишком большого рисунка
    figsize = (x_range * scale, y_range * scale)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([x_range, y_range, z_range])
    
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # Карта цветов по зонам
    zones = list({face.zone.name for face in mesh.faces})
    colors = np.random.rand(len(zones), 3)
    color_map = {z: c for z,c in zip(zones, colors)}

    # Все грани одним Poly3DCollection
    polys = []
    colors = []

    if faces_enable:
        for face in mesh.faces:
            coords = np.array([node.p for node in face.nodes])
            polys.append(coords)
            colors.append(color_map.get(face.zone.name))

    draw_face(ax = ax, faces_coord = polys, colors = colors, alpha = alpha, edge_enable = edge_enable)

    # Вершины
    if nodes_enable:
        coords = np.array([node.p for node in mesh.nodes])
        draw_nodes(ax = ax, coords = coords, color = "red", text_enable = False)

    # Граница
    if border_face_enable:
        if border_faces:
            border_polys = [np.array([node.p for node in f.nodes]) for f in border_faces]
            draw_face(ax = ax, faces_coord = border_polys, default_color = "blue", alpha = 0.4, edge_enable = edge_enable)
    if border_enable:
        for line in border_multiline:
            draw_line(ax = ax, coords = line, color = "red", linewidth = 3)
    
    if draw_new_faces_enable:
        new_faces_coord = [np.array([node.p for node in f.nodes]) for f in new_faces]
        colors = np.random.rand(len(new_faces_coord), 3)
        draw_face(ax = ax, faces_coord = new_faces_coord, colors = list(colors), alpha = 0.4, edge_enable = edge_enable)
        
            
    # Оси и подпись
    ax.set_title(mesh.title)
    
    data = [[k, v] for k, v in mesh_statistics(mesh).items()]

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


def face_plotter(border_faces_classification = [],
                 intersection_line = [], 
                 text_enable=True, 
                 new_faces = [],
                 intersection_enable=False, 
                 nodes_enable=False, 
                 edge_enable=False,
                 plane_enable=True,
                 stop_draw = 0,
                 alpha=0.3):
    
    num = 0
    for face_pair, line, new_face_pair in zip(border_faces_classification, intersection_line, new_faces):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        coords1 = np.array([node.p for node in face_pair["faces"][0].nodes])
        coords2 = np.array([node.p for node in face_pair["faces"][1].nodes])
        colors = np.random.rand(2, 3)
        draw_face(ax = ax, faces_coord = [coords1, coords2], colors = list(colors), alpha = alpha, edge_enable = edge_enable)
        coords1 = np.array([node.p 
                            for face in new_face_pair["face_0_triangulation"] 
                            for node in face.nodes])
        coords2 = np.array([node.p 
                            for face in new_face_pair["face_1_triangulation"] 
                            for node in face.nodes])
        draw_face(ax = ax, faces_coord = [coords1, coords2], colors = list(colors), alpha = alpha, edge_enable = edge_enable)
        
        if plane_enable:
            draw_planes(ax = ax, faces_coord = [coords1, coords2], colors = colors)
        
        if text_enable:
            centroid = coords1.mean(axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], f"f_{face_pair['faces'][0].glo_id}", color="black", fontsize=10)
            centroid = coords2.mean(axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], f"f_{face_pair['faces'][1].glo_id}", color="black", fontsize=10)
            
        
        # Вершины
        if nodes_enable:
            coords = np.array([node.p for node in face_pair["faces"][1].nodes + face_pair["faces"][0].nodes])
            ids = [node.glo_id for node in face_pair["faces"][1].nodes + face_pair["faces"][0].nodes]
            draw_nodes(ax = ax, coords = coords, color = "black", texts = ids, text_enable = True)
                    
        # Точки и линия пересечения двух плоскостей, отрезок при пересечении двух граней
        if intersection_enable: 
            coords = np.array([ip[0] for ip in face_pair["face0_vs_face1"] + face_pair["face1_vs_face0"]])
            ids = [f"{ip[2]}_{ip[1]}" for ip in face_pair["face0_vs_face1"] + face_pair["face1_vs_face0"]]
            draw_nodes(ax = ax, coords = coords, color = "red", texts = ids, text_enable = True)
            draw_line(ax = ax, coords = line, color = "red")
                    
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

