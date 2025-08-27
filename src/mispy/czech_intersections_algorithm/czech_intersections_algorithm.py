import logging

import numpy as np
from numpy import linalg as LA
from typing import List, Tuple, Optional
from collections import defaultdict
from itertools import combinations

from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node


# ==================================================================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class CzechIntersectionsAlgorithm:
    def __init__(self, mesh):
        self.mesh = mesh
        logging.info("Total face count: %s", len(self.mesh.faces))
        self.border_candidates = []
        self.border_faces = []
        self.border_multiline = []
    
    def tnoi_filtering(self) :
        """
        Filtering intersection candidates
        
        Returns
        -------
        candidates
            List of border candidates.
        """
        # Алгоритм background grid
        bbox_min = np.min([node.p for node in self.mesh.nodes], axis=0)
        bbox_max = np.max([node.p for node in self.mesh.nodes], axis=0)
        bbox_size = bbox_max - bbox_min
        logging.info("tnoi_filtering >> bbox_size: %s", bbox_size)
        
        # Размер ячейки фоновой сетки (cell_size) = среднему размеру face
        cell_size = np.cbrt(np.mean([np.linalg.norm(f.nodes[0].p - f.nodes[1].p) for f in self.mesh.faces]))
        logging.info("tnoi_filtering >> cell_size: %s", cell_size)
        
        # Привязка треугольников к ячейкам
        # Для каждого Face считаем его локальный bounding box
        # Находим, какие ячейки сетки он пересекает
        # Записываем face в список кандидатов ячейки
        grid = defaultdict(list)

        for face in self.mesh.faces:
            f_min = np.min([n.p for n in face.nodes], axis=0)
            f_max = np.max([n.p for n in face.nodes], axis=0)
            cell_min = np.floor((f_min - bbox_min) / cell_size).astype(int)
            cell_max = np.floor((f_max - bbox_min) / cell_size).astype(int)

            for ix in range(cell_min[0], cell_max[0]+1):
                for iy in range(cell_min[1], cell_max[1]+1):
                    for iz in range(cell_min[2], cell_max[2]+1):
                        grid[(ix, iy, iz)].append(face)
        logging.info("The triangles are bound to the cells, count: %s", len(grid))
        
        # Для каждой ячейки берём все грани (faces)
        # Если там несколько разных сеток -> формируем все пары (face_i, face_j)
        # Если в ячейке три и более сетки (A, B, D), то берём комбинации: (A, B), (A, D), (B, D)
        # Если в ячейке 1 зона, но несколько граней -> проверяем самопересечение
        candidates_set = set()
        for cell_faces in grid.values():
            # Группируем грани по zone
            by_zone = defaultdict(list)
            for f in cell_faces:
                by_zone[f.zone.name].append(f)

            zones = list(by_zone.keys())

            if len(zones) == 1:
                # Одна зона, но несколько граней -> проверяем самопересечения
                # faces = by_zone[zones[0]]
                # if len(faces) > 1:
                #     for f1, f2 in combinations(faces, 2):
                #         key = tuple(sorted((id(f1), id(f2))))
                #         candidates_set.add(key)
                pass 
            else:
                # Несколько зон -> формируем пары между зонами
                for i, zone1 in enumerate(zones):
                    for zone2 in zones[i+1:]:
                        for f1 in by_zone[zone1]:
                            for f2 in by_zone[zone2]:
                                key = tuple(sorted((id(f1), id(f2))))
                                candidates_set.add(key)

        # Восстанавливаем объекты граней из id
        all_faces = [f for faces in grid.values() for f in faces]
        id_to_face = {id(f): f for f in all_faces}
        self.border_candidates = [(id_to_face[i1], id_to_face[i2]) for i1, i2 in candidates_set]

        print(f"INFO: Candidates for checking the intersection have been formed, count: {len(self.border_candidates)}")
        
        sub = {f.glo_id for pair in self.border_candidates for f in pair}
        return [f for f in self.mesh.faces if f.glo_id in sub]


    def defining_intersections(self):
        """
        
        """
        pass
    

# ==================================================================================================

if __name__ == '__main__':
    pass

# ==================================================================================================
