import logging
import math

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import defaultdict

from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node
from .czech_classify import CzechClassify
from .bvh_tree import BVHTree
    
    
if __name__ == '__main__':
    #mesh = Mesh("tests/examples/small_sphere_double.dat")
    mesh = Mesh("tests/examples/sphere_double.dat")
    # bvh = BVHTree(mesh)
    # bvh.prepare_mesh(esc_enable=False)
    # bvh.build_tree()
    # graph = bvh.build_graph(bvh.root_node)
    #face.glo_id in [149, 75, 149, 66]:
    #[7349, 7481]
    #[4838 4841]
    #cs = CzechClassify((mesh.find_face_by_id(4838), mesh.find_face_by_id(4841)))
    #result, points = cs.classify()
    #print(result)
