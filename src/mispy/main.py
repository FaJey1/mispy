import logging

from mispy.extract_mesh import *
from mispy.transform_mesh import *
from mispy.visualization_mesh import *


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    mesh = Mesh("tests/examples/small_sphere_double.dat")
    bvh_tree = BVH(mesh)
    bvh_tree.build_tree()
    candidate_pairs = bvh_tree.find_candidates(zone_filter="different")
    
    candidates = {}
    for pair in candidate_pairs:
        candidates[pair[0].glo_id] = pair[0]
        candidates[pair[1].glo_id] = pair[1]
    print(len(candidates.values()))
    #draw_face_pair(candidate_pairs, nodes_enable=True, edge_enable=True)
    draw_mesh(mesh=mesh, border_faces=list(candidates.values()), border_face_enable=True, edge_enable=False, nodes_enable=False)


if __name__ == '__main__':
    main()
