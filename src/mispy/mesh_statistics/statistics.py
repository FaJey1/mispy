from mispy.extract_mesh import Mesh, Zone, Face, Edge, Node


def mesh_statistics(mesh):
    info = {
            "zones_count": len(mesh.zones),
            "nodes_count": len(mesh.nodes),
            "faces_count": len(mesh.faces),
            "edges_count": len(mesh.edges),
    }
    print("Mesh summary:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    

# ==================================================================================================

if __name__ == '__main__':
    pass

# ==================================================================================================