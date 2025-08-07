from mispy.extract_mesh import Mesh
from mispy.mesh_statistics import mesh_statistics

    
def test_mesh_print():
    mesh = Mesh("tests/examples/small_sphere_double.dat")
    mesh_statistics(mesh)
    assert mesh is not None
