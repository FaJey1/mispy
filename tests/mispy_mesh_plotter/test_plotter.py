from mispy.extract_mesh import Mesh
from mispy.mesh_plotter import mesh_plotter

    
def test_mesh_print():
    mesh = Mesh("tests/examples/small_sphere_double.dat")
    mesh_plotter(mesh)
    assert mesh is not None
