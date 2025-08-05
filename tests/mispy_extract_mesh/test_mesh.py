from mispy_extract_mesh import Mesh

def test_mesh_creation():
    mesh = Mesh("tests/examples/small_sphere_double.dat")
    assert mesh is not None