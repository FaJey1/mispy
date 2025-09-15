from mispy.czech_intersections_algorithm import CzechIntersectionsAlgorithm
from mispy.extract_mesh import Mesh
from mispy.mesh_plotter import border_plotter
    
def test_build_borders():
    mesh = Mesh("tests/examples/small_sphere_double.dat")
    cia = CzechIntersectionsAlgorithm(mesh)
    cia.describe_border_faces()
    faces = cia.get_border_faces()
    border_plotter(faces)
    assert mesh is not None
