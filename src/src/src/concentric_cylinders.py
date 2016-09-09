from dolfin import *
import mshr

# Parameters for geometry
a = 0.04
b = a + 0.004
c = a + 0.01
L = 0.5

# Define cylinders
inner = mshr.CSGCGALDomain3D(mshr.Cylinder(Point(0, 0, 0), Point(0, 0, L), a, a))
mid = mshr.CSGCGALDomain3D(mshr.Cylinder(Point(0, 0, 0), Point(0, 0, L), b, b))
outer = mshr.CSGCGALDomain3D(mshr.Cylinder(Point(0, 0, 0), Point(0, 0, L), c, c))

generator = mshr.TetgenMeshGenerator3D()
generator.parameters["preserve_surface"] = True
generator.parameters["mesh_resolution"] = 16.

# Mesh inner cylinder
inner_mesh = generator.generate(inner)


# Mesh mid part
mid_mesh = generator.generate(mshr.CSGCGALDomain3D(mid-inner))

# Mesh outer part
outer_mesh = generator.generate(mshr.CSGCGALDomain3D(outer-mid))

# Glue together inner and mid mesh
inner_mid_mesh = mshr.DolfinMeshUtils.merge_meshes(inner_mesh, mid_mesh)

# Glue outer and inner/mid
the_entire_domain = mshr.DolfinMeshUtils.merge_meshes(inner_mid_mesh, outer_mesh)

# Save to file
File('pipe_mesh.pvd') << the_entire_domain

plot(the_entire_domain, interactive=True)
