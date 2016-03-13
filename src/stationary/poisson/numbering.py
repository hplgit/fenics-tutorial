# The following shows that the order in which the mesh coordinates are stored is
# in general different from the ordering of the degrees of freedom.
from dolfin import *

mesh = UnitSquareMesh(2, 2)
coor = mesh.coordinates()
coor

V = FunctionSpace(mesh, 'CG', 1)

# We make some element in the function space V
u = interpolate(Expression('x[0]+x[1]'), V)
# Vector of degrees of freedom is the representation of function w.r.t to the
# basis V. The degrees of freedom are point evaluations at mesh vertices.
u_array = u.vector().array()
u_array

# plot(u) enables toggling the mesh with numbering of vertices
plot(u, interactive=True)

# Different ways of testing for P1 elements
assert mesh.num_vertices() == V.dim()
assert len(coor) == len(u_array)
assert V.ufl_element().degree() == 1
assert coor.shape[0] == u_array.shape[0]

u_at_vertices = u.compute_vertex_values()
for i, x in enumerate(coor):
    print('vertex %d: u_at_vertices[%d]=%g\tu(%s)=%g' %
          (i, i, u_at_vertices[i], x, u(x)))

# The next two methods shall use the fact that while the ordering of dofs and
# vertices is different, having mesh.num_vertices() == V.dim() ensures that
# there exists a one-to-one mapping between the two numberings. The map from
# vertex ordering to dof ordering is suggestively called vertex_to_dof_map
v2d = vertex_to_dof_map(V)
# If you print the object you will see that it is simply a list. The i-th entry
# in the list is a degree of freedom index such that i-th vertex is v2d[i]-th
# degree of freedom. Here's how it works
print all(near(u(x), u_array[v2d[i]], 1E-10) for i, x in enumerate(coor))

# We can go the other way around, i.e. dof->vertex ordering, too. The mapping is
# computed via dof_to_vertex_map
d2v = dof_to_vertex_map(V)
# This is again a list such that i-th degree of freedom is a point evaluation at
# d2v[i]-th vertex of the mesh. The usage is as follows
print all(near(u(coor[d2v[i]]), dof, 1E-10) for i, dof in enumerate(u_array))

# We can now tie the methods together and write our own compute_vertex_value
# function. The idea is to use u_array and the mapping:
print near(max(u_array[v2d] - u_at_vertices), 0)
# Note that we did not evaluate (expensive)! We simple reoreder the existing
# degrees of freeom (cheap).
