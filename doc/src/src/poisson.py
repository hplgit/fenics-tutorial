"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f   in the unit square
            u = u_b  on the boundary

  u = 1 + x^2 + 2y^2 = u_b
  f = -6
"""

from __future__ import print_function
from fenics import *

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_b = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_b, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Solve variational problem
u = Function(V)
solve(a == L, u, bc)

# Plot solution
u.rename('u', 'solution')
plot(u)
plot(mesh)

# Save solution to file in VTK format
vtkfile = File('poisson.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2norm = errornorm(u_b, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_b = u_b.compute_vertex_values(mesh)
vertex_values_u  = u.compute_vertex_values(mesh)
import numpy as np
error_vertices = np.max(np.abs(vertex_values_u_b - vertex_values_u))

# Print errors
print('error_L2norm   =', error_L2norm)
print('error_vertices =', error_vertices)

# Hold plot
interactive()
