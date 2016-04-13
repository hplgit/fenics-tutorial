"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Simplest example of computation and visualization with FEniCS.

  -Laplace(u) = f  on the unit square
            u = u0 on the boundary

  u = 1 + x^2 + 2y^2 = u0
  f = -6
"""

from __future__ import print_function
from fenics import *

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution on the screen
u.rename('u', 'solution')
plot(u)
plot(mesh)

# Dump solution to file in VTK format
vtkfile = File('poisson.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2norm = errornorm(u0, u, 'L2')

# Compute maximum error at vertices
vertex_values_u0 = u0.compute_vertex_values(mesh)
vertex_values_u  = u.compute_vertex_values(mesh)
error_vertices = max(abs(vertex_values_u0 - vertex_values_u))

# Print errors
print('error_L2norm   =', error_L2norm)
print('error_vertices =', error_vertices)

# Hold plot
interactive()
