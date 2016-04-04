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
u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=1)

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

# Compute and print error
u_e = interpolate(u0, V)
error = max(abs(u_e.vector().array() - u.vector().array()))
import numpy as np
error = np.abs(u_e.vector().array() - u.vector().array()).max()
print('error =', error)

# Hold plot
interactive()
