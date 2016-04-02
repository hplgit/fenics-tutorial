"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Simplest example of computation and visualization with FEniCS.

-Laplace(u) = f on the unit square.
u = u0 on the boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""
from __future__ import print_function
from fenics import *

# Create mesh and define function space
mesh = UnitSquareMesh(6, 4)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)

# Dump solution to file in VTK format
file = File("poisson.pvd")
file << u

# Find max error
u0_Function = interpolate(u0, V)         # exact solution
u0_array = u0_Function.vector().array()  # dof values
import numpy as np
max_error = np.abs(u0_array - u.vector().array()).max()
print('max error:', max_error)

# Hold plot
interactive()
