"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Simplest example of computation and visualization with FEniCS.

-div(q(u)*grad(u)) = f on the unit square.
u = u0 on the boundary.
"""
from __future__ import print_function

# Warning: from fenics import * imports f, q, and sym
# (which overwrites the f and q (function) objects
# and also sym if we do import sympy as sym).
# Therefore, do fenics import first and then overwrite
from fenics import *

def q(u):
    """Nonlinear coefficient in the PDE."""
    return 1 + u**2

# Use sympy to compute f given manufactured solution u
import sympy as sym
x, y = sym.symbols('x[0] x[1]')
u = 1 + x + 2*y
f = - sym.diff(q(u)*sym.diff(u, x), x) - \
      sym.diff(q(u)*sym.diff(u, y), y)
f = sym.simplify(f)
u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)
print('u=', u_code)
print('f=', f_code)

# Create mesh and define function space
mesh = UnitSquareMesh(16, 14)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u0 = Expression(u_code)

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = Function(V)  # not TrialFunction!
v = TestFunction(V)
f = Expression(f_code)
F = dot(q(u)*grad(u), grad(v))*dx - f*v*dx

# Compute solution
solve(F == 0, u, bc)

plot(u)

# Find max error
u0_Function = interpolate(u0, V)         # exact solution
u0_array = u0_Function.vector().array()  # dof values
import numpy as np
max_error = np.abs(u0_array - u.vector().array()).max()
print('max error:', max_error)

"""
u0_at_vertices = u0_Function.compute_vertex_values()
u_at_vertices = u.compute_vertex_values()
coor = V.mesh().coordinates()
for i, x in enumerate(coor):
    print('vertex %2d (%9g,%9g): error=%g'
          % (i, x[0], x[1],
             u0_at_vertices[i] - u_at_vertices[i]))
"""
# Hold plot
interactive()
