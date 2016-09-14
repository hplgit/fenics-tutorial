"""
FEniCS tutorial demo program:
Poisson equation with Dirichlet and Neumann conditions.
The solution is checked to coincide with the exact solution at all nodes.

-Laplace(u) = f on the unit square.
u = u0 on x=0 and x=1.
-du/dn = g on y=0 and y=1.
u0 = u = 1 + x^2 + 2y^2, f = -6, g = -4y.
"""

from __future__ import print_function
from fenics import *
import numpy

# Create mesh and define function space
mesh = UnitSquareMesh(3, 2)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define Dirichlet boundary conditions
u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')

def Dirichlet_boundary(x, on_boundary):
    tol = 1E-14   # tolerance for coordinate comparisons
    return on_boundary and \
           (abs(x[0]) < tol or abs(x[0] - 1) < tol)

bc = DirichletBC(V, u0, Dirichlet_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
g = Expression('-4*x[1]')
a = inner(grad(u), grad(v))*dx
L = f*v*dx - g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

#plot(u)

print("""
Solution of the Poisson problem -Laplace(u) = f,
with u = u0 on x=0,1 and -du/dn = g at y=0,1.
%s
""" % mesh)

# Verification
u_e = interpolate(u0, V)
u_e_array = u_e.vector().array()
u_array = u.vector().array()
print('Max error:', numpy.abs(u_e_array - u_array).max())

# Compare numerical and exact solution
center = (0.5, 0.5)
print('numerical u at the center point:', u(center))
print('exact     u at the center point:', u0(center))

#interactive()
