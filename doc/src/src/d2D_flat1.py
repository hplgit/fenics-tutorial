"""
FEniCS tutorial demo program: Diffusion equation with Dirichlet
conditions and a solution that will be exact at all nodes.
"""

from __future__ import print_function
from fenics import *
import numpy as np

# Create mesh and define function space
nx = ny = 4
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
alpha = 3; beta = 1.2
u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, boundary)

# Initial condition
u_1 = interpolate(u0, V)
#project(u0, V) will not result in exact solution at the nodes!

dt = 0.3      # time step

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)
F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_1 + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Compute solution
u = Function(V)   # the unknown at a new time level
T = 1.9           # total simulation time
t = dt
while t <= T:
    print('time =', t)
    u0.t = t
    solve(a == L, u, bc)

    # Verify
    u_e = interpolate(u0, V)
    max_error = np.abs(u_e.vector().array() -
                       u.vector().array()).max()
    print('max error, t=%.2f: %-10.3g' % (t, max_error))

    t += dt
    u_1.assign(u)
