"""
FEniCS tutorial demo program: Diffusion of a Gaussian hill
with u=0 on the boundaries.
"""

from __future__ import print_function
from fenics import *
import time

# Create mesh and define function space
nx = ny = 30
mesh = RectangleMesh(Point(-2,-2), Point(2,2), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# Initial condition
I = Expression('exp(-a*pow(x[0],2)-a*pow(x[1],2))', a=5)
u_1 = interpolate(I, V)
u_1.rename('u', 'initial condition')
vtkfile = File('diffusion.pvd')
vtkfile << (u_1, 0.0)
#project(u0, V) will not result in exact solution at the nodes!

dt = 0.01    # time step

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_1 + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Compute solution
u = Function(V)             # the unknown at a new time level
u.rename('u', 'solution')   # name and label for u
T = 0.5                     # total simulation time
t = dt
while t <= T:
    print('time =', t)
    solve(a == L, u, bc)
    vtkfile << (u, float(t))
    plot(u)
    time.sleep(0.3)

    t += dt
    u_1.assign(u)
