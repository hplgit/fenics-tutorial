"""
FEniCS tutorial demo program: Diffusion of a Gaussian hill.
"""

from __future__ import print_function
from fenics import *
import time

T = 2.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
nx = ny = 30
mesh = RectangleMesh(Point(-2,-2), Point(2,2), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# Define initial value
u_0 = Expression('exp(-a*pow(x[0],2) - a*pow(x[1],2))',
                 degree=2, a=5)
u_p = interpolate(u_0, V)
u_p.rename('u', 'initial value')
vtkfile = File('gaussian_diffusion.pvd')
vtkfile << (u_p, 0.0)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_p + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Compute solution
u = Function(V)
u.rename('u', 'solution')
t = 0
for n in xrange(num_steps):

    # Update current time
    t += dt

    # Solve variational problem
    solve(a == L, u, bc)

    # Save to file and plot solution
    vtkfile << (u, float(t))
    plot(u)
    time.sleep(0.3)

    # Update previous solution
    u_p.assign(u)
