"""
FEniCS tutorial demo program: Convection-diffusion-reaction for a system
describing the concentration of two species A and B undergoing an
autocatalytic reaction A + 2B --> B + 2B. The convective velocity is
given by the flow field w from the Navier-Stokes demo navier_stokes.py.

  u_1' + w . nabla(u_1) - div(epsilon*grad(u_1)) = -u_1*u_2^2
  u_2' + w . nabla(u_2) - div(epsilon*grad(u_2)) = +u_1*u_2^2

"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

T = 5.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size
theta = 0.5        # implicitness parameter for time-stepping

# FIXME: mpi_comm_world should not be needed here, fix in FEniCS!

# Read mesh from file
mesh = Mesh()
timeseries_m = TimeSeries(mpi_comm_world(), 'ns/mesh')
timeseries_m.retreive(mesh, 0)

# Define function space for velocity
W = VectorFunctionSpace(mesh, 'P', 2)

# Define function space for system
V1 = FunctionSpace(mesh, 'P', 1)
V2 = FunctionSpace(mesh, 'P', 1)
V = V1 * V2

# Define initial condition
u_01 = Expression('x[0] < 0.5 ? 1 : 0')
u_02 = Expression('x[0] < 0.5 ? 0 : 1')

# Define test functions
v_1, v_2 = TestFunctions(V)

# Define functions for velocity and time-stepping
w = Function(W)
u = Function(V)
u_p = Function(V)

# Split function for system to access components
u_1, u_2 = split(u)
u_p1, u_p2 = split(u_p)

# Define expressions used in variational forms
theta = Constant(theta)
U_1 = (1 - theta)*u_p1 + theta*u_1
U_2 = (1 - theta)*u_p2 + theta*u_2
k = Constant(dt)

# Define variational problem
F = ((u_1 - u_p1) / dt)*v_1*dx + dot(w, grad(U_1))*v_1*dx + \
    epsilon*dot(grad(U_1), grad(v_1))*dx + U_1*U_2**2*v_1*dx +
    ((u_2 - u_p2) / dt)*v_2*dx + dot(w, grad(U_2))*v_1*dx + \
    epsilon*dot(grad(U_2), grad(v_2))*dx - U_1*U_2**2*v_2*dx


# Time-stepping
t = 0
for n in xrange(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u1.vector(), b1, 'bic
