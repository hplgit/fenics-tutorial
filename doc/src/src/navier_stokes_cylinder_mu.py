"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma) = f
                          div(u) = 0
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

T = 5.0            # final time
num_steps = 5000   # number of time steps
dt = T / num_steps # time step size
mu = 0.001         # dynamic viscosity
rho = 1            # density

# Create mesh
channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
geometry = channel - cylinder
mesh = generate_mesh(geometry, 64)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u0 = Function(V)
u1 = Function(V)
p0 = Function(Q)
p1 = Function(Q)

# Define expressions used in variational forms
U   = 0.5*(u0 + u)
n   = FacetNormal(mesh)
f   = Constant((0, 0))
k   = Constant(dt)
mu  = Constant(mu)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u0) / k, v)*dx \
   + rho*dot(dot(u0, nabla_grad(u0)), v)*dx \
   + inner(sigma(U, p0), epsilon(v))*dx \
   + dot(p0*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - rho*dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p0), nabla_grad(q))*dx - (1/k)*div(u1)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u1, v)*dx - k*dot(nabla_grad(p1 - p0), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Create VTK files for saving solution for later visualization
vtkfile_u = File('ns/velocity.pvd')
vtkfile_p = File('ns/pressure.pvd')

# FIXME: mpi_comm_world should not be needed here, fix in FEniCS!

# Create time series for saving solution for later computation
timeseries_u = TimeSeries(mpi_comm_world(), 'ns/velocity')
timeseries_p = TimeSeries(mpi_comm_world(), 'ns/pressure')
timeseries_m = TimeSeries(mpi_comm_world(), 'ns/mesh')

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

# Time-stepping
t = 0
for n in xrange(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u1.vector(), b1, 'bicgstab', 'ilu')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p1.vector(), b2, 'bicgstab', 'ilu')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u1.vector(), b3, 'bicgstab')

    # Plot solution
    plot(u1, title='Velocity')
    plot(p1, title='Pressure')

    # Save solution to file
    vtkfile_u << (u1, t)
    vtkfile_p << (p1, t)

    # Save solution to file (HDF5)
    timeseries_u.store(u1.vector(), t)
    timeseries_p.store(p1.vector(), t)
    timeseries_m.store(mesh, t)

    # Update previous solution
    u0.assign(u1)
    p0.assign(p1)

    # Update progress bar
    progress.update(t / T)
    print('u max:', u1.vector().array().max())

# Hold plot
interactive()
