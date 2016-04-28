"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Simplest example of computation and visualization with FEniCS.

-Laplace(u) = f on the unit square.
u = u0 on the boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""
from __future__ import print_function
from fenics import *

# Scaled variables
L = 1; W = 0.2
lambda_ = 1
rho = 1
delta = W/L
gamma = 0.25*delta**2
beta = 0.8
mu = beta
g = gamma

# Create mesh and define function space
mesh = BoxMesh(Point(0,0,0), Point(L,W,W), 10, 3, 3)
V = VectorFunctionSpace(mesh, 'P', 1)

# Define boundary conditions
tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and (x[0] < tol)

bc = DirichletBC(V, Constant((0,0,0)), clamped_boundary)

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # no of space dim
v = TestFunction(V)
f = rho*Constant((0,0,g))
T = Constant((0,0,0))
a = inner(sigma(u), epsilon(v))*dx
L = -dot(f, v)*dx + dot(T, v)*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u, title='Displacement', mode='displacement')

von_Mises = inner(sigma(u), sigma(u)) - div(u)
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
plot(von_Mises, title='Stress intensity', mode='displacement')
u_magnitude = sqrt(dot(u,u))
u_magnitude = project(u_magnitude, V)
plot(u_magnitude, 'Displacement magnitude', mode='displacement')
print('min/max u:', u_magnitude.vector().array().min(),
      u_magnitude.vector().array().max())

# Dump solution to file in VTK format
file = File("elasticity.pvd")
file << u
file << von_Mises
file << u_magnitude

# Find max error
#u0_Function = interpolate(u0, V)         # exact solution
#u0_array = u0_Function.vector().array()  # dof values
#import numpy as np
#max_error = np.abs(u0_array - u.vector().array()).max()
#print('max error:', max_error)

#grad, eps:        -3.04536573498e-06 0.120769911819
#grad, grad:       -3.04536573498e-06 0.120769911819
#nabla_grad, grad: -3.04536573498e-06 0.120769911819
#nabla_grad, eps:  same

# Hold plot
interactive()
