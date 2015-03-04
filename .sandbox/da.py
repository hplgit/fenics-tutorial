"""
Solve a Poisson equation and its adjoint.
"""

from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(6, 6)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
bc = DirichletBC(V, 0, "on_boundary")

nu = 1

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6)
a = nu*inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Compute adjoint solution (when the adjoint PDE is derived by hand)
lambda_ = TrialFunction(V)
a = nu*inner(nabla_grad(v), nabla_grad(lambda_))*dx
u_d = Expression('sin(pi*x[0])')
L = -(u - u_d)*v*dx
lambda_ = Function(V)
solve(a == L, lambda_, bc)

# Plot solution and mesh
plot(u, interactive=True)
plot(mesh, interactive=True)

# Dump solution to file in VTK format
file = File('poisson.pvd')
file << u
