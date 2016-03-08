"""
Solve a Poisson equation and its adjoint.
"""

from dolfin import *
from dolfin_adjoint import *

# Create mesh and define function space
mesh = UnitSquareMesh(6, 6)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
bc = DirichletBC(V, 0.0, "on_boundary")

# All parameters in dolfin-adjoint must be Constant or Function
nu = Constant(1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.0)
f = interpolate(Constant(1.0), V)  # Must be function for da to work
a = nu*inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Compute adjoint solution
u_d = Expression('sin(pi*x[0])')
J = Functional(0.5*inner(u-u_d, u-u_d)*dx)
m1 = SteadyParameter(f)
m2 = ScalarParameter(nu)
dJdm = compute_gradient(J, [m1, m2], project=True)
print type(dJdm), dJdm.__class__.__name__, dJdm[0].__class__.__name__, dJdm[1].__class__.__name__

# Should compare with da.py

# Plot solution and mesh
plot(dJdm[0], interactive=True)
print float(dJdm[1])
