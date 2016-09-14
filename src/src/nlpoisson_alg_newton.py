"""
FEniCS tutorial demo program:
Nonlinear Poisson equation with Dirichlet conditions
in x-direction and homogeneous Neumann (symmetry) conditions
in all other directions. The domain is the unit hypercube in
of a given dimension.

-div(q(u)*grad(u)) = 0,
u = 0 at x=0, u=1 at x=1, du/dn=0 at all other boundaries.
q(u) = (1+u)^m

Solution method: Newton iteration at the algebraic level.
"""

from __future__ import print_function
from fenics import *
import numpy, sys

# Create mesh and define function space
degree = int(sys.argv[1])
divisions = [int(arg) for arg in sys.argv[2:]]
d = len(divisions)
domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
mesh = domain_type[d-1](*divisions)
V = FunctionSpace(mesh, 'P', degree)

# Define boundary conditions at x=0,1 for initial guess
tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < tol

Gamma_0 = DirichletBC(V, Constant(0.0), left_boundary)
Gamma_1 = DirichletBC(V, Constant(1.0), right_boundary)
bcs = [Gamma_0, Gamma_1]

# Define variational problem for initial guess (q(u)=1, i.e., m=0)
u = TrialFunction(V)
v = TestFunction(V)
F = dot(grad(u), grad(v))*dx + Constant(0)*v*dx
A, b = assemble_system(lhs(F), rhs(F), bcs)
u_ = Function(V)
solve(A, u_.vector(), b, 'lu')

# Note that all Dirichlet conditions must be zero for
# the correction function in a Newton-type method
Gamma_0_du = DirichletBC(V, Constant(0.0), left_boundary)
Gamma_1_du = DirichletBC(V, Constant(0.0), right_boundary)
bcs_du = [Gamma_0_du, Gamma_1_du]

# Choice of nonlinear coefficient
m = 2

def q(u):
    return (1+u)**m

def Dq(u):
    return m*(1+u)**(m-1)

# Define variational problem in a Newton iteration
du = TrialFunction(V) # u = u_ + omega*du
F = dot(q(u_)*grad(du), grad(v))*dx + \
    dot(Dq(u_)*du*grad(u_), grad(v))*dx + \
    dot(q(u_)*grad(u_), grad(v))*dx

# Newton iteration at the algebraic level
du = Function(V)
u  = Function(V)  # u = u_ + omega*du
omega = 1.0       # relaxation parameter
eps = 1.0
tol = 1.0E-5
num_iter = 0
max_iter = 25
# u_ must have right boundary conditions
while eps > tol and iter < max_iter:
    num_iter += 1
    print(num_iter, 'iteration', end=' ')
    A, b = assemble_system(lhs(F), rhs(F), bcs_du)
    solve(A, du.vector(), b)
    eps = numpy.linalg.norm(du.vector().array(), ord=numpy.Inf)
    print('Norm:', eps)
    u.vector()[:] = u_.vector() + omega*du.vector()
    u_.assign(u)


convergence = 'convergence after %d Newton iterations at the algebraic level' % num_iter
if num_iter >= max_iter:
    convergence = 'no ' + convergence

print("""
Solution of the nonlinear Poisson problem div(q(u)*grad(u)) = f,
with f=0, q(u) = (1+u)^m, u=0 at x=0 and u=1 at x=1.
%s
%s
""" % (mesh, convergence))

# Find max error
u_exact = Expression('pow((pow(2, m+1)-1)*x[0] + 1, 1.0/(m+1)) - 1', m=m)
u_e = interpolate(u_exact, V)
error = numpy.abs(u_e.vector().array() - u.vector().array()).max()
print('error:', error)
