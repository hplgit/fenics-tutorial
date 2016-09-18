"""
FEniCS tutorial demo program:
Nonlinear Poisson equation with Dirichlet conditions
in x-direction and homogeneous Neumann (symmetry) conditions
in all other directions. The domain is the unit hypercube in
of a given dimension.

-div(q(u)*grad(u)) = 0,
u = 0 at x=0, u=1 at x=1, du/dn=0 at all other boundaries.
q(u) = (1+u)^m

Solution method: Picard iteration (successive substitutions).
"""

from __future__ import print_function
import sys
from fenics import *
import numpy as np

# Create mesh and define function space
degree = int(sys.argv[1])
divisions = [int(arg) for arg in sys.argv[2:]]
d = len(divisions)
domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
mesh = domain_type[d-1](*divisions)
V = FunctionSpace(mesh, 'P', degree)

# Define boundary conditions at x=0 and x=1

tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < tol

Gamma_0 = DirichletBC(V, Constant(0.0), left_boundary)
Gamma_1 = DirichletBC(V, Constant(1.0), right_boundary)
bcs = [Gamma_0, Gamma_1]

# Choice of nonlinear coefficient
m = 2

def q(u):
    return (1+u)**m

# Define variational problem for Picard iteration
u = TrialFunction(V)
v = TestFunction(V)
u_ = interpolate(Constant(0.0), V)  # previous (known) u
a = dot(q(u_)*grad(u), grad(v))*dx
f = Constant(0.0)
L = f*v*dx

# Picard iterations
u = Function(V)     # new unknown function
eps = 1.0           # error measure ||u-u_||
tol = 1.0E-5        # tolerance
num_iter = 0        # iteration counter
maxiter = 25        # max no of iterations allowed
while eps > tol and iter < max_iter:
    num_iter += 1
    solve(a == L, u, bcs)
    du = u.vector().array() - u_.vector().array()
    eps = np.linalg.norm(du, ord=np.Inf)
    print('iter=%d: norm=%g' % (num_iter, eps))
    u_.assign(u)   # update for next iteration

convergence = 'convergence after %d Picard iterations' % num_iter
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
error = np.abs(u_e.vector().array() - u.vector().array()).max()
print('error:', error)
