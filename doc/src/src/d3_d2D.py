"""
FEniCS tutorial demo program: Diffusion equation with Dirichlet
conditions and a solution that will be exact at all nodes.
As d2_d2D.py, but here we test various start vectors for iterative
solution of the linear system at each time level.
The script d3_d2D_script.py runs experiments with different start
vectors and prints out the number of iterations.
"""

from fenics import *
import numpy, sys
numpy.random.seed(12)

# zero, random, default, last
initial_guess = 'zero' if len(sys.argv) == 1 else sys.argv[1]
# PETSc, Epetra, MTL4, 
la_backend = 'PETSc' if len(sys.argv) <= 2 else sys.argv[2]

parameters['linear_algebra_backend'] = la_backend


# Create mesh and define function space
nx = ny = 40
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
alpha = 3; beta = 1.2
u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                alpha=alpha, beta=beta, t=0)

class Boundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
        return on_boundary

boundary = Boundary()
bc = DirichletBC(V, u0, boundary)

# Initial condition
u_1 = interpolate(u0, V)
u_2 = Function(V)
#u_1 = project(u0, V)  # will not result in exact solution!

dt = 0.9      # time step
T = 10*dt        # total simulation time

# Define variational problem

# Laplace term
u = TrialFunction(V)
v = TestFunction(V)
a_K = dot(grad(u), grad(v))*dx

# "Mass matrix" term
a_M = u*v*dx

M = assemble(a_M)
K = assemble(a_K)
A = M + dt*K
bc.apply(A)

# f term
f = Expression('beta - 2 - 2*alpha', beta=beta, alpha=alpha)

# Linear solver initialization
#solver = KrylovSolver('cg', 'ilu')
solver = KrylovSolver('gmres', 'ilu')
#solver = KrylovSolver('gmres', 'none')  # cg doesn't work, probably because matrix bc makes it nonsymmetric
solver.parameters['absolute_tolerance'] = 1E-5
solver.parameters['relative_tolerance'] = 1E-17  # irrelevant
solver.parameters['maximum_iterations'] = 10000
if initial_guess == 'default':
    solver.parameters['nonzero_initial_guess'] = False
else:
    solver.parameters['nonzero_initial_guess'] = True
u = Function(V)
set_log_level(DEBUG)

print 'nonzero initial guess:', solver.parameters['nonzero_initial_guess']

# Compute solution
u = Function(V)
t = dt
while t <= T:
    print 'time =', t
    # f.t = t  # if time-dep f
    f_k = interpolate(f, V)
    F_k = f_k.vector()
    b = M*u_1.vector() + dt*M*F_k
    u0.t = t
    bc.apply(b)   # BIG POINT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if initial_guess == 'zero':
        u.vector()[:] = 0
    elif initial_guess == 'last':
        pass
    elif initial_guess == 'random':
        u.vector()[:] = numpy.random.uniform(-1, 1, V.dim())
    elif t >= 2*dt and initial_guess == 'extrapolate':
        u.vector()[:] = 2*u_1.vector() - u_2.vector()
    solver.solve(A, u.vector(), b)

    # Verify
    u_e = interpolate(u0, V)
    u_e_array = u_e.vector().array()
    u_array = u.vector().array()
    print 'Max error, t=%-10.3f:' % t, numpy.abs(u_e_array - u_array).max()

    t += dt
    u_2.assign(u_1)
    u_1.assign(u)

