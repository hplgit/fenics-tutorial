"""
FEniCS program for the deflection w(x,y) of a membrane:
-Laplace(w) = p = Gaussian function, in a unit circle,
with w = 0 on the boundary.
As membrane1.py, but with computation of the energy (a
quantity derived from the solution).
"""

from dolfin import *
import numpy

# Set pressure function:
T = 10.0  # tension
A = 1.0   # pressure amplitude
R = 0.3   # radius of domain
theta = 0.2
x0 = 0.6*R*cos(theta)
y0 = 0.6*R*sin(theta)
sigma = 0.025
n = 40   # approx no of elements in radial direction
mesh = UnitCircle(n)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary condition w=0

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.0), boundary)

# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
a = inner(nabla_grad(w), nabla_grad(v))*dx
f = Expression('4*exp(-0.5*(pow((R*x[0] - x0)/sigma, 2)) '
               '     - 0.5*(pow((R*x[1] - y0)/sigma, 2)))',
               R=R, x0=x0, y0=y0, sigma=sigma)
L = f*v*dx

# Compute solution
w = Function(V)
problem = LinearVariationalProblem(a, L, w, bc)
solver  = LinearVariationalSolver(problem)
solver.parameters['linear_solver'] = 'cg'
solver.parameters['preconditioner'] = 'ilu'
solver.solve()

# Plot solution and mesh
plot(mesh, title='Mesh over scaled domain')
plot(w, title='Scaled deflection')
p = interpolate(f, V)
plot(p, title='Scaled pressure')

# Find maximum real deflection
max_w = w.vector().array().max()
max_D = A*max_w/(8*pi*sigma*T)
print 'Maximum real deflection is', max_D

# Compute elastic energy: integral of T*abs(grad(w))^2
E_functional = 0.5*(A*R/(8*pi*sigma))**2*inner(nabla_grad(w), nabla_grad(w))*dx
E = assemble(E_functional)
#E = assemble(E_functional, mesh=mesh)
print 'Elastic energy:', E

# Should be at the end
#interactive()
