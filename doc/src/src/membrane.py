"""
FEniCS tutorial demo program: Deflection of a membrane.

  -Laplace(w) = p = Gaussian function

Computed on the unit circle with w = 0 on the boundary.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

# Create mesh and define function space
domain = Circle(Point(0.0, 0.0), 1.0)
mesh = generate_mesh(domain, 20)
V = FunctionSpace(mesh, 'P', 2)

# Define boundary condition
u_D = Constant(0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define load
beta = 8
R0 = 0.6
p = Expression(
    '4*exp(-pow(beta,2)*(pow(x[0], 2) + pow(x[1]-R0, 2)))',
    beta=beta, R0=R0)

# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(w), grad(v))*dx
L = p*v*dx

# Solve variational problem
w = Function(V)
solve(a == L, w, bc)

# Plot solution
p = interpolate(p, V)
w.rename('w', 'deflection')
p.rename('p', 'load')
plot(w, title='Deflection')
plot(p, title='Load')

# Save solution to file in VTK format
vtkfile_w = File('membrane_deflection.pvd')
vtkfile_w << w
vtkfile_p = File('membrane_load.pvd')
vtkfile_p << p

# Curve plot along x = 0 comparing p and w
import numpy as np
import matplotlib.pyplot as plt
tol = 1E-8  # avoid hitting points outside the domain
y = np.linspace(-1+tol, 1-tol, 101)
points = [(0, y_) for y_ in y]  # 2D points
w_line = np.array([w(point) for point in points])
p_line = np.array([p(point) for point in points])
plt.plot(y, 100*w_line, 'r-', y, p_line, 'b--') # magnify w
plt.legend(['100 x deflection', 'load'], loc='upper left')
plt.xlabel('y'); plt.ylabel('$p$ and $100u$')
plt.savefig('plot.pdf'); plt.savefig('plot.png')

# Hold plots
interactive()
plt.show()
