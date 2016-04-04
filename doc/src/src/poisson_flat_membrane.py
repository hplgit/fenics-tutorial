"""
FEniCS program for the deflection w(x,y) of a membrane:
-Laplace(w) = p = Gaussian function, in a unit circle,
with w = 0 on the boundary.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

domain = Circle(Point(0.0, 0.0), 1.0)
mesh = generate_mesh(domain, 20)
V = FunctionSpace(mesh, 'P', 2)

def u0_boundary(x, on_boundary):
    return on_boundary

u0 = Constant(0)
bc = DirichletBC(V, u0, u0_boundary)

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

# Compute solution
w = Function(V)
solve(a == L, w, bc)

p = interpolate(p, V)
w.rename('w', 'deflection')
p.rename('p', 'load')
plot(w, title='Deflection')
plot(p, title='Load')

# Dump p and w to file in VTK format
vtkfile = File('membrane.pvd')
vtkfile << w
vtkfile << p

# Should be at the end
interactive()
