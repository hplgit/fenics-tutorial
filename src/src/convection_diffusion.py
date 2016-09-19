"""
FEniCS tutorial demo program: Convection-diffusion in a cylinder
with particular focus on mesh generation, subdomains and boundary
conditions.

  -div(lmbda*grad(u)) + div(c*beta*u) = f
"""

from __future__ import print_function
from fenics import *
from mshr import *

# Parameters for geometry
a = 0.04
b = a + 0.004
c = a + 0.01
L = 0.5

# Define cylinders
cylinder_a = Cylinder(Point(0, 0, 0), Point(0, 0, L), a, a)
cylinder_b = Cylinder(Point(0, 0, 0), Point(0, 0, L), b, b)
cylinder_c = Cylinder(Point(0, 0, 0), Point(0, 0, L), c, c)

# Define domain and set subdomains
domain = cylinder_c
domain.set_subdomain(1, cylinder_b)
domain.set_subdomain(2, cylinder_a)

# Generate mesh
mesh = generate_mesh(domain, 16)

xmlfile = File('pipe.xml')
xmlfile << mesh

vtkfile = File('pipe.pvd')
vtkfile << mesh
