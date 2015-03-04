"""As membrane1.py, but with more Viper visualization."""
from dolfin import *

# Set pressure function:
T = 10.0  # tension
A = 1.0   # pressure amplitude
R = 0.3   # radius of domain
theta = 0.2
x0 = 0.6*R*cos(theta)
y0 = 0.6*R*sin(theta)
sigma = 0.025
#sigma = 50  # large value for verification

n = 40   # approx no of elements in radial direction
mesh = UnitCircleMesh(n)
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
               '      -0.5*(pow((R*x[1] - y0)/sigma, 2)))',
               R=R, x0=x0, y0=y0, sigma=sigma)
L = f*v*dx

# Compute solution
w = Function(V)
problem = LinearVariationalProblem(a, L, w, bc)
solver  = LinearVariationalSolver(problem)
solver.parameters['linear_solver'] = 'cg'
solver.parameters['preconditioner'] = 'ilu'
solver.solve()

# Find maximum real deflection
max_w = w.vector().array().max()
max_D = A*max_w/(8*pi*sigma*T)
print 'Maximum real deflection is', max_D

# Demonstrate some visualization

# Cannot do plot(w) first and then grab viz object!
import time
viz_w = plot(w,
             mode='warp',     # 'color' gives flat color plot
             wireframe=False, # True: elevated mesh
             title='Scaled membrane deflection',
             scale=3.0,       # stretch z axis by a factor of 3
             elevate=-75.0,   # tilt camera -75 degrees
             scalarbar=True,  # show colorbar
             axes=False,      # do not show X, Y, Z axes
             window_width=800,
             window_height=600,
             )

viz_w.write_png('membrane_deflection')
viz_w.write_pdf('tmp')
# Rotate pdf file (right) from landscape to portrait
import os
os.system('pdftk tmp.pdf cat 1-endR output membrane_deflection.pdf')

f = interpolate(f, V)
viz_f = plot(f,
             title='Scaled pressure',
             elevate=-65.0,
             scale=3.0)
viz_f.write_png('pressure')
viz_f.write_pdf('tmp')
# Rotate pdf file (right) from landscape to portrait
os.system('pdftk tmp.pdf cat 1-endR output pressure.pdf')

viz_m = plot(mesh, title='Finite element mesh')

#time.sleep(15)

# Should be at the end
interactive()
