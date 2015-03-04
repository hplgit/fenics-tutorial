"""
FEniCS tutorial demo program 1. -Laplace(u) = f on the unit square.
u = u0 on x=0,
u0 = u = 1 + x^2 + 2y^2,
f = -6.
"""

from dolfin import *
import numpy

# Create mesh and define function space
mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, 'CG', 1)


# Define boundary conditions
u0 = Function(V, '1 + x[0]*x[0] + 2*x[1]*x[1]')

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary

u0_boundary = DirichletBoundary()
bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
p = Function(V, 'x[0] + x[1]')
f = Function(V, '-8*x[0] - 10*x[1]')
a = p*dot(grad(v), grad(u))*dx
L = v*f*dx

# Compute solution
problem = VariationalProblem(a, L, bc)
u = problem.solve()

#plot(u)
#interactive()

print """
Solution of the Poisson problem -Laplace(u) = f,
with u = u0 on x=0,1 and -du/dn = g at y=0,1.
%s
""" % mesh

# Dump solution to the screen
u_nodal_values = u.vector()
u_array = u_nodal_values.array()
coor = mesh.coordinates()
for i in range(len(u_array)):
    print 'u(%8g,%8g) = %g' % (coor[i][0], coor[i][1], u_array[i])

# Verification
tolerance = 1E-14
u0.interpolate()
u0_array = u0.vector().array()
ok = numpy.allclose(u_array, u0_array, 
                    rtol=tolerance, atol=tolerance)
print 'Solution is', 'ok' if ok else 'not correct'

# Compare numerical and exact solution at (0.5, 0.5)
center_point = numpy.array((0.5, 0.5))
u_value = numpy.zeros(1)
u.eval(u_value, center_point)
u0_value = numpy.zeros(1)
u0.eval(u0_value, center_point)
print 'numerical u at the center point:', u_value
print 'exact     u at the center point:', u0_value
    

