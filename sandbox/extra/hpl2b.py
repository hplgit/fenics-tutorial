"""
FEniCS tutorial demo program 2. -Laplace(u) = f, u = u0 on the boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""

from dolfin import *
import numpy

# Create mesh and define function space
mesh = UnitSquare(2, 1)
V = FunctionSpace(mesh, 'CG', 1)


# Define boundary conditions
u0 = Function(V, '1 + x[0]*x[0] + 2*x[1]*x[1]')

class Boundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
        return on_boundary

u0_boundary = Boundary()
bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V, '-6')
a = dot(grad(v), grad(u))*dx
L = v*f*dx

# Assemble and solve linear system
A = assemble(a)
b = assemble(L)
bc.apply(A, b)
u = Function(V)
solve(A, u.vector(), b)


#plot(u)
#interactive()

print """
Solution of the Poisson problem -Laplace(u) = f,
with u = u0 on the boundary and a
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
    

