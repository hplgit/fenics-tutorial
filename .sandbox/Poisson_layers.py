from dolfin import *
import numpy

# Read mesh and subdomains from file
mesh = Mesh('hypercube_mesh.xml.gz')
subdomains = MeshFunction('uint', mesh, 'layers.xml.gz')

from u_layered import L, k as k_values, s, u_exact

V0 = FunctionSpace(mesh, 'DG', 0)
k = Function(V0)
"""
# Loop over all cell numbers, find corresponding
# subdomain number and fill cell value in k
for cell_no in range(len(subdomains.values())):
    subdomain_no = subdomains.values()[cell_no]
    k.vector()[cell_no] = k_values[subdomain_no]
"""
# Vectorized computation of k
help = numpy.asarray(subdomains.values(), dtype=numpy.int32)
k.vector()[:] = numpy.choose(help, k_values)


if mesh.num_cells() < 50:
    print 'k:', k.vector().array()

V = FunctionSpace(mesh, 'CG', 1)

# Define Dirichlet conditions for x=0 boundary

u_L = Constant(0)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0]) < tol

Gamma_0 = DirichletBC(V, u_L, LeftBoundary())

# Define Dirichlet conditions for x=1 boundary

u_R = Constant(1)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0] - 1) < tol
 
Gamma_1 = DirichletBC(V, u_R, RightBoundary())

bc = [Gamma_0, Gamma_1]

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Constant(0)
a = k*inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
problem = VariationalProblem(a, L, bc)
u = problem.solve()

plot(u, wireframe=True, title='u')
plot(k, title='continuous function version of k')
plot(mesh, title='mesh')
plot(subdomains, title='subdomains')

k_meshfunc = MeshFunction('double', mesh, mesh.topology().dim())

"""
# Scalar version
for i in range(len(subdomains.values())):
    k_meshfunc.values()[i] = k_values[subdomains.values()[i]]
"""
# Vectorized version
help = numpy.asarray(subdomains.values(), dtype=numpy.int32)
k_meshfunc.values()[:] = numpy.choose(help, k_values)

plot(k_meshfunc, title='k as mesh function')

# Find maximum error (must loop since u_exact(x) assumes scalar x)
u_array = u.vector().array()
max_diff = 0
for i in range(mesh.num_vertices()):
    x0 = mesh.coordinates()[i][0]
    diff = abs(u_exact(x0) - u_array[i])
    if diff > max_diff:
        max_diff = diff
print 'Max error (infinity norm):', max_diff
interactive()

