from __future__ import print_function
from fenics import *

Nx = Ny = 20
error = []
for i in range(2):
    Nx *= (i+1)
    Ny *= (i+1)

    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'Lagrange', 1)

    # Define boundary conditions
    u0 = Constant(0)

    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, u0_boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression('-2*exp(-2*x[0])*sin(pi*x[1])*('
                   '(4-5*pow(pi,2))*sin(2*pi*x[0]) '
                   ' - 8*pi*cos(2*pi*x[0]))')
    # Note: no need for pi=DOLFIN_PI in f, pi is valid variable
    a = inner(nabla_grad(u), nabla_grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    u_e = Expression(
        '2*exp(-2*x[0])*sin(2*pi*x[0])*sin(pi*x[1])')

    u_e_Function = interpolate(u_e, V)         # exact solution
    u_e_array = u_e_Function.vector().array()  # dof values
    max_error = (u_e_array - u.vector().array()).max()
    print('max error:', max_error, '%dx%d mesh' % (Nx, Ny))
    error.append(max_error)

print('Error reduction:', error[1]/error[0])

# Plot solution and mesh
plot(u)

# Dump solution to file in VTK format
file = File("poisson.pvd")
file << u

# Hold plot
interactive()
