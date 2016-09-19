"""
Solve -Laplace(u) = f on the unit square
with u = u0 on the boundary.
"""
from __future__ import print_function
from fenics import *

def solver(f, u0, Nx, Ny, Nz, degree=1):
    """
    Solve -Laplace(u)=f on [0,1]x[0,1]x[0,1] with 2*Nx*Ny*Nz Lagrange
    elements of specified degree and u=u0 (Expresssion) on
    the boundary.
    """
    # Create mesh and define function space
    mesh = UnitCubeMesh(Nx, Ny, Nz)
    V = FunctionSpace(mesh, 'Lagrange', degree)

    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, u0_boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(nabla_grad(u), nabla_grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    return u

def test_solver():
    """Reproduce u=1+x^2+2y^2 to "machine precision"."""
    tol = 1E-11  # This problem's precision
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1] - 4*x[2]*x[2]')
    f = Constant(2.0)
    for Nx, Ny, Nz in [(3,3,3), (3,5,3), (5,3,5)]:
        for degree in 1, 2, 3:
            print('solving on 2(%dx%dx%d) mesh with P%d elements'
                  % (Nx, Ny, Nz, degree))
            u = solver(f, u0, Nx, Ny, Nz, degree)
            # Make a finite element function of the exact u0
            V = u.function_space()
            u0_Function = interpolate(u0, V)  # exact solution
            # Check that dof arrays are equal
            u0_array = u0_Function.vector().array()  # dof values
            max_error = (u0_array - u.vector().array()).max()
            msg = 'max error: %g for 2(%dx%d) mesh and degree=%d' %\
                  (max_error, Nx, Ny, degree)
            assert max_error < tol, msg

def application_test():
    """Plot the solution in the test problem."""
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1] - 4*x[2]*x[2]')
    f = Constant(2.0)
    u = solver(f, u0, 6, 4, 3, 1)
    # Dump solution to file in VTK format
    file = File("poisson.pvd")
    file << u
    # Plot solution and mesh
    plot(u)

if __name__ == '__main__':
    application_test()
    # Hold plot
    interactive()
