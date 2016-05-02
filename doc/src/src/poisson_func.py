"""
Solve -Laplace(u) = f on the unit square
with u = u_b on the boundary.
"""
from __future__ import print_function
from fenics import *

def solver(f, u_b, Nx, Ny, degree=1):
    """
    Solve -Laplace(u) = f on [0,1] x [0,1] with 2*Nx*Ny Lagrange
    elements of specified degree and u=u_D (Expresssion) on
    the boundary.
    """

    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'P', degree)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_b, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    return u

def test_solver():
    """Reproduce u = 1 + x^2 + 2y^2 to "machine precision"."""

    # Set up parameters for testing
    tol = 1E-11
    u_b = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    f = Constant(-6.0)

    # Iterate over mesh sizes and degrees
    for Nx, Ny in [(3,3), (3,5), (5,3), (20,20)]:
        for degree in 1, 2, 3:
            print('Solving on a 2 x (%d x %d) mesh with P%d elements.'
                  % (Nx, Ny, degree))

            # Compute solution
            u = solver(f, u_b, Nx, Ny, degree)

            # Compute maximum error at vertices
            vertex_values_u_D = u_D.compute_vertex_values(mesh)
            vertex_values_u  = u.compute_vertex_values(mesh)
            import numpy as np
            error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

            # Check maximum error
            msg = 'error_max = %g' % error_max
            assert max_error < tol, msg

def application_test():
    """Compute and post-process solution"""

    # Set up problem parameters and call solver
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    f = Constant(-6.0)
    u = solver(f, u_b, 6, 4, 1)

    # Plot solution
    u.rename('u', 'u')
    plot(u)
    plot(mesh)

    # Save solution to file in VTK format
    vtkfile = File('poisson.pvd')
    vtkfile << u

if __name__ == '__main__':
    application_test()
    interactive()
