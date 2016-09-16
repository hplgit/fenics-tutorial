from __future__ import print_function
#from fenics import *  # not necessary, need only a few
import os, sys
path = os.path.join(os.pardir, os.pardir, os.pardir,
                    'src', 'stationary', 'poisson')
sys.path.insert(0, path)
from poisson_solver import (
    solver, Expression, Constant, interpolate, File, plot,
    interactive)

def data():
    """Return data for this Poisson problem."""
    u0 = Constant(0)
    u_e = Expression(
        '2*exp(-2*x[0])*sin(2*pi*x[0])*sin(pi*x[1])')
    f = Expression('-2*exp(-2*x[0])*sin(pi*x[1])*('
                   '(4-5*pow(pi,2))*sin(2*pi*x[0]) '
                   ' - 8*pi*cos(2*pi*x[0]))')
    return u0, f, u_e

def test_solver():
    """Check convergence rate of solver."""
    u0, f, u_e = data()
    Nx = 20
    Ny = Nx
    error = []
    # Loop over refined meshes
    for i in range(2):
        Nx *= i+1
        Ny *= i+1
        print('solving on 2(%dx%d) mesh' % (Nx, Ny))
        u = solver(f, u0, Nx, Ny, degree=1)
        # Make a finite element function of the exact u_e
        V = u.function_space()
        u_e_array = interpolate(u_e, V).vector().array()
        max_error = (u_e_array - u.vector().array()).max()  # Linf norm
        error.append(max_error)
        print('max error:', max_error)
    for i in range(1, len(error)):
        error_reduction = error[i]/error[i-1]
        print('error reduction:', error_reduction)
        assert abs(error_reduction - 0.25) < 0.1

def application():
    """Plot the solution."""
    u0, f, u_e = data()
    Nx = 40
    Ny = Nx
    u = solver(f, u0, Nx, Ny, 1)
    # Dump solution to file in VTK format
    file = File("poisson.pvd")
    file << u
    # Plot solution and mesh
    plot(u)

if __name__ == '__main__':
    test_solver()
    application()
    # Hold plot
    interactive()
