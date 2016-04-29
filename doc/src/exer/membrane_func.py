"""
FEniCS program for the deflection w(x,y) of a membrane:
-Laplace(w) = p = Gaussian function, in a unit circle,
with w = 0 on the boundary.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

def solver(
    f, u0, mesh, degree=1,
    linear_solver='Krylov', # Alt: 'direct'
    abs_tol=1E-5,           # Absolute tolerance in Krylov solver
    rel_tol=1E-3,           # Relative tolerance in Krylov solver
    max_iter=1000,          # Max no of iterations in Krylov solver
    log_level=PROGRESS,     # Amount of solver output
    dump_parameters=False,  # Write out parameter database?
    ):
    """
    Solve -Laplace(u)=f on given mesh with Lagrange elements
    of specified degree and u=u0 (Expresssion) on the boundary.
    """
    V = FunctionSpace(mesh, 'P', degree)

    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, u0_boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)

    if linear_solver == 'Krylov':
        prm = parameters['krylov_solver'] # short form
        prm['absolute_tolerance'] = abs_tol
        prm['relative_tolerance'] = rel_tol
        prm['maximum_iterations'] = max_iter
        print(parameters['linear_algebra_backend'])
        set_log_level(log_level)
        if dump_parameters:
            info(parameters, True)
        solver_parameters = {'linear_solver': 'gmres',
                             'preconditioner': 'ilu'}
    else:
        solver_parameters = {'linear_solver': 'lu'}

    solve(a == L, u, bc, solver_parameters=solver_parameters)
    return u


def application(beta, R0, num_elements_radial_dir):
    # Scaled pressure function
    p = Expression(
        '4*exp(-pow(beta,2)*(pow(x[0], 2) + pow(x[1]-R0, 2)))',
        beta=beta, R0=R0)

    # Generate mesh over the unit circle
    domain = Circle(Point(0.0, 0.0), 1.0)
    mesh = generate_mesh(domain, num_elements_radial_dir)

    w = solver(p, Constant(0), mesh, degree=1,
               linear_solver='direct')
    w.rename('w', 'deflection')  # set name and label (description)

    # Plot scaled solution, mesh and pressure
    plot(mesh, title='Mesh over scaled domain')
    plot(w, title='Scaled ' + w.label())
    V = w.function_space()
    p = interpolate(p, V)
    p.rename('p', 'pressure')
    plot(p, title='Scaled ' + p.label())

    # Dump p and w to file in VTK format
    vtkfile1 = File('membrane_deflection.pvd')
    vtkfile1 << w
    vtkfile2 = File('membrane_load.pvd')
    vtkfile2 << p

def test_membrane():
    """Verification for constant pressure."""
    p = Constant(4)
    # Generate mesh over the unit circle
    domain = Circle(Point(0.0, 0.0), 1.0)
    for degree in 2, 3:
        print('********* P%d elements:' % degree)
        n = 5
        for i in range(4):  # Run some resolutions
            n *= (i+1)
            mesh = generate_mesh(domain, n)
            #info(mesh)
            w = solver(p, Constant(0), mesh, degree=degree,
                       linear_solver='direct')
            print('max w: %g, w(0,0)=%g, h=%.3E, dofs=%d' %
                  (w.vector().array().max(), w((0,0)),
                   1/np.sqrt(mesh.num_vertices()),
                   w.function_space().dim()))

            w_exact = Expression('1 - x[0]*x[0] - x[1]*x[1]')
            w_e = interpolate(w_exact, w.function_space())
            error = np.abs(w_e.vector().array() -
                           w.vector().array()).max()
            print('error: %.3E' % error)
            assert error < 9.61E-03

def application2(
    beta, R0, num_elements_radial_dir):
    """Explore more built-in visulization features."""
    # Scaled pressure function
    p = Expression(
        '4*exp(-pow(beta,2)*(pow(x[0], 2) + pow(x[1]-R0, 2)))',
        beta=beta, R0=R0)

    # Generate mesh over the unit circle
    domain = Circle(Point(0.0, 0.0), 1.0)
    mesh = generate_mesh(domain, num_elements_radial_dir)

    w = solver(p, Constant(0), mesh, degree=1,
               linear_solver='direct')
    w.rename('w', 'deflection')

    # Plot scaled solution, mesh and pressure
    plot(mesh, title='Mesh over scaled domain')
    viz_w = plot(w,
                 wireframe=False,
                 title='Scaled membrane deflection',
                 axes=False,
                 interactive=False,
                 )
    viz_w.elevate(-10) # adjust (lift) camera from default view
    viz_w.plot(w)      # bring new settings into action
    viz_w.write_png('deflection')
    viz_w.write_pdf('deflection')

    V = w.function_space()
    p = interpolate(p, V)
    p.rename('p', 'pressure')
    viz_p = plot(p, title='Scaled pressure', interactive=False)
    viz_p.elevate(-10)
    viz_p.plot(p)
    viz_p.write_png('pressure')
    viz_p.write_pdf('pressure')

    # Dump w and p to file in VTK format
    vtkfile1 = File('membrane_deflection.pvd')
    vtkfile1 << w
    vtkfile2 = File('membrane_load.pvd')
    vtkfile2 << p

if __name__ == '__main__':
    application2(8, 0.6, 20)
    # Should be at the end
    interactive()
