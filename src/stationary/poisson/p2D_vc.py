"""As p2D_iter.py, but the PDE is -div(p*grad(u)=f."""
from __future__ import print_function
from dolfin import *

def solver(
    p, f, u0, Nx, Ny, degree=1,
    linear_solver='Krylov', # Alt: 'direct'
    abs_tol=1E-5,           # Absolute tolerance in Krylov solver
    rel_tol=1E-3,           # Relative tolerance in Krylov solver
    max_iter=1000,          # Max no of iterations in Krylov solver
    log_level=PROGRESS,     # Amount of solver output
    dump_parameters=False,  # Write out parameter database?
    ):
    """
    Solve -div(p*grad(u)=f on [0,1]x[0,1] with 2*Nx*Ny Lagrange
    elements of specified degree and u=u0 (Expresssion) on
    the boundary.
    """
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'Lagrange', degree)

    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, u0_boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(p*nabla_grad(u), nabla_grad(v))*dx
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

def solver_objects(
    p, f, u0, Nx, Ny, degree=1,
    linear_solver='Krylov', # Alt: 'direct'
    abs_tol=1E-5,           # Absolute tolerance in Krylov solver
    rel_tol=1E-3,           # Relative tolerance in Krylov solver
    max_iter=1000,          # Max no of iterations in Krylov solver
    log_level=PROGRESS,     # Amount of solver output
    dump_parameters=False,  # Write out parameter database?
    ):
    """As solver, but use objects for linear variational problem
    and solver."""
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'Lagrange', degree)

    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, u0_boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(p*nabla_grad(u), nabla_grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    problem = LinearVariationalProblem(a, L, u, bc)
    solver  = LinearVariationalSolver(problem)

    if linear_solver == 'Krylov':
        solver.parameters['linear_solver'] = 'gmres'
        solver.parameters['preconditioner'] = 'ilu'
        prm = solver.parameters['krylov_solver'] # short form
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

    solver.solve()
    return u

def test_solvers():
    """Reproduce u=1+x^2+2y^2 to with different solvers."""
    # With P1 elements we have an error E-15 with Krylov solver
    # tolerances of 1E-12, but with P2 elements the error is E-6.
    # P3 elements drive the tolerance down to E-3.
    # For higher mesh resolution we also need reduced tolerances.
    # The tol dict maps degree to expected tolerance for the coarse
    # meshes in the test.
    tol = {'direct': {1: 1E-11, 2: 1E-11, 3: 1E-11},
           'Krylov': {1: 1E-14, 2: 1E-05, 3: 1E-03}}
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    p = Expression('x[0] + x[1]')
    f = Expression('-8*x[0] - 10*x[1]')
    for Nx, Ny in [(3,3), (3,5), (5,3)]:
        for degree in 1, 2, 3:
            for linear_solver in 'direct', 'Krylov':
                for solver_func in solver, solver_objects:
                    print('solving on 2(%dx%dx) mesh with P%d elements'
                          % (Nx, Ny, degree)),
                    print(' %s solver, %s function' %
                          (linear_solver, solver_func.__name__))
                    # Important: Krylov solver error must be smaller
                    # than tol!
                    u = solver_func(
                         p, f, u0, Nx, Ny, degree,
                         linear_solver=linear_solver,
                         abs_tol=0.1*tol[linear_solver][degree],
                         rel_tol=0.1*tol[linear_solver][degree])
                    # Make a finite element function of the exact u0
                    V = u.function_space()
                    u0_Function = interpolate(u0, V)  # exact solution
                    # Check that dof arrays are equal
                    u0_array = u0_Function.vector().array()  # dof values
                    max_error = (u0_array - u.vector().array()).max()
                    msg = 'max error: %g for 2(%dx%d) mesh, degree=%d,'\
                          ' %s solver, %s' % \
                      (max_error, Nx, Ny, degree, linear_solver,
                       solver_func.__name__)
                    print(msg)
                    assert max_error < tol[linear_solver][degree], msg

def application_test():
    """Plot the solution in the test problem."""
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    p = Expression('x[0] + x[1]')
    f = Expression('-8*x[0] - 10*x[1]')
    u = solver(p, f, u0, 6, 4, 1)
    # Dump solution to file in VTK format
    file = File("poisson.pvd")
    file << u
    # Plot solution and mesh
    plot(u)

def compare_exact_and_numerical_solution(Nx, Ny, degree=1):
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    p = Expression('x[0] + x[1]')
    f = Expression('-8*x[0] - 10*x[1]')
    u = solver(p, f, u0, Nx, Ny, degree, linear_solver='direct')
    # Grab exact and numerical solution at the vertices and compare
    V = u.function_space()
    u0_Function = interpolate(u0, V)
    u0_at_vertices = u0_Function.compute_vertex_values()
    u_at_vertices = u.compute_vertex_values()
    coor = V.mesh().coordinates()
    for i, x in enumerate(coor):
        print('vertex %2d (%9g,%9g): error=%g'
              % (i, x[0], x[1],
                 u0_at_vertices[i] - u_at_vertices[i]))
        # Could compute u0(x) - u_at_vertices[i] but this
        # is much more expensive and gives more rounding errors
    center = (0.5, 0.5)
    error = u0(center) - u(center)
    print('numerical error at %s: %g' % (center, error))

def normalize_solution(u):
    """Normalize u: return u divided by max(u)."""
    u_array = u.vector().array()
    u_max = u_array.max()
    u_array /= u_max
    u.vector()[:] = u_array
    u.vector().set_local(u_array)  # alternative
    return u

def test_normalize_solution():
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    f = Constant(-6.0)
    u = solver(f, u0, 4, 2, 1, linear_solver='direct')
    u = normalize_solution(u)
    computed = u.vector().array().max()
    expected = 1.0
    assert abs(expected - computed) < 1E-15

def flux(u, p):
    """Return p*grad(u) projected onto same space as u."""
    V = u.function_space()
    mesh = V.mesh()
    degree = u.ufl_element().degree()
    V_g = VectorFunctionSpace(mesh, 'Lagrange', degree)
    grad_u = project(-p*grad(u), V_g)
    grad_u.rename('flux(u)', 'continuous flux field')
    return grad_u

def application_test_gradient(Nx=6, Ny=4):
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    p = Expression('x[0] + x[1]')
    f = Expression('-8*x[0] - 10*x[1]')
    u = solver(p, f, u0, Nx, Ny, 1, linear_solver='direct')
    u.rename('u', 'solution')
    flux_u = flux(u, p)
    # Grab each component as a scalar field
    flux_u_x, flux_u_y = flux_u.split(deepcopy=True)
    flux_u_x.rename('flux(u)_x', 'x-component of flux(u)')
    flux_u_y.rename('flux(u)_y', 'y-component of flux(u)')
    plot(u, title=u.label())
    plot(flux_u,   title=flux_u.label())
    plot(flux_u_x, title=flux_u_x.label())
    plot(flux_u_y, title=flux_u_y.label())

    u_exact = lambda x, y: 1 + x**2 + 2*y**2
    flux_x_exact = lambda x, y: -(x+y)*2*x
    flux_y_exact = lambda x, y: -(x+y)*4*y

    coor = u.function_space().mesh().coordinates()
    if len(coor) < 50:
        # Quite large errors for coarse meshes, but the error
        # decreases with increasing resolution
        for i, value in enumerate(flux_u_x.compute_vertex_values()):
            print('vertex %d, %s, -p*u_x=%g, error=%g' %
                  (i, tuple(coor[i]), value,
                   flux_x_exact(*coor[i]) - value))
        for i, value in enumerate(flux_u_y.compute_vertex_values()):
            print('vertex %d, %s, -p*u_y=%g, error=%g' %
                  (i, tuple(coor[i]), value,
                   flux_y_exact(*coor[i]) - value))
    else:
        # Compute integrated L2 error of the flux components
        # (Will this work for unstructured mesh? Need to think about that)
        xv = coor.T[0]
        yv = coor.T[1]


if __name__ == '__main__':
    #application_test()
    application_test_gradient(Nx=20, Ny=20)
    # Hold plot
    interactive()
