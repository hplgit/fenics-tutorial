"""As poisson_func.py, but iterative linear solver."""
from __future__ import print_function
from fenics import *
import numpy as np

def solver(
    f, u_D, Nx, Ny, degree=1,
    linear_solver='Krylov', # Alt: 'direct'
    abs_tol=1E-5,           # Absolute tolerance in Krylov solver
    rel_tol=1E-3,           # Relative tolerance in Krylov solver
    max_iter=1000,          # Max no of iterations in Krylov solver
    log_level=PROGRESS,     # Amount of solver output
    print_parameters=False, # Write out parameter database?
    ):
    """
    Solve -Laplace(u)=f on [0,1]x[0,1] with 2*Nx*Ny Lagrange
    elements of specified degree and u=u_D (Expresssion) on
    the boundary.
    """
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'P', degree)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

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
        if print_parameters:
            info(parameters, True)
        solver_parameters = {'linear_solver': 'gmres',
                             'preconditioner': 'ilu'}
    else:
        solver_parameters = {'linear_solver': 'lu'}

    solve(a == L, u, bc, solver_parameters=solver_parameters)
    return u

def solver_objects(
    f, u_D, Nx, Ny, degree=1,
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
    V = FunctionSpace(mesh, 'P', degree)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
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
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    f = Constant(-6.0)
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
                         f, u_D, Nx, Ny, degree,
                         linear_solver=linear_solver,
                         abs_tol=0.1*tol[linear_solver][degree],
                         rel_tol=0.1*tol[linear_solver][degree])
                    # Make a finite element function of the exact u_D
                    V = u.function_space()
                    u_D_Function = interpolate(u_D, V)  # exact solution
                    # Check that dof arrays are equal
                    u_D_array = u_D_Function.vector().array()  # dof values
                    max_error = (u_D_array - u.vector().array()).max()
                    msg = 'max error: %g for 2(%dx%d) mesh, degree=%d,'\
                          ' %s solver, %s' % \
                      (max_error, Nx, Ny, degree, linear_solver,
                       solver_func.__name__)
                    print(msg)
                    assert max_error < tol[linear_solver][degree], msg

def demo_test():
    """Plot the solution in the test problem."""
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    f = Constant(-6.0)
    u = solver(f, u_D, 6, 4, 1)
    # Dump solution to file in VTK format
    u.rename('u', 'potential')  # name 'u' is used in plot
    vtkfile = File("poisson.pvd")
    vtkfile << u
    # Plot solution on the screen
    plot(u)

def compare_exact_and_numerical_solution(Nx, Ny, degree=1):
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    f = Constant(-6.0)
    u = solver(f, u_D, Nx, Ny, degree, linear_solver='direct')
    # Grab exact and numerical solution at the vertices and compare
    V = u.function_space()
    u_D_Function = interpolate(u_D, V)
    u_D_at_vertices = u_D_Function.compute_vertex_values()
    u_at_vertices = u.compute_vertex_values()
    coor = V.mesh().coordinates()
    for i, x in enumerate(coor):
        print('vertex %2d (%9g,%9g): error=%g'
              % (i, x[0], x[1],
                 u_D_at_vertices[i] - u_at_vertices[i]))
        # Could compute u_D(x) - u_at_vertices[i] but this
        # is much more expensive and gives more rounding errors
    center = (0.5, 0.5)
    error = u_D(center) - u(center)
    print('numerical error at %s: %g' % (center, error))

def normalize_solution(u):
    """Normalize solution by dividing by max(|u|)."""
    nodal_values = u.vector().array()
    u_max = np.abs(nodal_values).max()
    nodal_values /= u_max
    u.vector()[:] = nodal_values
    #u.vector().set_local(dofs) # alternative

def test_normalize_solution():
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    f = Constant(-6.0)
    u = solver(f, u_D, 4, 2, 1, linear_solver='direct')
    normalize_solution(u)
    computed = u.vector().array().max()
    expected = 1.0
    assert abs(expected - computed) < 1E-15

def gradient(u):
    """Return grad(u) projected onto same space as u."""
    V = u.function_space()
    mesh = V.mesh()
    V_g = VectorFunctionSpace(mesh, 'P', 1)
    grad_u = project(grad(u), V_g)
    grad_u.rename('grad(u)', 'continuous gradient field')
    return grad_u

def demo_test_gradient(Nx=6, Ny=4):
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    f = Constant(-6.0)
    u = solver(f, u_D, Nx, Ny, 1, linear_solver='direct')
    u.rename('u', 'solution')
    grad_u = gradient(u)
    # Grab each component as a scalar field
    grad_u_x, grad_u_y = grad_u.split(deepcopy=True)
    grad_u_x.rename('grad(u)_x', 'x-component of grad(u)')
    grad_u_y.rename('grad(u)_y', 'y-component of grad(u)')
    plot(u, title=u.label())
    plot(grad_u,   title=grad_u.label())
    plot(grad_u_x, title=grad_u_x.label())
    plot(grad_u_y, title=grad_u_y.label())

    coor = u.function_space().mesh().coordinates()
    if len(coor) < 50:
        for i, value in enumerate(grad_u_x.compute_vertex_values()):
            print('vertex %d, %s, u_x=%g (2x), error=%g' %
                  (i, tuple(coor[i]), value, 2*coor[i][0] - value))
        for i, value in enumerate(grad_u_y.compute_vertex_values()):
            print('vertex %d, %s, u_y=%g (4y), error=%g' %
                  (i, tuple(coor[i]), value, 4*coor[i][1] - value))

def solver_linalg(
    p, f, u_D, Nx, Ny, degree=1,
    linear_solver='Krylov', # Alt: 'direct'
    abs_tol=1E-5,           # Absolute tolerance in Krylov solver
    rel_tol=1E-3,           # Relative tolerance in Krylov solver
    max_iter=1000,          # Max no of iterations in Krylov solver
    log_level=PROGRESS,     # Amount of solver output
    dump_parameters=False,  # Write out parameter database?
    assembly='variational', # or 'matvec' or 'system'
    start_vector='zero',    # or 'random'
    ):
    """
    Solve -div(p*grad(u)=f on [0,1]x[0,1] with 2*Nx*Ny Lagrange
    elements of specified degree and u=u_D (Expresssion) on
    the boundary.
    """
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'P', degree)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(p*grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    U = u.vector()
    if initial_guess == 'random':
        import numpy as np
        np.random.seed(10)  # for testing
        U[:] = numpy.random.uniform(-100, 100, n)

    if assembly == 'variational':
        if linear_solver == 'Krylov':
            prm = parameters['krylov_solver'] # short form
            prm['absolute_tolerance'] = abs_tol
            prm['relative_tolerance'] = rel_tol
            prm['maximum_iterations'] = max_iter
            prm['nonzero_initial_guess'] = True
            print(parameters['linear_algebra_backend'])
            set_log_level(log_level)
            if dump_parameters:
                info(parameters, True)
            solver_parameters = {'linear_solver': 'gmres',
                                 'preconditioner': 'ilu'}
        else:
            solver_parameters = {'linear_solver': 'lu'}

        solve(a == L, u, bc, solver_parameters=solver_parameters)
        A = None # Cannot return cofficient matrix
    else:
        if assembly == 'matvec':
            A = assemble(a)
            b = assemble(L)
            bc.apply(A, b)
            if linear_solver == 'direct':
                solve(A, U, b)
            else:
                solver = KrylovSolver('gmres', 'ilu')
                prm = solver.parameters
                prm['absolute_tolerance'] = abs_tol
                prm['relative_tolerance'] = rel_tol
                prm['maximum_iterations'] = max_iter
                prm['nonzero_initial_guess'] = True
                solver.solve(A, U, b)
        elif assembly == 'system':
            A, b = assemble_system(a, L, [bc])
            if linear_solver == 'direct':
                solve(A, U, b)
            else:
                solver = KrylovSolver('cg', 'ilu')
                prm = solver.parameters
                prm['absolute_tolerance'] = abs_tol
                prm['relative_tolerance'] = rel_tol
                prm['maximum_iterations'] = max_iter
                prm['nonzero_initial_guess'] = True
                solver.solve(A, U, b)
    return u, A

def demo_linalg():
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    p = Expression('x[0] + x[1]')
    f = Expression('-8*x[0] - 10*x[1]')
    meshes = [2, 8, 32, 128]
    for n in meshes:
        for assembly in 'variational', 'matvec', 'system':
            print('--- %dx%d mesh, %s assembly ---' % (n, n, assembly))
            u, A = solver_linalg(
                p, f, u_D, n, n, linear_solver='Krylov',
                assembly=assembly)
            if A is not None and u.function_space().dim() < 10:
                import numpy as np
                np.set_printoptions(precision=2)
                print('A: %s assembly\n' % assembly, A.array())

def efficiency():
    """Measure CPU time: direct vs Krylov solver."""
    import time
    # This solution is an eigenfunction, CG may have superlinear
    # convergence in such cases but GMRES is seemingly not affected
    u_exact = Expression('sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1])')
    f = Expression('2*pow(DOLFIN_PI,2)*sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1])')
    u_D = Constant(0)

    # Establish what the errors for P1, P2 and P3 elements are,
    # because initial tolerances for Krylov solvers must be lower
    # than these errors.
    # For finer meshes, we let the Krylov solver tolerances be
    # reduced in the same manner as the error, i.e., as
    # h**(-(degree+1)). With h halved, we get a reduction factor
    # 2**(degree+1).
    n = 80
    tol = {}
    for degree in 1, 2, 3:
        u = solver(f, u_D, n, n, degree, linear_solver='direct')
        u_e = interpolate(u_exact, u.function_space())
        error = np.abs(u_e.vector().array() - u.vector().array()).max()
        print('error degree=%d: %g' % (degree, error))
        tol[degree] = 0.1*error  # suitable Krylov solver tolerance
    timings_direct = []
    timings_Krylov = []
    for i in range(2):
        n *= 2
        for degree in 1, 2, 3:
            # Run direct solver
            print('n=%d, degree=%d, N:' % (n, degree)),
            t0 = time.clock()
            u = solver(f, u_D, n, n, degree, linear_solver='direct')
            t1 = time.clock()
            N = u.function_space().dim()
            print(N),
            timings_direct.append((N, t1-t0))
            # Run Krylov solver
            # (with tolerance reduced as the error)
            tol[degree] /= 2.0**(degree+1)
            print(' tol=%E' % tol[degree])
            t0 = time.clock()
            u = solver(f, u_D, n, n, degree, linear_solver='Krylov',
                       abs_tol=tol[degree], rel_tol=tol[degree])
            t1 = time.clock()
            timings_Krylov.append((N, t1-t0))
    for i in range(len(timings_direct)):
        print('LU decomp N=%d: %g' %
              (timings_direct[i][0], timings_direct[i][1]))
        print('GMRES+ILU N=%d: %g' %
              (timings_Krylov[i][0], timings_Krylov[i][1]))


if __name__ == '__main__':
    #demo_test()
    #demo_test_gradient()
    efficiency()
    # Hold plot
    interactive()
