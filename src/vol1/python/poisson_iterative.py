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

def demo_efficiency():
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
    demo_linalg()
    demo_efficiency()
    # Hold plot
    interactive()
