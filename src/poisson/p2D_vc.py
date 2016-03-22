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

def compute_errors(u, u_exact):
    """Compute various measures of the error u - u_exact, where
    u is a finite element Function and u_exact is an Expression."""

    # Compute error norm (for very small errors, the value can be
    # negative so we run abs(assemble(error)) to avoid failure in sqrt

    V = u.function_space()

    # Function - Expression
    error = (u - u_exact)**2*dx
    E1 = sqrt(abs(assemble(error)))

    # Explicit interpolation of u_e onto the same space as u:
    u_e = interpolate(u_exact, V)
    error = (u - u_e)**2*dx
    E2 = sqrt(abs(assemble(error)))

    # Explicit interpolation of u_exact to higher-order elements,
    # u will also be interpolated to the space Ve before integration
    Ve = FunctionSpace(V.mesh(), 'Lagrange', 5)  # mesh here: BUG, module mesh in dolfin...make warning box
    u_e = interpolate(u_exact, Ve)
    error = (u - u_e)**2*dx
    E3 = sqrt(abs(assemble(error)))

    # dolfin.errornorm interpolates u and u_e to a space with
    # given degree, and creates the error field by subtracting
    # the degrees of freedom, then the error field is integrated
    # TEMPORARY BUG - doesn't accept Expression for u_e
    #E4 = errornorm(u_e, u, normtype='l2', degree=3)
    # Manual implementation errornorm to get around the bug:
    def errornorm(u_exact, u, Ve):
        u_Ve = interpolate(u, Ve)
        u_e_Ve = interpolate(u_exact, Ve)
        e_Ve = Function(Ve)
        # Subtract degrees of freedom for the error field
        e_Ve.vector()[:] = u_e_Ve.vector().array() - u_Ve.vector().array()
        # More efficient computation (avoids the rhs array result above)
        #e_Ve.assign(u_e_Ve)                      # e_Ve = u_e_Ve
        #e_Ve.vector().axpy(-1.0, u_Ve.vector())  # e_Ve += -1.0*u_Ve
        error = e_Ve**2*dx(Ve.mesh())
        return sqrt(abs(assemble(error))), e_Ve
    E4, e_Ve = errornorm(u_exact, u, Ve)

    # Infinity norm based on nodal values
    u_e = interpolate(u_exact, V)
    E5 = abs(u_e.vector().array() - u.vector().array()).max()

    # H1 seminorm
    error = inner(grad(e_Ve), grad(e_Ve))*dx
    E6 = sqrt(abs(assemble(error)))

    # Collect error measures in a dictionary with self-explanatory keys
    errors = {'u - u_exact': E1,
              'u - interpolate(u_exact,V)': E2,
              'interpolate(u,Ve) - interpolate(u_exact,Ve)': E3,
              'errornorm': E4,
              'infinity norm (of dofs)': E5,
              'grad(error) H1 seminorm': E6}

    return errors

def convergence_rate(u_exact, f, u0, p, degrees,
                     n=[2**(k+3) for k in range(5)]):
    """
    Compute convergence rates for various error norms for a
    sequence of meshes with Nx=Ny=b and P1, P2, ...,
    Pdegrees elements. Return rates for two consecutive meshes:
    rates[degree][error_type] = r0, r1, r2, ...
    """

    h = {}  # Discretization parameter, h[degree][experiment]
    E = {}  # Error measure(s), E[degree][experiment][error_type]
    P_degrees = 1,2,3,4
    num_meshes = 5

    # Perform experiments with meshes and element types
    for degree in P_degrees:
        n = 4   # Coarsest mesh division
        h[degree] = []
        E[degree] = []
        for i in range(num_meshes):
            n *= 2
            h[degree].append(1.0/n)
            u = solver(p, f, u0, n, n, degree,
                       linear_solver='direct')
            errors = compute_errors(u, u_exact)
            E[degree].append(errors)
            print('2*(%dx%d) P%d mesh, %d unknowns, E1=%g' %
                  (n, n, degree, u.function_space().dim(),
                   errors['u - u_exact']))
    # Convergence rates
    from math import log as ln  # log is a dolfin name too
    error_types = list(E[1][0].keys())
    rates = {}
    for degree in P_degrees:
        rates[degree] = {}
        for error_type in sorted(error_types):
            rates[degree][error_type] = []
            for i in range(num_meshes):
                Ei   = E[degree][i][error_type]
                Eim1 = E[degree][i-1][error_type]
                r = ln(Ei/Eim1)/ln(h[degree][i]/h[degree][i-1])
                rates[degree][error_type].append(round(r,2))
    return rates

def convergence_rate_sin():
    """Compute convergence rates for u=sin(x)*sin(y) solution."""
    omega = 1.0
    u_exact = Expression('sin(omega*pi*x[0])*sin(omega*pi*x[1])',
                         omega=omega)
    f = 2*omega**2*pi**2*u_exact
    u0 = Constant(0)
    p = Constant(1)
    # Note: P4 for n>=128 seems to break down
    rates = convergence_rates(u_exact, f, u0, p, degrees=4,
                              n=[2**(k+3) for k in range(5)])
    # Print rates
    print('\n\n')
    for error_type in error_types:
        print(error_type)
        for degree in P_degrees:
            print('P%d: %s' %
                  (degree, str(rates[degree][error_type])[1:-1]))

def structured_mesh(u, divisions):
    """Represent u on a structured mesh."""
    # u must have P1 elements, otherwise interpolate to P1 elements
    u2 = u if u.ufl_element().degree() == 1 else \
         interpolate(u, FunctionSpace(mesh, 'Lagrange', 1))
    mesh = u.function_space().mesh()
    from BoxField import dolfin_function2BoxField
    u_box = dolfin_function2BoxField(
        u2, mesh, divisions, uniform_mesh=True)
    return u_box

def application_structured_mesh(model_problem=1):
    if model_problem == 1:
        # Numerical solution is exact
        u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
        p = Expression('x[0] + x[1]')
        f = Expression('-8*x[0] - 10*x[1]')
        flux_u_x_exact = lambda x, y: -(x + y)*2*x
        nx = 6;  ny = 4
    elif model_problem == 2:
        # Mexican hat solution
        from sympy import exp, sin, pi  # for use in math formulas
        import sympy as sym
        H = lambda x: exp(-16*(x-0.5)**2)*sin(3*pi*x)
        x, y = sym.symbols('x[0], x[1]')
        u = H(x)*H(y)
        u_c = sym.printing.ccode(u)
        # '-exp(-16*pow(x - 0.5, 2) - 16*pow(y - 0.5, 2))*'
        # 'sin(3*M_PI*x)*sin(3*M_PI*y)'
        u_c = u_c.replace('M_PI', 'DOLFIN_PI')
        print('u in C:', u_c)
        u0 = Expression(u_c)

        p = 1  # Don't use Constant(1) here (!)
        f = sym.diff(-p*sym.diff(u, x), x) + \
            sym.diff(-p*sym.diff(u, y), y)
        f = sym.simplify(f)
        f_c = sym.printing.ccode(f)
        f_c = f_c.replace('M_PI', 'DOLFIN_PI')
        f = Expression(f_c)
        flux_u_x_exact = sym.lambdify([x, y], -p*sym.diff(u, x),
                                      modules='numpy')
        print('f in C:', f_c)
        p = Constant(1)
        nx = 22;  ny = 22

    u = solver(p, f, u0, nx, ny, 1, linear_solver='direct')
    u_box = structured_mesh(u, (nx, ny))
    u_ = u_box.values  # numpy array
    X = 0;  Y = 1      # for indexing in x and y direction

    # Iterate over 2D mesh points (i,j)
    print('u_ is defined on a structured mesh with %s points'
          % str(u_.shape))
    if u.function_space().dim() < 100:
        for j in range(u_.shape[1]):
            for i in range(u_.shape[0]):
                print('u[%d,%d]=u(%g,%g)=%g' %
                      (i, j,
                       u_box.grid.coor[X][i], u_box.grid.coor[X][j],
                       u_[i,j]))

    # Make surface plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cv = u_box.grid.coorv  # vectorized mesh coordinates
    ax.plot_surface(cv[X], cv[Y], u_, cmap=cm.coolwarm,
                    rstride=1, cstride=1)
    plt.title('Surface plot of solution')
    plt.savefig('tmp0.png'); plt.savefig('tmp0.pdf')

    # Make contour plot
    fig = plt.figure()
    ax = fig.gca()
    cs = ax.contour(cv[X], cv[Y], u_, 7)  # 7 levels
    plt.clabel(cs)  # add labels to contour lines
    plt.axis('equal')
    plt.title('Contour plot of solution')
    plt.savefig('tmp1.png'); plt.savefig('tmp1.pdf')

    # Plot u along a line y=const and compare with exact solution
    start = (0, 0.4)
    x, u_val, y_fixed, snapped = u_box.gridline(start, direction=X)
    u_e_val = [u0((x_, y_fixed)) for x_ in x]

    plt.figure()
    plt.plot(x, u_val, 'r-')
    plt.plot(x, u_e_val, 'bo')
    plt.legend(['P1 elements', 'exact'], loc='best')
    plt.title('Solution along line y=%g' % y_fixed)
    plt.xlabel('x');  plt.ylabel('u')
    plt.savefig('tmp2.png'); plt.savefig('tmp2.pdf')

    flux_u = flux(u, p)
    flux_u_x, flux_u_y = flux_u.split(deepcopy=True)

    # Plot the numerical and exact flux along the same line
    flux2_x = flux_u_x if flux_u_x.ufl_element().degree() == 1 \
              else interpolate(flux_x,
                   FunctionSpace(u.function_space().mesh(),
                                 'Lagrange', 1))
    flux_u_x_box = structured_mesh(flux_u_x, (nx,ny))
    x, flux_u_val, y_fixed, snapped = \
       flux_u_x_box.gridline(start, direction=X)
    y = y_fixed

    plt.figure()
    plt.plot(x, flux_u_val, 'r-')
    plt.plot(x, flux_u_x_exact(x, y_fixed), 'bo')
    plt.legend(['P1 elements', 'exact'], loc='best')
    plt.title('Flux along line y=%g' % y_fixed)
    plt.xlabel('x');  plt.ylabel('u')
    plt.savefig('tmp3.png'); plt.savefig('tmp3.pdf')

    plt.show()

def solver_linalg(
    p, f, u0, Nx, Ny, degree=1,
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

def application_linalg():
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    p = Expression('x[0] + x[1]')
    f = Expression('-8*x[0] - 10*x[1]')
    meshes = [2, 8, 32, 128]
    for n in meshes:
        for assembly in 'variational', 'matvec', 'system':
            print('--- %dx%d mesh, %s assembly ---' % (n, n, assembly))
            u, A = solver_linalg(
                p, f, u0, n, n, linear_solver='Krylov',
                assembly=assembly)
            if A is not None and u.function_space().dim() < 10:
                import numpy as np
                np.set_printoptions(precision=2)
                print('A: %s assembly\n' % assembly, A.array())

if __name__ == '__main__':
    #application_test()
    #application_test_gradient(Nx=20, Ny=20)
    #convergence_rate()
    #application_structured_mesh(2)
    application_linalg()
    # Hold plot
    interactive()
