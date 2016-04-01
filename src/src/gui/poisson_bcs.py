"""As poisson_iter.py, but the PDE is -div(kappa*grad(u)=f."""
from __future__ import print_function
from fenics import *

def solver(
    kappa, f, u_D, Nx, Ny, degree=1,
    linear_solver='Krylov', # Alternative: 'direct'
    abs_tol=1E-5,           # Absolute tolerance in Krylov solver
    rel_tol=1E-3,           # Relative tolerance in Krylov solver
    max_iter=1000,          # Max no of iterations in Krylov solver
    log_level=PROGRESS,     # Amount of solver output
    dump_parameters=False,  # Write out parameter database?
    ):
    """
    Solve -div(kappa*grad(u)=f on [0,1]x[0,1] with 2*Nx*Ny Lagrange
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
    a = kappa*dot(grad(u), grad(v))*dx
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
    kappa, f, u_D, Nx, Ny, degree=1,
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
    a = kappa*dot(grad(u), grad(v))*dx
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
    kappa = Expression('x[0] + x[1]')
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
                         kappa, f, u_D, Nx, Ny, degree,
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
    kappa = Expression('x[0] + x[1]')
    f = Expression('-8*x[0] - 10*x[1]')
    u = solver(kappa, f, u_D, 6, 4, 1)
    # Dump solution to file in VTK format
    file = File("poisson.pvd")
    file << u
    # Plot solution and mesh
    plot(u)

def compare_exact_and_numerical_solution(Nx, Ny, degree=1):
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    kappa = Expression('x[0] + x[1]')
    f = Expression('-8*x[0] - 10*x[1]')
    u = solver(kappa, f, u_D, Nx, Ny, degree, linear_solver='direct')
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
    """Normalize u: return u divided by max(u)."""
    u_array = u.vector().array()
    u_max = u_array.max()
    u_array /= u_max
    u.vector()[:] = u_array
    u.vector().set_local(u_array)  # alternative
    return u

def test_normalize_solution():
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    f = Constant(-6.0)
    u = solver(f, u_D, 4, 2, 1, linear_solver='direct')
    u = normalize_solution(u)
    computed = u.vector().array().max()
    expected = 1.0
    assert abs(expected - computed) < 1E-15

def flux(u, kappa):
    """Return -kappa*grad(u) projected into same space as u."""
    V = u.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'P', degree)
    flux_u = project(-kappa*grad(u), W)
    flux_u.rename('flux(u)', 'continuous flux field')
    return flux_u

def demo_test_flux(Nx=6, Ny=4):
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    kappa = Expression('x[0] + x[1]')
    f = Expression('-8*x[0] - 10*x[1]')
    u = solver(kappa, f, u_D, Nx, Ny, 1, linear_solver='direct')
    u.rename('u', 'solution')
    flux_u = flux(u, kappa)
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

def compute_errors(u_exact, u):
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
    Ve = FunctionSpace(V.mesh(), 'P', 5)
    u_e = interpolate(u_exact, Ve)
    error = (u - u_e)**2*dx
    E3 = sqrt(abs(assemble(error)))

    # fenics.errornorm interpolates u and u_e to a space with
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
    error = dot(grad(e_Ve), grad(e_Ve))*dx
    E6 = sqrt(abs(assemble(error)))

    # Collect error measures in a dictionary with self-explanatory keys
    errors = {'u - u_exact': E1,
              'u - interpolate(u_exact,V)': E2,
              'interpolate(u,Ve) - interpolate(u_exact,Ve)': E3,
              'errornorm': E4,
              'infinity norm (of dofs)': E5,
              'grad(error) H1 seminorm': E6}

    return errors

def convergence_rate(u_exact, f, u_D, kappa):
    """
    Compute convergence rates for various error norms for a
    sequence of meshes and elements.
    """

    h = {}  # discretization parameter: h[degree][level]
    E = {}  # error measure(s): E[degree][level][error_type]
    degrees = 1, 2, 3, 4
    num_levels = 5

    # Iterate over degrees and mesh refinement levels
    for degree in degrees:
        n = 4  # coarsest mesh division
        h[degree] = []
        E[degree] = []
        for i in range(num_levels):
            n *= 2
            h[degree].append(1.0 / n)
            u = solver(kappa, f, u_D, n, n, degree,
                       linear_solver='direct')
            errors = compute_errors(u_exact, u)
            E[degree].append(errors)
            print('2 x (%d x %d) P%d mesh, %d unknowns, E1=%g' %
                  (n, n, degree, u.function_space().dim(),
                   errors['u - u_exact']))

    # Compute convergence rates
    from math import log as ln  # log is a fenics name too
    error_types = list(E[1][0].keys())
    rates = {}
    for degree in degrees:
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
    u_D = Constant(0)
    kappa = Constant(1)
    # Note: P4 for n>=128 seems to break down
    rates = convergence_rates(u_exact, f, u_D, kappa, degrees=4)
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
    mesh = u.function_space().mesh()
    u2 = u if u.ufl_element().degree() == 1 else \
         interpolate(u, FunctionSpace(mesh, 'P', 1))
    from BoxField import fenics_function2BoxField
    u_box = fenics_function2BoxField(
        u2, mesh, divisions, uniform_mesh=True)
    return u_box

def demo_structured_mesh(model_problem=1):
    if model_problem == 1:
        # Numerical solution is exact
        u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
        kappa = Expression('x[0] + x[1]')
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
        u_c = u_c.replace('M_PI', 'pi')  # or 'DOLFIN_PI'
        print('u in C:', u_c)
        u_D = Expression(u_c)

        kappa = 1  # Don't use Constant(1) here because of sym.diff (!)
        f = sym.diff(-kappa*sym.diff(u, x), x) + \
            sym.diff(-kappa*sym.diff(u, y), y)
        f = sym.simplify(f)
        f_c = sym.printing.ccode(f)
        f_c = f_c.replace('M_PI', 'pi')
        f = Expression(f_c)
        flux_u_x_exact = sym.lambdify([x, y], -kappa*sym.diff(u, x),
                                      modules='numpy')
        print('f in C:', f_c)
        kappa = Constant(1)   # wrap for FEniCS
        nx = 22;  ny = 22

    u = solver(kappa, f, u_D, nx, ny, 1, linear_solver='direct')
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
    u_e_val = [u_D((x_, y_fixed)) for x_ in x]

    plt.figure()
    plt.plot(x, u_val, 'r-')
    plt.plot(x, u_e_val, 'bo')
    plt.legend(['P1 elements', 'exact'], loc='best')
    plt.title('Solution along line y=%g' % y_fixed)
    plt.xlabel('x');  plt.ylabel('u')
    plt.savefig('tmp2.png'); plt.savefig('tmp2.pdf')

    flux_u = flux(u, kappa)
    flux_u_x, flux_u_y = flux_u.split(deepcopy=True)

    # Plot the numerical and exact flux along the same line
    flux2_x = flux_u_x if flux_u_x.ufl_element().degree() == 1 \
              else interpolate(flux_x,
                   FunctionSpace(u.function_space().mesh(),
                                 'P', 1))
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

def solver_bc(
    kappa, f,               # Coefficients in the PDE
    boundary_conditions,    # Dict of boundary conditions
    Nx, Ny,                 # Cell division of the domain
    degree=1,               # Polynomial degree
    subdomains=[],          # List of SubDomain objects in domain
    linear_solver='Krylov', # Alt: 'direct'
    abs_tol=1E-5,           # Absolute tolerance in Krylov solver
    rel_tol=1E-3,           # Relative tolerance in Krylov solver
    max_iter=1000,          # Max no of iterations in Krylov solver
    log_level=PROGRESS,     # Amount of solver output
    dump_parameters=False,  # Write out parameter database?
    debug=False,
    ):
    """
    Solve -div(kappa*grad(u)=f on [0,1]x[0,1] with 2*Nx*Ny Lagrange
    elements of specified degree and Dirichlet, Neumann, or Robin
    conditions on the boundary. Piecewise constant kappa over subdomains
    are also allowed.
    """
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'P', degree)

    tol = 1E-14

    # Subdomains in the domain?
    import numpy as np
    if subdomains:
        # subdomains is list of SubDomain objects,
        # p is array of corresponding constant values of p
        # in each subdomain
        if not isinstance(kappa, (list, tuple, np.ndarray)):
            raise TypeError(
                'kappa must be array if we have sudomains, not %s'
                % type(kappa))
        materials = CellFunction('size_t', mesh)
        materials.set_all(0)  # "the rest"
        for m, subdomain in enumerate(subdomains[1:], 1):
            subdomain.mark(materials, m)

        kappa_values = kappa
        V0 = FunctionSpace(mesh, 'DG', 0)
        kappa  = Function(V0)
        help = np.asarray(materials.array(), dtype=np.int32)
        kappa.vector()[:] = np.choose(help, kappa_values)
    else:
        if not isinstance(kappa, (Expression, Constant)):
            raise TypeError(
                'kappa is type %s, must be Expression or Constant'
                % type(kappa))

    # Boundary subdomains
    class BoundaryX0(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0]) < tol

    class BoundaryX1(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0] - 1) < tol

    class BoundaryY0(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1]) < tol

    class BoundaryY1(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1] - 1) < tol

    # Mark boundaries
    boundary_markers = FacetFunction('size_t', mesh)
    boundary_markers.set_all(9999)
    bx0 = BoundaryX0()
    bx1 = BoundaryX1()
    by0 = BoundaryY0()
    by1 = BoundaryY1()
    bx0.mark(boundary_markers, 0)
    bx1.mark(boundary_markers, 1)
    by0.mark(boundary_markers, 2)
    by1.mark(boundary_markers, 3)

    # Redefine boundary integration measure
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    # Collect Dirichlet conditions
    bcs = []
    for i in boundary_conditions:
        if 'Dirichlet' in boundary_conditions[i]:
            bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'],
                             boundary_markers, i)
            bcs.append(bc)

    if debug:
        # Print all vertices that belong to the boundary parts
        for x in mesh.coordinates():
            if bx0.inside(x, True): print('%s is on x = 0' % x)
            if bx1.inside(x, True): print('%s is on x = 1' % x)
            if by0.inside(x, True): print('%s is on y = 0' % x)
            if by1.inside(x, True): print('%s is on y = 1' % x)

        # Print the Dirichlet conditions
        print('Number of Dirichlet conditions:', len(bcs))
        if V.ufl_element().degree() == 1:  # P1 elements
            d2v = dof_to_vertex_map(V)
            coor = mesh.coordinates()
        for i, bc in enumerate(bcs):
            print('Dirichlet condition %d' % i)
            boundary_values = bc.get_boundary_values()
            for dof in boundary_values:
                print('   dof %2d: u=%g' % (dof, boundary_values[dof]))
                if V.ufl_element().degree() == 1:
                    print('    at point %s' %
                          (str(tuple(coor[d2v[dof]].tolist()))))

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Collect Neumann integrals
    integrals_N = []
    for i in boundary_conditions:
        if 'Neumann' in boundary_conditions[i]:
            if boundary_conditions[i]['Neumann'] != 0:
                g = boundary_conditions[i]['Neumann']
                integrals_N.append(g*v*ds(i))

    # Collect Robin integrals
    integrals_R_a = []
    integrals_R_L = []
    for i in boundary_conditions:
        if 'Robin' in boundary_conditions[i]:
            r, s = boundary_conditions[i]['Robin']
            integrals_R_a.append(r*u*v*ds(i))
            integrals_R_L.append(r*s*v*ds(i))

    # Simpler Robin integrals
    integrals_R = []
    for i in boundary_conditions:
        if 'Robin' in boundary_conditions[i]:
            r, s = boundary_conditions[i]['Robin']
            integrals_R.append(r*(u - s)*v*ds(n))

    # Define variational problem, solver_bc
    a = kappa*dot(grad(u), grad(v))*dx + sum(integrals_R_a)
    L = f*v*dx - sum(integrals_N) + sum(integrals_R_L)

    # Simpler variational formulation
    F = kappa*dot(grad(u), grad(v))*dx + \
        sum(integrals_R) - f*v*dx + sum(integrals_N)
    a, L = lhs(F), rhs(F)

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

    solve(a == L, u, bcs, solver_parameters=solver_parameters)
    return u, kappa  # Note: kappa may be modified (Function on V0)

def demo_bc_test():

    # Define manufactured solution in sympy and derive f, g, etc.
    import sympy as sym
    x, y = sym.symbols('x[0], x[1]')            # needed by UFL
    u = 1 + x**2 + 2*y**2                       # exact solution
    u_e = u                                     # exact solution
    u_00 = u.subs(x, 0)                         # restrict to x = 0
    u_01 = u.subs(x, 1)                         # restrict to x = 1
    f = -sym.diff(u, x, 2) - sym.diff(u, y, 2)  # -Laplace(u)
    f = sym.simplify(f)                         # simplify f
    g = -sym.diff(u, y).subs(y, 1)              # compute g = -du/dn
    r = 1000                                    # Robin data, arbitrary
    s = u                                       # Robin data, u = s

    # Collect variables
    variables = [u_e, u_00, u_01, f, g, r, s]

    # Turn into C/C++ code strings
    variables = [sym.printing.ccode(var) for var in variables]

    # Turn into FEniCS Expression
    variables = [Expression(var, degree=2) for var in variables]

    # Extract variables
    u_e, u_00, u_01, f, g, r, s = variables

    # Define boundary conditions
    boundary_conditions = {0: {'Dirichlet': u_00},   # x=0
                           1: {'Dirichlet': u_01},   # x=1
                           2: {'Robin':     (r, s)}, # y=0
                           3: {'Neumann':   g}}      # y=1

    # Compute solution
    p = Constant(1)
    Nx = Ny = 2
    u, kappa = solver_bc(kappa, f, boundary_conditions,
                         Nx, Ny, degree=1,
                         linear_solver='direct',
                         debug=2*Nx*Ny < 50)

    # Compute max error in infinity norm
    u_e = interpolate(u_e, u.function_space())
    import numpy as np
    max_error = np.abs(u_e.vector().array() -
                       u.vector().array()).max()
    print('Max error:', max_error)

    # Print numerical and exact solution at the vertices
    if u.function_space().dim() < 50:  # (small problems only)
        u_e_at_vertices = u_e.compute_vertex_values()
        u_at_vertices = u.compute_vertex_values()
        coor = u.function_space().mesh().coordinates()
        for i, x in enumerate(coor):
            print('vertex %2d (%9g,%9g): error=%g %g vs %g'
                  % (i, x[0], x[1],
                     u_e_at_vertices[i] - u_at_vertices[i],
                     u_e_at_vertices[i], u_at_vertices[i]))

def test_solvers_bc():
    """Reproduce u=1+x^2+2y^2 to machince precision with different solvers."""
    tol = 3E-12  # Appropriate tolerance for these tests (P2, 20x20 mesh)
    import sympy as sym
    x, y = sym.symbols('x[0], x[1]')
    u = 1 + x**2 + 2*y**2
    f = -sym.diff(u, x, 2) - sym.diff(u, y, 2)
    f = sym.simplify(f)
    u_00 = u.subs(x, 0)  # x=0 boundary
    u_01 = u.subs(x, 1)  # x=1 boundary
    g = -sym.diff(u, y).subs(y, 1)  # x=1 boundary
    r = 1000 # arbitrary function can go here
    s = u

    # Turn to C/C++ code for UFL expressions
    f = sym.printing.ccode(f)
    u_00 = sym.printing.ccode(u_00)
    u_01 = sym.printing.ccode(u_01)
    g = sym.printing.ccode(g)
    r = sym.printing.ccode(r)
    s = sym.printing.ccode(s)
    print('Test problem (C/C++):\nu = %s\nf = %s' % (u, f))
    print('u_00: %s\nu_01: %s\ng = %s\nr = %s\ns = %s' %
          (u_00, u_01, g, r, s))

    # Turn into FEniCS objects
    u_00 = Expression(u_00)
    u_01 = Expression(u_01)
    f = Expression(f)
    g = Expression(g)
    r = Expression(r)
    s = Expression(s)
    u_exact = Expression(sym.printing.ccode(u))

    # Define boundary conditions
    boundary_conditions = {0: {'Dirichlet': u_00},
                           1: {'Dirichlet': u_01},
                           2: {'Robin':     (r, s)},
                           3: {'Neumann':   g}}

    for Nx, Ny in [(3,3), (3,5), (5,3), (20,20)]:
        for degree in 1, 2, 3:
            for linear_solver in ['direct']:
                print('solving on 2(%dx%dx) mesh with P%d elements'
                      % (Nx, Ny, degree)),
                print(' %s solver, %s function' %
                      (linear_solver, solver_func.__name__))
                kappa = Constant(1)
                u, kappa = solver_bc(
                    kappa, f, boundary_conditions, Nx, Ny, degree,
                linear_solver=linear_solver,
                    abs_tol=0.1*tol,
                    rel_tol=0.1*tol)
                # Make a finite element function of the exact u_D
                V = u.function_space()
                u_e_Function = interpolate(u_exact, V)  # exact solution
                # Check that dof arrays are equal
                u_e_array = u_e_Function.vector().array()  # dof values
                max_error = (u_e_array - u.vector().array()).max()
                msg = 'max error: %g for 2(%dx%d) mesh, degree=%d,'\
                      ' %s solver, %s' % \
                      (max_error, Nx, Ny, degree, linear_solver,
                       solver_func.__name__)
                print(msg)
                assert max_error < tol, msg

def test_solvers_bc_2mat():
    tol = 2E-13  # Tolerance for comparisons

    class Omega0(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] <= 0.5+tol

    class Omega1(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] >= 0.5-tol

    subdomains = [Omega0(), Omega1()]
    kappa_values = [2.0, 13.0]
    boundary_conditions = {
        0: {'Neumann': 0},
        1: {'Neumann': 0},
        2: {'Dirichlet': Constant(0)}, # y=0
        3: {'Dirichlet': Constant(1)}, # y=1
        }

    f = Constant(0)
    u_exact = Expression(
        'x[1] <= 0.5? 2*x[1]*k_1/(k_0+k_1) : '
        '((2*x[1]-1)*k_0 + k_1)/(k_0+k_1)',
        k_0=kappa_values[0], k_1=kappa_values[1])

    for Nx, Ny in [(2,2), (2,4), (8,4)]:
        for degree in 1, 2, 3:
            u, kappa = solver_bc(
                kappa_values, f, boundary_conditions, Nx, Ny, degree,
                linear_solver='direct', subdomains=subdomains,
                debug=False)

            # Compute max error in infinity norm
            u_e = interpolate(u_exact, u.function_space())
            import numpy as np
            max_error = np.abs(u_e.vector().array() -
                           u.vector().array()).max()
            assert max_error < tol, 'max error: %g' % max_error


if __name__ == '__main__':
    #demo_test()
    #demo_test_flux(Nx=20, Ny=20)
    #convergence_rate()
    #demo_structured_mesh(2)
    #demo_linalg()
    #demo_bc_test_2mat()
    #test_solvers_bc()
    # Hold plot
    interactive()
